from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.dataset.types import Analyses, OptimizedCases, Structures
from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    denormalize_generalized_stiffness,
    mechanical_terms,
    normalize_generalized_stiffness,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import symmetrize_matrix, upper_triangle_edge_index


@dataclass(frozen=True)
class OptimizationLossWeights:
    stiffness: float = 1.0
    material: float = 5e2
    sparsity: float = 0.03
    short_beam: float = 2e4
    long_beam: float = 2e4
    thin_beam: float = 4e6
    thick_beam: float = 2e6
    free_spacing: float = 8e4
    domain: float = 40.0
    anchor_attachment: float = 5.0
    psd: float = 0.05


@dataclass(frozen=True)
class CaseOptimizationConfig:
    num_steps: int = 120
    learning_rate: float = 2e-2
    log_every: int = 10
    stiffness_perturbation_scale: float = 0.18
    weights: OptimizationLossWeights = OptimizationLossWeights()
    mechanics: Frame3DConfig = Frame3DConfig()
    geometry: GeometryPenaltyConfig = GeometryPenaltyConfig()


def _allowed_edge_mask(roles: torch.Tensor) -> torch.Tensor:
    fixed_mask, mobile_mask, _ = role_masks(roles.unsqueeze(0))
    fixed_mask = fixed_mask[0]
    mobile_mask = mobile_mask[0]
    forbidden = (
        (fixed_mask[:, None] & fixed_mask[None, :])
        | (mobile_mask[:, None] & mobile_mask[None, :])
        | (fixed_mask[:, None] & mobile_mask[None, :])
        | (mobile_mask[:, None] & fixed_mask[None, :])
    )
    allowed = ~forbidden
    allowed.fill_diagonal_(False)
    return allowed


def _logits_from_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    clamped = adjacency.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(clamped) - torch.log1p(-clamped)


def _build_adjacency(
    edge_logits: torch.Tensor,
    roles: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    edge_i, edge_j = upper_triangle_edge_index(num_nodes, edge_logits.device)
    allowed_mask = _allowed_edge_mask(roles)
    upper_mask = allowed_mask[edge_i, edge_j]
    adjacency = torch.zeros((num_nodes, num_nodes), device=edge_logits.device, dtype=edge_logits.dtype)
    adjacency[edge_i[upper_mask], edge_j[upper_mask]] = torch.sigmoid(edge_logits)
    return symmetrize_matrix(adjacency)


def sample_target_stiffness(
    structures: Structures,
    config: CaseOptimizationConfig | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    config = config or CaseOptimizationConfig()
    rng = random.Random(seed)
    structures.validate()
    if structures.batch_size != 1:
        raise ValueError("sample_target_stiffness expects a single-structure batch")
    base = mechanical_terms(
        positions=structures.positions,
        roles=structures.roles,
        adjacency=structures.adjacency,
        frame_config=config.mechanics,
        penalty_config=config.geometry,
    )["generalized_stiffness"][0]
    normalized_base = normalize_generalized_stiffness(base, config=config.mechanics)
    jitter = torch.tensor(
        [[rng.gauss(0.0, 1.0) for _ in range(6)] for _ in range(6)],
        dtype=normalized_base.dtype,
    )
    symmetric_jitter = symmetrize_matrix(jitter)
    scale = normalized_base.abs().mean().clamp_min(1e-3)
    proposal = normalized_base + config.stiffness_perturbation_scale * scale * symmetric_jitter
    eigenvalues, eigenvectors = torch.linalg.eigh(proposal)
    minimum_eigenvalue = max(float(normalized_base.diagonal().mean().item()) * 0.05, 1e-4)
    clamped = eigenvalues.clamp_min(minimum_eigenvalue)
    normalized_target = eigenvectors @ torch.diag(clamped) @ eigenvectors.transpose(0, 1)
    return denormalize_generalized_stiffness(normalized_target, config=config.mechanics)


def _domain_penalty(positions: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    _, _, free_mask = role_masks(roles.unsqueeze(0))
    free_mask = free_mask[0]
    violations = positions[free_mask].clamp_max(0.0).square().sum()
    violations = violations + (positions[free_mask] - 1.0).clamp_min(0.0).square().sum()
    if not int(free_mask.sum().item()):
        return torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
    return violations / int(free_mask.sum().item())


def _anchor_attachment_penalty(adjacency: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    fixed_mask, mobile_mask, free_mask = role_masks(roles.unsqueeze(0))
    anchor_mask = (fixed_mask | mobile_mask)[0]
    free_mask = free_mask[0]
    anchor_to_free = adjacency[anchor_mask][:, free_mask]
    if anchor_to_free.numel() == 0:
        return torch.tensor(0.0, device=adjacency.device, dtype=adjacency.dtype)
    attachment = anchor_to_free.sum(dim=1)
    return (1.5 - attachment).clamp_min(0.0).square().mean()


def _stiffness_loss(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    normalized_current = normalize_generalized_stiffness(current)
    normalized_target = normalize_generalized_stiffness(target)
    scale = normalized_target.abs().amax().clamp_min(1e-3)
    return ((normalized_current - normalized_target) / scale).square().mean()


def _psd_penalty(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(normalize_generalized_stiffness(matrix))
    return (-eigenvalues).clamp_min(0.0).square().mean()


def _loss_breakdown(
    structures: Structures,
    target_stiffness: torch.Tensor,
    config: CaseOptimizationConfig,
) -> dict[str, torch.Tensor]:
    structures.validate()
    terms = mechanical_terms(
        positions=structures.positions,
        roles=structures.roles,
        adjacency=structures.adjacency,
        frame_config=config.mechanics,
        penalty_config=config.geometry,
    )
    generalized_stiffness = terms["generalized_stiffness"][0]
    weights = config.weights
    sparsity = structures.adjacency[0].mean()
    breakdown = {
        "stiffness_loss": weights.stiffness * _stiffness_loss(generalized_stiffness, target_stiffness),
        "material_loss": weights.material * terms["material_usage"][0],
        "sparsity_loss": weights.sparsity * sparsity,
        "short_beam_loss": weights.short_beam * terms["short_beam_penalty"][0],
        "long_beam_loss": weights.long_beam * terms["long_beam_penalty"][0],
        "thin_beam_loss": weights.thin_beam * terms["thin_beam_penalty"][0],
        "thick_beam_loss": weights.thick_beam * terms["thick_beam_penalty"][0],
        "free_spacing_loss": weights.free_spacing * terms["free_node_spacing_penalty"][0],
        "domain_loss": weights.domain * _domain_penalty(structures.positions[0], structures.roles[0]),
        "anchor_attachment_loss": weights.anchor_attachment * _anchor_attachment_penalty(structures.adjacency[0], structures.roles[0]),
        "psd_loss": weights.psd * _psd_penalty(generalized_stiffness),
    }
    breakdown["total_loss"] = torch.stack(tuple(breakdown.values())).sum()
    breakdown["generalized_stiffness"] = generalized_stiffness
    breakdown["material_usage"] = terms["material_usage"][0]
    breakdown["short_beam_penalty"] = terms["short_beam_penalty"][0]
    breakdown["long_beam_penalty"] = terms["long_beam_penalty"][0]
    breakdown["thin_beam_penalty"] = terms["thin_beam_penalty"][0]
    breakdown["thick_beam_penalty"] = terms["thick_beam_penalty"][0]
    breakdown["free_node_spacing_penalty"] = terms["free_node_spacing_penalty"][0]
    return breakdown


def _optimize_single_case(
    structures: Structures,
    target_stiffness: torch.Tensor,
    config: CaseOptimizationConfig | None = None,
    logdir: str | Path | None = None,
) -> OptimizedCases:
    config = config or CaseOptimizationConfig()
    structures.validate()
    if structures.batch_size != 1:
        raise ValueError("_optimize_single_case expects a single-case batch")
    if target_stiffness.shape != (1, 6, 6):
        raise ValueError("target_stiffness must have shape [1, 6, 6] for single-case optimization")

    initial_structures = structures
    target_stiffness_single = target_stiffness[0].detach().clone()
    _, _, free_mask = role_masks(initial_structures.roles)
    free_mask = free_mask[0]
    free_positions = torch.nn.Parameter(initial_structures.positions[0][free_mask].clone())
    edge_i, edge_j = upper_triangle_edge_index(initial_structures.num_nodes, initial_structures.positions.device)
    allowed = _allowed_edge_mask(initial_structures.roles[0])
    active_upper = allowed[edge_i, edge_j]
    edge_logits = torch.nn.Parameter(
        _logits_from_adjacency(initial_structures.adjacency[0][edge_i[active_upper], edge_j[active_upper]])
    )
    optimizer = torch.optim.Adam([free_positions, edge_logits], lr=config.learning_rate)

    writer = SummaryWriter(log_dir=str(logdir)) if logdir is not None else None
    best_loss = None
    best_structures = initial_structures
    initial_loss = None
    best_breakdown: dict[str, torch.Tensor] | None = None

    for step in range(config.num_steps):
        optimizer.zero_grad()
        positions = initial_structures.positions[0].clone()
        positions[free_mask] = free_positions
        adjacency = _build_adjacency(
            edge_logits=edge_logits,
            roles=initial_structures.roles[0],
            num_nodes=initial_structures.num_nodes,
        )
        current_structures = Structures(
            positions=positions.unsqueeze(0),
            roles=initial_structures.roles.clone(),
            adjacency=adjacency.unsqueeze(0),
        )
        breakdown = _loss_breakdown(current_structures, target_stiffness_single, config)
        loss = breakdown["total_loss"]
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().item())
        if initial_loss is None:
            initial_loss = current_loss
        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_structures = Structures(
                positions=current_structures.positions.detach().clone(),
                roles=current_structures.roles.detach().clone(),
                adjacency=current_structures.adjacency.detach().clone(),
            )
            best_breakdown = {name: value.detach().clone() for name, value in breakdown.items()}

        if writer is not None and (step % config.log_every == 0 or step == config.num_steps - 1):
            for name, value in breakdown.items():
                if name in {
                    "generalized_stiffness",
                    "material_usage",
                    "short_beam_penalty",
                    "long_beam_penalty",
                    "thin_beam_penalty",
                    "thick_beam_penalty",
                    "free_node_spacing_penalty",
                }:
                    continue
                writer.add_scalar(
                    f"dataset/optimization/{name}",
                    float(value.detach().item()),
                    step,
                )

    if writer is not None:
        writer.close()

    if best_breakdown is None:
        best_breakdown = _loss_breakdown(best_structures, target_stiffness_single, config)

    result = OptimizedCases(
        raw_structures=Structures(
            positions=initial_structures.positions.detach().clone(),
            roles=initial_structures.roles.detach().clone(),
            adjacency=initial_structures.adjacency.detach().clone(),
        ),
        target_stiffness=target_stiffness.detach().clone(),
        optimized_structures=best_structures,
        initial_loss=torch.tensor([float(initial_loss if initial_loss is not None else 0.0)], dtype=torch.float32),
        best_loss=torch.tensor([float(best_loss if best_loss is not None else 0.0)], dtype=torch.float32),
        last_analyses=Analyses(
            generalized_stiffness=best_breakdown["generalized_stiffness"].unsqueeze(0),
            material_usage=best_breakdown["material_usage"].reshape(1),
            short_beam_penalty=best_breakdown["short_beam_penalty"].reshape(1),
            long_beam_penalty=best_breakdown["long_beam_penalty"].reshape(1),
            thin_beam_penalty=best_breakdown["thin_beam_penalty"].reshape(1),
            thick_beam_penalty=best_breakdown["thick_beam_penalty"].reshape(1),
            free_node_spacing_penalty=best_breakdown["free_node_spacing_penalty"].reshape(1),
        ),
    )
    result.validate()
    return result


def optimize_cases(
    structures: Structures,
    target_stiffness: torch.Tensor,
    config: CaseOptimizationConfig | None = None,
    logdir: str | Path | None = None,
) -> OptimizedCases:
    config = config or CaseOptimizationConfig()
    structures.validate()
    if target_stiffness.shape != (structures.batch_size, 6, 6):
        raise ValueError("target_stiffness must have shape [batch, 6, 6]")

    optimized_items = []
    for case_index in range(structures.batch_size):
        case_logdir = None if logdir is None else Path(logdir) / f"case_{case_index:04d}"
        optimized_items.append(
            _optimize_single_case(
                structures=structures.slice(case_index),
                target_stiffness=target_stiffness[case_index : case_index + 1],
                config=config,
                logdir=case_logdir,
            )
        )

    result = OptimizedCases(
        raw_structures=Structures(
            positions=torch.cat([item.raw_structures.positions for item in optimized_items], dim=0),
            roles=torch.cat([item.raw_structures.roles for item in optimized_items], dim=0),
            adjacency=torch.cat([item.raw_structures.adjacency for item in optimized_items], dim=0),
        ),
        target_stiffness=torch.cat([item.target_stiffness for item in optimized_items], dim=0),
        optimized_structures=Structures(
            positions=torch.cat([item.optimized_structures.positions for item in optimized_items], dim=0),
            roles=torch.cat([item.optimized_structures.roles for item in optimized_items], dim=0),
            adjacency=torch.cat([item.optimized_structures.adjacency for item in optimized_items], dim=0),
        ),
        initial_loss=torch.cat([item.initial_loss for item in optimized_items], dim=0),
        best_loss=torch.cat([item.best_loss for item in optimized_items], dim=0),
        last_analyses=Analyses(
            generalized_stiffness=torch.cat([item.last_analyses.generalized_stiffness for item in optimized_items], dim=0),
            material_usage=torch.cat([item.last_analyses.material_usage for item in optimized_items], dim=0),
            short_beam_penalty=torch.cat([item.last_analyses.short_beam_penalty for item in optimized_items], dim=0),
            long_beam_penalty=torch.cat([item.last_analyses.long_beam_penalty for item in optimized_items], dim=0),
            thin_beam_penalty=torch.cat([item.last_analyses.thin_beam_penalty for item in optimized_items], dim=0),
            thick_beam_penalty=torch.cat([item.last_analyses.thick_beam_penalty for item in optimized_items], dim=0),
            free_node_spacing_penalty=torch.cat([item.last_analyses.free_node_spacing_penalty for item in optimized_items], dim=0),
        ),
    )
    result.validate()
    return result
