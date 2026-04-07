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
    num_steps: int = 32
    learning_rate: float = 1e-3
    log_every: int = 10
    stiffness_perturbation_scale: float = 0.18
    weights: OptimizationLossWeights = OptimizationLossWeights()
    mechanics: Frame3DConfig = Frame3DConfig()
    geometry: GeometryPenaltyConfig = GeometryPenaltyConfig()


_LOGGED_BREAKDOWN_NAMES = (
    "stiffness_loss",
    "material_loss",
    "sparsity_loss",
    "short_beam_loss",
    "long_beam_loss",
    "thin_beam_loss",
    "thick_beam_loss",
    "free_spacing_loss",
    "domain_loss",
    "anchor_attachment_loss",
    "psd_loss",
    "total_loss",
)


def _allowed_edge_mask(roles: torch.Tensor) -> torch.Tensor:
    if roles.ndim == 1:
        roles = roles.unsqueeze(0)
    fixed_mask, mobile_mask, _ = role_masks(roles)
    forbidden = (
        (fixed_mask.unsqueeze(-1) & fixed_mask.unsqueeze(-2))
        | (mobile_mask.unsqueeze(-1) & mobile_mask.unsqueeze(-2))
        | (fixed_mask.unsqueeze(-1) & mobile_mask.unsqueeze(-2))
        | (mobile_mask.unsqueeze(-1) & fixed_mask.unsqueeze(-2))
    )
    allowed = ~forbidden
    diagonal = torch.eye(allowed.shape[-1], device=allowed.device, dtype=torch.bool).unsqueeze(0)
    allowed = allowed & ~diagonal
    return allowed[0] if allowed.shape[0] == 1 else allowed


def _logits_from_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    clamped = adjacency.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(clamped) - torch.log1p(-clamped)


def _build_adjacency(
    edge_logits: torch.Tensor,
    roles: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    edge_i, edge_j = upper_triangle_edge_index(num_nodes, edge_logits.device)
    if roles.ndim == 1:
        upper_mask = _allowed_edge_mask(roles)[edge_i, edge_j]
        adjacency = torch.zeros((num_nodes, num_nodes), device=edge_logits.device, dtype=edge_logits.dtype)
        adjacency[edge_i[upper_mask], edge_j[upper_mask]] = torch.sigmoid(edge_logits)
        return symmetrize_matrix(adjacency)

    upper_mask = _allowed_edge_mask(roles[0])[edge_i, edge_j]
    adjacency = torch.zeros((roles.shape[0], num_nodes, num_nodes), device=edge_logits.device, dtype=edge_logits.dtype)
    adjacency[:, edge_i[upper_mask], edge_j[upper_mask]] = torch.sigmoid(edge_logits)
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
    squeeze = False
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
        roles = roles.unsqueeze(0)
        squeeze = True
    _, _, free_mask = role_masks(roles)
    below = positions.clamp_max(0.0).square().sum(dim=-1)
    above = (positions - 1.0).clamp_min(0.0).square().sum(dim=-1)
    violations = (below + above) * free_mask.to(dtype=positions.dtype)
    free_count = free_mask.sum(dim=1).clamp_min(1)
    penalty = violations.sum(dim=1) / free_count.to(dtype=positions.dtype)
    return penalty[0] if squeeze else penalty


def _anchor_attachment_penalty(adjacency: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if adjacency.ndim == 2:
        adjacency = adjacency.unsqueeze(0)
        roles = roles.unsqueeze(0)
        squeeze = True
    fixed_mask, mobile_mask, free_mask = role_masks(roles)
    penalties = []
    for batch_index in range(adjacency.shape[0]):
        anchor_mask = fixed_mask[batch_index] | mobile_mask[batch_index]
        anchor_to_free = adjacency[batch_index][anchor_mask][:, free_mask[batch_index]]
        if anchor_to_free.numel() == 0:
            penalties.append(torch.tensor(0.0, device=adjacency.device, dtype=adjacency.dtype))
            continue
        attachment = anchor_to_free.sum(dim=1)
        penalties.append((1.5 - attachment).clamp_min(0.0).square().mean())
    penalty = torch.stack(penalties)
    return penalty[0] if squeeze else penalty


def _stiffness_loss(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    normalized_current = normalize_generalized_stiffness(current)
    normalized_target = normalize_generalized_stiffness(target)
    if normalized_target.ndim == 2:
        scale = normalized_target.abs().amax().clamp_min(1e-3)
        return ((normalized_current - normalized_target) / scale).square().mean()
    scale = normalized_target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    return ((normalized_current - normalized_target) / scale).square().mean(dim=(-2, -1))


def _psd_penalty(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(normalize_generalized_stiffness(matrix))
    if eigenvalues.ndim == 1:
        return (-eigenvalues).clamp_min(0.0).square().mean()
    return (-eigenvalues).clamp_min(0.0).square().mean(dim=-1)


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
    generalized_stiffness = terms["generalized_stiffness"]
    weights = config.weights
    sparsity = structures.adjacency.mean(dim=(-2, -1))
    breakdown = {
        "stiffness_loss": weights.stiffness * _stiffness_loss(generalized_stiffness, target_stiffness),
        "material_loss": weights.material * terms["material_usage"],
        "sparsity_loss": weights.sparsity * sparsity,
        "short_beam_loss": weights.short_beam * terms["short_beam_penalty"],
        "long_beam_loss": weights.long_beam * terms["long_beam_penalty"],
        "thin_beam_loss": weights.thin_beam * terms["thin_beam_penalty"],
        "thick_beam_loss": weights.thick_beam * terms["thick_beam_penalty"],
        "free_spacing_loss": weights.free_spacing * terms["free_node_spacing_penalty"],
        "domain_loss": weights.domain * _domain_penalty(structures.positions, structures.roles),
        "anchor_attachment_loss": weights.anchor_attachment * _anchor_attachment_penalty(structures.adjacency, structures.roles),
        "psd_loss": weights.psd * _psd_penalty(generalized_stiffness),
    }
    breakdown["total_loss_per_case"] = torch.stack(tuple(breakdown.values())).sum(dim=0)
    breakdown["total_loss"] = breakdown["total_loss_per_case"].mean()
    breakdown["generalized_stiffness"] = generalized_stiffness
    breakdown["material_usage"] = terms["material_usage"]
    breakdown["short_beam_penalty"] = terms["short_beam_penalty"]
    breakdown["long_beam_penalty"] = terms["long_beam_penalty"]
    breakdown["thin_beam_penalty"] = terms["thin_beam_penalty"]
    breakdown["thick_beam_penalty"] = terms["thick_beam_penalty"]
    breakdown["free_node_spacing_penalty"] = terms["free_node_spacing_penalty"]
    return breakdown


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

    writer = SummaryWriter(log_dir=str(logdir)) if logdir is not None else None
    initial_structures = structures
    free_mask = role_masks(initial_structures.roles)[2]
    free_positions = torch.nn.Parameter(initial_structures.positions.clone())
    edge_i, edge_j = upper_triangle_edge_index(initial_structures.num_nodes, initial_structures.positions.device)
    active_upper = _allowed_edge_mask(initial_structures.roles[0])[edge_i, edge_j]
    initial_edge_values = initial_structures.adjacency[:, edge_i[active_upper], edge_j[active_upper]]
    edge_logits = torch.nn.Parameter(_logits_from_adjacency(initial_edge_values))
    optimizer = torch.optim.Adam([free_positions, edge_logits], lr=config.learning_rate)

    best_loss = torch.full((structures.batch_size,), float("inf"), dtype=torch.float32, device=structures.positions.device)
    best_positions = initial_structures.positions.detach().clone()
    best_adjacency = initial_structures.adjacency.detach().clone()
    initial_loss: torch.Tensor | None = None
    try:
        for step in range(config.num_steps):
            optimizer.zero_grad()
            positions = initial_structures.positions.clone()
            clamped_free_positions = free_positions.clamp(0.0, 1.0)
            positions[free_mask] = clamped_free_positions[free_mask]
            adjacency = _build_adjacency(
                edge_logits=edge_logits,
                roles=initial_structures.roles,
                num_nodes=initial_structures.num_nodes,
            )
            current_structures = Structures(
                positions=positions,
                roles=initial_structures.roles,
                adjacency=adjacency,
            )
            breakdown = _loss_breakdown(current_structures, target_stiffness, config)
            breakdown["total_loss"].backward()
            optimizer.step()

            current_loss = breakdown["total_loss_per_case"].detach()
            if initial_loss is None:
                initial_loss = current_loss.clone()
            improved = current_loss < best_loss
            best_loss = torch.minimum(best_loss, current_loss)
            improved_positions = improved.unsqueeze(-1).unsqueeze(-1)
            improved_adjacency = improved.unsqueeze(-1).unsqueeze(-1)
            best_positions = torch.where(improved_positions, current_structures.positions.detach(), best_positions)
            best_adjacency = torch.where(improved_adjacency, current_structures.adjacency.detach(), best_adjacency)

            if writer is not None and (step % config.log_every == 0 or step == config.num_steps - 1):
                for name in _LOGGED_BREAKDOWN_NAMES:
                    value = breakdown[name]
                    mean_value = float(value.detach().mean().item()) if value.ndim > 0 else float(value.detach().item())
                    writer.add_scalar(f"dataset/optimization/{name}", mean_value, step)
                writer.flush()
    finally:
        if writer is not None:
            writer.close()

    if initial_loss is None:
        initial_loss = torch.zeros((structures.batch_size,), dtype=torch.float32, device=structures.positions.device)

    best_structures = Structures(
        positions=best_positions,
        roles=initial_structures.roles.detach().clone(),
        adjacency=best_adjacency,
    )
    best_breakdown = _loss_breakdown(best_structures, target_stiffness, config)
    result = OptimizedCases(
        raw_structures=Structures(
            positions=initial_structures.positions.detach().clone(),
            roles=initial_structures.roles.detach().clone(),
            adjacency=initial_structures.adjacency.detach().clone(),
        ),
        target_stiffness=target_stiffness.detach().clone(),
        optimized_structures=best_structures,
        initial_loss=initial_loss.detach().cpu(),
        best_loss=best_loss.detach().cpu(),
        last_analyses=Analyses(
            generalized_stiffness=best_breakdown["generalized_stiffness"].detach().cpu(),
            material_usage=best_breakdown["material_usage"].detach().cpu(),
            short_beam_penalty=best_breakdown["short_beam_penalty"].detach().cpu(),
            long_beam_penalty=best_breakdown["long_beam_penalty"].detach().cpu(),
            thin_beam_penalty=best_breakdown["thin_beam_penalty"].detach().cpu(),
            thick_beam_penalty=best_breakdown["thick_beam_penalty"].detach().cpu(),
            free_node_spacing_penalty=best_breakdown["free_node_spacing_penalty"].detach().cpu(),
        ),
    )
    result.validate()
    return result
