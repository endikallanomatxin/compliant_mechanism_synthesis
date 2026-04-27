from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.adjacency import (
    allowed_edge_mask,
    build_adjacency,
    logits_from_adjacency,
    split_legacy_adjacency,
)
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Scaffolds,
    Structures,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    PrimitiveConfig,
    materialize_scaffold,
)
from compliant_mechanism_synthesis.losses import (
    psd_penalty,
    stiffness_interest_loss,
    stiffness_step_loss,
)
from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    mechanical_terms,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import upper_triangle_edge_index


@dataclass(frozen=True)
class OptimizationLossWeights:
    stiffness: float = 1.0
    stiffness_interest: float = 0.02
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
    scaffold_num_steps: int = 8
    scaffold_learning_rate: float = 5e-4
    num_steps: int = 32
    learning_rate: float = 1e-3
    log_every: int = 10
    weights: OptimizationLossWeights = OptimizationLossWeights()
    mechanics: Frame3DConfig = Frame3DConfig()
    geometry: GeometryPenaltyConfig = GeometryPenaltyConfig()


_LOGGED_BREAKDOWN_NAMES = (
    "stiffness_loss",
    "stiffness_interest_loss",
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


def _anchor_attachment_penalty(
    adjacency: torch.Tensor, roles: torch.Tensor
) -> torch.Tensor:
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
            penalties.append(
                torch.tensor(0.0, device=adjacency.device, dtype=adjacency.dtype)
            )
            continue
        attachment = anchor_to_free.sum(dim=1)
        penalties.append((1.5 - attachment).clamp_min(0.0).square().mean())
    penalty = torch.stack(penalties)
    return penalty[0] if squeeze else penalty


def _loss_breakdown(
    structures: Structures,
    previous_stiffness: torch.Tensor,
    config: CaseOptimizationConfig,
) -> dict[str, torch.Tensor]:
    structures.validate()
    terms = mechanical_terms(
        positions=structures.positions,
        roles=structures.roles,
        adjacency=structures.adjacency,
        edge_radius=structures.edge_radius,
        frame_config=config.mechanics,
        penalty_config=config.geometry,
    )
    generalized_stiffness = terms["generalized_stiffness"]
    weights = config.weights
    sparsity = structures.adjacency.mean(dim=(-2, -1))
    breakdown = {
        "stiffness_loss": weights.stiffness
        * stiffness_step_loss(generalized_stiffness, previous_stiffness),
        "stiffness_interest_loss": weights.stiffness_interest
        * stiffness_interest_loss(generalized_stiffness),
        "material_loss": weights.material * terms["material_usage"],
        "sparsity_loss": weights.sparsity * sparsity,
        "short_beam_loss": weights.short_beam * terms["short_beam_penalty"],
        "long_beam_loss": weights.long_beam * terms["long_beam_penalty"],
        "thin_beam_loss": weights.thin_beam * terms["thin_beam_penalty"],
        "thick_beam_loss": weights.thick_beam * terms["thick_beam_penalty"],
        "free_spacing_loss": weights.free_spacing * terms["free_node_spacing_penalty"],
        "domain_loss": weights.domain
        * _domain_penalty(structures.positions, structures.roles),
        "anchor_attachment_loss": weights.anchor_attachment
        * _anchor_attachment_penalty(structures.adjacency, structures.roles),
        "psd_loss": weights.psd * psd_penalty(generalized_stiffness),
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
    breakdown["nodal_displacements"] = terms["nodal_displacements"]
    breakdown["edge_von_mises"] = terms["edge_von_mises"]
    return breakdown


def optimize_cases(
    structures: Structures,
    config: CaseOptimizationConfig | None = None,
    logdir: str | Path | None = None,
) -> OptimizedCases:
    config = config or CaseOptimizationConfig()
    structures.validate()

    writer = SummaryWriter(log_dir=str(logdir)) if logdir is not None else None
    initial_structures = structures
    initial_edge_radius = (
        initial_structures.edge_radius
        if initial_structures.edge_radius is not None
        else split_legacy_adjacency(initial_structures.adjacency)[1]
    )
    free_mask = role_masks(initial_structures.roles)[2]
    free_positions = torch.nn.Parameter(initial_structures.positions.clone())
    edge_i, edge_j = upper_triangle_edge_index(
        initial_structures.num_nodes, initial_structures.positions.device
    )
    active_upper = allowed_edge_mask(initial_structures.roles[0])[edge_i, edge_j]
    initial_edge_values = initial_structures.adjacency[
        :, edge_i[active_upper], edge_j[active_upper]
    ]
    edge_logits = torch.nn.Parameter(logits_from_adjacency(initial_edge_values))
    optimizer = torch.optim.Adam([free_positions, edge_logits], lr=config.learning_rate)
    previous_terms = mechanical_terms(
        positions=initial_structures.positions,
        roles=initial_structures.roles,
        adjacency=initial_structures.adjacency,
        edge_radius=initial_edge_radius,
        frame_config=config.mechanics,
        penalty_config=config.geometry,
    )
    previous_stiffness = previous_terms["generalized_stiffness"].detach()

    best_loss = torch.full(
        (structures.batch_size,),
        float("inf"),
        dtype=torch.float32,
        device=structures.positions.device,
    )
    best_positions = initial_structures.positions.detach().clone()
    best_adjacency = initial_structures.adjacency.detach().clone()
    initial_loss: torch.Tensor | None = None
    try:
        for step in range(config.num_steps):
            optimizer.zero_grad()
            positions = initial_structures.positions.clone()
            clamped_free_positions = free_positions.clamp(0.0, 1.0)
            positions[free_mask] = clamped_free_positions[free_mask]
            adjacency = build_adjacency(
                edge_logits=edge_logits,
                roles=initial_structures.roles,
                num_nodes=initial_structures.num_nodes,
            )
            current_structures = Structures(
                positions=positions,
                roles=initial_structures.roles,
                adjacency=adjacency,
                edge_radius=initial_edge_radius,
            )
            breakdown = _loss_breakdown(
                current_structures,
                previous_stiffness,
                config,
            )
            breakdown["total_loss"].backward()
            optimizer.step()
            previous_stiffness = breakdown["generalized_stiffness"].detach()

            current_loss = breakdown["total_loss_per_case"].detach()
            if initial_loss is None:
                initial_loss = current_loss.clone()
            improved = current_loss < best_loss
            best_loss = torch.minimum(best_loss, current_loss)
            improved_positions = improved.unsqueeze(-1).unsqueeze(-1)
            improved_adjacency = improved.unsqueeze(-1).unsqueeze(-1)
            best_positions = torch.where(
                improved_positions,
                current_structures.positions.detach(),
                best_positions,
            )
            best_adjacency = torch.where(
                improved_adjacency,
                current_structures.adjacency.detach(),
                best_adjacency,
            )

            if writer is not None and (
                step % config.log_every == 0 or step == config.num_steps - 1
            ):
                for name in _LOGGED_BREAKDOWN_NAMES:
                    value = breakdown[name]
                    mean_value = (
                        float(value.detach().mean().item())
                        if value.ndim > 0
                        else float(value.detach().item())
                    )
                    writer.add_scalar(f"dataset/optimization/{name}", mean_value, step)
                writer.flush()
    finally:
        if writer is not None:
            writer.close()

    if initial_loss is None:
        initial_loss = torch.zeros(
            (structures.batch_size,),
            dtype=torch.float32,
            device=structures.positions.device,
        )

    best_structures = Structures(
        positions=best_positions,
        roles=initial_structures.roles.detach().clone(),
        adjacency=best_adjacency,
        edge_radius=initial_edge_radius.detach().clone(),
    )
    best_breakdown = _loss_breakdown(
        best_structures,
        previous_stiffness,
        config,
    )
    result = OptimizedCases(
        target_stiffness=best_breakdown["generalized_stiffness"].detach().cpu().clone(),
        optimized_structures=best_structures,
        initial_loss=initial_loss.detach().cpu(),
        best_loss=best_loss.detach().cpu(),
        last_analyses=Analyses(
            generalized_stiffness=best_breakdown["generalized_stiffness"]
            .detach()
            .cpu(),
            material_usage=best_breakdown["material_usage"].detach().cpu(),
            short_beam_penalty=best_breakdown["short_beam_penalty"].detach().cpu(),
            long_beam_penalty=best_breakdown["long_beam_penalty"].detach().cpu(),
            thin_beam_penalty=best_breakdown["thin_beam_penalty"].detach().cpu(),
            thick_beam_penalty=best_breakdown["thick_beam_penalty"].detach().cpu(),
            free_node_spacing_penalty=best_breakdown["free_node_spacing_penalty"]
            .detach()
            .cpu(),
            nodal_displacements=best_breakdown["nodal_displacements"].detach().cpu(),
            edge_von_mises=best_breakdown["edge_von_mises"].detach().cpu(),
        ),
    )
    result = result.to("cpu")
    result.validate()
    return result


def optimize_scaffolds(
    scaffolds: Scaffolds,
    primitive_config: PrimitiveConfig,
    config: CaseOptimizationConfig | None = None,
    logdir: str | Path | None = None,
) -> tuple[Scaffolds, Structures]:
    config = config or CaseOptimizationConfig()
    scaffolds.validate()
    if config.scaffold_num_steps <= 0:
        structures = materialize_scaffold(scaffolds, config=primitive_config)
        return scaffolds, structures

    def _scaffold_with_positions(positions: torch.Tensor) -> Scaffolds:
        return Scaffolds(
            positions=positions,
            roles=scaffolds.roles,
            adjacency=scaffolds.adjacency,
            edge_primitive_ids=scaffolds.edge_primitive_ids,
            edge_primitive_types=scaffolds.edge_primitive_types,
            edge_sheet_width_nodes=scaffolds.edge_sheet_width_nodes,
            edge_orientation_start=scaffolds.edge_orientation_start,
            edge_orientation_end=scaffolds.edge_orientation_end,
            edge_offset_start=scaffolds.edge_offset_start,
            edge_offset_end=scaffolds.edge_offset_end,
            edge_helix_phase=scaffolds.edge_helix_phase,
            edge_helix_pitch=scaffolds.edge_helix_pitch,
            edge_width_start=scaffolds.edge_width_start,
            edge_width_end=scaffolds.edge_width_end,
            edge_thickness_start=scaffolds.edge_thickness_start,
            edge_thickness_end=scaffolds.edge_thickness_end,
            edge_twist_start=scaffolds.edge_twist_start,
            edge_twist_end=scaffolds.edge_twist_end,
            edge_sweep_phase=scaffolds.edge_sweep_phase,
            edge_radius=scaffolds.edge_radius,
        )

    writer = SummaryWriter(log_dir=str(logdir)) if logdir is not None else None
    free_mask = role_masks(scaffolds.roles)[2]
    free_positions = torch.nn.Parameter(scaffolds.positions.clone())
    optimizer = torch.optim.Adam([free_positions], lr=config.scaffold_learning_rate)
    initial_structures = materialize_scaffold(scaffolds, config=primitive_config)
    initial_terms = mechanical_terms(
        positions=initial_structures.positions,
        roles=initial_structures.roles,
        adjacency=initial_structures.adjacency,
        edge_radius=initial_structures.edge_radius,
        frame_config=config.mechanics,
        penalty_config=config.geometry,
    )
    previous_stiffness = initial_terms["generalized_stiffness"].detach()

    best_loss = torch.full(
        (scaffolds.batch_size,),
        float("inf"),
        dtype=torch.float32,
        device=scaffolds.positions.device,
    )
    best_positions = scaffolds.positions.detach().clone()
    best_structures = Structures(
        positions=initial_structures.positions.detach(),
        roles=initial_structures.roles.detach(),
        adjacency=initial_structures.adjacency.detach(),
        edge_radius=(
            None
            if initial_structures.edge_radius is None
            else initial_structures.edge_radius.detach()
        ),
    )
    try:
        for step in range(config.scaffold_num_steps):
            optimizer.zero_grad()
            clamped_positions = torch.where(
                free_mask.unsqueeze(-1),
                free_positions.clamp(0.0, 1.0),
                scaffolds.positions,
            )
            current_scaffolds = _scaffold_with_positions(clamped_positions)
            current_structures = materialize_scaffold(
                current_scaffolds, config=primitive_config
            )
            breakdown = _loss_breakdown(
                current_structures,
                previous_stiffness,
                config,
            )
            breakdown["total_loss"].backward()
            optimizer.step()
            current_loss = breakdown["total_loss_per_case"].detach()
            previous_stiffness = breakdown["generalized_stiffness"].detach()
            improved = current_loss < best_loss
            best_loss = torch.minimum(best_loss, current_loss)
            best_positions = torch.where(
                improved.unsqueeze(-1).unsqueeze(-1),
                current_scaffolds.positions.detach(),
                best_positions,
            )
            if torch.any(improved):
                best_structures = Structures(
                    positions=current_structures.positions.detach(),
                    roles=current_structures.roles.detach(),
                    adjacency=current_structures.adjacency.detach(),
                    edge_radius=(
                        None
                        if current_structures.edge_radius is None
                        else current_structures.edge_radius.detach()
                    ),
                )

            if writer is not None and (
                step % config.log_every == 0 or step == config.scaffold_num_steps - 1
            ):
                for name in _LOGGED_BREAKDOWN_NAMES:
                    value = breakdown[name]
                    mean_value = (
                        float(value.detach().mean().item())
                        if value.ndim > 0
                        else float(value.detach().item())
                    )
                    writer.add_scalar(
                        f"dataset/optimization/scaffold/{name}",
                        mean_value,
                        step,
                    )
                writer.flush()
    finally:
        if writer is not None:
            writer.close()

    best_scaffolds = Scaffolds(
        positions=best_positions,
        roles=scaffolds.roles.detach().clone(),
        adjacency=scaffolds.adjacency.detach().clone(),
        edge_primitive_ids=scaffolds.edge_primitive_ids.detach().clone(),
        edge_primitive_types=scaffolds.edge_primitive_types.detach().clone(),
        edge_sheet_width_nodes=scaffolds.edge_sheet_width_nodes.detach().clone(),
        edge_orientation_start=scaffolds.edge_orientation_start.detach().clone(),
        edge_orientation_end=scaffolds.edge_orientation_end.detach().clone(),
        edge_offset_start=scaffolds.edge_offset_start.detach().clone(),
        edge_offset_end=scaffolds.edge_offset_end.detach().clone(),
        edge_helix_phase=scaffolds.edge_helix_phase.detach().clone(),
        edge_helix_pitch=scaffolds.edge_helix_pitch.detach().clone(),
        edge_width_start=scaffolds.edge_width_start.detach().clone(),
        edge_width_end=scaffolds.edge_width_end.detach().clone(),
        edge_thickness_start=scaffolds.edge_thickness_start.detach().clone(),
        edge_thickness_end=scaffolds.edge_thickness_end.detach().clone(),
        edge_twist_start=scaffolds.edge_twist_start.detach().clone(),
        edge_twist_end=scaffolds.edge_twist_end.detach().clone(),
        edge_sweep_phase=scaffolds.edge_sweep_phase.detach().clone(),
    )
    best_scaffolds.validate()
    best_structures.validate()
    return best_scaffolds, best_structures
