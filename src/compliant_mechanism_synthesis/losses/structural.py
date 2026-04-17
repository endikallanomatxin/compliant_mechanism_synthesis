from __future__ import annotations

from dataclasses import dataclass

import torch

from compliant_mechanism_synthesis.dataset.types import Analyses
from compliant_mechanism_synthesis.losses.stiffness import (
    log_generalized_stiffness_error,
)


@dataclass(frozen=True)
class StructuralObjectiveWeights:
    stiffness: float
    stress: float
    material: float
    short_beam: float
    long_beam: float
    thin_beam: float
    thick_beam: float
    free_node_spacing: float


def stress_violation_terms(
    edge_von_mises: torch.Tensor,
    adjacency: torch.Tensor,
    allowable_von_mises: float,
    stress_activation_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate per-edge von Mises stress into smooth structure-level penalties."""
    if allowable_von_mises <= 0.0:
        raise ValueError("allowable_von_mises must be positive")
    if not 0.0 <= stress_activation_threshold <= 1.0:
        raise ValueError("stress_activation_threshold must be in [0.0, 1.0]")
    if adjacency.ndim != 3:
        raise ValueError("adjacency must have shape [batch, nodes, nodes]")
    if edge_von_mises.ndim != 4:
        raise ValueError("edge_von_mises must have shape [batch, nodes, nodes, loads]")
    if edge_von_mises.shape[:3] != adjacency.shape:
        raise ValueError(
            "edge_von_mises must have matching [batch, nodes, nodes] dimensions"
        )

    _, num_nodes, _ = adjacency.shape
    edge_i, edge_j = torch.triu_indices(
        num_nodes,
        num_nodes,
        offset=1,
        device=adjacency.device,
    )
    edge_activation = adjacency[:, edge_i, edge_j].clamp_min(0.0)
    activation_weight = (
        torch.sigmoid((edge_activation - stress_activation_threshold) / 0.05)
        * edge_activation
    )
    safe_activation_weight = activation_weight.sum(dim=-1).clamp_min(1.0)
    edge_stresses = torch.nan_to_num(
        edge_von_mises[:, edge_i, edge_j],
        nan=0.0,
        posinf=allowable_von_mises * 1e6,
        neginf=0.0,
    )
    edge_rms_stress = torch.sqrt(edge_stresses.square().mean(dim=-1) + 1e-12)
    stress_ratio = edge_rms_stress / allowable_von_mises
    stress_excess = torch.relu(stress_ratio - 1.0)
    # Use a log-domain penalty to avoid float32 overflow for extreme ratios.
    stress_violation = torch.log1p(stress_excess).square()
    mean_stress_violation = (stress_violation * activation_weight).sum(
        dim=-1
    ) / safe_activation_weight
    mean_stress_ratio = (stress_ratio * activation_weight).sum(
        dim=-1
    ) / safe_activation_weight
    max_stress_ratio = torch.where(
        edge_activation >= stress_activation_threshold,
        stress_ratio,
        torch.zeros_like(stress_ratio),
    ).amax(dim=-1)
    return mean_stress_violation, mean_stress_ratio, max_stress_ratio


def structural_objective_terms(
    analyses: Analyses,
    adjacency: torch.Tensor,
    target_stiffness: torch.Tensor,
    *,
    weights: StructuralObjectiveWeights,
    allowable_von_mises: float,
    stress_activation_threshold: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stiffness_error = log_generalized_stiffness_error(
        analyses.generalized_stiffness,
        target_stiffness,
    )
    material_usage = analyses.material_usage
    short_beam_penalty = analyses.short_beam_penalty
    long_beam_penalty = analyses.long_beam_penalty
    thin_beam_penalty = analyses.thin_beam_penalty
    thick_beam_penalty = analyses.thick_beam_penalty
    free_node_spacing_penalty = analyses.free_node_spacing_penalty
    if analyses.edge_von_mises is None:
        raise ValueError("analyses must provide edge_von_mises")
    stress_violation, mean_stress_ratio, max_stress_ratio = stress_violation_terms(
        edge_von_mises=analyses.edge_von_mises,
        adjacency=adjacency,
        allowable_von_mises=allowable_von_mises,
        stress_activation_threshold=stress_activation_threshold,
    )

    loss_contributions = {
        "stiffness_loss_contribution": weights.stiffness * stiffness_error,
        "stress_loss_contribution": weights.stress * stress_violation,
        "material_loss_contribution": weights.material * material_usage,
        "short_beam_loss_contribution": weights.short_beam * short_beam_penalty,
        "long_beam_loss_contribution": weights.long_beam * long_beam_penalty,
        "thin_beam_loss_contribution": weights.thin_beam * thin_beam_penalty,
        "thick_beam_loss_contribution": weights.thick_beam * thick_beam_penalty,
        "free_node_spacing_loss_contribution": weights.free_node_spacing
        * free_node_spacing_penalty,
    }
    metrics = {
        "stiffness_error": stiffness_error.mean(),
        "stress_violation": stress_violation.mean(),
        "mean_stress_ratio": mean_stress_ratio.mean(),
        "max_stress_ratio": max_stress_ratio.mean(),
        "material_usage": material_usage.mean(),
        "short_beam_penalty": short_beam_penalty.mean(),
        "long_beam_penalty": long_beam_penalty.mean(),
        "thin_beam_penalty": thin_beam_penalty.mean(),
        "thick_beam_penalty": thick_beam_penalty.mean(),
        "free_node_spacing_penalty": free_node_spacing_penalty.mean(),
    }
    return loss_contributions, metrics
