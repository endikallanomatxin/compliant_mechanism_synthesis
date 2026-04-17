from __future__ import annotations

import torch


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
