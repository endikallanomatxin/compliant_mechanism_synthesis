from __future__ import annotations

import torch

from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import (
    symmetrize_matrix,
    upper_triangle_edge_index,
)


def allowed_edge_mask(roles: torch.Tensor) -> torch.Tensor:
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
    diagonal = torch.eye(
        allowed.shape[-1], device=allowed.device, dtype=torch.bool
    ).unsqueeze(0)
    allowed = allowed & ~diagonal
    return allowed[0] if allowed.shape[0] == 1 else allowed


def logits_from_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    clamped = adjacency.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(clamped) - torch.log1p(-clamped)


def build_adjacency(
    edge_logits: torch.Tensor,
    roles: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    edge_i, edge_j = upper_triangle_edge_index(num_nodes, edge_logits.device)
    if roles.ndim == 1:
        upper_mask = allowed_edge_mask(roles)[edge_i, edge_j]
        adjacency = torch.zeros(
            (num_nodes, num_nodes), device=edge_logits.device, dtype=edge_logits.dtype
        )
        adjacency[edge_i[upper_mask], edge_j[upper_mask]] = torch.sigmoid(edge_logits)
        return symmetrize_matrix(adjacency)

    upper_mask = allowed_edge_mask(roles[0])[edge_i, edge_j]
    adjacency = torch.zeros(
        (roles.shape[0], num_nodes, num_nodes),
        device=edge_logits.device,
        dtype=edge_logits.dtype,
    )
    adjacency[:, edge_i[upper_mask], edge_j[upper_mask]] = torch.sigmoid(edge_logits)
    return symmetrize_matrix(adjacency)
