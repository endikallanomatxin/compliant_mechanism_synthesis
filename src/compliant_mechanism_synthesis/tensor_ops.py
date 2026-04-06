from __future__ import annotations

import torch

from compliant_mechanism_synthesis.roles import role_masks


def symmetrize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    diagonal = torch.diagonal(symmetric, dim1=-2, dim2=-1)
    return symmetric - torch.diag_embed(diagonal)


def upper_triangle_edge_index(num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    return edge_index[0], edge_index[1]


def symmetric_matrix_unique_values(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    index = torch.triu_indices(size, size, offset=0, device=matrix.device)
    return matrix[..., index[0], index[1]]


def distance_affinity(
    positions: torch.Tensor,
    length_scale: float = 0.22,
) -> torch.Tensor:
    pairwise = torch.linalg.vector_norm(
        positions[:, :, None, :] - positions[:, None, :, :],
        dim=-1,
    )
    affinity = torch.exp(-pairwise.square() / max(length_scale**2, 1e-8))
    return symmetrize_matrix(affinity)


def max_length_gate(
    positions: torch.Tensor,
    max_distance: float = 0.24,
    transition_width: float = 0.08,
) -> torch.Tensor:
    pairwise = torch.linalg.vector_norm(
        positions[:, :, None, :] - positions[:, None, :, :],
        dim=-1,
    )
    excess = (pairwise - max_distance).clamp_min(0.0)
    gate = torch.exp(-excess.square() / max(transition_width**2, 1e-8))
    return symmetrize_matrix(gate)


def enforce_role_adjacency_constraints(
    adjacency: torch.Tensor,
    roles: torch.Tensor,
) -> torch.Tensor:
    adjacency = symmetrize_matrix(adjacency)
    fixed, mobile, _ = role_masks(roles)
    fixed_pair = fixed.unsqueeze(-1) & fixed.unsqueeze(-2)
    mobile_pair = mobile.unsqueeze(-1) & mobile.unsqueeze(-2)
    fixed_mobile_pair = (fixed.unsqueeze(-1) & mobile.unsqueeze(-2)) | (
        mobile.unsqueeze(-1) & fixed.unsqueeze(-2)
    )
    forbidden = fixed_pair | mobile_pair | fixed_mobile_pair
    return adjacency.masked_fill(forbidden, 0.0)
