from __future__ import annotations

import torch


ROLE_FIXED = 0
ROLE_MOBILE = 1
ROLE_FREE = 2
NUM_ROLES = 3


def role_masks(roles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed = roles == ROLE_FIXED
    mobile = roles == ROLE_MOBILE
    free = roles == ROLE_FREE
    return fixed, mobile, free


def symmetrize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    diagonal = torch.diagonal(symmetric, dim1=-2, dim2=-1)
    return symmetric - torch.diag_embed(diagonal)


def enforce_role_adjacency_constraints(
    adjacency: torch.Tensor,
    roles: torch.Tensor,
) -> torch.Tensor:
    adjacency = symmetrize_adjacency(adjacency)
    fixed, mobile, _ = role_masks(roles)
    fixed_pair = fixed.unsqueeze(-1) & fixed.unsqueeze(-2)
    mobile_pair = mobile.unsqueeze(-1) & mobile.unsqueeze(-2)
    fixed_mobile_pair = (fixed.unsqueeze(-1) & mobile.unsqueeze(-2)) | (
        mobile.unsqueeze(-1) & fixed.unsqueeze(-2)
    )
    forbidden = fixed_pair | mobile_pair | fixed_mobile_pair
    return adjacency.masked_fill(forbidden, 0.0)


def adjacency_logits(adjacency: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    clamped = adjacency.clamp(eps, 1.0 - eps)
    return torch.log(clamped) - torch.log1p(-clamped)


def logits_to_adjacency(logits: torch.Tensor) -> torch.Tensor:
    return symmetrize_adjacency(torch.sigmoid(logits))


def apply_free_node_update(
    positions: torch.Tensor, delta: torch.Tensor, roles: torch.Tensor, step_size: float
) -> torch.Tensor:
    _, _, free = role_masks(roles)
    free_mask = free.unsqueeze(-1).to(dtype=positions.dtype)
    return positions + step_size * torch.tanh(delta) * free_mask


def distance_affinity(
    positions: torch.Tensor,
    length_scale: float = 0.18,
) -> torch.Tensor:
    pairwise = torch.linalg.vector_norm(
        positions[:, :, None, :] - positions[:, None, :, :],
        dim=-1,
    )
    affinity = torch.exp(-pairwise.square() / (length_scale**2))
    return symmetrize_adjacency(affinity)


def max_length_gate(
    positions: torch.Tensor,
    max_distance: float = 0.10,
    transition_width: float = 0.05,
) -> torch.Tensor:
    pairwise = torch.linalg.vector_norm(
        positions[:, :, None, :] - positions[:, None, :, :],
        dim=-1,
    )
    excess = (pairwise - max_distance).clamp_min(0.0)
    gate = torch.exp(-excess.square() / (transition_width**2))
    return symmetrize_adjacency(gate)


def symmetric_matrix_unique_values(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[-1]
    idx = torch.triu_indices(n, n, offset=0, device=matrix.device)
    return matrix[..., idx[0], idx[1]]


def unique_values_to_symmetric_matrix(values: torch.Tensor, size: int) -> torch.Tensor:
    idx = torch.triu_indices(size, size, offset=0, device=values.device)
    matrix = torch.zeros(
        values.shape[:-1] + (size, size),
        device=values.device,
        dtype=values.dtype,
    )
    matrix[..., idx[0], idx[1]] = values
    matrix[..., idx[1], idx[0]] = values
    return matrix
