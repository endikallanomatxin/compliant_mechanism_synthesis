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


def adjacency_logits(adjacency: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    clamped = adjacency.clamp(eps, 1.0 - eps)
    return torch.log(clamped) - torch.log1p(-clamped)


def logits_to_adjacency(logits: torch.Tensor) -> torch.Tensor:
    return symmetrize_adjacency(torch.sigmoid(logits))


def clamp_positions(positions: torch.Tensor) -> torch.Tensor:
    return positions.clamp(0.0, 1.0)


def apply_free_node_update(
    positions: torch.Tensor, delta: torch.Tensor, roles: torch.Tensor, step_size: float
) -> torch.Tensor:
    _, _, free = role_masks(roles)
    free_mask = free.unsqueeze(-1).to(dtype=positions.dtype)
    updated = positions + step_size * delta * free_mask
    return clamp_positions(updated)


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
