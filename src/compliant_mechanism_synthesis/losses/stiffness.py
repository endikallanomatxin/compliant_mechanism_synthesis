from __future__ import annotations

import torch

from compliant_mechanism_synthesis.mechanics import normalize_generalized_stiffness


def generalized_stiffness_error(
    generalized_stiffness: torch.Tensor,
    target_stiffness: torch.Tensor,
) -> torch.Tensor:
    normalized_current = normalize_generalized_stiffness(generalized_stiffness)
    normalized_target = normalize_generalized_stiffness(target_stiffness)
    return (normalized_current - normalized_target).square().mean(dim=(-2, -1))


def log_generalized_stiffness_error(
    generalized_stiffness: torch.Tensor,
    target_stiffness: torch.Tensor,
) -> torch.Tensor:
    return torch.log1p(
        generalized_stiffness_error(generalized_stiffness, target_stiffness)
    )


def stiffness_step_loss(
    current: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    normalized_current = normalize_generalized_stiffness(current)
    normalized_previous = normalize_generalized_stiffness(previous)
    if normalized_previous.ndim == 2:
        scale = normalized_previous.abs().amax().clamp_min(1e-3)
        return ((normalized_current - normalized_previous) / scale).square().mean()
    scale = normalized_previous.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    return (
        ((normalized_current - normalized_previous) / scale).square().mean(dim=(-2, -1))
    )


def psd_penalty(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(normalize_generalized_stiffness(matrix))
    if eigenvalues.ndim == 1:
        return (-eigenvalues).clamp_min(0.0).square().mean()
    return (-eigenvalues).clamp_min(0.0).square().mean(dim=-1)


def stiffness_interest_loss(matrix: torch.Tensor) -> torch.Tensor:
    normalized = normalize_generalized_stiffness(matrix)
    eigenvalues = torch.linalg.eigvalsh(normalized)
    if eigenvalues.ndim == 1:
        return normalized.new_zeros((1,))
    if eigenvalues.shape[0] <= 1:
        return normalized.new_zeros((eigenvalues.shape[0],))
    batch_mean = eigenvalues.mean(dim=0, keepdim=True)
    return -(eigenvalues - batch_mean).square().mean(dim=-1)
