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
