from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CharacteristicScales:
    length: float
    force: float
    moment: float
    stress: float
    material: float


def center_positions(positions: torch.Tensor) -> torch.Tensor:
    return 2.0 * positions - 1.0


def uncenter_positions(centered_positions: torch.Tensor) -> torch.Tensor:
    return 0.5 * (centered_positions + 1.0)


def _generalized_displacement_scale(
    scales: CharacteristicScales,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [scales.length, scales.length, 1.0],
        device=device,
        dtype=dtype,
    )


def _generalized_force_scale(
    scales: CharacteristicScales,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [scales.force, scales.force, scales.moment],
        device=device,
        dtype=dtype,
    )


def normalize_generalized_response_matrix(
    response_matrix: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    displacement_scale = _generalized_displacement_scale(
        scales,
        device=response_matrix.device,
        dtype=response_matrix.dtype,
    )
    force_scale = _generalized_force_scale(
        scales,
        device=response_matrix.device,
        dtype=response_matrix.dtype,
    )
    return response_matrix * (
        force_scale.unsqueeze(-2) / displacement_scale.unsqueeze(-1)
    )


def denormalize_generalized_response_matrix(
    normalized_response_matrix: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    displacement_scale = _generalized_displacement_scale(
        scales,
        device=normalized_response_matrix.device,
        dtype=normalized_response_matrix.dtype,
    )
    force_scale = _generalized_force_scale(
        scales,
        device=normalized_response_matrix.device,
        dtype=normalized_response_matrix.dtype,
    )
    return normalized_response_matrix * (
        displacement_scale.unsqueeze(-1) / force_scale.unsqueeze(-2)
    )


def normalize_generalized_stiffness_matrix(
    stiffness_matrix: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    displacement_scale = _generalized_displacement_scale(
        scales,
        device=stiffness_matrix.device,
        dtype=stiffness_matrix.dtype,
    )
    force_scale = _generalized_force_scale(
        scales,
        device=stiffness_matrix.device,
        dtype=stiffness_matrix.dtype,
    )
    return stiffness_matrix * (
        displacement_scale.unsqueeze(-2) / force_scale.unsqueeze(-1)
    )


def denormalize_generalized_stiffness_matrix(
    normalized_stiffness_matrix: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    displacement_scale = _generalized_displacement_scale(
        scales,
        device=normalized_stiffness_matrix.device,
        dtype=normalized_stiffness_matrix.dtype,
    )
    force_scale = _generalized_force_scale(
        scales,
        device=normalized_stiffness_matrix.device,
        dtype=normalized_stiffness_matrix.dtype,
    )
    return normalized_stiffness_matrix * (
        force_scale.unsqueeze(-1) / displacement_scale.unsqueeze(-2)
    )


def normalize_translations(
    translations: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    return translations / scales.length


def normalize_stress(
    nodal_stress: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    return nodal_stress / scales.stress


def normalize_material(
    material: torch.Tensor,
    scales: CharacteristicScales,
) -> torch.Tensor:
    return material / scales.material
