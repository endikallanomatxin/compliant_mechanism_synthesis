"""Euler–Bernoulli spatial beam stiffness kernel."""

import torch

from .mechanics import Material


def _choose_reference_axis(x_axis: torch.Tensor) -> torch.Tensor:
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=x_axis.device, dtype=x_axis.dtype)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=x_axis.device, dtype=x_axis.dtype)
    z_ref = z_axis.expand_as(x_axis)
    y_ref = y_axis.expand_as(x_axis)
    use_y = x_axis[..., 2].abs() > 0.9
    return torch.where(use_y.unsqueeze(-1), y_ref, z_ref)


def _local_axes(delta: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    x_axis = delta / length.unsqueeze(-1)
    reference = _choose_reference_axis(x_axis)
    y_axis = torch.linalg.cross(reference, x_axis, dim=-1)
    y_axis = y_axis / torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-2)


def _frame_local_stiffness(
    length: torch.Tensor,
    area: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    polar_inertia: torch.Tensor,
    config: Material,
) -> torch.Tensor:
    axial = config.young_modulus * area / length
    torsion = config.shear_modulus * polar_inertia / length
    by = 12.0 * config.young_modulus * iy / length.pow(3)
    bz = 12.0 * config.young_modulus * iz / length.pow(3)
    cy = 6.0 * config.young_modulus * iy / length.square()
    cz = 6.0 * config.young_modulus * iz / length.square()
    dy = 4.0 * config.young_modulus * iy / length
    dz = 4.0 * config.young_modulus * iz / length
    ey = 2.0 * config.young_modulus * iy / length
    ez = 2.0 * config.young_modulus * iz / length

    k = torch.zeros(length.shape + (12, 12), device=length.device, dtype=length.dtype)
    k[..., 0, 0] = axial
    k[..., 0, 6] = -axial
    k[..., 6, 0] = -axial
    k[..., 6, 6] = axial

    k[..., 1, 1] = bz
    k[..., 1, 5] = cz
    k[..., 1, 7] = -bz
    k[..., 1, 11] = cz

    k[..., 2, 2] = by
    k[..., 2, 4] = -cy
    k[..., 2, 8] = -by
    k[..., 2, 10] = -cy

    k[..., 3, 3] = torsion
    k[..., 3, 9] = -torsion

    k[..., 4, 2] = -cy
    k[..., 4, 4] = dy
    k[..., 4, 8] = cy
    k[..., 4, 10] = ey

    k[..., 5, 1] = cz
    k[..., 5, 5] = dz
    k[..., 5, 7] = -cz
    k[..., 5, 11] = ez

    k[..., 7, 1] = -bz
    k[..., 7, 5] = -cz
    k[..., 7, 7] = bz
    k[..., 7, 11] = -cz

    k[..., 8, 2] = -by
    k[..., 8, 4] = cy
    k[..., 8, 8] = by
    k[..., 8, 10] = cy

    k[..., 9, 3] = -torsion
    k[..., 9, 9] = torsion

    k[..., 10, 2] = -cy
    k[..., 10, 4] = ey
    k[..., 10, 8] = cy
    k[..., 10, 10] = dy

    k[..., 11, 1] = cz
    k[..., 11, 5] = ez
    k[..., 11, 7] = -cz
    k[..., 11, 11] = dz
    return (k + k.transpose(-1, -2)) / 2


def _frame_transform(rotation: torch.Tensor) -> torch.Tensor:
    transform = torch.zeros(
        rotation.shape[:-2] + (12, 12),
        device=rotation.device,
        dtype=rotation.dtype,
    )
    for block in range(4):
        start = 3 * block
        transform[..., start : start + 3, start : start + 3] = rotation
    return transform
