"""Detached, device-local equilibrium factorizations in normalized coordinates."""

from dataclasses import dataclass

import torch


def coordinate_scale(like, dof, length):
    scale = like.new_ones(like.shape[-1])
    dimension = 2 if dof == 3 else 3
    scale.reshape(-1, dof)[:, :dimension] = length
    return scale


@dataclass
class EquilibriumSystem:
    factor: torch.Tensor
    scale: torch.Tensor
    valid: torch.Tensor
    pivots: torch.Tensor | None = None

    def solve(self, rhs):
        vector = rhs.ndim == self.factor.ndim - 1
        rhs = rhs[..., None] if vector else rhs
        scaled = rhs * self.scale[:, None]
        scaled = torch.where(self.valid[..., None, None], scaled, 0)
        if self.pivots is None:
            result = torch.cholesky_solve(scaled, self.factor)
        else:
            result = torch.linalg.lu_solve(self.factor, self.pivots, scaled)
        result = result * self.scale[:, None]
        return result.squeeze(-1) if vector else result


def factor_system(matrix, scale):
    from .mechanics_work import record

    record("factorizations")
    matrix = matrix.detach()
    scaled = matrix * scale[:, None] * scale[None, :]
    finite = torch.isfinite(scaled).all((-2, -1))
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    scaled = torch.where(finite[..., None, None], scaled, eye)
    if matrix.device.type == "cpu":
        factor, info = torch.linalg.cholesky_ex(scaled, check_errors=False)
        valid = finite & info.eq(0)
        factor = torch.where(valid[..., None, None], factor, eye)
        return EquilibriumSystem(factor, scale, valid)
    # Intel XPU Cholesky is not supported in this environment. Keep validity
    # and solve on-device; never trigger an implicit fallback to CPU.
    from .spectral import positive_definite

    valid = finite & positive_definite(scaled)
    safe = torch.where(valid[..., None, None], scaled, eye)
    factor, pivots, info = torch.linalg.lu_factor_ex(safe, check_errors=False)
    valid &= info.eq(0)
    return EquilibriumSystem(factor, scale, valid, pivots)


def scaled_descent(matrix, gradient, dof, length, regularization):
    scale = coordinate_scale(matrix, dof, length)
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    regularized = matrix + regularization * eye
    system = factor_system(regularized, scale)
    step = -system.solve(gradient)
    failed = (
        ~system.valid | ~torch.isfinite(step).all(-1) | ((step * gradient).sum(-1) >= 0)
    )
    if bool(failed.any()):
        scaled = regularized * scale[:, None] * scale[None, :]
        diagonal = scaled.diagonal(dim1=-2, dim2=-1)
        lower = (diagonal - (scaled.abs().sum(-1) - diagonal.abs())).amin(-1)
        shift = (-lower).clamp_min(0) + scaled.abs().amax((-2, -1)).clamp_min(1) * 1e-8
        fallback = (
            torch.linalg.solve(scaled + shift[..., None, None] * eye, -gradient * scale)
            * scale
        )
        step = torch.where(failed[..., None], fallback, step)
    return step
