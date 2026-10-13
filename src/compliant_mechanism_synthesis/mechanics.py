"""SI-unit, float64 frame FEM with clamping and independent rigid-body ties."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Material:
    young_modulus: float = 2.1e11
    poisson_ratio: float = 0.3
    regularization: float = 1e-9

    @property
    def shear_modulus(self):
        return self.young_modulus / (2 * (1 + self.poisson_ratio))


DEFAULT_MATERIAL = Material()


@dataclass
class Design:
    positions: torch.Tensor
    roles: tuple[str, ...]
    edges: torch.Tensor
    radii: torch.Tensor
    presence: torch.Tensor

    @property
    def dimension(self):
        return self.positions.shape[1]

    def validate(self):
        n, dim = self.positions.shape
        if dim not in (2, 3) or len(self.roles) != n:
            raise ValueError("Expected N×2 or N×3 positions and N roles")
        if any(r not in ("fixed", "active", "passive", "free") for r in self.roles):
            raise ValueError("Unknown node role")
        if "fixed" not in self.roles or "passive" not in self.roles:
            raise ValueError("At least one fixed and one passive node required")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Expected E×2 edges")
        if (
            self.radii.shape != (len(self.edges),)
            or self.presence.shape != self.radii.shape
        ):
            raise ValueError("One radius and presence per edge required")
        if (
            (self.edges < 0).any()
            or (self.edges >= n).any()
            or (self.edges[:, 0] == self.edges[:, 1]).any()
        ):
            raise ValueError("Invalid edge indices")
        if len(torch.unique(torch.sort(self.edges, dim=1).values, dim=0)) != len(
            self.edges
        ):
            raise ValueError("Duplicate edges")
        if not all(
            torch.isfinite(t).all() for t in (self.positions, self.radii, self.presence)
        ):
            raise ValueError("Nonfinite design")
        if (
            (self.radii <= 0).any()
            or ((self.presence != 0) & (self.presence != 1)).any()
            or self.presence.requires_grad
        ):
            raise ValueError(
                "Radii must be positive; beam presence must be binary and non-differentiable"
            )
        if (
            torch.linalg.vector_norm(
                self.positions[self.edges[:, 1]] - self.positions[self.edges[:, 0]],
                dim=1,
            )
            < 1e-8
        ).any():
            raise ValueError("Coincident beam endpoints")


def assemble(design: Design, material: Material = DEFAULT_MATERIAL):
    from .elements import _frame_local_stiffness, _frame_transform, _local_axes

    design.validate()
    p = design.positions
    spatial = (
        p if design.dimension == 3 else torch.cat((p, torch.zeros_like(p[:, :1])), 1)
    )
    delta = spatial[design.edges[:, 1]] - spatial[design.edges[:, 0]]
    length = torch.linalg.vector_norm(delta, dim=1)
    area = math.pi * design.radii.square()
    inertia = math.pi * design.radii.pow(4) / 4
    local = _frame_local_stiffness(
        length, area, inertia, inertia, 2 * inertia, material
    )
    transform = _frame_transform(_local_axes(delta, length))
    element = (transform.transpose(-1, -2) @ local @ transform) * design.presence[
        :, None, None
    ]
    if design.dimension == 2:
        selected = torch.tensor([0, 1, 5, 6, 7, 11], device=p.device)
        element = element[:, selected][:, :, selected]
    dof = 3 if design.dimension == 2 else 6
    indices = (
        design.edges[:, :, None] * dof + torch.arange(dof, device=p.device)
    ).reshape(-1, 2 * dof)
    total = len(p) * dof
    linear = (indices[:, :, None] * total + indices[:, None, :]).reshape(-1)
    full = (
        p.new_zeros(total * total)
        .scatter_add(0, linear, element.reshape(-1))
        .reshape(total, total)
    )
    return (full + full.T) / 2


def rigid_roles(roles):
    return tuple(role for role in ("passive", "active") if role in roles)


def reduction(design: Design):
    p = design.positions
    dof = 3 if design.dimension == 2 else 6
    free = [i for i, r in enumerate(design.roles) if r == "free"]
    bodies = rigid_roles(design.roles)
    t = p.new_zeros(len(p) * dof, (len(free) + len(bodies)) * dof)
    eye = torch.eye(dof, dtype=p.dtype, device=p.device)
    for k, i in enumerate(free):
        t[i * dof : (i + 1) * dof, k * dof : (k + 1) * dof] = eye
    for k, role in enumerate(bodies):
        ids = [i for i, r in enumerate(design.roles) if r == role]
        center = p[ids].mean(0)
        offset = (len(free) + k) * dof
        for i in ids:
            block = eye.clone()
            r = p[i] - center
            if design.dimension == 2:
                block[0, 2], block[1, 2] = -r[1], r[0]
            else:
                zero = r[0] * 0
                skew = torch.stack(
                    (zero, -r[2], r[1], r[2], zero, -r[0], -r[1], r[0], zero)
                ).reshape(3, 3)
                block[:3, 3:] = -skew
            t[i * dof : (i + 1) * dof, offset : offset + dof] = block
    return t, t.shape[1] - dof


@dataclass
class Response:
    stiffness: torch.Tensor
    displacement: torch.Tensor
    output: torch.Tensor
    volume: torch.Tensor
    residual: torch.Tensor


def solve(design: Design, load: torch.Tensor, material: Material = DEFAULT_MATERIAL):
    full = assemble(design, material)
    t, base = reduction(design)
    reduced = t.T @ full @ t
    regularized = reduced + material.regularization * torch.eye(
        len(reduced), dtype=full.dtype, device=full.device
    )
    rhs = torch.cat((load.new_zeros(base), load))
    q = torch.linalg.solve(regularized, rhs)
    a, b = regularized[:base, :base], regularized[:base, base:]
    effective = (
        regularized[base:, base:] - b.T @ torch.linalg.solve(a, b)
        if base
        else regularized
    )
    length = torch.linalg.vector_norm(
        design.positions[design.edges[:, 1]] - design.positions[design.edges[:, 0]],
        dim=1,
    )
    volume = (math.pi * design.radii.square() * length * design.presence).sum()
    return Response(
        effective,
        (t @ q).reshape(len(design.positions), -1),
        q[base:],
        volume,
        torch.linalg.vector_norm(reduced @ q - rhs),
    )
