"""Corotational beam energy and load-controlled equilibrium continuation.

Large rotations, small local strains. Continuation stops at loss of convergence;
load control cannot trace snap-through beyond a limit point.
"""

import math
from dataclasses import dataclass

import torch

from .mechanics import DEFAULT_MATERIAL, Design, Material, reduction, rigid_roles


@dataclass(frozen=True)
class ContinuationConfig:
    steps: int = 12
    beam_segments: int = 1
    hessian_chunk_size: int = 8
    max_iterations: int = 45
    tolerance: float = 1e-7
    minimum_increment: float = 1e-4
    max_backtracks: int = 18
    characteristic_length: float = 0.05
    newton_batch_mode: str = "auto"
    predictor: int = 1
    path_position_tolerance: float = 0.0001
    path_orientation_tolerance: float = 0.01

    def __post_init__(self):
        if type(self.predictor) is not int or self.predictor not in (0, 1):
            raise ValueError("Predictor must be 0 or 1")
        for value in (self.path_position_tolerance, self.path_orientation_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Path tolerances must be positive and finite")
        if self.newton_batch_mode not in ("auto", "compact", "masked"):
            raise ValueError("Newton batch mode must be auto, compact or masked")
        for name in (
            "steps",
            "beam_segments",
            "hessian_chunk_size",
            "max_iterations",
            "max_backtracks",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not math.isfinite(self.tolerance) or not math.isfinite(
            self.minimum_increment
        ):
            raise ValueError("Continuation tolerances must be finite")
        if self.beam_segments < 1:
            raise ValueError("Beam segments must be positive")
        if self.hessian_chunk_size < 1:
            raise ValueError("Hessian chunk size must be positive")
        if self.steps < 1 or self.max_iterations < 1 or self.max_backtracks < 1:
            raise ValueError("Continuation counts must be positive")
        if self.tolerance <= 0 or not 0 < self.minimum_increment <= 1:
            raise ValueError("Invalid continuation tolerances")

        if (
            not math.isfinite(self.characteristic_length)
            or self.characteristic_length <= 0
        ):
            raise ValueError("Characteristic length must be positive and finite")


DEFAULT_CONTINUATION = ContinuationConfig()


def skew(v):
    z = v[..., 0] * 0
    return torch.stack(
        (z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z),
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def rotation(v):
    # Smooth Rodrigues coefficients avoid a zero-angle singularity and keep
    # both first and second derivatives on the selected compute device.
    theta2 = v.square().sum(-1)
    safe = theta2.clamp_min(1e-12)
    theta = torch.sqrt(safe)
    a = torch.where(
        theta2 < 1e-6,
        1 - theta2 / 6 + theta2.square() / 120 - theta2.pow(3) / 5040,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        theta2 < 1e-6,
        0.5 - theta2 / 24 + theta2.square() / 720 - theta2.pow(3) / 40320,
        (1 - torch.cos(theta)) / safe,
    )
    k = skew(v)
    return (
        torch.eye(3, dtype=v.dtype, device=v.device)
        + a[..., None, None] * k
        + b[..., None, None] * (k @ k)
    )


def rotation_vector(r):
    v = (
        torch.stack(
            (
                r[..., 2, 1] - r[..., 1, 2],
                r[..., 0, 2] - r[..., 2, 0],
                r[..., 1, 0] - r[..., 0, 1],
            ),
            dim=-1,
        )
        / 2
    )
    s = torch.sqrt(v.square().sum(-1) + 1e-24)
    c = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return v * (torch.atan2(s, c) / s)[..., None]


def configuration(design, q):
    dof = 3 if design.dimension == 2 else 6
    free = [i for i, r in enumerate(design.roles) if r == "free"]
    poses = {}
    centers = {}
    matrices = {}
    for k, role in enumerate(rigid_roles(design.roles)):
        ids = [i for i, r in enumerate(design.roles) if r == role]
        centers[role] = design.positions[ids].mean(0)
        body = q[(len(free) + k) * dof : (len(free) + k + 1) * dof]
        poses[role] = body
        if design.dimension == 2:
            c, s = body[2].cos(), body[2].sin()
            matrices[role] = torch.stack((c, -s, s, c)).reshape(2, 2)
        else:
            matrices[role] = rotation(body[3:])
    coordinates, angles = [], []
    for i, role in enumerate(design.roles):
        if role == "free":
            local = q[free.index(i) * dof : (free.index(i) + 1) * dof]
            coordinates.append(design.positions[i] + local[: design.dimension])
            angles.append(local[design.dimension :])
        elif role in poses:
            body = poses[role]
            center = centers[role]
            coordinates.append(
                center
                + body[: design.dimension]
                + matrices[role] @ (design.positions[i] - center)
            )
            angles.append(body[design.dimension :])
        else:
            coordinates.append(design.positions[i])
            angles.append(q.new_zeros(dof - design.dimension))
    return torch.stack(coordinates), torch.stack(angles)


def energy(design: Design, q, material: Material = DEFAULT_MATERIAL):
    current, angles = configuration(design, q)
    i, j = design.edges.T
    return beam_energies(
        design.positions[j] - design.positions[i],
        current[j] - current[i],
        torch.stack((angles[i], angles[j]), dim=1),
        design.radii,
        design.presence,
        design.dimension,
        material,
    ).sum()


def beam_energies(original, delta, end_angles, radii, presence, dimension, material):
    """Independent corotational element energies, shared by both tangent paths."""
    from .elements import _local_axes

    length0 = torch.linalg.vector_norm(original, dim=1)
    length = torch.linalg.vector_norm(delta, dim=1)
    area = math.pi * radii.square()
    inertia = math.pi * radii.pow(4) / 4
    axial = material.young_modulus * area / (2 * length0) * (length - length0).square()
    if dimension == 2:
        # atan2(cross,dot) avoids an arbitrary branch at the reference angle.
        chord = torch.atan2(
            original[:, 0] * delta[:, 1] - original[:, 1] * delta[:, 0],
            (original * delta).sum(1),
        )
        a, b = end_angles[:, 0, 0] - chord, end_angles[:, 1, 0] - chord
        bending = (
            2
            * material.young_modulus
            * inertia
            / length0
            * (a.square() + a * b + b.square())
        )
    else:
        reference = _local_axes(original, length0)
        ra, rb = rotation(end_angles[:, 0]), rotation(end_angles[:, 1])
        x = delta / length[:, None]
        director = (
            (ra @ reference[:, 1, :, None]) + (rb @ reference[:, 1, :, None])
        ).squeeze(-1) / 2
        y = director - x * (x * director).sum(-1, keepdim=True)
        y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        frame = torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), dim=-2)
        a = rotation_vector(frame @ ra @ reference.transpose(-1, -2))
        b = rotation_vector(frame @ rb @ reference.transpose(-1, -2))
        bending = (
            2
            * material.young_modulus
            * inertia
            / length0
            * (
                a[:, 1:].square().sum(-1)
                + (a[:, 1:] * b[:, 1:]).sum(-1)
                + b[:, 1:].square().sum(-1)
            )
        )
        bending = (
            bending
            + material.shear_modulus * inertia / length0 * (b[:, 0] - a[:, 0]).square()
        )
    return (axial + bending) * presence


@dataclass
class PathState:
    factor: float
    q: torch.Tensor
    positions: torch.Tensor
    rotations: torch.Tensor
    reaction: torch.Tensor
    energy: torch.Tensor
    residual: float
    iterations: int
    minimum_eigenvalue: float
    beam_nodes: torch.Tensor | None = None


@dataclass
class PathResult:
    states: list[PathState]
    converged: bool
    message: str


def equilibrium(design, load, initial, config, material):
    from .tangents import assembled_tangent, descent_step

    primal = Design(
        design.positions.detach(),
        design.roles,
        design.edges,
        design.radii.detach(),
        design.presence.detach(),
    )
    base = len(initial) - len(load)
    rhs = torch.cat((load.new_zeros(base), load))
    q = initial.detach().clone()
    for iteration in range(config.max_iterations):
        q = q.detach().requires_grad_(True)
        potential = lambda x: energy(design, x, material) - torch.dot(rhs, x)
        value = potential(q)
        g = torch.autograd.grad(value, q, create_graph=True)[0]
        residual = float(g.detach().norm())
        h = assembled_tangent(
            primal, q.detach(), material, config.hessian_chunk_size
        ).detach()
        if residual <= config.tolerance * max(1.0, float(rhs.norm())):
            # Implicit equilibrium derivative: one Newton correction at the
            # converged solution, using a detached tangent inverse.
            corrected = q.detach() - torch.linalg.solve(h.detach(), g)
            eig = float(torch.linalg.eigvalsh(h.cpu()).min())
            return corrected, residual, iteration, eig
        step = descent_step(h[None], g.detach()[None], material.regularization)[0]
        slope = float(torch.dot(g.detach(), step))
        accepted = False
        for backtrack in range(config.max_backtracks):
            candidate = q.detach() + step * (0.5**backtrack)
            if (
                float(potential(candidate).detach())
                <= float(value.detach()) + 1e-4 * (0.5**backtrack) * slope
            ):
                q = candidate
                accepted = True
                break
        if not accepted:
            break
    eig = float(torch.linalg.eigvalsh(h.cpu()).min())
    return None, residual, iteration, eig


def trace(
    design: Design,
    load,
    config: ContinuationConfig = DEFAULT_CONTINUATION,
    material: Material = DEFAULT_MATERIAL,
    rest_factor: float = 0.0,
):
    design.validate()
    _, base = reduction(design)
    dof = 3 if design.dimension == 2 else 6
    if load.shape != (dof,) or not torch.isfinite(load).all():
        raise ValueError("Invalid generalized load")
    q = design.positions.new_zeros(base + dof)
    if not 0 <= rest_factor <= 1:
        raise ValueError("Rest progress must lie between 0 and 1")
    span = max(rest_factor, 1 - rest_factor)
    if rest_factor > 0:
        primal = Design(
            design.positions.detach(),
            design.roles,
            design.edges,
            design.radii.detach(),
            design.presence.detach(),
        )
        preload = trace(primal, load.detach() * (-rest_factor / span), config, material)
        if not preload.converged:
            return PathResult(
                [], False, "Could not reach the start of the requested path"
            )
        q = preload.states[-1].q.detach()
    states = []
    factor, increment = 0.0, 1 / config.steps
    while True:
        target = min(1.0, factor + increment) if states else 0.0
        if factor < rest_factor < target:
            target = rest_factor
        if 1 - target < 1e-12:
            target = 1.0
        result, residual, iterations, eig = equilibrium(
            design, load * ((target - rest_factor) / span), q, config, material
        )
        if result is None:
            increment /= 2
            if increment < config.minimum_increment or not states:
                return PathResult(
                    states,
                    False,
                    "Equilibrium failed; load control stopped before the requested load.",
                )
            continue
        q = result
        coords, rotations = configuration(design, q)
        e = energy(design, q, material)
        reaction = torch.autograd.grad(e, q, create_graph=True)[0][-dof:]
        states.append(
            PathState(
                target, q, coords, rotations, reaction, e, residual, iterations, eig
            )
        )
        factor = target
        if factor >= 1:
            return PathResult(states, True, "Requested load reached.")
        increment = min(1 / config.steps, increment * 1.5)
