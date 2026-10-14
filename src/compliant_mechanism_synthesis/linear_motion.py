"""Small-displacement exploration: one rest tangent and batched input RHS."""

import torch

from .batching import active_beams, kernels
from .equilibrium_system import coordinate_scale, factor_system
from .mechanics import DEFAULT_MATERIAL
from .prescribed_motion import MotionPath, MotionState
from .spectral import Spectrum


def trace_linear(batch, trajectory, config):
    batch = active_beams(batch)
    free = batch.roles.count("free") * 6
    q0 = batch.positions.new_zeros(batch.size, free + 12)
    _, _, tangent = kernels(batch, DEFAULT_MATERIAL, config)
    h = tangent(batch.positions, batch.radii, batch.presence, q0)
    base = free + 6
    system = factor_system(
        h[:, :base, :base],
        coordinate_scale(h[:, :base, :base], 6, config.characteristic_length),
    )
    # Detached factorization + differentiable RHS residual contraction gives the
    # exact first derivative of H(p) u = b(p) without differentiating pivots.
    factors = sorted(
        set(
            [i / config.steps for i in range(config.steps + 1)]
            + trajectory.factors.detach().cpu().tolist()
        )
    )
    progress = q0.new_tensor(factors)
    body = trajectory.at(progress)
    rhs = -(h[:, :base, base:] @ body.T)
    unknown = system.solve(rhs)
    residual = h[:, :base, :base] @ unknown - rhs
    unknown = unknown - system.solve(residual)
    k = h[:, free : free + 6, free : free + 6]
    if free:
        coupling = h[:, :free, free : free + 6]
        # Invalid candidates are masked before a solve, never stabilized into a
        # physically valid mechanism.
        eye = torch.eye(free, dtype=h.dtype, device=h.device)
        safe = torch.where(system.valid[:, None, None], h[:, :free, :free], eye)
        k = k - coupling.transpose(-1, -2) @ torch.linalg.solve(safe, coupling)
    scale = k.new_ones(6)
    scale[3:] = 1 / config.characteristic_length
    spectrum = Spectrum.apply(k * scale[:, None] * scale[None, :])
    ok = system.valid & torch.isfinite(spectrum).all(-1)
    states = []
    for j, factor in enumerate(factors):
        q = torch.cat((unknown[:, :, j], body[j].expand(batch.size, -1)), -1)
        norm = residual[:, :, j].detach().norm(dim=-1)
        states.append(
            MotionState(
                q.new_full((batch.size,), factor),
                q,
                norm,
                torch.zeros(batch.size, dtype=torch.long, device=q.device),
                spectrum[:, 0].detach(),
                ok,
                spectrum,
                None,
            )
        )
    return MotionPath(states, ok, 1)
