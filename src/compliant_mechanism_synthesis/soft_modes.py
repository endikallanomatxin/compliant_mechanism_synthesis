"""Natural compliant motion: relaxed tangent, least-stiff twist and continuation.

The body is advanced along its minimum incremental-energy direction, not along
an imposed force. Free nodes minimize elastic energy at each prescribed body
pose. Translation and spatial angular increments use a physical length metric.
"""

from dataclasses import dataclass

import torch

from .batching import active_beams, kernels
from .mechanics import DEFAULT_MATERIAL
from .nonlinear import rotation, rotation_vector
from .tangents import descent_step


def symmetric_eigh(matrix):
    """Tiny spectral problems stay on Intel XPU (native eigh falls back to CPU)."""
    if matrix.device.type != "xpu":
        return torch.linalg.eigh(matrix)
    a = matrix.clone()
    n = a.shape[-1]
    vectors = torch.eye(n, dtype=a.dtype, device=a.device).expand_as(a).clone()
    for _ in range(10):
        for i in range(n):
            for j in range(i + 1, n):
                angle = 0.5 * torch.atan2(2 * a[..., i, j], a[..., j, j] - a[..., i, i])
                c, s = angle.cos(), angle.sin()
                transform = (
                    torch.eye(n, dtype=a.dtype, device=a.device).expand_as(a).clone()
                )
                transform[..., i, i] = c
                transform[..., j, j] = c
                transform[..., i, j] = s
                transform[..., j, i] = -s
                a = transform.transpose(-1, -2) @ a @ transform
                vectors = vectors @ transform
    values, order = a.diagonal(dim1=-2, dim2=-1).sort(-1)
    return values, vectors.gather(-1, order[..., None, :].expand_as(vectors))


class MinimumMode(torch.autograd.Function):
    """Differentiate only the lowest eigenvector; repeated higher modes are safe.

    The lowest mode must be isolated. Near a crossing the trace is rejected;
    the backward denominator floor only keeps rejected samples numerically safe.
    """

    @staticmethod
    def forward(ctx, matrix):
        values, vectors = symmetric_eigh((matrix + matrix.transpose(-1, -2)) / 2)
        ctx.save_for_backward(values, vectors)
        return values, vectors[..., 0]

    @staticmethod
    def backward(ctx, values_gradient, vector_gradient):
        values, vectors = ctx.saved_tensors
        v = vectors[..., 0]
        gradient = torch.zeros_like(vectors)
        if values_gradient is not None:
            gradient = (vectors * values_gradient[..., None, :]) @ vectors.transpose(
                -1, -2
            )
        if vector_gradient is not None:
            differences = values[..., :1] - values[..., 1:]
            margin = values.abs().amax(-1, keepdim=True).clamp_min(1) * 1e-12
            differences = differences.clamp_max(-margin)
            other = vectors[..., 1:]
            coefficients = (
                other.transpose(-1, -2) @ vector_gradient[..., None]
            ).squeeze(-1) / differences
            w = (other @ coefficients[..., None]).squeeze(-1)
            gradient = (
                gradient
                + (
                    w[..., :, None] * v[..., None, :]
                    + v[..., :, None] * w[..., None, :]
                )
                / 2
            )
        return gradient


def body_step(body, twist, dimension):
    translation = body[..., :dimension] + twist[..., :dimension]
    if dimension == 2:
        angles = body[..., dimension:] + twist[..., dimension:]
    else:
        angles = rotation_vector(
            rotation(twist[..., dimension:]) @ rotation(body[..., dimension:])
        )
    return torch.cat((translation, angles), -1)


def pose_delta(a, b, dimension):
    translation = b[..., :dimension] - a[..., :dimension]
    angles = (
        b[..., dimension:] - a[..., dimension:]
        if dimension == 2
        else rotation_vector(
            rotation(b[..., dimension:])
            @ rotation(a[..., dimension:]).transpose(-1, -2)
        )
    )
    return torch.cat((translation, angles), -1)


def target_arc_length(trajectory, start, end, dimension, length):
    """Four-point Gauss quadrature of physical translation/angular speed."""
    points = trajectory.poses.new_tensor(
        [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ]
    )
    weights = trajectory.poses.new_tensor(
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    )
    samples = (start + end) / 2 + (end - start) * points / 2

    pose, derivative = torch.func.jvp(
        trajectory.at, (samples,), (torch.ones_like(samples),)
    )
    if dimension == 3:
        r, dr = torch.func.jvp(rotation, (pose[..., 3:],), (derivative[..., 3:],))
        skew = dr @ r.transpose(-1, -2)
        angular = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), -1)
    else:
        angular = derivative[..., 2:]
    speed = torch.cat((derivative[..., :dimension], angular * length), -1).norm(dim=-1)
    return (speed * weights).sum() * abs(end - start) / 2


def condensed_stiffness(hessian, gradient, body, dimension, length):
    """Schur complement plus exact spatial rotation chart curvature."""
    dof = body.shape[-1]
    base = hessian.shape[-1] - dof
    hbb = hessian[..., base:, base:]
    if base:
        hff, hfb = hessian[..., :base, :base], hessian[..., :base, base:]
        hbb = hbb - hfb.transpose(-1, -2) @ torch.linalg.solve(hff, hfb)
    scale = body.new_ones(dof)
    scale[dimension:] = 1 / length
    if dimension == 2:
        return hbb * scale[:, None] * scale[None, :]

    def chart(b, g):
        def change(z):
            return body_step(b, z * scale, dimension)

        zero = torch.zeros_like(b)
        jac = torch.func.jacrev(change)(zero)
        curvature = torch.func.jacrev(torch.func.grad(lambda z: (change(z) * g).sum()))(
            zero
        )
        return jac, curvature

    jac, curvature = torch.func.vmap(chart)(body, gradient[..., base:])
    result = jac.transpose(-1, -2) @ hbb @ jac + curvature
    return (result + result.transpose(-1, -2)) / 2


@dataclass
class SoftState:
    factors: torch.Tensor
    q: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    minimum_eigenvalue: torch.Tensor
    accepted: torch.Tensor
    stiffness: torch.Tensor
    mode: torch.Tensor


@dataclass
class SoftPath:
    states: list[SoftState]
    completed: torch.Tensor
    evaluations: int


def relax_free(batch, body, initial, config, cached=None):
    """Prescribed rigid-body pose; free-node equilibrium and implicit derivative."""
    dof = body.shape[-1]
    base = initial.shape[-1] - dof
    primal = active_beams(batch.detached())
    value, grad, tangent = kernels(primal, DEFAULT_MATERIAL, config)
    q = torch.cat((initial[..., :base].detach(), body.detach()), -1)
    counts = torch.zeros(batch.size, dtype=torch.long, device=q.device)
    ok = torch.ones(batch.size, dtype=torch.bool, device=q.device)
    if cached is None and base:
        pending = ok.clone()
        for _ in range(config.max_iterations):
            indices = torch.nonzero(pending).flatten()
            if not len(indices):
                break
            p = primal.take(indices)
            x = q[indices]
            full_gradient = grad(p.positions, p.radii, p.presence, x)
            g = full_gradient[..., :base]
            tolerance = config.tolerance * full_gradient[..., base:].norm(
                dim=-1
            ).clamp_min(1)
            h = tangent(p.positions, p.radii, p.presence, x)[..., :base, :base].detach()
            finite = torch.isfinite(g).all(-1) & torch.isfinite(h).all(dim=(-2, -1))
            solved = finite & (g.norm(dim=-1) <= tolerance)
            pending[indices] = finite & ~solved
            ok[indices] &= finite
            work = torch.nonzero(finite & ~solved).flatten()
            if not len(work):
                continue
            p = p.take(work)
            x = x[work]
            g = g[work]
            h = h[work]
            indices = indices[work]
            step = descent_step(h, g, DEFAULT_MATERIAL.regularization)
            baseline = value(p.positions, p.radii, p.presence, x)
            slope = (g * step).sum(-1)
            waiting = torch.ones(len(indices), dtype=torch.bool, device=q.device)
            for k in range(config.max_backtracks):
                todo = torch.nonzero(waiting).flatten()
                if not len(todo):
                    break
                part = p.take(todo)
                candidate = torch.cat(
                    (x[todo, :base] + (0.5**k) * step[todo], x[todo, base:]), -1
                )
                e = value(part.positions, part.radii, part.presence, candidate)
                take = torch.isfinite(e) & (
                    e
                    <= baseline[todo]
                    + 1e-4 * (0.5**k) * slope[todo]
                    + 32 * torch.finfo(e.dtype).eps * baseline[todo].abs().clamp_min(1)
                )
                accepted = todo[take]
                q[indices[accepted]] = candidate[take].detach()
                waiting[accepted] = False
            counts[indices] += 1
            pending[indices] = ~waiting
            ok[indices] &= ~waiting
        ok &= ~pending
    # Implicit equilibrium sensitivity preserves dependence on the integrated body.
    q = torch.cat((q[..., :base], body), -1)
    _, original_grad, original_tangent = kernels(batch, DEFAULT_MATERIAL, config)
    g = original_grad(batch.positions, batch.radii, batch.presence, q)
    h = original_tangent(batch.positions, batch.radii, batch.presence, q)
    residual = g[..., :base].norm(dim=-1) if base else q.new_zeros(batch.size)
    ok &= torch.isfinite(h).all(dim=(-2, -1)) & (
        residual.detach()
        <= config.tolerance * g[..., base:].detach().norm(dim=-1).clamp_min(1) * 1.01
    )
    if base:
        # Failed samples are excluded by the caller; safe solves avoid NaN gradients.
        eye = torch.eye(base, dtype=q.dtype, device=q.device)
        hff = torch.where(ok[:, None, None], h[..., :base, :base].detach(), eye)
        correction = torch.linalg.solve(
            hff, torch.where(ok[:, None], g[..., :base], 0)[..., None]
        ).squeeze(-1)
        q = torch.cat((q[..., :base] - correction, body), -1)
        g = original_grad(batch.positions, batch.radii, batch.presence, q)
        h = original_tangent(batch.positions, batch.radii, batch.presence, q)
    return q, ok, residual.detach(), counts, g, h


def trace_soft(batch, trajectory, config, cached=None):
    """Integrate least-stiff spatial twists on both sides of the rest pose.

    Explicit Euler in arc length; refinement diagnostics compare steps vs 2*steps.
    Sign is chosen by the desired local twist, with continuity for zero alignment.
    Cached equilibria bypass Newton but never detach the mode/path sensitivity.
    """
    dof = trajectory.poses.shape[-1]
    dim = batch.dimension
    base = sum(role == "free" for role in batch.roles) * dof
    zero = batch.positions.new_zeros(batch.size, base + dof)
    scale = zero.new_ones(dof)
    scale[dim:] = config.characteristic_length
    rest = trajectory.rest_factor
    grid = sorted(
        set(
            [i / config.steps for i in range(config.steps + 1)]
            + trajectory.factors.detach().cpu().tolist()
            + [rest]
        )
    )
    record_maps = (
        None
        if cached is None
        else [{round(r[0], 12): r for r in records} for records in cached]
    )
    states = {}
    completed = torch.ones(batch.size, dtype=torch.bool, device=zero.device)
    evaluations = 0

    def evaluate(factor, body, initial):
        nonlocal evaluations, completed
        indices = torch.nonzero(completed).flatten()
        if not len(indices):
            return None
        part = batch.take(indices)
        guess = initial[indices]
        if record_maps is not None:
            guess = torch.stack(
                [
                    record_maps[i][round(factor, 12)][1]
                    for i in indices.detach().cpu().tolist()
                ]
            ).to(zero.device)
        q, ok, residual, counts, g, h = relax_free(
            part, body[indices], guess, config, record_maps
        )
        evaluations += int(record_maps is None)
        # Singular free tangents are invalid; the Schur solve must not hide them.
        good = torch.nonzero(ok).flatten()
        completed[indices[~ok]] = False
        if not len(good):
            return None
        indices = indices[good]
        q = q[good]
        g = g[good]
        h = h[good]
        residual = residual[good]
        counts = counts[good]
        stiffness = condensed_stiffness(
            h, g, q[..., -dof:], dim, config.characteristic_length
        )
        values, mode = MinimumMode.apply(stiffness)
        valid = (
            torch.isfinite(values).all(-1)
            & (
                values[:, 0]
                >= -config.mode_gap_tolerance * values[:, -1].abs().clamp_min(1)
            )
            & (
                (values[:, 1] - values[:, 0])
                > config.mode_gap_tolerance * values[:, 1].abs().clamp_min(1)
            )
        )
        completed[indices[~valid]] = False
        good = torch.nonzero(valid).flatten()
        if not len(good):
            return None
        indices = indices[good]
        q = q[good]
        values = values[good]
        mode = mode[good]
        residual = residual[good]
        counts = counts[good]
        full = zero.clone().index_copy(0, indices, q)
        modes = zero.new_zeros(batch.size, dof).index_copy(0, indices, mode)
        spectrum = zero.new_zeros(batch.size, dof).index_copy(0, indices, values)
        res = zero.new_zeros(batch.size).index_copy(0, indices, residual)
        iterations = torch.zeros(
            batch.size, dtype=torch.long, device=zero.device
        ).index_copy(0, indices, counts)
        accepted = torch.zeros_like(completed)
        accepted[indices] = True
        state = SoftState(
            zero.new_full((batch.size,), factor),
            full,
            res,
            iterations,
            spectrum[:, 0],
            accepted,
            spectrum,
            modes,
        )
        states[factor] = state
        return state

    initial = evaluate(rest, zero[:, -dof:], zero)
    if initial is not None:
        for branch in (
            [t for t in reversed(grid) if t < rest],
            [t for t in grid if t > rest],
        ):
            current = initial
            previous = zero.new_zeros(batch.size, dof)
            for factor in branch:
                if current is None:
                    break
                start = float(current.factors[0])
                desired = (
                    pose_delta(
                        trajectory.at(zero.new_tensor(start)),
                        trajectory.at(zero.new_tensor(factor)),
                        dim,
                    )
                    * scale
                )
                distance = target_arc_length(
                    trajectory, start, factor, dim, config.characteristic_length
                )
                alignment = (current.mode * desired).sum(-1)
                continuity = (current.mode * previous).sum(-1)
                sign = torch.where(
                    alignment.abs() > 1e-12,
                    alignment.sign(),
                    torch.where(
                        continuity.abs() > 1e-12,
                        continuity.sign(),
                        torch.ones_like(alignment),
                    ),
                )
                direction = current.mode * sign[:, None]
                body = body_step(current.q[:, -dof:], direction * distance / scale, dim)
                previous = direction
                current = evaluate(factor, body, current.q)
    result = [states[k] for k in sorted(states)]
    completed &= bool(result) and 0 in states and 1 in states
    return SoftPath(result, completed, evaluations)
