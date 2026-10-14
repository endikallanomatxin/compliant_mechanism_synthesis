"""Nonlinear input-output continuation with a prescribed active rigid body.

Unknowns comprise free-node degrees of freedom and the passive rigid body.
Each active pose is held exactly; unknowns minimize the corotational energy.
"""

from dataclasses import dataclass

import torch

from .batching import active_beams, kernels
from .equilibrium_system import scaled_descent
from .mechanics import DEFAULT_MATERIAL
from .nonlinear import rotation, rotation_vector
from .spectral import Spectrum, positive_definite


def spatial_stiffness(stiffness, body, force, dimension, length):
    scale = body.new_ones(body.shape[-1])
    scale[dimension:] = 1 / length
    if dimension == 2:
        return stiffness * scale[:, None] * scale[None, :]

    def chart(b, g):
        def change(z):
            z = z * scale
            return torch.cat(
                (b[:3] + z[:3], rotation_vector(rotation(z[3:]) @ rotation(b[3:])))
            )

        zero = torch.zeros_like(b)
        jac = torch.func.jacrev(change)(zero)
        curvature = torch.func.jacrev(torch.func.grad(lambda z: (change(z) * g).sum()))(
            zero
        )
        return jac, curvature

    jac, curvature = torch.func.vmap(chart)(body, force)
    result = jac.transpose(-1, -2) @ stiffness @ jac + curvature
    return (result + result.transpose(-1, -2)) / 2


def newton_compact(primal, body, initial, config):
    dof = body.shape[-1]
    base = initial.shape[-1] - dof
    value, grad, tangent = kernels(primal, DEFAULT_MATERIAL, config)
    q = initial.detach().clone()
    counts = torch.zeros(primal.size, dtype=torch.long, device=q.device)
    ok = torch.ones(primal.size, dtype=torch.bool, device=q.device)
    pending = ok.clone()
    for _ in range(config.max_iterations):
        indices = torch.nonzero(pending).flatten()
        if not len(indices):
            break
        p = primal.take(indices)
        x = q[indices]
        full_gradient = grad(p.positions, p.radii, p.presence, x)
        g = full_gradient[..., :base]
        tolerance = config.tolerance * full_gradient[..., base:].norm(dim=-1).clamp_min(
            1
        )
        finite = torch.isfinite(full_gradient).all(-1)
        solved = finite & (g.norm(dim=-1) <= tolerance)
        pending[indices] = finite & ~solved
        ok[indices] &= finite
        work = torch.nonzero(finite & ~solved).flatten()
        if not len(work):
            continue
        p = p.take(work)
        x = x[work]
        g = g[work]
        indices = indices[work]
        # A converged residual needs no Newton direction or tangent here.
        # Stability and implicit derivatives use the final tangent below.
        h = tangent(p.positions, p.radii, p.presence, x)[..., :base, :base].detach()
        finite = torch.isfinite(h).all(dim=(-2, -1))
        ok[indices] &= finite
        pending[indices] &= finite
        work = torch.nonzero(finite).flatten()
        if not len(work):
            continue
        p = p.take(work)
        x = x[work]
        g = g[work]
        h = h[work]
        indices = indices[work]
        step = scaled_descent(
            h, g, dof, config.characteristic_length, DEFAULT_MATERIAL.regularization
        )
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
    return q, ok, counts


def masked_descent(h, g, working):
    eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
    matrix = torch.where(working[:, None, None], h, eye)
    gradient = torch.where(working[:, None], g, 0)
    step, info = torch.linalg.solve_ex(
        matrix + DEFAULT_MATERIAL.regularization * eye, -gradient, check_errors=False
    )
    descending = (
        (info == 0) & torch.isfinite(step).all(-1) & ((gradient * step).sum(-1) < 0)
    )
    failed = working & ~descending
    if bool(failed.any()):
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        lower = (diagonal - (matrix.abs().sum(-1) - diagonal.abs())).amin(-1)
        shift = (
            (-lower).clamp_min(0)
            + matrix.abs().amax(dim=(-2, -1)).clamp_min(1) * 1e-8
            + DEFAULT_MATERIAL.regularization
        )
        fallback = torch.linalg.solve(matrix + shift[:, None, None] * eye, -gradient)
        step = torch.where(failed[:, None], fallback, step)
    return torch.where(working[:, None], step, 0)


def newton_masked(primal, body, initial, config):
    """Fixed-shape batches avoid nonzero/index compaction inside Newton.

    Finished members are frozen; masks keep their systems harmless. One scalar
    per Newton/line-search exit remains, instead of dynamic output-size queries.
    """
    dof = body.shape[-1]
    base = initial.shape[-1] - dof
    value, grad, tangent = kernels(primal, DEFAULT_MATERIAL, config)
    q = initial.detach().clone()
    counts = torch.zeros(primal.size, dtype=torch.long, device=q.device)
    pending = torch.ones(primal.size, dtype=torch.bool, device=q.device)
    ok = pending.clone()
    for _ in range(config.max_iterations):
        full_gradient = grad(primal.positions, primal.radii, primal.presence, q)
        g = full_gradient[..., :base]
        tolerance = config.tolerance * full_gradient[..., base:].norm(dim=-1).clamp_min(
            1
        )
        finite = torch.isfinite(full_gradient).all(-1)
        ok &= ~pending | finite
        pending &= finite & (g.norm(dim=-1) > tolerance)
        if not bool(pending.any()):
            break
        h = tangent(primal.positions, primal.radii, primal.presence, q)[
            ..., :base, :base
        ].detach()
        finite = torch.isfinite(h).all(dim=(-2, -1))
        ok &= ~pending | finite
        pending &= finite
        step = scaled_descent(
            torch.where(
                pending[:, None, None],
                h,
                torch.eye(base, dtype=q.dtype, device=q.device),
            ),
            torch.where(pending[:, None], g, 0),
            dof,
            config.characteristic_length,
            DEFAULT_MATERIAL.regularization,
        )
        slope = (torch.where(pending[:, None], g, 0) * step).sum(-1)
        baseline = value(primal.positions, primal.radii, primal.presence, q)
        waiting = pending.clone()
        original = q
        for k in range(config.max_backtracks):
            candidate = torch.cat(
                (original[..., :base] + (0.5**k) * step, original[..., base:]), -1
            )
            e = value(primal.positions, primal.radii, primal.presence, candidate)
            accepted = (
                waiting
                & torch.isfinite(e)
                & (
                    e
                    <= baseline
                    + 1e-4 * (0.5**k) * slope
                    + 32 * torch.finfo(e.dtype).eps * baseline.abs().clamp_min(1)
                )
            )
            q = torch.where(accepted[:, None], candidate, q)
            waiting &= ~accepted
            if not bool(waiting.any()):
                break
        counts += pending.to(counts.dtype)
        ok &= ~waiting
        pending &= ~waiting
    ok &= ~pending
    return q, ok, counts


def relax_equilibrium(
    batch, body, initial, config, cached=None, differentiate_stiffness=True
):
    """Prescribed rigid-body pose; free-node equilibrium and implicit derivative."""
    if batch.dimension == 3:
        from .implicit_equilibrium import relax_spatial

        return relax_spatial(
            batch, body, initial, config, cached, differentiate_stiffness
        )
    dof = body.shape[-1]
    base = initial.shape[-1] - dof
    q = torch.cat((initial[..., :base].detach(), body.detach()), -1)
    counts = torch.zeros(batch.size, dtype=torch.long, device=q.device)
    ok = torch.ones(batch.size, dtype=torch.bool, device=q.device)
    if cached is None and base:
        primal = batch.detached()
        mode = config.newton_batch_mode
        masked = mode == "masked" or (mode == "auto" and q.device.type == "xpu")
        q, ok, counts = (newton_masked if masked else newton_compact)(
            primal, body, q, config
        )
    # Implicit equilibrium sensitivity preserves dependence on the integrated body.
    q = torch.cat((q[..., :base], body), -1)
    _, original_grad, original_tangent = kernels(batch, DEFAULT_MATERIAL, config)
    g = original_grad(batch.positions, batch.radii, batch.presence, q)
    # The inverse tangent in an implicit first derivative is held fixed.
    # Do not construct a design-gradient graph that is immediately detached.
    h = original_tangent(
        batch.positions.detach(), batch.radii.detach(), batch.presence, q.detach()
    )
    residual = g[..., :base].norm(dim=-1) if base else q.new_zeros(batch.size)
    ok &= torch.isfinite(h).all(dim=(-2, -1)) & (
        residual.detach()
        <= config.tolerance * g[..., base:].detach().norm(dim=-1).clamp_min(1) * 1.01
    )
    if base:
        ok &= positive_definite(h[..., :base, :base])
        # Failed samples are excluded by the caller; safe solves avoid NaN gradients.
        eye = torch.eye(base, dtype=q.dtype, device=q.device)
        hff = torch.where(ok[:, None, None], h[..., :base, :base].detach(), eye)
        correction = torch.linalg.solve(
            hff, torch.where(ok[:, None], g[..., :base], 0)[..., None]
        ).squeeze(-1)
        q = torch.cat((q[..., :base] - correction, body), -1)
        g = original_grad(batch.positions, batch.radii, batch.presence, q)
        differentiating = any(
            t.requires_grad for t in (batch.positions, batch.radii, body)
        )
        if differentiating and batch.dimension == 2:
            h = original_tangent(batch.positions, batch.radii, batch.presence, q)
        # Primal cleanup does not need an intermediate tangent assembly.
        # Retain it for implicit derivatives where Hessian dependence matters.
        if batch.dimension == 2:
            hff = torch.where(ok[:, None, None], h[..., :base, :base].detach(), eye)
            correction = torch.linalg.solve(
                hff, torch.where(ok[:, None], g[..., :base], 0)[..., None]
            ).squeeze(-1)
            q = torch.cat((q[..., :base] - correction, body), -1)
            g = original_grad(batch.positions, batch.radii, batch.presence, q)
        if differentiate_stiffness:
            h = original_tangent(batch.positions, batch.radii, batch.presence, q)
        else:
            h = original_tangent(
                batch.positions.detach(),
                batch.radii.detach(),
                batch.presence,
                q.detach(),
            )
    return q, ok, residual.detach(), counts, g, h, None


@dataclass
class MotionState:
    factors: torch.Tensor
    q: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    minimum_eigenvalue: torch.Tensor
    accepted: torch.Tensor
    stiffness: torch.Tensor
    beam_nodes: torch.Tensor | None = None
    linearization: object = None


@dataclass
class MotionPath:
    states: list[MotionState]
    completed: torch.Tensor
    evaluations: int


def predict_equilibrium(q, sensitivity, body):
    """First-order implicit equilibrium predictor; detached Newton initial guess."""
    dof = body.shape[-1]
    change = body.detach() - q[..., -dof:].detach()
    unknown = q[..., :-dof].detach() + (
        sensitivity.detach() @ change[..., None]
    ).squeeze(-1)
    return torch.cat((unknown, body.detach()), -1)


def predictor_error(result, guess, dimension, config):
    """Passive pose correction in SI units; an indicator, not an error bound."""
    dof = 3 if dimension == 2 else 6
    actual = result[..., -2 * dof : -dof].detach()
    predicted = guess[..., -2 * dof : -dof].detach()
    position = (actual[..., :dimension] - predicted[..., :dimension]).norm(dim=-1)
    if dimension == 2:
        angle = actual[..., 2] - predicted[..., 2]
        orientation = torch.atan2(angle.sin(), angle.cos()).abs()
    else:
        orientation = rotation_vector(
            rotation(actual[..., 3:]) @ rotation(predicted[..., 3:]).transpose(-1, -2)
        ).norm(dim=-1)
    return torch.maximum(
        position / config.path_position_tolerance,
        orientation / config.path_orientation_tolerance,
    )


def trace_motion(
    batch,
    input_trajectory,
    config,
    cached=None,
    differentiate_stiffness=True,
    initial_path=None,
):
    if "active" not in batch.roles or "passive" not in batch.roles:
        raise ValueError("Provide separate active and passive rigid bodies")
    # Binary topology is fixed throughout a path, including its derivatives.
    # Prune its edge union once; each equilibrium reuses this graph.
    from .mechanics_work import record

    original_batch = batch
    hints = (
        [
            None if hint is None else hint.to(batch.positions.device)
            for hint in initial_path
        ]
        if initial_path is not None and cached is None
        else None
    )
    edge_indices = torch.nonzero((batch.presence != 0).any(dim=0)).flatten()
    batch = active_beams(batch)
    dof = input_trajectory.poses.shape[-1]
    retain_linearization = not any(
        t.requires_grad for t in (batch.positions, batch.radii, input_trajectory.poses)
    )
    free = sum(r == "free" for r in batch.roles) * dof
    q = batch.positions.new_zeros(batch.size, free + 2 * dof)
    beam_state = None
    if config.beam_segments > 1:
        from .condensed_mechanics import nodal_map, relax_condensed
        from .subdivided import BeamMesh, initial_nodes, reference_nodes

        ref = reference_nodes(
            batch.positions[:, batch.edges], BeamMesh(config.beam_segments)
        )
        nodes, _, _ = nodal_map(batch, q)
        beam_state = initial_nodes(ref, nodes[:, batch.edges])

    hint_private_valid = torch.zeros(batch.size, dtype=torch.bool, device=q.device)

    def solve(part, body, guess, indices, records=None, cached_interiors=None):
        if config.beam_segments == 1:
            result = relax_equilibrium(
                part,
                body,
                guess,
                config,
                records,
                differentiate_stiffness=differentiate_stiffness,
            )
            return (*result[:-1], None, result[-1])
        interior = (
            beam_state[indices, :, 1:-1, :]
            if cached_interiors is None
            else cached_interiors[:, edge_indices]
        )
        if records is None and config.predictor and states:
            from .condensed_mechanics import predict_interiors

            current_q = torch.cat((guess[..., :-dof], body), -1)
            predicted_inner = predict_interiors(
                part,
                current_q,
                beam_state[indices],
                config.beam_segments,
                DEFAULT_MATERIAL,
            )
            interior = (
                torch.where(
                    hint_private_valid[indices, None, None, None],
                    interior,
                    predicted_inner,
                )
                if cached_interiors is not None
                else predicted_inner
            )
        return relax_condensed(
            part,
            body,
            guess,
            config,
            interior,
            cached=records is not None,
            differentiate_stiffness=differentiate_stiffness,
        )

    active = torch.ones(batch.size, dtype=torch.bool, device=q.device)
    states = []
    evaluations = 0
    grid = sorted(
        set(
            [i / config.steps for i in range(config.steps + 1)]
            + input_trajectory.factors.detach().cpu().tolist()
        )
    )
    record_maps = (
        None
        if cached is None
        else [{round(r[0], 12): r for r in records} for records in cached]
    )
    if cached is not None:
        grid = sorted({r[0] for records in cached for r in records})
    cached_edge_ids = None
    if cached is not None and beam_state is not None:
        presence_cpu = original_batch.presence.detach().cpu()
        cached_edge_ids = [torch.nonzero(row != 0).flatten() for row in presence_cpu]

    # Prepare the requested start from undeformed input; this warmup is not scored.
    if input_trajectory.rest_factor > 0 and record_maps is None:
        rest = input_trajectory.rest_factor
        preparation = sorted(
            {rest * (1 - i / config.steps) for i in range(1, config.steps + 1)}
            | {t for t in grid if t < rest},
            reverse=True,
        )
        previous_preparation = rest
        while preparation:
            factor = preparation.pop(0)
            indices = torch.nonzero(active).flatten()
            if not len(indices):
                break
            body = (
                input_trajectory.at(q.new_tensor(factor))
                .detach()
                .expand(len(indices), -1)
            )
            result, ok, _, _, _, _, local_beams, _ = solve(
                batch.take(indices).detached(), body, q[indices], indices
            )
            evaluations += 1
            if (
                not bool(ok.all())
                and previous_preparation - factor > 2 * config.minimum_increment
            ):
                preparation = [
                    (previous_preparation + factor) / 2,
                    factor,
                ] + preparation
                continue
            q = q.index_copy(0, indices, result.detach())
            if beam_state is not None:
                beam_state = beam_state.index_copy(0, indices, local_beams.detach())
            active[indices] &= ok
            previous_preparation = factor
    pending = list(grid)
    previous = 0.0
    sensitivity = None
    while pending:
        factor = pending.pop(0)
        indices = torch.nonzero(active).flatten()
        if not len(indices):
            break
        part = batch.take(indices)
        guess = q[indices]
        if record_maps is not None:
            guess = torch.stack(
                [
                    record_maps[i][round(factor, 12)][1]
                    for i in indices.detach().cpu().tolist()
                ]
            ).to(q.device)
        body = input_trajectory.at(q.new_tensor(factor)).expand(len(indices), -1)
        predicted = record_maps is None and config.predictor and sensitivity is not None
        if predicted:
            guess = predict_equilibrium(guess, sensitivity[indices], body)
        continuation_guess = guess
        hint_private_valid.zero_()
        hint_used = torch.zeros(batch.size, dtype=torch.bool, device=q.device)
        cached_interiors = None
        if hints is not None:
            # Only guesses close to the child's own continuation predictor are
            # eligible; a parent solution cannot replace Newton or preloading.
            rows = range(batch.size)
            full_guess = q.clone().index_copy(0, indices, guess)
            candidate_guesses = []
            candidate_interiors = []
            for i in rows:
                candidate, private = (
                    (None, None)
                    if hints[i] is None
                    else hints[i].at(
                        factor, original_batch.positions[i], original_batch.edges
                    )
                )
                eligible = q.new_tensor(False, dtype=torch.bool)
                if candidate is not None:
                    difference = (candidate - full_guess[i]).reshape(-1, dof)
                    eligible = (
                        difference[:, :3].norm(dim=-1).max()
                        <= config.path_position_tolerance
                    ) & (
                        difference[:, 3:].norm(dim=-1).max()
                        <= config.path_orientation_tolerance
                    )
                hint_used[i] = eligible
                candidate_guesses.append(
                    full_guess[i]
                    if candidate is None
                    else torch.where(eligible, candidate, full_guess[i])
                )
                if beam_state is not None:
                    fallback = q.new_zeros(
                        len(original_batch.edges), config.beam_segments - 1, dof
                    ).index_copy(0, edge_indices, beam_state[i, :, 1:-1, :])
                    hint_private_valid[i] = eligible & (private is not None)
                    candidate_interiors.append(
                        fallback
                        if private is None
                        else torch.where(eligible, private, fallback)
                    )
            record("parent_hint_trials", len(indices))
            record("parent_hint_used", int(hint_used[indices].sum()))
            guess = torch.stack(candidate_guesses)[indices]
            if candidate_interiors:
                cached_interiors = torch.stack(candidate_interiors)[indices]
        if record_maps is not None and beam_state is not None:
            meshes = []
            for i in indices.detach().cpu().tolist():
                compact = record_maps[i][round(factor, 12)][6]
                meshes.append(
                    compact.new_zeros(
                        len(original_batch.edges), config.beam_segments - 1, dof
                    ).index_copy(0, cached_edge_ids[i], compact)
                )
            cached_interiors = torch.stack(meshes).to(q.device)
        result, ok, residual, counts, _g, h, local_beams, linearization = solve(
            part, body, guess, indices, record_maps, cached_interiors
        )
        evaluations += int(record_maps is None)
        if hints is not None and bool(hint_used[indices].any()):
            correction = (result - continuation_guess).reshape(len(indices), -1, dof)
            ambiguous = (
                correction[..., :3].norm(dim=-1).amax(-1)
                > config.path_position_tolerance
            ) | (
                correction[..., 3:].norm(dim=-1).amax(-1)
                > config.path_orientation_tolerance
            )
            if bool((hint_used[indices] & (~ok | ambiguous)).any()):
                record("parent_hint_fallbacks", len(indices))
                hint_private_valid.zero_()
                old_counts = counts
                result, ok, residual, counts, _g, h, local_beams, linearization = solve(
                    part, body, continuation_guess, indices
                )
                counts = counts + old_counts
                evaluations += 1
        inaccurate = False
        if predicted:
            error = predictor_error(result, continuation_guess, batch.dimension, config)
            inaccurate = bool(((error > 1) & ok).any())
        # Failed Newton steps are retried with smaller input-progress increments.
        # A common grid keeps batch continuation and cached replay consistent.
        if (
            record_maps is None
            and (not bool(ok.all()) or inaccurate)
            and states
            and factor - previous > 2 * config.minimum_increment
        ):
            pending = [(previous + factor) / 2, factor] + pending
            continue
        active[indices[~ok]] = False
        good = torch.nonzero(ok).flatten()
        if not len(good):
            continue
        indices = indices[good]
        result = result[good]
        if linearization is not None and retain_linearization:
            linearization = linearization.take(good)
        if local_beams is not None:
            local_beams = local_beams[good]
        h = h[good]
        residual = residual[good]
        counts = counts[good]
        unknown = h[..., :-dof, :-dof]
        stability = positive_definite(unknown)
        # Block the actuator and relax free nodes to evaluate passive support.
        k = h[..., free : free + dof, free : free + dof]
        if free:
            coupling = h[..., :free, free : free + dof]
            k = k - coupling.transpose(-1, -2) @ torch.linalg.solve(
                h[..., :free, :free], coupling
            )
        k = spatial_stiffness(
            k,
            result[..., free : free + dof],
            _g[good, free : free + dof],
            batch.dimension,
            config.characteristic_length,
        )
        spectrum = Spectrum.apply(k)
        stable = stability & torch.isfinite(spectrum).all(-1)
        active[indices[~stable]] = False
        good = torch.nonzero(stable).flatten()
        if not len(good):
            continue
        indices = indices[good]
        result = result[good]
        if linearization is not None and retain_linearization:
            linearization = linearization.take(good)
        if record_maps is None and config.predictor:
            tangent = h[good].detach()
            local_sensitivity = -torch.linalg.solve(
                tangent[..., :-dof, :-dof], tangent[..., :-dof, -dof:]
            )
            if sensitivity is None:
                sensitivity = q.new_zeros(batch.size, q.shape[-1] - dof, dof)
            sensitivity = sensitivity.index_copy(0, indices, local_sensitivity)
        spectrum = spectrum[good]
        residual = residual[good]
        counts = counts[good]
        q = q.index_copy(0, indices, result)
        stored_beams = None
        if beam_state is not None:
            beam_state = beam_state.index_copy(0, indices, local_beams[good])
            # Keep original edge indexing for viewer and cache records.
            stored_beams = q.new_zeros(
                batch.size, len(original_batch.edges), config.beam_segments + 1, dof
            ).index_copy(1, edge_indices, beam_state)
        accepted = torch.zeros_like(active)
        accepted[indices] = True
        states.append(
            MotionState(
                q.new_full((batch.size,), factor),
                q.clone(),
                q.new_zeros(batch.size).index_copy(0, indices, residual),
                torch.zeros(batch.size, dtype=torch.long, device=q.device).index_copy(
                    0, indices, counts
                ),
                q.new_zeros(batch.size).index_copy(0, indices, spectrum[:, 0]),
                accepted,
                q.new_zeros(batch.size, dof).index_copy(0, indices, spectrum),
                stored_beams,
                None
                if linearization is None or not retain_linearization
                else (linearization, indices),
            )
        )
        previous = factor
    active &= bool(states) and states[-1].factors[0] >= 1
    return MotionPath(states, active, evaluations)
