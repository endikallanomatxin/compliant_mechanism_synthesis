"""Simultaneous nonlinear equilibrium and continuation for independent designs."""

from dataclasses import dataclass

import torch

from .mechanics import DEFAULT_MATERIAL, Design
from .nonlinear import DEFAULT_CONTINUATION, configuration, energy
from .tangents import assembled_tangent, descent_step


@dataclass
class DesignBatch:
    positions: torch.Tensor
    roles: tuple[str, ...]
    edges: torch.Tensor
    radii: torch.Tensor
    presence: torch.Tensor

    @property
    def size(self):
        return self.positions.shape[0]

    @property
    def dimension(self):
        return self.positions.shape[-1]

    def item(self, index):
        return Design(
            self.positions[index],
            self.roles,
            self.edges,
            self.radii[index],
            self.presence[index],
        )

    def take(self, indices):
        return DesignBatch(
            self.positions[indices],
            self.roles,
            self.edges,
            self.radii[indices],
            self.presence[indices],
        )

    def detached(self):
        return DesignBatch(
            self.positions.detach(),
            self.roles,
            self.edges,
            self.radii.detach(),
            self.presence.detach(),
        )


def kernels(batch, material, config=DEFAULT_CONTINUATION):
    def single(p, r, a, q):
        return energy(Design(p, batch.roles, batch.edges, r, a), q, material)

    value = torch.func.vmap(single)
    gradient = torch.func.vmap(torch.func.grad(single, argnums=3))
    hessian = torch.func.vmap(
        lambda p, r, a, q: assembled_tangent(
            Design(p, batch.roles, batch.edges, r, a),
            q,
            material,
            config.hessian_chunk_size,
        )
    )
    return value, gradient, hessian


def active_beams(batch):
    """Prune beams absent from every member of a detached Newton batch.

    The original batch is retained for implicit design derivatives: a zero
    presence can still have a nonzero energy derivative with respect to presence.
    Keep a shared edge list so the mechanics remain vectorized across designs.
    """
    indices = torch.nonzero((batch.presence != 0).any(dim=0)).flatten()
    return DesignBatch(
        batch.positions,
        batch.roles,
        batch.edges[indices],
        batch.radii[:, indices],
        batch.presence[:, indices],
    )


@dataclass
class BatchState:
    factors: torch.Tensor
    q: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    minimum_eigenvalue: torch.Tensor
    accepted: torch.Tensor


@dataclass
class BatchPath:
    states: list[BatchState]
    completed: torch.Tensor
    evaluations: int


def equilibria(batch, loads, initial, enabled, config, material):
    """Compact active designs for Newton, solves and line-search evaluations."""
    primal = active_beams(batch.detached())
    values, gradients, hessians = kernels(primal, material, config)
    q = initial.detach().clone()
    rhs = torch.cat(
        (loads.new_zeros(batch.size, q.shape[-1] - loads.shape[-1]), loads), -1
    )
    identity = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    active = enabled.clone()
    converged = torch.zeros_like(active)
    counts = torch.zeros(batch.size, dtype=torch.long, device=q.device)
    residual = q.new_full((batch.size,), float("inf"))
    final_h = identity.expand(batch.size, -1, -1).clone()
    final_g = torch.zeros_like(q)
    last_h = final_h.clone()
    for _ in range(config.max_iterations):
        indices = torch.nonzero(active).flatten()
        if not len(indices):
            break
        part = primal.take(indices)
        x, force = q[indices], rhs[indices]
        g = gradients(part.positions, part.radii, part.presence, x) - force
        h = hessians(part.positions, part.radii, part.presence, x).detach()
        finite = torch.isfinite(h).all(dim=(-2, -1)) & torch.isfinite(g).all(-1)
        h = torch.where(finite[:, None, None], h, identity)
        last_h[indices] = h
        norms = g.detach().norm(dim=-1)
        residual[indices] = norms
        solved = finite & (
            norms <= config.tolerance * loads[indices].norm(dim=-1).clamp_min(1)
        )
        finished = indices[solved]
        final_h[finished] = h[solved]
        final_g[finished] = g[solved].detach()
        converged[finished] = True
        active[indices] = finite & ~solved
        working = torch.nonzero(finite & ~solved).flatten()
        if not len(working):
            continue
        indices = indices[working]
        part = primal.take(indices)
        x, force, g, h = x[working], force[working], g[working].detach(), h[working]
        step = descent_step(h, g, material.regularization)
        slope = (g * step).sum(-1)
        baseline = values(part.positions, part.radii, part.presence, x) - (
            force * x
        ).sum(-1)
        pending = torch.ones(len(indices), dtype=torch.bool, device=q.device)
        for k in range(config.max_backtracks):
            todo = torch.nonzero(pending).flatten()
            if not len(todo):
                break
            trial = part.take(todo)
            rate = 0.5**k
            candidate = x[todo] + rate * step[todo]
            potential = values(
                trial.positions, trial.radii, trial.presence, candidate
            ) - (force[todo] * candidate).sum(-1)
            take = torch.isfinite(potential) & (
                potential <= baseline[todo] + 1e-4 * rate * slope[todo]
            )
            accepted = todo[take]
            q[indices[accepted]] = candidate[take].detach()
            pending[accepted] = False
        counts[indices] += 1
        active[indices] = ~pending
    # Transfer only diagnostics, once per equilibrium call, outside Newton.
    eigen = torch.linalg.eigvalsh(last_h.cpu()).amin(-1).to(q.device)
    finished = torch.nonzero(converged).flatten()
    result = q
    if len(finished):
        if torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (batch.positions, batch.radii, batch.presence, loads)
        ):
            original = batch.take(finished)
            _, original_gradients, _ = kernels(original, material, config)
            g = (
                original_gradients(
                    original.positions, original.radii, original.presence, q[finished]
                )
                - rhs[finished]
            )
        else:
            g = final_g[finished]
        corrected = q[finished] - torch.linalg.solve(
            final_h[finished] + material.regularization * identity, g
        )
        result = q.index_copy(0, finished, corrected)
    return result, converged, residual, counts, eigen


def trace_batch(
    batch: DesignBatch,
    loads,
    config=DEFAULT_CONTINUATION,
    material=DEFAULT_MATERIAL,
    valid=None,
    rest_factor=0.0,
):
    """Adaptive load increments are independent within a shared tensor batch."""
    dof = 3 if batch.dimension == 2 else 6
    base = sum(r == "free" for r in batch.roles) * dof
    if loads.shape == (dof,):
        loads = loads.expand(batch.size, -1)
    if loads.shape != (batch.size, dof):
        raise ValueError("Batch loads must have shape B×rigid_dofs")
    valid = (
        torch.ones(batch.size, dtype=torch.bool, device=loads.device)
        if valid is None
        else valid
    )
    if not 0 <= rest_factor <= 1:
        raise ValueError("Rest progress must lie between 0 and 1")
    span = max(rest_factor, 1 - rest_factor)
    q = loads.new_zeros(batch.size, base + dof)
    preload_evaluations = 0
    if rest_factor > 0:
        preload = trace_batch(
            batch.detached(),
            loads.detach() * (-rest_factor / span),
            config,
            material,
            valid,
        )
        valid = valid & preload.completed
        if preload.states:
            q = preload.states[-1].q.detach()
        preload_evaluations = preload.evaluations
    factors = loads.new_zeros(batch.size)
    increments = loads.new_full((batch.size,), 1 / config.steps)
    failed = ~valid.clone()
    states = []
    initial = True
    evaluations = preload_evaluations
    while bool((~failed & (factors < 1)).any()):
        enabled = ~failed & (factors < 1)
        targets = factors if initial else (factors + increments).clamp_max(1)
        targets = torch.where(
            (factors < rest_factor) & (targets > rest_factor),
            targets.new_full(targets.shape, rest_factor),
            targets,
        )
        targets = torch.where(1 - targets < 1e-12, torch.ones_like(targets), targets)
        solved, ok, residual, iterations, eigen = equilibria(
            batch,
            loads * ((targets - rest_factor) / span)[:, None],
            q,
            enabled,
            config,
            material,
        )
        evaluations += 1
        success = enabled & ok
        if initial:
            failed |= enabled & ~ok
        else:
            increments = torch.where(enabled & ~ok, increments / 2, increments)
            failed |= enabled & ~ok & (increments < config.minimum_increment)
            increments = torch.where(
                success, (increments * 1.5).clamp_max(1 / config.steps), increments
            )
        q = torch.where(success[:, None], solved, q)
        factors = torch.where(success, targets, factors)
        if states:
            # Diagnostics belong to the accepted state, not to a later failed
            # attempt or to an already completed, disabled member of the batch.
            residual = torch.where(success, residual, states[-1].residual)
            iterations = torch.where(success, iterations, states[-1].iterations)
            eigen = torch.where(success, eigen, states[-1].minimum_eigenvalue)
        states.append(
            BatchState(
                factors.clone(),
                q,
                residual.detach(),
                iterations,
                eigen.detach(),
                success.clone(),
            )
        )
        initial = False
    return BatchPath(states, (factors >= 1) & ~failed, evaluations)


def configurations_batch(batch, q):
    return torch.func.vmap(
        lambda p, r, a, x: configuration(Design(p, batch.roles, batch.edges, r, a), x)
    )(batch.positions, batch.radii, batch.presence, q)


def path_from_equilibria(
    batch,
    loads,
    records,
    config=DEFAULT_CONTINUATION,
    material=DEFAULT_MATERIAL,
    differentiate=True,
    rest_factor=0.0,
):
    """Restore stored equilibria, optionally attaching implicit derivatives.

    Records contain only accepted states from an identical design and load.
    Different path lengths are padded by retaining the last accepted state.
    """
    dof = loads.shape[-1]
    n = (sum(role == "free" for role in batch.roles) + 1) * dof
    q = batch.positions.new_zeros(batch.size, n)
    factors = q.new_zeros(batch.size)
    residual = q.new_zeros(batch.size)
    iterations = torch.zeros(batch.size, dtype=torch.long, device=q.device)
    eigen = q.new_zeros(batch.size)
    states = []
    eye = torch.eye(n, dtype=q.dtype, device=q.device)
    for k in range(max(map(len, records))):
        members = [i for i, path in enumerate(records) if k < len(path)]
        indices = torch.tensor(members, dtype=torch.long, device=q.device)
        entries = [records[i][k] for i in members]
        reference = torch.stack([entry[1] for entry in entries]).to(q.device)
        at = q.new_tensor([entry[0] for entry in entries])
        if differentiate:
            part = batch.take(indices)
            primal = active_beams(part.detached())
            _, _, hessians = kernels(primal, material, config)
            _, gradients, _ = kernels(part, material, config)
            h = hessians(
                primal.positions, primal.radii, primal.presence, reference
            ).detach()
            load = (
                loads.expand(batch.size, -1)[indices]
                * ((at - rest_factor) / max(rest_factor, 1 - rest_factor))[:, None]
            )
            rhs = torch.cat((load.new_zeros(len(indices), n - dof), load), -1)
            g = gradients(part.positions, part.radii, part.presence, reference) - rhs
            corrected = reference - torch.linalg.solve(
                h + material.regularization * eye, g
            )
        else:
            corrected = reference
        q = q.index_copy(0, indices, corrected)
        factors = factors.index_copy(0, indices, at)
        residual[indices] = residual.new_tensor([entry[2] for entry in entries])
        iterations[indices] = iterations.new_tensor([entry[3] for entry in entries])
        eigen[indices] = eigen.new_tensor([entry[4] for entry in entries])
        accepted = torch.zeros(batch.size, dtype=torch.bool, device=q.device)
        accepted[indices] = True
        states.append(
            BatchState(
                factors.clone(),
                q,
                residual.clone(),
                iterations.clone(),
                eigen.clone(),
                accepted,
            )
        )
    return BatchPath(
        states, torch.ones(batch.size, dtype=torch.bool, device=q.device), 0
    )
