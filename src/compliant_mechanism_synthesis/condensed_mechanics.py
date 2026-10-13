"""Beam condensation with exact nonlinear rigid-body kinematics."""

import torch

from .mechanics import rigid_roles
from .nonlinear import rotation
from .spectral import positive_definite
from .subdivided import (
    BeamMesh,
    chain_derivatives,
    chain_energy,
    condense,
    initial_nodes,
    reference_nodes,
)
from .tangents import descent_step


def nodal_nodes(batch, q):
    """Exact fixed clamps and rigid ties, without tangent-map preparation."""
    dimension = batch.dimension
    dof = 3 if dimension == 2 else 6
    free = [i for i, role in enumerate(batch.roles) if role == "free"]
    nodes = torch.cat(
        (batch.positions, q.new_zeros(batch.size, len(batch.roles), dof - dimension)),
        -1,
    )
    ids = torch.tensor(free, dtype=torch.long, device=q.device)
    nodes = nodes.index_add(
        1, ids, q[:, : len(free) * dof].reshape(batch.size, len(free), dof)
    )
    for k, role in enumerate(rigid_roles(batch.roles)):
        body_ids = torch.tensor(
            [i for i, r in enumerate(batch.roles) if r == role], device=q.device
        )
        points = batch.positions[:, body_ids]
        center = points.mean(1, keepdim=True)
        body = q[:, (len(free) + k) * dof : (len(free) + k + 1) * dof]
        if dimension == 2:
            c, s = body[:, 2].cos(), body[:, 2].sin()
            rot = torch.stack((c, -s, s, c), -1).reshape(batch.size, 2, 2)
        else:
            rot = rotation(body[:, 3:])
        current = (
            center
            + body[:, None, :dimension]
            + (points - center) @ rot.transpose(-1, -2)
        )
        nodes = nodes.index_copy(
            1,
            body_ids,
            torch.cat(
                (current, body[:, None, dimension:].expand(-1, len(body_ids), -1)), -1
            ),
        )
    return nodes


def nodal_map(batch, q):
    """Current nodal coordinates, Jacobian and rigid-body curvature data."""
    nodes = nodal_nodes(batch, q)
    b, n, dof = nodes.shape
    dimension, m = batch.dimension, q.shape[-1]
    free = [i for i, r in enumerate(batch.roles) if r == "free"]
    eye = torch.eye(m, dtype=q.dtype, device=q.device)
    ids = torch.tensor(free, dtype=torch.long, device=q.device)
    free_map = (
        eye[: len(free) * dof].reshape(1, len(free), dof, m).expand(b, -1, -1, -1)
    )
    mapping = q.new_zeros(b, n, dof, m).index_copy(1, ids, free_map)
    arms = []
    for k, role in enumerate(rigid_roles(batch.roles)):
        ids = [i for i, r in enumerate(batch.roles) if r == role]
        offset = (len(free) + k) * dof
        local = (
            torch.eye(dof, dtype=q.dtype, device=q.device)
            .expand(b, len(ids), dof, dof)
            .clone()
        )
        if dimension == 2:
            arm = nodes[:, ids, :2] - nodes[:, ids, :2].mean(1, keepdim=True)
            local[:, :, 0, 2] = -arm[:, :, 1]
            local[:, :, 1, 2] = arm[:, :, 0]
        else:
            arm = batch.positions[:, ids] - batch.positions[:, ids].mean(
                1, keepdim=True
            )
            derivative = torch.func.vmap(torch.func.jacrev(rotation))(
                q[:, offset + 3 : offset + 6]
            )
            local[:, :, :3, 3:] = torch.einsum("bijk,bnj->bnik", derivative, arm)
        mapping[:, ids, :, offset : offset + dof] = local
        arms.append((ids, offset + dimension, arm))
    return nodes, mapping.flatten(1, 2), arms


def evaluate(batch, q, interior, segments, material, detached_tangent=False):
    """Condense before global assembly, including nonlinear map curvature."""
    nodes, jac, arms = nodal_map(batch, q)
    dof, dimension = nodes.shape[-1], batch.dimension
    ref = reference_nodes(batch.positions[:, batch.edges], BeamMesh(segments))
    if interior is None:
        interior = initial_nodes(ref, nodes[:, batch.edges])[..., 1:-1, :]
    chains = torch.cat(
        (
            nodes[:, batch.edges[:, 0], None],
            interior,
            nodes[:, batch.edges[:, 1], None],
        ),
        -2,
    )
    chains = torch.where(
        batch.presence[..., None, None] != 0,
        chains,
        torch.cat((ref, ref.new_zeros(ref.shape[:-1] + (dof - dimension,))), -1),
    )
    values, g, h = chain_derivatives(ref, chains, batch.radii, material)
    present = batch.presence != 0
    g = torch.where(present[..., None], g, 0)
    # Absent members have harmless private identity systems but zero endpoints.
    n = h.shape[-1]
    identity = torch.zeros(n, n, dtype=h.dtype, device=h.device)
    identity[dof:-dof, dof:-dof] = torch.eye(
        n - 2 * dof, dtype=h.dtype, device=h.device
    )
    h = torch.where(present[..., None, None], h, identity)
    if detached_tangent:
        h = h.detach()
    blocks = condense(g, h, dof)
    i, j = batch.edges.T
    if q.device.type == "xpu":
        # Dense batched GEMM avoids contended FP64 atomic scatter on Intel XPU.
        # Contract the endpoint axis first; never materialize E x M x M.
        nodal_j = jac.reshape(batch.size, -1, dof, q.shape[-1])
        edge_j = torch.stack((nodal_j[:, i], nodal_j[:, j]), 2).flatten(2, 3)
        flat_j = edge_j.flatten(1, 2)
        reduced_g = (
            flat_j.transpose(-1, -2) @ blocks.residual.flatten(1)[..., None]
        ).squeeze(-1)
        transformed = (blocks.tangent @ edge_j).flatten(1, 2)
        reduced_h = flat_j.transpose(-1, -2) @ transformed
    else:
        free = [k for k, role in enumerate(batch.roles) if role == "free"]
        offsets = {k: index * dof for index, k in enumerate(free)}
        rigid_offsets = {
            role: (len(free) + index) * dof
            for index, role in enumerate(rigid_roles(batch.roles))
        }
        columns = torch.tensor(
            [
                [
                    offsets[k] if role == "free" else rigid_offsets.get(role, 0)
                    for _ in range(dof)
                ]
                for k, role in enumerate(batch.roles)
            ],
            dtype=torch.long,
            device=q.device,
        ) + torch.arange(dof, device=q.device)
        nodal_j = jac.reshape(batch.size, -1, dof, q.shape[-1]).gather(
            -1, columns[None, :, None, :].expand(batch.size, -1, dof, -1)
        )
        ja, jb = nodal_j[:, i], nodal_j[:, j]
        zero = torch.zeros_like(ja)
        local_j = torch.cat((torch.cat((ja, zero), -1), torch.cat((zero, jb), -1)), -2)
        equation_ids = torch.cat((columns[i], columns[j]), -1)
        local_g = (local_j.transpose(-1, -2) @ blocks.residual[..., None]).squeeze(-1)
        local_h = local_j.transpose(-1, -2) @ blocks.tangent @ local_j
        size = q.shape[-1]
        reduced_g = q.new_zeros(batch.size, size).index_add(
            -1, equation_ids.flatten(), local_g.flatten(-2)
        )
        entries = (equation_ids[:, :, None] * size + equation_ids[:, None, :]).flatten()
        reduced_h = (
            q.new_zeros(batch.size, size * size)
            .index_add(-1, entries, local_h.flatten(-3))
            .reshape(batch.size, size, size)
        )
        edge_j = (local_j, equation_ids)
    external = torch.cat((g[..., :dof], g[..., -dof:]), -1)
    forces = (
        q.new_zeros(batch.size, len(batch.roles), dof)
        .index_add(1, i, external[..., :dof])
        .index_add(1, j, external[..., dof:])
    )
    curvature = q.new_zeros(batch.size, q.shape[-1], q.shape[-1])
    for ids, column, arm in arms:
        if dimension == 2:
            curvature[:, column, column] = -(arm * forces[:, ids, :2]).sum((1, 2))
        else:

            def work(theta, points, force):
                return ((points @ rotation(theta).T) * force).sum()

            block = torch.func.vmap(torch.func.jacrev(torch.func.grad(work)))(
                q[:, column : column + 3], arm, forces[:, ids, :3]
            )
            curvature[:, column : column + 3, column : column + 3] = block
    reduced_h = reduced_h + curvature
    stable = positive_definite(h[..., dof:-dof, dof:-dof]).all(-1) & (
        blocks.info == 0
    ).all(-1)
    return (
        (values * batch.presence).sum(-1),
        reduced_g,
        reduced_h,
        blocks,
        chains,
        edge_j,
        stable,
        g,
    )


def apply_endpoint_step(edge_map, step, detach=False):
    if isinstance(edge_map, torch.Tensor):
        mapping = edge_map.detach() if detach else edge_map
        return (mapping @ step[:, None, :, None]).squeeze(-1)
    mapping, indices = edge_map
    if detach:
        mapping = mapping.detach()
    return (mapping @ step[:, indices, None]).squeeze(-1)


def trial_energy(batch, q, interior, segments, material):
    nodes = nodal_nodes(batch, q)
    dof, dimension = nodes.shape[-1], batch.dimension
    ref = reference_nodes(batch.positions[:, batch.edges], BeamMesh(segments))
    chains = torch.cat(
        (
            nodes[:, batch.edges[:, 0], None],
            interior,
            nodes[:, batch.edges[:, 1], None],
        ),
        -2,
    )
    chains = torch.where(
        batch.presence[..., None, None] != 0,
        chains,
        torch.cat((ref, ref.new_zeros(ref.shape[:-1] + (dof - dimension,))), -1),
    )
    return (chain_energy(ref, chains, batch.radii, material) * batch.presence).sum(-1)


def relax_condensed(
    batch, body, initial, config, interior=None, cached=False, material=None
):
    """Global Newton on graph/body DOFs; private mesh corrections in batch."""
    from .mechanics import DEFAULT_MATERIAL

    material = DEFAULT_MATERIAL if material is None else material
    dof = body.shape[-1]
    base = initial.shape[-1] - dof
    q = torch.cat((initial[..., :base].detach(), body.detach()), -1)
    primal = batch.detached()
    if interior is None:
        nodes, _, _ = nodal_map(primal, q)
        ref = reference_nodes(
            primal.positions[:, primal.edges], BeamMesh(config.beam_segments)
        )
        interior = initial_nodes(ref, nodes[:, primal.edges])[..., 1:-1, :]
    interior = interior.detach().clone()
    counts = torch.zeros(batch.size, dtype=torch.long, device=q.device)
    ok = torch.ones(batch.size, dtype=torch.bool, device=q.device)
    pending = ok.clone()
    if not cached:
        for _ in range(config.max_iterations):
            value, g, h, blocks, _, edge_j, stable, local_g = evaluate(
                primal, q, interior, config.beam_segments, material
            )
            tolerance = config.tolerance * g[..., base:].norm(dim=-1).clamp_min(1)
            residual = torch.maximum(
                g[..., :base].norm(dim=-1),
                local_g[..., dof:-dof].flatten(1).norm(dim=-1),
            )
            finite = (
                torch.isfinite(h).all((-2, -1)) & torch.isfinite(g).all(-1) & stable
            )
            ok &= ~pending | finite
            pending &= finite & (residual > tolerance)
            if not bool(pending.any()):
                break
            eye = torch.eye(base, dtype=q.dtype, device=q.device)
            hh = torch.where(pending[:, None, None], h[..., :base, :base], eye)
            step = descent_step(
                hh,
                torch.where(pending[:, None], g[..., :base], 0),
                material.regularization,
            )
            full_step = torch.cat((step, torch.zeros_like(body)), -1)
            endpoint_step = apply_endpoint_step(edge_j, full_step)
            inner_step = blocks.recover(endpoint_step).reshape_as(interior)
            inner_step = torch.where(pending[:, None, None, None], inner_step, 0)
            # Full residual dot full correction includes private-node contributions.
            local_step = torch.cat(
                (
                    endpoint_step[..., :dof],
                    inner_step.flatten(-2),
                    endpoint_step[..., dof:],
                ),
                -1,
            )
            slope = (local_g * local_step).sum((1, 2))
            original_q, original_interior = q, interior
            waiting = pending.clone()
            for k in range(config.max_backtracks):
                rate = 0.5**k
                candidate = original_q + rate * full_step
                candidate_inner = original_interior + rate * inner_step
                potential = trial_energy(
                    primal, candidate, candidate_inner, config.beam_segments, material
                )
                accepted = (
                    waiting
                    & torch.isfinite(potential)
                    & (
                        potential
                        <= value
                        + 1e-4 * rate * slope
                        + 32 * torch.finfo(value.dtype).eps * value.abs().clamp_min(1)
                    )
                )
                q = torch.where(accepted[:, None], candidate, q)
                interior = torch.where(
                    accepted[:, None, None, None], candidate_inner, interior
                )
                waiting &= ~accepted
                if not bool(waiting.any()):
                    break
            counts += pending.to(counts.dtype)
            ok &= ~waiting
            pending &= ~waiting
        ok &= ~pending
    q = torch.cat((q[..., :base], body), -1)
    # One primal cleanup suffices when no design/input derivative is requested.
    # Keep both implicit corrections for graph and private-node sensitivities.
    differentiating = any(t.requires_grad for t in (batch.positions, batch.radii, body))
    for _ in range(2 if differentiating else 1):
        _, g, h, blocks, _, edge_j, stable, _ = evaluate(
            batch, q, interior, config.beam_segments, material, detached_tangent=True
        )
        valid = ok & stable & positive_definite(h[..., :base, :base])
        eye = torch.eye(base, dtype=q.dtype, device=q.device)
        matrix = torch.where(valid[:, None, None], h[..., :base, :base].detach(), eye)
        step = torch.linalg.solve(
            matrix, -torch.where(valid[:, None], g[..., :base], 0)
        )
        full_step = torch.cat((step, torch.zeros_like(body)), -1)
        endpoint_step = apply_endpoint_step(edge_j, full_step, detach=True)
        correction = blocks.recover(endpoint_step).reshape_as(interior)
        interior = interior + torch.where(valid[:, None, None, None], correction, 0)
        q = q + full_step
        ok &= valid
    value, g, h, _, chains, _, stable, local_g = evaluate(
        batch, q, interior, config.beam_segments, material
    )
    residual = torch.maximum(
        g[..., :base].norm(dim=-1), local_g[..., dof:-dof].flatten(1).norm(dim=-1)
    )
    ok &= (
        stable
        & positive_definite(h[..., :base, :base])
        & torch.isfinite(residual)
        & (
            residual.detach()
            <= config.tolerance
            * g[..., base:].detach().norm(dim=-1).clamp_min(1)
            * 1.01
        )
    )
    return q, ok, residual.detach(), counts, g, h, chains


def predict_interiors(batch, q, previous_beams, segments, material):
    """Extend the macroscopic predictor to private nodes using their local tangent."""
    primal = batch.detached()
    dof = previous_beams.shape[-1]
    ref = reference_nodes(primal.positions[:, primal.edges], BeamMesh(segments))
    _, g, h = chain_derivatives(ref, previous_beams.detach(), primal.radii, material)
    present = primal.presence != 0
    n = h.shape[-1]
    identity = torch.zeros(n, n, dtype=h.dtype, device=h.device)
    identity[dof:-dof, dof:-dof] = torch.eye(
        n - 2 * dof, dtype=h.dtype, device=h.device
    )
    blocks = condense(
        torch.where(present[..., None], g, 0),
        torch.where(present[..., None, None], h, identity),
        dof,
    )
    nodes = nodal_nodes(primal, q.detach())
    endpoints = nodes[:, primal.edges].flatten(-2)
    old = torch.cat((previous_beams[..., 0, :], previous_beams[..., -1, :]), -1)
    correction = blocks.recover(endpoints - old).reshape_as(
        previous_beams[..., 1:-1, :]
    )
    good = (blocks.info == 0) & torch.isfinite(correction).all((-2, -1))
    return previous_beams[..., 1:-1, :].detach() + torch.where(
        good[..., None, None], correction, 0
    )
