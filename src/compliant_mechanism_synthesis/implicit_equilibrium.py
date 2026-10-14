"""First-order equilibrium adjoint with saved global/private factorizations.

Newton and stability are primal operations. Autograd differentiates the residual
contraction in backward, including exact rigid ties and private beam coordinates.
"""

from dataclasses import dataclass

import torch

from .batching import DesignBatch
from .equilibrium_system import coordinate_scale, factor_system


@dataclass
class AdjointContext:
    system: object
    private_system: object | None
    response: torch.Tensor
    edge_map: object
    segments: int
    material: object
    work: object = None

    def take(self, ids):
        from .equilibrium_system import EquilibriumSystem

        def sliced(system):
            return (
                None
                if system is None
                else EquilibriumSystem(
                    system.factor[ids].clone(),
                    system.scale,
                    system.valid[ids].clone(),
                    None if system.pivots is None else system.pivots[ids].clone(),
                )
            )

        mapping = (
            (self.edge_map[0][ids].clone(), self.edge_map[1])
            if isinstance(self.edge_map, tuple)
            else self.edge_map[ids].clone()
        )
        return AdjointContext(
            sliced(self.system),
            sliced(self.private_system),
            self.response[ids].clone(),
            mapping,
            self.segments,
            self.material,
            None,
        )


def prepare_adjoint(batch, q, interior, config, material, evaluated=None):
    from .condensed_mechanics import evaluate

    if evaluated is None:
        evaluated = evaluate(
            batch.detached(),
            q.detach(),
            interior.detach(),
            config.beam_segments,
            material,
        )
    _, _, h, blocks, _, edge_map, stable, _ = evaluated
    dof = 6 if batch.dimension == 3 else 3
    base = q.shape[-1] - dof
    system = factor_system(
        h[..., :base, :base],
        coordinate_scale(h[..., :base, :base], dof, config.characteristic_length),
    )
    system.valid &= stable
    private = None
    if interior.numel():
        hii = blocks.interior_matrix
        private = factor_system(
            hii, coordinate_scale(hii, dof, config.characteristic_length)
        )
        system.valid &= private.valid.all(-1)
    if isinstance(edge_map, tuple):
        edge_map = (edge_map[0].detach(), edge_map[1])
    else:
        edge_map = edge_map.detach()
    from .mechanics_work import CURRENT

    return AdjointContext(
        system,
        private,
        blocks.interior_response.detach(),
        edge_map,
        config.beam_segments,
        material,
        CURRENT.get(),
    )


def _endpoint_transpose(context, force, size):
    mapping = context.edge_map
    if isinstance(mapping, torch.Tensor):
        return (mapping.transpose(-1, -2) @ force[..., None]).squeeze(-1).sum(1)
    local, ids = mapping
    value = (local.transpose(-1, -2) @ force[..., None]).squeeze(-1)
    return force.new_zeros(force.shape[0], size).index_add(
        -1, ids.flatten(), value.flatten(1)
    )


class EquilibriumAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, positions, radii, body, q, interior, roles, edges, presence, context
    ):
        ctx.save_for_backward(positions, radii, body, q, interior, edges, presence)
        ctx.roles = roles
        ctx.context = context
        return q.clone(), interior.clone()

    @staticmethod
    def backward(ctx, gradient_q, gradient_inner):
        from .condensed_mechanics import apply_endpoint_step, trial_energy

        positions, radii, body, q, interior, edges, presence = ctx.saved_tensors
        context = ctx.context
        from .mechanics_work import record

        record("adjoint_solves", work=context.work)
        dof = body.shape[-1]
        base = q.shape[-1] - dof
        gradient_q = torch.zeros_like(q) if gradient_q is None else gradient_q
        gradient_inner = (
            torch.zeros_like(interior) if gradient_inner is None else gradient_inner
        )
        rhs = gradient_q[..., :base]
        private_rhs = gradient_inner.flatten(-2)
        if context.private_system is not None:
            eliminated = (
                context.response.transpose(-1, -2) @ private_rhs[..., None]
            ).squeeze(-1)
            rhs = (
                rhs - _endpoint_transpose(context, eliminated, q.shape[-1])[..., :base]
            )
        lam = context.system.solve(rhs)
        full_lam = torch.cat((lam, torch.zeros_like(body)), -1)
        private_lam = None
        if context.private_system is not None:
            endpoint = apply_endpoint_step(context.edge_map, full_lam, detach=True)
            private_lam = context.private_system.solve(private_rhs) - (
                context.response @ endpoint[..., None]
            ).squeeze(-1)
            private_lam = torch.where(
                context.system.valid[:, None, None], private_lam, 0
            ).reshape_as(interior)
        # Rebuild only energy and residual, never tangent/factorization.
        with torch.enable_grad():
            p = positions.detach().requires_grad_(True)
            r = radii.detach().requires_grad_(True)
            a = body.detach().requires_grad_(True)
            unknown = q[..., :base].detach().requires_grad_(True)
            inner = interior.detach().requires_grad_(True)
            full_q = torch.cat((unknown, a), -1)
            batch = DesignBatch(p, ctx.roles, edges, r, presence)
            value = trial_energy(
                batch, full_q, inner, context.segments, context.material
            ).sum()
            variables = (unknown, inner) if inner.numel() else (unknown,)
            residual = torch.autograd.grad(value, variables, create_graph=True)
            adjoints = (-lam, -private_lam) if inner.numel() else (-lam,)
            gp, gr, ga = torch.autograd.grad(
                residual, (p, r, a), grad_outputs=adjoints, allow_unused=True
            )
        ga = torch.zeros_like(body) if ga is None else ga
        ga = ga + gradient_q[..., -dof:]
        return gp, gr, ga, None, None, None, None, None, None


def attach_equilibrium(batch, body, q, interior, config, context=None, material=None):
    from .mechanics import DEFAULT_MATERIAL

    material = DEFAULT_MATERIAL if material is None else material
    interior = (
        q.new_empty(batch.size, len(batch.edges), 0, body.shape[-1])
        if interior is None
        else interior
    )
    context = (
        prepare_adjoint(batch, q, interior, config, material)
        if context is None
        else context
    )
    return EquilibriumAdjoint.apply(
        batch.positions,
        batch.radii,
        body,
        q.detach(),
        interior.detach(),
        batch.roles,
        batch.edges,
        batch.presence,
        context,
    )


def relax_spatial(batch, body, initial, config, cached, differentiate_stiffness):
    """Primal equilibrium followed by one reusable first-derivative attachment."""
    from .condensed_mechanics import evaluate
    from .mechanics import DEFAULT_MATERIAL
    from .prescribed_motion import newton_compact, newton_masked

    primal = batch.detached()
    dof = body.shape[-1]
    base = initial.shape[-1] - dof
    q = torch.cat((initial[..., :base].detach(), body.detach()), -1)
    counts = torch.zeros(batch.size, dtype=torch.long, device=q.device)
    ok = torch.ones(batch.size, dtype=torch.bool, device=q.device)
    if cached is None:
        masked = config.newton_batch_mode == "masked" or (
            config.newton_batch_mode == "auto" and q.device.type == "xpu"
        )
        q, ok, counts = (newton_masked if masked else newton_compact)(
            primal, body.detach(), q, config
        )
    interior = q.new_empty(batch.size, len(batch.edges), 0, dof)
    # Numerical cleanup and sensitivity use native scaled solves. Detach the
    # primal completely; its iterations are not part of the backward graph.
    evaluated = evaluate(primal, q, interior, 1, DEFAULT_MATERIAL)
    context = prepare_adjoint(primal, q, interior, config, DEFAULT_MATERIAL, evaluated)
    context.system.valid &= ok
    q = torch.cat(
        (q[..., :base] - context.system.solve(evaluated[1][..., :base]), body.detach()),
        -1,
    )
    evaluated = evaluate(primal, q, interior, 1, DEFAULT_MATERIAL)
    _, g, h, _, _, _, stable, _ = evaluated
    residual = g[..., :base].norm(dim=-1)
    ok &= (
        stable
        & torch.isfinite(residual)
        & (
            residual
            <= config.tolerance * g[..., base:].norm(dim=-1).clamp_min(1) * 1.01
        )
    )
    context = prepare_adjoint(primal, q, interior, config, DEFAULT_MATERIAL, evaluated)
    ok &= context.system.valid
    context.system.valid &= ok
    differentiating = any(t.requires_grad for t in (batch.positions, batch.radii, body))
    if differentiating:
        q, _ = attach_equilibrium(batch, body, q, interior, config, context)
        if differentiate_stiffness:
            _, g, h, *_ = evaluate(batch, q, interior, 1, DEFAULT_MATERIAL)
    return q, ok, residual.detach(), counts, g, h, context


def combine_contexts(stored, edge_ids):
    """Repack saved per-design systems onto a new batch's present-edge union."""
    from .equilibrium_system import EquilibriumSystem
    from .mechanics_work import CURRENT, record

    contexts = [pair[1] for pair in stored]

    def join(systems):
        if systems[0] is None:
            return None
        return EquilibriumSystem(
            torch.cat([s.factor for s in systems]),
            systems[0].scale,
            torch.cat([s.valid for s in systems]),
            None
            if systems[0].pivots is None
            else torch.cat([s.pivots for s in systems]),
        )

    global_system = join([c.system for c in contexts])
    private_systems = []
    responses = []
    maps = []
    columns = None
    for original_ids, c in stored:
        index = torch.searchsorted(original_ids, edge_ids).clamp_max(
            len(original_ids) - 1
        )
        match = original_ids[index] == edge_ids
        responses.append(
            torch.where(match[None, :, None, None], c.response[:, index], 0)
        )
        if isinstance(c.edge_map, tuple):
            local, ids = c.edge_map
            maps.append(torch.where(match[None, :, None, None], local[:, index], 0))
            columns = (
                ids.new_zeros(len(edge_ids), ids.shape[-1])
                if columns is None
                else columns
            )
            columns = torch.where(match[:, None], ids[index], columns)
        else:
            maps.append(
                torch.where(match[None, :, None, None], c.edge_map[:, index], 0)
            )
        if c.private_system is not None:
            s = c.private_system
            n = s.factor.shape[-1]
            eye = torch.eye(n, dtype=s.factor.dtype, device=s.factor.device)
            factor = torch.where(match[None, :, None, None], s.factor[:, index], eye)
            valid = torch.where(match[None], s.valid[:, index], True)
            pivots = (
                None
                if s.pivots is None
                else torch.where(
                    match[None, :, None],
                    s.pivots[:, index],
                    torch.arange(1, n + 1, device=eye.device, dtype=s.pivots.dtype),
                )
            )
            private_systems.append(EquilibriumSystem(factor, s.scale, valid, pivots))
    edge_map = torch.cat(maps) if columns is None else (torch.cat(maps), columns)
    record("cached_factorizations_reused", len(contexts))
    return AdjointContext(
        global_system,
        join(private_systems) if private_systems else None,
        torch.cat(responses),
        edge_map,
        contexts[0].segments,
        contexts[0].material,
        CURRENT.get(),
    )
