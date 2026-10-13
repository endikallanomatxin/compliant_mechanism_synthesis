"""Vectorized spatial/planar beam chains and exact Newton block condensation.

Internal mesh nodes belong to individual beams; they are never graph junctions.
Arrays have shape (..., beam, mesh_node, component). No local Newton loop is
nested inside global Newton: residual and tangent are condensed together.
"""

import math
from dataclasses import dataclass

import torch

from .mechanics import DEFAULT_MATERIAL
from .nonlinear import beam_energies
from .tangents import planar_element_derivatives


@dataclass(frozen=True)
class BeamMesh:
    segments: int = 4

    def __post_init__(self):
        if type(self.segments) is not int or self.segments < 1:
            raise ValueError("Beam segments must be a positive integer")

    @property
    def internal_dofs(self):
        return 3 * (self.segments - 1)


DEFAULT_BEAM_MESH = BeamMesh()


def reference_nodes(endpoints, mesh=DEFAULT_BEAM_MESH):
    """(..., beam, 2, dimension) endpoints -> private straight reference mesh."""
    fraction = torch.linspace(
        0, 1, mesh.segments + 1, dtype=endpoints.dtype, device=endpoints.device
    )
    return endpoints[..., :1, :] + fraction[:, None] * (
        endpoints[..., 1:2, :] - endpoints[..., :1, :]
    )


def initial_nodes(reference, endpoints):
    """Straight deformed chord and interpolated endpoint rotation coordinates."""
    fraction = torch.linspace(
        0, 1, reference.shape[-2], dtype=reference.dtype, device=reference.device
    )
    return endpoints[..., :1, :] + fraction[:, None] * (
        endpoints[..., 1:2, :] - endpoints[..., :1, :]
    )


def chain_energy(reference, nodes, radii, material=DEFAULT_MATERIAL):
    """Vectorized segment energy for line search; no tangent preparation."""
    dimension = reference.shape[-1]
    original = reference[..., 1:, :] - reference[..., :-1, :]
    delta = nodes[..., 1:, :dimension] - nodes[..., :-1, :dimension]
    angles = torch.stack((nodes[..., :-1, dimension:], nodes[..., 1:, dimension:]), -2)
    radius = radii[..., None].expand(original.shape[:-1]).reshape(-1)
    value = beam_energies(
        original.reshape(-1, dimension),
        delta.reshape(-1, dimension),
        angles.reshape(-1, 2, nodes.shape[-1] - dimension),
        radius,
        torch.ones_like(radius),
        dimension,
        material,
    )
    return value.reshape(original.shape[:-1]).sum(-1)


def chain_derivatives(reference, nodes, radii, material=DEFAULT_MATERIAL):
    """Batch all private segments, then assemble each beam's local system."""
    dimension, dof = reference.shape[-1], nodes.shape[-1]
    segments = nodes.shape[-2] - 1
    original = reference[..., 1:, :] - reference[..., :-1, :]
    radius = radii[..., None].expand(original.shape[:-1]).reshape(-1)
    if dimension == 2:
        delta = nodes[..., 1:, :2] - nodes[..., :-1, :2]
        angles = torch.stack((nodes[..., :-1, 2], nodes[..., 1:, 2]), -1)
        g, h = planar_element_derivatives(
            original.reshape(-1, 2),
            delta.reshape(-1, 2),
            angles.reshape(-1, 2),
            radius,
            torch.ones_like(radius),
            material,
        )
    else:
        # Translation invariance leaves nine independent coordinates, not twelve.
        # Differentiate once: jacrev returns the force as auxiliary output too.
        local = torch.cat(
            (
                nodes[..., 1:, :3] - nodes[..., :-1, :3],
                nodes[..., :-1, 3:],
                nodes[..., 1:, 3:],
            ),
            -1,
        ).reshape(-1, 9)

        def element(z, reference, radius):
            return beam_energies(
                reference[None],
                z[:3][None],
                z[3:].reshape(1, 2, 3),
                radius[None],
                torch.ones_like(radius[None]),
                3,
                material,
            )[0]

        derivative = torch.func.grad_and_value(element)

        def force_with_aux(z, reference, radius):
            force, value = derivative(z, reference, radius)
            return force, (force, value)

        h, (g, values) = torch.func.vmap(
            torch.func.jacrev(force_with_aux, has_aux=True)
        )(local, original.reshape(-1, 3), radius)
        # Constant change of coordinates restores the endpoint assembly format.
        indices = torch.tensor(
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 6, 7, 8], device=nodes.device
        )
        signs = nodes.new_tensor([-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        g = g[:, indices] * signs
        h = h[:, indices[:, None], indices[None, :]] * signs[:, None] * signs[None, :]
    shape = original.shape[:-2]
    g = g.reshape(*shape, segments, 2 * dof)
    h = h.reshape(*shape, segments, 2 * dof, 2 * dof)
    n = dof * (segments + 1)
    ids = (
        torch.arange(segments, device=nodes.device)[:, None] * dof
        + torch.arange(2 * dof, device=nodes.device)[None, :]
    )
    residual = nodes.new_zeros(*shape, n).index_add(-1, ids.flatten(), g.flatten(-2))
    matrix_ids = (ids[:, :, None] * n + ids[:, None, :]).flatten()
    tangent = (
        nodes.new_zeros(*shape, n * n)
        .index_add(-1, matrix_ids, h.flatten(-3))
        .reshape(*shape, n, n)
    )
    energy = (
        chain_energy(reference, nodes, radii, material)
        if dimension == 2
        else values.reshape(*shape, segments).sum(-1)
    )
    return energy, residual, tangent


@dataclass
class CondensedBeam:
    residual: torch.Tensor
    tangent: torch.Tensor
    interior_response: torch.Tensor
    interior_offset: torch.Tensor
    info: torch.Tensor

    def recover(self, endpoint_step):
        """Interior Newton correction after the assembled endpoint solve."""
        return -self.interior_offset - (
            self.interior_response @ endpoint_step[..., None]
        ).squeeze(-1)


def condense(residual, tangent, endpoint_dofs=3):
    """Schur complement including nonzero interior residual; no explicit inverse.

    A failed solve is reported in info, never replaced with artificial stiffness.
    Internal stability must be checked separately from condensed stability.
    """
    n = residual.shape[-1]
    external = torch.cat(
        (
            torch.arange(endpoint_dofs, device=residual.device),
            torch.arange(n - endpoint_dofs, n, device=residual.device),
        )
    )
    interior = torch.arange(endpoint_dofs, n - endpoint_dofs, device=residual.device)
    ge, gi = residual[..., external], residual[..., interior]
    hee = tangent[..., external[:, None], external[None, :]]
    hii = tangent[..., interior[:, None], interior[None, :]]
    hie = tangent[..., interior[:, None], external[None, :]]
    hei = tangent[..., external[:, None], interior[None, :]]
    if not len(interior):
        return CondensedBeam(
            ge,
            hee,
            hie,
            gi,
            torch.zeros(residual.shape[:-1], dtype=torch.int32, device=residual.device),
        )
    solution, info = torch.linalg.solve_ex(
        hii, torch.cat((hie, gi[..., None]), -1), check_errors=False
    )
    response, offset = solution[..., :-1], solution[..., -1]
    reduced_g = ge - (hei @ offset[..., None]).squeeze(-1)
    reduced_h = hee - hei @ response
    return CondensedBeam(
        reduced_g, (reduced_h + reduced_h.transpose(-1, -2)) / 2, response, offset, info
    )


def assemble_condensed(beams, edges, node_count):
    """Assemble the macroscopic nodal residual/tangent; shared graph, batch axes.

    Rigid-body tying/clamping follows assembly. Endpoint rotations are retained.
    """
    n = 3 * node_count
    ids = (edges[..., None] * 3 + torch.arange(3, device=edges.device)).flatten(-2)
    shape = beams.residual.shape[:-2]
    residual = beams.residual.new_zeros(*shape, n).index_add(
        -1, ids.flatten(), beams.residual.flatten(-2)
    )
    matrix_ids = (ids[:, :, None] * n + ids[:, None, :]).flatten()
    tangent = (
        beams.tangent.new_zeros(*shape, n * n)
        .index_add(-1, matrix_ids, beams.tangent.flatten(-3))
        .reshape(*shape, n, n)
    )
    return residual, tangent


@dataclass(frozen=True)
class NetworkConfig:
    segments: int = DEFAULT_BEAM_MESH.segments
    max_iterations: int = 45
    max_backtracks: int = 18
    tolerance: float = 1e-7

    def __post_init__(self):
        BeamMesh(self.segments)
        if (
            any(
                type(value) is not int or value < 1
                for value in (self.max_iterations, self.max_backtracks)
            )
            or not math.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("Invalid subdivided network solver settings")


DEFAULT_NETWORK = NetworkConfig()


@dataclass
class NetworkState:
    nodes: torch.Tensor
    beam_nodes: torch.Tensor
    energy: torch.Tensor
    residual: torch.Tensor
    tangent: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor


def solve_network(
    reference,
    edges,
    radii,
    prescribed,
    boundary,
    config=DEFAULT_NETWORK,
    material=DEFAULT_MATERIAL,
    initial=None,
):
    """Batched planar networks with arbitrary prescribed nodal coordinates.

    reference: (B,N,2), radii: (B,E); shared edges: (E,2), prescribed: (N,3).
    boundary: (B,N,3), absolute translations and planar rotations. Beam interiors
    are private. Supply the previous NetworkState for nonlinear continuation.
    This validation solver uses exact condensation, including interior residuals.
    """
    from .spectral import positive_definite

    mesh = BeamMesh(config.segments)
    if reference.shape[-1] != 2 or boundary.shape[-1] != 3:
        raise ValueError("Subdivided network solver currently supports 2D only")
    count = reference.shape[-2]
    free = torch.nonzero(~prescribed.flatten()).flatten()
    ref = reference_nodes(reference[:, edges], mesh)
    # Newton is a numerical solve; design derivatives use implicit corrections.
    original_ref, original_radii = ref, radii
    ref, radii = ref.detach(), radii.detach()
    nodes = torch.cat(
        (reference.detach(), reference.new_zeros(*reference.shape[:-1], 1)), -1
    )
    if initial is not None:
        nodes = initial.nodes.detach().clone()
    nodes = torch.where(prescribed[None], boundary.detach(), nodes)
    chains = (
        initial_nodes(ref, nodes[:, edges])
        if initial is None
        else initial.beam_nodes.detach().clone()
    )
    chains = torch.cat(
        (
            nodes[:, edges[:, 0], None],
            chains[..., 1:-1, :],
            nodes[:, edges[:, 1], None],
        ),
        -2,
    )
    finished = torch.zeros(
        reference.shape[0], dtype=torch.bool, device=reference.device
    )
    iterations = torch.zeros_like(finished, dtype=torch.long)
    for _ in range(config.max_iterations):
        values, g, h = chain_derivatives(ref, chains, radii, material)
        blocks = condense(g, h)
        global_g, global_h = assemble_condensed(blocks, edges, count)
        interior_g = g[..., 3:-3]
        norm = torch.maximum(
            global_g[:, free].norm(dim=-1), interior_g.flatten(1).norm(dim=-1)
        )
        solved = (norm <= config.tolerance) & (blocks.info == 0).all(-1)
        finished |= solved
        working = ~finished
        if not bool(working.any()):
            break
        matrix = global_h[:, free[:, None], free[None, :]]
        eye = torch.eye(len(free), dtype=matrix.dtype, device=matrix.device)
        safe = torch.where(working[:, None, None], matrix, eye)
        step, info = torch.linalg.solve_ex(
            safe,
            -torch.where(working[:, None], global_g[:, free], 0),
            check_errors=False,
        )
        valid = (info == 0) & (blocks.info == 0).all(-1) & torch.isfinite(step).all(-1)
        delta = (
            torch.zeros_like(nodes)
            .flatten(1)
            .index_copy(1, free, step)
            .reshape_as(nodes)
        )
        edge_step = delta[:, edges].flatten(-2)
        interior_step = blocks.recover(edge_step).reshape(
            *chains.shape[:-2], mesh.segments - 1, 3
        )
        change = torch.cat(
            (delta[:, edges[:, 0], None], interior_step, delta[:, edges[:, 1], None]),
            -2,
        )
        slope = (g * change.flatten(-2)).sum((-1, -2))
        waiting = working & valid
        accepted_any = torch.zeros_like(waiting)
        for k in range(config.max_backtracks):
            rate = 0.5**k
            trial = chains + rate * change
            potential = chain_energy(ref, trial, radii, material).sum(-1)
            accept = (
                waiting
                & torch.isfinite(potential)
                & (
                    potential
                    <= values.sum(-1)
                    + 1e-4 * rate * slope
                    + 32
                    * torch.finfo(potential.dtype).eps
                    * values.sum(-1).abs().clamp_min(1)
                )
            )
            nodes = torch.where(accept[:, None, None], nodes + rate * delta, nodes)
            chains = torch.where(accept[:, None, None, None], trial, chains)
            waiting &= ~accept
            accepted_any |= accept
            if not bool(waiting.any()):
                break
        iterations += working.to(iterations.dtype)
        if bool((working & ~accepted_any).any()):
            # Return unsuccessful branches explicitly; callers may subdivide input.
            break
    # Two detached-Jacobian corrections give implicit first derivatives while
    # avoiding an autograd graph through Newton/backtracking decisions.
    nodes = torch.where(prescribed[None], boundary, nodes.detach())
    chains = torch.cat(
        (
            nodes[:, edges[:, 0], None],
            chains[..., 1:-1, :].detach(),
            nodes[:, edges[:, 1], None],
        ),
        -2,
    )
    for _ in range(2):
        values, g, h = chain_derivatives(original_ref, chains, original_radii, material)
        blocks = condense(g, h.detach())
        global_g, global_h = assemble_condensed(blocks, edges, count)
        internal_stable = (
            positive_definite(h.detach()[..., 3:-3, 3:-3])
            if mesh.internal_dofs
            else torch.ones_like(blocks.info, dtype=torch.bool)
        )
        matrix = global_h[:, free[:, None], free[None, :]].detach()
        stable = (
            internal_stable.all(-1)
            & positive_definite(matrix)
            & (blocks.info == 0).all(-1)
        )
        eye = torch.eye(len(free), dtype=matrix.dtype, device=matrix.device)
        step = torch.linalg.solve(
            torch.where(stable[:, None, None], matrix, eye),
            -torch.where(stable[:, None], global_g[:, free], 0),
        )
        delta = (
            torch.zeros_like(nodes)
            .flatten(1)
            .index_copy(1, free, step)
            .reshape_as(nodes)
        )
        interior_step = blocks.recover(delta[:, edges].flatten(-2)).reshape(
            *chains.shape[:-2], mesh.segments - 1, 3
        )
        nodes = nodes + delta
        chains = torch.cat(
            (
                nodes[:, edges[:, 0], None],
                chains[..., 1:-1, :]
                + torch.where(stable[:, None, None, None], interior_step, 0),
                nodes[:, edges[:, 1], None],
            ),
            -2,
        )
    values, g, h = chain_derivatives(original_ref, chains, original_radii, material)
    blocks = condense(g, h)
    global_g, global_h = assemble_condensed(blocks, edges, count)
    residual = torch.maximum(
        global_g[:, free].norm(dim=-1), g[..., 3:-3].flatten(1).norm(dim=-1)
    )
    ok = stable & torch.isfinite(residual) & (residual <= config.tolerance)
    return NetworkState(
        nodes, chains, values.sum(-1), residual.detach(), global_h, ok, iterations
    )
