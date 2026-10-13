"""Element tangents assembled through exact nonlinear rigid-body kinematics."""

import torch
import torch.nn.functional as F

from .mechanics import rigid_roles
from .nonlinear import beam_energies, configuration, rotation


def planar_element_derivatives(
    original, delta, angles, radii, presence, material, with_hessian=True
):
    """Exact corotational 2D endpoint gradient/Hessian, differentiable in design.

    Analytic derivatives avoid nested reverse-mode transforms in Newton.
    """
    length0 = original.norm(dim=-1)
    length = delta.norm(dim=-1)
    unit = delta / length[:, None]
    axial = material.young_modulus * torch.pi * radii.square() / length0
    bending = material.young_modulus * torch.pi * radii.pow(4) / (2 * length0)
    chord = torch.atan2(
        original[:, 0] * delta[:, 1] - original[:, 1] * delta[:, 0],
        (original * delta).sum(-1),
    )
    a, b = angles[:, 0] - chord, angles[:, 1] - chord
    phi_gradient = (
        torch.stack((-delta[:, 1], delta[:, 0]), -1) / length.square()[:, None]
    )
    phi_force = -3 * bending * (a + b)
    force_delta = (
        axial[:, None] * (length - length0)[:, None] * unit
        + phi_force[:, None] * phi_gradient
    )
    force_angles = bending[:, None] * torch.stack((2 * a + b, a + 2 * b), -1)
    gradient = torch.cat((force_delta, force_angles), -1)
    mapping = delta.new_tensor(
        [
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    if not with_hessian:
        return (gradient @ mapping) * presence[:, None], None
    x, y = delta.unbind(-1)
    phi_hessian = (
        torch.stack(
            (2 * x * y, y.square() - x.square(), y.square() - x.square(), -2 * x * y),
            -1,
        ).reshape(-1, 2, 2)
        / length.pow(4)[:, None, None]
    )
    outer = unit[:, :, None] * unit[:, None, :]
    eye = torch.eye(2, dtype=delta.dtype, device=delta.device)
    hdd = axial[:, None, None] * (
        outer + ((length - length0) / length)[:, None, None] * (eye - outer)
    )
    hdd = (
        hdd
        + 6
        * bending[:, None, None]
        * (phi_gradient[:, :, None] * phi_gradient[:, None, :])
        + phi_force[:, None, None] * phi_hessian
    )
    hdtheta = (-3 * bending[:, None] * phi_gradient)[:, :, None].expand(-1, -1, 2)
    htt = bending[:, None, None] * delta.new_tensor([[2, 1], [1, 2]])
    h = torch.cat(
        (
            torch.cat((hdd, hdtheta), -1),
            torch.cat((hdtheta.transpose(-1, -2), htt), -1),
        ),
        -2,
    )
    return (gradient @ mapping) * presence[:, None], (
        mapping.T @ h @ mapping
    ) * presence[:, None, None]


def planar_gradient(design, q, material):
    current, angles = configuration(design, q)
    i, j = design.edges.T
    gradient, _ = planar_element_derivatives(
        design.positions[j] - design.positions[i],
        current[j] - current[i],
        torch.stack((angles[i, 0], angles[j, 0]), -1),
        design.radii,
        design.presence,
        material,
        with_hessian=False,
    )
    forces = (
        q.new_zeros(len(design.roles), 3)
        .index_add(0, i, gradient[:, :3])
        .index_add(0, j, gradient[:, 3:])
    )
    free = [i for i, r in enumerate(design.roles) if r == "free"]
    blocks = [forces[free].flatten()]
    for role in rigid_roles(design.roles):
        ids = [i for i, r in enumerate(design.roles) if r == role]
        arms = current[ids] - current[ids].mean(0)
        force = forces[ids]
        torque = (
            arms[:, 0] * force[:, 1] - arms[:, 1] * force[:, 0] + force[:, 2]
        ).sum()
        blocks.append(torch.cat((force[:, :2].sum(0), torque[None])))
    return torch.cat(blocks)


def assembled_tangent(design, q, material, chunk_size=None):
    dimension = design.dimension
    dof = 3 if dimension == 2 else 6
    current, angles = configuration(design, q)
    nodal = torch.cat((current, angles), dim=-1)
    start, end = design.edges.T
    original = design.positions[end] - design.positions[start]
    local = torch.stack((nodal[start], nodal[end]), dim=1).flatten(1)

    def element(z, reference, radius, presence):
        ends = z.reshape(2, dof)
        return beam_energies(
            reference[None],
            (ends[1, :dimension] - ends[0, :dimension])[None],
            ends[:, dimension:][None],
            radius[None],
            presence[None],
            dimension,
            material,
        )[0]

    def gradient_with_aux(z, reference, radius, presence):
        gradient = torch.func.grad(element)(z, reference, radius, presence)
        return gradient, gradient

    # jacrev already computes the element gradient; retain it as auxiliary
    # output for rigid-body curvature instead of differentiating it twice.
    if dimension == 2:
        ends = local.reshape(-1, 2, dof)
        local_grad, local_h = planar_element_derivatives(
            original,
            ends[:, 1, :2] - ends[:, 0, :2],
            ends[:, :, 2],
            design.radii,
            design.presence,
            material,
        )
    else:
        local_h, local_grad = torch.func.vmap(
            torch.func.jacrev(gradient_with_aux, chunk_size=chunk_size, has_aux=True)
        )(local, original, design.radii, design.presence)
    # Free-node maps are exactly linear. Differentiate only each rigid body's
    # small pose map instead of the full nodal map over all generalized DOFs.
    free = [k for k, r in enumerate(design.roles) if r == "free"]
    eye = torch.eye(len(q), dtype=q.dtype, device=q.device)
    mappings = [q.new_zeros(dof, len(q)) for _ in design.roles]
    for k, i in enumerate(free):
        mappings[i] = eye[k * dof : (k + 1) * dof]
    body_maps = []
    for k, role in enumerate(rigid_roles(design.roles)):
        ids = [i for i, r in enumerate(design.roles) if r == role]
        points = design.positions[ids]
        center = points.mean(0)
        offset = (len(free) + k) * dof

        def body_nodes(body, points=points, center=center):
            if dimension == 2:
                c, s = body[2].cos(), body[2].sin()
                rot = torch.stack((c, -s, s, c)).reshape(2, 2)
            else:
                rot = rotation(body[3:])
            coordinates = center + body[:dimension] + (points - center) @ rot.T
            return torch.cat(
                (coordinates, body[dimension:].expand(len(points), -1)), -1
            )

        jac = torch.func.jacrev(body_nodes)(q[offset : offset + dof])
        for local, i in enumerate(ids):
            mappings[i] = F.pad(jac[local], (offset, len(q) - offset - dof))
        body_maps.append((ids, offset, body_nodes))
    mapping = torch.stack(mappings)
    edge_j = torch.stack((mapping[start], mapping[end]), 1).flatten(1, 2)
    tangent = torch.einsum("eai,eab,ebj->ij", edge_j, local_h, edge_j)
    forces = q.new_zeros(len(design.roles), dof)
    forces = forces.index_add(0, start, local_grad[:, :dof]).index_add(
        0, end, local_grad[:, dof:]
    )
    for ids, offset, body_nodes in body_maps:

        def contracted(body, body_nodes=body_nodes, ids=ids):
            return (body_nodes(body) * forces[ids]).sum()

        curvature = torch.func.jacrev(torch.func.grad(contracted))(
            q[offset : offset + dof]
        )
        padding = (offset, len(q) - offset - dof, offset, len(q) - offset - dof)
        tangent = tangent + F.pad(curvature, padding)
    return tangent


def descent_step(hessian, gradient, regularization):
    """Newton descent with a device-local, diagonally dominant fallback.

    Avoid an eigensolve or unsupported XPU Cholesky inside Newton. Backtracking
    still tests the true potential; the fallback only determines the direction.
    """
    eye = torch.eye(hessian.shape[-1], dtype=hessian.dtype, device=hessian.device)
    matrix = hessian + regularization * eye
    step, info = torch.linalg.solve_ex(matrix, -gradient, check_errors=False)
    descending = (
        (info == 0) & torch.isfinite(step).all(-1) & ((gradient * step).sum(-1) < 0)
    )
    indices = torch.nonzero(~descending).flatten()
    if len(indices):
        h = hessian[indices]
        diagonal = h.diagonal(dim1=-2, dim2=-1)
        lower = (diagonal - (h.abs().sum(-1) - diagonal.abs())).amin(-1)
        margin = h.abs().amax(dim=(-2, -1)).clamp_min(1) * 1e-8 + regularization
        shift = (-lower).clamp_min(0) + margin
        fallback = torch.linalg.solve(
            h + shift[:, None, None] * eye, -gradient[indices]
        )
        step = step.index_copy(0, indices, fallback)
    return step
