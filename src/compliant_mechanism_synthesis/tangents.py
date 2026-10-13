"""Element tangents assembled through exact nonlinear rigid-body kinematics."""

import torch
import torch.nn.functional as F

from .nonlinear import beam_energies, configuration, rotation


def assembled_tangent(design, q, material, chunk_size=None):
    dimension = design.dimension
    dof = 3 if dimension == 2 else 6
    current, angles = configuration(design, q)
    nodal = torch.cat((current, angles), dim=-1)
    i, j = design.edges.T
    original = design.positions[j] - design.positions[i]
    local = torch.stack((nodal[i], nodal[j]), dim=1).flatten(1)

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

    local_grad = torch.func.vmap(torch.func.grad(element))(
        local, original, design.radii, design.presence
    )
    local_h = torch.func.vmap(
        torch.func.jacrev(torch.func.grad(element), chunk_size=chunk_size)
    )(local, original, design.radii, design.presence)
    mobile = [k for k, role in enumerate(design.roles) if role == "mobile"]
    free = [k for k, role in enumerate(design.roles) if role == "free"]
    base = len(free) * dof
    mobile_points = design.positions[mobile]
    center = mobile_points.mean(0)

    def body_nodes(body):
        if dimension == 2:
            c, s = torch.cos(body[2]), torch.sin(body[2])
            r = torch.stack((c, -s, s, c)).reshape(2, 2)
        else:
            r = rotation(body[3:])
        positions = center + body[:dimension] + (mobile_points - center) @ r.T
        return torch.cat((positions, body[dimension:].expand(len(mobile), -1)), -1)

    body_j = torch.func.jacrev(body_nodes)(q[base:])
    eye = torch.eye(len(q), dtype=q.dtype, device=q.device)
    mappings = []
    for k, role in enumerate(design.roles):
        if role == "free":
            offset = free.index(k) * dof
            mappings.append(eye[offset : offset + dof])
        elif role == "mobile":
            mappings.append(F.pad(body_j[mobile.index(k)], (base, 0)))
        else:
            mappings.append(q.new_zeros(dof, len(q)))
    mapping = torch.stack(mappings)
    edge_j = torch.stack((mapping[i], mapping[j]), 1).flatten(1, 2)
    tangent = torch.einsum("eai,eab,ebj->ij", edge_j, local_h, edge_j)
    forces = q.new_zeros(len(design.roles), dof)
    forces = forces.index_add(0, i, local_grad[:, :dof])
    forces = forces.index_add(0, j, local_grad[:, dof:])

    # Jᵀ H J alone misses curvature of mobile attachments under rigid rotation.
    def contracted(body):
        return (body_nodes(body) * forces[mobile]).sum()

    curvature = torch.func.jacrev(torch.func.grad(contracted))(q[base:])
    return tangent + F.pad(curvature, (base, 0, base, 0))


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
