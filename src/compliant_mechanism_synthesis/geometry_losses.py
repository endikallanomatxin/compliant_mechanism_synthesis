"""Spatial capsule contact and full-beam length costs on sampled deformation paths.

This scores geometry, not contact forces. It does not alter equilibrium.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .condensed_mechanics import nodal_nodes
from .nonlinear import rotation, rotation_vector


def interpolate_poses(a, b, fraction):
    """Interpolate translation and shortest relative spatial rotation."""
    a, b = a.reshape(a.shape[0], -1, 6), b.reshape(b.shape[0], -1, 6)
    ra, rb = rotation(a[..., 3:]), rotation(b[..., 3:])
    relative = rotation_vector(ra.transpose(-1, -2) @ rb)
    angles = rotation_vector(ra @ rotation(relative * fraction))
    return torch.cat(
        (a[..., :3] + fraction * (b[..., :3] - a[..., :3]), angles), -1
    ).flatten(1)


def beam_curves(batch, q, mesh, subdivisions):
    """Cubic beam centerlines, using computed private nodes when available."""
    nodal = nodal_nodes(batch, q)
    if mesh is None:
        mesh = nodal[:, batch.edges]
    else:
        mesh = torch.cat(
            (
                nodal[:, batch.edges[:, 0], None],
                mesh[..., 1:-1, :],
                nodal[:, batch.edges[:, 1], None],
            ),
            -2,
        )
    native = mesh.shape[-2] - 1
    count = max(native, subdivisions)
    samples = torch.linspace(0, native, count + 1, dtype=q.dtype, device=q.device)
    index = samples.floor().long().clamp_max(native - 1)
    t = (samples - index)[None, None, :, None]
    a, b = mesh[:, :, index], mesh[:, :, index + 1]
    original = (
        batch.positions[:, batch.edges[:, 1]] - batch.positions[:, batch.edges[:, 0]]
    )
    unit = original / original.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    length = (b[..., :3] - a[..., :3]).norm(dim=-1, keepdim=True)
    ta = (rotation(a[..., 3:]) @ unit[:, :, None, :, None]).squeeze(-1) * length
    tb = (rotation(b[..., 3:]) @ unit[:, :, None, :, None]).squeeze(-1) * length
    return (
        (2 * t**3 - 3 * t**2 + 1) * a[..., :3]
        + (t**3 - 2 * t**2 + t) * ta
        + (-2 * t**3 + 3 * t**2) * b[..., :3]
        + (t**3 - t**2) * tb
    )


def path_curves(batch, qs, meshes, factors, config, input_trajectory):
    """Accepted equilibria and visual interpolation between them; no extra solves."""
    curves = []
    for j, q in enumerate(qs):
        curves.append(beam_curves(batch, q, meshes[j], config.collision_subdivisions))
        if j == len(qs) - 1:
            continue
        for k in range(1, config.collision_path_substeps):
            t = k / config.collision_path_substeps
            mid = interpolate_poses(q, qs[j + 1], t)
            progress = factors[j] + t * (factors[j + 1] - factors[j])
            mid = torch.cat((mid[..., :-6], input_trajectory.at(progress)), -1)
            mesh = None
            if meshes[j] is not None:
                a, b = meshes[j], meshes[j + 1]
                angles = interpolate_poses(
                    a.reshape(-1, 6), b.reshape(-1, 6), t
                ).reshape_as(a)[..., 3:]
                mesh = torch.cat(
                    (a[..., :3] + t * (b[..., :3] - a[..., :3]), angles), -1
                )
            curves.append(beam_curves(batch, mid, mesh, config.collision_subdivisions))
    return torch.stack(curves)


def _trim(curve, head, tail, allowance):
    """Exempt only the physical vicinity of a shared joint, not whole beam pairs."""
    delta = curve[..., 1:, :] - curve[..., :-1, :]
    length = delta.norm(dim=-1).clamp_min(1e-12)
    cumulative = torch.cat((torch.zeros_like(length[..., :1]), length.cumsum(-1)), -1)
    start = torch.maximum(cumulative[..., :-1], head * allowance)
    end = torch.minimum(cumulative[..., 1:], cumulative[..., -1:] - tail * allowance)
    lo = ((start - cumulative[..., :-1]) / length).clamp(0, 1)[..., None]
    hi = ((end - cumulative[..., :-1]) / length).clamp(0, 1)[..., None]
    return curve[..., :-1, :] + delta * lo, curve[..., :-1, :] + delta * hi, end > start


def capsule_depth(a0, a1, b0, b1, radius_sum):
    """Closest finite segments, with a separating subgradient at exact crossings.

    Squared-distance penalties alone have zero gradient at intersecting axes.
    A spatial normal supplies a valid one-sided separation direction there.
    """
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb = u.square().sum(-1).clamp_min(1e-24), v.square().sum(-1).clamp_min(1e-24)
    ab, d, e = (u * v).sum(-1), (u * w).sum(-1), (v * w).sum(-1)
    denominator = aa * bb - ab.square()
    valid = denominator > 1e-10 * aa * bb
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    s, t = (ab * e - bb * d) / safe, (aa * e - ab * d) / safe
    differences = torch.stack(
        (
            w + s[..., None] * u - t[..., None] * v,
            w - (e / bb).clamp(0, 1)[..., None] * v,
            w + u - ((e + ab) / bb).clamp(0, 1)[..., None] * v,
            w + (-d / aa).clamp(0, 1)[..., None] * u,
            w + ((ab - d) / aa).clamp(0, 1)[..., None] * u - v,
        ),
        -2,
    )
    squared = differences.square().sum(-1)
    valid = valid & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    squared = torch.cat(
        (squared[..., :1].masked_fill(~valid[..., None], torch.inf), squared[..., 1:]),
        -1,
    )
    index = squared.argmin(-1)
    closest = differences.gather(
        -2, index[..., None, None].expand(*index.shape, 1, 3)
    ).squeeze(-2)
    unit = u / aa.sqrt()[..., None]
    cross = torch.linalg.cross(unit, v / bb.sqrt()[..., None])
    cross_length = cross.norm(dim=-1, keepdim=True)
    basis_weights = unit.abs().masked_fill(unit.abs() < 1e-9, 0)
    basis = torch.nn.functional.one_hot(basis_weights.argmin(-1), 3).to(unit.dtype)
    fallback = torch.linalg.cross(unit, basis)
    fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normal = torch.where(
        cross_length > 1e-8, cross / cross_length.clamp_min(1e-12), fallback
    )
    distance = torch.sqrt(closest.square().sum(-1, keepdim=True) + 1e-24)
    # A fixed one-sided normal within numerical coincidence avoids sign flips
    # from roundoff between fresh and cached equilibria.
    signed = (closest * normal).sum(-1)
    distance = torch.where(distance[..., 0] > 1e-10, distance[..., 0], signed)
    return (radius_sum - distance).clamp_min(0)


def _pair_depth(a, b, radii, flags, joint_allowance):
    allowance = (joint_allowance * radii)[None, ..., None]
    a0, a1, valid_a = _trim(a, flags[0], flags[1], allowance)
    b0, b1, valid_b = _trim(b, flags[2], flags[3], allowance)
    depth = capsule_depth(
        a0[..., :, None, :],
        a1[..., :, None, :],
        b0[..., None, :, :],
        b1[..., None, :, :],
        radii[None, ..., None, None],
    )
    depth = torch.where(valid_a[..., :, None] & valid_b[..., None, :], depth, 0)
    return depth.amax((-2, -1)).amax(0)


def collision_loss(curves, batch, config):
    """Worst sampled penetration per present beam pair, with AABB broad phase."""
    ids = torch.nonzero((batch.presence != 0).any(0)).flatten()
    curves, radii = curves[:, :, ids], batch.radii[:, ids]
    presence, edges = batch.presence[:, ids] != 0, batch.edges[ids]
    i, j = torch.triu_indices(len(ids), len(ids), 1, device=curves.device)
    margin = radii + config.collision_clearance / 2
    lower = curves.detach().amin((0, 3)) - margin.detach()[..., None]
    upper = curves.detach().amax((0, 3)) + margin.detach()[..., None]
    active = presence[:, i] & presence[:, j]
    overlap = (upper[:, i] >= lower[:, j]).all(-1) & (upper[:, j] >= lower[:, i]).all(
        -1
    )
    selected = (active & overlap).any(0)
    i, j, active = i[selected], j[selected], active[:, selected]
    loss = curves.sum((0, 2, 3, 4)) * 0 + radii.sum(-1) * 0
    peak = loss.clone()
    count = torch.zeros(batch.size, dtype=torch.long, device=curves.device)
    for start in range(0, len(i), 128):
        ia, ib = i[start : start + 128], j[start : start + 128]
        a, b = curves[:, :, ia], curves[:, :, ib]
        radius_sum = radii[:, ia] + radii[:, ib] + config.collision_clearance
        ea, eb = edges[ia], edges[ib]
        flags = tuple(
            flag[None, None, :, None]
            for flag in (
                (ea[:, 0, None] == eb).any(-1),
                (ea[:, 1, None] == eb).any(-1),
                (eb[:, 0, None] == ea).any(-1),
                (eb[:, 1, None] == ea).any(-1),
            )
        )

        def evaluate(a, b, radius_sum, flags=flags):
            return _pair_depth(
                a, b, radius_sum, flags, config.collision_joint_allowance
            )

        if curves.requires_grad or radii.requires_grad:
            depth = checkpoint(evaluate, a, b, radius_sum, use_reentrant=False)
        else:
            depth = evaluate(a, b, radius_sum)
        depth = torch.where(active[:, start : start + 128], depth, 0)
        loss = loss + (depth / radius_sum.clamp_min(1e-12)).square().sum(-1)
        peak = torch.maximum(peak, depth.amax(-1))
        count = count + (depth > 1e-10).sum(-1)
    return loss, peak, count


def length_loss(curves, batch, config):
    """Preferred lengths concern complete beams, never private FEM segments."""
    lengths = (curves[..., 1:, :] - curves[..., :-1, :]).norm(dim=-1).sum(-1)
    rest = (
        batch.positions[:, batch.edges[:, 1]] - batch.positions[:, batch.edges[:, 0]]
    ).norm(dim=-1)
    shortest = torch.minimum(rest, lengths.amin(0))
    longest = torch.maximum(rest, lengths.amax(0))
    penalty = (
        (config.beam_length_min - shortest).clamp_min(0) / config.beam_length_min
    ).square() + (
        (longest - config.beam_length_max).clamp_min(0) / config.beam_length_max
    ).square()
    return (penalty * batch.presence).sum(-1), shortest, longest
