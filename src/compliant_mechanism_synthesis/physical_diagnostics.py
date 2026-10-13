"""Elastic beam diagnostics in the corotational model, not a yield criterion."""

import torch

from .elements import _local_axes
from .nonlinear import configuration, rotation, rotation_vector


def beam_strains(design, q):
    """Peak endpoint normal strain and torsional engineering shear per beam.

    Euler–Bernoulli curvature is linear along an unloaded uniform beam. The
    normal-strain envelope combines axial extension with extreme-fiber bending.
    Joint stress concentrations, transverse shear and material failure are absent.
    """
    positions, angles = configuration(design, q)
    i, j = design.edges.T
    reference_delta = design.positions[j] - design.positions[i]
    delta = positions[j] - positions[i]
    length0 = reference_delta.norm(dim=-1)
    length = delta.norm(dim=-1)
    axial = (length / length0 - 1).abs()
    if design.dimension == 2:
        chord = torch.atan2(
            reference_delta[:, 0] * delta[:, 1] - reference_delta[:, 1] * delta[:, 0],
            (reference_delta * delta).sum(-1),
        )
        a, b = angles[i, 0] - chord, angles[j, 0] - chord
        curvature = (
            torch.maximum((4 * a + 2 * b).abs(), (2 * a + 4 * b).abs()) / length0
        )
        torsion = torch.zeros_like(axial)
    else:
        reference = _local_axes(reference_delta, length0)
        ra, rb = rotation(angles[i]), rotation(angles[j])
        x = delta / length[:, None]
        director = (
            (ra @ reference[:, 1, :, None]) + (rb @ reference[:, 1, :, None])
        ).squeeze(-1) / 2
        y = director - x * (x * director).sum(-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        frame = torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), dim=-2)
        a = rotation_vector(frame @ ra @ reference.transpose(-1, -2))
        b = rotation_vector(frame @ rb @ reference.transpose(-1, -2))
        curvature = (
            torch.maximum(
                (4 * a[:, 1:] + 2 * b[:, 1:]).norm(dim=-1),
                (2 * a[:, 1:] + 4 * b[:, 1:]).norm(dim=-1),
            )
            / length0
        )
        torsion = design.radii * (b[:, 0] - a[:, 0]).abs() / length0
    present = design.presence != 0
    return torch.where(present, axial + design.radii * curvature, 0), torch.where(
        present, torsion, 0
    )


def path_diagnostics(design, states):
    """Evaluate existing states only; no equilibrium solves or retained graphs."""
    if not states:
        return None
    with torch.no_grad():
        if states[0].beam_nodes is None:
            normal, shear = torch.func.vmap(lambda q: beam_strains(design, q))(
                torch.stack([state.q for state in states])
            )
        else:
            from .subdivided import BeamMesh, reference_nodes

            ids = torch.nonzero(design.presence != 0).flatten()
            segments = states[0].beam_nodes.shape[-2] - 1
            ref = reference_nodes(
                design.positions[design.edges[ids]], BeamMesh(segments)
            )
            positions = ref.reshape(-1, design.dimension)
            edges = torch.stack(
                (
                    torch.arange(len(ids) * (segments + 1), device=positions.device)
                    .reshape(-1, segments + 1)[:, :-1]
                    .flatten(),
                    torch.arange(len(ids) * (segments + 1), device=positions.device)
                    .reshape(-1, segments + 1)[:, 1:]
                    .flatten(),
                ),
                -1,
            )
            from .mechanics import Design

            mesh_design = Design(
                positions,
                ("free",) * len(positions),
                edges,
                design.radii[ids, None].expand(-1, segments).flatten(),
                positions.new_ones(len(edges)),
            )
            meshes = torch.stack([state.beam_nodes[ids] for state in states])
            q = torch.cat(
                (
                    meshes[..., : design.dimension] - ref,
                    meshes[..., design.dimension :],
                ),
                -1,
            ).flatten(1)
            n, sh = torch.func.vmap(lambda x: beam_strains(mesh_design, x))(q)
            normal = positions.new_zeros(len(states), len(design.edges)).index_copy(
                1, ids, n.reshape(len(states), len(ids), segments).amax(-1)
            )
            shear = positions.new_zeros(len(states), len(design.edges)).index_copy(
                1, ids, sh.reshape(len(states), len(ids), segments).amax(-1)
            )
        reactions = torch.stack([state.reaction for state in states])
        d = design.dimension
        flat = int(normal.argmax())
        sample, beam = divmod(flat, normal.shape[-1])
        return {
            "maximum_normal_strain": float(normal.max()),
            "maximum_torsional_shear_strain": float(shear.max()),
            "critical_beam": beam,
            "critical_factor": states[sample].factor,
            "maximum_actuator_force": float(reactions[:, :d].norm(dim=-1).max()),
            "maximum_actuator_generalized_moment": float(
                reactions[:, d:].norm(dim=-1).max()
            ),
        }
