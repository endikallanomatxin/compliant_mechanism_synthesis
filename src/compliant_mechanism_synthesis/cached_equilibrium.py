"""Exact cached path replay, with reused first-derivative factorizations."""

from dataclasses import replace

import torch

from .mechanics import DEFAULT_MATERIAL
from .prescribed_motion import MotionPath, MotionState


def replay_path(
    design,
    records,
    stored_linearizations,
    config,
    input_trajectory,
    differentiate_stiffness,
    sample_geometry,
):
    linearized = stored_linearizations is not None
    need_mesh = config.beam_segments > 1 and (sample_geometry or linearized)
    present_ids = (
        [torch.nonzero(mask != 0).flatten() for mask in design.presence.detach().cpu()]
        if need_mesh
        else []
    )
    edge_ids = torch.nonzero((design.presence != 0).any(0)).flatten()
    replay_design = replace(
        design,
        edges=design.edges[edge_ids],
        radii=design.radii[:, edge_ids],
        presence=design.presence[:, edge_ids],
    )
    cached_states = []
    for j in range(len(records[0])):
        q = torch.stack([r[j][1] for r in records]).to(design.positions.device)
        mesh = None
        interior = None
        if need_mesh:
            from .condensed_mechanics import nodal_nodes

            interior = torch.stack(
                [
                    r[j][6]
                    .new_zeros(
                        len(design.edges),
                        config.beam_segments - 1,
                        6,
                    )
                    .index_copy(0, present_ids[i], r[j][6])
                    for i, r in enumerate(records)
                ]
            ).to(design.positions.device)
        spectrum = torch.stack([r[j][5] for r in records]).to(design.positions.device)
        if linearized:
            from .implicit_equilibrium import (
                attach_equilibrium,
                combine_contexts,
            )

            context = combine_contexts(
                [(entry[0], entry[1][j]) for entry in stored_linearizations],
                edge_ids,
            )
            private = (
                q.new_empty(design.size, len(edge_ids), 0, 6)
                if interior is None
                else interior[:, edge_ids]
            )
            factors = q.new_tensor([r[j][0] for r in records])
            body = input_trajectory.at(factors)
            q, private = attach_equilibrium(
                replay_design, body, q, private, config, context
            )
            if interior is not None:
                interior = private.new_zeros(
                    design.size,
                    len(design.edges),
                    config.beam_segments - 1,
                    6,
                ).index_copy(1, edge_ids, private)
            if differentiate_stiffness:
                from .condensed_mechanics import evaluate
                from .prescribed_motion import spatial_stiffness
                from .spectral import Spectrum

                _, g, h, *_ = evaluate(
                    replay_design,
                    q,
                    private,
                    config.beam_segments,
                    DEFAULT_MATERIAL,
                )
                free = design.roles.count("free") * 6
                k = h[:, free : free + 6, free : free + 6]
                if free:
                    coupling = h[:, :free, free : free + 6]
                    k = k - coupling.transpose(-1, -2) @ torch.linalg.solve(
                        h[:, :free, :free], coupling
                    )
                k = spatial_stiffness(
                    k,
                    q[:, free : free + 6],
                    g[:, free : free + 6],
                    3,
                    config.characteristic_length,
                )
                spectrum = Spectrum.apply(k)
        if need_mesh:
            from .condensed_mechanics import nodal_nodes

            nodal = nodal_nodes(design, q)
            mesh = torch.cat(
                (
                    nodal[:, design.edges[:, 0], None],
                    interior,
                    nodal[:, design.edges[:, 1], None],
                ),
                -2,
            )
        cached_states.append(
            MotionState(
                design.positions.new_tensor([r[j][0] for r in records]),
                q,
                design.positions.new_tensor([r[j][2] for r in records]),
                torch.tensor(
                    [r[j][3] for r in records], device=design.positions.device
                ),
                design.positions.new_tensor([r[j][4] for r in records]),
                torch.ones(
                    design.size, dtype=torch.bool, device=design.positions.device
                ),
                spectrum,
                mesh,
            )
        )
    return MotionPath(
        cached_states,
        torch.ones(design.size, dtype=torch.bool, device=design.positions.device),
        0,
    )
