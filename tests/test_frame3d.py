from __future__ import annotations

import torch

from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    assemble_global_stiffness,
    effective_output_stiffness,
    mechanical_terms,
)
from compliant_mechanism_synthesis.roles import NodeRole
from compliant_mechanism_synthesis.visualization import plot_design_3d


def _sample_design() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        [
            [
                [0.10, 0.10, 0.10],
                [0.30, 0.10, 0.10],
                [0.10, 0.30, 0.10],
                [0.70, 0.70, 0.70],
                [0.90, 0.70, 0.70],
                [0.70, 0.90, 0.70],
                [0.45, 0.30, 0.30],
                [0.55, 0.50, 0.45],
            ]
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor(
        [
            [
                NodeRole.FIXED,
                NodeRole.FIXED,
                NodeRole.FIXED,
                NodeRole.MOBILE,
                NodeRole.MOBILE,
                NodeRole.MOBILE,
                NodeRole.FREE,
                NodeRole.FREE,
            ]
        ],
        dtype=torch.long,
    )
    adjacency = torch.zeros((1, 8, 8), dtype=torch.float32)
    edges = [
        (0, 6),
        (1, 6),
        (2, 6),
        (6, 7),
        (7, 3),
        (7, 4),
        (7, 5),
        (1, 7),
        (2, 7),
    ]
    for i, j in edges:
        adjacency[:, i, j] = 0.8
        adjacency[:, j, i] = 0.8
    return positions, roles, adjacency


def test_global_stiffness_is_symmetric() -> None:
    positions, _, adjacency = _sample_design()
    stiffness = assemble_global_stiffness(positions, adjacency)
    assert stiffness.shape == (1, 48, 48)
    assert torch.allclose(stiffness, stiffness.transpose(1, 2), atol=1e-5)


def test_effective_output_stiffness_is_symmetric_and_finite() -> None:
    positions, roles, adjacency = _sample_design()
    stiffness = effective_output_stiffness(positions, roles, adjacency)
    assert stiffness.shape == (1, 6, 6)
    assert torch.isfinite(stiffness).all()
    assert torch.allclose(stiffness, stiffness.transpose(1, 2), atol=1e-5)


def test_mechanical_terms_backward_is_finite() -> None:
    positions, roles, adjacency = _sample_design()
    positions = positions.clone().requires_grad_(True)
    adjacency = adjacency.clone().requires_grad_(True)

    terms = mechanical_terms(
        positions=positions,
        roles=roles,
        adjacency=adjacency,
        frame_config=Frame3DConfig(),
    )
    loss = terms["generalized_stiffness"].square().mean() + terms["material_usage"].mean()
    loss.backward()

    assert positions.grad is not None
    assert adjacency.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert torch.isfinite(adjacency.grad).all()


def test_plot_design_3d_returns_a_figure() -> None:
    positions, roles, adjacency = _sample_design()
    figure = plot_design_3d(positions[0], roles[0], adjacency[0], title="sample")
    assert len(figure.axes) == 1
