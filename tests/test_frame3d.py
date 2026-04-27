from __future__ import annotations

import torch

from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    assemble_global_stiffness,
    effective_output_stiffness,
    material_usage,
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


def _edge_radius_from_presence(presence: torch.Tensor, value: float) -> torch.Tensor:
    radius = torch.zeros_like(presence)
    active = presence > 0.0
    radius[active] = value
    return radius


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
    loss = (
        terms["generalized_stiffness"].square().mean() + terms["material_usage"].mean()
    )
    loss.backward()

    assert positions.grad is not None
    assert adjacency.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert torch.isfinite(adjacency.grad).all()


def test_mechanical_terms_include_nodal_displacements() -> None:
    positions, roles, adjacency = _sample_design()
    terms = mechanical_terms(
        positions=positions,
        roles=roles,
        adjacency=adjacency,
        frame_config=Frame3DConfig(),
    )

    nodal_displacements = terms["nodal_displacements"]
    assert nodal_displacements.shape == (1, positions.shape[1], 18)
    assert torch.isfinite(nodal_displacements).all()
    edge_von_mises = terms["edge_von_mises"]
    assert edge_von_mises.shape == (1, positions.shape[1], positions.shape[1], 6)
    assert torch.isfinite(edge_von_mises).all()


def test_zero_presence_removes_contribution_even_with_valid_edge_radius() -> None:
    positions, _, adjacency = _sample_design()
    zero_presence = torch.zeros_like(adjacency)
    edge_radius_a = _edge_radius_from_presence(torch.ones_like(adjacency), 0.25)
    edge_radius_b = _edge_radius_from_presence(torch.ones_like(adjacency), 0.75)

    stiffness_a = assemble_global_stiffness(
        positions,
        zero_presence,
        edge_radius=edge_radius_a,
    )
    stiffness_b = assemble_global_stiffness(
        positions,
        zero_presence,
        edge_radius=edge_radius_b,
    )
    usage = material_usage(
        positions,
        zero_presence,
        edge_radius=edge_radius_b,
    )

    assert torch.allclose(stiffness_a, stiffness_b)
    assert torch.allclose(usage, torch.zeros_like(usage))


def test_larger_edge_radius_increases_stiffness_and_material() -> None:
    positions, roles, adjacency = _sample_design()
    presence = (adjacency > 0.0).to(dtype=adjacency.dtype)
    small_radius = _edge_radius_from_presence(presence, 0.2)
    large_radius = _edge_radius_from_presence(presence, 0.8)

    small_stiffness = effective_output_stiffness(
        positions,
        roles,
        presence,
        edge_radius=small_radius,
    )
    large_stiffness = effective_output_stiffness(
        positions,
        roles,
        presence,
        edge_radius=large_radius,
    )
    small_usage = material_usage(positions, presence, edge_radius=small_radius)
    large_usage = material_usage(positions, presence, edge_radius=large_radius)

    assert (
        torch.linalg.matrix_norm(large_stiffness).item()
        > torch.linalg.matrix_norm(small_stiffness).item()
    )
    assert torch.all(large_usage > small_usage)


def test_presence_scales_contribution_with_fixed_edge_radius() -> None:
    positions, roles, adjacency = _sample_design()
    full_presence = (adjacency > 0.0).to(dtype=adjacency.dtype)
    low_presence = 0.25 * full_presence
    edge_radius = _edge_radius_from_presence(full_presence, 0.6)

    low_stiffness = effective_output_stiffness(
        positions,
        roles,
        low_presence,
        edge_radius=edge_radius,
    )
    full_stiffness = effective_output_stiffness(
        positions,
        roles,
        full_presence,
        edge_radius=edge_radius,
    )
    low_usage = material_usage(positions, low_presence, edge_radius=edge_radius)
    full_usage = material_usage(positions, full_presence, edge_radius=edge_radius)

    assert (
        torch.linalg.matrix_norm(full_stiffness).item()
        > torch.linalg.matrix_norm(low_stiffness).item()
    )
    assert torch.all(full_usage > low_usage)


def test_plot_design_3d_returns_a_figure() -> None:
    positions, roles, adjacency = _sample_design()
    figure = plot_design_3d(positions[0], roles[0], adjacency[0], title="sample")
    assert len(figure.axes) == 1
