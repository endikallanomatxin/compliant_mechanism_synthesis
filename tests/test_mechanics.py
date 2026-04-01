from __future__ import annotations

import torch

from compliant_mechanism_synthesis.common import symmetrize_adjacency
from compliant_mechanism_synthesis.data import (
    generate_graph_sample,
    generate_noise_sample,
    proximity_bias_matrix,
)
from compliant_mechanism_synthesis.mechanics import (
    FrameFEMConfig,
    GeometryRegularizationConfig,
    assemble_global_stiffness,
    effective_response,
    fixed_mobile_connectivity_penalty,
    geometric_regularization_terms,
    load_case_deformations,
    mechanical_terms,
    refine_connectivity,
    threshold_connectivity,
)


def test_global_stiffness_is_symmetric() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    stiffness = assemble_global_stiffness(
        positions.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert stiffness.shape == (1, 30, 30)
    assert torch.allclose(stiffness, stiffness.transpose(1, 2), atol=1e-5)


def test_rigid_body_effective_response_is_finite_and_symmetric() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    response_matrix, stiffness_matrix = effective_response(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert response_matrix.shape == (1, 3, 3)
    assert stiffness_matrix.shape == (1, 3, 3)
    assert torch.isfinite(response_matrix).all()
    assert torch.isfinite(stiffness_matrix).all()
    assert torch.allclose(response_matrix, response_matrix.transpose(1, 2), atol=1e-5)
    assert torch.allclose(stiffness_matrix, stiffness_matrix.transpose(1, 2), atol=1e-5)


def test_load_case_deformations_match_response_shape() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    deformations, response_matrix = load_case_deformations(
        positions.unsqueeze(0),
        roles.unsqueeze(0),
        adjacency.unsqueeze(0),
    )
    assert deformations.shape == (1, 3, 10, 2)
    assert response_matrix.shape == (1, 3, 3)
    assert torch.isfinite(deformations).all()
    assert torch.isfinite(response_matrix).all()


def test_mechanical_terms_have_expected_keys() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    response_matrix, stiffness_matrix = effective_response(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    terms = mechanical_terms(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert terms["response_matrix"].shape == (1, 3, 3)
    assert terms["stiffness_matrix"].shape == (1, 3, 3)
    assert torch.allclose(terms["response_matrix"], response_matrix)
    assert torch.allclose(terms["stiffness_matrix"], stiffness_matrix)
    assert terms["connectivity_penalty"].shape == (1,)
    assert terms["fixed_mobile_connectivity_penalty"].shape == (1,)
    assert terms["sparsity"].shape == (1,)
    assert terms["material"].shape == (1,)
    assert terms["short_beam_penalty"].shape == (1,)
    assert terms["long_beam_penalty"].shape == (1,)
    assert terms["thin_diameter_penalty"].shape == (1,)
    assert terms["thick_diameter_penalty"].shape == (1,)
    assert terms["node_spacing_penalty"].shape == (1,)
    assert terms["boundary_penalty"].shape == (1,)


def test_geometric_regularization_penalizes_extreme_lengths_and_diameters() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [0.0025, 0.0], [0.15, 0.0]]], dtype=torch.float32
    )
    roles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    adjacency = torch.tensor(
        [[[0.0, 0.1, 0.95], [0.1, 0.0, 0.0], [0.95, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(
            min_length=1e-3,
            max_length=2e-2,
            min_diameter=2.5e-4,
            max_diameter=1.5e-3,
        ),
        frame_config=FrameFEMConfig(workspace_size=0.2, r_max=1e-3),
    )
    assert penalties["short_beam_penalty"][0] > 0.0
    assert penalties["long_beam_penalty"][0] > 0.0
    assert penalties["thin_diameter_penalty"][0] > 0.0
    assert penalties["thick_diameter_penalty"][0] > 0.0


def test_thin_diameter_penalty_is_zero_for_absent_or_fabricable_bars() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]], dtype=torch.float32
    )
    roles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    config = GeometryRegularizationConfig(min_diameter=2e-4, max_diameter=2e-3)
    frame = FrameFEMConfig(workspace_size=0.2, r_max=1e-3)

    absent = geometric_regularization_terms(
        positions,
        roles,
        torch.zeros((1, 3, 3), dtype=torch.float32),
        config,
        frame_config=frame,
    )
    fabricable = geometric_regularization_terms(
        positions,
        roles,
        torch.tensor(
            [[[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        config,
        frame_config=frame,
    )

    assert torch.isclose(absent["thin_diameter_penalty"][0], torch.tensor(0.0))
    assert torch.isclose(fabricable["thin_diameter_penalty"][0], torch.tensor(0.0))


def test_geometric_regularization_penalizes_node_clustering_and_boundary_crowding() -> (
    None
):
    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.01, 0.02],
                [0.015, 0.021],
            ]
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)
    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(
            min_free_node_spacing=5e-3,
            boundary_margin=5e-3,
        ),
        frame_config=FrameFEMConfig(workspace_size=0.2, r_max=1e-3),
    )
    assert penalties["node_spacing_penalty"][0] > 0.0
    assert penalties["boundary_penalty"][0] > 0.0


def test_thresholded_connectivity_is_symmetric_and_zero_diagonal() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    thresholded = threshold_connectivity(
        adjacency.unsqueeze(0), roles.unsqueeze(0), threshold=0.4
    )
    assert thresholded.shape == (1, 10, 10)
    assert torch.allclose(thresholded, thresholded.transpose(1, 2))
    assert torch.allclose(torch.diagonal(thresholded[0]), torch.zeros(10))
    assert torch.allclose(thresholded, symmetrize_adjacency(thresholded))


def test_rigid_endpoint_nodes_never_connect_directly() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    fixed = torch.where(roles == 0)[0]
    mobile = torch.where(roles == 1)[0]
    assert torch.isclose(adjacency[fixed[0], fixed[1]], torch.tensor(0.0))
    assert torch.isclose(adjacency[mobile[0], mobile[1]], torch.tensor(0.0))
    for fixed_idx in fixed.tolist():
        for mobile_idx in mobile.tolist():
            assert torch.isclose(adjacency[fixed_idx, mobile_idx], torch.tensor(0.0))

    forced = torch.ones((1, 10, 10), dtype=torch.float32)
    thresholded = threshold_connectivity(forced, roles.unsqueeze(0), threshold=0.4)
    assert torch.isclose(thresholded[0, fixed[0], fixed[1]], torch.tensor(0.0))
    assert torch.isclose(thresholded[0, mobile[0], mobile[1]], torch.tensor(0.0))
    for fixed_idx in fixed.tolist():
        for mobile_idx in mobile.tolist():
            assert torch.isclose(
                thresholded[0, fixed_idx, mobile_idx], torch.tensor(0.0)
            )


def test_sparsity_penalty_increases_with_more_active_edges() -> None:
    positions, roles, _ = generate_graph_sample(10)
    sparse = torch.zeros((1, 10, 10), dtype=torch.float32)
    sparse[:, 0, 4] = 0.2
    sparse[:, 4, 0] = 0.2
    dense = torch.full((1, 10, 10), 0.2, dtype=torch.float32)
    dense = dense - torch.diag_embed(torch.diagonal(dense, dim1=1, dim2=2))
    sparse_terms = mechanical_terms(positions.unsqueeze(0), roles.unsqueeze(0), sparse)
    dense_terms = mechanical_terms(positions.unsqueeze(0), roles.unsqueeze(0), dense)
    assert dense_terms["sparsity"][0] > sparse_terms["sparsity"][0]


def test_fixed_mobile_connectivity_penalty_detects_flying_mobile_nodes() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)

    penalty = fixed_mobile_connectivity_penalty(roles, adjacency)

    assert torch.isclose(penalty[0], torch.tensor(1.0))


def test_fixed_mobile_connectivity_penalty_drops_when_mobile_is_anchored() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)
    adjacency[0, 0, 4] = 1.0
    adjacency[0, 4, 0] = 1.0
    adjacency[0, 4, 2] = 1.0
    adjacency[0, 2, 4] = 1.0
    adjacency[0, 1, 5] = 1.0
    adjacency[0, 5, 1] = 1.0
    adjacency[0, 5, 3] = 1.0
    adjacency[0, 3, 5] = 1.0

    penalty = fixed_mobile_connectivity_penalty(roles, adjacency)

    assert torch.isclose(penalty[0], torch.tensor(0.0))


def test_proximity_bias_prefers_closer_pairs() -> None:
    positions = torch.tensor(
        [[0.10, 0.10], [0.18, 0.10], [0.75, 0.10], [0.90, 0.10]],
        dtype=torch.float32,
    )
    bias = proximity_bias_matrix(positions, length_scale=0.2)
    assert bias[0, 1] > bias[0, 2]
    assert bias[2, 3] > bias[0, 3]


def test_noise_connectivity_gives_each_node_a_local_allowed_edge() -> None:
    _, _, adjacency = generate_noise_sample(10)
    incident = adjacency.sum(dim=1)
    assert torch.all(incident > 0.0)


def test_refine_connectivity_updates_near_edges_more_than_far_edges() -> None:
    positions = torch.tensor(
        [[[0.10, 0.10], [0.18, 0.10], [0.85, 0.10], [0.92, 0.10]]],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 2, 2, 1]], dtype=torch.long)
    adjacency = torch.zeros((1, 4, 4), dtype=torch.float32)
    delta = torch.ones((1, 4, 4), dtype=torch.float32)
    updated = refine_connectivity(adjacency, positions, roles, delta, step_size=0.1)
    assert updated[0, 0, 1] > updated[0, 0, 2]


def test_refine_connectivity_decays_far_active_edges() -> None:
    positions = torch.tensor(
        [[[0.10, 0.10], [0.18, 0.10], [0.85, 0.10], [0.92, 0.10]]],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 2, 2, 1]], dtype=torch.long)
    adjacency = torch.zeros((1, 4, 4), dtype=torch.float32)
    adjacency[0, 0, 2] = 0.8
    adjacency[0, 2, 0] = 0.8
    delta = torch.ones((1, 4, 4), dtype=torch.float32)
    updated = refine_connectivity(adjacency, positions, roles, delta, step_size=0.1)
    assert updated[0, 0, 2] < adjacency[0, 0, 2]
