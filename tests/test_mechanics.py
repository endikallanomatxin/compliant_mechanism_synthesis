from __future__ import annotations

import torch

from compliant_mechanism_synthesis.common import symmetrize_adjacency
from compliant_mechanism_synthesis.data import generate_graph_sample
from compliant_mechanism_synthesis.mechanics import (
    GeometryRegularizationConfig,
    assemble_global_stiffness,
    effective_response,
    geometric_regularization_terms,
    mechanical_terms,
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
    assert terms["material"].shape == (1,)
    assert terms["binarization"].shape == (1,)
    assert terms["short_beam_penalty"].shape == (1,)
    assert terms["long_beam_penalty"].shape == (1,)
    assert terms["thin_diameter_penalty"].shape == (1,)
    assert terms["thick_diameter_penalty"].shape == (1,)


def test_geometric_regularization_penalizes_extreme_lengths_and_diameters() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [0.05, 0.0], [1.0, 0.0]]], dtype=torch.float32
    )
    adjacency = torch.tensor(
        [[[0.0, 0.1, 0.95], [0.1, 0.0, 0.0], [0.95, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    penalties = geometric_regularization_terms(
        positions,
        adjacency,
        GeometryRegularizationConfig(
            min_length=0.10,
            max_length=0.80,
            min_diameter=0.02,
            max_diameter=0.08,
        ),
    )
    assert penalties["short_beam_penalty"][0] > 0.0
    assert penalties["long_beam_penalty"][0] > 0.0
    assert penalties["thin_diameter_penalty"][0] > 0.0
    assert penalties["thick_diameter_penalty"][0] > 0.0


def test_thresholded_connectivity_is_symmetric_and_zero_diagonal() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    thresholded = threshold_connectivity(adjacency.unsqueeze(0), threshold=0.4)
    assert thresholded.shape == (1, 10, 10)
    assert torch.allclose(thresholded, thresholded.transpose(1, 2))
    assert torch.allclose(torch.diagonal(thresholded[0]), torch.zeros(10))
    assert torch.allclose(thresholded, symmetrize_adjacency(thresholded))
