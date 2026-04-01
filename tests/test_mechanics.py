from __future__ import annotations

import torch

from compliant_mechanism_synthesis.common import symmetrize_adjacency
from compliant_mechanism_synthesis.data import generate_graph_sample
from compliant_mechanism_synthesis.mechanics import (
    assemble_global_stiffness,
    effective_properties,
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


def test_rigid_body_effective_properties_are_finite() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    properties, compliance = effective_properties(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert properties.shape == (1, 3)
    assert compliance.shape == (1, 3)
    assert torch.isfinite(properties).all()
    assert torch.isfinite(compliance).all()


def test_mechanical_terms_have_expected_keys() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    properties, compliance = effective_properties(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    terms = mechanical_terms(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert terms["properties"].shape == (1, 3)
    assert torch.allclose(terms["properties"], properties)
    assert torch.allclose(terms["compliance"], compliance)
    assert terms["connectivity_penalty"].shape == (1,)
    assert terms["material"].shape == (1,)
    assert terms["binarization"].shape == (1,)


def test_thresholded_connectivity_is_symmetric_and_zero_diagonal() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    thresholded = threshold_connectivity(adjacency.unsqueeze(0), threshold=0.4)
    assert thresholded.shape == (1, 10, 10)
    assert torch.allclose(thresholded, thresholded.transpose(1, 2))
    assert torch.allclose(torch.diagonal(thresholded[0]), torch.zeros(10))
    assert torch.allclose(thresholded, symmetrize_adjacency(thresholded))
