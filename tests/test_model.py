from __future__ import annotations

import math

import torch

from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    symmetrize_adjacency,
)
from compliant_mechanism_synthesis.model import GraphRefinementModel


def _dummy_inputs(batch_size: int = 2, num_nodes: int = 8) -> tuple[torch.Tensor, ...]:
    positions = torch.rand(batch_size, num_nodes, 2)
    roles = torch.tensor(
        [
            [ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_MOBILE]
            + [ROLE_FREE] * (num_nodes - 4)
        ]
        * batch_size,
        dtype=torch.long,
    )
    adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency, dim1=1, dim2=2))
    targets = torch.rand(batch_size, 3)
    timesteps = torch.rand(batch_size)
    return positions, roles, adjacency, targets, timesteps


def test_model_output_shapes() -> None:
    positions, roles, adjacency, targets, timesteps = _dummy_inputs()
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs = model(positions, roles, adjacency, targets, timesteps)
    assert outputs["displacements"].shape == positions.shape
    assert outputs["node_latents"].shape == (2, 8, 32)
    assert outputs["delta_scores"].shape == adjacency.shape
    assert outputs["predicted_adjacency"].shape == adjacency.shape


def test_connectivity_update_is_symmetric_and_zero_diagonal() -> None:
    positions, roles, adjacency, targets, timesteps = _dummy_inputs()
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs = model(positions, roles, adjacency, targets, timesteps)
    scores = outputs["delta_scores"]
    predicted = outputs["predicted_adjacency"]
    assert torch.allclose(scores, scores.transpose(1, 2), atol=1e-6)
    assert torch.allclose(predicted, predicted.transpose(1, 2), atol=1e-6)
    assert torch.allclose(torch.diagonal(scores, dim1=1, dim2=2), torch.zeros(2, 8))
    assert torch.allclose(torch.diagonal(predicted, dim1=1, dim2=2), torch.zeros(2, 8))


def test_connectivity_delta_comes_from_node_latent_dot_products() -> None:
    positions, roles, adjacency, targets, timesteps = _dummy_inputs()
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs = model(positions, roles, adjacency, targets, timesteps)
    expected = torch.matmul(
        outputs["node_latents"], outputs["node_latents"].transpose(1, 2)
    ) / math.sqrt(outputs["node_latents"].shape[-1])
    expected = symmetrize_adjacency(expected)
    assert torch.allclose(outputs["delta_scores"], expected, atol=1e-6)


def test_attention_layers_alternate_conditioning() -> None:
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=6, latent_dim=32)
    flags = [layer.conditioned for layer in model.layers]
    assert flags == [True, False, True, False, True, False]
