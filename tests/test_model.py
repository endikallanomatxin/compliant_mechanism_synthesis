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
    target_stiffness = torch.rand(batch_size, 3, 3)
    target_stiffness = 0.5 * (target_stiffness + target_stiffness.transpose(1, 2))
    current_stiffness = torch.rand(batch_size, 3, 3)
    current_stiffness = 0.5 * (current_stiffness + current_stiffness.transpose(1, 2))
    residual_stiffness = target_stiffness - current_stiffness
    nodal_mechanics = torch.rand(batch_size, num_nodes, 7)
    timesteps = torch.rand(batch_size)
    position_noise_levels = torch.rand(batch_size)
    connectivity_noise_levels = torch.rand(batch_size)
    return (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )


def test_model_output_shapes() -> None:
    (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    ) = _dummy_inputs()
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )
    assert outputs["displacements"].shape == positions.shape
    assert outputs["node_latents"].shape == (2, 8, 32)
    assert outputs["delta_scores"].shape == adjacency.shape
    assert outputs["predicted_adjacency"].shape == adjacency.shape


def test_connectivity_update_is_symmetric_and_zero_diagonal() -> None:
    (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    ) = _dummy_inputs()
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )
    scores = outputs["delta_scores"]
    predicted = outputs["predicted_adjacency"]
    assert torch.allclose(scores, scores.transpose(1, 2), atol=1e-6)
    assert torch.allclose(predicted, predicted.transpose(1, 2), atol=1e-6)
    assert torch.allclose(torch.diagonal(scores, dim1=1, dim2=2), torch.zeros(2, 8))
    assert torch.allclose(torch.diagonal(predicted, dim1=1, dim2=2), torch.zeros(2, 8))


def test_connectivity_delta_comes_from_node_latent_dot_products() -> None:
    (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    ) = _dummy_inputs()
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )
    expected = torch.matmul(
        outputs["node_latents"], outputs["node_latents"].transpose(1, 2)
    ) / math.sqrt(outputs["node_latents"].shape[-1])
    expected = symmetrize_adjacency(expected)
    assert torch.allclose(outputs["delta_scores"], expected, atol=1e-6)


def test_attention_layers_cycle_distance_connectivity_free() -> None:
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=6, latent_dim=32)
    modes = [layer.mode for layer in model.layers]
    assert modes == [
        "distance",
        "connectivity",
        "free",
        "distance",
        "connectivity",
        "free",
    ]


def test_model_conditioning_depends_on_current_stiffness() -> None:
    (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    ) = _dummy_inputs(batch_size=1)
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs_a = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )
    shifted_current = current_stiffness + 0.25 * torch.eye(3).unsqueeze(0)
    shifted_residual = target_stiffness - shifted_current
    outputs_b = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        shifted_current,
        shifted_residual,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )

    assert not torch.allclose(outputs_a["displacements"], outputs_b["displacements"])


def test_model_conditioning_depends_on_noise_levels() -> None:
    (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    ) = _dummy_inputs(batch_size=1)
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs_a = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )
    outputs_b = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels + 0.25,
        connectivity_noise_levels + 0.25,
    )

    assert not torch.allclose(outputs_a["displacements"], outputs_b["displacements"])


def test_model_conditioning_depends_on_nodal_mechanics() -> None:
    (
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    ) = _dummy_inputs(batch_size=1)
    model = GraphRefinementModel(d_model=128, nhead=4, num_layers=4, latent_dim=32)
    outputs_a = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )
    outputs_b = model(
        positions,
        roles,
        adjacency,
        target_stiffness,
        current_stiffness,
        residual_stiffness,
        nodal_mechanics + 0.25,
        timesteps,
        position_noise_levels,
        connectivity_noise_levels,
    )

    assert not torch.allclose(outputs_a["displacements"], outputs_b["displacements"])
