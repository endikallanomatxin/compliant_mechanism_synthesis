from __future__ import annotations

import torch

from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.models.refiner import GraphAttentionBlock


def test_graph_attention_block_supports_context_and_stress_heads() -> None:
    block = GraphAttentionBlock(hidden_dim=512, num_heads=16)
    hidden = torch.randn(2, 5, 512)
    adjacency = torch.rand(2, 5, 5)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    adjacency[:, torch.arange(5), torch.arange(5)] = 0.0
    positions = torch.rand(2, 5, 3)
    context_tokens = torch.randn(2, 2, 512)
    edge_head_conditioning = torch.randn(2, 4, 5, 5)

    output = block(
        hidden,
        adjacency,
        positions,
        context_tokens=context_tokens,
        edge_head_conditioning=edge_head_conditioning,
    )

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


def test_supervised_refiner_uses_hybrid_attention_defaults() -> None:
    model = SupervisedRefiner(SupervisedRefinerConfig())

    assert model.config.hidden_dim == 1024
    assert model.config.latent_dim == 512
    assert model.config.num_attention_layers == 6
    assert model.config.num_heads == 16
    assert model.edge_von_mises_mlp[-1].out_features == 4
    assert all(layer.num_heads == 16 for layer in model.layers)
