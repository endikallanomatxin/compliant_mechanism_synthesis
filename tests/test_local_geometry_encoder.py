from __future__ import annotations

import torch

from compliant_mechanism_synthesis.models.refiner import (
    LocalNodeGeometryEncoder,
    SupervisedRefinerConfig,
    _incident_bar_pair_features,
)


def test_incident_bar_pair_features_encode_minimum_relative_angle() -> None:
    directions_2d = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
    weights = torch.tensor([[[0.75, 0.25]]], dtype=torch.float32)
    radii = torch.tensor([[[0.40, 0.60]]], dtype=torch.float32)

    features = _incident_bar_pair_features(directions_2d, lengths, weights, radii)

    pair_01 = features[0, 0, 0, 1]
    pair_10 = features[0, 0, 1, 0]
    assert torch.isclose(pair_01[0], torch.tensor(0.0))
    assert torch.isclose(pair_01[1], torch.tensor(1.0))
    assert torch.isclose(pair_01[2], torch.tensor(2.0))
    assert torch.isclose(pair_01[3], torch.tensor(3.0))
    assert torch.isclose(pair_01[4], torch.tensor(0.75))
    assert torch.isclose(pair_01[5], torch.tensor(0.25))
    assert torch.isclose(pair_01[6], torch.tensor(0.40))
    assert torch.isclose(pair_01[7], torch.tensor(0.60))
    assert torch.isclose(pair_10[0], torch.tensor(0.0))
    assert torch.isclose(pair_10[1], torch.tensor(1.0))


def test_local_geometry_encoder_is_invariant_to_incident_bar_permutations() -> None:
    torch.manual_seed(3)
    encoder = LocalNodeGeometryEncoder(
        SupervisedRefinerConfig(
            hidden_dim=32,
            connectivity_latent_dim=16,
            num_heads=4,
            local_relation_hidden_dim=16,
            local_bar_hidden_dim=32,
            local_num_heads=4,
            local_incident_bar_limit=5,
        )
    )
    encoder.eval()

    positions_a = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    adjacency_a = torch.zeros((1, 5, 5), dtype=torch.float32)
    adjacency_a[0, 0, 1:] = 1.0
    adjacency_a[0, 1:, 0] = 1.0
    edge_radius_a = torch.zeros((1, 5, 5), dtype=torch.float32)
    edge_radius_a[0, 0, 1:] = torch.tensor([0.2, 0.4, 0.6, 0.8])
    edge_radius_a[0, 1:, 0] = torch.tensor([0.2, 0.4, 0.6, 0.8])

    permutation = torch.tensor([0, 3, 1, 4, 2], dtype=torch.long)
    positions_b = positions_a[:, permutation]
    adjacency_b = adjacency_a[:, permutation][:, :, permutation]
    edge_radius_b = edge_radius_a[:, permutation][:, :, permutation]

    encoded_a = encoder(positions_a, adjacency_a, edge_radius_a)
    encoded_b = encoder(positions_b, adjacency_b, edge_radius_b)

    assert torch.allclose(encoded_a[:, 0], encoded_b[:, 0], atol=1e-5, rtol=1e-5)


def test_local_geometry_encoder_returns_hidden_dim_features() -> None:
    encoder = LocalNodeGeometryEncoder(
        SupervisedRefinerConfig(
            hidden_dim=48,
            connectivity_latent_dim=16,
            num_heads=4,
            local_relation_hidden_dim=16,
            local_bar_hidden_dim=32,
            local_num_heads=4,
            local_incident_bar_limit=5,
        )
    )
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    adjacency = torch.tensor(
        [[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    edge_radius = torch.tensor(
        [[[0.0, 0.3, 0.7], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    encoded = encoder(positions, adjacency, edge_radius)

    assert encoded.shape == (1, 3, 48)
    assert torch.isfinite(encoded).all()
