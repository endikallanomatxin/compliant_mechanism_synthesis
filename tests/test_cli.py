from __future__ import annotations

import torch

from compliant_mechanism_synthesis.cli import (
    _inject_rollout_noise,
    _monotonic_improvement_loss,
    _pure_noise_batch,
    _resolve_sample_seed,
    ResponseStatistics,
)
from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    symmetric_matrix_unique_values,
)


def test_rollout_noise_preserves_adjacency_symmetry_and_zero_diagonal() -> None:
    positions = torch.rand(2, 8, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_MOBILE] + [ROLE_FREE] * 4] * 2,
        dtype=torch.long,
    )
    adjacency = torch.rand(2, 8, 8)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency, dim1=1, dim2=2))
    noisy_positions, noisy_adjacency = _inject_rollout_noise(
        positions,
        roles,
        adjacency,
        torch.tensor([1.0, 0.5]),
        position_noise_scale=0.05,
        connectivity_noise_scale=0.10,
    )
    assert noisy_positions.shape == positions.shape
    assert noisy_adjacency.shape == adjacency.shape
    assert torch.allclose(noisy_adjacency, noisy_adjacency.transpose(1, 2), atol=1e-6)
    assert torch.allclose(
        torch.diagonal(noisy_adjacency, dim1=1, dim2=2), torch.zeros(2, 8)
    )


def test_monotonic_improvement_loss_penalizes_regressions_only() -> None:
    errors = [
        torch.tensor(3.0),
        torch.tensor(2.0),
        torch.tensor(2.5),
        torch.tensor(1.0),
    ]
    loss = _monotonic_improvement_loss(errors)
    assert torch.isclose(loss, torch.tensor((0.0 + 0.5 + 0.0) / 3.0))


def test_response_statistics_clone_freezes_sampling_buffer() -> None:
    stats = ResponseStatistics.empty(
        device=torch.device("cpu"),
        capacity=8,
        covariance_regularization=1e-3,
        sampling_temperature=1.0,
    )
    first = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    second = (2.0 * torch.eye(3, dtype=torch.float32)).unsqueeze(0)
    stats.update(first)
    frozen = stats.clone()
    stats.update(second)

    assert torch.allclose(
        frozen._current_buffer(), symmetric_matrix_unique_values(first)
    )
    assert not torch.allclose(stats._current_buffer(), frozen._current_buffer())


def test_pure_noise_batch_is_reproducible_with_explicit_seed() -> None:
    batch_a = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)
    batch_b = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)

    for a, b in zip(batch_a, batch_b):
        assert torch.allclose(a, b)


def test_resolve_sample_seed_uses_override_when_provided() -> None:
    assert _resolve_sample_seed(17) == 17
    assert isinstance(_resolve_sample_seed(None), int)
