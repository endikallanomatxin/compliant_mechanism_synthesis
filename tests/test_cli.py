from __future__ import annotations

import torch

from compliant_mechanism_synthesis.cli import (
    _fixed_stiffness_target_specs,
    _inject_rollout_noise,
    _mechanics_condition_matrices,
    _matrix_loss,
    _monotonic_improvement_loss,
    _normalize_residual_matrix,
    _pure_noise_batch,
    _resolve_sample_seed,
    _sample_stiffness_targets,
    _stiffness_to_response,
    _supervised_reconstruction_losses,
    _sample_supervised_denoising_batch,
    _target_normalization,
)
from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
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


def test_fixed_stiffness_targets_are_positive_definite() -> None:
    specs = _fixed_stiffness_target_specs(torch.device("cpu"))
    matrices = torch.stack([matrix for _, matrix in specs], dim=0)
    eigenvalues = torch.linalg.eigvalsh(matrices)

    assert len(specs) == 6
    assert torch.all(eigenvalues > 0.0)


def test_stiffness_target_sampling_draws_from_fixed_library() -> None:
    library = torch.stack(
        [matrix for _, matrix in _fixed_stiffness_target_specs(torch.device("cpu"))],
        dim=0,
    )
    sampled = _sample_stiffness_targets(8, torch.device("cpu"))

    assert sampled.shape == (8, 3, 3)
    assert torch.allclose(sampled, sampled.transpose(1, 2))
    assert all(
        any(torch.allclose(target, base) for base in library) for target in sampled
    )


def test_matrix_loss_is_zero_for_exact_match_under_target_normalization() -> None:
    specs = torch.stack(
        [matrix for _, matrix in _fixed_stiffness_target_specs(torch.device("cpu"))],
        dim=0,
    )
    normalization = _target_normalization(specs)
    loss = _matrix_loss(specs[:1], specs[:1], normalization)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_stiffness_to_response_inverts_each_target_matrix() -> None:
    stiffness = torch.stack(
        [
            matrix
            for _, matrix in _fixed_stiffness_target_specs(torch.device("cpu"))[:2]
        ],
        dim=0,
    )
    response = _stiffness_to_response(stiffness)
    eye = torch.eye(3).expand_as(stiffness)

    assert torch.allclose(stiffness @ response, eye, atol=1e-5)


def test_supervised_denoising_batch_preserves_fixed_and_mobile_positions() -> None:
    positions = torch.rand(2, 8, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_MOBILE] + [ROLE_FREE] * 4] * 2,
        dtype=torch.long,
    )
    adjacency = torch.rand(2, 8, 8)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    noisy_positions, noisy_adjacency = _sample_supervised_denoising_batch(
        positions,
        roles,
        adjacency,
        position_noise_scale=0.05,
        connectivity_noise_scale=0.10,
    )

    assert torch.allclose(noisy_positions[:, :4], positions[:, :4])
    assert noisy_adjacency.shape == adjacency.shape


def test_supervised_reconstruction_losses_vanish_on_exact_match() -> None:
    positions = torch.rand(2, 8, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_MOBILE] + [ROLE_FREE] * 4] * 2,
        dtype=torch.long,
    )
    adjacency = torch.rand(2, 8, 8)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    position_loss, adjacency_loss = _supervised_reconstruction_losses(
        positions,
        positions,
        roles,
        adjacency,
        adjacency,
    )

    assert torch.isclose(position_loss, torch.tensor(0.0))
    assert torch.isclose(adjacency_loss, torch.tensor(0.0))


def test_mechanics_condition_residual_matches_difference() -> None:
    targets = torch.stack(
        [
            matrix
            for _, matrix in _fixed_stiffness_target_specs(torch.device("cpu"))[:2]
        ],
        dim=0,
    )
    current = 0.5 * targets
    normalization = _target_normalization(targets)
    _, _, residual = _mechanics_condition_matrices(targets, current, normalization)
    expected = _normalize_residual_matrix(targets - current, normalization)

    assert residual.shape == targets.shape
    assert torch.allclose(residual, expected, atol=1e-6)


def test_pure_noise_batch_is_reproducible_with_explicit_seed() -> None:
    batch_a = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)
    batch_b = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)

    for a, b in zip(batch_a, batch_b):
        assert torch.allclose(a, b)


def test_resolve_sample_seed_uses_override_when_provided() -> None:
    assert _resolve_sample_seed(17) == 17
    assert isinstance(_resolve_sample_seed(None), int)
