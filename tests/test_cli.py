from __future__ import annotations

import random

import torch

from compliant_mechanism_synthesis.cli import (
    _bootstrap_repertoire,
    _inject_rollout_noise,
    _mechanics_condition_matrices,
    _matrix_loss,
    _monotonic_improvement_loss,
    _pure_noise_batch,
    _resolve_sample_seed,
    _sample_target_stiffnesses,
    _scheduled_learning_rate,
    _scheduled_supervised_priority,
    _scheduled_training_phase,
    _stiffness_to_response,
    _supervised_reconstruction_losses,
    TrainConfig,
)
from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    apply_free_node_update,
)
from compliant_mechanism_synthesis.mechanics import characteristic_scales, FrameFEMConfig
from compliant_mechanism_synthesis.repertoire import SimulationRepertoire
from compliant_mechanism_synthesis.scaling import normalize_generalized_stiffness_matrix


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


def test_rollout_noise_does_not_clip_existing_strong_connections() -> None:
    positions = torch.rand(1, 8, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_MOBILE] + [ROLE_FREE] * 4],
        dtype=torch.long,
    )
    adjacency = torch.zeros(1, 8, 8)
    adjacency[:, 4, 5] = 1.2
    adjacency[:, 5, 4] = 1.2

    _, noisy_adjacency = _inject_rollout_noise(
        positions,
        roles,
        adjacency,
        torch.tensor([0.0]),
        position_noise_scale=0.0,
        connectivity_noise_scale=0.0,
    )

    assert torch.isclose(noisy_adjacency[0, 4, 5], torch.tensor(1.2))
    assert torch.isclose(noisy_adjacency[0, 5, 4], torch.tensor(1.2))


def test_apply_free_node_update_does_not_clamp_free_nodes() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.95, 0.05]]],
        dtype=torch.float32,
    )
    roles = torch.tensor([[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_MOBILE, ROLE_FREE]])
    delta = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, -1.0]]],
        dtype=torch.float32,
    )

    updated = apply_free_node_update(positions, delta, roles, step_size=0.2)

    assert torch.allclose(updated[:, :4], positions[:, :4])
    assert updated[0, 4, 0] > 0.95
    assert updated[0, 4, 0] < 1.15
    assert updated[0, 4, 1] < 0.05
    assert updated[0, 4, 1] > -0.15


def test_monotonic_improvement_loss_penalizes_regressions_only() -> None:
    errors = [
        torch.tensor(3.0),
        torch.tensor(2.0),
        torch.tensor(2.5),
        torch.tensor(1.0),
    ]
    loss = _monotonic_improvement_loss(errors)
    assert torch.isclose(loss, torch.tensor((0.0 + 0.5 + 0.0) / 3.0))


def test_bootstrap_repertoire_contains_positive_definite_stiffness_cases() -> None:
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=16, repertoire_max_cases=32),
        torch.device("cpu"),
    )
    matrices = repertoire.stiffness
    eigenvalues = torch.linalg.eigvalsh(matrices)

    assert len(repertoire) == 16
    assert torch.all(eigenvalues > -5e-2)


def test_stiffness_target_sampling_from_repertoire_is_positive_definite() -> None:
    random.seed(7)
    torch.manual_seed(7)
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=24, repertoire_max_cases=32),
        torch.device("cpu"),
    )
    sampled = _sample_target_stiffnesses(8, torch.device("cpu"), repertoire)
    eigenvalues = torch.linalg.eigvalsh(sampled)

    assert sampled.shape == (8, 3, 3)
    assert torch.allclose(sampled, sampled.transpose(1, 2))
    assert torch.all(eigenvalues > -5e-2)


def test_repertoire_add_discards_nonfinite_cases() -> None:
    repertoire = SimulationRepertoire.empty(num_nodes=4, max_cases=8)
    positions = torch.rand(2, 4, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_FREE]] * 2,
        dtype=torch.long,
    )
    adjacency = torch.rand(2, 4, 4)
    stiffness = torch.eye(3).repeat(2, 1, 1)
    stiffness[1, 0, 0] = torch.nan

    repertoire.add(positions, roles, adjacency, stiffness, source_code=0)

    assert len(repertoire) == 1
    assert torch.isfinite(repertoire.stiffness).all()


def test_stiffness_target_sampling_falls_back_when_repertoire_has_nonfinite_components() -> (
    None
):
    repertoire = SimulationRepertoire.empty(num_nodes=4, max_cases=8)
    positions = torch.rand(3, 4, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_FREE]] * 3,
        dtype=torch.long,
    )
    adjacency = torch.rand(3, 4, 4)
    stiffness = torch.eye(3).repeat(3, 1, 1)
    repertoire.add(positions, roles, adjacency, stiffness, source_code=0)
    repertoire.stiffness[2, 0, 0] = torch.nan

    sampled = repertoire.sample_target_stiffness(4, torch.device("cpu"))

    assert sampled.shape == (4, 3, 3)
    assert torch.isfinite(sampled).all()


def test_matrix_loss_is_zero_for_exact_match_under_characteristic_scaling() -> None:
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=8, repertoire_max_cases=16),
        torch.device("cpu"),
    )
    specs = repertoire.stiffness
    loss = _matrix_loss(specs[:1], specs[:1])

    assert torch.isclose(loss, torch.tensor(0.0))


def test_stiffness_to_response_inverts_each_target_matrix() -> None:
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=8, repertoire_max_cases=16),
        torch.device("cpu"),
    )
    stiffness = repertoire.stiffness[:2]
    response = _stiffness_to_response(stiffness)
    eye = torch.eye(3).expand_as(stiffness)

    assert torch.allclose(stiffness @ response, eye, atol=1e-5)


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
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=8, repertoire_max_cases=16),
        torch.device("cpu"),
    )
    targets = repertoire.stiffness[:2]
    current = 0.5 * targets
    _, _, residual = _mechanics_condition_matrices(targets, current)
    expected = normalize_generalized_stiffness_matrix(
        targets - current,
        characteristic_scales(FrameFEMConfig()),
    )

    assert residual.shape == targets.shape
    assert torch.allclose(residual, expected, atol=1e-6)


def test_scheduled_supervised_priority_decays_linearly_to_one() -> None:
    assert _scheduled_supervised_priority(1, 1000, 200, 1) == 200
    assert _scheduled_supervised_priority(1000, 1000, 200, 1) == 1
    assert _scheduled_supervised_priority(1500, 1000, 200, 1) == 1


def test_scheduled_learning_rate_warms_up_linearly() -> None:
    base_lr = 5e-5

    assert torch.isclose(
        torch.tensor(_scheduled_learning_rate(1, 100, base_lr, 10, 0.1)),
        torch.tensor(base_lr / 10.0),
    )
    assert torch.isclose(
        torch.tensor(_scheduled_learning_rate(10, 100, base_lr, 10, 0.1)),
        torch.tensor(base_lr),
    )


def test_scheduled_learning_rate_cosine_anneals_to_min_scale() -> None:
    base_lr = 5e-5
    lr_mid = _scheduled_learning_rate(55, 100, base_lr, 10, 0.1)
    lr_final = _scheduled_learning_rate(100, 100, base_lr, 10, 0.1)

    assert base_lr * 0.1 < lr_mid < base_lr
    assert torch.isclose(torch.tensor(lr_final), torch.tensor(base_lr * 0.1))


def test_scheduled_training_phase_starts_with_long_supervised_blocks() -> None:
    phases = [
        _scheduled_training_phase(step, 200, 1, 1000)
        for step in range(1, 205)
    ]
    assert all(phase == "supervised" for phase in phases[:200])
    assert phases[200] == "rl"
    assert phases[201] == "supervised"


def test_scheduled_training_phase_converges_to_one_one_alternation() -> None:
    phases = [
        _scheduled_training_phase(step, 200, 1, 1000)
        for step in range(1200, 1208)
    ]
    assert phases == ["rl", "supervised", "rl", "supervised", "rl", "supervised", "rl", "supervised"]


def test_repertoire_canonical_specs_choose_positive_definite_cases() -> None:
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=16, repertoire_max_cases=32),
        torch.device("cpu"),
    )
    specs = repertoire.canonical_specs(torch.device("cpu"), max_specs=4)
    eigenvalues = torch.linalg.eigvalsh(torch.stack([matrix for _, matrix in specs], dim=0))

    assert len(specs) == 4
    assert torch.all(eigenvalues > 0.0)


def test_pure_noise_batch_is_reproducible_with_explicit_seed() -> None:
    batch_a = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)
    batch_b = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)

    for a, b in zip(batch_a, batch_b):
        assert torch.allclose(a, b)


def test_resolve_sample_seed_uses_override_when_provided() -> None:
    assert _resolve_sample_seed(17) == 17
    assert isinstance(_resolve_sample_seed(None), int)
