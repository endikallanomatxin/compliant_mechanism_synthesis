from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from compliant_mechanism_synthesis.cli import (
    _bootstrap_repertoire,
    _inject_rollout_noise,
    _mechanics_condition_matrices,
    _matrix_loss,
    _rollout_continuous_improvement_loss,
    _pure_noise_batch,
    _resolve_sample_seed,
    _sample_target_stiffnesses,
    _sample_mixed_rl_targets,
    _sample_mixed_rl_starts,
    _sample_mixed_supervised_teachers,
    _scheduled_learning_rate,
    _scheduled_supervised_priority,
    _scheduled_training_phase,
    _stiffness_to_response,
    _supervised_reconstruction_losses,
    _geometry_regularization_config,
    TrainConfig,
    rollout_refinement,
)
from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    apply_free_node_update,
)
from compliant_mechanism_synthesis.mechanics import (
    characteristic_scales,
    FrameFEMConfig,
    mechanical_terms,
)
from compliant_mechanism_synthesis.model import GraphRefinementModel
from compliant_mechanism_synthesis.repertoire import (
    SimulationRepertoire,
    SOURCE_RANDOM_INITIALIZATION,
    SOURCE_RL,
)
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


def test_rollout_continuous_improvement_loss_penalizes_regressions_more_than_improvements() -> None:
    errors = [
        torch.tensor(3.0),
        torch.tensor(2.0),
        torch.tensor(2.5),
        torch.tensor(1.0),
    ]
    loss = _rollout_continuous_improvement_loss(errors, scale=0.1)
    expected = torch.stack(
        [
            F.softplus(torch.tensor(-1.0 / 0.1)),
            F.softplus(torch.tensor(0.5 / 0.1)),
            F.softplus(torch.tensor(-1.5 / 0.1)),
        ]
    ).mean()
    assert torch.isclose(loss, expected)


def test_rollout_continuous_improvement_loss_rewards_larger_improvements() -> None:
    small_improvement = _rollout_continuous_improvement_loss(
        [torch.tensor(3.0), torch.tensor(2.9)],
        scale=0.1,
    )
    large_improvement = _rollout_continuous_improvement_loss(
        [torch.tensor(3.0), torch.tensor(2.0)],
        scale=0.1,
    )
    regression = _rollout_continuous_improvement_loss(
        [torch.tensor(3.0), torch.tensor(3.2)],
        scale=0.1,
    )

    assert large_improvement < small_improvement
    assert regression > small_improvement


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


def test_stiffness_target_sampling_with_noise_stays_positive_definite() -> None:
    random.seed(7)
    torch.manual_seed(7)
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=24, repertoire_max_cases=32),
        torch.device("cpu"),
    )
    sampled = repertoire.sample_target_stiffness(
        8,
        torch.device("cpu"),
        target_noise_scale=0.1,
    )
    eigenvalues = torch.linalg.eigvalsh(sampled)

    assert sampled.shape == (8, 3, 3)
    assert torch.allclose(sampled, sampled.transpose(1, 2))
    assert torch.all(eigenvalues > -5e-2)


def test_repertoire_case_sampling_with_rarity_weights_returns_valid_shapes() -> None:
    repertoire = _bootstrap_repertoire(
        TrainConfig(batch_size=8, repertoire_bootstrap_cases=24, repertoire_max_cases=32),
        torch.device("cpu"),
    )
    positions, roles, adjacency, stiffness = repertoire.sample_cases(
        6,
        torch.device("cpu"),
    )

    assert positions.shape == (6, repertoire.positions.shape[1], 2)
    assert roles.shape == (6, repertoire.roles.shape[1])
    assert adjacency.shape == (6, repertoire.adjacency.shape[1], repertoire.adjacency.shape[2])
    assert stiffness.shape == (6, 3, 3)


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


def test_mixed_supervised_teachers_use_fresh_and_rl_repertoire_halves() -> None:
    device = torch.device("cpu")
    fresh_positions = torch.rand(4, 4, 2)
    fresh_roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_FREE]] * 4,
        dtype=torch.long,
    )
    fresh_adjacency = torch.rand(4, 4, 4)
    repertoire = SimulationRepertoire.empty(num_nodes=4, max_cases=8)
    rl_positions = torch.rand(4, 4, 2) + 10.0
    rl_adjacency = torch.rand(4, 4, 4)
    repertoire.add(
        rl_positions,
        fresh_roles,
        rl_adjacency,
        torch.eye(3).repeat(4, 1, 1),
        source_code=SOURCE_RL,
    )

    teacher_positions, _, _ = _sample_mixed_supervised_teachers(
        fresh_positions,
        fresh_roles,
        fresh_adjacency,
        repertoire,
        device,
    )

    assert torch.allclose(teacher_positions[:2], fresh_positions[:2])
    assert torch.all(teacher_positions[2:] > 1.0)


def test_mixed_rl_targets_use_random_initialization_and_rl_repertoire_halves() -> None:
    device = torch.device("cpu")
    repertoire = SimulationRepertoire.empty(num_nodes=4, max_cases=8)
    positions = torch.rand(1, 4, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_FREE]],
        dtype=torch.long,
    )
    adjacency = torch.rand(1, 4, 4)
    random_initialization_stiffness = torch.eye(3).repeat(1, 1, 1)
    rl_stiffness = 2.0 * torch.eye(3).repeat(1, 1, 1)
    repertoire.add(
        positions,
        roles,
        adjacency,
        random_initialization_stiffness,
        source_code=SOURCE_RANDOM_INITIALIZATION,
    )
    repertoire.add(
        positions + 1.0,
        roles,
        adjacency,
        rl_stiffness,
        source_code=SOURCE_RL,
    )

    targets = _sample_mixed_rl_targets(4, device, repertoire, target_noise_scale=0.0)

    assert torch.allclose(
        targets[:2],
        random_initialization_stiffness.expand(2, -1, -1),
    )
    assert torch.allclose(targets[2:], rl_stiffness.expand(2, -1, -1))


def test_mixed_rl_targets_apply_target_noise() -> None:
    device = torch.device("cpu")
    repertoire = SimulationRepertoire.empty(num_nodes=4, max_cases=8)
    positions = torch.rand(2, 4, 2)
    roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_FREE]] * 2,
        dtype=torch.long,
    )
    adjacency = torch.rand(2, 4, 4)
    repertoire.add(
        positions[:1],
        roles[:1],
        adjacency[:1],
        torch.eye(3).unsqueeze(0),
        source_code=SOURCE_RANDOM_INITIALIZATION,
    )
    repertoire.add(
        positions[1:],
        roles[1:],
        adjacency[1:],
        (2.0 * torch.eye(3)).unsqueeze(0),
        source_code=SOURCE_RL,
    )

    torch.manual_seed(7)
    targets = _sample_mixed_rl_targets(4, device, repertoire, target_noise_scale=0.1)

    assert torch.allclose(targets, targets.transpose(1, 2))
    assert torch.all(torch.linalg.eigvalsh(targets) > -5e-2)
    assert not torch.allclose(targets[:2], torch.eye(3).expand(2, -1, -1))


def test_mixed_rl_starts_use_fresh_random_init_and_rl_sources() -> None:
    device = torch.device("cpu")
    fresh_positions = torch.rand(6, 4, 2)
    fresh_roles = torch.tensor(
        [[ROLE_FIXED, ROLE_FIXED, ROLE_MOBILE, ROLE_FREE]] * 6,
        dtype=torch.long,
    )
    fresh_adjacency = torch.rand(6, 4, 4)
    repertoire = SimulationRepertoire.empty(num_nodes=4, max_cases=8)
    repertoire.add(
        torch.rand(2, 4, 2) + 10.0,
        fresh_roles[:2],
        torch.rand(2, 4, 4),
        torch.eye(3).repeat(2, 1, 1),
        source_code=SOURCE_RANDOM_INITIALIZATION,
    )
    repertoire.add(
        torch.rand(2, 4, 2) + 20.0,
        fresh_roles[:2],
        torch.rand(2, 4, 4),
        2.0 * torch.eye(3).repeat(2, 1, 1),
        source_code=SOURCE_RL,
    )

    positions, roles, adjacency = _sample_mixed_rl_starts(
        fresh_positions,
        fresh_roles,
        fresh_adjacency,
        repertoire,
        device,
        position_noise_scale=0.0,
        connectivity_noise_scale=0.0,
    )

    assert positions.shape == fresh_positions.shape
    assert roles.shape == fresh_roles.shape
    assert adjacency.shape == fresh_adjacency.shape
    assert torch.allclose(positions[:2], fresh_positions[:2])
    assert torch.all(positions[2:4] > 1.0)
    assert torch.all(positions[4:] > 10.0)


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


def test_repertoire_canonical_specs_avoid_duplicate_fallback_cases() -> None:
    positions, roles, adjacency = _pure_noise_batch(256, 64, torch.device("cpu"), seed=123)
    repertoire = SimulationRepertoire.empty(num_nodes=64, max_cases=512)
    generated_terms = mechanical_terms(positions, roles, adjacency)
    repertoire.add(
        positions,
        roles,
        adjacency,
        generated_terms["stiffness_matrix"],
        source_code=0,
    )

    specs = repertoire.canonical_specs(torch.device("cpu"), max_specs=6)
    canonical_matrices = torch.stack([matrix for _, matrix in specs], dim=0)
    unique_components = torch.unique(canonical_matrices.reshape(6, -1), dim=0)

    assert len(specs) == 6
    assert unique_components.shape[0] == 6


def test_pure_noise_batch_is_reproducible_with_explicit_seed() -> None:
    batch_a = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)
    batch_b = _pure_noise_batch(2, 8, torch.device("cpu"), seed=123)

    for a, b in zip(batch_a, batch_b):
        assert torch.allclose(a, b)


def test_rollout_terms_are_computed_on_refined_state_before_noise() -> None:
    device = torch.device("cpu")
    config = TrainConfig(
        device="cpu",
        batch_size=2,
        num_nodes=8,
        rollout_steps=2,
        rollout_position_noise=0.05,
        rollout_connectivity_noise=0.10,
    )
    model = GraphRefinementModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=2,
        node_effect_dim=config.node_effect_dim,
    )
    positions, roles, adjacency = _pure_noise_batch(
        config.batch_size,
        config.num_nodes,
        device,
        seed=123,
    )
    target_stiffness = torch.eye(3).repeat(config.batch_size, 1, 1)
    rollout = rollout_refinement(
        model,
        positions,
        roles,
        adjacency,
        target_stiffness,
        steps=config.rollout_steps,
        position_step_size=config.position_step_size,
        connectivity_step_size=config.connectivity_step_size,
        base_time=torch.ones((config.batch_size,), device=device),
        position_noise_scale=config.rollout_position_noise,
        connectivity_noise_scale=config.rollout_connectivity_noise,
        geometry_config=_geometry_regularization_config(config),
    )

    state = rollout[0]
    refined_terms = mechanical_terms(
        state["refined_positions"],
        roles,
        state["refined_adjacency"],
        geometry_config=_geometry_regularization_config(config),
    )
    noisy_terms = mechanical_terms(
        state["positions"],
        roles,
        state["adjacency"],
        geometry_config=_geometry_regularization_config(config),
    )

    assert torch.allclose(
        state["terms"]["stiffness_matrix"],
        refined_terms["stiffness_matrix"],
    )
    assert not torch.allclose(
        state["terms"]["stiffness_matrix"],
        noisy_terms["stiffness_matrix"],
    )


def test_resolve_sample_seed_uses_override_when_provided() -> None:
    assert _resolve_sample_seed(17) == 17
    assert isinstance(_resolve_sample_seed(None), int)
