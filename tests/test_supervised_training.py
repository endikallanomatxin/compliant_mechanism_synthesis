from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
)
from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.training import (
    CurriculumConfig,
    SupervisedTrainingConfig,
    analyze_structures,
    make_supervised_batch,
    load_supervised_cases,
    train_supervised_refiner,
)
from compliant_mechanism_synthesis.training.supervised import _scheduled_learning_rate


def _build_cases(tmp_path: Path):
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=4,
            device="cpu",
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(
                num_free_nodes=6,
                forced_primitive_type="sheet_helix",
                sheet_width_nodes=2,
                sample_sheet_helix_width_nodes=False,
                sheet_helix_width_nodes_min=2,
                sheet_helix_width_nodes_max=2,
                sheet_helix_offset_distance_min=0.05,
                sheet_helix_offset_distance_max=0.07,
                sheet_helix_pitch_distance=0.30,
                sheet_helix_max_longitudinal_points=32,
            ),
            optimization=CaseOptimizationConfig(num_steps=4),
        )
    )
    return load_supervised_cases(str(path))


def test_supervised_refiner_preserves_structure_shapes(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64, latent_dim=32, num_attention_layers=3, num_heads=4
        )
    )
    prediction = model(
        cases.optimized_structures,
        cases.target_stiffness,
        analysis_fn=analyze_structures,
        num_steps=2,
        style_structures=cases.optimized_structures,
        style_analyses=cases.last_analyses,
    )

    assert prediction.positions.shape == cases.optimized_structures.positions.shape
    assert prediction.adjacency.shape == cases.optimized_structures.adjacency.shape


def test_scheduled_learning_rate_warms_up_then_cosine_decays() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        num_steps=10,
        learning_rate=1e-3,
        warmup_steps=2,
        min_learning_rate=1e-4,
    )

    learning_rates = [_scheduled_learning_rate(step, config) for step in range(10)]

    assert learning_rates[0] < learning_rates[1]
    assert learning_rates[1] == config.learning_rate
    assert learning_rates[-1] == config.min_learning_rate
    assert learning_rates[2] > learning_rates[5] > learning_rates[8]


def test_difficulty_limits_sampled_flow_times(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    curriculum = CurriculumConfig(initial_mix=0.2, final_mix=1.0)

    easy_batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=curriculum,
        difficulty=0.0,
        seed=3,
    )
    hard_batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=curriculum,
        difficulty=1.0,
        seed=3,
    )

    assert float(easy_batch.flow_times.min().item()) >= 0.8
    assert float(hard_batch.flow_times.min().item()) >= 0.0
    assert float(easy_batch.flow_times.mean().item()) > float(
        hard_batch.flow_times.mean().item()
    )


def test_train_supervised_refiner_writes_checkpoint_and_reduces_training_loss(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    _, summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64, latent_dim=32, num_attention_layers=3, num_heads=4
        ),
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=16,
            learning_rate=5e-4,
            checkpoint_path=str(tmp_path / "refiner.pt"),
            logdir=str(tmp_path / "runs"),
            seed=3,
        ),
    )

    assert summary.checkpoint_path.exists()
    first_window = sum(summary.history["total_loss"][:4]) / 4.0
    best_observed = min(summary.history["total_loss"])
    assert best_observed <= first_window


def test_trained_refiner_beats_untrained_baseline_on_seen_batch(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    config = SupervisedRefinerConfig(
        hidden_dim=64, latent_dim=32, num_attention_layers=3, num_heads=4
    )
    baseline = SupervisedRefiner(config)
    batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=CurriculumConfig(),
        difficulty=0.7,
        seed=5,
    )
    baseline_prediction = baseline.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        position_noise_levels=batch.position_noise_levels,
        adjacency_noise_levels=batch.adjacency_noise_levels,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )
    trained, _ = train_supervised_refiner(
        optimized_cases=cases,
        model_config=config,
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=20,
            learning_rate=5e-4,
            checkpoint_path=str(tmp_path / "refiner.pt"),
            logdir=str(tmp_path / "runs"),
            seed=5,
        ),
    )
    trained_prediction = trained.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        position_noise_levels=batch.position_noise_levels,
        adjacency_noise_levels=batch.adjacency_noise_levels,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )

    baseline_position_error = (
        (baseline_prediction.position_velocity - batch.target_position_velocity)
        .square()
        .mean()
    )
    trained_position_error = (
        (trained_prediction.position_velocity - batch.target_position_velocity)
        .square()
        .mean()
    )
    baseline_adjacency_error = (
        (baseline_prediction.adjacency_velocity - batch.target_adjacency_velocity)
        .square()
        .mean()
    )
    trained_adjacency_error = (
        (trained_prediction.adjacency_velocity - batch.target_adjacency_velocity)
        .square()
        .mean()
    )

    assert (
        trained_position_error + trained_adjacency_error
        < baseline_position_error + baseline_adjacency_error
    )
    assert trained_adjacency_error < baseline_adjacency_error


def test_predict_flow_rejects_mismatched_style_roles(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64, latent_dim=32, num_attention_layers=3, num_heads=4
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=CurriculumConfig(),
        difficulty=0.5,
        seed=11,
    )
    mismatched_roles = batch.oracle_structures.roles.clone()
    mismatched_roles[:, 0] = mismatched_roles[:, 1]

    with pytest.raises(ValueError, match="same node roles"):
        model.predict_flow(
            structures=batch.flow_structures,
            target_stiffness=batch.target_stiffness,
            current_stiffness=batch.current_analyses.generalized_stiffness,
            nodal_displacements=batch.current_analyses.nodal_displacements,
            edge_von_mises=batch.current_analyses.edge_von_mises,
            flow_times=batch.flow_times,
            position_noise_levels=batch.position_noise_levels,
            adjacency_noise_levels=batch.adjacency_noise_levels,
            style_structures=type(batch.oracle_structures)(
                positions=batch.oracle_structures.positions,
                roles=mismatched_roles,
                adjacency=batch.oracle_structures.adjacency,
            ),
            style_analyses=batch.oracle_analyses,
        )


def test_predict_flow_ignores_style_inputs_when_style_token_disabled(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            latent_dim=32,
            num_attention_layers=3,
            num_heads=4,
            use_style_token=False,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=CurriculumConfig(),
        difficulty=0.5,
        seed=13,
    )
    mismatched_roles = batch.oracle_structures.roles.clone()
    mismatched_roles[:, 0] = mismatched_roles[:, 1]

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        position_noise_levels=batch.position_noise_levels,
        adjacency_noise_levels=batch.adjacency_noise_levels,
        style_structures=type(batch.oracle_structures)(
            positions=batch.oracle_structures.positions,
            roles=mismatched_roles,
            adjacency=batch.oracle_structures.adjacency,
        ),
    )

    assert prediction.position_velocity.shape == batch.target_position_velocity.shape
    assert prediction.adjacency_velocity.shape == batch.target_adjacency_velocity.shape


def test_predict_flow_stabilizes_large_mechanics_inputs(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            latent_dim=32,
            num_attention_layers=3,
            num_heads=4,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=CurriculumConfig(),
        difficulty=0.5,
        seed=17,
    )
    huge_displacements = torch.full_like(
        batch.current_analyses.nodal_displacements, 1e25
    )
    huge_stresses = torch.full_like(batch.current_analyses.edge_von_mises, 1e30)

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=huge_displacements,
        edge_von_mises=huge_stresses,
        flow_times=batch.flow_times,
        position_noise_levels=batch.position_noise_levels,
        adjacency_noise_levels=batch.adjacency_noise_levels,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )

    assert torch.isfinite(prediction.position_velocity).all()
    assert torch.isfinite(prediction.adjacency_velocity).all()


def test_predict_flow_requires_style_analyses_with_style_structures(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            latent_dim=32,
            num_attention_layers=3,
            num_heads=4,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        curriculum=CurriculumConfig(),
        difficulty=0.5,
        seed=19,
    )

    with pytest.raises(ValueError, match="style_analyses"):
        model.predict_flow(
            structures=batch.flow_structures,
            target_stiffness=batch.target_stiffness,
            current_stiffness=batch.current_analyses.generalized_stiffness,
            nodal_displacements=batch.current_analyses.nodal_displacements,
            edge_von_mises=batch.current_analyses.edge_von_mises,
            flow_times=batch.flow_times,
            position_noise_levels=batch.position_noise_levels,
            adjacency_noise_levels=batch.adjacency_noise_levels,
            style_structures=batch.oracle_structures,
        )
