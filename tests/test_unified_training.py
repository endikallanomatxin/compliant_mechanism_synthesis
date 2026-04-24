from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
)
from compliant_mechanism_synthesis.models import SupervisedRefinerConfig
from compliant_mechanism_synthesis.training import (
    FlowCurriculumTrainingConfig,
    flow_step_predictions,
    load_training_cases,
    local_flow_step_targets,
    local_flow_targets,
    make_training_batch,
    rollout_step_schedule,
    train_flow_refiner,
)
from compliant_mechanism_synthesis.training.unified import _aggregate_rollout_losses


def _build_cases(tmp_path: Path):
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=4,
            device="cpu",
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )
    return path, load_training_cases(str(path))


def test_rollout_step_schedule_reaches_final_time_one() -> None:
    initial_times = torch.tensor([0.0, 0.25, 0.9])
    step_times, step_sizes = rollout_step_schedule(
        initial_times, num_integration_steps=4
    )

    assert len(step_times) == 4
    assert torch.allclose(step_times[0], initial_times)
    assert torch.allclose(step_times[-1] + step_sizes, torch.ones_like(initial_times))


def test_local_flow_targets_use_current_state_and_remaining_time(
    tmp_path: Path,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)

    target_position_velocity, target_adjacency_velocity = local_flow_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        epsilon=1e-4,
    )
    remaining = (1.0 - batch.initial_times).clamp_min(1e-4)[:, None, None]
    assert torch.allclose(
        batch.initial_structures.positions + remaining * target_position_velocity,
        batch.oracle_structures.positions,
        atol=1e-5,
    )
    assert torch.allclose(
        batch.initial_structures.adjacency + remaining * target_adjacency_velocity,
        batch.oracle_structures.adjacency,
        atol=1e-5,
    )


def test_local_flow_step_targets_use_dt_correctly(tmp_path: Path) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    _, step_sizes = rollout_step_schedule(batch.initial_times, num_integration_steps=4)

    target_position_velocity, target_adjacency_velocity = local_flow_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        epsilon=1e-4,
    )
    target_position_step, target_adjacency_step = local_flow_step_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        step_sizes,
        epsilon=1e-4,
    )

    step_scale = step_sizes[:, None, None]
    assert torch.allclose(target_position_step, step_scale * target_position_velocity)
    assert torch.allclose(target_adjacency_step, step_scale * target_adjacency_velocity)


def test_flow_step_predictions_supervise_effective_step_not_raw_velocity() -> None:
    position_velocity = torch.tensor([[[10.0, 0.0, 0.0]]])
    adjacency_velocity = torch.tensor([[[0.0]]])
    step_sizes = torch.tensor([0.1])

    predicted_position_step, predicted_adjacency_step = flow_step_predictions(
        position_velocity=position_velocity,
        adjacency_velocity=adjacency_velocity,
        step_sizes=step_sizes,
    )

    assert torch.allclose(predicted_position_step, torch.tensor([[[1.0, 0.0, 0.0]]]))
    assert torch.allclose(predicted_adjacency_step, torch.tensor([[[0.0]]]))


def test_local_flow_step_targets_are_coherent_for_single_integration_step(
    tmp_path: Path,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    _, step_sizes = rollout_step_schedule(batch.initial_times, num_integration_steps=1)

    target_position_step, target_adjacency_step = local_flow_step_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        step_sizes,
        epsilon=1e-4,
    )

    assert torch.allclose(
        batch.initial_structures.positions + target_position_step,
        batch.oracle_structures.positions,
        atol=1e-5,
    )
    assert torch.allclose(
        batch.initial_structures.adjacency + target_adjacency_step,
        batch.oracle_structures.adjacency,
        atol=1e-5,
    )


def test_aggregate_rollout_losses_means_steps_and_applies_time_weighted_physics() -> (
    None
):
    total_loss, loss_terms = _aggregate_rollout_losses(
        supervised_step_losses=[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        physical_step_losses=[torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])],
        weighted_physical_step_losses=[
            torch.tensor([0.5, 0.6]),
            torch.tensor([1.4, 1.6]),
        ],
        weighted_stiffness_step_losses=[
            torch.tensor([0.2, 0.3]),
            torch.tensor([0.5, 0.7]),
        ],
        weighted_stress_step_losses=[
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.9, 0.9]),
        ],
        style_kl=torch.tensor([0.1, 0.2]),
        lambda_sup=0.5,
        lambda_phys=2.0,
        lambda_kl=3.0,
    )

    expected = torch.tensor(
        [
            0.5 * ((1.0 + 3.0) / 2.0) + 2.0 * ((0.5 + 1.4) / 2.0) + 3.0 * 0.1,
            0.5 * ((2.0 + 4.0) / 2.0) + 2.0 * ((0.6 + 1.6) / 2.0) + 3.0 * 0.2,
        ]
    ).mean()
    assert torch.allclose(total_loss, expected)
    assert torch.allclose(loss_terms["supervised_loss"], torch.tensor(2.5))
    assert torch.allclose(loss_terms["physical_loss"], torch.tensor(6.5))
    assert torch.allclose(
        loss_terms["time_weighted_physical_loss"], torch.tensor(1.025)
    )


def test_train_flow_refiner_writes_checkpoint_and_history(tmp_path: Path) -> None:
    dataset_path, cases = _build_cases(tmp_path)

    _, summary = train_flow_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=FlowCurriculumTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=3,
            eval_fraction=0.0,
            checkpoint_path=str(tmp_path / "refiner.pt"),
            logdir=str(tmp_path / "runs_train"),
            seed=5,
        ),
    )

    checkpoint = torch.load(summary.checkpoint_path, map_location="cpu")
    assert summary.checkpoint_path.exists()
    assert "total_loss" in summary.history
    assert "supervised_loss_contribution" in summary.history
    assert "physical_loss_contribution" in summary.history
    assert checkpoint["train_config"]["num_integration_steps"] == 4
    assert "style_sample_dropout" not in checkpoint["train_config"]


def test_train_flow_refiner_supports_warm_start_from_checkpoint(tmp_path: Path) -> None:
    dataset_path, cases = _build_cases(tmp_path)
    initial_checkpoint = tmp_path / "initial.pt"
    _, initial_summary = train_flow_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=FlowCurriculumTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=2,
            eval_fraction=0.0,
            checkpoint_path=str(initial_checkpoint),
            logdir=str(tmp_path / "runs_initial"),
            seed=7,
        ),
    )

    _, summary = train_flow_refiner(
        optimized_cases=cases,
        train_config=FlowCurriculumTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=2,
            eval_fraction=0.0,
            init_checkpoint_path=str(initial_summary.checkpoint_path),
            checkpoint_path=str(tmp_path / "warm_started.pt"),
            logdir=str(tmp_path / "runs_warm_started"),
            seed=11,
        ),
    )

    checkpoint = torch.load(summary.checkpoint_path, map_location="cpu")
    assert checkpoint["train_config"]["init_checkpoint_path"] == str(
        initial_summary.checkpoint_path
    )


def test_flow_curriculum_config_has_no_sample_style_dropout() -> None:
    config = FlowCurriculumTrainingConfig(dataset_path="dataset.pt")
    assert not hasattr(config, "style_sample_dropout")
