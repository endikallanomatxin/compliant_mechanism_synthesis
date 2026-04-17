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
    RLTrainingConfig,
    SupervisedTrainingConfig,
    load_supervised_cases,
    train_rl_refiner,
    train_supervised_refiner,
)


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
    return path, load_supervised_cases(str(path))


def test_train_rl_refiner_writes_checkpoint_and_eval_history(tmp_path: Path) -> None:
    dataset_path, cases = _build_cases(tmp_path)

    _, summary = train_rl_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=RLTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=4,
            rollout_steps=2,
            eval_every_steps=2,
            checkpoint_path=str(tmp_path / "refiner_rl.pt"),
            logdir=str(tmp_path / "runs_rl"),
            seed=5,
        ),
    )

    assert summary.checkpoint_path.exists()
    assert "total_loss" in summary.history
    assert "stiffness_loss_contribution" in summary.history
    assert "rollout_monotonicity_loss_contribution" in summary.history
    assert "eval_total_loss" in summary.history
    assert "eval_metric_reward" in summary.history


def test_train_rl_refiner_supports_gradient_accumulation(
    tmp_path: Path,
) -> None:
    dataset_path, cases = _build_cases(tmp_path)

    _, summary = train_rl_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=RLTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=1,
            gradient_accumulation_steps=2,
            num_steps=2,
            rollout_steps=2,
            eval_every_steps=2,
            checkpoint_path=str(tmp_path / "refiner_rl_accum.pt"),
            logdir=str(tmp_path / "runs_rl_accum"),
            seed=11,
        ),
    )

    checkpoint = torch.load(summary.checkpoint_path, map_location="cpu")
    assert summary.checkpoint_path.exists()
    assert checkpoint["train_config"]["gradient_accumulation_steps"] == 2
    assert len(summary.history["total_loss"]) == 2


def test_train_rl_refiner_can_initialize_from_supervised_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_path, cases = _build_cases(tmp_path)
    supervised_checkpoint = tmp_path / "refiner_supervised.pt"
    _, supervised_summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=SupervisedTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=2,
            checkpoint_path=str(supervised_checkpoint),
            logdir=str(tmp_path / "runs_supervised"),
            seed=3,
        ),
    )
    checkpoint = torch.load(supervised_summary.checkpoint_path, map_location="cpu")
    expected_model_config = checkpoint["model_config"]

    _, rl_summary = train_rl_refiner(
        optimized_cases=cases,
        train_config=RLTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=2,
            rollout_steps=2,
            eval_every_steps=2,
            init_checkpoint_path=str(supervised_summary.checkpoint_path),
            checkpoint_path=str(tmp_path / "refiner_rl_from_supervised.pt"),
            logdir=str(tmp_path / "runs_rl_from_supervised"),
            seed=7,
        ),
    )

    rl_checkpoint = torch.load(rl_summary.checkpoint_path, map_location="cpu")
    assert rl_summary.checkpoint_path.exists()
    assert rl_checkpoint["train_config"]["init_checkpoint_path"] == str(
        supervised_summary.checkpoint_path
    )
    assert rl_checkpoint["model_config"] == expected_model_config
