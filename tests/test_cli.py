from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.cli import (
    dataset_generate_main,
    sample_supervised_main,
    train_main,
    visualize_dataset_main,
)
from compliant_mechanism_synthesis.dataset import load_offline_dataset


def _generate_dataset(tmp_path: Path, name: str = "dataset.pt") -> Path:
    output_path = tmp_path / name
    dataset_generate_main(
        [
            "--num-cases",
            "2",
            "--device",
            "cpu",
            "--num-free-nodes",
            "6",
            "--optimization-steps",
            "3",
            "--output-path",
            str(output_path),
            "--logdir",
            str(tmp_path / f"runs_{name}"),
        ]
    )
    return output_path


def test_dataset_generate_main_generates_offline_dataset_and_preview(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset.pt"
    preview_dir = tmp_path / "preview"
    dataset_generate_main(
        [
            "--num-cases",
            "2",
            "--device",
            "cpu",
            "--num-free-nodes",
            "6",
            "--optimization-steps",
            "3",
            "--output-path",
            str(output_path),
            "--logdir",
            str(tmp_path / "runs"),
            "--preview-dir",
            str(preview_dir),
            "--preview-case-number",
            "2",
        ]
    )

    optimized_cases, _ = load_offline_dataset(output_path)
    assert optimized_cases.optimized_structures.positions.shape[0] == 2
    assert (preview_dir / "case_0000_primitives.png").exists()
    assert (preview_dir / "summary.txt").exists()


def test_visualize_dataset_main_renders_existing_dataset(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    output_dir = tmp_path / "viz"
    visualize_dataset_main(
        [
            "--dataset-path",
            str(output_path),
            "--output-dir",
            str(output_dir),
            "--max-cases",
            "1",
        ]
    )

    assert (output_dir / "summary.txt").exists()
    assert any(path.suffix == ".png" for path in output_dir.iterdir())


def test_train_main_writes_checkpoint(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    checkpoint_path = tmp_path / "refiner.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_train"),
        ]
    )
    assert checkpoint_path.exists()


def test_train_main_writes_default_checkpoint_inside_run_dir(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    runs_dir = tmp_path / "runs_named"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--logdir",
            str(runs_dir),
            "--name",
            "curriculum",
        ]
    )

    run_dirs = [entry for entry in runs_dir.iterdir() if entry.is_dir()]
    assert len(run_dirs) == 1
    assert run_dirs[0].name.endswith("-curriculum")
    assert (run_dirs[0] / "refiner.pt").exists()


def test_train_main_can_disable_style_conditioning(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    checkpoint_path = tmp_path / "refiner_no_style.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--no-style-conditioning",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_no_style"),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["model_config"]["use_style_conditioning"] is False
    assert checkpoint["train_config"]["use_style_conditioning"] is False


def test_train_main_uses_default_curriculum_config(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    checkpoint_path = tmp_path / "refiner_defaults.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_defaults"),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["train_config"]["num_integration_steps"] == 3
    assert checkpoint["train_config"]["physical_weight_start"] == 0.0
    assert checkpoint["train_config"]["physical_weight_end"] == 0.01
    assert checkpoint["train_config"]["supervised_weight_start"] == 1.0
    assert checkpoint["train_config"]["supervised_weight_end"] == 0.0
    assert checkpoint["train_config"]["stress_loss_weight"] == 0.01
    assert "style_sample_dropout" not in checkpoint["train_config"]


def test_train_main_accepts_curriculum_flags(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    checkpoint_path = tmp_path / "refiner_flags.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--num-integration-steps",
            "6",
            "--supervised-transition-start-step",
            "10",
            "--supervised-transition-end-step",
            "20",
            "--physical-transition-start-step",
            "30",
            "--physical-transition-end-step",
            "40",
            "--style-token-dropout",
            "0.25",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_flags"),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["train_config"]["num_integration_steps"] == 6
    assert checkpoint["train_config"]["supervised_transition_start_step"] == 10
    assert checkpoint["train_config"]["supervised_transition_end_step"] == 20
    assert checkpoint["train_config"]["physical_transition_start_step"] == 30
    assert checkpoint["train_config"]["physical_transition_end_step"] == 40
    assert checkpoint["train_config"]["style_token_dropout"] == 0.25


def test_train_main_supports_warm_start(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    initial_checkpoint_path = tmp_path / "refiner_initial.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--checkpoint-path",
            str(initial_checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_initial"),
        ]
    )

    warm_started_checkpoint_path = tmp_path / "refiner_warm_started.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--init-checkpoint-path",
            str(initial_checkpoint_path),
            "--checkpoint-path",
            str(warm_started_checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_warm_started"),
        ]
    )

    checkpoint = torch.load(warm_started_checkpoint_path, map_location="cpu")
    assert checkpoint["train_config"]["init_checkpoint_path"] == str(
        initial_checkpoint_path
    )


def test_sample_supervised_main_writes_comparison_outputs(tmp_path: Path) -> None:
    output_path = _generate_dataset(tmp_path)
    checkpoint_path = tmp_path / "refiner.pt"
    train_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_train"),
        ]
    )

    output_dir = tmp_path / "samples"
    sample_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--max-cases",
            "1",
        ]
    )

    assert (output_dir / "summary.txt").exists()
    assert any(path.suffix == ".png" for path in output_dir.iterdir())
