from __future__ import annotations

from pathlib import Path

from compliant_mechanism_synthesis.cli import (
    dataset_generate_main,
    sample_supervised_main,
    train_rl_main,
    train_rl_optimizer_supported_main,
    train_supervised_main,
    upgrade_supervised_checkpoint_main,
    visualize_dataset_main,
)
from compliant_mechanism_synthesis.dataset import load_offline_dataset
from compliant_mechanism_synthesis.models import SupervisedRefiner, SupervisedRefinerConfig
import pytest
import torch


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
    assert optimized_cases.optimized_structures.positions.shape[-1] == 3
    assert (preview_dir / "case_0000_primitives.png").exists()
    assert (preview_dir / "summary.txt").exists()


def test_dataset_generate_main_samples_preview_subset_for_large_datasets(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset_large.pt"
    preview_dir = tmp_path / "preview_large"
    dataset_generate_main(
        [
            "--num-cases",
            "10",
            "--batch-size",
            "3",
            "--device",
            "cpu",
            "--num-free-nodes",
            "6",
            "--optimization-steps",
            "3",
            "--output-path",
            str(output_path),
            "--logdir",
            str(tmp_path / "runs_large"),
            "--preview-dir",
            str(preview_dir),
            "--preview-case-number",
            "8",
            "--seed",
            "11",
        ]
    )

    summary = (preview_dir / "summary.txt").read_text(encoding="utf-8")
    primitive_previews = list(preview_dir.glob("case_*_primitives.png"))
    assert len(primitive_previews) == 8
    assert "cases=10" in summary
    assert "preview_cases=8" in summary
    assert "preview_case_indices=" in summary


def test_dataset_generate_main_logs_batch_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "dataset_progress.pt"
    dataset_generate_main(
        [
            "--num-cases",
            "3",
            "--batch-size",
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
            str(tmp_path / "runs_progress"),
        ]
    )

    captured = capsys.readouterr()
    assert "dataset generation started" in captured.out
    assert "dataset batch 1 " in captured.out
    assert "dataset batch 2 " in captured.out
    assert "cases=2/3" in captured.out
    assert "cases=3/3" in captured.out


def test_dataset_generate_main_names_run(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
    runs_dir = tmp_path / "runs_named"
    create_logdir = runs_dir
    name = "fancygen"
    dataset_generate_main(
        [
            "--num-cases",
            "1",
            "--device",
            "cpu",
            "--num-free-nodes",
            "6",
            "--optimization-steps",
            "3",
            "--output-path",
            str(output_path),
            "--logdir",
            str(create_logdir),
            "--name",
            name,
        ]
    )
    entries = [entry for entry in create_logdir.iterdir() if entry.is_dir()]
    assert len(entries) == 1
    assert entries[0].name.endswith(f"-{name}")


def test_dataset_generate_can_run_sample_check(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    dataset_generate_main(
        [
            "--just-check-sample",
            "--device",
            "cpu",
            "--sample-num-free-nodes",
            "6",
            "--sample-optimization-steps",
            "3",
            "--sample-output-dir",
            str(sample_dir),
        ]
    )

    assert (sample_dir / "initial.png").exists()
    assert (sample_dir / "optimized.png").exists()


def test_visualize_dataset_main_renders_existing_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
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
        ]
    )

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


def test_train_supervised_main_writes_checkpoint(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
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
        ]
    )

    checkpoint_path = tmp_path / "refiner.pt"
    train_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "6",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_supervised"),
        ]
    )
    assert checkpoint_path.exists()


def test_train_supervised_main_writes_default_checkpoint_inside_run_dir(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    runs_dir = tmp_path / "runs_supervised"
    train_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "6",
            "--logdir",
            str(runs_dir),
            "--name",
            "testrun",
        ]
    )

    run_dirs = [entry for entry in runs_dir.iterdir() if entry.is_dir()]
    assert len(run_dirs) == 1
    assert run_dirs[0].name.endswith("-testrun")
    assert (run_dirs[0] / "refiner.pt").exists()


def test_train_supervised_main_can_disable_style_token(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    checkpoint_path = tmp_path / "refiner_no_style.pt"
    train_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "4",
            "--no-style-token",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_supervised_no_style"),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["model_config"]["use_style_token"] is False
    assert checkpoint["train_config"]["use_style_token"] is False


def test_train_supervised_main_uses_default_style_token_count(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    checkpoint_path = tmp_path / "refiner_style_count.pt"
    train_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_supervised_style_count"),
        ]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["model_config"]["style_token_count"] == 1
    assert checkpoint["train_config"]["style_token_count"] == 1


def test_train_rl_main_writes_checkpoint_and_supports_warm_start(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    supervised_checkpoint_path = tmp_path / "refiner_supervised.pt"
    train_supervised_main(
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
            str(supervised_checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_supervised"),
        ]
    )

    rl_checkpoint_path = tmp_path / "refiner_rl.pt"
    train_rl_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "2",
            "--num-steps",
            "2",
            "--rollout-steps",
            "2",
            "--init-checkpoint-path",
            str(supervised_checkpoint_path),
            "--checkpoint-path",
            str(rl_checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_rl"),
        ]
    )

    checkpoint = torch.load(rl_checkpoint_path, map_location="cpu")
    assert rl_checkpoint_path.exists()
    assert checkpoint["train_config"]["gradient_accumulation_steps"] == 2
    assert checkpoint["train_config"]["init_checkpoint_path"] == str(
        supervised_checkpoint_path
    )


def test_train_rl_optimizer_supported_main_writes_checkpoint_and_supports_warm_start(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    supervised_checkpoint_path = tmp_path / "refiner_supervised.pt"
    train_supervised_main(
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
            str(supervised_checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_supervised"),
        ]
    )

    hybrid_checkpoint_path = tmp_path / "refiner_explore_optimize.pt"
    train_rl_optimizer_supported_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--num-steps",
            "2",
            "--explore-steps",
            "2",
            "--optimize-steps",
            "2",
            "--init-checkpoint-path",
            str(supervised_checkpoint_path),
            "--checkpoint-path",
            str(hybrid_checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_explore_optimize"),
        ]
    )

    checkpoint = torch.load(hybrid_checkpoint_path, map_location="cpu")
    assert hybrid_checkpoint_path.exists()
    assert checkpoint["train_config"]["init_checkpoint_path"] == str(
        supervised_checkpoint_path
    )


def test_upgrade_supervised_checkpoint_main_rewrites_old_checkpoint_format(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    checkpoint_path = tmp_path / "refiner.pt"
    train_supervised_main(
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
            str(tmp_path / "runs_supervised"),
        ]
    )

    old_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_checkpoint["model_state_dict"].pop("style_token_encoder.token_seed")
    old_checkpoint["model_config"].pop("style_token_count")
    old_checkpoint["train_config"].pop("style_token_count")
    torch.save(old_checkpoint, checkpoint_path)

    upgrade_supervised_checkpoint_main(["--checkpoint-path", str(checkpoint_path)])

    backup_path = tmp_path / "refiner-original.pt"
    upgraded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert backup_path.exists()
    assert upgraded_checkpoint["model_config"]["style_token_count"] == 1
    assert upgraded_checkpoint["train_config"]["style_token_count"] == 1

    model = SupervisedRefiner(SupervisedRefinerConfig(**upgraded_checkpoint["model_config"]))
    model.load_state_dict(upgraded_checkpoint["model_state_dict"])


def test_sample_supervised_main_writes_comparison_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
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
            str(tmp_path / "runs_dataset"),
        ]
    )

    checkpoint_path = tmp_path / "refiner.pt"
    train_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-steps",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--logdir",
            str(tmp_path / "runs_supervised"),
        ]
    )

    samples_dir = tmp_path / "samples"
    sample_supervised_main(
        [
            "--dataset-path",
            str(output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(samples_dir),
            "--device",
            "cpu",
            "--max-cases",
            "1",
            "--num-steps",
            "2",
        ]
    )

    summary = (samples_dir / "summary.txt").read_text(encoding="utf-8")
    assert (samples_dir / "summary.txt").exists()
    assert (samples_dir / "case_0000_comparison.png").exists()
    assert (samples_dir / "case_0000_rollout.gif").exists()
    assert "noisy_position_error=" in summary
    assert "generated_no_style_position_error=" in summary
