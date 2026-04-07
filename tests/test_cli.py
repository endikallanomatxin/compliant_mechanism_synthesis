from __future__ import annotations

from pathlib import Path

from compliant_mechanism_synthesis.cli import (
    dataset_generate_main,
    train_supervised_main,
    visualize_dataset_main,
)
from compliant_mechanism_synthesis.dataset import load_offline_dataset
import pytest


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
    assert "dataset batch 1/2" in captured.out
    assert "dataset batch 2/2" in captured.out


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
