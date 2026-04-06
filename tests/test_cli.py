from __future__ import annotations

from pathlib import Path

from compliant_mechanism_synthesis.cli import (
    generate_dataset_main,
    sample_main,
    visualize_dataset_main,
)
from compliant_mechanism_synthesis.dataset import load_offline_dataset


def test_generate_dataset_main_generates_offline_dataset_and_preview(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
    preview_dir = tmp_path / "preview"
    generate_dataset_main(
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
            "--preview-dir",
            str(preview_dir),
            "--preview-cases",
            "2",
        ]
    )

    optimized_cases, primitive_kinds, _ = load_offline_dataset(output_path)
    assert optimized_cases.optimized_structures.positions.shape == (2, 12, 3)
    assert len(primitive_kinds) == 2
    assert (preview_dir / "summary.txt").exists()


def test_sample_main_writes_images(tmp_path: Path) -> None:
    sample_main(
        [
            "--primitive",
            "straight_beam",
            "--num-free-nodes",
            "6",
            "--optimization-steps",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert (tmp_path / "initial.png").exists()
    assert (tmp_path / "optimized.png").exists()


def test_visualize_dataset_main_renders_existing_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
    generate_dataset_main(
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
