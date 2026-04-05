from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.cli import sample_main, train_main


def test_train_main_generates_offline_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.pt"
    train_main(
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

    payload = torch.load(output_path)
    assert payload["optimized_positions"].shape == (2, 12, 3)


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
