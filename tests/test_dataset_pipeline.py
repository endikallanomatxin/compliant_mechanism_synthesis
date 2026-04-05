from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
    optimize_case,
    sample_primitive_design,
    sample_target_stiffness,
)


def test_sample_primitive_design_is_valid_in_3d() -> None:
    design = sample_primitive_design(
        "curved_lattice_sheet",
        config=PrimitiveConfig(num_free_nodes=8),
        seed=7,
    )
    design.validate()
    assert design.positions.shape == (14, 3)
    assert torch.count_nonzero(design.adjacency) > 0


def test_case_optimizer_improves_best_loss_against_initial_loss(tmp_path: Path) -> None:
    initial_design = sample_primitive_design(
        "straight_beam",
        config=PrimitiveConfig(num_free_nodes=6),
        seed=3,
    )
    optimization = CaseOptimizationConfig(num_steps=6)
    target = sample_target_stiffness(initial_design, config=optimization, seed=11)
    result = optimize_case(
        primitive_kind="straight_beam",
        initial_design=initial_design,
        target_stiffness=target,
        config=optimization,
        logdir=tmp_path / "tb",
    )

    assert result.best_loss <= result.initial_loss


def test_generate_offline_dataset_persists_payload(tmp_path: Path) -> None:
    payload = generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=2,
            output_path=str(tmp_path / "dataset.pt"),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )

    assert payload["optimized_adjacency"].shape == (2, 12, 12)
    assert (tmp_path / "dataset.pt").exists()
