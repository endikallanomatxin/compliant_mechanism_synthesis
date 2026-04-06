from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
    load_offline_dataset,
    optimize_cases,
    sample_primitive_design,
    sample_target_stiffness,
)


def test_sample_primitive_design_is_valid_in_3d() -> None:
    structures = sample_primitive_design(
        "curved_lattice_sheet",
        config=PrimitiveConfig(num_free_nodes=8),
        seed=7,
    )
    structures.validate()
    assert structures.positions.shape == (1, 14, 3)
    assert torch.count_nonzero(structures.adjacency) > 0


def test_case_optimizer_improves_best_loss_against_initial_loss(tmp_path: Path) -> None:
    initial_structures = sample_primitive_design(
        "straight_beam",
        config=PrimitiveConfig(num_free_nodes=6),
        seed=3,
    )
    optimization = CaseOptimizationConfig(num_steps=6)
    target = sample_target_stiffness(initial_structures, config=optimization, seed=11)
    result = optimize_cases(
        structures=initial_structures,
        target_stiffness=target.unsqueeze(0),
        config=optimization,
        logdir=tmp_path / "tb",
    )

    assert result.best_loss[0] <= result.initial_loss[0]


def test_generate_offline_dataset_persists_payload(tmp_path: Path) -> None:
    optimized_cases = generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=2,
            output_path=str(tmp_path / "dataset.pt"),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )

    loaded_cases, primitive_kinds, loaded_config = load_offline_dataset(tmp_path / "dataset.pt")
    assert optimized_cases.optimized_structures.adjacency.shape == (2, 12, 12)
    assert loaded_cases.last_analyses.generalized_stiffness.shape == (2, 6, 6)
    assert len(primitive_kinds) == 2
    assert loaded_config.num_cases == 2
    assert (tmp_path / "dataset.pt").exists()
