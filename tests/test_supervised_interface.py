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
from compliant_mechanism_synthesis.dataset.types import Structures
from compliant_mechanism_synthesis.evaluation import evaluate_refinement_step
from compliant_mechanism_synthesis.training import (
    CurriculumConfig,
    analyze_structures,
    iter_supervised_minibatches,
    load_supervised_cases,
    make_supervised_batch,
    select_batch,
)


def _build_cases(tmp_path: Path):
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=3,
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )
    return path, load_supervised_cases(str(path))


def test_make_supervised_batch_returns_noisy_structures(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    batch = make_supervised_batch(
        optimized_cases=optimized_cases,
        curriculum=CurriculumConfig(),
        difficulty=0.6,
        seed=7,
    )

    assert batch.noisy_structures.positions.shape == optimized_cases.raw_structures.positions.shape
    assert batch.target_stiffness.shape == optimized_cases.target_stiffness.shape
    assert not torch.allclose(batch.noisy_structures.positions, batch.oracle_structures.positions)


def test_iter_supervised_minibatches_covers_all_cases(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    minibatches = list(iter_supervised_minibatches(optimized_cases, batch_size=2, shuffle=False))

    assert len(minibatches) == 2
    assert sum(batch.raw_structures.batch_size for batch in minibatches) == 3


def test_evaluate_refinement_step_compares_noisy_refined_and_oracle(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)

    def oracle_refiner(noisy_structures: Structures, _target_stiffness: torch.Tensor) -> Structures:
        return select_batch(optimized_cases, torch.arange(optimized_cases.raw_structures.batch_size)).optimized_structures

    metrics = evaluate_refinement_step(
        refiner=oracle_refiner,
        optimized_cases=optimized_cases,
        difficulty=0.5,
        seed=3,
    )

    relative_gap = abs(metrics.refined_target_error - metrics.oracle_target_error) / max(
        metrics.oracle_target_error,
        1e-6,
    )
    assert relative_gap < 0.05
    assert metrics.noisy_target_error >= 0.0


def test_analyze_structures_matches_dataset_batch_shape(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    analyses = analyze_structures(optimized_cases.optimized_structures)

    assert analyses.generalized_stiffness.shape == optimized_cases.target_stiffness.shape
    assert analyses.material_usage.shape == optimized_cases.initial_loss.shape
