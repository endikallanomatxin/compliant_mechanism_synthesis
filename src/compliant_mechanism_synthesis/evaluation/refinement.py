from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from compliant_mechanism_synthesis.dataset.types import OptimizedCases, Structures
from compliant_mechanism_synthesis.models import SupervisedRefiner
from compliant_mechanism_synthesis.training.supervised import (
    CurriculumConfig,
    analyze_structures,
    generalized_stiffness_error,
    make_supervised_batch,
)


@dataclass(frozen=True)
class RefinementMetrics:
    noisy_target_error: float
    refined_target_error: float
    oracle_target_error: float


def evaluate_refinement_step(
    refiner: SupervisedRefiner | Callable[[Structures, object], Structures],
    optimized_cases: OptimizedCases,
    curriculum: CurriculumConfig | None = None,
    difficulty: float = 1.0,
    seed: int | None = None,
) -> RefinementMetrics:
    curriculum = curriculum or CurriculumConfig()
    batch = make_supervised_batch(
        optimized_cases=optimized_cases,
        curriculum=curriculum,
        difficulty=difficulty,
        seed=seed,
    )
    if isinstance(refiner, SupervisedRefiner):
        refined_structures = refiner(
            batch.noisy_structures,
            batch.target_stiffness,
            analysis_fn=analyze_structures,
        )
    else:
        refined_structures = refiner(batch.noisy_structures, batch.target_stiffness)
    if not isinstance(refined_structures, Structures):
        raise TypeError("refiner must return Structures")
    refined_structures.validate()

    noisy_analyses = analyze_structures(batch.noisy_structures)
    refined_analyses = analyze_structures(refined_structures)
    oracle_analyses = analyze_structures(batch.oracle_structures)

    noisy_target_error = generalized_stiffness_error(
        noisy_analyses.generalized_stiffness,
        batch.target_stiffness,
    ).mean()
    refined_target_error = generalized_stiffness_error(
        refined_analyses.generalized_stiffness,
        batch.target_stiffness,
    ).mean()
    oracle_target_error = generalized_stiffness_error(
        oracle_analyses.generalized_stiffness,
        batch.target_stiffness,
    ).mean()
    return RefinementMetrics(
        noisy_target_error=float(noisy_target_error.detach().item()),
        refined_target_error=float(refined_target_error.detach().item()),
        oracle_target_error=float(oracle_target_error.detach().item()),
    )
