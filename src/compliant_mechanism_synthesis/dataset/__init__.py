"""Offline dataset generation and case optimization."""

from compliant_mechanism_synthesis.dataset.offline import OfflineDatasetConfig, generate_offline_dataset
from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    OptimizationLossWeights,
    optimize_cases,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    PRIMITIVE_LIBRARY,
    PrimitiveConfig,
    sample_primitive_design,
    sample_random_primitive,
)
from compliant_mechanism_synthesis.dataset.types import Analyses, OptimizedCases, Structures

__all__ = [
    "Analyses",
    "CaseOptimizationConfig",
    "OfflineDatasetConfig",
    "OptimizationLossWeights",
    "OptimizedCases",
    "PRIMITIVE_LIBRARY",
    "PrimitiveConfig",
    "Structures",
    "generate_offline_dataset",
    "optimize_cases",
    "sample_primitive_design",
    "sample_random_primitive",
    "sample_target_stiffness",
]
