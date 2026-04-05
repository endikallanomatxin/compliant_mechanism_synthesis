"""Offline dataset generation and case optimization."""

from compliant_mechanism_synthesis.dataset.offline import OfflineDatasetConfig, generate_offline_dataset
from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    CaseOptimizationResult,
    OptimizationLossWeights,
    optimize_case,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    PRIMITIVE_LIBRARY,
    PrimitiveConfig,
    sample_primitive_design,
    sample_random_primitive,
)

__all__ = [
    "CaseOptimizationConfig",
    "CaseOptimizationResult",
    "OfflineDatasetConfig",
    "OptimizationLossWeights",
    "PRIMITIVE_LIBRARY",
    "PrimitiveConfig",
    "generate_offline_dataset",
    "optimize_case",
    "sample_primitive_design",
    "sample_random_primitive",
    "sample_target_stiffness",
]
