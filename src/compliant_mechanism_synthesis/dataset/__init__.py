"""Offline dataset generation and case optimization."""

from compliant_mechanism_synthesis.dataset.offline import (
    OfflineDatasetConfig,
    generate_offline_dataset,
    load_offline_dataset,
    save_offline_dataset,
)
from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    OptimizationLossWeights,
    optimize_cases,
    optimize_scaffolds,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    CHAIN_PRIMITIVE_LIBRARY,
    PrimitiveConfig,
    materialize_scaffold,
    sample_primitive_design,
    sample_random_primitive,
)
from compliant_mechanism_synthesis.dataset.types import Analyses, OptimizedCases, Scaffolds, Structures

__all__ = [
    "Analyses",
    "CaseOptimizationConfig",
    "CHAIN_PRIMITIVE_LIBRARY",
    "OfflineDatasetConfig",
    "OptimizationLossWeights",
    "OptimizedCases",
    "PrimitiveConfig",
    "Scaffolds",
    "Structures",
    "generate_offline_dataset",
    "load_offline_dataset",
    "materialize_scaffold",
    "optimize_cases",
    "optimize_scaffolds",
    "save_offline_dataset",
    "sample_primitive_design",
    "sample_random_primitive",
]
