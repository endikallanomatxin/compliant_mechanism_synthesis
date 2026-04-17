"""Reusable loss helpers shared across training and optimization flows."""

from compliant_mechanism_synthesis.losses.stiffness import (
    generalized_stiffness_error,
    log_generalized_stiffness_error,
    psd_penalty,
    stiffness_interest_loss,
    stiffness_step_loss,
)
from compliant_mechanism_synthesis.losses.structural import (
    StructuralObjectiveWeights,
    stress_violation_terms,
    structural_objective_terms,
)

__all__ = [
    "StructuralObjectiveWeights",
    "generalized_stiffness_error",
    "log_generalized_stiffness_error",
    "psd_penalty",
    "stiffness_interest_loss",
    "stiffness_step_loss",
    "stress_violation_terms",
    "structural_objective_terms",
]
