"""Differentiable structural mechanics backends."""

from compliant_mechanism_synthesis.mechanics.frame3d import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    assemble_global_stiffness,
    denormalize_generalized_stiffness,
    effective_output_stiffness,
    geometry_penalties,
    material_usage,
    mechanical_terms,
    normalize_generalized_stiffness,
)

__all__ = [
    "Frame3DConfig",
    "GeometryPenaltyConfig",
    "assemble_global_stiffness",
    "denormalize_generalized_stiffness",
    "effective_output_stiffness",
    "geometry_penalties",
    "material_usage",
    "mechanical_terms",
    "normalize_generalized_stiffness",
]
