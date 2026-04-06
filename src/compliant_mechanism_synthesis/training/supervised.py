from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from compliant_mechanism_synthesis.dataset import load_offline_dataset
from compliant_mechanism_synthesis.dataset.types import Analyses, OptimizedCases, Structures
from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    mechanical_terms,
    normalize_generalized_stiffness,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import symmetrize_matrix


@dataclass(frozen=True)
class CurriculumConfig:
    initial_mix: float = 0.15
    final_mix: float = 1.0
    position_noise: float = 0.01
    adjacency_noise: float = 0.04


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    dataset_path: str
    batch_size: int = 16
    num_steps: int = 10_000


@dataclass(frozen=True)
class SupervisedBatch:
    noisy_structures: Structures
    target_stiffness: torch.Tensor
    oracle_structures: Structures
    oracle_analyses: Analyses


def _difficulty_fraction(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return min(max(step / (total_steps - 1), 0.0), 1.0)


def load_supervised_cases(dataset_path: str) -> OptimizedCases:
    optimized_cases, _, _ = load_offline_dataset(dataset_path)
    return optimized_cases


def sample_noisy_structures(
    optimized_cases: OptimizedCases,
    curriculum: CurriculumConfig,
    difficulty: float,
    seed: int | None = None,
) -> Structures:
    optimized_cases.validate()
    difficulty = float(min(max(difficulty, 0.0), 1.0))
    mix = curriculum.initial_mix + difficulty * (curriculum.final_mix - curriculum.initial_mix)

    base_positions = optimized_cases.optimized_structures.positions + mix * (
        optimized_cases.raw_structures.positions - optimized_cases.optimized_structures.positions
    )
    base_adjacency = optimized_cases.optimized_structures.adjacency + mix * (
        optimized_cases.raw_structures.adjacency - optimized_cases.optimized_structures.adjacency
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=base_positions.device).manual_seed(seed)

    position_noise = curriculum.position_noise * difficulty * torch.randn(
        base_positions.shape,
        generator=generator,
        device=base_positions.device,
        dtype=base_positions.dtype,
    )
    noisy_positions = base_positions + position_noise

    adjacency_noise = curriculum.adjacency_noise * difficulty * torch.randn(
        base_adjacency.shape,
        generator=generator,
        device=base_adjacency.device,
        dtype=base_adjacency.dtype,
    )
    noisy_adjacency = symmetrize_matrix((base_adjacency + adjacency_noise).clamp(0.0, 1.0))

    _, _, free_mask = role_masks(optimized_cases.raw_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=noisy_positions.dtype)
    noisy_positions = (
        optimized_cases.raw_structures.positions * (1.0 - free_mask)
        + noisy_positions * free_mask
    )

    structures = Structures(
        positions=noisy_positions,
        roles=optimized_cases.raw_structures.roles,
        adjacency=noisy_adjacency,
    )
    structures.validate()
    return structures


def make_supervised_batch(
    optimized_cases: OptimizedCases,
    curriculum: CurriculumConfig,
    difficulty: float,
    seed: int | None = None,
) -> SupervisedBatch:
    noisy_structures = sample_noisy_structures(
        optimized_cases=optimized_cases,
        curriculum=curriculum,
        difficulty=difficulty,
        seed=seed,
    )
    return SupervisedBatch(
        noisy_structures=noisy_structures,
        target_stiffness=optimized_cases.target_stiffness,
        oracle_structures=optimized_cases.optimized_structures,
        oracle_analyses=optimized_cases.last_analyses,
    )


def select_batch(
    optimized_cases: OptimizedCases,
    batch_indices: torch.Tensor,
) -> OptimizedCases:
    if batch_indices.ndim != 1:
        raise ValueError("batch_indices must be one-dimensional")
    return OptimizedCases(
        raw_structures=Structures(
            positions=optimized_cases.raw_structures.positions.index_select(0, batch_indices),
            roles=optimized_cases.raw_structures.roles.index_select(0, batch_indices),
            adjacency=optimized_cases.raw_structures.adjacency.index_select(0, batch_indices),
        ),
        target_stiffness=optimized_cases.target_stiffness.index_select(0, batch_indices),
        optimized_structures=Structures(
            positions=optimized_cases.optimized_structures.positions.index_select(0, batch_indices),
            roles=optimized_cases.optimized_structures.roles.index_select(0, batch_indices),
            adjacency=optimized_cases.optimized_structures.adjacency.index_select(0, batch_indices),
        ),
        initial_loss=optimized_cases.initial_loss.index_select(0, batch_indices),
        best_loss=optimized_cases.best_loss.index_select(0, batch_indices),
        last_analyses=Analyses(
            generalized_stiffness=optimized_cases.last_analyses.generalized_stiffness.index_select(0, batch_indices),
            material_usage=optimized_cases.last_analyses.material_usage.index_select(0, batch_indices),
            short_beam_penalty=optimized_cases.last_analyses.short_beam_penalty.index_select(0, batch_indices),
            long_beam_penalty=optimized_cases.last_analyses.long_beam_penalty.index_select(0, batch_indices),
            thin_beam_penalty=optimized_cases.last_analyses.thin_beam_penalty.index_select(0, batch_indices),
            thick_beam_penalty=optimized_cases.last_analyses.thick_beam_penalty.index_select(0, batch_indices),
            free_node_spacing_penalty=optimized_cases.last_analyses.free_node_spacing_penalty.index_select(0, batch_indices),
        ),
    )


def iter_supervised_minibatches(
    optimized_cases: OptimizedCases,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    optimized_cases.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(optimized_cases.raw_structures.batch_size))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(indices[start : start + batch_size], dtype=torch.long)
        yield select_batch(optimized_cases, batch_indices)


def generalized_stiffness_error(
    generalized_stiffness: torch.Tensor,
    target_stiffness: torch.Tensor,
) -> torch.Tensor:
    normalized_current = normalize_generalized_stiffness(generalized_stiffness)
    normalized_target = normalize_generalized_stiffness(target_stiffness)
    return (normalized_current - normalized_target).square().mean(dim=(1, 2))


def analyze_structures(
    structures: Structures,
    frame_config: Frame3DConfig | None = None,
    geometry_config: GeometryPenaltyConfig | None = None,
) -> Analyses:
    frame_config = frame_config or Frame3DConfig()
    geometry_config = geometry_config or GeometryPenaltyConfig()
    structures.validate()
    terms = mechanical_terms(
        positions=structures.positions,
        roles=structures.roles,
        adjacency=structures.adjacency,
        frame_config=frame_config,
        penalty_config=geometry_config,
    )
    return Analyses(
        generalized_stiffness=terms["generalized_stiffness"],
        material_usage=terms["material_usage"],
        short_beam_penalty=terms["short_beam_penalty"],
        long_beam_penalty=terms["long_beam_penalty"],
        thin_beam_penalty=terms["thin_beam_penalty"],
        thick_beam_penalty=terms["thick_beam_penalty"],
        free_node_spacing_penalty=terms["free_node_spacing_penalty"],
    )


def run_supervised_training(config: SupervisedTrainingConfig) -> None:
    optimized_cases = load_supervised_cases(config.dataset_path)
    optimized_cases.validate()
    raise NotImplementedError(
        "Supervised training is still intentionally deferred. The dataset loader, "
        "curriculum noise sampler, minibatch iterator, and refinement evaluation "
        "interfaces are now in place for stage 2."
    )
