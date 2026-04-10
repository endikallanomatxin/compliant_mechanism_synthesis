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
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Structures,
)
from compliant_mechanism_synthesis.evaluation import evaluate_refinement_step
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.training import (
    analyze_structures,
    iter_supervised_batches,
    load_supervised_cases,
    make_supervised_batch,
    sample_noisy_structures,
    select_batch,
)


def _build_cases(tmp_path: Path):
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=3,
            device="cpu",
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
        seed=7,
    )

    assert (
        batch.noisy_structures.positions.shape
        == optimized_cases.optimized_structures.positions.shape
    )
    assert batch.target_stiffness.shape == optimized_cases.target_stiffness.shape
    assert batch.current_analyses.nodal_displacements is not None
    assert batch.current_analyses.nodal_displacements.shape[:2] == (
        optimized_cases.optimized_structures.batch_size,
        optimized_cases.optimized_structures.positions.shape[1],
    )
    assert batch.current_analyses.edge_von_mises is not None
    assert batch.current_analyses.edge_von_mises.shape[:3] == (
        optimized_cases.optimized_structures.batch_size,
        optimized_cases.optimized_structures.positions.shape[1],
        optimized_cases.optimized_structures.positions.shape[1],
    )
    assert not torch.allclose(
        batch.noisy_structures.positions, batch.oracle_structures.positions
    )
    assert torch.allclose(
        batch.target_position_velocity,
        batch.oracle_structures.positions - batch.source_structures.positions,
    )
    assert torch.allclose(
        batch.target_adjacency_velocity,
        batch.oracle_structures.adjacency - batch.source_structures.adjacency,
    )


def test_sample_noisy_structures_is_seeded_gaussian_from_dataset_stats(
    tmp_path: Path,
) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    structures_a = sample_noisy_structures(
        optimized_cases=optimized_cases,
        seed=7,
    )
    structures_b = sample_noisy_structures(
        optimized_cases=optimized_cases,
        seed=7,
    )
    structures_c = sample_noisy_structures(
        optimized_cases=optimized_cases,
        seed=8,
    )

    assert torch.allclose(structures_a.positions, structures_b.positions)
    assert torch.allclose(structures_a.adjacency, structures_b.adjacency)
    assert not torch.allclose(structures_a.positions, structures_c.positions)
    assert not torch.allclose(
        structures_a.positions, optimized_cases.optimized_structures.positions
    )


def test_make_supervised_batch_can_use_global_noise_statistics(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    _, _, free_mask = role_masks(optimized_cases.optimized_structures.roles)
    free_positions = optimized_cases.optimized_structures.positions[free_mask]
    position_mean = free_positions.mean(dim=0, keepdim=True).view(1, 1, 3)
    position_std = (
        free_positions.std(dim=0, unbiased=False).clamp_min(1e-3).view(1, 1, 3)
    )
    fixed_mask, mobile_mask, free_mask = role_masks(
        optimized_cases.optimized_structures.roles
    )
    diagonal = torch.eye(
        optimized_cases.optimized_structures.num_nodes, dtype=torch.bool
    ).unsqueeze(0)
    free_free = (free_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)) & ~diagonal
    free_fixed = (free_mask.unsqueeze(-1) & fixed_mask.unsqueeze(-2)) | (
        fixed_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    free_mobile = (free_mask.unsqueeze(-1) & mobile_mask.unsqueeze(-2)) | (
        mobile_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    adjacency_values = optimized_cases.optimized_structures.adjacency
    adjacency_mean = torch.stack(
        [
            adjacency_values[free_free].mean(),
            adjacency_values[free_fixed].mean(),
            adjacency_values[free_mobile].mean(),
        ]
    )
    adjacency_std = torch.stack(
        [
            adjacency_values[free_free].std(unbiased=False).clamp_min(1e-3),
            adjacency_values[free_fixed].std(unbiased=False).clamp_min(1e-3),
            adjacency_values[free_mobile].std(unbiased=False).clamp_min(1e-3),
        ]
    )

    batch = make_supervised_batch(
        optimized_cases=optimized_cases,
        position_mean=position_mean,
        position_std=position_std,
        adjacency_mean=adjacency_mean,
        adjacency_std=adjacency_std,
        seed=9,
    )

    assert batch.noisy_structures.positions.shape == (
        optimized_cases.optimized_structures.batch_size,
        optimized_cases.optimized_structures.num_nodes,
        3,
    )


def test_make_supervised_batch_matches_permuted_free_oracle_nodes(
    tmp_path: Path,
) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    _, _, free_mask = role_masks(optimized_cases.optimized_structures.roles)
    free_indices = torch.nonzero(free_mask[0], as_tuple=False).squeeze(-1)
    reversed_free = torch.flip(free_indices, dims=[0])
    permutation = torch.arange(
        optimized_cases.optimized_structures.num_nodes, dtype=torch.long
    )
    permutation[free_indices] = reversed_free

    permuted_cases = OptimizedCases(
        target_stiffness=optimized_cases.target_stiffness,
        optimized_structures=Structures(
            positions=optimized_cases.optimized_structures.positions.index_select(
                1, permutation
            ),
            roles=optimized_cases.optimized_structures.roles,
            adjacency=optimized_cases.optimized_structures.adjacency.index_select(
                1, permutation
            ).index_select(2, permutation),
        ),
        initial_loss=optimized_cases.initial_loss,
        best_loss=optimized_cases.best_loss,
        last_analyses=Analyses(
            generalized_stiffness=optimized_cases.last_analyses.generalized_stiffness,
            material_usage=optimized_cases.last_analyses.material_usage,
            short_beam_penalty=optimized_cases.last_analyses.short_beam_penalty,
            long_beam_penalty=optimized_cases.last_analyses.long_beam_penalty,
            thin_beam_penalty=optimized_cases.last_analyses.thin_beam_penalty,
            thick_beam_penalty=optimized_cases.last_analyses.thick_beam_penalty,
            free_node_spacing_penalty=optimized_cases.last_analyses.free_node_spacing_penalty,
            nodal_displacements=optimized_cases.last_analyses.nodal_displacements.index_select(
                1, permutation
            ),
            edge_von_mises=optimized_cases.last_analyses.edge_von_mises.index_select(
                1, permutation
            ).index_select(2, permutation),
        ),
        scaffolds=optimized_cases.scaffolds,
    )
    all_free_positions = optimized_cases.optimized_structures.positions[free_mask]
    position_mean = all_free_positions.mean(dim=0, keepdim=True).view(1, 1, 3)
    position_std = (
        all_free_positions.std(dim=0, unbiased=False).clamp_min(1e-3).view(1, 1, 3)
    )
    fixed_mask, mobile_mask, free_mask = role_masks(
        optimized_cases.optimized_structures.roles
    )
    diagonal = torch.eye(
        optimized_cases.optimized_structures.num_nodes, dtype=torch.bool
    ).unsqueeze(0)
    free_free = (free_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)) & ~diagonal
    free_fixed = (free_mask.unsqueeze(-1) & fixed_mask.unsqueeze(-2)) | (
        fixed_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    free_mobile = (free_mask.unsqueeze(-1) & mobile_mask.unsqueeze(-2)) | (
        mobile_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    adjacency_values = optimized_cases.optimized_structures.adjacency
    adjacency_mean = torch.stack(
        [
            adjacency_values[free_free].mean(),
            adjacency_values[free_fixed].mean(),
            adjacency_values[free_mobile].mean(),
        ]
    )
    adjacency_std = torch.stack(
        [
            adjacency_values[free_free].std(unbiased=False).clamp_min(1e-3),
            adjacency_values[free_fixed].std(unbiased=False).clamp_min(1e-3),
            adjacency_values[free_mobile].std(unbiased=False).clamp_min(1e-3),
        ]
    )

    reference_batch = make_supervised_batch(
        optimized_cases=optimized_cases,
        position_mean=position_mean,
        position_std=position_std,
        adjacency_mean=adjacency_mean,
        adjacency_std=adjacency_std,
        seed=12,
    )
    permuted_batch = make_supervised_batch(
        optimized_cases=permuted_cases,
        position_mean=position_mean,
        position_std=position_std,
        adjacency_mean=adjacency_mean,
        adjacency_std=adjacency_std,
        seed=12,
    )

    assert torch.allclose(
        reference_batch.source_structures.positions,
        permuted_batch.source_structures.positions,
    )
    assert torch.allclose(
        reference_batch.oracle_structures.positions,
        permuted_batch.oracle_structures.positions,
    )
    assert torch.allclose(
        reference_batch.target_position_velocity,
        permuted_batch.target_position_velocity,
    )
    assert torch.allclose(
        reference_batch.target_adjacency_velocity,
        permuted_batch.target_adjacency_velocity,
    )


def test_iter_supervised_batches_covers_all_cases(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    batches = list(
        iter_supervised_batches(optimized_cases, batch_size=2, shuffle=False)
    )

    assert len(batches) == 2
    assert sum(batch.optimized_structures.batch_size for batch in batches) == 3


def test_evaluate_refinement_step_compares_noisy_refined_and_oracle(
    tmp_path: Path,
) -> None:
    _, optimized_cases = _build_cases(tmp_path)

    def oracle_refiner(
        noisy_structures: Structures, _target_stiffness: torch.Tensor
    ) -> Structures:
        return select_batch(
            optimized_cases,
            torch.arange(optimized_cases.optimized_structures.batch_size),
        ).optimized_structures

    metrics = evaluate_refinement_step(
        refiner=oracle_refiner,
        optimized_cases=optimized_cases,
        seed=3,
    )

    assert abs(metrics.refined_target_error - metrics.oracle_target_error) < 1e-3
    assert metrics.noisy_target_error >= 0.0


def test_analyze_structures_matches_dataset_batch_shape(tmp_path: Path) -> None:
    _, optimized_cases = _build_cases(tmp_path)
    analyses = analyze_structures(optimized_cases.optimized_structures)

    assert (
        analyses.generalized_stiffness.shape == optimized_cases.target_stiffness.shape
    )
    assert analyses.material_usage.shape == optimized_cases.initial_loss.shape
