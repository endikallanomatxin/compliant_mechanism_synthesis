from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    OptimizationLossWeights,
    PrimitiveConfig,
    generate_offline_dataset,
    load_offline_dataset,
    materialize_scaffold,
    optimize_cases,
    optimize_scaffolds,
    sample_primitive_design,
    sample_random_primitive,
)
from compliant_mechanism_synthesis.mechanics import normalize_generalized_stiffness
from compliant_mechanism_synthesis.roles import NodeRole


def _stiffness_interest_score(matrix: torch.Tensor) -> torch.Tensor:
    normalized = normalize_generalized_stiffness(matrix)
    eigenvalues = torch.linalg.eigvalsh(normalized)
    batch_mean = eigenvalues.mean(dim=0, keepdim=True)
    return (eigenvalues - batch_mean).square().mean(dim=-1)


def test_sample_primitive_design_is_valid_in_3d() -> None:
    structures = sample_primitive_design(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=7,
    )
    structures.validate()
    assert structures.positions.shape[0] == 1
    assert structures.positions.shape[-1] == 3
    assert structures.num_nodes >= 8
    assert torch.count_nonzero(structures.adjacency) > 0


def test_sample_primitive_design_places_fixed_anchors_below_mobile_anchors() -> None:
    structures = sample_primitive_design(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=7,
    )

    roles = structures.roles[0]
    fixed_z = structures.positions[0, roles == int(NodeRole.FIXED), 2]
    mobile_z = structures.positions[0, roles == int(NodeRole.MOBILE), 2]
    assert torch.max(fixed_z) < torch.min(mobile_z)


def test_sample_primitive_design_materializes_triplets_with_no_dangling_nodes() -> None:
    structures = sample_primitive_design(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=11,
    )

    degree = structures.adjacency[0].sum(dim=1)
    assert torch.all(degree >= 2.0)


def test_sample_random_primitive_assigns_multiple_segment_families() -> None:
    _, scaffolds = sample_random_primitive(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=13,
    )

    edge_types = scaffolds.edge_primitive_types[0]
    used_types = torch.unique(edge_types[edge_types >= 0])
    assert used_types.numel() >= 3


def test_sample_random_primitive_builds_intertwined_scaffold_graph() -> None:
    _, scaffolds = sample_random_primitive(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=13,
    )

    degree = scaffolds.adjacency[0].sum(dim=1)
    assert torch.count_nonzero(degree >= 3.0) >= 2
    assert torch.all(degree[1:-1] >= 2.0)


def test_materialize_scaffold_returns_valid_structure() -> None:
    structures, scaffold = sample_random_primitive(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=15,
    )

    rematerialized = materialize_scaffold(scaffold, config=PrimitiveConfig(num_free_nodes=8))

    rematerialized.validate()
    assert rematerialized.batch_size == structures.batch_size
    assert rematerialized.positions.shape == structures.positions.shape
    assert torch.allclose(rematerialized.positions, structures.positions)
    assert torch.equal(rematerialized.roles, structures.roles)
    assert torch.allclose(rematerialized.adjacency, structures.adjacency)


def test_optimize_scaffolds_returns_valid_batch() -> None:
    primitive_config = PrimitiveConfig(num_free_nodes=6, forced_primitive_type="rod")
    _, scaffold_a = sample_random_primitive(
        config=primitive_config,
        seed=27,
    )
    _, scaffold_b = sample_random_primitive(
        config=primitive_config,
        seed=33,
    )
    scaffolds = type(scaffold_a)(
        positions=torch.cat([scaffold_a.positions, scaffold_b.positions], dim=0),
        roles=torch.cat([scaffold_a.roles, scaffold_b.roles], dim=0),
        adjacency=torch.cat([scaffold_a.adjacency, scaffold_b.adjacency], dim=0),
        edge_primitive_types=torch.cat(
            [scaffold_a.edge_primitive_types, scaffold_b.edge_primitive_types], dim=0
        ),
        edge_sheet_width_nodes=torch.cat(
            [scaffold_a.edge_sheet_width_nodes, scaffold_b.edge_sheet_width_nodes],
            dim=0,
        ),
        edge_orientation_start=torch.cat(
            [scaffold_a.edge_orientation_start, scaffold_b.edge_orientation_start],
            dim=0,
        ),
        edge_orientation_end=torch.cat(
            [scaffold_a.edge_orientation_end, scaffold_b.edge_orientation_end],
            dim=0,
        ),
        edge_offset_start=torch.cat(
            [scaffold_a.edge_offset_start, scaffold_b.edge_offset_start], dim=0
        ),
        edge_offset_end=torch.cat(
            [scaffold_a.edge_offset_end, scaffold_b.edge_offset_end], dim=0
        ),
        edge_helix_phase=torch.cat(
            [scaffold_a.edge_helix_phase, scaffold_b.edge_helix_phase], dim=0
        ),
        edge_helix_pitch=torch.cat(
            [scaffold_a.edge_helix_pitch, scaffold_b.edge_helix_pitch], dim=0
        ),
        edge_width_start=torch.cat(
            [scaffold_a.edge_width_start, scaffold_b.edge_width_start], dim=0
        ),
        edge_width_end=torch.cat(
            [scaffold_a.edge_width_end, scaffold_b.edge_width_end], dim=0
        ),
        edge_thickness_start=torch.cat(
            [scaffold_a.edge_thickness_start, scaffold_b.edge_thickness_start], dim=0
        ),
        edge_thickness_end=torch.cat(
            [scaffold_a.edge_thickness_end, scaffold_b.edge_thickness_end], dim=0
        ),
        edge_twist_start=torch.cat(
            [scaffold_a.edge_twist_start, scaffold_b.edge_twist_start], dim=0
        ),
        edge_twist_end=torch.cat(
            [scaffold_a.edge_twist_end, scaffold_b.edge_twist_end], dim=0
        ),
        edge_sweep_phase=torch.cat(
            [scaffold_a.edge_sweep_phase, scaffold_b.edge_sweep_phase], dim=0
        ),
    )

    optimized_scaffolds, structures = optimize_scaffolds(
        scaffolds=scaffolds,
        primitive_config=primitive_config,
        config=CaseOptimizationConfig(scaffold_num_steps=2, num_steps=4),
    )

    optimized_scaffolds.validate()
    structures.validate()
    assert optimized_scaffolds.positions.shape == scaffolds.positions.shape
    assert structures.batch_size == scaffolds.batch_size


def test_sheet_width_nodes_increases_materialized_node_count() -> None:
    narrow = sample_primitive_design(
        config=PrimitiveConfig(
            num_free_nodes=6,
            sheet_width_nodes=2,
            forced_primitive_type="sheet_helix",
            sample_sheet_helix_width_nodes=False,
            sheet_helix_width_nodes_min=2,
            sheet_helix_width_nodes_max=2,
        ),
        seed=17,
    )
    wide = sample_primitive_design(
        config=PrimitiveConfig(
            num_free_nodes=6,
            sheet_width_nodes=4,
            forced_primitive_type="sheet_helix",
            sample_sheet_helix_width_nodes=False,
            sheet_helix_width_nodes_min=4,
            sheet_helix_width_nodes_max=4,
        ),
        seed=17,
    )

    assert wide.positions.shape[1] > narrow.positions.shape[1]


def test_sheet_helix_responds_to_pitch_and_offset_controls() -> None:
    relaxed = sample_primitive_design(
        config=PrimitiveConfig(
            num_free_nodes=6,
            forced_primitive_type="sheet_helix",
            sample_sheet_helix_width_nodes=False,
            sheet_helix_width_nodes_min=2,
            sheet_helix_width_nodes_max=2,
            sheet_helix_offset_distance_min=0.04,
            sheet_helix_offset_distance_max=0.04,
            sheet_helix_pitch_distance=0.40,
        ),
        seed=19,
    )
    tighter = sample_primitive_design(
        config=PrimitiveConfig(
            num_free_nodes=6,
            forced_primitive_type="sheet_helix",
            sample_sheet_helix_width_nodes=False,
            sheet_helix_width_nodes_min=2,
            sheet_helix_width_nodes_max=2,
            sheet_helix_offset_distance_min=0.20,
            sheet_helix_offset_distance_max=0.20,
            sheet_helix_pitch_distance=0.10,
        ),
        seed=19,
    )

    assert tighter.positions.shape[1] > relaxed.positions.shape[1]


def test_truss_materialization_creates_dense_skip_link_bracing() -> None:
    structures = sample_primitive_design(
        config=PrimitiveConfig(num_free_nodes=6),
        seed=23,
    )

    degree = structures.adjacency[0].sum(dim=1)
    assert torch.count_nonzero(degree >= 4.0) > 0


def test_case_optimizer_improves_best_loss_against_initial_loss(tmp_path: Path) -> None:
    initial_structures = sample_primitive_design(
        config=PrimitiveConfig(num_free_nodes=6),
        seed=3,
    )
    optimization = CaseOptimizationConfig(num_steps=6)
    result = optimize_cases(
        structures=initial_structures,
        config=optimization,
        logdir=tmp_path / "tb",
    )

    assert result.best_loss[0] <= result.initial_loss[0]


def test_case_optimizer_can_increase_batch_stiffness_diversity(
    tmp_path: Path,
) -> None:
    primitive_config = PrimitiveConfig(
        num_free_nodes=6,
        forced_primitive_type="rod",
    )
    structure_a = sample_primitive_design(
        config=primitive_config,
        seed=29,
    )
    structure_b = sample_primitive_design(
        config=primitive_config,
        seed=31,
    )
    initial_structures = type(structure_a)(
        positions=torch.cat([structure_a.positions, structure_b.positions], dim=0),
        roles=torch.cat([structure_a.roles, structure_b.roles], dim=0),
        adjacency=torch.cat([structure_a.adjacency, structure_b.adjacency], dim=0),
    )
    neutral = optimize_cases(
        structures=initial_structures,
        config=CaseOptimizationConfig(
            num_steps=12,
            weights=OptimizationLossWeights(stiffness_interest=0.0),
        ),
        logdir=tmp_path / "tb_neutral",
    )
    interested = optimize_cases(
        structures=initial_structures,
        config=CaseOptimizationConfig(
            num_steps=12,
            weights=OptimizationLossWeights(stiffness_interest=0.4),
        ),
        logdir=tmp_path / "tb_interested",
    )

    neutral_score = _stiffness_interest_score(neutral.last_analyses.generalized_stiffness)
    interested_score = _stiffness_interest_score(
        interested.last_analyses.generalized_stiffness
    )

    assert interested_score.mean() >= neutral_score.mean()


def test_generate_offline_dataset_persists_payload(tmp_path: Path) -> None:
    optimized_cases = generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=3,
            batch_size=2,
            device="cpu",
            output_path=str(tmp_path / "dataset.pt"),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )

    loaded_cases, loaded_config = load_offline_dataset(tmp_path / "dataset.pt")
    assert optimized_cases.optimized_structures.adjacency.shape[0] == 3
    assert (
        optimized_cases.optimized_structures.adjacency.shape[1]
        == optimized_cases.optimized_structures.num_nodes
    )
    assert loaded_cases.last_analyses.generalized_stiffness.shape == (3, 6, 6)
    assert loaded_cases.last_analyses.nodal_displacements is not None
    assert loaded_cases.last_analyses.nodal_displacements.shape == (
        3,
        loaded_cases.optimized_structures.num_nodes,
        18,
    )
    assert loaded_cases.last_analyses.edge_von_mises is not None
    assert loaded_cases.last_analyses.edge_von_mises.shape == (
        3,
        loaded_cases.optimized_structures.num_nodes,
        loaded_cases.optimized_structures.num_nodes,
        6,
    )
    assert loaded_cases.scaffolds is not None
    assert loaded_cases.scaffolds.positions.shape == (3, 8, 3)
    assert loaded_config.num_cases == 3
    assert loaded_config.batch_size == 2
    assert (tmp_path / "dataset.pt").exists()
