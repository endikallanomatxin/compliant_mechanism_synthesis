from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
    load_offline_dataset,
    optimize_cases,
    sample_primitive_design,
    sample_random_primitive,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.roles import NodeRole


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
    assert torch.count_nonzero(degree >= 3.0) >= 4


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
    target = sample_target_stiffness(initial_structures, config=optimization, seed=11)
    result = optimize_cases(
        structures=initial_structures,
        target_stiffness=target.unsqueeze(0),
        config=optimization,
        logdir=tmp_path / "tb",
    )

    assert result.best_loss[0] <= result.initial_loss[0]


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
    assert loaded_cases.scaffolds is not None
    assert loaded_cases.scaffolds.positions.shape == (3, 8, 3)
    assert loaded_config.num_cases == 3
    assert loaded_config.batch_size == 2
    assert (tmp_path / "dataset.pt").exists()
