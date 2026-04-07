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


def test_sample_random_primitive_uses_single_debug_chain_family() -> None:
    _, scaffolds = sample_random_primitive(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=13,
    )

    edge_types = scaffolds.edge_primitive_types[0]
    used_types = torch.unique(edge_types[edge_types >= 0])
    assert used_types.tolist() == [1]


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
    assert loaded_cases.scaffolds is not None
    assert loaded_cases.scaffolds.positions.shape == (3, 8, 3)
    assert loaded_config.num_cases == 3
    assert loaded_config.batch_size == 2
    assert (tmp_path / "dataset.pt").exists()
