from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    OfflineDatasetConfig,
    PrimitiveConfig,
    CaseOptimizationConfig,
    generate_offline_dataset,
    load_offline_dataset,
)


def test_offline_dataset_roundtrip_preserves_shapes(tmp_path: Path) -> None:
    path = tmp_path / "dataset.pt"
    generated = generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=2,
            device="cpu",
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )

    loaded_cases, _ = load_offline_dataset(path)

    assert (
        loaded_cases.optimized_structures.adjacency.shape
        == generated.optimized_structures.adjacency.shape
    )
    assert (
        loaded_cases.last_analyses.generalized_stiffness.shape
        == generated.last_analyses.generalized_stiffness.shape
    )
    assert loaded_cases.last_analyses.nodal_displacements is not None
    assert generated.last_analyses.nodal_displacements is not None
    assert (
        loaded_cases.last_analyses.nodal_displacements.shape
        == generated.last_analyses.nodal_displacements.shape
    )
    assert loaded_cases.last_analyses.edge_von_mises is not None
    assert generated.last_analyses.edge_von_mises is not None
    assert (
        loaded_cases.last_analyses.edge_von_mises.shape
        == generated.last_analyses.edge_von_mises.shape
    )
    assert loaded_cases.scaffolds is not None
    assert generated.scaffolds is not None
    assert (
        loaded_cases.scaffolds.edge_primitive_types.shape
        == generated.scaffolds.edge_primitive_types.shape
    )
    assert loaded_cases.optimized_structures.edge_radius is not None
    assert generated.optimized_structures.edge_radius is not None
    assert torch.allclose(
        loaded_cases.optimized_structures.edge_radius,
        generated.optimized_structures.edge_radius,
    )


def test_load_offline_dataset_migrates_legacy_adjacency_to_presence_and_radius(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=2,
            device="cpu",
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )

    legacy_payload = torch.load(path, map_location="cpu")
    original_adjacency = legacy_payload["optimized_structures"]["adjacency"].clone()
    legacy_payload["optimized_structures"].pop("edge_radius", None)
    legacy_payload["scaffolds"].pop("edge_radius", None)
    legacy_path = tmp_path / "legacy_dataset.pt"
    torch.save(legacy_payload, legacy_path)

    loaded_cases, _ = load_offline_dataset(legacy_path)

    assert loaded_cases.optimized_structures.edge_radius is not None
    assert torch.equal(
        loaded_cases.optimized_structures.adjacency,
        (original_adjacency.clamp(0.0, 1.0) > 0.15).to(dtype=original_adjacency.dtype),
    )
    assert torch.allclose(
        loaded_cases.optimized_structures.edge_radius,
        original_adjacency.clamp(0.0, 1.0),
    )
