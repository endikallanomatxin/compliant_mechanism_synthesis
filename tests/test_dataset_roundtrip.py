from __future__ import annotations

from pathlib import Path

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
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )

    loaded_cases, _ = load_offline_dataset(path)

    assert loaded_cases.raw_structures.positions.shape == generated.raw_structures.positions.shape
    assert loaded_cases.optimized_structures.adjacency.shape == generated.optimized_structures.adjacency.shape
    assert loaded_cases.last_analyses.generalized_stiffness.shape == generated.last_analyses.generalized_stiffness.shape
    assert loaded_cases.scaffolds is not None
    assert generated.scaffolds is not None
    assert loaded_cases.scaffolds.edge_primitive_types.shape == generated.scaffolds.edge_primitive_types.shape
