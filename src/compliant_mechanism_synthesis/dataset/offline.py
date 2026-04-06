from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random

import torch

from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    optimize_cases,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    PRIMITIVE_LIBRARY,
    PrimitiveConfig,
    sample_primitive_design,
)
from compliant_mechanism_synthesis.dataset.types import OptimizedCases, Structures
from compliant_mechanism_synthesis.visualization import write_dataset_visualizations


@dataclass(frozen=True)
class OfflineDatasetConfig:
    num_cases: int = 32
    seed: int = 7
    output_path: str = "artifacts/offline_dataset.pt"
    logdir: str = "runs/offline_dataset"
    preview_dir: str | None = None
    preview_cases: int = 6
    primitive: PrimitiveConfig = PrimitiveConfig()
    optimization: CaseOptimizationConfig = field(default_factory=CaseOptimizationConfig)


def generate_offline_dataset(config: OfflineDatasetConfig | None = None) -> dict[str, object]:
    config = config or OfflineDatasetConfig()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.logdir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    primitive_kinds: list[str] = []
    structure_batches: list[Structures] = []
    target_batches = []
    for case_index in range(config.num_cases):
        primitive_kind = PRIMITIVE_LIBRARY[case_index % len(PRIMITIVE_LIBRARY)]
        primitive_seed = rng.randrange(0, 2**31)
        target_seed = rng.randrange(0, 2**31)
        initial_structures = sample_primitive_design(
            primitive_kind,
            config=config.primitive,
            seed=primitive_seed,
        )
        target = sample_target_stiffness(
            initial_structures,
            config=config.optimization,
            seed=target_seed,
        )
        primitive_kinds.append(primitive_kind)
        structure_batches.append(initial_structures)
        target_batches.append(target.unsqueeze(0))

    raw_structures = Structures(
        positions=torch.cat([item.positions for item in structure_batches], dim=0),
        roles=torch.cat([item.roles for item in structure_batches], dim=0),
        adjacency=torch.cat([item.adjacency for item in structure_batches], dim=0),
    )
    optimized_cases = optimize_cases(
        structures=raw_structures,
        target_stiffness=torch.cat(target_batches, dim=0),
        config=config.optimization,
        logdir=Path(config.logdir),
    )
    payload = _serialize_optimized_cases(optimized_cases, primitive_kinds, config)
    torch.save(payload, output_path)
    preview_dir = (
        Path(config.preview_dir)
        if config.preview_dir is not None
        else output_path.parent / f"{output_path.stem}_preview"
    )
    write_dataset_visualizations(payload, preview_dir, max_cases=config.preview_cases)
    return payload


def _serialize_optimized_cases(
    optimized_cases: OptimizedCases,
    primitive_kinds: list[str],
    config: OfflineDatasetConfig,
) -> dict[str, object]:
    optimized_cases.validate()
    return {
        "primitive_kinds": list(primitive_kinds),
        "raw_structures": {
            "positions": optimized_cases.raw_structures.positions,
            "roles": optimized_cases.raw_structures.roles,
            "adjacency": optimized_cases.raw_structures.adjacency,
        },
        "target_stiffness": optimized_cases.target_stiffness,
        "optimized_structures": {
            "positions": optimized_cases.optimized_structures.positions,
            "roles": optimized_cases.optimized_structures.roles,
            "adjacency": optimized_cases.optimized_structures.adjacency,
        },
        "initial_loss": optimized_cases.initial_loss,
        "best_loss": optimized_cases.best_loss,
        "last_analyses": {
            "generalized_stiffness": optimized_cases.last_analyses.generalized_stiffness,
            "material_usage": optimized_cases.last_analyses.material_usage,
            "short_beam_penalty": optimized_cases.last_analyses.short_beam_penalty,
            "long_beam_penalty": optimized_cases.last_analyses.long_beam_penalty,
            "thin_beam_penalty": optimized_cases.last_analyses.thin_beam_penalty,
            "thick_beam_penalty": optimized_cases.last_analyses.thick_beam_penalty,
            "free_node_spacing_penalty": optimized_cases.last_analyses.free_node_spacing_penalty,
        },
        "config": asdict(config),
    }
