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
    PrimitiveConfig,
    sample_random_primitive,
)
from compliant_mechanism_synthesis.dataset.types import Analyses, OptimizedCases, Scaffolds, Structures
from compliant_mechanism_synthesis.visualization import write_dataset_visualizations


@dataclass(frozen=True)
class OfflineDatasetConfig:
    num_cases: int = 32
    seed: int = 7
    output_path: str = "datasets/offline_dataset.pt"
    logdir: str = "runs/offline_dataset"
    preview_dir: str | None = None
    preview_cases: int = 6
    primitive: PrimitiveConfig = PrimitiveConfig()
    optimization: CaseOptimizationConfig = field(default_factory=CaseOptimizationConfig)


def generate_offline_dataset(config: OfflineDatasetConfig | None = None) -> OptimizedCases:
    config = config or OfflineDatasetConfig()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.logdir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    structure_batches: list[Structures] = []
    scaffold_batches: list[Scaffolds] = []
    target_batches = []
    for _case_index in range(config.num_cases):
        primitive_seed = rng.randrange(0, 2**31)
        target_seed = rng.randrange(0, 2**31)
        initial_structures, scaffold = sample_random_primitive(
            config=config.primitive,
            seed=primitive_seed,
        )
        target = sample_target_stiffness(
            initial_structures,
            config=config.optimization,
            seed=target_seed,
        )
        structure_batches.append(initial_structures)
        scaffold_batches.append(scaffold)
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
    optimized_cases = OptimizedCases(
        raw_structures=optimized_cases.raw_structures,
        target_stiffness=optimized_cases.target_stiffness,
        optimized_structures=optimized_cases.optimized_structures,
        initial_loss=optimized_cases.initial_loss,
        best_loss=optimized_cases.best_loss,
        last_analyses=optimized_cases.last_analyses,
        scaffolds=Scaffolds(
            positions=torch.cat([item.positions for item in scaffold_batches], dim=0),
            roles=torch.cat([item.roles for item in scaffold_batches], dim=0),
            adjacency=torch.cat([item.adjacency for item in scaffold_batches], dim=0),
            edge_primitive_types=torch.cat(
                [item.edge_primitive_types for item in scaffold_batches],
                dim=0,
            ),
        ),
    )
    optimized_cases.validate()
    save_offline_dataset(output_path, optimized_cases, config)
    preview_dir = (
        Path(config.preview_dir)
        if config.preview_dir is not None
        else output_path.parent / f"{output_path.stem}_preview"
    )
    write_dataset_visualizations(
        optimized_cases=optimized_cases,
        output_dir=preview_dir,
        max_cases=config.preview_cases,
    )
    return optimized_cases


def save_offline_dataset(
    path: str | Path,
    optimized_cases: OptimizedCases,
    config: OfflineDatasetConfig,
) -> Path:
    optimized_cases.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_serialize_optimized_cases(optimized_cases, config), path)
    return path


def load_offline_dataset(
    path: str | Path,
) -> tuple[OptimizedCases, OfflineDatasetConfig]:
    payload = torch.load(Path(path), map_location="cpu")
    optimized_cases = OptimizedCases(
        raw_structures=Structures(
            positions=payload["raw_structures"]["positions"],
            roles=payload["raw_structures"]["roles"],
            adjacency=payload["raw_structures"]["adjacency"],
        ),
        target_stiffness=payload["target_stiffness"],
        optimized_structures=Structures(
            positions=payload["optimized_structures"]["positions"],
            roles=payload["optimized_structures"]["roles"],
            adjacency=payload["optimized_structures"]["adjacency"],
        ),
        initial_loss=payload["initial_loss"],
        best_loss=payload["best_loss"],
        last_analyses=Analyses(
            generalized_stiffness=payload["last_analyses"]["generalized_stiffness"],
            material_usage=payload["last_analyses"]["material_usage"],
            short_beam_penalty=payload["last_analyses"]["short_beam_penalty"],
            long_beam_penalty=payload["last_analyses"]["long_beam_penalty"],
            thin_beam_penalty=payload["last_analyses"]["thin_beam_penalty"],
            thick_beam_penalty=payload["last_analyses"]["thick_beam_penalty"],
            free_node_spacing_penalty=payload["last_analyses"]["free_node_spacing_penalty"],
        ),
        scaffolds=Scaffolds(
            positions=payload["scaffolds"]["positions"],
            roles=payload["scaffolds"]["roles"],
            adjacency=payload["scaffolds"]["adjacency"],
            edge_primitive_types=payload["scaffolds"]["edge_primitive_types"],
        ),
    )
    optimized_cases.validate()
    return optimized_cases, OfflineDatasetConfig(**payload["config"])


def _serialize_optimized_cases(
    optimized_cases: OptimizedCases,
    config: OfflineDatasetConfig,
) -> dict[str, object]:
    optimized_cases.validate()
    if optimized_cases.scaffolds is None:
        raise ValueError("offline datasets require scaffold metadata for primitive previews")
    return {
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
        "scaffolds": {
            "positions": optimized_cases.scaffolds.positions,
            "roles": optimized_cases.scaffolds.roles,
            "adjacency": optimized_cases.scaffolds.adjacency,
            "edge_primitive_types": optimized_cases.scaffolds.edge_primitive_types,
        },
        "config": asdict(config),
    }
