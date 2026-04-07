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
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Scaffolds,
    Structures,
)
from compliant_mechanism_synthesis.utils import resolve_torch_device
from compliant_mechanism_synthesis.visualization import write_dataset_visualizations


@dataclass(frozen=True)
class OfflineDatasetConfig:
    num_cases: int = 32
    seed: int = 7
    device: str = "auto"
    output_path: str = "datasets/offline_dataset.pt"
    logdir: str = "runs/offline_dataset"
    preview_dir: str | None = None
    preview_case_number: int = 8
    primitive: PrimitiveConfig = PrimitiveConfig()
    optimization: CaseOptimizationConfig = field(default_factory=CaseOptimizationConfig)


def _sample_preview_case_indices(
    num_cases: int,
    preview_case_number: int,
    seed: int,
) -> list[int]:
    if num_cases < 0:
        raise ValueError("num_cases must be non-negative")
    if preview_case_number <= 0:
        raise ValueError("preview_case_number must be positive")
    if num_cases <= preview_case_number:
        return list(range(num_cases))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_cases), k=preview_case_number))


def generate_offline_dataset(
    config: OfflineDatasetConfig | None = None,
) -> OptimizedCases:
    config = config or OfflineDatasetConfig()
    device = resolve_torch_device(config.device)
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
        initial_structures = initial_structures.to(device)
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
    preview_case_indices = _sample_preview_case_indices(
        num_cases=optimized_cases.raw_structures.batch_size,
        preview_case_number=config.preview_case_number,
        seed=config.seed,
    )
    write_dataset_visualizations(
        optimized_cases=optimized_cases,
        output_dir=preview_dir,
        case_indices=preview_case_indices,
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
            free_node_spacing_penalty=payload["last_analyses"][
                "free_node_spacing_penalty"
            ],
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
        raise ValueError(
            "offline datasets require scaffold metadata for primitive previews"
        )
    serialized = optimized_cases.to("cpu")
    return {
        "raw_structures": {
            "positions": serialized.raw_structures.positions,
            "roles": serialized.raw_structures.roles,
            "adjacency": serialized.raw_structures.adjacency,
        },
        "target_stiffness": serialized.target_stiffness,
        "optimized_structures": {
            "positions": serialized.optimized_structures.positions,
            "roles": serialized.optimized_structures.roles,
            "adjacency": serialized.optimized_structures.adjacency,
        },
        "initial_loss": serialized.initial_loss,
        "best_loss": serialized.best_loss,
        "last_analyses": {
            "generalized_stiffness": serialized.last_analyses.generalized_stiffness,
            "material_usage": serialized.last_analyses.material_usage,
            "short_beam_penalty": serialized.last_analyses.short_beam_penalty,
            "long_beam_penalty": serialized.last_analyses.long_beam_penalty,
            "thin_beam_penalty": serialized.last_analyses.thin_beam_penalty,
            "thick_beam_penalty": serialized.last_analyses.thick_beam_penalty,
            "free_node_spacing_penalty": serialized.last_analyses.free_node_spacing_penalty,
        },
        "scaffolds": {
            "positions": serialized.scaffolds.positions,
            "roles": serialized.scaffolds.roles,
            "adjacency": serialized.scaffolds.adjacency,
            "edge_primitive_types": serialized.scaffolds.edge_primitive_types,
        },
        "config": asdict(config),
    }
