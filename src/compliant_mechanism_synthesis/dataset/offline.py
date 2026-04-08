from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import random

import torch

from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    optimize_cases,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    CHAIN_PRIMITIVE_LIBRARY,
    PrimitiveConfig,
    _intertwined_scaffold_edges,
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
    batch_size: int = 32
    seed: int = 7
    device: str = "auto"
    output_path: str = "datasets/dataset.pt"
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


def _concatenate_structures(batches: list[Structures]) -> Structures:
    return Structures(
        positions=torch.cat([batch.positions for batch in batches], dim=0),
        roles=torch.cat([batch.roles for batch in batches], dim=0),
        adjacency=torch.cat([batch.adjacency for batch in batches], dim=0),
    )


def _concatenate_scaffolds(batches: list[Scaffolds]) -> Scaffolds:
    return Scaffolds(
        positions=torch.cat([batch.positions for batch in batches], dim=0),
        roles=torch.cat([batch.roles for batch in batches], dim=0),
        adjacency=torch.cat([batch.adjacency for batch in batches], dim=0),
        edge_primitive_types=torch.cat(
            [batch.edge_primitive_types for batch in batches],
            dim=0,
        ),
    )


def _concatenate_analyses(batches: list[Analyses]) -> Analyses:
    nodal_mechanics = None
    has_nodal_mechanics = [batch.nodal_mechanics is not None for batch in batches]
    if any(has_nodal_mechanics) and not all(has_nodal_mechanics):
        raise ValueError("all analyses must consistently include nodal_mechanics")
    if batches and all(has_nodal_mechanics):
        nodal_mechanics = torch.cat(
            [
                batch.nodal_mechanics
                for batch in batches
                if batch.nodal_mechanics is not None
            ],
            dim=0,
        )
    return Analyses(
        generalized_stiffness=torch.cat(
            [batch.generalized_stiffness for batch in batches], dim=0
        ),
        material_usage=torch.cat([batch.material_usage for batch in batches], dim=0),
        short_beam_penalty=torch.cat(
            [batch.short_beam_penalty for batch in batches], dim=0
        ),
        long_beam_penalty=torch.cat(
            [batch.long_beam_penalty for batch in batches], dim=0
        ),
        thin_beam_penalty=torch.cat(
            [batch.thin_beam_penalty for batch in batches], dim=0
        ),
        thick_beam_penalty=torch.cat(
            [batch.thick_beam_penalty for batch in batches], dim=0
        ),
        free_node_spacing_penalty=torch.cat(
            [batch.free_node_spacing_penalty for batch in batches], dim=0
        ),
        nodal_mechanics=nodal_mechanics,
    )


def _concatenate_optimized_case_batches(
    optimized_batches: list[OptimizedCases],
    scaffold_batches: list[Scaffolds],
) -> OptimizedCases:
    return OptimizedCases(
        raw_structures=_concatenate_structures(
            [batch.raw_structures for batch in optimized_batches]
        ),
        target_stiffness=torch.cat(
            [batch.target_stiffness for batch in optimized_batches], dim=0
        ),
        optimized_structures=_concatenate_structures(
            [batch.optimized_structures for batch in optimized_batches]
        ),
        initial_loss=torch.cat(
            [batch.initial_loss for batch in optimized_batches], dim=0
        ),
        best_loss=torch.cat([batch.best_loss for batch in optimized_batches], dim=0),
        last_analyses=_concatenate_analyses(
            [batch.last_analyses for batch in optimized_batches]
        ),
        scaffolds=_concatenate_scaffolds(scaffold_batches),
    )


def generate_offline_dataset(
    config: OfflineDatasetConfig | None = None,
) -> OptimizedCases:
    config = config or OfflineDatasetConfig()
    device = resolve_torch_device(config.device)
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.logdir).mkdir(parents=True, exist_ok=True)
    total_batches = (config.num_cases + config.batch_size - 1) // config.batch_size
    processed_cases = 0

    print(
        f"dataset generation started num_cases={config.num_cases} "
        f"batch_size={config.batch_size} total_batches={total_batches}",
        flush=True,
    )

    rng = random.Random(config.seed)
    primitive_config = config.primitive
    if (
        primitive_config.forced_segment_primitive_types is None
        and primitive_config.forced_primitive_type is None
    ):
        primitive_config = replace(
            primitive_config,
            forced_segment_primitive_types=tuple(
                rng.choice(CHAIN_PRIMITIVE_LIBRARY)
                for _ in range(
                    len(
                        _intertwined_scaffold_edges(primitive_config.num_free_nodes + 2)
                    )
                )
            ),
        )
    if primitive_config.sample_sheet_helix_width_nodes:
        if (
            primitive_config.sheet_helix_width_nodes_min
            > primitive_config.sheet_helix_width_nodes_max
        ):
            raise ValueError(
                "sheet_helix_width_nodes_min must be <= sheet_helix_width_nodes_max"
            )
        primitive_config = replace(
            primitive_config,
            sheet_width_nodes=rng.randint(
                primitive_config.sheet_helix_width_nodes_min,
                primitive_config.sheet_helix_width_nodes_max,
            ),
            sample_sheet_helix_width_nodes=False,
        )
    current_structures: list[Structures] = []
    current_scaffolds: list[Scaffolds] = []
    current_targets: list[torch.Tensor] = []
    optimized_batches: list[OptimizedCases] = []
    scaffold_batches: list[Scaffolds] = []

    def flush_batch(batch_index: int) -> None:
        nonlocal processed_cases
        if not current_structures:
            return
        batch_structures = _concatenate_structures(current_structures)
        batch_targets = torch.cat(current_targets, dim=0)
        optimized_batch = optimize_cases(
            structures=batch_structures,
            target_stiffness=batch_targets,
            config=config.optimization,
            logdir=Path(config.logdir) / f"batch_{batch_index:04d}",
        )
        optimized_batches.append(optimized_batch)
        scaffold_batches.append(_concatenate_scaffolds(current_scaffolds))
        processed_cases += batch_structures.batch_size
        print(
            f"dataset batch {batch_index + 1}/{total_batches} "
            f"cases={processed_cases}/{config.num_cases} "
            f"mean_best_loss={float(optimized_batch.best_loss.mean().item()):.6f}",
            flush=True,
        )
        current_structures.clear()
        current_scaffolds.clear()
        current_targets.clear()

    for case_index in range(config.num_cases):
        primitive_seed = rng.randrange(0, 2**31)
        target_seed = rng.randrange(0, 2**31)
        initial_structures, scaffold = sample_random_primitive(
            config=primitive_config,
            seed=primitive_seed,
        )
        initial_structures = initial_structures.to(device)
        target = sample_target_stiffness(
            initial_structures,
            config=config.optimization,
            seed=target_seed,
        )
        current_structures.append(initial_structures)
        current_scaffolds.append(scaffold)
        current_targets.append(target.unsqueeze(0))
        if len(current_structures) == config.batch_size:
            flush_batch(batch_index=len(optimized_batches))

    flush_batch(batch_index=len(optimized_batches))
    optimized_cases = _concatenate_optimized_case_batches(
        optimized_batches=optimized_batches,
        scaffold_batches=scaffold_batches,
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
            nodal_mechanics=payload["last_analyses"].get("nodal_mechanics"),
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
            "nodal_mechanics": serialized.last_analyses.nodal_mechanics,
        },
        "scaffolds": {
            "positions": serialized.scaffolds.positions,
            "roles": serialized.scaffolds.roles,
            "adjacency": serialized.scaffolds.adjacency,
            "edge_primitive_types": serialized.scaffolds.edge_primitive_types,
        },
        "config": asdict(config),
    }
