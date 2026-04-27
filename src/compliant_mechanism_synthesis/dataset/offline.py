from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import random

import torch

from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    optimize_cases,
    optimize_scaffolds,
)
from compliant_mechanism_synthesis.adjacency import split_legacy_adjacency
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
    num_cases: int = 16_384
    batch_size: int = 64
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
    edge_radii = [batch.edge_radius for batch in batches]
    if any(edge_radius is not None for edge_radius in edge_radii) and not all(
        edge_radius is not None for edge_radius in edge_radii
    ):
        raise ValueError("all structure batches must consistently include edge_radius")
    return Structures(
        positions=torch.cat([batch.positions for batch in batches], dim=0),
        roles=torch.cat([batch.roles for batch in batches], dim=0),
        adjacency=torch.cat([batch.adjacency for batch in batches], dim=0),
        edge_radius=(
            None
            if not edge_radii or edge_radii[0] is None
            else torch.cat(
                [edge_radius for edge_radius in edge_radii if edge_radius is not None],
                dim=0,
            )
        ),
    )


def _concatenate_scaffolds(batches: list[Scaffolds]) -> Scaffolds:
    edge_radii = [batch.edge_radius for batch in batches]
    if any(edge_radius is not None for edge_radius in edge_radii) and not all(
        edge_radius is not None for edge_radius in edge_radii
    ):
        raise ValueError("all scaffold batches must consistently include edge_radius")
    return Scaffolds(
        positions=torch.cat([batch.positions for batch in batches], dim=0),
        roles=torch.cat([batch.roles for batch in batches], dim=0),
        adjacency=torch.cat([batch.adjacency for batch in batches], dim=0),
        edge_radius=(
            None
            if not edge_radii or edge_radii[0] is None
            else torch.cat(
                [edge_radius for edge_radius in edge_radii if edge_radius is not None],
                dim=0,
            )
        ),
        edge_primitive_ids=torch.cat(
            [batch.edge_primitive_ids for batch in batches], dim=0
        ),
        edge_primitive_types=torch.cat(
            [batch.edge_primitive_types for batch in batches],
            dim=0,
        ),
        edge_sheet_width_nodes=torch.cat(
            [batch.edge_sheet_width_nodes for batch in batches],
            dim=0,
        ),
        edge_orientation_start=torch.cat(
            [batch.edge_orientation_start for batch in batches],
            dim=0,
        ),
        edge_orientation_end=torch.cat(
            [batch.edge_orientation_end for batch in batches],
            dim=0,
        ),
        edge_offset_start=torch.cat(
            [batch.edge_offset_start for batch in batches],
            dim=0,
        ),
        edge_offset_end=torch.cat(
            [batch.edge_offset_end for batch in batches],
            dim=0,
        ),
        edge_helix_phase=torch.cat(
            [batch.edge_helix_phase for batch in batches],
            dim=0,
        ),
        edge_helix_pitch=torch.cat(
            [batch.edge_helix_pitch for batch in batches],
            dim=0,
        ),
        edge_width_start=torch.cat(
            [batch.edge_width_start for batch in batches],
            dim=0,
        ),
        edge_width_end=torch.cat(
            [batch.edge_width_end for batch in batches],
            dim=0,
        ),
        edge_thickness_start=torch.cat(
            [batch.edge_thickness_start for batch in batches],
            dim=0,
        ),
        edge_thickness_end=torch.cat(
            [batch.edge_thickness_end for batch in batches],
            dim=0,
        ),
        edge_twist_start=torch.cat(
            [batch.edge_twist_start for batch in batches],
            dim=0,
        ),
        edge_twist_end=torch.cat(
            [batch.edge_twist_end for batch in batches],
            dim=0,
        ),
        edge_sweep_phase=torch.cat(
            [batch.edge_sweep_phase for batch in batches],
            dim=0,
        ),
    )


def _concatenate_analyses(batches: list[Analyses]) -> Analyses:
    nodal_displacements = None
    edge_von_mises = None
    has_nodal_displacements = [
        batch.nodal_displacements is not None for batch in batches
    ]
    has_edge_von_mises = [batch.edge_von_mises is not None for batch in batches]
    if any(has_nodal_displacements) and not all(has_nodal_displacements):
        raise ValueError("all analyses must consistently include nodal_displacements")
    if any(has_edge_von_mises) and not all(has_edge_von_mises):
        raise ValueError("all analyses must consistently include edge_von_mises")
    if batches and all(has_nodal_displacements):
        nodal_displacements = torch.cat(
            [
                batch.nodal_displacements
                for batch in batches
                if batch.nodal_displacements is not None
            ],
            dim=0,
        )
    if batches and all(has_edge_von_mises):
        edge_von_mises = torch.cat(
            [
                batch.edge_von_mises
                for batch in batches
                if batch.edge_von_mises is not None
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
        nodal_displacements=nodal_displacements,
        edge_von_mises=edge_von_mises,
    )


def _concatenate_optimized_case_batches(
    optimized_batches: list[OptimizedCases],
    scaffold_batches: list[Scaffolds],
) -> OptimizedCases:
    return OptimizedCases(
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
    optimized_batches: list[OptimizedCases] = []
    scaffold_batches: list[Scaffolds] = []
    target_node_count: int | None = None
    target_roles: torch.Tensor | None = None

    def flush_batch(batch_index: int) -> None:
        nonlocal processed_cases
        if not current_structures:
            return
        batch_scaffolds = _concatenate_scaffolds(current_scaffolds)
        if config.optimization.scaffold_num_steps > 0:
            batch_scaffolds, batch_structures = optimize_scaffolds(
                scaffolds=batch_scaffolds,
                primitive_config=primitive_config,
                config=config.optimization,
                logdir=Path(config.logdir) / f"batch_{batch_index:04d}_scaffold",
            )
        else:
            batch_structures = _concatenate_structures(current_structures)
        optimized_batch = optimize_cases(
            structures=batch_structures,
            config=config.optimization,
            logdir=Path(config.logdir) / f"batch_{batch_index:04d}",
        )
        optimized_batches.append(optimized_batch)
        scaffold_batches.append(batch_scaffolds)
        processed_cases += batch_structures.batch_size
        print(
            f"dataset batch {batch_index + 1} "
            f"node_count={target_node_count} "
            f"cases={processed_cases}/{config.num_cases} "
            f"mean_best_loss={float(optimized_batch.best_loss.mean().item()):.6f}",
            flush=True,
        )
        current_structures.clear()
        current_scaffolds.clear()

    while processed_cases + len(current_structures) < config.num_cases:
        primitive_seed = rng.randrange(0, 2**31)
        initial_structures, scaffold = sample_random_primitive(
            config=primitive_config,
            seed=primitive_seed,
        )
        initial_structures = initial_structures.to(device)
        scaffold = scaffold.to(device)
        node_count = initial_structures.num_nodes
        if target_node_count is None:
            target_node_count = node_count
            target_roles = initial_structures.roles.detach().cpu().clone()
        if node_count != target_node_count:
            continue
        if target_roles is None or not torch.equal(
            initial_structures.roles.detach().cpu(),
            target_roles,
        ):
            continue
        current_structures.append(initial_structures)
        current_scaffolds.append(scaffold)
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
        num_cases=optimized_cases.optimized_structures.batch_size,
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
        target_stiffness=payload["target_stiffness"],
        optimized_structures=Structures(
            adjacency=(
                payload["optimized_structures"]["adjacency"]
                if "edge_radius" in payload["optimized_structures"]
                else split_legacy_adjacency(
                    payload["optimized_structures"]["adjacency"]
                )[0]
            ),
            positions=payload["optimized_structures"]["positions"],
            roles=payload["optimized_structures"]["roles"],
            edge_radius=payload["optimized_structures"].get("edge_radius")
            if "edge_radius" in payload["optimized_structures"]
            else split_legacy_adjacency(payload["optimized_structures"]["adjacency"])[
                1
            ],
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
            nodal_displacements=payload["last_analyses"].get(
                "nodal_displacements",
                payload["last_analyses"].get("nodal_mechanics"),
            ),
            edge_von_mises=payload["last_analyses"].get("edge_von_mises"),
        ),
        scaffolds=Scaffolds(
            positions=payload["scaffolds"]["positions"],
            roles=payload["scaffolds"]["roles"],
            adjacency=payload["scaffolds"]["adjacency"],
            edge_primitive_ids=payload["scaffolds"]["edge_primitive_ids"],
            edge_primitive_types=payload["scaffolds"]["edge_primitive_types"],
            edge_sheet_width_nodes=payload["scaffolds"]["edge_sheet_width_nodes"],
            edge_orientation_start=payload["scaffolds"]["edge_orientation_start"],
            edge_orientation_end=payload["scaffolds"]["edge_orientation_end"],
            edge_offset_start=payload["scaffolds"]["edge_offset_start"],
            edge_offset_end=payload["scaffolds"]["edge_offset_end"],
            edge_helix_phase=payload["scaffolds"]["edge_helix_phase"],
            edge_helix_pitch=payload["scaffolds"]["edge_helix_pitch"],
            edge_width_start=payload["scaffolds"]["edge_width_start"],
            edge_width_end=payload["scaffolds"]["edge_width_end"],
            edge_thickness_start=payload["scaffolds"]["edge_thickness_start"],
            edge_thickness_end=payload["scaffolds"]["edge_thickness_end"],
            edge_twist_start=payload["scaffolds"]["edge_twist_start"],
            edge_twist_end=payload["scaffolds"]["edge_twist_end"],
            edge_sweep_phase=payload["scaffolds"]["edge_sweep_phase"],
            edge_radius=payload["scaffolds"].get("edge_radius"),
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
        "target_stiffness": serialized.target_stiffness,
        "optimized_structures": {
            "positions": serialized.optimized_structures.positions,
            "roles": serialized.optimized_structures.roles,
            "adjacency": serialized.optimized_structures.adjacency,
            "edge_radius": serialized.optimized_structures.edge_radius,
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
            "nodal_displacements": serialized.last_analyses.nodal_displacements,
            "edge_von_mises": serialized.last_analyses.edge_von_mises,
        },
        "scaffolds": {
            "positions": serialized.scaffolds.positions,
            "roles": serialized.scaffolds.roles,
            "adjacency": serialized.scaffolds.adjacency,
            "edge_radius": serialized.scaffolds.edge_radius,
            "edge_primitive_ids": serialized.scaffolds.edge_primitive_ids,
            "edge_primitive_types": serialized.scaffolds.edge_primitive_types,
            "edge_sheet_width_nodes": serialized.scaffolds.edge_sheet_width_nodes,
            "edge_orientation_start": serialized.scaffolds.edge_orientation_start,
            "edge_orientation_end": serialized.scaffolds.edge_orientation_end,
            "edge_offset_start": serialized.scaffolds.edge_offset_start,
            "edge_offset_end": serialized.scaffolds.edge_offset_end,
            "edge_helix_phase": serialized.scaffolds.edge_helix_phase,
            "edge_helix_pitch": serialized.scaffolds.edge_helix_pitch,
            "edge_width_start": serialized.scaffolds.edge_width_start,
            "edge_width_end": serialized.scaffolds.edge_width_end,
            "edge_thickness_start": serialized.scaffolds.edge_thickness_start,
            "edge_thickness_end": serialized.scaffolds.edge_thickness_end,
            "edge_twist_start": serialized.scaffolds.edge_twist_start,
            "edge_twist_end": serialized.scaffolds.edge_twist_end,
            "edge_sweep_phase": serialized.scaffolds.edge_sweep_phase,
        },
        "config": asdict(config),
    }
