from __future__ import annotations

from pathlib import Path
import random

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
from compliant_mechanism_synthesis.dataset.primitives import (
    ChainPrimitiveAssignment,
    _build_sheet_helix_centerline,
    _build_sheet_lateral_axes,
    _build_truss_helix_positions,
    _discretize_sheet_chain,
    _extract_primitive_segments,
    _materialize_truss_helix,
    _sample_chain_primitives,
)
from compliant_mechanism_synthesis.mechanics import normalize_generalized_stiffness
from compliant_mechanism_synthesis.roles import NodeRole


def _stiffness_interest_score(matrix: torch.Tensor) -> torch.Tensor:
    normalized = normalize_generalized_stiffness(matrix)
    eigenvalues = torch.linalg.eigvalsh(normalized)
    batch_mean = eigenvalues.mean(dim=0, keepdim=True)
    return (eigenvalues - batch_mean).square().mean(dim=-1)


def _sample_matching_random_primitives(
    config: PrimitiveConfig,
    *,
    seed_start: int,
    count: int,
) -> list[tuple[torch.Tensor, object]]:
    candidates_by_size: dict[int, list[tuple[torch.Tensor, object]]] = {}
    for seed in range(seed_start, seed_start + 256):
        structures, scaffold = sample_random_primitive(config=config, seed=seed)
        candidates = candidates_by_size.setdefault(structures.num_nodes, [])
        candidates.append((structures, scaffold))
        if len(candidates) == count:
            return candidates
    raise AssertionError("unable to sample matching primitive structures")


def _sheet_helix_assignment(sheet_width_nodes: int = 2) -> ChainPrimitiveAssignment:
    return ChainPrimitiveAssignment(
        chain=[0, 1, 2],
        primitive_type="sheet_helix",
        sheet_width_nodes=sheet_width_nodes,
        sheet_orientations=(0.0, 0.0, 0.0),
        offset_distances=(0.06, 0.06, 0.06),
        helix_phase=0.0,
        helix_pitch=0.30,
        width_start=0.015,
        width_end=0.015,
        thickness_start=0.010,
        thickness_end=0.010,
        twist_start=0.0,
        twist_end=0.0,
        sweep_phase=0.0,
    )


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


def test_extract_primitive_segments_merges_degree_two_runs() -> None:
    adjacency = torch.zeros((6, 6), dtype=torch.float32)
    for source, target in ((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)):
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    assert _extract_primitive_segments(adjacency) == [[0, 1, 2], [2, 3], [2, 4, 5]]


def test_sample_chain_primitives_disables_helices_on_longer_runs() -> None:
    observed_types = {
        _sample_chain_primitives(
            chains=[[0, 1, 2, 3]],
            config=PrimitiveConfig(),
            rng=random.Random(seed),
        )[0].primitive_type
        for seed in range(32)
    }

    assert observed_types <= {"rod", "sheet", "truss"}
    assert "rod_helix" not in observed_types
    assert "sheet_helix" not in observed_types


def test_sample_chain_primitives_downgrades_long_run_helices_without_forcing_truss() -> (
    None
):
    observed_types = {
        _sample_chain_primitives(
            chains=[[0, 1, 2, 3, 4]],
            config=PrimitiveConfig(),
            rng=random.Random(seed),
        )[0].primitive_type
        for seed in range(64)
    }

    assert observed_types <= {"rod", "sheet", "truss"}
    assert "rod" in observed_types or "sheet" in observed_types


def test_sample_random_primitive_builds_degree_capped_scaffold_graph() -> None:
    _, scaffolds = sample_random_primitive(
        config=PrimitiveConfig(
            num_free_nodes=12,
            max_scaffold_degree=4,
            long_range_connection_probability=1.0,
        ),
        seed=13,
    )

    degree = scaffolds.adjacency[0].sum(dim=1)
    edge_pairs = torch.nonzero(torch.triu(scaffolds.adjacency[0], diagonal=1) > 0.0)
    index_gaps = torch.abs(edge_pairs[:, 1] - edge_pairs[:, 0])

    assert torch.count_nonzero(degree >= 3.0) >= 2
    assert torch.all(degree[1:-1] >= 2.0)
    assert torch.all(degree <= 4.0)
    assert torch.any(index_gaps >= 4)


def test_materialize_scaffold_returns_valid_structure() -> None:
    structures, scaffold = sample_random_primitive(
        config=PrimitiveConfig(num_free_nodes=8),
        seed=15,
    )

    rematerialized = materialize_scaffold(
        scaffold, config=PrimitiveConfig(num_free_nodes=8)
    )

    rematerialized.validate()
    assert rematerialized.batch_size == structures.batch_size
    assert rematerialized.positions.shape == structures.positions.shape
    assert torch.allclose(rematerialized.positions, structures.positions)
    assert torch.equal(rematerialized.roles, structures.roles)
    assert torch.allclose(rematerialized.adjacency, structures.adjacency)


def test_optimize_scaffolds_returns_valid_batch() -> None:
    primitive_config = PrimitiveConfig(num_free_nodes=6)
    (_, scaffold_a), (_, scaffold_b) = _sample_matching_random_primitives(
        primitive_config,
        seed_start=27,
        count=2,
    )
    scaffolds = type(scaffold_a)(
        positions=torch.cat([scaffold_a.positions, scaffold_b.positions], dim=0),
        roles=torch.cat([scaffold_a.roles, scaffold_b.roles], dim=0),
        adjacency=torch.cat([scaffold_a.adjacency, scaffold_b.adjacency], dim=0),
        edge_primitive_ids=torch.cat(
            [scaffold_a.edge_primitive_ids, scaffold_b.edge_primitive_ids], dim=0
        ),
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
    chain_positions = torch.tensor(
        [[0.12, 0.50, 0.20], [0.50, 0.55, 0.50], [0.88, 0.50, 0.80]],
        dtype=torch.float32,
    )
    narrow_assignment = _sheet_helix_assignment(sheet_width_nodes=2)
    wide_assignment = _sheet_helix_assignment(sheet_width_nodes=4)
    narrow_centerline, narrow_radial_axes = _build_sheet_helix_centerline(
        chain_positions,
        narrow_assignment,
        target_edge_length=0.05,
        max_longitudinal_points=32,
    )
    wide_centerline, wide_radial_axes = _build_sheet_helix_centerline(
        chain_positions,
        wide_assignment,
        target_edge_length=0.05,
        max_longitudinal_points=32,
    )
    narrow_rows = _discretize_sheet_chain(
        narrow_centerline,
        _build_sheet_lateral_axes(
            narrow_centerline, narrow_assignment, narrow_radial_axes
        ),
        sheet_width_distance=0.02,
        sheet_width_nodes=2,
        helix_turns=0.0,
        helix_phase=0.0,
    )
    wide_rows = _discretize_sheet_chain(
        wide_centerline,
        _build_sheet_lateral_axes(wide_centerline, wide_assignment, wide_radial_axes),
        sheet_width_distance=0.02,
        sheet_width_nodes=4,
        helix_turns=0.0,
        helix_phase=0.0,
    )

    assert sum(row.shape[0] for row in wide_rows) > sum(
        row.shape[0] for row in narrow_rows
    )


def test_sheet_helix_responds_to_pitch_and_offset_controls() -> None:
    chain_positions = torch.tensor(
        [[0.12, 0.50, 0.20], [0.50, 0.55, 0.50], [0.88, 0.50, 0.80]],
        dtype=torch.float32,
    )
    relaxed_assignment = ChainPrimitiveAssignment(
        **{
            **_sheet_helix_assignment(sheet_width_nodes=2).__dict__,
            "offset_distances": (0.04, 0.04, 0.04),
            "helix_pitch": 0.40,
        }
    )
    tighter_assignment = ChainPrimitiveAssignment(
        **{
            **_sheet_helix_assignment(sheet_width_nodes=2).__dict__,
            "offset_distances": (0.20, 0.20, 0.20),
            "helix_pitch": 0.10,
        }
    )
    relaxed_centerline, _ = _build_sheet_helix_centerline(
        chain_positions,
        relaxed_assignment,
        target_edge_length=0.05,
        max_longitudinal_points=128,
    )
    tighter_centerline, _ = _build_sheet_helix_centerline(
        chain_positions,
        tighter_assignment,
        target_edge_length=0.05,
        max_longitudinal_points=128,
    )

    assert tighter_centerline.shape[0] > relaxed_centerline.shape[0]


def test_truss_materialization_creates_dense_skip_link_bracing() -> None:
    assignment = ChainPrimitiveAssignment(
        chain=[0, 1, 2, 3, 4],
        primitive_type="truss",
        sheet_width_nodes=2,
        sheet_orientations=(0.0, 0.0, 0.0, 0.0, 0.0),
        offset_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
        helix_phase=0.0,
        helix_pitch=0.30,
        width_start=0.015,
        width_end=0.015,
        thickness_start=0.010,
        thickness_end=0.010,
        twist_start=0.0,
        twist_end=0.0,
        sweep_phase=0.0,
    )
    chain_positions = torch.tensor(
        [
            [0.12, 0.50, 0.20],
            [0.31, 0.52, 0.35],
            [0.50, 0.55, 0.50],
            [0.69, 0.52, 0.65],
            [0.88, 0.50, 0.80],
        ],
        dtype=torch.float32,
    )
    truss_positions = _build_truss_helix_positions(chain_positions, assignment)
    next_node_index = 2
    edge_pairs: set[tuple[int, int]] = set()

    def add_node(position: torch.Tensor, role: NodeRole) -> int:
        nonlocal next_node_index
        assert role == NodeRole.FREE
        node_index = next_node_index
        next_node_index += 1
        return node_index

    def add_edge(source: int, target: int) -> None:
        edge_pairs.add(tuple(sorted((source, target))))

    _materialize_truss_helix(
        truss_positions,
        add_node=add_node,
        add_edge=add_edge,
        start_index=0,
        end_index=1,
    )
    node_count = next_node_index
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float32)
    for source, target in edge_pairs:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    degree = adjacency.sum(dim=1)
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
    )
    (structure_a, _), (structure_b, _) = _sample_matching_random_primitives(
        primitive_config,
        seed_start=20,
        count=2,
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

    neutral_score = _stiffness_interest_score(
        neutral.last_analyses.generalized_stiffness
    )
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
