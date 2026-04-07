from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from compliant_mechanism_synthesis.dataset.types import Scaffolds, Structures
from compliant_mechanism_synthesis.roles import NodeRole
from compliant_mechanism_synthesis.tensor_ops import symmetrize_matrix

CHAIN_PRIMITIVE_LIBRARY = (
    "rod",
    "rod_helix",
    "sheet",
    "sheet_helix",
    "truss",
)

# During scaffold-to-mesh debugging we intentionally keep a single active
# primitive family so we can inspect one geometry/conectivity pattern at a
# time before re-introducing the others.
ACTIVE_CHAIN_PRIMITIVE_LIBRARY = ("truss",)


@dataclass(frozen=True)
class PrimitiveConfig:
    # This controls the sparse scaffolding graph, not the final number of free
    # FEM nodes. We keep the default sparse on purpose during primitive
    # debugging so the scaffold-to-mesh conversion remains visually readable.
    num_free_nodes: int = 6
    width: float = 0.18
    thickness: float = 0.10
    anchor_radius: float = 0.04
    # The final mesh primitives should read as thin composable parts rather
    # than a few oversized blobs. Using a 0.04 workspace-wide diameter keeps
    # the primitive family legible once many of them are composed together.
    primitive_radius: float = 0.015
    neighbor_count: int = 3
    extra_connection_probability: float = 0.18
    connection_length_scale: float = 0.22
    chain_connection_probability: float = 0.72
    free_z_min: float = 0.28
    free_z_max: float = 0.72
    fixed_anchor_z: float = 0.10
    mobile_anchor_z: float = 0.90
    target_edge_length: float = 0.05
    helix_radius: float = 0.06
    helix_turns: float = 12.0
    truss_target_edge_length_scale: float = 0.33
    sheet_width_nodes: int = 4
    sheet_width_distance: float = 0.02
    sample_sheet_helix_width_nodes: bool = True
    sheet_helix_width_nodes_min: int = 2
    sheet_helix_width_nodes_max: int = 4
    sheet_helix_offset_distance_min: float = 0.06
    sheet_helix_offset_distance_max: float = 0.10
    sheet_helix_pitch_distance: float = 0.16
    sheet_helix_max_longitudinal_points: int = 128
    forced_primitive_type: str | None = None


@dataclass(frozen=True)
class ChainPrimitiveAssignment:
    chain: list[int]
    primitive_type: str
    sheet_width_nodes: int
    sheet_orientations: tuple[float, ...]
    offset_distances: tuple[float, ...]
    helix_phase: float
    helix_pitch: float
    width_start: float
    width_end: float
    thickness_start: float
    thickness_end: float
    twist_start: float
    twist_end: float
    sweep_phase: float


def _orthonormal_frame(
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    reference = torch.tensor(
        [0.0, 0.0, 1.0], dtype=direction.dtype, device=direction.device
    )
    if abs(float(direction[2].item())) > 0.85:
        reference = torch.tensor(
            [0.0, 1.0, 0.0], dtype=direction.dtype, device=direction.device
        )
    normal_1 = torch.linalg.cross(reference, direction)
    normal_1 = normal_1 / torch.linalg.vector_norm(normal_1).clamp_min(1e-8)
    normal_2 = torch.linalg.cross(direction, normal_1)
    return direction, normal_1, normal_2


def _sample_random_scaffold_centers(
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    start_center = torch.tensor(
        [0.12, 0.50, config.fixed_anchor_z], dtype=torch.float32
    )
    end_center = torch.tensor([0.88, 0.50, config.mobile_anchor_z], dtype=torch.float32)

    base_x = torch.linspace(
        0.22, 0.78, steps=config.num_free_nodes, dtype=torch.float32
    )
    free_positions = torch.stack(
        [
            base_x
            + torch.tensor(
                [rng.uniform(-0.025, 0.025) for _ in range(config.num_free_nodes)],
                dtype=torch.float32,
            ),
            torch.tensor(
                [rng.uniform(0.20, 0.80) for _ in range(config.num_free_nodes)],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    rng.uniform(config.free_z_min, config.free_z_max)
                    for _ in range(config.num_free_nodes)
                ],
                dtype=torch.float32,
            ),
        ],
        dim=-1,
    )
    free_positions = free_positions[torch.argsort(free_positions[:, 0])]
    return torch.cat(
        [start_center.unsqueeze(0), free_positions, end_center.unsqueeze(0)], dim=0
    )


def _localized_edge_probability(
    distance: float,
    config: PrimitiveConfig,
) -> float:
    return math.exp(-(distance**2) / max(config.connection_length_scale**2, 1e-8))


def _ensure_min_degree_two(
    centers: torch.Tensor,
    adjacency: torch.Tensor,
) -> None:
    pairwise = torch.linalg.vector_norm(
        centers[:, None, :] - centers[None, :, :], dim=-1
    )
    num_nodes = centers.shape[0]
    for node_index in range(num_nodes):
        while int(adjacency[node_index].sum().item()) < 2:
            candidate_order = torch.argsort(pairwise[node_index])
            for candidate in candidate_order.tolist():
                if candidate == node_index or adjacency[node_index, candidate] > 0.0:
                    continue
                adjacency[node_index, candidate] = 1.0
                adjacency[candidate, node_index] = 1.0
                break


def _sample_scaffold_connectivity(
    centers: torch.Tensor,
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    num_nodes = centers.shape[0]
    if set(ACTIVE_CHAIN_PRIMITIVE_LIBRARY).issubset(
        {"rod", "rod_helix", "sheet", "sheet_helix", "truss"}
    ):
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        remaining = list(range(1, num_nodes - 1))
        order = [0]
        current = 0
        while remaining:
            weights = []
            for candidate in remaining:
                distance = float(
                    torch.linalg.vector_norm(
                        centers[candidate] - centers[current]
                    ).item()
                )
                forward_bias = max(
                    float(centers[candidate, 0].item() - centers[current, 0].item()),
                    0.02,
                )
                weights.append(
                    _localized_edge_probability(distance, config) * forward_bias
                )
            next_index = rng.choices(remaining, weights=weights, k=1)[0]
            order.append(next_index)
            remaining.remove(next_index)
            current = next_index
        order.append(num_nodes - 1)
        for source, target in zip(order[:-1], order[1:]):
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        return adjacency

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    pairwise = torch.linalg.vector_norm(
        centers[:, None, :] - centers[None, :, :], dim=-1
    )

    # First build a localized spanning tree so every scaffold node belongs to a
    # connected support graph before we add random local redundancy.
    for node_index in range(1, num_nodes):
        candidate_indices = list(range(node_index))
        candidate_indices.sort(
            key=lambda index: float(pairwise[node_index, index].item())
        )
        local_candidates = candidate_indices[: max(2, config.neighbor_count)]
        weights = [
            _localized_edge_probability(
                float(pairwise[node_index, candidate].item()), config
            )
            for candidate in local_candidates
        ]
        connected_to = rng.choices(local_candidates, weights=weights, k=1)[0]
        adjacency[node_index, connected_to] = 1.0
        adjacency[connected_to, node_index] = 1.0

    for node_index in range(num_nodes):
        nearest = torch.argsort(pairwise[node_index])[1 : config.neighbor_count + 2]
        for candidate in nearest.tolist():
            if adjacency[node_index, candidate] > 0.0:
                continue
            probability = (
                config.extra_connection_probability
                * _localized_edge_probability(
                    float(pairwise[node_index, candidate].item()),
                    config,
                )
            )
            if rng.random() < probability:
                adjacency[node_index, candidate] = 1.0
                adjacency[candidate, node_index] = 1.0

    _ensure_min_degree_two(centers, adjacency)
    return adjacency


def _sample_chain_primitives(
    chains: list[list[int]],
    config: PrimitiveConfig,
    rng: random.Random,
) -> list[ChainPrimitiveAssignment]:
    assignments = []
    if (
        config.forced_primitive_type is not None
        and config.forced_primitive_type not in CHAIN_PRIMITIVE_LIBRARY
    ):
        raise ValueError("forced_primitive_type must be a known primitive family")
    for chain in chains:
        width_scale = rng.uniform(0.8, 1.15)
        thickness_scale = rng.uniform(0.45, 0.85)
        primitive_type = (
            config.forced_primitive_type
            if config.forced_primitive_type is not None
            else rng.choice(ACTIVE_CHAIN_PRIMITIVE_LIBRARY)
        )
        sheet_width_nodes = config.sheet_width_nodes
        if primitive_type == "sheet_helix" and config.sample_sheet_helix_width_nodes:
            if config.sheet_helix_width_nodes_min > config.sheet_helix_width_nodes_max:
                raise ValueError(
                    "sheet_helix_width_nodes_min must be <= sheet_helix_width_nodes_max"
                )
            sheet_width_nodes = rng.randint(
                config.sheet_helix_width_nodes_min,
                config.sheet_helix_width_nodes_max,
            )
        assignments.append(
            ChainPrimitiveAssignment(
                chain=chain,
                primitive_type=primitive_type,
                sheet_width_nodes=sheet_width_nodes,
                sheet_orientations=tuple(
                    rng.uniform(0.0, 2.0 * math.pi) for _ in range(len(chain))
                ),
                offset_distances=tuple(
                    rng.uniform(
                        config.sheet_helix_offset_distance_min,
                        config.sheet_helix_offset_distance_max,
                    )
                    if primitive_type == "sheet_helix"
                    else 0.0
                    for _ in range(len(chain))
                ),
                helix_phase=rng.uniform(0.0, 2.0 * math.pi),
                helix_pitch=config.sheet_helix_pitch_distance,
                width_start=config.primitive_radius
                * width_scale
                * rng.uniform(0.8, 1.1),
                width_end=config.primitive_radius * width_scale * rng.uniform(0.8, 1.1),
                thickness_start=config.primitive_radius
                * thickness_scale
                * rng.uniform(0.8, 1.1),
                thickness_end=config.primitive_radius
                * thickness_scale
                * rng.uniform(0.8, 1.1),
                twist_start=rng.uniform(-0.8, 0.8),
                twist_end=rng.uniform(-0.8, 0.8),
                sweep_phase=rng.uniform(0.0, 2.0 * math.pi),
            )
        )
    return assignments


def _extract_primitive_chains(adjacency: torch.Tensor) -> list[list[int]]:
    num_nodes = adjacency.shape[0]
    degree = adjacency.sum(dim=1).to(dtype=torch.long)
    visited_edges: set[tuple[int, int]] = set()
    chains: list[list[int]] = []

    def walk_chain(start: int, nxt: int) -> list[int]:
        chain = [start, nxt]
        previous = start
        current = nxt
        visited_edges.add(tuple(sorted((start, nxt))))
        while degree[current].item() == 2:
            neighbors = (
                torch.nonzero(adjacency[current] > 0.0, as_tuple=False)
                .flatten()
                .tolist()
            )
            candidates = [neighbor for neighbor in neighbors if neighbor != previous]
            if not candidates:
                break
            following = candidates[0]
            edge_key = tuple(sorted((current, following)))
            if edge_key in visited_edges:
                break
            chain.append(following)
            visited_edges.add(edge_key)
            previous, current = current, following
        return chain

    branch_nodes = [index for index in range(num_nodes) if degree[index].item() != 2]
    for start in branch_nodes:
        neighbors = (
            torch.nonzero(adjacency[start] > 0.0, as_tuple=False).flatten().tolist()
        )
        for nxt in neighbors:
            edge_key = tuple(sorted((start, nxt)))
            if edge_key in visited_edges:
                continue
            chains.append(walk_chain(start, nxt))

    for start in range(num_nodes):
        neighbors = (
            torch.nonzero(adjacency[start] > 0.0, as_tuple=False).flatten().tolist()
        )
        for nxt in neighbors:
            edge_key = tuple(sorted((start, nxt)))
            if edge_key in visited_edges:
                continue
            chains.append(walk_chain(start, nxt))

    return chains


def _estimate_scaffold_tangent(
    centers: torch.Tensor,
    adjacency: torch.Tensor,
    node_index: int,
) -> torch.Tensor:
    neighbors = torch.nonzero(adjacency[node_index] > 0.0, as_tuple=False).flatten()
    if neighbors.numel() == 0:
        return torch.tensor([1.0, 0.0, 0.0], dtype=centers.dtype)
    offsets = centers[neighbors] - centers[node_index]
    tangent = offsets.mean(dim=0)
    if torch.linalg.vector_norm(tangent).item() < 1e-8:
        tangent = offsets[0]
    if torch.linalg.vector_norm(tangent).item() < 1e-8:
        tangent = torch.tensor([1.0, 0.0, 0.0], dtype=centers.dtype)
    return tangent


def _primitive_center_offset(
    primitive_type: str,
    normal_1: torch.Tensor,
    normal_2: torch.Tensor,
    width: float,
    thickness: float,
    twist: float,
    sweep_phase: float,
) -> torch.Tensor:
    if primitive_type == "rod":
        # The debug rod should stay centered on the scaffold so we can assess
        # whether the expanded FEM mesh actually follows the sparse graph path.
        return torch.zeros_like(normal_1)
    if primitive_type == "rod_helix":
        return torch.zeros_like(normal_1)
    if primitive_type == "sheet":
        return torch.zeros_like(normal_1)
    if primitive_type == "sheet_helix":
        return torch.zeros_like(normal_1)
    if primitive_type == "truss":
        return torch.zeros_like(normal_1)
    raise ValueError(f"unknown primitive type: {primitive_type}")


def _apply_chain_style_offsets(
    centers: torch.Tensor,
    adjacency: torch.Tensor,
    assignments: list[ChainPrimitiveAssignment],
    config: PrimitiveConfig,
) -> torch.Tensor:
    adjusted = centers.clone()
    offset_sum = torch.zeros_like(centers)
    offset_count = torch.zeros((centers.shape[0], 1), dtype=centers.dtype)

    for assignment in assignments:
        chain = assignment.chain
        if len(chain) < 2:
            continue

        for local_index, node_index in enumerate(chain):
            if node_index in {0, centers.shape[0] - 1}:
                continue
            if local_index == 0 or local_index == len(chain) - 1:
                continue

            tangent = _estimate_scaffold_tangent(centers, adjacency, node_index)
            _, normal_1, normal_2 = _orthonormal_frame(tangent)
            fraction = local_index / max(len(chain) - 1, 1)
            width = (
                1.0 - fraction
            ) * assignment.width_start + fraction * assignment.width_end
            thickness = (
                1.0 - fraction
            ) * assignment.thickness_start + fraction * assignment.thickness_end
            twist = (
                1.0 - fraction
            ) * assignment.twist_start + fraction * assignment.twist_end
            offset = _primitive_center_offset(
                primitive_type=assignment.primitive_type,
                normal_1=normal_1,
                normal_2=normal_2,
                width=width,
                thickness=thickness,
                twist=twist,
                sweep_phase=assignment.sweep_phase,
            )

            offset_sum[node_index] = offset_sum[node_index] + offset
            offset_count[node_index] = offset_count[node_index] + 1.0

    valid = offset_count.squeeze(-1) > 0.0
    adjusted[valid] = adjusted[valid] + offset_sum[valid] / offset_count[valid]
    adjusted[:, 2] = adjusted[:, 2].clamp(config.free_z_min, config.free_z_max)
    adjusted[0, 2] = config.fixed_anchor_z
    adjusted[-1, 2] = config.mobile_anchor_z
    return adjusted


def _materialize_scaffold_node_triplets(
    centers: torch.Tensor,
    adjacency: torch.Tensor,
    assignments: list[ChainPrimitiveAssignment],
    config: PrimitiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    degree = adjacency.sum(dim=1).to(dtype=torch.long)

    positions: list[torch.Tensor] = []
    roles: list[int] = []
    edges: set[tuple[int, int]] = set()
    connection_node_indices: dict[int, int] = {}

    def add_node(position: torch.Tensor, role: NodeRole) -> int:
        positions.append(position)
        roles.append(int(role))
        return len(positions) - 1

    def add_edge(source: int, target: int) -> None:
        if source == target:
            return
        edges.add(tuple(sorted((source, target))))

    def ensure_connection_node(scaffold_index: int) -> int:
        existing = connection_node_indices.get(scaffold_index)
        if existing is not None:
            return existing
        connection_index = add_node(centers[scaffold_index], NodeRole.FREE)
        connection_node_indices[scaffold_index] = connection_index
        if scaffold_index == 0 or scaffold_index == centers.shape[0] - 1:
            tangent = _estimate_scaffold_tangent(centers, adjacency, scaffold_index)
            _, normal_1, normal_2 = _orthonormal_frame(tangent)
            anchor_role = NodeRole.FIXED if scaffold_index == 0 else NodeRole.MOBILE
            anchor_offsets = (
                config.anchor_radius * normal_1,
                config.anchor_radius * (-0.5 * normal_1 + 0.866 * normal_2),
                config.anchor_radius * (-0.5 * normal_1 - 0.866 * normal_2),
            )
            anchor_indices = []
            for offset in anchor_offsets:
                anchor_index = add_node(centers[scaffold_index] + offset, anchor_role)
                anchor_indices.append(anchor_index)
                add_edge(connection_index, anchor_index)
            for source, target in ((0, 1), (1, 2), (2, 0)):
                add_edge(anchor_indices[source], anchor_indices[target])
        return connection_index

    for scaffold_index in range(centers.shape[0]):
        if (
            scaffold_index in {0, centers.shape[0] - 1}
            or degree[scaffold_index].item() != 2
        ):
            ensure_connection_node(scaffold_index)

    for assignment in assignments:
        if assignment.primitive_type not in {
            "rod",
            "rod_helix",
            "sheet",
            "sheet_helix",
            "truss",
        }:
            raise ValueError(
                "debugging path only supports the current primitive families"
            )
        chain_positions = _discretize_rod_chain(
            centers=centers[assignment.chain],
            target_edge_length=config.target_edge_length,
        )
        start_index = ensure_connection_node(assignment.chain[0])
        end_index = ensure_connection_node(assignment.chain[-1])
        if assignment.primitive_type == "rod":
            _materialize_single_rail(
                chain_positions=chain_positions,
                add_node=add_node,
                add_edge=add_edge,
                start_index=start_index,
                end_index=end_index,
            )
            continue

        if assignment.primitive_type == "rod_helix":
            helix_radius = config.helix_radius
            helical_chain = _discretize_rod_helix_chain(
                chain_positions=chain_positions,
                helix_radius=helix_radius,
                turns=config.helix_turns,
                phase=assignment.sweep_phase,
                target_edge_length=config.target_edge_length,
            )
            _materialize_single_rail(
                chain_positions=helical_chain,
                add_node=add_node,
                add_edge=add_edge,
                start_index=start_index,
                end_index=end_index,
            )
            continue

        if assignment.primitive_type == "truss":
            truss_target_edge_length = max(
                1e-6,
                config.target_edge_length * config.truss_target_edge_length_scale,
            )
            truss_num_points = max(
                chain_positions.shape[0],
                int(
                    math.ceil(
                        (chain_positions.shape[0] - 1)
                        * config.target_edge_length
                        / truss_target_edge_length
                    )
                )
                + 1,
            )
            chain_positions = _resample_polyline_by_spacing(
                chain_positions,
                target_edge_length=truss_target_edge_length,
                num_points=truss_num_points,
            )
            truss_positions = _build_truss_helix_positions(
                chain_positions=chain_positions,
                assignment=assignment,
            )
            _materialize_truss_helix(
                truss_positions=truss_positions,
                add_node=add_node,
                add_edge=add_edge,
                start_index=start_index,
                end_index=end_index,
            )
            continue

        if assignment.primitive_type == "sheet_helix":
            sheet_centerline, radial_axes = _build_sheet_helix_centerline(
                chain_positions=chain_positions,
                assignment=assignment,
                target_edge_length=config.target_edge_length,
                max_longitudinal_points=config.sheet_helix_max_longitudinal_points,
            )
        else:
            sheet_centerline = chain_positions
            radial_axes = None
        lateral_axes = _build_sheet_lateral_axes(
            chain_positions=sheet_centerline,
            assignment=assignment,
            radial_axes=radial_axes,
        )
        sheet_rows = _discretize_sheet_chain(
            chain_positions=sheet_centerline,
            lateral_axes=lateral_axes,
            sheet_width_distance=config.sheet_width_distance,
            sheet_width_nodes=assignment.sheet_width_nodes,
            helix_turns=0.0,
            helix_phase=0.0,
        )
        _materialize_sheet_lattice(
            rows=sheet_rows,
            add_node=add_node,
            add_edge=add_edge,
            start_index=start_index,
            end_index=end_index,
        )

    edge_index = torch.tensor(sorted(edges), dtype=torch.long)
    return (
        torch.stack(positions, dim=0),
        torch.tensor(roles, dtype=torch.long),
        edge_index,
    )


def _catmull_rom_point(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float,
) -> torch.Tensor:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _sample_catmull_rom_polyline(
    control_points: torch.Tensor,
    target_edge_length: float,
) -> torch.Tensor:
    if control_points.shape[0] <= 2:
        return control_points

    samples = [control_points[0]]
    for segment_index in range(control_points.shape[0] - 1):
        p0 = control_points[max(segment_index - 1, 0)]
        p1 = control_points[segment_index]
        p2 = control_points[segment_index + 1]
        p3 = control_points[min(segment_index + 2, control_points.shape[0] - 1)]
        segment_length = float(torch.linalg.vector_norm(p2 - p1).item())
        subdivisions = max(
            4, int(math.ceil(segment_length / max(target_edge_length, 1e-6))) * 6
        )
        for sample_index in range(1, subdivisions + 1):
            samples.append(
                _catmull_rom_point(
                    p0,
                    p1,
                    p2,
                    p3,
                    sample_index / subdivisions,
                )
            )
    return torch.stack(samples, dim=0)


def _resample_polyline_by_spacing(
    polyline: torch.Tensor,
    target_edge_length: float,
    num_points: int | None = None,
) -> torch.Tensor:
    if polyline.shape[0] <= 2:
        return polyline

    segment_vectors = polyline[1:] - polyline[:-1]
    segment_lengths = torch.linalg.vector_norm(segment_vectors, dim=1)
    cumulative = torch.cat(
        [torch.zeros(1, dtype=polyline.dtype), torch.cumsum(segment_lengths, dim=0)],
        dim=0,
    )
    total_length = float(cumulative[-1].item())
    if total_length < 1e-8:
        return polyline[[0, -1]]

    if num_points is None:
        num_segments = max(
            1, int(math.ceil(total_length / max(target_edge_length, 1e-6)))
        )
        num_points = num_segments + 1
    else:
        num_points = max(2, num_points)
    targets = torch.linspace(0.0, total_length, steps=num_points, dtype=polyline.dtype)
    resampled = []
    segment_index = 0
    for target in targets.tolist():
        while (
            segment_index < segment_lengths.shape[0] - 1
            and float(cumulative[segment_index + 1].item()) < target
        ):
            segment_index += 1
        start = polyline[segment_index]
        end = polyline[segment_index + 1]
        start_distance = float(cumulative[segment_index].item())
        end_distance = float(cumulative[segment_index + 1].item())
        if end_distance - start_distance < 1e-8:
            resampled.append(start)
            continue
        fraction = (target - start_distance) / (end_distance - start_distance)
        resampled.append((1.0 - fraction) * start + fraction * end)
    return torch.stack(resampled, dim=0)


def _discretize_rod_chain(
    centers: torch.Tensor,
    target_edge_length: float,
) -> torch.Tensor:
    # Rods should read as one-dimensional members that follow the scaffold
    # trajectory. We therefore interpolate a smooth centerline through the
    # scaffold control points and discretize it by arc length, instead of
    # inflating every scaffold node into a local volume.
    smoothed = _sample_catmull_rom_polyline(centers, target_edge_length)
    # The downstream pipeline is fully batched, so rod debugging still needs a
    # stable tensor shape across cases. We therefore resample every rod chain
    # to a fixed number of points derived from the fixed workspace span and the
    # requested target edge length. The curve geometry still comes from the
    # scaffold; only the sample count is normalized for batching.
    workspace_span = float(torch.linalg.vector_norm(centers[-1] - centers[0]).item())
    num_points = max(
        2, int(math.ceil(workspace_span / max(target_edge_length, 1e-6))) + 1
    )
    return _resample_polyline_by_spacing(
        smoothed,
        target_edge_length,
        num_points=num_points,
    )


def _materialize_single_rail(
    chain_positions: torch.Tensor,
    add_node,
    add_edge,
    start_index: int,
    end_index: int,
) -> None:
    previous_index = start_index
    for position in chain_positions[1:-1]:
        node_index = add_node(position, NodeRole.FREE)
        add_edge(previous_index, node_index)
        previous_index = node_index
    add_edge(previous_index, end_index)


def _estimate_polyline_tangent(
    polyline: torch.Tensor, sample_index: int
) -> torch.Tensor:
    if sample_index == 0:
        tangent = polyline[1] - polyline[0]
    elif sample_index == polyline.shape[0] - 1:
        tangent = polyline[-1] - polyline[-2]
    else:
        tangent = polyline[sample_index + 1] - polyline[sample_index - 1]
    if torch.linalg.vector_norm(tangent).item() < 1e-8:
        tangent = torch.tensor([1.0, 0.0, 0.0], dtype=polyline.dtype)
    return tangent


def _cumulative_arc_lengths(polyline: torch.Tensor) -> torch.Tensor:
    if polyline.shape[0] <= 1:
        return torch.zeros(
            (polyline.shape[0],), dtype=polyline.dtype, device=polyline.device
        )
    segment_lengths = torch.linalg.vector_norm(polyline[1:] - polyline[:-1], dim=1)
    return torch.cat(
        [
            torch.zeros((1,), dtype=polyline.dtype, device=polyline.device),
            torch.cumsum(segment_lengths, dim=0),
        ],
        dim=0,
    )


def _interpolate_scalar_controls(
    control_values: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if control_values.shape[0] == num_samples:
        return control_values
    if control_values.shape[0] == 1:
        return control_values.repeat(num_samples)

    control_positions = torch.linspace(
        0.0,
        1.0,
        steps=control_values.shape[0],
        dtype=control_values.dtype,
        device=control_values.device,
    )
    sample_positions = torch.linspace(
        0.0,
        1.0,
        steps=num_samples,
        dtype=control_values.dtype,
        device=control_values.device,
    )
    samples = []
    for sample_position in sample_positions.tolist():
        upper_index = 1
        while (
            upper_index < control_positions.shape[0] - 1
            and float(control_positions[upper_index].item()) < sample_position
        ):
            upper_index += 1
        lower_index = upper_index - 1
        lower_position = float(control_positions[lower_index].item())
        upper_position = float(control_positions[upper_index].item())
        if upper_position - lower_position < 1e-8:
            interpolated = control_values[lower_index]
        else:
            fraction = (sample_position - lower_position) / (
                upper_position - lower_position
            )
            interpolated = (1.0 - fraction) * control_values[
                lower_index
            ] + fraction * control_values[upper_index]
        samples.append(interpolated)
    return torch.stack(samples, dim=0)


def _discretize_sheet_chain(
    chain_positions: torch.Tensor,
    lateral_axes: torch.Tensor,
    sheet_width_distance: float,
    sheet_width_nodes: int,
    helix_turns: float,
    helix_phase: float,
) -> list[torch.Tensor]:
    if sheet_width_nodes < 2 or sheet_width_nodes > 6:
        raise ValueError("sheet_width_nodes must be between 2 and 6")
    if sheet_width_distance <= 0.0:
        raise ValueError("sheet_width_distance must be positive")

    rows = []
    sample_count = max(chain_positions.shape[0] - 1, 1)
    row_offsets = (
        torch.linspace(
            -0.5 * (sheet_width_nodes - 1),
            0.5 * (sheet_width_nodes - 1),
            steps=sheet_width_nodes,
            dtype=chain_positions.dtype,
            device=chain_positions.device,
        )
        * sheet_width_distance
    )
    for row_index, row_offset in enumerate(row_offsets.tolist()):
        if row_index % 2 == 0:
            row_centers = chain_positions
            row_axes = lateral_axes
        else:
            row_centers = 0.5 * (chain_positions[:-1] + chain_positions[1:])
            row_axes = lateral_axes[:-1] + lateral_axes[1:]
            row_axes = row_axes / torch.linalg.vector_norm(
                row_axes, dim=1, keepdim=True
            ).clamp_min(1e-8)

        if helix_turns != 0.0:
            rotated_axes = []
            row_sample_count = max(row_centers.shape[0] - 1, 1)
            for sample_index in range(row_centers.shape[0]):
                tangent = _estimate_polyline_tangent(row_centers, sample_index)
                _, normal_1, normal_2 = _orthonormal_frame(tangent)
                base_axis = row_axes[sample_index]
                base_angle = math.atan2(
                    float(torch.dot(base_axis, normal_2).item()),
                    float(torch.dot(base_axis, normal_1).item()),
                )
                angle = (
                    base_angle
                    + helix_phase
                    + 2.0 * math.pi * helix_turns * (sample_index / row_sample_count)
                )
                rotated_axes.append(
                    math.cos(angle) * normal_1 + math.sin(angle) * normal_2
                )
            row_axes = torch.stack(rotated_axes, dim=0)

        rows.append(row_centers + row_offset * row_axes)
    return rows


def _interpolate_control_vectors(
    control_vectors: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if control_vectors.shape[0] == num_samples:
        return control_vectors
    if control_vectors.shape[0] == 1:
        return control_vectors.repeat(num_samples, 1)

    control_positions = torch.linspace(
        0.0,
        1.0,
        steps=control_vectors.shape[0],
        dtype=control_vectors.dtype,
        device=control_vectors.device,
    )
    sample_positions = torch.linspace(
        0.0,
        1.0,
        steps=num_samples,
        dtype=control_vectors.dtype,
        device=control_vectors.device,
    )
    samples = []
    for sample_position in sample_positions.tolist():
        upper_index = 1
        while (
            upper_index < control_positions.shape[0] - 1
            and float(control_positions[upper_index].item()) < sample_position
        ):
            upper_index += 1
        lower_index = upper_index - 1
        lower_position = float(control_positions[lower_index].item())
        upper_position = float(control_positions[upper_index].item())
        if upper_position - lower_position < 1e-8:
            interpolated = control_vectors[lower_index]
        else:
            fraction = (sample_position - lower_position) / (
                upper_position - lower_position
            )
            interpolated = (1.0 - fraction) * control_vectors[
                lower_index
            ] + fraction * control_vectors[upper_index]
        samples.append(interpolated)
    stacked = torch.stack(samples, dim=0)
    return stacked / torch.linalg.vector_norm(stacked, dim=1, keepdim=True).clamp_min(
        1e-8
    )


def _build_sheet_helix_centerline(
    chain_positions: torch.Tensor,
    assignment: ChainPrimitiveAssignment,
    target_edge_length: float,
    max_longitudinal_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    control_offset_distances = torch.tensor(
        assignment.offset_distances,
        dtype=chain_positions.dtype,
        device=chain_positions.device,
    )
    base_arc_lengths = _cumulative_arc_lengths(chain_positions)
    base_arc_length = float(base_arc_lengths[-1].item())
    effective_radius = float(control_offset_distances.abs().max().item())
    total_turns = base_arc_length / max(assignment.helix_pitch, 1e-6)
    helical_circumference_travel = abs(total_turns) * 2.0 * math.pi * effective_radius
    helical_length = math.sqrt(base_arc_length**2 + helical_circumference_travel**2)
    num_points = max(
        chain_positions.shape[0],
        int(math.ceil(helical_length / max(target_edge_length, 1e-6))) + 1,
    )
    num_points = min(num_points, max_longitudinal_points)
    helix_support = _resample_polyline_by_spacing(
        chain_positions,
        target_edge_length=target_edge_length,
        num_points=num_points,
    )
    offset_distances = _interpolate_scalar_controls(
        control_offset_distances.to(
            dtype=helix_support.dtype, device=helix_support.device
        ),
        helix_support.shape[0],
    )
    arc_lengths = _cumulative_arc_lengths(helix_support)
    centers = []
    radial_axes = []
    for sample_index in range(helix_support.shape[0]):
        tangent = _estimate_polyline_tangent(helix_support, sample_index)
        _, normal_1, normal_2 = _orthonormal_frame(tangent)
        angle = assignment.helix_phase + 2.0 * math.pi * float(
            arc_lengths[sample_index].item()
        ) / max(assignment.helix_pitch, 1e-6)
        radial_axis = math.cos(angle) * normal_1 + math.sin(angle) * normal_2
        centers.append(
            helix_support[sample_index] + offset_distances[sample_index] * radial_axis
        )
        radial_axes.append(radial_axis)
    return torch.stack(centers, dim=0), torch.stack(radial_axes, dim=0)


def _build_sheet_lateral_axes(
    chain_positions: torch.Tensor,
    assignment: ChainPrimitiveAssignment,
    radial_axes: torch.Tensor | None = None,
) -> torch.Tensor:
    orientation_angles = _interpolate_scalar_controls(
        torch.tensor(
            assignment.sheet_orientations,
            dtype=chain_positions.dtype,
            device=chain_positions.device,
        ),
        chain_positions.shape[0],
    )
    lateral_axes = []
    for sample_index in range(chain_positions.shape[0]):
        tangent = _estimate_polyline_tangent(chain_positions, sample_index)
        _, normal_1, normal_2 = _orthonormal_frame(tangent)
        if radial_axes is None:
            lateral_axis = (
                math.cos(float(orientation_angles[sample_index].item())) * normal_1
                + math.sin(float(orientation_angles[sample_index].item())) * normal_2
            )
        else:
            radial_axis = radial_axes[sample_index]
            circumferential_axis = torch.linalg.cross(radial_axis, tangent)
            if torch.linalg.vector_norm(circumferential_axis).item() < 1e-8:
                circumferential_axis = normal_1
            circumferential_axis = circumferential_axis / torch.linalg.vector_norm(
                circumferential_axis
            ).clamp_min(1e-8)
            orientation = float(orientation_angles[sample_index].item())
            lateral_axis = (
                math.cos(orientation) * circumferential_axis
                + math.sin(orientation) * radial_axis
            )
        lateral_axes.append(
            lateral_axis / torch.linalg.vector_norm(lateral_axis).clamp_min(1e-8)
        )
    return torch.stack(lateral_axes, dim=0)


def _build_truss_helix_positions(
    chain_positions: torch.Tensor,
    assignment: ChainPrimitiveAssignment,
) -> torch.Tensor:
    positions = []
    for sample_index in range(chain_positions.shape[0]):
        tangent = _estimate_polyline_tangent(chain_positions, sample_index)
        _, normal_1, normal_2 = _orthonormal_frame(tangent)
        fraction = sample_index / max(chain_positions.shape[0] - 1, 1)
        width = (
            1.0 - fraction
        ) * assignment.width_start + fraction * assignment.width_end
        thickness = (
            1.0 - fraction
        ) * assignment.thickness_start + fraction * assignment.thickness_end
        twist = (
            1.0 - fraction
        ) * assignment.twist_start + fraction * assignment.twist_end
        base_angle = assignment.sweep_phase + twist * math.pi
        triangle_phase = base_angle + (2.0 * math.pi / 3.0) * sample_index
        center = chain_positions[sample_index]
        radius = max(2.2 * max(width, thickness), 0.02)
        offset = radius * (
            math.cos(triangle_phase) * normal_1 + math.sin(triangle_phase) * normal_2
        )
        positions.append(center + offset)
    return torch.stack(positions, dim=0)


def _materialize_truss_helix(
    truss_positions: torch.Tensor,
    add_node,
    add_edge,
    start_index: int,
    end_index: int,
) -> None:
    if truss_positions.shape[0] <= 2:
        add_edge(start_index, end_index)
        return

    node_indices = [
        add_node(position, NodeRole.FREE) for position in truss_positions[1:-1]
    ]
    if not node_indices:
        add_edge(start_index, end_index)
        return

    ordered_indices = [start_index, *node_indices, end_index]
    for current_index in range(1, len(ordered_indices)):
        current_node = ordered_indices[current_index]
        for offset in (1, 2, 3):
            previous_index = current_index - offset
            if previous_index < 0:
                continue
            add_edge(current_node, ordered_indices[previous_index])


def _discretize_rod_helix_chain(
    chain_positions: torch.Tensor,
    helix_radius: float,
    turns: float,
    phase: float,
    target_edge_length: float,
) -> torch.Tensor:
    centerline_span = float(
        torch.linalg.vector_norm(chain_positions[-1] - chain_positions[0]).item()
    )
    helix_circumference_travel = abs(turns) * 2.0 * math.pi * helix_radius
    helical_length = math.sqrt(centerline_span**2 + helix_circumference_travel**2)
    # Helices need denser sampling than their centerline, but the offline
    # dataset still needs fixed tensor shapes across cases. We therefore
    # normalize the sample count against the expected helix geometry for the
    # whole family rather than the exact per-case arc length.
    num_points = max(
        2, int(math.ceil(helical_length / max(1e-6, target_edge_length))) + 1
    )
    helix_support = _resample_polyline_by_spacing(
        chain_positions,
        target_edge_length=target_edge_length,
        num_points=num_points,
    )

    helical_positions = []
    sample_count = max(helix_support.shape[0] - 1, 1)
    for sample_index in range(helix_support.shape[0]):
        tangent = _estimate_polyline_tangent(helix_support, sample_index)
        _, normal_1, normal_2 = _orthonormal_frame(tangent)
        angle = phase + 2.0 * math.pi * turns * (sample_index / sample_count)
        offset = helix_radius * (
            math.cos(angle) * normal_1 + math.sin(angle) * normal_2
        )
        helical_positions.append(helix_support[sample_index] + offset)
    return torch.stack(helical_positions, dim=0)


def _materialize_sheet_lattice(
    rows: list[torch.Tensor],
    add_node,
    add_edge,
    start_index: int,
    end_index: int,
) -> None:
    row_node_indices: list[list[int]] = []
    for row_index, row in enumerate(rows):
        if row_index % 2 == 0:
            node_indices = [start_index]
            previous_index = start_index
            for position in row[1:-1]:
                node_index = add_node(position, NodeRole.FREE)
                node_indices.append(node_index)
                add_edge(previous_index, node_index)
                previous_index = node_index
            add_edge(previous_index, end_index)
            node_indices.append(end_index)
        else:
            node_indices = []
            previous_index = None
            for position in row:
                node_index = add_node(position, NodeRole.FREE)
                node_indices.append(node_index)
                if previous_index is not None:
                    add_edge(previous_index, node_index)
                previous_index = node_index
        row_node_indices.append(node_indices)

    for upper_row, lower_row in zip(row_node_indices[:-1], row_node_indices[1:]):
        if len(upper_row) == len(lower_row) + 1:
            longer_row = upper_row
            shorter_row = lower_row
        elif len(lower_row) == len(upper_row) + 1:
            longer_row = lower_row
            shorter_row = upper_row
        else:
            raise ValueError("sheet rows must alternate between N and N-1 nodes")
        for node_index, shorter_node in enumerate(shorter_row):
            add_edge(shorter_node, longer_row[node_index])
            add_edge(shorter_node, longer_row[node_index + 1])


def _edge_index_to_adjacency(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if edge_index.numel() == 0:
        return adjacency
    adjacency[edge_index[:, 0], edge_index[:, 1]] = 1.0
    adjacency[edge_index[:, 1], edge_index[:, 0]] = 1.0
    return adjacency


def _build_scaffold_primitive_types(
    scaffold_adjacency: torch.Tensor,
    assignments: list[ChainPrimitiveAssignment],
) -> torch.Tensor:
    edge_primitive_types = -torch.ones(scaffold_adjacency.shape, dtype=torch.long)
    primitive_to_index = {
        primitive_type: index
        for index, primitive_type in enumerate(CHAIN_PRIMITIVE_LIBRARY)
    }

    for assignment in assignments:
        primitive_index = primitive_to_index[assignment.primitive_type]
        for source, target in zip(assignment.chain[:-1], assignment.chain[1:]):
            if scaffold_adjacency[source, target] <= 0.0:
                continue
            edge_primitive_types[source, target] = primitive_index
            edge_primitive_types[target, source] = primitive_index

    return edge_primitive_types


def _build_scaffold_roles(num_nodes: int) -> torch.Tensor:
    roles = torch.full((num_nodes,), int(NodeRole.FREE), dtype=torch.long)
    roles[0] = int(NodeRole.FIXED)
    roles[-1] = int(NodeRole.MOBILE)
    return roles


def _sample_primitive_case(
    config: PrimitiveConfig,
    seed: int | None,
) -> tuple[Structures, Scaffolds]:
    rng = random.Random(seed)

    # The generator is intentionally split into a sparse graph stage and a
    # materialization stage. This keeps the high-level combinatorics separate
    # from the final FEM mesh and leaves a clean place for a future two-stage
    # optimizer: first over the scaffold/primitive parameters, then over the
    # final expanded mesh.
    scaffold_centers = _sample_random_scaffold_centers(config, rng)
    scaffold_adjacency = _sample_scaffold_connectivity(scaffold_centers, config, rng)
    primitive_chains = _extract_primitive_chains(scaffold_adjacency)
    primitive_assignments = _sample_chain_primitives(primitive_chains, config, rng)
    styled_centers = _apply_chain_style_offsets(
        centers=scaffold_centers,
        adjacency=scaffold_adjacency,
        assignments=primitive_assignments,
        config=config,
    )
    positions, roles, edge_index = _materialize_scaffold_node_triplets(
        centers=styled_centers,
        adjacency=scaffold_adjacency,
        assignments=primitive_assignments,
        config=config,
    )
    adjacency = _edge_index_to_adjacency(positions.shape[0], edge_index)

    design = Structures(
        positions=positions.unsqueeze(0).clamp(0.02, 0.98),
        roles=roles.unsqueeze(0),
        adjacency=symmetrize_matrix(adjacency).unsqueeze(0),
    )
    scaffold = Scaffolds(
        positions=styled_centers.unsqueeze(0).clamp(0.02, 0.98),
        roles=_build_scaffold_roles(styled_centers.shape[0]).unsqueeze(0),
        adjacency=symmetrize_matrix(scaffold_adjacency).unsqueeze(0),
        edge_primitive_types=_build_scaffold_primitive_types(
            scaffold_adjacency,
            primitive_assignments,
        ).unsqueeze(0),
    )
    design.validate()
    scaffold.validate()
    return design, scaffold


def sample_primitive_design(
    config: PrimitiveConfig | None = None,
    seed: int | None = None,
) -> Structures:
    config = config or PrimitiveConfig()
    design, _ = _sample_primitive_case(config, seed)
    return design


def sample_random_primitive(
    config: PrimitiveConfig | None = None,
    seed: int | None = None,
) -> tuple[Structures, Scaffolds]:
    return _sample_primitive_case(config or PrimitiveConfig(), seed)
