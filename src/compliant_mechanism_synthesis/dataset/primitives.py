from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
import torch.nn.functional as F

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


@dataclass(frozen=True)
class PrimitiveConfig:
    # This controls the sparse scaffolding graph, not the final number of free
    # FEM nodes. We keep the default sparse on purpose during primitive
    # debugging so the scaffold-to-mesh conversion remains visually readable.
    num_free_nodes: int = 12
    width: float = 0.18
    thickness: float = 0.10
    anchor_radius: float = 0.04
    # The final mesh primitives should read as thin composable parts rather
    # than a few oversized blobs. Using a 0.04 workspace-wide diameter keeps
    # the primitive family legible once many of them are composed together.
    primitive_radius: float = 0.015
    neighbor_count: int = 2
    extra_connection_probability: float = 0.10
    connection_length_scale: float = 0.22
    max_scaffold_degree: int = 4
    long_range_connection_probability: float = 0.10
    long_range_min_index_gap: int = 4
    free_z_min: float = 0.28
    free_z_max: float = 0.72
    fixed_anchor_z: float = 0.10
    mobile_anchor_z: float = 0.90
    target_edge_length: float = 0.05
    helix_radius: float = 0.06
    helix_turns: float = 4.0
    truss_target_edge_length_scale: float = 0.33
    sheet_width_nodes: int = 4
    sheet_width_distance: float = 0.02
    sample_sheet_helix_width_nodes: bool = True
    sheet_helix_width_nodes_min: int = 2
    sheet_helix_width_nodes_max: int = 4
    sheet_helix_offset_distance_min: float = 0.06
    sheet_helix_offset_distance_max: float = 0.10
    sheet_helix_pitch_distance: float = 0.24
    sheet_helix_max_longitudinal_points: int = 128


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


def _try_add_scaffold_edge(
    adjacency: torch.Tensor,
    source: int,
    target: int,
    max_degree: int,
) -> bool:
    if source == target or adjacency[source, target] > 0.0:
        return False
    if int(adjacency[source].sum().item()) >= max_degree:
        return False
    if int(adjacency[target].sum().item()) >= max_degree:
        return False
    adjacency[source, target] = 1.0
    adjacency[target, source] = 1.0
    return True


def _default_extra_scaffold_edge_count(num_nodes: int) -> int:
    if num_nodes <= 2:
        return 0
    return max(2, (num_nodes - 2) // 4)


def _default_scaffold_primitive_count(num_nodes: int) -> int:
    if num_nodes <= 1:
        return 0
    if num_nodes == 2:
        return 1
    return 3 * _default_extra_scaffold_edge_count(num_nodes) - 1


def _sample_scaffold_connectivity(
    centers: torch.Tensor,
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    num_nodes = centers.shape[0]
    if config.max_scaffold_degree < 2:
        raise ValueError("max_scaffold_degree must be at least 2")
    if config.long_range_min_index_gap < 2:
        raise ValueError("long_range_min_index_gap must be at least 2")

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    pairwise = torch.linalg.vector_norm(
        centers[:, None, :] - centers[None, :, :], dim=-1
    )

    def sample_candidate(
        long_range: bool,
        required_node: int | None = None,
    ) -> tuple[int, int] | None:
        candidates: list[tuple[int, int]] = []
        weights: list[float] = []
        for source in range(num_nodes - 1):
            if int(adjacency[source].sum().item()) >= config.max_scaffold_degree:
                continue
            for target in range(source + 1, num_nodes):
                if required_node is not None and required_node not in {source, target}:
                    continue
                if adjacency[source, target] > 0.0:
                    continue
                if abs(target - source) <= 1:
                    continue
                if required_node is None:
                    if int(adjacency[source].sum().item()) != 2:
                        continue
                    if int(adjacency[target].sum().item()) != 2:
                        continue
                if int(adjacency[target].sum().item()) >= config.max_scaffold_degree:
                    continue
                index_gap = abs(target - source)
                if long_range and index_gap < config.long_range_min_index_gap:
                    continue
                distance = float(pairwise[source, target].item())
                weight = (
                    distance * index_gap
                    if long_range
                    else _localized_edge_probability(distance, config)
                    * max(index_gap - 1, 1)
                )
                candidates.append((source, target))
                weights.append(weight)
        if not candidates:
            return None
        return rng.choices(candidates, weights=weights, k=1)[0]

    for source in range(num_nodes - 1):
        adjacency[source, source + 1] = 1.0
        adjacency[source + 1, source] = 1.0

    extra_edges_to_add = _default_extra_scaffold_edge_count(num_nodes)
    extra_edges_added = 0
    for endpoint in (0, num_nodes - 1):
        if (
            extra_edges_added >= extra_edges_to_add
            or int(adjacency[endpoint].sum().item()) >= 2
        ):
            continue
        candidate = sample_candidate(long_range=False, required_node=endpoint)
        if candidate is None:
            candidate = sample_candidate(long_range=True, required_node=endpoint)
        if candidate is None:
            raise ValueError(
                "unable to connect scaffold endpoint within the degree cap"
            )
        source, target = candidate
        if not _try_add_scaffold_edge(
            adjacency,
            source,
            target,
            config.max_scaffold_degree,
        ):
            raise ValueError("sampled scaffold endpoint edge violates degree cap")
        extra_edges_added += 1

    for _ in range(extra_edges_added, extra_edges_to_add):
        prefer_long_range = rng.random() < config.long_range_connection_probability
        candidate = sample_candidate(long_range=prefer_long_range)
        if candidate is None and prefer_long_range:
            candidate = sample_candidate(long_range=False)
        if candidate is None:
            raise ValueError("unable to sample scaffold edges within the degree cap")
        source, target = candidate
        if not _try_add_scaffold_edge(
            adjacency,
            source,
            target,
            config.max_scaffold_degree,
        ):
            raise ValueError("sampled scaffold edge violates degree cap")

    return adjacency


def _intertwined_scaffold_edges(num_nodes: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for source in range(num_nodes - 1):
        edges.append((source, source + 1))
    for source in range(0, num_nodes - 2, 3):
        edges.append((source, source + 2))
    return edges


def _sample_chain_primitives(
    chains: list[list[int]],
    config: PrimitiveConfig,
    rng: random.Random,
) -> list[ChainPrimitiveAssignment]:
    assignments = []

    def coerce_primitive_type(chain: list[int], primitive_type: str) -> str:
        if len(chain) <= 3:
            return primitive_type
        if primitive_type == "rod_helix":
            return "rod"
        if primitive_type == "sheet_helix":
            return "sheet"
        return primitive_type

    primitive_types = [
        coerce_primitive_type(chain, rng.choice(CHAIN_PRIMITIVE_LIBRARY))
        for chain in chains
    ]

    for chain, primitive_type in zip(chains, primitive_types):
        width_scale = rng.uniform(0.8, 1.15)
        thickness_scale = rng.uniform(0.45, 0.85)
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


def _extract_primitive_segments(adjacency: torch.Tensor) -> list[list[int]]:
    num_nodes = adjacency.shape[0]
    degrees = adjacency.sum(dim=1).to(dtype=torch.long)
    terminals = {
        node_index
        for node_index in range(num_nodes)
        if node_index in {0, num_nodes - 1} or int(degrees[node_index].item()) != 2
    }
    visited_edges: set[tuple[int, int]] = set()
    segments: list[list[int]] = []

    def edge_key(source: int, target: int) -> tuple[int, int]:
        return tuple(sorted((source, target)))

    def neighbors(node_index: int) -> list[int]:
        return (
            torch.nonzero(adjacency[node_index] > 0.0, as_tuple=False)
            .flatten()
            .tolist()
        )

    def trace_chain(start: int, next_node: int) -> list[int]:
        chain = [start, next_node]
        visited_edges.add(edge_key(start, next_node))
        previous = start
        current = next_node
        while current not in terminals:
            current_neighbors = [
                neighbor for neighbor in neighbors(current) if neighbor != previous
            ]
            if not current_neighbors:
                break
            previous, current = current, current_neighbors[0]
            chain.append(current)
            visited_edges.add(edge_key(previous, current))
        return chain

    for start in sorted(terminals):
        for next_node in sorted(neighbors(start)):
            if edge_key(start, next_node) in visited_edges:
                continue
            segments.append(trace_chain(start, next_node))

    for source in range(num_nodes):
        for target in neighbors(source):
            if source >= target or edge_key(source, target) in visited_edges:
                continue
            segments.append(trace_chain(source, target))

    return segments


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
    offset_sum = torch.zeros_like(centers)
    offset_count = torch.zeros(
        (centers.shape[0], 1),
        dtype=centers.dtype,
        device=centers.device,
    )

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

            selector = F.one_hot(
                torch.tensor(node_index, device=centers.device),
                num_classes=centers.shape[0],
            ).to(dtype=centers.dtype)
            offset_sum = offset_sum + selector[:, None] * offset[None, :]
            offset_count = offset_count + selector[:, None]

    safe_count = offset_count.clamp_min(1.0)
    adjusted = centers + offset_sum / safe_count
    valid = (offset_count > 0.0).to(dtype=centers.dtype)
    adjusted = valid * adjusted + (1.0 - valid) * centers
    adjusted_z = adjusted[:, 2].clamp(config.free_z_min, config.free_z_max)
    adjusted_z = torch.cat(
        [
            adjusted_z[:1] * 0.0 + config.fixed_anchor_z,
            adjusted_z[1:-1],
            adjusted_z[-1:] * 0.0 + config.mobile_anchor_z,
        ],
        dim=0,
    )
    adjusted = torch.cat([adjusted[:, :2], adjusted_z.unsqueeze(-1)], dim=1)
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
        segment_num_points = _segment_num_points(assignment.primitive_type, config)
        chain_positions = _discretize_rod_chain(
            centers=centers[assignment.chain],
            target_edge_length=config.target_edge_length,
            num_points=segment_num_points,
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
                num_points=segment_num_points,
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
                num_points=segment_num_points,
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

    edge_index = torch.tensor(sorted(edges), dtype=torch.long, device=centers.device)
    return (
        torch.stack(positions, dim=0),
        torch.tensor(roles, dtype=torch.long, device=centers.device),
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
    if polyline.shape[0] <= 1:
        return polyline

    segment_vectors = polyline[1:] - polyline[:-1]
    segment_lengths = torch.linalg.vector_norm(segment_vectors, dim=1)
    cumulative = torch.cat(
        [
            torch.zeros(1, dtype=polyline.dtype, device=polyline.device),
            torch.cumsum(segment_lengths, dim=0),
        ],
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
    targets = torch.linspace(
        0.0,
        total_length,
        steps=num_points,
        dtype=polyline.dtype,
        device=polyline.device,
    )
    segment_index = torch.searchsorted(cumulative[1:], targets, right=False)
    segment_index = segment_index.clamp_max(segment_lengths.shape[0] - 1)

    start = polyline.index_select(0, segment_index)
    end = polyline.index_select(0, segment_index + 1)
    start_distance = cumulative.index_select(0, segment_index)
    end_distance = cumulative.index_select(0, segment_index + 1)
    safe_denominator = (end_distance - start_distance).clamp_min(1e-8)
    fraction = ((targets - start_distance) / safe_denominator).unsqueeze(-1)
    return torch.lerp(start, end, fraction)


def _discretize_rod_chain(
    centers: torch.Tensor,
    target_edge_length: float,
    num_points: int | None = None,
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
    if num_points is None:
        workspace_span = float(
            torch.linalg.vector_norm(centers[-1] - centers[0]).item()
        )
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


def _segment_num_points(
    primitive_type: str,
    config: PrimitiveConfig,
) -> int:
    nominal_segment_span = 0.76 / max(config.num_free_nodes + 1, 1)
    if primitive_type == "rod":
        effective_length = nominal_segment_span
    elif primitive_type == "rod_helix":
        effective_length = math.sqrt(
            nominal_segment_span**2
            + (2.0 * math.pi * config.helix_radius * abs(config.helix_turns)) ** 2
        )
    elif primitive_type == "sheet":
        effective_length = max(nominal_segment_span, 2.5 * config.sheet_width_distance)
    elif primitive_type == "sheet_helix":
        effective_radius = 0.5 * (
            config.sheet_helix_offset_distance_min
            + config.sheet_helix_offset_distance_max
        )
        turns = nominal_segment_span / max(config.sheet_helix_pitch_distance, 1e-6)
        effective_length = math.sqrt(
            max(nominal_segment_span, 2.5 * config.sheet_width_distance) ** 2
            + (2.0 * math.pi * effective_radius * abs(turns)) ** 2
        )
    elif primitive_type == "truss":
        effective_length = nominal_segment_span / max(
            config.truss_target_edge_length_scale,
            1e-6,
        )
    else:
        raise ValueError(f"unknown primitive type: {primitive_type}")

    minimum_points = 2
    if primitive_type in {"sheet", "sheet_helix"}:
        minimum_points = 4
    if primitive_type == "truss":
        minimum_points = 5
    return max(
        minimum_points,
        int(math.ceil(effective_length / max(config.target_edge_length, 1e-6))) + 1,
    )


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
    upper_index = (
        torch.searchsorted(control_positions[1:], sample_positions, right=False) + 1
    )
    upper_index = upper_index.clamp_max(control_positions.shape[0] - 1)
    lower_index = upper_index - 1
    lower_position = control_positions.index_select(0, lower_index)
    upper_position = control_positions.index_select(0, upper_index)
    lower_values = control_values.index_select(0, lower_index)
    upper_values = control_values.index_select(0, upper_index)
    safe_denominator = (upper_position - lower_position).clamp_min(1e-8)
    fraction = (sample_positions - lower_position) / safe_denominator
    return torch.lerp(lower_values, upper_values, fraction)


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
    upper_index = (
        torch.searchsorted(control_positions[1:], sample_positions, right=False) + 1
    )
    upper_index = upper_index.clamp_max(control_positions.shape[0] - 1)
    lower_index = upper_index - 1
    lower_position = control_positions.index_select(0, lower_index)
    upper_position = control_positions.index_select(0, upper_index)
    lower_vectors = control_vectors.index_select(0, lower_index)
    upper_vectors = control_vectors.index_select(0, upper_index)
    safe_denominator = (upper_position - lower_position).clamp_min(1e-8)
    fraction = ((sample_positions - lower_position) / safe_denominator).unsqueeze(-1)
    stacked = torch.lerp(lower_vectors, upper_vectors, fraction)
    return stacked / torch.linalg.vector_norm(stacked, dim=1, keepdim=True).clamp_min(
        1e-8
    )


def _build_sheet_helix_centerline(
    chain_positions: torch.Tensor,
    assignment: ChainPrimitiveAssignment,
    target_edge_length: float,
    max_longitudinal_points: int,
    num_points: int | None = None,
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
    if num_points is None:
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
    num_points: int | None = None,
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
    if num_points is None:
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
    adjacency = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device
    )
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


def _build_scaffold_primitive_ids(
    scaffold_adjacency: torch.Tensor,
    assignments: list[ChainPrimitiveAssignment],
) -> torch.Tensor:
    edge_primitive_ids = -torch.ones(scaffold_adjacency.shape, dtype=torch.long)
    for primitive_id, assignment in enumerate(assignments):
        for source, target in zip(assignment.chain[:-1], assignment.chain[1:]):
            if scaffold_adjacency[source, target] <= 0.0:
                continue
            edge_primitive_ids[source, target] = primitive_id
            edge_primitive_ids[target, source] = primitive_id
    return edge_primitive_ids


def _build_scaffold_roles(num_nodes: int) -> torch.Tensor:
    roles = torch.full((num_nodes,), int(NodeRole.FREE), dtype=torch.long)
    roles[0] = int(NodeRole.FIXED)
    roles[-1] = int(NodeRole.MOBILE)
    return roles


def _build_scaffold_assignment_tensors(
    scaffold_adjacency: torch.Tensor,
    assignments: list[ChainPrimitiveAssignment],
) -> dict[str, torch.Tensor]:
    shape = scaffold_adjacency.shape
    float_params = {
        "edge_orientation_start": torch.zeros(shape, dtype=torch.float32),
        "edge_orientation_end": torch.zeros(shape, dtype=torch.float32),
        "edge_offset_start": torch.zeros(shape, dtype=torch.float32),
        "edge_offset_end": torch.zeros(shape, dtype=torch.float32),
        "edge_helix_phase": torch.zeros(shape, dtype=torch.float32),
        "edge_helix_pitch": torch.zeros(shape, dtype=torch.float32),
        "edge_width_start": torch.zeros(shape, dtype=torch.float32),
        "edge_width_end": torch.zeros(shape, dtype=torch.float32),
        "edge_thickness_start": torch.zeros(shape, dtype=torch.float32),
        "edge_thickness_end": torch.zeros(shape, dtype=torch.float32),
        "edge_twist_start": torch.zeros(shape, dtype=torch.float32),
        "edge_twist_end": torch.zeros(shape, dtype=torch.float32),
        "edge_sweep_phase": torch.zeros(shape, dtype=torch.float32),
    }
    int_params = {
        "edge_sheet_width_nodes": -torch.ones(shape, dtype=torch.long),
    }
    for assignment in assignments:
        for local_index, (source, target) in enumerate(
            zip(assignment.chain[:-1], assignment.chain[1:])
        ):
            int_params["edge_sheet_width_nodes"][source, target] = (
                assignment.sheet_width_nodes
            )
            int_params["edge_sheet_width_nodes"][target, source] = (
                assignment.sheet_width_nodes
            )
            float_params["edge_orientation_start"][source, target] = (
                assignment.sheet_orientations[local_index]
            )
            float_params["edge_orientation_end"][source, target] = (
                assignment.sheet_orientations[local_index + 1]
            )
            float_params["edge_orientation_start"][target, source] = (
                assignment.sheet_orientations[local_index + 1]
            )
            float_params["edge_orientation_end"][target, source] = (
                assignment.sheet_orientations[local_index]
            )
            float_params["edge_offset_start"][source, target] = (
                assignment.offset_distances[local_index]
            )
            float_params["edge_offset_end"][source, target] = (
                assignment.offset_distances[local_index + 1]
            )
            float_params["edge_offset_start"][target, source] = (
                assignment.offset_distances[local_index + 1]
            )
            float_params["edge_offset_end"][target, source] = (
                assignment.offset_distances[local_index]
            )
            float_params["edge_helix_phase"][source, target] = assignment.helix_phase
            float_params["edge_helix_phase"][target, source] = assignment.helix_phase
            float_params["edge_helix_pitch"][source, target] = assignment.helix_pitch
            float_params["edge_helix_pitch"][target, source] = assignment.helix_pitch
            float_params["edge_width_start"][source, target] = assignment.width_start
            float_params["edge_width_end"][source, target] = assignment.width_end
            float_params["edge_width_start"][target, source] = assignment.width_end
            float_params["edge_width_end"][target, source] = assignment.width_start
            float_params["edge_thickness_start"][source, target] = (
                assignment.thickness_start
            )
            float_params["edge_thickness_end"][source, target] = (
                assignment.thickness_end
            )
            float_params["edge_thickness_start"][target, source] = (
                assignment.thickness_end
            )
            float_params["edge_thickness_end"][target, source] = (
                assignment.thickness_start
            )
            float_params["edge_twist_start"][source, target] = assignment.twist_start
            float_params["edge_twist_end"][source, target] = assignment.twist_end
            float_params["edge_twist_start"][target, source] = assignment.twist_end
            float_params["edge_twist_end"][target, source] = assignment.twist_start
            float_params["edge_sweep_phase"][source, target] = assignment.sweep_phase
            float_params["edge_sweep_phase"][target, source] = assignment.sweep_phase
    return {**int_params, **float_params}


def _ordered_chain_from_primitive_edges(
    adjacency: torch.Tensor,
    primitive_ids: torch.Tensor,
    primitive_id: int,
) -> list[int]:
    nodes: set[int] = set()
    primitive_neighbors: dict[int, list[int]] = {}
    for source in range(adjacency.shape[0]):
        for target in range(source + 1, adjacency.shape[1]):
            if adjacency[source, target] <= 0.0:
                continue
            if int(primitive_ids[source, target].item()) != primitive_id:
                continue
            nodes.update((source, target))
            primitive_neighbors.setdefault(source, []).append(target)
            primitive_neighbors.setdefault(target, []).append(source)
    if not nodes:
        raise ValueError("primitive id does not map to any scaffold edges")
    endpoints = sorted(
        node_index
        for node_index, node_neighbors in primitive_neighbors.items()
        if len(node_neighbors) == 1
    )
    start = endpoints[0] if endpoints else min(nodes)
    chain = [start]
    previous = -1
    current = start
    while True:
        candidates = sorted(
            node_index
            for node_index in primitive_neighbors[current]
            if node_index != previous
        )
        if not candidates:
            break
        next_node = candidates[0]
        chain.append(next_node)
        previous, current = current, next_node
        if endpoints and current == endpoints[-1]:
            break
    return chain


def _assignments_from_scaffold(
    scaffold: Scaffolds,
    batch_index: int,
    config: PrimitiveConfig,
) -> list[ChainPrimitiveAssignment]:
    adjacency = scaffold.adjacency[batch_index]
    primitive_ids = scaffold.edge_primitive_ids[batch_index]
    primitive_types = scaffold.edge_primitive_types[batch_index]
    primitive_names = list(CHAIN_PRIMITIVE_LIBRARY)
    assignments: list[ChainPrimitiveAssignment] = []
    used_ids = sorted(
        {
            int(primitive_ids[source, target].item())
            for source in range(adjacency.shape[0])
            for target in range(source + 1, adjacency.shape[1])
            if adjacency[source, target] > 0.0
        }
    )
    for primitive_id in used_ids:
        chain = _ordered_chain_from_primitive_edges(
            adjacency, primitive_ids, primitive_id
        )
        edge_pairs = list(zip(chain[:-1], chain[1:]))
        first_source, first_target = edge_pairs[0]
        last_source, last_target = edge_pairs[-1]
        primitive_index = int(primitive_types[first_source, first_target].item())
        primitive_type = primitive_names[primitive_index]
        assignments.append(
            ChainPrimitiveAssignment(
                chain=chain,
                primitive_type=primitive_type,
                sheet_width_nodes=int(
                    scaffold.edge_sheet_width_nodes[
                        batch_index, first_source, first_target
                    ].item()
                ),
                sheet_orientations=tuple(
                    [
                        float(
                            scaffold.edge_orientation_start[
                                batch_index, first_source, first_target
                            ].item()
                        )
                    ]
                    + [
                        float(
                            scaffold.edge_orientation_end[
                                batch_index, source, target
                            ].item()
                        )
                        for source, target in edge_pairs
                    ]
                ),
                offset_distances=tuple(
                    [
                        float(
                            scaffold.edge_offset_start[
                                batch_index, first_source, first_target
                            ].item()
                        )
                    ]
                    + [
                        float(
                            scaffold.edge_offset_end[batch_index, source, target].item()
                        )
                        for source, target in edge_pairs
                    ]
                ),
                helix_phase=float(
                    scaffold.edge_helix_phase[
                        batch_index, first_source, first_target
                    ].item()
                ),
                helix_pitch=float(
                    scaffold.edge_helix_pitch[
                        batch_index, first_source, first_target
                    ].item()
                ),
                width_start=float(
                    scaffold.edge_width_start[
                        batch_index, first_source, first_target
                    ].item()
                ),
                width_end=float(
                    scaffold.edge_width_end[
                        batch_index, last_source, last_target
                    ].item()
                ),
                thickness_start=float(
                    scaffold.edge_thickness_start[
                        batch_index, first_source, first_target
                    ].item()
                ),
                thickness_end=float(
                    scaffold.edge_thickness_end[
                        batch_index, last_source, last_target
                    ].item()
                ),
                twist_start=float(
                    scaffold.edge_twist_start[
                        batch_index, first_source, first_target
                    ].item()
                ),
                twist_end=float(
                    scaffold.edge_twist_end[
                        batch_index, last_source, last_target
                    ].item()
                ),
                sweep_phase=float(
                    scaffold.edge_sweep_phase[
                        batch_index, first_source, first_target
                    ].item()
                ),
            )
        )
    return assignments


def materialize_scaffold(
    scaffold: Scaffolds,
    config: PrimitiveConfig | None = None,
) -> Structures:
    config = config or PrimitiveConfig()
    scaffold.validate()
    designs: list[Structures] = []
    for batch_index in range(scaffold.batch_size):
        centers = scaffold.positions[batch_index]
        adjacency = scaffold.adjacency[batch_index]
        assignments = _assignments_from_scaffold(scaffold, batch_index, config)
        styled_centers = _apply_chain_style_offsets(
            centers=centers,
            adjacency=adjacency,
            assignments=assignments,
            config=config,
        )
        positions, roles, edge_index = _materialize_scaffold_node_triplets(
            centers=styled_centers,
            adjacency=adjacency,
            assignments=assignments,
            config=config,
        )
        dense_adjacency = _edge_index_to_adjacency(positions.shape[0], edge_index)
        designs.append(
            Structures(
                positions=positions.unsqueeze(0).clamp(0.02, 0.98),
                roles=roles.unsqueeze(0),
                adjacency=symmetrize_matrix(dense_adjacency).unsqueeze(0),
            )
        )
    design = Structures(
        positions=torch.cat([item.positions for item in designs], dim=0),
        roles=torch.cat([item.roles for item in designs], dim=0),
        adjacency=torch.cat([item.adjacency for item in designs], dim=0),
    )
    design.validate()
    return design


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
    primitive_segments = _extract_primitive_segments(scaffold_adjacency)
    primitive_assignments = _sample_chain_primitives(primitive_segments, config, rng)
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
    assignment_tensors = _build_scaffold_assignment_tensors(
        scaffold_adjacency,
        primitive_assignments,
    )
    scaffold = Scaffolds(
        positions=styled_centers.unsqueeze(0).clamp(0.02, 0.98),
        roles=_build_scaffold_roles(styled_centers.shape[0]).unsqueeze(0),
        adjacency=symmetrize_matrix(scaffold_adjacency).unsqueeze(0),
        edge_primitive_ids=_build_scaffold_primitive_ids(
            scaffold_adjacency,
            primitive_assignments,
        ).unsqueeze(0),
        edge_primitive_types=_build_scaffold_primitive_types(
            scaffold_adjacency,
            primitive_assignments,
        ).unsqueeze(0),
        edge_sheet_width_nodes=assignment_tensors["edge_sheet_width_nodes"].unsqueeze(
            0
        ),
        edge_orientation_start=assignment_tensors["edge_orientation_start"].unsqueeze(
            0
        ),
        edge_orientation_end=assignment_tensors["edge_orientation_end"].unsqueeze(0),
        edge_offset_start=assignment_tensors["edge_offset_start"].unsqueeze(0),
        edge_offset_end=assignment_tensors["edge_offset_end"].unsqueeze(0),
        edge_helix_phase=assignment_tensors["edge_helix_phase"].unsqueeze(0),
        edge_helix_pitch=assignment_tensors["edge_helix_pitch"].unsqueeze(0),
        edge_width_start=assignment_tensors["edge_width_start"].unsqueeze(0),
        edge_width_end=assignment_tensors["edge_width_end"].unsqueeze(0),
        edge_thickness_start=assignment_tensors["edge_thickness_start"].unsqueeze(0),
        edge_thickness_end=assignment_tensors["edge_thickness_end"].unsqueeze(0),
        edge_twist_start=assignment_tensors["edge_twist_start"].unsqueeze(0),
        edge_twist_end=assignment_tensors["edge_twist_end"].unsqueeze(0),
        edge_sweep_phase=assignment_tensors["edge_sweep_phase"].unsqueeze(0),
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
