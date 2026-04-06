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

# During scaffold-to-mesh debugging we intentionally keep the active primitive
# family to the simplest one. That gives us one geometry/conectivity pattern to
# validate visually before re-introducing wider families.
ACTIVE_CHAIN_PRIMITIVE_LIBRARY = ("rod_helix",)


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
    primitive_radius: float = 0.02
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
    sheet_width_nodes: int = 4
    sheet_width_distance: float = 0.03


@dataclass(frozen=True)
class ChainPrimitiveAssignment:
    chain: list[int]
    primitive_type: str
    width_start: float
    width_end: float
    thickness_start: float
    thickness_end: float
    twist_start: float
    twist_end: float
    sweep_phase: float


def _orthonormal_frame(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=direction.dtype, device=direction.device)
    if abs(float(direction[2].item())) > 0.85:
        reference = torch.tensor([0.0, 1.0, 0.0], dtype=direction.dtype, device=direction.device)
    normal_1 = torch.linalg.cross(reference, direction)
    normal_1 = normal_1 / torch.linalg.vector_norm(normal_1).clamp_min(1e-8)
    normal_2 = torch.linalg.cross(direction, normal_1)
    return direction, normal_1, normal_2


def _sample_random_scaffold_centers(
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    start_center = torch.tensor([0.12, 0.50, config.fixed_anchor_z], dtype=torch.float32)
    end_center = torch.tensor([0.88, 0.50, config.mobile_anchor_z], dtype=torch.float32)

    base_x = torch.linspace(0.22, 0.78, steps=config.num_free_nodes, dtype=torch.float32)
    free_positions = torch.stack(
        [
            base_x + torch.tensor([rng.uniform(-0.025, 0.025) for _ in range(config.num_free_nodes)], dtype=torch.float32),
            torch.tensor([rng.uniform(0.20, 0.80) for _ in range(config.num_free_nodes)], dtype=torch.float32),
            torch.tensor(
                [rng.uniform(config.free_z_min, config.free_z_max) for _ in range(config.num_free_nodes)],
                dtype=torch.float32,
            ),
        ],
        dim=-1,
    )
    free_positions = free_positions[torch.argsort(free_positions[:, 0])]
    return torch.cat([start_center.unsqueeze(0), free_positions, end_center.unsqueeze(0)], dim=0)


def _localized_edge_probability(
    distance: float,
    config: PrimitiveConfig,
) -> float:
    return math.exp(-(distance**2) / max(config.connection_length_scale**2, 1e-8))


def _ensure_min_degree_two(
    centers: torch.Tensor,
    adjacency: torch.Tensor,
) -> None:
    pairwise = torch.linalg.vector_norm(centers[:, None, :] - centers[None, :, :], dim=-1)
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
    if set(ACTIVE_CHAIN_PRIMITIVE_LIBRARY).issubset({"rod", "rod_helix", "sheet", "sheet_helix", "truss"}):
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        remaining = list(range(1, num_nodes - 1))
        order = [0]
        current = 0
        while remaining:
            weights = []
            for candidate in remaining:
                distance = float(torch.linalg.vector_norm(centers[candidate] - centers[current]).item())
                forward_bias = max(float(centers[candidate, 0].item() - centers[current, 0].item()), 0.02)
                weights.append(_localized_edge_probability(distance, config) * forward_bias)
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
    pairwise = torch.linalg.vector_norm(centers[:, None, :] - centers[None, :, :], dim=-1)

    # First build a localized spanning tree so every scaffold node belongs to a
    # connected support graph before we add random local redundancy.
    for node_index in range(1, num_nodes):
        candidate_indices = list(range(node_index))
        candidate_indices.sort(key=lambda index: float(pairwise[node_index, index].item()))
        local_candidates = candidate_indices[: max(2, config.neighbor_count)]
        weights = [
            _localized_edge_probability(float(pairwise[node_index, candidate].item()), config)
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
            probability = config.extra_connection_probability * _localized_edge_probability(
                float(pairwise[node_index, candidate].item()),
                config,
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
    for chain in chains:
        width_scale = rng.uniform(0.8, 1.15)
        thickness_scale = rng.uniform(0.45, 0.85)
        assignments.append(
            ChainPrimitiveAssignment(
                chain=chain,
                primitive_type=rng.choice(ACTIVE_CHAIN_PRIMITIVE_LIBRARY),
                width_start=config.primitive_radius * width_scale * rng.uniform(0.8, 1.1),
                width_end=config.primitive_radius * width_scale * rng.uniform(0.8, 1.1),
                thickness_start=config.primitive_radius * thickness_scale * rng.uniform(0.8, 1.1),
                thickness_end=config.primitive_radius * thickness_scale * rng.uniform(0.8, 1.1),
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
            neighbors = torch.nonzero(adjacency[current] > 0.0, as_tuple=False).flatten().tolist()
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
        neighbors = torch.nonzero(adjacency[start] > 0.0, as_tuple=False).flatten().tolist()
        for nxt in neighbors:
            edge_key = tuple(sorted((start, nxt)))
            if edge_key in visited_edges:
                continue
            chains.append(walk_chain(start, nxt))

    for start in range(num_nodes):
        neighbors = torch.nonzero(adjacency[start] > 0.0, as_tuple=False).flatten().tolist()
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
        return 0.12 * width * normal_1
    if primitive_type == "sheet_helix":
        angle = sweep_phase + twist * math.pi
        return 0.12 * width * (math.cos(angle) * normal_1 + math.sin(angle) * normal_2)
    if primitive_type == "truss":
        return 0.12 * width * normal_1 - 0.12 * thickness * normal_2
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
            width = (1.0 - fraction) * assignment.width_start + fraction * assignment.width_end
            thickness = (1.0 - fraction) * assignment.thickness_start + fraction * assignment.thickness_end
            twist = (1.0 - fraction) * assignment.twist_start + fraction * assignment.twist_end
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
        if scaffold_index in {0, centers.shape[0] - 1} or degree[scaffold_index].item() != 2:
            ensure_connection_node(scaffold_index)

    for assignment in assignments:
        if assignment.primitive_type not in {"rod", "rod_helix", "sheet", "sheet_helix", "truss"}:
            raise ValueError("debugging path only supports the current primitive families")
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

        sheet_width = max(config.sheet_width_distance, 0.75 * (assignment.width_start + assignment.width_end))
        left_chain, right_chain = _discretize_sheet_chain(
            chain_positions=chain_positions,
            sheet_width=sheet_width,
            helix_turns=(config.helix_turns if assignment.primitive_type == "sheet_helix" else 0.0),
            helix_phase=assignment.sweep_phase,
        )
        _materialize_two_rail_sheet(
            left_chain=left_chain,
            right_chain=right_chain,
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
        subdivisions = max(4, int(math.ceil(segment_length / max(target_edge_length, 1e-6))) * 6)
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
        num_segments = max(1, int(math.ceil(total_length / max(target_edge_length, 1e-6))))
        num_points = num_segments + 1
    else:
        num_points = max(2, num_points)
    targets = torch.linspace(0.0, total_length, steps=num_points, dtype=polyline.dtype)
    resampled = []
    segment_index = 0
    for target in targets.tolist():
        while segment_index < segment_lengths.shape[0] - 1 and float(cumulative[segment_index + 1].item()) < target:
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
    num_points = max(2, int(math.ceil(workspace_span / max(target_edge_length, 1e-6))) + 1)
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


def _estimate_polyline_tangent(polyline: torch.Tensor, sample_index: int) -> torch.Tensor:
    if sample_index == 0:
        tangent = polyline[1] - polyline[0]
    elif sample_index == polyline.shape[0] - 1:
        tangent = polyline[-1] - polyline[-2]
    else:
        tangent = polyline[sample_index + 1] - polyline[sample_index - 1]
    if torch.linalg.vector_norm(tangent).item() < 1e-8:
        tangent = torch.tensor([1.0, 0.0, 0.0], dtype=polyline.dtype)
    return tangent


def _discretize_sheet_chain(
    chain_positions: torch.Tensor,
    sheet_width: float,
    helix_turns: float,
    helix_phase: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = []
    right = []
    sample_count = max(chain_positions.shape[0] - 1, 1)
    for sample_index in range(chain_positions.shape[0]):
        tangent = _estimate_polyline_tangent(chain_positions, sample_index)
        _, normal_1, normal_2 = _orthonormal_frame(tangent)
        if helix_turns != 0.0:
            angle = helix_phase + 2.0 * math.pi * helix_turns * (sample_index / sample_count)
            lateral_axis = math.cos(angle) * normal_1 + math.sin(angle) * normal_2
        else:
            lateral_axis = normal_1
        offset = 0.5 * sheet_width * lateral_axis
        center = chain_positions[sample_index]
        left.append(center + offset)
        right.append(center - offset)
    return torch.stack(left, dim=0), torch.stack(right, dim=0)


def _discretize_rod_helix_chain(
    chain_positions: torch.Tensor,
    helix_radius: float,
    turns: float,
    phase: float,
    target_edge_length: float,
) -> torch.Tensor:
    centerline_span = float(torch.linalg.vector_norm(chain_positions[-1] - chain_positions[0]).item())
    helix_circumference_travel = abs(turns) * 2.0 * math.pi * helix_radius
    helical_length = math.sqrt(centerline_span**2 + helix_circumference_travel**2)
    # Helices need denser sampling than their centerline, but the offline
    # dataset still needs fixed tensor shapes across cases. We therefore
    # normalize the sample count against the expected helix geometry for the
    # whole family rather than the exact per-case arc length.
    num_points = max(2, int(math.ceil(helical_length / max(1e-6, target_edge_length))) + 1)
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
        offset = helix_radius * (math.cos(angle) * normal_1 + math.sin(angle) * normal_2)
        helical_positions.append(helix_support[sample_index] + offset)
    return torch.stack(helical_positions, dim=0)


def _materialize_two_rail_sheet(
    left_chain: torch.Tensor,
    right_chain: torch.Tensor,
    add_node,
    add_edge,
    start_index: int,
    end_index: int,
) -> None:
    previous_left = start_index
    previous_right = start_index
    for sample_index in range(1, left_chain.shape[0] - 1):
        left_index = add_node(left_chain[sample_index], NodeRole.FREE)
        right_index = add_node(right_chain[sample_index], NodeRole.FREE)
        add_edge(previous_left, left_index)
        add_edge(previous_right, right_index)
        add_edge(left_index, right_index)
        if sample_index > 1:
            add_edge(previous_left, right_index)
            add_edge(previous_right, left_index)
        previous_left = left_index
        previous_right = right_index
    add_edge(previous_left, end_index)
    add_edge(previous_right, end_index)
    add_edge(previous_left, previous_right)


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
        primitive_type: index for index, primitive_type in enumerate(CHAIN_PRIMITIVE_LIBRARY)
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
