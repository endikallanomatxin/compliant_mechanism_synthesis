from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from compliant_mechanism_synthesis.dataset.types import Scaffolds, Structures
from compliant_mechanism_synthesis.roles import NodeRole
from compliant_mechanism_synthesis.tensor_ops import symmetrize_matrix


PRIMITIVE_LIBRARY = (
    "straight_lattice_sheet",
    "curved_lattice_sheet",
    "helix_lattice_sheet",
    "straight_beam",
    "curved_beam",
    "path_truss",
    "loose_cloud",
)

CHAIN_PRIMITIVE_LIBRARY = (
    "rod",
    "ribbon",
    "blade",
    "twist",
    "fin",
    "truss",
)


@dataclass(frozen=True)
class PrimitiveConfig:
    # This controls the sparse scaffolding graph, not the final number of free
    # FEM nodes. The scaffold is expanded into connection triplets later.
    num_free_nodes: int = 14
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


def _path(kind: str, t: torch.Tensor) -> torch.Tensor:
    x = 0.15 + 0.70 * t
    if kind == "straight_beam":
        y = torch.full_like(t, 0.50)
        z = torch.full_like(t, 0.50)
    elif kind == "curved_beam":
        y = 0.50 + 0.16 * torch.sin(math.pi * t)
        z = 0.50 + 0.10 * torch.sin(2.0 * math.pi * t + 0.25)
    elif kind == "straight_lattice_sheet":
        y = torch.full_like(t, 0.50)
        z = torch.full_like(t, 0.45)
    elif kind == "curved_lattice_sheet":
        y = 0.50 + 0.15 * torch.sin(math.pi * t)
        z = 0.45 + 0.08 * torch.sin(2.0 * math.pi * t)
    elif kind == "helix_lattice_sheet":
        angle = 2.5 * math.pi * t
        y = 0.50 + 0.15 * torch.cos(angle)
        z = 0.50 + 0.15 * torch.sin(angle)
    elif kind == "path_truss":
        y = 0.50 + 0.08 * torch.sin(2.0 * math.pi * t)
        z = 0.50 + 0.08 * torch.cos(2.0 * math.pi * t)
    elif kind == "loose_cloud":
        y = 0.50 + 0.05 * torch.sin(3.0 * math.pi * t)
        z = 0.50 + 0.05 * torch.cos(2.0 * math.pi * t)
    else:
        raise ValueError(f"unknown primitive kind: {kind}")
    return torch.stack([x, y, z], dim=-1)


def _sample_free_scaffold_positions(
    kind: str,
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=config.num_free_nodes + 2, dtype=torch.float32)[1:-1]
    centerline = _path(kind, t)

    if kind in {"straight_beam", "curved_beam"}:
        jitter = torch.tensor(
            [
                [rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)]
                for _ in range(config.num_free_nodes)
            ],
            dtype=torch.float32,
        )
        positions = centerline + jitter
    elif kind in {"straight_lattice_sheet", "curved_lattice_sheet", "helix_lattice_sheet"}:
        positions = []
        lateral = config.width / 2.0
        for index, point in enumerate(centerline):
            left_right = -1.0 if index % 2 == 0 else 1.0
            tangent_t0 = max(index - 1, 0)
            tangent_t1 = min(index + 1, centerline.shape[0] - 1)
            tangent = centerline[tangent_t1] - centerline[tangent_t0]
            _, normal_1, normal_2 = _orthonormal_frame(tangent)
            layer = normal_1 if index % 3 else normal_2
            positions.append(point + lateral * left_right * layer)
        positions = torch.stack(positions, dim=0)
    elif kind == "path_truss":
        positions = []
        for index, point in enumerate(centerline):
            tangent_t0 = max(index - 1, 0)
            tangent_t1 = min(index + 1, centerline.shape[0] - 1)
            tangent = centerline[tangent_t1] - centerline[tangent_t0]
            _, normal_1, normal_2 = _orthonormal_frame(tangent)
            angle = 2.0 * math.pi * (index % 3) / 3.0
            positions.append(
                point
                + 0.5 * config.width * math.cos(angle) * normal_1
                + 0.5 * config.thickness * math.sin(angle) * normal_2
            )
        positions = torch.stack(positions, dim=0)
    elif kind == "loose_cloud":
        positions = centerline + torch.tensor(
            [
                [rng.uniform(-0.14, 0.14), rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18)]
                for _ in range(config.num_free_nodes)
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"unknown primitive kind: {kind}")

    positions = positions.clamp(0.05, 0.95)
    z_values = positions[:, 2]
    z_min = float(z_values.min().item())
    z_max = float(z_values.max().item())
    if z_max - z_min < 1e-6:
        positions[:, 2] = 0.5 * (config.free_z_min + config.free_z_max)
    else:
        normalized = (z_values - z_min) / (z_max - z_min)
        positions[:, 2] = config.free_z_min + normalized * (config.free_z_max - config.free_z_min)
    return positions


def _sample_scaffold_centers(
    kind: str,
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    start_center = _path(kind, torch.tensor([0.0], dtype=torch.float32))[0]
    end_center = _path(kind, torch.tensor([1.0], dtype=torch.float32))[0]
    start_center[2] = config.fixed_anchor_z
    end_center[2] = config.mobile_anchor_z
    free_positions = _sample_free_scaffold_positions(kind, config, rng)
    return torch.cat(
        [start_center.unsqueeze(0), free_positions, end_center.unsqueeze(0)],
        dim=0,
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
                primitive_type=rng.choice(CHAIN_PRIMITIVE_LIBRARY),
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
        return 0.08 * width * (math.cos(sweep_phase) * normal_1 + math.sin(sweep_phase) * normal_2)
    if primitive_type == "ribbon":
        return 0.18 * width * normal_1
    if primitive_type == "blade":
        return 0.18 * thickness * normal_2
    if primitive_type == "twist":
        angle = sweep_phase + twist * math.pi
        return 0.14 * width * (math.cos(angle) * normal_1 + math.sin(angle) * normal_2)
    if primitive_type == "fin":
        return 0.16 * width * normal_1 + 0.08 * thickness * normal_2
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
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_parameters: dict[int, tuple[str, float, float, float]] = {}
    for assignment in assignments:
        for local_index, node_index in enumerate(assignment.chain):
            if node_index in {0, centers.shape[0] - 1}:
                continue
            fraction = local_index / max(len(assignment.chain) - 1, 1)
            width = (1.0 - fraction) * assignment.width_start + fraction * assignment.width_end
            thickness = (1.0 - fraction) * assignment.thickness_start + fraction * assignment.thickness_end
            twist = (1.0 - fraction) * assignment.twist_start + fraction * assignment.twist_end
            if node_index not in node_parameters:
                node_parameters[node_index] = (assignment.primitive_type, width, thickness, twist)

    positions = []
    roles = []
    for node_index in range(centers.shape[0]):
        center = centers[node_index]
        tangent = _estimate_scaffold_tangent(centers, adjacency, node_index)
        _, normal_1, normal_2 = _orthonormal_frame(tangent)

        if node_index == 0:
            role = NodeRole.FIXED
            offsets = (
                config.anchor_radius * normal_1,
                config.anchor_radius * (-0.5 * normal_1 + 0.866 * normal_2),
                config.anchor_radius * (-0.5 * normal_1 - 0.866 * normal_2),
            )
        elif node_index == centers.shape[0] - 1:
            role = NodeRole.MOBILE
            offsets = (
                config.anchor_radius * normal_1,
                config.anchor_radius * (-0.5 * normal_1 + 0.866 * normal_2),
                config.anchor_radius * (-0.5 * normal_1 - 0.866 * normal_2),
            )
        else:
            role = NodeRole.FREE
            primitive_type, width, thickness, twist = node_parameters.get(
                node_index,
                ("rod", config.primitive_radius, 0.75 * config.primitive_radius, 0.0),
            )
            if primitive_type == "rod":
                offsets = (
                    width * normal_1,
                    width * (-0.5 * normal_1 + 0.866 * normal_2),
                    width * (-0.5 * normal_1 - 0.866 * normal_2),
                )
            elif primitive_type == "ribbon":
                offsets = (
                    width * normal_1 + 0.2 * thickness * normal_2,
                    -width * normal_1 + 0.2 * thickness * normal_2,
                    -0.4 * thickness * normal_2,
                )
            elif primitive_type == "blade":
                offsets = (
                    thickness * normal_2,
                    -0.5 * thickness * normal_2 + width * normal_1,
                    -0.5 * thickness * normal_2 - width * normal_1,
                )
            elif primitive_type == "twist":
                angle = twist * math.pi
                rotated_1 = math.cos(angle) * normal_1 + math.sin(angle) * normal_2
                rotated_2 = -math.sin(angle) * normal_1 + math.cos(angle) * normal_2
                offsets = (
                    width * rotated_1,
                    -0.5 * width * rotated_1 + thickness * rotated_2,
                    -0.5 * width * rotated_1 - thickness * rotated_2,
                )
            elif primitive_type == "fin":
                offsets = (
                    1.2 * width * normal_1,
                    -0.4 * width * normal_1 + thickness * normal_2,
                    -0.4 * width * normal_1 - thickness * normal_2,
                )
            elif primitive_type == "truss":
                offsets = (
                    width * normal_1 + thickness * normal_2,
                    -width * normal_1 + thickness * normal_2,
                    -thickness * normal_2,
                )
            else:
                raise ValueError(f"unknown primitive type: {primitive_type}")

        for offset in offsets:
            positions.append(center + offset)
            roles.append(role)

    return torch.stack(positions, dim=0), torch.tensor(roles, dtype=torch.long)


def _materialize_primitive_connectivity(scaffold_adjacency: torch.Tensor) -> torch.Tensor:
    num_scaffold_nodes = scaffold_adjacency.shape[0]
    adjacency = torch.zeros((num_scaffold_nodes * 3, num_scaffold_nodes * 3), dtype=torch.float32)

    for source in range(num_scaffold_nodes):
        for target in range(source + 1, num_scaffold_nodes):
            if scaffold_adjacency[source, target] <= 0.0:
                continue
            source_nodes = range(3 * source, 3 * source + 3)
            target_nodes = range(3 * target, 3 * target + 3)
            for source_node in source_nodes:
                for target_node in target_nodes:
                    adjacency[source_node, target_node] = 1.0
                    adjacency[target_node, source_node] = 1.0

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
    kind: str,
    config: PrimitiveConfig,
    seed: int | None,
) -> tuple[Structures, Scaffolds]:
    rng = random.Random(seed)

    # The generator is intentionally split into a sparse graph stage and a
    # materialization stage. This keeps the high-level combinatorics separate
    # from the final FEM mesh and leaves a clean place for a future two-stage
    # optimizer: first over the scaffold/primitive parameters, then over the
    # final expanded mesh.
    scaffold_centers = _sample_scaffold_centers(kind, config, rng)
    scaffold_adjacency = _sample_scaffold_connectivity(scaffold_centers, config, rng)
    primitive_chains = _extract_primitive_chains(scaffold_adjacency)
    primitive_assignments = _sample_chain_primitives(primitive_chains, config, rng)
    styled_centers = _apply_chain_style_offsets(
        centers=scaffold_centers,
        adjacency=scaffold_adjacency,
        assignments=primitive_assignments,
        config=config,
    )
    positions, roles = _materialize_scaffold_node_triplets(
        centers=styled_centers,
        adjacency=scaffold_adjacency,
        assignments=primitive_assignments,
        config=config,
        rng=rng,
    )
    adjacency = _materialize_primitive_connectivity(scaffold_adjacency)

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
    kind: str,
    config: PrimitiveConfig | None = None,
    seed: int | None = None,
) -> Structures:
    config = config or PrimitiveConfig()
    design, _ = _sample_primitive_case(kind, config, seed)
    return design


def sample_random_primitive(
    config: PrimitiveConfig | None = None,
    seed: int | None = None,
) -> tuple[Structures, Scaffolds]:
    rng = random.Random(seed)
    kind = rng.choice(PRIMITIVE_LIBRARY)
    kind_seed = None if seed is None else rng.randrange(0, 2**31)
    return _sample_primitive_case(kind, config or PrimitiveConfig(), kind_seed)
