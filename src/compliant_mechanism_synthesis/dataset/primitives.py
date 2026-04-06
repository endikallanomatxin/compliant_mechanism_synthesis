from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from compliant_mechanism_synthesis.dataset.types import Structures
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


@dataclass(frozen=True)
class PrimitiveConfig:
    num_free_nodes: int = 18
    width: float = 0.18
    thickness: float = 0.10
    anchor_radius: float = 0.04
    neighbor_count: int = 3
    free_z_min: float = 0.28
    free_z_max: float = 0.72
    fixed_anchor_z: float = 0.10
    mobile_anchor_z: float = 0.90


def _orthonormal_frame(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=direction.dtype)
    if abs(float(direction[2].item())) > 0.85:
        reference = torch.tensor([0.0, 1.0, 0.0], dtype=direction.dtype)
    normal_1 = torch.linalg.cross(reference, direction)
    normal_1 = normal_1 / torch.linalg.vector_norm(normal_1).clamp_min(1e-8)
    normal_2 = torch.linalg.cross(direction, normal_1)
    return direction, normal_1, normal_2


def _anchor_triangle(center: torch.Tensor, radius: float, anchor_z: float) -> torch.Tensor:
    angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    anchors = []
    anchor_center = center.clone()
    anchor_center[2] = anchor_z
    for angle in angles:
        planar_offset = torch.tensor(
            [radius * math.cos(angle), radius * math.sin(angle), 0.0],
            dtype=center.dtype,
        )
        anchors.append(
            anchor_center + planar_offset
        )
    return torch.stack(anchors, dim=0)


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


def _sample_free_positions(
    kind: str,
    config: PrimitiveConfig,
    rng: random.Random,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=config.num_free_nodes + 2, dtype=torch.float32)[1:-1]
    centerline = _path(kind, t)

    if kind in {"straight_beam", "curved_beam"}:
        jitter = torch.tensor(
            [[rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)] for _ in range(config.num_free_nodes)],
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
            offset = lateral * left_right * layer
            positions.append(point + offset)
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
            [[rng.uniform(-0.14, 0.14), rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18)] for _ in range(config.num_free_nodes)],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"unknown primitive kind: {kind}")

    return positions.clamp(0.05, 0.95)


def _remap_free_positions_z(
    positions: torch.Tensor,
    config: PrimitiveConfig,
) -> torch.Tensor:
    remapped = positions.clone()
    z_values = remapped[:, 2]
    z_min = float(z_values.min().item())
    z_max = float(z_values.max().item())
    if z_max - z_min < 1e-6:
        remapped[:, 2] = 0.5 * (config.free_z_min + config.free_z_max)
        return remapped
    normalized_z = (z_values - z_min) / (z_max - z_min)
    remapped[:, 2] = config.free_z_min + normalized_z * (config.free_z_max - config.free_z_min)
    return remapped


def _connect_local_neighbors(positions: torch.Tensor, adjacency: torch.Tensor, neighbor_count: int) -> None:
    pairwise = torch.linalg.vector_norm(positions[:, None, :] - positions[None, :, :], dim=-1)
    for node_index in range(positions.shape[0]):
        nearest = torch.argsort(pairwise[node_index])[1 : neighbor_count + 1]
        for neighbor in nearest.tolist():
            adjacency[node_index, neighbor] = max(float(adjacency[node_index, neighbor].item()), 0.65)


def sample_primitive_design(
    kind: str,
    config: PrimitiveConfig | None = None,
    seed: int | None = None,
) -> Structures:
    config = config or PrimitiveConfig()
    rng = random.Random(seed)
    free_positions = _remap_free_positions_z(_sample_free_positions(kind, config, rng), config)
    start_center = _path(kind, torch.tensor([0.0], dtype=torch.float32))[0]
    end_center = _path(kind, torch.tensor([1.0], dtype=torch.float32))[0]
    fixed_anchors = _anchor_triangle(start_center, config.anchor_radius, anchor_z=config.fixed_anchor_z)
    mobile_anchors = _anchor_triangle(end_center, config.anchor_radius, anchor_z=config.mobile_anchor_z)
    positions = torch.cat([fixed_anchors, mobile_anchors, free_positions], dim=0).clamp(0.02, 0.98)

    roles = torch.tensor(
        [NodeRole.FIXED] * 3 + [NodeRole.MOBILE] * 3 + [NodeRole.FREE] * config.num_free_nodes,
        dtype=torch.long,
    )

    adjacency = torch.zeros((positions.shape[0], positions.shape[0]), dtype=torch.float32)
    free_offset = 6
    _connect_local_neighbors(positions[free_offset:], adjacency[free_offset:, free_offset:], config.neighbor_count)

    first_free = list(range(free_offset, min(free_offset + 3, positions.shape[0])))
    last_free = list(range(max(free_offset, positions.shape[0] - 3), positions.shape[0]))
    for anchor_index in range(3):
        for node_index in first_free:
            adjacency[anchor_index, node_index] = 0.85
            adjacency[node_index, anchor_index] = 0.85
        for node_index in last_free:
            mobile_anchor = 3 + anchor_index
            adjacency[mobile_anchor, node_index] = 0.85
            adjacency[node_index, mobile_anchor] = 0.85

    if kind in {"straight_lattice_sheet", "curved_lattice_sheet", "helix_lattice_sheet"}:
        for node_index in range(free_offset, positions.shape[0] - 2, 2):
            adjacency[node_index, node_index + 2] = 0.55
            adjacency[node_index + 2, node_index] = 0.55

    if kind == "path_truss":
        for node_index in range(free_offset, positions.shape[0] - 3):
            adjacency[node_index, node_index + 3] = 0.70
            adjacency[node_index + 3, node_index] = 0.70

    if kind == "loose_cloud":
        pairwise = torch.linalg.vector_norm(positions[:, None, :] - positions[None, :, :], dim=-1)
        for anchor_index in range(6):
            nearest = torch.argsort(pairwise[anchor_index])[1:4]
            for node_index in nearest.tolist():
                adjacency[anchor_index, node_index] = 0.55
                adjacency[node_index, anchor_index] = 0.55

    design = Structures(
        positions=positions.unsqueeze(0),
        roles=roles.unsqueeze(0),
        adjacency=symmetrize_matrix(adjacency).unsqueeze(0),
    )
    design.validate()
    return design


def sample_random_primitive(
    config: PrimitiveConfig | None = None,
    seed: int | None = None,
) -> tuple[str, Structures]:
    rng = random.Random(seed)
    kind = rng.choice(PRIMITIVE_LIBRARY)
    kind_seed = None if seed is None else rng.randrange(0, 2**31)
    return kind, sample_primitive_design(kind, config=config, seed=kind_seed)
