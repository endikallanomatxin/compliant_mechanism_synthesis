from __future__ import annotations

import math
import random

import torch

from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    symmetrize_adjacency,
)


def _sample_point(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    existing: list[tuple[float, float]],
    min_spacing: float,
) -> tuple[float, float]:
    for _ in range(500):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_spacing**2 for px, py in existing):
            return x, y
    raise RuntimeError("failed to sample point with requested spacing")


def _adaptive_interior_spacing(
    num_nodes: int,
    requested_spacing: float,
) -> float:
    free_nodes = max(num_nodes - 4, 1)
    interior_area = 0.8 * 0.8
    spacing_from_area = 0.75 * math.sqrt(interior_area / free_nodes)
    return min(requested_spacing, spacing_from_area)


def generate_points(
    num_nodes: int, min_spacing: float = 0.12
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_nodes < 6:
        raise ValueError("num_nodes must be at least 6")

    coords: list[tuple[float, float]] = []
    roles = torch.full((num_nodes,), ROLE_FREE, dtype=torch.long)
    effective_spacing = _adaptive_interior_spacing(num_nodes, min_spacing)

    coords.append(_sample_point((0.0, 0.2), (0.0, 0.1), coords, effective_spacing))
    roles[0] = ROLE_FIXED
    coords.append(_sample_point((0.8, 1.0), (0.0, 0.1), coords, effective_spacing))
    roles[1] = ROLE_FIXED
    coords.append(_sample_point((0.0, 0.2), (0.9, 1.0), coords, effective_spacing))
    roles[2] = ROLE_MOBILE
    coords.append(_sample_point((0.8, 1.0), (0.9, 1.0), coords, effective_spacing))
    roles[3] = ROLE_MOBILE

    while len(coords) < num_nodes:
        coords.append(_sample_point((0.1, 0.9), (0.1, 0.9), coords, effective_spacing))

    return torch.tensor(coords, dtype=torch.float32), roles


def _activate_edge(
    adjacency: torch.Tensor, i: int, j: int, low: float, high: float
) -> None:
    value = random.uniform(low, high)
    adjacency[i, j] = value
    adjacency[j, i] = value


def _sorted_free_indices(positions: torch.Tensor, roles: torch.Tensor) -> list[int]:
    free = [idx for idx, role in enumerate(roles.tolist()) if role == ROLE_FREE]
    return sorted(
        free, key=lambda idx: (positions[idx, 1].item(), positions[idx, 0].item())
    )


def generate_connectivity(positions: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    num_nodes = positions.shape[0]
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    free_sorted = _sorted_free_indices(positions, roles)
    fixed = [0, 1]
    mobile = [2, 3]

    backbone_count = max(2, min(len(free_sorted), num_nodes // 3))
    backbone = free_sorted[:backbone_count]
    if not backbone:
        backbone = free_sorted

    motif = random.choice(["chain", "arch", "fan", "lattice", "brace"])

    # A connected backbone gives every motif a mechanically meaningful path.
    start = random.choice(fixed)
    previous = start
    for node in backbone:
        _activate_edge(adjacency, previous, node, 0.65, 1.0)
        previous = node
    _activate_edge(adjacency, previous, random.choice(mobile), 0.65, 1.0)

    if motif in {"chain", "brace", "lattice"}:
        for left, right in zip(backbone[:-1], backbone[1:]):
            _activate_edge(adjacency, left, right, 0.5, 0.9)

    if motif in {"arch", "brace"}:
        for node in backbone:
            if positions[node, 0] < 0.5:
                _activate_edge(adjacency, 0, node, 0.4, 0.8)
            else:
                _activate_edge(adjacency, 1, node, 0.4, 0.8)

    if motif in {"fan", "brace"}:
        fan_sources = fixed + mobile
        for source in fan_sources:
            chosen = random.sample(free_sorted, k=min(3, len(free_sorted)))
            for target in chosen:
                _activate_edge(adjacency, source, target, 0.35, 0.7)

    if motif in {"lattice", "arch"}:
        for i, node in enumerate(free_sorted[:-1]):
            neighbors = sorted(
                free_sorted[i + 1 :],
                key=lambda idx: torch.linalg.vector_norm(
                    positions[idx] - positions[node]
                ).item(),
            )[:2]
            for target in neighbors:
                _activate_edge(adjacency, node, target, 0.3, 0.65)

    if motif in {"brace", "lattice"} and len(free_sorted) >= 4:
        left_nodes = [idx for idx in free_sorted if positions[idx, 0] < 0.5]
        right_nodes = [idx for idx in free_sorted if positions[idx, 0] >= 0.5]
        for left in left_nodes[:2]:
            for right in right_nodes[:2]:
                _activate_edge(adjacency, left, right, 0.25, 0.55)

    for mobile_idx in mobile:
        nearest_free = sorted(
            free_sorted,
            key=lambda idx: torch.linalg.vector_norm(
                positions[idx] - positions[mobile_idx]
            ).item(),
        )[:2]
        for free_idx in nearest_free:
            _activate_edge(adjacency, mobile_idx, free_idx, 0.45, 0.8)

    return symmetrize_adjacency(adjacency)


def generate_graph_sample(
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, roles = generate_points(num_nodes)
    adjacency = generate_connectivity(positions, roles)
    return positions, roles, adjacency


def generate_noise_connectivity(num_nodes: int) -> torch.Tensor:
    adjacency = torch.rand((num_nodes, num_nodes), dtype=torch.float32).pow(1.5)
    return symmetrize_adjacency(adjacency)


def generate_noise_sample(
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, roles = generate_points(num_nodes)
    adjacency = generate_noise_connectivity(num_nodes)
    return positions, roles, adjacency
