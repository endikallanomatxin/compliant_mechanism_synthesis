from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


SPRING_NEIGHBORS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _bridge_component_ratio(mask: np.ndarray) -> tuple[float, float]:
    height, width = mask.shape
    top_coords = [(0, c) for c in range(width) if mask[0, c]]
    if not top_coords:
        return 0.0, 1.0

    visited: set[tuple[int, int]] = set(top_coords)
    stack = list(top_coords)
    touches_bottom = any(r == height - 1 for r, _ in top_coords)

    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if not mask[nr, nc] or (nr, nc) in visited:
                    continue
                visited.add((nr, nc))
                touches_bottom = touches_bottom or nr == height - 1
                stack.append((nr, nc))

    active = int(mask.sum())
    bridge_mass = len(visited) if touches_bottom else 0
    if active == 0:
        return 0.0, 1.0
    connectivity_penalty = 1.0 - (bridge_mass / active)
    return bridge_mass / mask.size, connectivity_penalty


def interface_length(occupancy: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(occupancy[:, :, :, 1:] - occupancy[:, :, :, :-1]).mean(dim=(1, 2, 3))
    dy = torch.abs(occupancy[:, :, 1:, :] - occupancy[:, :, :-1, :]).mean(dim=(1, 2, 3))
    return dx + dy


def topology_regularizers(occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
    occupancy_mass = occupancy.mean(dim=(1, 2, 3))
    bottom = torch.zeros_like(occupancy)
    top = torch.zeros_like(occupancy)
    bottom[:, :, -1, :] = 1.0
    top[:, :, 0, :] = 1.0

    reach = occupancy * top
    for _ in range(occupancy.shape[-2]):
        reach = occupancy * F.max_pool2d(reach, kernel_size=3, stride=1, padding=1)

    bridge_fraction = reach[:, :, -1, :].mean(dim=(1, 2))
    connectivity_penalty = (1.0 - bridge_fraction).clamp(0.0, 1.0)
    return {
        "surface": interface_length(occupancy),
        "connectivity_penalty": connectivity_penalty,
        "occupancy_mass": occupancy_mass,
    }


def _spring_matrix(dr: int, dc: int) -> np.ndarray:
    direction = np.array([float(dc), float(-dr)], dtype=np.float64)
    length = np.linalg.norm(direction)
    unit = direction / length
    axial = np.outer(unit, unit)
    stiffness = 1.0 / length
    return stiffness * np.block([[axial, -axial], [-axial, axial]])


def _assemble_stiffness(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=np.float64), coords

    node_ids = -np.ones(mask.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords):
        node_ids[r, c] = idx

    total_dofs = len(coords) * 2
    stiffness = np.zeros((total_dofs, total_dofs), dtype=np.float64)

    for r, c in coords:
        source = node_ids[r, c]
        for dr, dc in SPRING_NEIGHBORS:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= mask.shape[0] or nc < 0 or nc >= mask.shape[1]:
                continue
            if not mask[nr, nc]:
                continue

            target = node_ids[nr, nc]
            element = _spring_matrix(dr, dc)
            dofs = np.array(
                [2 * source, 2 * source + 1, 2 * target, 2 * target + 1], dtype=np.int32
            )
            stiffness[np.ix_(dofs, dofs)] += element

    return stiffness, coords


def _top_plate_projection(coords: np.ndarray, top_indices: list[int]) -> np.ndarray:
    projection = np.zeros((len(coords) * 2, 3), dtype=np.float64)
    top_coords = coords[top_indices]
    center_x = float(top_coords[:, 1].mean()) if len(top_coords) else 0.0
    center_y = float(top_coords[:, 0].mean()) if len(top_coords) else 0.0

    for idx in top_indices:
        r, c = coords[idx]
        projection[2 * idx, 0] = 1.0
        projection[2 * idx + 1, 1] = 1.0
        projection[2 * idx, 2] = -(float(r) - center_y)
        projection[2 * idx + 1, 2] = float(c) - center_x

    return projection


def _reduced_stiffness(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    stiffness, coords = _assemble_stiffness(mask)
    if stiffness.shape[0] == 0:
        return np.zeros((3, 3), dtype=np.float64), 0.0, 1.0

    height = mask.shape[0]
    bottom_indices = [idx for idx, (r, _) in enumerate(coords) if r == height - 1]
    top_indices = [idx for idx, (r, _) in enumerate(coords) if r == 0]
    connected_mass, connectivity_penalty = _bridge_component_ratio(mask)

    if not top_indices or not bottom_indices:
        return np.zeros((3, 3), dtype=np.float64), connected_mass, 1.0

    total_dofs = stiffness.shape[0]
    free_dof_indices: list[int] = []
    for idx in range(len(coords)):
        if idx in bottom_indices or idx in top_indices:
            continue
        free_dof_indices.extend([2 * idx, 2 * idx + 1])

    projection = np.zeros((total_dofs, len(free_dof_indices) + 3), dtype=np.float64)
    for col, dof in enumerate(free_dof_indices):
        projection[dof, col] = 1.0
    projection[:, len(free_dof_indices) :] = _top_plate_projection(coords, top_indices)

    reduced = projection.T @ stiffness @ projection
    free_count = len(free_dof_indices)
    plate = reduced[free_count:, free_count:]
    if free_count == 0:
        return plate, connected_mass, connectivity_penalty

    coupling = reduced[:free_count, free_count:]
    free = reduced[:free_count, :free_count]
    regularizer = 1e-6 * max(float(np.trace(free)) / max(free_count, 1), 1.0)
    stabilized = free + regularizer * np.eye(free_count, dtype=np.float64)
    condensed = plate - coupling.T @ np.linalg.solve(stabilized, coupling)
    return condensed, connected_mass, connectivity_penalty


def _raw_properties(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    condensed, connected_mass, connectivity_penalty = _reduced_stiffness(mask)
    if not np.any(condensed):
        return np.zeros(3, dtype=np.float64), connected_mass, connectivity_penalty

    regularizer = 1e-6 * max(float(np.trace(condensed)) / 3.0, 1.0)
    compliance = np.linalg.inv(condensed + regularizer * np.eye(3, dtype=np.float64))
    diagonal = np.clip(np.diag(compliance), 1e-6, None)
    raw = 1.0 / diagonal
    return raw, connected_mass, connectivity_penalty


@lru_cache(maxsize=16)
def _reference_properties(grid_size: int) -> np.ndarray:
    full = np.ones((grid_size, grid_size), dtype=bool)
    reference, _, _ = _raw_properties(full)
    return np.maximum(reference, 1e-6)


def _single_sample_terms(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    raw, connected_mass, connectivity_penalty = _raw_properties(mask)
    # Normalize against a fraction of the full-solid plate so the dataset spans
    # a more useful range than near-zero values only.
    reference = np.maximum(0.1 * _reference_properties(mask.shape[0]), 1e-6)
    scaled = np.log1p(raw) / np.log1p(reference)
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled.astype(np.float32), float(connected_mass), float(connectivity_penalty)


def mechanical_terms(occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
    device = occupancy.device
    occupancy = occupancy.float()
    regularizers = topology_regularizers(occupancy)

    batch_masks = occupancy.detach().cpu().numpy()[:, 0] > 0.5
    properties: list[np.ndarray] = []
    connected_mass: list[float] = []
    connectivity_penalty: list[float] = []

    for mask in batch_masks:
        mask = mask.copy()
        mask[0, :] = True
        mask[-1, :] = True
        sample_props, sample_connected_mass, sample_connectivity_penalty = (
            _single_sample_terms(mask)
        )
        properties.append(sample_props)
        connected_mass.append(sample_connected_mass)
        connectivity_penalty.append(sample_connectivity_penalty)

    return {
        "properties": torch.tensor(
            np.stack(properties), device=device, dtype=torch.float32
        ),
        "surface": regularizers["surface"],
        "connectivity_penalty": torch.tensor(
            connectivity_penalty, device=device, dtype=torch.float32
        ),
        "connected_mass": torch.tensor(
            connected_mass, device=device, dtype=torch.float32
        ),
        "occupancy_mass": regularizers["occupancy_mass"],
    }
