from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


AXIAL_SPRING_NEIGHBORS = ((1, 0), (0, 1))
DIAGONAL_SPRING_NEIGHBORS = ((1, 1), (1, -1))
GRID_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_STIFFNESS_SCALE = 0.25


@dataclass(frozen=True)
class GridFEMData:
    edge_sources: torch.Tensor
    edge_targets: torch.Tensor
    edge_flat_indices: torch.Tensor
    edge_base_values: torch.Tensor
    transform: torch.Tensor
    free_count: int


def threshold_occupancy(
    occupancy: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    return (occupancy >= threshold).to(dtype=torch.float32)


def binarization_penalty(occupancy: torch.Tensor) -> torch.Tensor:
    return (occupancy * (1.0 - occupancy)).mean(dim=(1, 2, 3))


def _cross_neighbor_max(values: torch.Tensor) -> torch.Tensor:
    padded = F.pad(values, (1, 1, 1, 1), mode="constant", value=0.0)
    up = padded[:, :, :-2, 1:-1]
    down = padded[:, :, 2:, 1:-1]
    left = padded[:, :, 1:-1, :-2]
    right = padded[:, :, 1:-1, 2:]
    return torch.maximum(torch.maximum(up, down), torch.maximum(left, right))


def interface_length(occupancy: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(occupancy[:, :, :, 1:] - occupancy[:, :, :, :-1]).mean(dim=(1, 2, 3))
    dy = torch.abs(occupancy[:, :, 1:, :] - occupancy[:, :, :-1, :]).mean(dim=(1, 2, 3))
    return dx + dy


def topology_regularizers(occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
    occupancy_mass = occupancy.mean(dim=(1, 2, 3))
    top = torch.zeros_like(occupancy)
    top[:, :, 0, :] = 1.0

    reach = occupancy * top
    for _ in range(occupancy.shape[-2]):
        reach = occupancy * _cross_neighbor_max(reach)

    bridge_fraction = reach[:, :, -1, :].mean(dim=(1, 2))
    connectivity_penalty = (1.0 - bridge_fraction).clamp(0.0, 1.0)
    return {
        "surface": interface_length(occupancy),
        "connectivity_penalty": connectivity_penalty,
        "connected_mass": bridge_fraction,
        "occupancy_mass": occupancy_mass,
    }


def _spring_matrix(dr: int, dc: int, scale: float = 1.0) -> torch.Tensor:
    direction = torch.tensor([float(dc), float(-dr)], dtype=torch.float32)
    length = torch.linalg.vector_norm(direction)
    unit = direction / length
    axial = torch.outer(unit, unit)
    stiffness = scale / length
    return stiffness * torch.cat(
        [
            torch.cat([axial, -axial], dim=1),
            torch.cat([-axial, axial], dim=1),
        ],
        dim=0,
    )


def _node_index(width: int, row: int, col: int) -> int:
    return row * width + col


@lru_cache(maxsize=16)
def _grid_fem_data(height: int, width: int) -> GridFEMData:
    node_count = height * width
    dof_count = node_count * 2

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_flat_indices: list[list[int]] = []
    edge_base_values: list[list[float]] = []

    for row in range(height):
        for col in range(width):
            source = _node_index(width, row, col)
            for dr, dc in AXIAL_SPRING_NEIGHBORS:
                nr = row + dr
                nc = col + dc
                if nr >= height or nc >= width:
                    continue
                target = _node_index(width, nr, nc)
                element = _spring_matrix(dr, dc)
                dofs = [2 * source, 2 * source + 1, 2 * target, 2 * target + 1]
                edge_sources.append(source)
                edge_targets.append(target)
                edge_flat_indices.append(
                    [a * dof_count + b for a in dofs for b in dofs]
                )
                edge_base_values.append(element.reshape(-1).tolist())

            for dr, dc in DIAGONAL_SPRING_NEIGHBORS:
                nr = row + dr
                nc = col + dc
                if nr >= height or nc < 0 or nc >= width:
                    continue
                target = _node_index(width, nr, nc)
                element = _spring_matrix(dr, dc, scale=DIAGONAL_STIFFNESS_SCALE)
                dofs = [2 * source, 2 * source + 1, 2 * target, 2 * target + 1]
                edge_sources.append(source)
                edge_targets.append(target)
                edge_flat_indices.append(
                    [a * dof_count + b for a in dofs for b in dofs]
                )
                edge_base_values.append(element.reshape(-1).tolist())

    free_count = (height - 2) * width * 2
    transform = torch.zeros((dof_count, free_count + 3), dtype=torch.float32)
    free_col = 0
    for row in range(1, height - 1):
        for col in range(width):
            node = _node_index(width, row, col)
            transform[2 * node, free_col] = 1.0
            transform[2 * node + 1, free_col + 1] = 1.0
            free_col += 2

    center_x = (width - 1) / 2.0
    for col in range(width):
        node = _node_index(width, 0, col)
        transform[2 * node, free_count + 0] = 1.0
        transform[2 * node + 1, free_count + 1] = 1.0
        transform[2 * node + 1, free_count + 2] = float(col) - center_x

    return GridFEMData(
        edge_sources=torch.tensor(edge_sources, dtype=torch.long),
        edge_targets=torch.tensor(edge_targets, dtype=torch.long),
        edge_flat_indices=torch.tensor(edge_flat_indices, dtype=torch.long),
        edge_base_values=torch.tensor(edge_base_values, dtype=torch.float32),
        transform=transform,
        free_count=free_count,
    )


def _assemble_stiffness(occupancy: torch.Tensor) -> tuple[torch.Tensor, GridFEMData]:
    batch, _, height, width = occupancy.shape
    data = _grid_fem_data(height, width)
    device = occupancy.device
    dtype = occupancy.dtype

    sources = data.edge_sources.to(device)
    targets = data.edge_targets.to(device)
    flat_indices = data.edge_flat_indices.to(device)
    base_values = data.edge_base_values.to(device=device, dtype=dtype)

    occ_flat = occupancy.reshape(batch, -1)
    edge_weights = occ_flat[:, sources] * occ_flat[:, targets]

    dof_count = height * width * 2
    stiffness_flat = torch.zeros(
        (batch, dof_count * dof_count), device=device, dtype=dtype
    )
    contributions = (edge_weights[:, :, None] * base_values[None, :, :]).reshape(
        batch, -1
    )
    stiffness_flat.scatter_add_(
        1, flat_indices.reshape(1, -1).expand(batch, -1), contributions
    )
    stiffness = stiffness_flat.view(batch, dof_count, dof_count)
    return 0.5 * (stiffness + stiffness.transpose(1, 2)), data


def _condensed_stiffness(occupancy: torch.Tensor) -> torch.Tensor:
    stiffness, data = _assemble_stiffness(occupancy)
    device = occupancy.device
    dtype = occupancy.dtype

    transform = data.transform.to(device=device, dtype=dtype)
    reduced = torch.matmul(
        transform.transpose(0, 1).unsqueeze(0), torch.matmul(stiffness, transform)
    )

    free_count = data.free_count
    plate = reduced[:, free_count:, free_count:]
    if free_count == 0:
        return plate

    free = reduced[:, :free_count, :free_count]
    coupling = reduced[:, :free_count, free_count:]
    free_trace = free.diagonal(dim1=1, dim2=2).sum(dim=1)
    regularizer = (1e-4 + 1e-4 * free_trace / max(free_count, 1)).to(dtype=dtype)
    eye = torch.eye(free_count, device=device, dtype=dtype).unsqueeze(0)
    stabilized_free = free + regularizer[:, None, None] * eye
    solved = torch.linalg.solve(stabilized_free, coupling)
    return plate - torch.matmul(coupling.transpose(1, 2), solved)


def _raw_properties(occupancy: torch.Tensor) -> torch.Tensor:
    condensed = _condensed_stiffness(occupancy)
    batch = condensed.shape[0]
    dtype = condensed.dtype
    device = condensed.device

    trace = condensed.diagonal(dim1=1, dim2=2).sum(dim=1)
    regularizer = (1e-4 + 1e-4 * trace / 3.0).to(dtype=dtype)
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    compliance = torch.linalg.inv(condensed + regularizer[:, None, None] * eye)
    diagonal = compliance.diagonal(dim1=1, dim2=2).clamp_min(1e-6)
    return 1.0 / diagonal.reshape(batch, 3)


@lru_cache(maxsize=16)
def _reference_properties(height: int, width: int) -> torch.Tensor:
    full = torch.ones((1, 1, height, width), dtype=torch.float32)
    return _raw_properties(full)[0].cpu().clamp_min(1e-6)


def mechanical_terms(occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
    occupancy = occupancy.float().clamp(0.0, 1.0)
    regularizers = topology_regularizers(occupancy)
    raw_properties = _raw_properties(occupancy)
    reference = _reference_properties(occupancy.shape[-2], occupancy.shape[-1]).to(
        device=occupancy.device, dtype=occupancy.dtype
    )
    properties = torch.log1p(raw_properties) / torch.log1p(0.1 * reference[None, :])
    properties = properties.clamp(0.0, 1.0)

    return {
        "properties": properties,
        "surface": regularizers["surface"],
        "connectivity_penalty": regularizers["connectivity_penalty"],
        "connected_mass": regularizers["connected_mass"],
        "occupancy_mass": regularizers["occupancy_mass"],
    }
