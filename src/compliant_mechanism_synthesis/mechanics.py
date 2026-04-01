from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    adjacency_logits,
    logits_to_adjacency,
    role_masks,
    symmetrize_adjacency,
)


@dataclass(frozen=True)
class FrameFEMConfig:
    young_modulus: float = 1.0
    r_max: float = 0.06
    stiffness_regularization: float = 1e-4


@dataclass(frozen=True)
class GeometryRegularizationConfig:
    min_length: float = 0.08
    max_length: float = 0.85
    min_diameter: float = 0.01
    max_diameter: float = 0.10


def threshold_connectivity(
    adjacency: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    return symmetrize_adjacency((adjacency >= threshold).to(dtype=adjacency.dtype))


def _edge_index_pairs(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.triu_indices(num_nodes, num_nodes, offset=1)
    return pairs[0], pairs[1]


@lru_cache(maxsize=32)
def _cached_edge_index_pairs(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _edge_index_pairs(num_nodes)


def _frame_local_stiffness(
    length: torch.Tensor,
    area: torch.Tensor,
    inertia: torch.Tensor,
    young_modulus: float,
) -> torch.Tensor:
    ea_over_l = young_modulus * area / length
    ei = young_modulus * inertia
    l2 = length.square()
    l3 = l2 * length

    k = torch.zeros(length.shape + (6, 6), device=length.device, dtype=length.dtype)
    k[..., 0, 0] = ea_over_l
    k[..., 0, 3] = -ea_over_l
    k[..., 3, 0] = -ea_over_l
    k[..., 3, 3] = ea_over_l

    k[..., 1, 1] = 12.0 * ei / l3
    k[..., 1, 2] = 6.0 * ei / l2
    k[..., 1, 4] = -12.0 * ei / l3
    k[..., 1, 5] = 6.0 * ei / l2

    k[..., 2, 1] = 6.0 * ei / l2
    k[..., 2, 2] = 4.0 * ei / length
    k[..., 2, 4] = -6.0 * ei / l2
    k[..., 2, 5] = 2.0 * ei / length

    k[..., 4, 1] = -12.0 * ei / l3
    k[..., 4, 2] = -6.0 * ei / l2
    k[..., 4, 4] = 12.0 * ei / l3
    k[..., 4, 5] = -6.0 * ei / l2

    k[..., 5, 1] = 6.0 * ei / l2
    k[..., 5, 2] = 2.0 * ei / length
    k[..., 5, 4] = -6.0 * ei / l2
    k[..., 5, 5] = 4.0 * ei / length
    return k


def _frame_transform(delta: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    c = delta[..., 0] / length
    s = delta[..., 1] / length
    transform = torch.zeros(
        delta.shape[:-1] + (6, 6), device=delta.device, dtype=delta.dtype
    )
    transform[..., 0, 0] = c
    transform[..., 0, 1] = s
    transform[..., 1, 0] = -s
    transform[..., 1, 1] = c
    transform[..., 2, 2] = 1.0
    transform[..., 3, 3] = c
    transform[..., 3, 4] = s
    transform[..., 4, 3] = -s
    transform[..., 4, 4] = c
    transform[..., 5, 5] = 1.0
    return transform


def _rigid_transform_for_sample(
    positions: torch.Tensor, roles: torch.Tensor
) -> torch.Tensor:
    num_nodes = positions.shape[0]
    free_indices = torch.where(roles == ROLE_FREE)[0].tolist()
    mobile_indices = torch.where(roles == ROLE_MOBILE)[0]
    free_count = len(free_indices)
    reduced_dofs = 3 * free_count + 3

    transform = torch.zeros(
        (3 * num_nodes, reduced_dofs), device=positions.device, dtype=positions.dtype
    )

    for free_slot, node_idx in enumerate(free_indices):
        transform[3 * node_idx + 0, 3 * free_slot + 0] = 1.0
        transform[3 * node_idx + 1, 3 * free_slot + 1] = 1.0
        transform[3 * node_idx + 2, 3 * free_slot + 2] = 1.0

    centroid = positions[mobile_indices].mean(dim=0)
    base = 3 * free_count
    for node_idx in mobile_indices.tolist():
        dx = positions[node_idx, 0] - centroid[0]
        dy = positions[node_idx, 1] - centroid[1]
        transform[3 * node_idx + 0, base + 0] = 1.0
        transform[3 * node_idx + 0, base + 2] = -dy
        transform[3 * node_idx + 1, base + 1] = 1.0
        transform[3 * node_idx + 1, base + 2] = dx
        transform[3 * node_idx + 2, base + 2] = 1.0

    return transform


def assemble_global_stiffness(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> torch.Tensor:
    config = config or FrameFEMConfig()
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    batch_size, num_nodes, _ = positions.shape
    edge_i, edge_j = _cached_edge_index_pairs(num_nodes)
    edge_i = edge_i.to(positions.device)
    edge_j = edge_j.to(positions.device)

    stiffness = torch.zeros(
        (batch_size, 3 * num_nodes, 3 * num_nodes),
        device=positions.device,
        dtype=positions.dtype,
    )

    activations = adjacency[:, edge_i, edge_j]
    delta = positions[:, edge_j] - positions[:, edge_i]
    length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-4)
    radius = config.r_max * activations
    area = math.pi * radius.square()
    inertia = math.pi * radius.pow(4) / 4.0
    local = _frame_local_stiffness(length, area, inertia, config.young_modulus)
    transform = _frame_transform(delta, length)
    element = transform.transpose(-1, -2) @ local @ transform

    dofs = torch.stack(
        [
            3 * edge_i + 0,
            3 * edge_i + 1,
            3 * edge_i + 2,
            3 * edge_j + 0,
            3 * edge_j + 1,
            3 * edge_j + 2,
        ],
        dim=1,
    )
    batch_index = torch.arange(batch_size, device=positions.device).view(-1, 1, 1, 1)
    row_index = dofs.view(1, -1, 6, 1)
    col_index = dofs.view(1, -1, 1, 6)
    stiffness.index_put_(
        (
            batch_index.expand(-1, dofs.shape[0], 6, 6).reshape(-1),
            row_index.expand(batch_size, -1, 6, 6).reshape(-1),
            col_index.expand(batch_size, -1, 6, 6).reshape(-1),
        ),
        element.reshape(-1),
        accumulate=True,
    )

    return 0.5 * (stiffness + stiffness.transpose(1, 2))


def _reduced_stiffness(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_nodes, _ = positions.shape
    full = assemble_global_stiffness(positions, adjacency, config=config)
    free_count = num_nodes - 4
    reduced_size = 3 * free_count + 3
    reduced = torch.zeros(
        (batch_size, reduced_size, reduced_size),
        device=positions.device,
        dtype=positions.dtype,
    )
    transforms = []

    for batch_idx in range(batch_size):
        transform = _rigid_transform_for_sample(positions[batch_idx], roles[batch_idx])
        transforms.append(transform)
        reduced[batch_idx] = transform.transpose(0, 1) @ full[batch_idx] @ transform

    return reduced, torch.stack(transforms, dim=0)


def effective_response(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = config or FrameFEMConfig()
    reduced, _ = _reduced_stiffness(positions, roles, adjacency, config)
    batch_size, reduced_size, _ = reduced.shape
    eye = torch.eye(reduced_size, device=reduced.device, dtype=reduced.dtype)
    trace = reduced.diagonal(dim1=1, dim2=2).sum(dim=1)
    stabilized = (
        reduced
        + (config.stiffness_regularization * (1.0 + trace / max(reduced_size, 1)))[
            :, None, None
        ]
        * eye[None, :, :]
    )

    free_count = positions.shape[1] - 4
    mobile_base = 3 * free_count
    load_cases = torch.zeros(
        (3, reduced_size), device=reduced.device, dtype=reduced.dtype
    )
    load_cases[0, mobile_base + 0] = 1.0
    load_cases[1, mobile_base + 1] = 1.0
    load_cases[2, mobile_base + 2] = 1.0

    rhs = load_cases.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
    solved = torch.linalg.solve(stabilized, rhs)

    response_matrix = solved[:, mobile_base : mobile_base + 3, :]
    response_matrix = 0.5 * (response_matrix + response_matrix.transpose(1, 2))
    response_eye = torch.eye(3, device=reduced.device, dtype=reduced.dtype)
    response_trace = response_matrix.diagonal(dim1=1, dim2=2).sum(dim=1)
    stabilized_response = (
        response_matrix
        + (config.stiffness_regularization * (1.0 + response_trace / 3.0))[
            :, None, None
        ]
        * response_eye[None, :, :]
    )
    stiffness_matrix = torch.linalg.solve(
        stabilized_response,
        response_eye[None, :, :].expand(batch_size, -1, -1),
    )
    stiffness_matrix = 0.5 * (stiffness_matrix + stiffness_matrix.transpose(1, 2))
    return response_matrix, stiffness_matrix


def _graph_reachability(adjacency: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    reach = seeds
    diagonal = torch.eye(
        adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype
    )
    adjacency = symmetrize_adjacency(adjacency) + diagonal.unsqueeze(0)
    for _ in range(adjacency.shape[-1]):
        propagated = torch.amax(reach.unsqueeze(1) * adjacency, dim=2)
        reach = torch.maximum(reach, propagated)
    return reach.clamp(0.0, 1.0)


def connectivity_penalty(
    roles: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    fixed, mobile, _ = role_masks(roles)
    fixed_seed = fixed.to(dtype=adjacency.dtype)
    mobile_seed = mobile.to(dtype=adjacency.dtype)
    fixed_reach = _graph_reachability(adjacency, fixed_seed)
    mobile_reach = _graph_reachability(adjacency, mobile_seed)

    expanded_fixed = fixed_reach
    expanded_mobile = mobile_reach
    penalties = []
    dense = symmetrize_adjacency(adjacency) + torch.eye(
        adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype
    ).unsqueeze(0)
    for _ in range(adjacency.shape[-1]):
        overlap = torch.amax(expanded_fixed * expanded_mobile, dim=1)
        penalties.append(1.0 - overlap)
        expanded_fixed = torch.maximum(
            expanded_fixed, torch.amax(expanded_fixed.unsqueeze(1) * dense, dim=2)
        )
        expanded_mobile = torch.maximum(
            expanded_mobile, torch.amax(expanded_mobile.unsqueeze(1) * dense, dim=2)
        )
    return torch.stack(penalties, dim=0).mean(dim=0).clamp(0.0, 1.0)


def beam_material(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> torch.Tensor:
    config = config or FrameFEMConfig()
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    edge_i, edge_j = _cached_edge_index_pairs(positions.shape[1])
    edge_i = edge_i.to(positions.device)
    edge_j = edge_j.to(positions.device)
    edge_lengths = torch.linalg.vector_norm(
        positions[:, edge_j] - positions[:, edge_i], dim=-1
    ).clamp_min(1e-4)
    edge_radius = config.r_max * adjacency[:, edge_i, edge_j]
    area = math.pi * edge_radius.square()
    return (edge_lengths * area).sum(dim=1)


def geometric_regularization_terms(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    geometry_config: GeometryRegularizationConfig,
    frame_config: FrameFEMConfig | None = None,
) -> dict[str, torch.Tensor]:
    frame_config = frame_config or FrameFEMConfig()
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    edge_i, edge_j = _cached_edge_index_pairs(positions.shape[1])
    edge_i = edge_i.to(positions.device)
    edge_j = edge_j.to(positions.device)

    activations = adjacency[:, edge_i, edge_j]
    lengths = torch.linalg.vector_norm(
        positions[:, edge_j] - positions[:, edge_i], dim=-1
    ).clamp_min(1e-4)
    diameters = 2.0 * frame_config.r_max * activations
    normalizer = activations.sum(dim=1).clamp_min(1e-6)

    short = (activations * (geometry_config.min_length - lengths).clamp_min(0.0)).sum(
        dim=1
    ) / normalizer
    long = (activations * (lengths - geometry_config.max_length).clamp_min(0.0)).sum(
        dim=1
    ) / normalizer
    thin = (
        activations * (geometry_config.min_diameter - diameters).clamp_min(0.0)
    ).sum(dim=1) / normalizer
    thick = (
        activations * (diameters - geometry_config.max_diameter).clamp_min(0.0)
    ).sum(dim=1) / normalizer
    return {
        "short_beam_penalty": short,
        "long_beam_penalty": long,
        "thin_diameter_penalty": thin,
        "thick_diameter_penalty": thick,
    }


def mechanical_terms(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
    geometry_config: GeometryRegularizationConfig | None = None,
) -> dict[str, torch.Tensor]:
    config = config or FrameFEMConfig()
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    response_matrix, stiffness_matrix = effective_response(
        positions, roles, adjacency, config
    )
    if geometry_config is None:
        zeros = torch.zeros(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        geometry_terms = {
            "short_beam_penalty": zeros,
            "long_beam_penalty": zeros,
            "thin_diameter_penalty": zeros,
            "thick_diameter_penalty": zeros,
        }
    else:
        geometry_terms = geometric_regularization_terms(
            positions,
            adjacency,
            geometry_config,
            frame_config=config,
        )
    return {
        "response_matrix": response_matrix,
        "stiffness_matrix": stiffness_matrix,
        "connectivity_penalty": connectivity_penalty(roles, adjacency),
        "material": beam_material(positions, adjacency, config),
        **geometry_terms,
    }


def refine_connectivity(
    adjacency: torch.Tensor, delta_scores: torch.Tensor, step_size: float
) -> torch.Tensor:
    logits = adjacency_logits(adjacency)
    return logits_to_adjacency(logits + step_size * delta_scores)
