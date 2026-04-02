from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from compliant_mechanism_synthesis.common import (
    distance_affinity,
    max_length_gate,
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    enforce_role_adjacency_constraints,
    role_masks,
    symmetrize_adjacency,
)
from compliant_mechanism_synthesis.scaling import (
    CharacteristicScales,
    center_positions,
    normalize_generalized_response_matrix,
    normalize_generalized_stiffness_matrix,
    normalize_material,
    normalize_stress,
    normalize_translations,
)


@dataclass(frozen=True)
class FrameFEMConfig:
    young_modulus: float = 210e9
    workspace_size: float = 0.2
    r_max: float = 1.5e-3
    stiffness_regularization: float = 1e-4
    yield_stress: float = 250e6


@dataclass(frozen=True)
class GeometryRegularizationConfig:
    min_length: float = 1e-3
    max_length: float = 2e-2
    min_diameter: float = 2e-4
    max_diameter: float = 2e-3
    min_free_node_spacing: float = 5e-3


def characteristic_scales(
    config: FrameFEMConfig | None = None,
) -> CharacteristicScales:
    config = config or FrameFEMConfig()
    area_scale = math.pi * config.r_max**2
    length_scale = 0.5 * config.workspace_size
    force_scale = config.young_modulus * area_scale
    return CharacteristicScales(
        length=length_scale,
        force=force_scale,
        moment=force_scale * length_scale,
        stress=config.yield_stress,
        material=area_scale * length_scale,
    )


def threshold_connectivity(
    adjacency: torch.Tensor, roles: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    thresholded = symmetrize_adjacency(
        (adjacency >= threshold).to(dtype=adjacency.dtype)
    )
    return enforce_role_adjacency_constraints(thresholded, roles)


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


def _physical_positions(
    positions: torch.Tensor,
    config: FrameFEMConfig,
) -> torch.Tensor:
    return positions * config.workspace_size


def _rigid_transforms(positions: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    batch_size, num_nodes, _ = positions.shape
    free_mask = roles == ROLE_FREE
    mobile_mask = roles == ROLE_MOBILE
    free_count = int(free_mask[0].sum().item())
    mobile_count = int(mobile_mask[0].sum().item())
    reduced_dofs = 3 * free_count + 3

    node_ids = torch.arange(num_nodes, device=positions.device).expand(batch_size, -1)
    free_indices = node_ids.masked_select(free_mask).view(batch_size, free_count)
    mobile_indices = node_ids.masked_select(mobile_mask).view(batch_size, mobile_count)

    transform = torch.zeros(
        (batch_size, 3 * num_nodes, reduced_dofs),
        device=positions.device,
        dtype=positions.dtype,
    )
    batch_ids = torch.arange(batch_size, device=positions.device)[:, None]
    free_slots = torch.arange(free_count, device=positions.device)[None, :]

    for dof in range(3):
        transform[
            batch_ids,
            3 * free_indices + dof,
            3 * free_slots + dof,
        ] = 1.0

    mobile_positions = positions[batch_ids, mobile_indices]
    centroid = mobile_positions.mean(dim=1, keepdim=True)
    dx = mobile_positions[..., 0] - centroid[..., 0]
    dy = mobile_positions[..., 1] - centroid[..., 1]
    base = 3 * free_count

    transform[batch_ids, 3 * mobile_indices + 0, base + 0] = 1.0
    transform[batch_ids, 3 * mobile_indices + 0, base + 2] = -dy
    transform[batch_ids, 3 * mobile_indices + 1, base + 1] = 1.0
    transform[batch_ids, 3 * mobile_indices + 1, base + 2] = dx
    transform[batch_ids, 3 * mobile_indices + 2, base + 2] = 1.0
    return transform


def assemble_global_stiffness(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> torch.Tensor:
    config = config or FrameFEMConfig()
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    physical_positions = _physical_positions(positions, config)
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
    delta = physical_positions[:, edge_j] - physical_positions[:, edge_i]
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
    _, num_nodes, _ = positions.shape
    full = assemble_global_stiffness(positions, adjacency, config=config)
    free_count = num_nodes - 4
    transforms = _rigid_transforms(_physical_positions(positions, config), roles)
    reduced = transforms.transpose(1, 2) @ full @ transforms
    return reduced, transforms


def _stabilized_reduced_system(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced, transforms = _reduced_stiffness(positions, roles, adjacency, config)
    _, reduced_size, _ = reduced.shape
    eye = torch.eye(reduced_size, device=reduced.device, dtype=reduced.dtype)
    trace = reduced.diagonal(dim1=1, dim2=2).sum(dim=1)
    stabilized = (
        reduced
        + (config.stiffness_regularization * (1.0 + trace / max(reduced_size, 1)))[
            :, None, None
        ]
        * eye[None, :, :]
    )
    return stabilized, transforms


def _mobile_load_cases(
    positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    reduced_size = 3 * (positions.shape[1] - 4) + 3
    free_count = positions.shape[1] - 4
    mobile_base = 3 * free_count
    load_cases = torch.zeros((3, reduced_size), device=device, dtype=dtype)
    load_cases[0, mobile_base + 0] = 1.0
    load_cases[1, mobile_base + 1] = 1.0
    load_cases[2, mobile_base + 2] = 1.0
    return load_cases


def _solve_load_cases(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stabilized, transforms = _stabilized_reduced_system(
        positions, roles, adjacency, config
    )
    batch_size = positions.shape[0]
    load_cases = _mobile_load_cases(
        positions,
        device=stabilized.device,
        dtype=stabilized.dtype,
    )
    rhs = load_cases.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
    reduced_displacements = torch.linalg.solve(stabilized, rhs)
    return stabilized, transforms, reduced_displacements


def _response_and_stiffness_from_solution(
    positions: torch.Tensor,
    stabilized: torch.Tensor,
    reduced_displacements: torch.Tensor,
    config: FrameFEMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = positions.shape[0]
    free_count = positions.shape[1] - 4
    mobile_base = 3 * free_count
    response_matrix = reduced_displacements[:, mobile_base : mobile_base + 3, :]
    response_matrix = 0.5 * (response_matrix + response_matrix.transpose(1, 2))
    response_eye = torch.eye(3, device=stabilized.device, dtype=stabilized.dtype)
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


def _stress_response_fields(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    transforms: torch.Tensor,
    reduced_displacements: torch.Tensor,
    config: FrameFEMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_positions = _physical_positions(positions, config)
    batch_size, num_nodes, _ = positions.shape
    edge_i, edge_j = _cached_edge_index_pairs(num_nodes)
    edge_i = edge_i.to(positions.device)
    edge_j = edge_j.to(positions.device)
    activations = adjacency[:, edge_i, edge_j]
    delta = physical_positions[:, edge_j] - physical_positions[:, edge_i]
    length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-4)
    radius = (config.r_max * activations).clamp_min(1e-8)
    area = (math.pi * radius.square()).clamp_min(1e-12)
    inertia = (math.pi * radius.pow(4) / 4.0).clamp_min(1e-16)
    local = _frame_local_stiffness(length, area, inertia, config.young_modulus)
    transform = _frame_transform(delta, length)
    full_displacements = transforms @ reduced_displacements
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
    edge_displacements = full_displacements[:, dofs.reshape(-1), :].view(
        batch_size,
        -1,
        6,
        3,
    )
    local_displacements = torch.einsum("beij,bejk->beik", transform, edge_displacements)
    local_forces = torch.einsum("beij,bejk->beik", local, local_displacements)

    axial_force = torch.maximum(
        local_forces[..., 0, :].abs(),
        local_forces[..., 3, :].abs(),
    )
    moment_i = local_forces[..., 2, :].abs()
    moment_j = local_forces[..., 5, :].abs()
    axial_stress = axial_force / area.unsqueeze(-1)
    bending_i = moment_i * radius.unsqueeze(-1) / inertia.unsqueeze(-1)
    bending_j = moment_j * radius.unsqueeze(-1) / inertia.unsqueeze(-1)
    edge_stress = torch.maximum(axial_stress + bending_i, axial_stress + bending_j)
    edge_stress = edge_stress.masked_fill(activations.unsqueeze(-1) <= 1e-6, 0.0)
    max_edge_stress = edge_stress.max(dim=-1).values

    weighted_stress = activations * max_edge_stress
    stress_sum = torch.zeros(
        (batch_size, num_nodes), device=positions.device, dtype=positions.dtype
    )
    weight_sum = torch.zeros(
        (batch_size, num_nodes), device=positions.device, dtype=positions.dtype
    )
    scatter_i = edge_i.unsqueeze(0).expand(batch_size, -1)
    scatter_j = edge_j.unsqueeze(0).expand(batch_size, -1)
    stress_sum.scatter_add_(1, scatter_i, weighted_stress)
    stress_sum.scatter_add_(1, scatter_j, weighted_stress)
    weight_sum.scatter_add_(1, scatter_i, activations)
    weight_sum.scatter_add_(1, scatter_j, activations)
    nodal_stress = stress_sum / weight_sum.clamp_min(1e-6)
    yield_ratio = max_edge_stress / config.yield_stress
    yield_penalty = (activations * yield_ratio.square()).sum(dim=1) / activations.sum(
        dim=1
    ).clamp_min(1e-6)
    return nodal_stress, yield_penalty


def mechanical_response_fields(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> dict[str, torch.Tensor]:
    config = config or FrameFEMConfig()
    scales = characteristic_scales(config)
    stabilized, transforms, reduced_displacements = _solve_load_cases(
        positions,
        roles,
        adjacency,
        config,
    )
    response_matrix, stiffness_matrix = _response_and_stiffness_from_solution(
        positions,
        stabilized,
        reduced_displacements,
        config,
    )
    full_displacements = transforms @ reduced_displacements
    batch_size, num_nodes, _ = positions.shape
    full_displacements = full_displacements.view(batch_size, num_nodes, 3, 3).permute(
        0, 3, 1, 2
    )
    translations = full_displacements[..., :2]
    nodal_stress, yield_penalty = _stress_response_fields(
        positions,
        adjacency,
        transforms,
        reduced_displacements,
        config,
    )
    normalized_response_matrix = normalize_generalized_response_matrix(
        response_matrix,
        scales,
    )
    normalized_stiffness_matrix = normalize_generalized_stiffness_matrix(
        stiffness_matrix,
        scales,
    )
    normalized_translations = normalize_translations(translations, scales)
    normalized_nodal_stress = normalize_stress(nodal_stress, scales)
    return {
        "response_matrix": response_matrix,
        "stiffness_matrix": stiffness_matrix,
        "translations": translations,
        "nodal_stress": nodal_stress,
        "normalized_response_matrix": normalized_response_matrix,
        "normalized_stiffness_matrix": normalized_stiffness_matrix,
        "normalized_translations": normalized_translations,
        "normalized_nodal_stress": normalized_nodal_stress,
        "yield_stress_penalty": yield_penalty,
    }


def effective_response(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fields = mechanical_response_fields(positions, roles, adjacency, config)
    return fields["response_matrix"], fields["stiffness_matrix"]


def load_case_deformations(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fields = mechanical_response_fields(positions, roles, adjacency, config)
    return fields["translations"], fields["response_matrix"]


def stiffness_and_deformations(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = mechanical_response_fields(positions, roles, adjacency, config)
    return (
        fields["stiffness_matrix"],
        fields["translations"],
        fields["response_matrix"],
    )


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


def fixed_mobile_connectivity_penalty(
    roles: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    fixed, mobile, _ = role_masks(roles)
    fixed_seed = fixed.to(dtype=adjacency.dtype)
    mobile_mask = mobile.to(dtype=adjacency.dtype)
    fixed_reach = _graph_reachability(adjacency, fixed_seed)
    mobile_support = (fixed_reach * mobile_mask).sum(dim=1) / mobile_mask.sum(
        dim=1
    ).clamp_min(1.0)
    return (1.0 - mobile_support).clamp(0.0, 1.0)


def beam_material(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
) -> torch.Tensor:
    config = config or FrameFEMConfig()
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    physical_positions = _physical_positions(positions, config)
    edge_i, edge_j = _cached_edge_index_pairs(positions.shape[1])
    edge_i = edge_i.to(positions.device)
    edge_j = edge_j.to(positions.device)
    edge_lengths = torch.linalg.vector_norm(
        physical_positions[:, edge_j] - physical_positions[:, edge_i], dim=-1
    ).clamp_min(1e-4)
    edge_radius = config.r_max * adjacency[:, edge_i, edge_j]
    area = math.pi * edge_radius.square()
    return (edge_lengths * area).sum(dim=1)


def connectivity_sparsity(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    edge_i, edge_j = _cached_edge_index_pairs(adjacency.shape[1])
    edge_i = edge_i.to(adjacency.device)
    edge_j = edge_j.to(adjacency.device)
    return adjacency[:, edge_i, edge_j].mean(dim=1)


def geometric_regularization_terms(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    geometry_config: GeometryRegularizationConfig,
    frame_config: FrameFEMConfig | None = None,
) -> dict[str, torch.Tensor]:
    frame_config = frame_config or FrameFEMConfig()
    scales = characteristic_scales(frame_config)
    adjacency = symmetrize_adjacency(adjacency.float().clamp(0.0, 1.0))
    physical_positions = _physical_positions(positions, frame_config)
    edge_i, edge_j = _cached_edge_index_pairs(positions.shape[1])
    edge_i = edge_i.to(positions.device)
    edge_j = edge_j.to(positions.device)

    activations = adjacency[:, edge_i, edge_j]
    lengths = torch.linalg.vector_norm(
        physical_positions[:, edge_j] - physical_positions[:, edge_i], dim=-1
    ).clamp_min(1e-4)
    diameters = 2.0 * frame_config.r_max * activations
    diameter_scale = 2.0 * frame_config.r_max
    normalizer = activations.sum(dim=1).clamp_min(1e-6)

    short = (activations * (geometry_config.min_length - lengths).clamp_min(0.0)).sum(
        dim=1
    ) / (normalizer * scales.length)
    long = (activations * (lengths - geometry_config.max_length).clamp_min(0.0)).sum(
        dim=1
    ) / (normalizer * scales.length)
    normalized_diameter = (diameters / geometry_config.min_diameter).clamp(0.0, 1.0)
    thin_profile = normalized_diameter * (1.0 - normalized_diameter)
    thin = thin_profile.sum(dim=1) / normalizer
    thick = ((diameters - geometry_config.max_diameter).clamp_min(0.0).square()).sum(
        dim=1
    ) / (normalizer * diameter_scale**2)

    free_mask = roles == ROLE_FREE
    free_positions = physical_positions
    pairwise = torch.linalg.vector_norm(
        free_positions[:, :, None, :] - free_positions[:, None, :, :], dim=-1
    )
    pair_mask = free_mask[:, :, None] & free_mask[:, None, :]
    upper_mask = torch.triu(
        torch.ones(pairwise.shape[-2:], device=pairwise.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0)
    active_pairs = pair_mask & upper_mask
    spacing_violation = (geometry_config.min_free_node_spacing - pairwise).clamp_min(
        0.0
    )
    spacing_penalty = (spacing_violation * active_pairs.to(dtype=pairwise.dtype)).sum(
        dim=(1, 2)
    ) / (
        active_pairs.to(dtype=pairwise.dtype).sum(dim=(1, 2)).clamp_min(1.0)
        * scales.length
    )

    centered_positions = center_positions(positions)
    soft_domain_violation = (centered_positions.abs() - 1.0).clamp_min(0.0)
    soft_domain_penalty = (
        soft_domain_violation.square().sum(dim=-1) * free_mask.to(dtype=positions.dtype)
    ).sum(dim=1) / free_mask.to(dtype=positions.dtype).sum(dim=1).clamp_min(1.0)
    return {
        "short_beam_penalty": short,
        "long_beam_penalty": long,
        "thin_diameter_penalty": thin,
        "thick_diameter_penalty": thick,
        "node_spacing_penalty": spacing_penalty,
        "soft_domain_penalty": soft_domain_penalty,
    }


def mechanical_terms(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: FrameFEMConfig | None = None,
    geometry_config: GeometryRegularizationConfig | None = None,
) -> dict[str, torch.Tensor]:
    config = config or FrameFEMConfig()
    adjacency = enforce_role_adjacency_constraints(
        adjacency.float().clamp(0.0, 1.0),
        roles,
    )
    response_fields = mechanical_response_fields(positions, roles, adjacency, config)
    if geometry_config is None:
        zeros = torch.zeros(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        geometry_terms = {
            "short_beam_penalty": zeros,
            "long_beam_penalty": zeros,
            "thin_diameter_penalty": zeros,
            "thick_diameter_penalty": zeros,
            "node_spacing_penalty": zeros,
            "soft_domain_penalty": zeros,
        }
    else:
        geometry_terms = geometric_regularization_terms(
            positions,
            roles,
            adjacency,
            geometry_config,
            frame_config=config,
        )
    return {
        "response_matrix": response_fields["response_matrix"],
        "stiffness_matrix": response_fields["stiffness_matrix"],
        "translations": response_fields["translations"],
        "nodal_stress": response_fields["nodal_stress"],
        "normalized_response_matrix": response_fields["normalized_response_matrix"],
        "normalized_stiffness_matrix": response_fields["normalized_stiffness_matrix"],
        "normalized_translations": response_fields["normalized_translations"],
        "normalized_nodal_stress": response_fields["normalized_nodal_stress"],
        "yield_stress_penalty": response_fields["yield_stress_penalty"],
        "connectivity_penalty": connectivity_penalty(roles, adjacency),
        "fixed_mobile_connectivity_penalty": fixed_mobile_connectivity_penalty(
            roles,
            adjacency,
        ),
        "sparsity": connectivity_sparsity(adjacency),
        "material": beam_material(positions, adjacency, config),
        "normalized_material": normalize_material(
            beam_material(positions, adjacency, config),
            characteristic_scales(config),
        ),
        **geometry_terms,
    }


def refine_connectivity(
    adjacency: torch.Tensor,
    positions: torch.Tensor,
    roles: torch.Tensor,
    delta_scores: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    gate = max_length_gate(positions, max_distance=0.10, transition_width=0.05)
    updated = adjacency + step_size * (
        torch.tanh(delta_scores) * gate - adjacency * (1.0 - gate)
    )
    return enforce_role_adjacency_constraints(updated.clamp(0.0, 1.0), roles)
