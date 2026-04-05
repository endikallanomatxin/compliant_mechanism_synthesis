from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from compliant_mechanism_synthesis.roles import NodeRole, role_masks
from compliant_mechanism_synthesis.tensor_ops import symmetrize_matrix, upper_triangle_edge_index


@dataclass(frozen=True)
class Frame3DConfig:
    young_modulus: float = 210e9
    poisson_ratio: float = 0.30
    workspace_size: float = 0.2
    radius_max: float = 1.5e-3
    global_regularization: float = 1e-8
    free_dof_regularization: float = 1e-7
    minimum_length: float = 1e-4

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass(frozen=True)
class GeometryPenaltyConfig:
    min_length: float = 2.0e-3
    max_length: float = 3.5e-2
    min_radius: float = 2.0e-4
    max_radius: float = 1.5e-3
    min_free_spacing: float = 4.0e-3


def normalize_generalized_stiffness(
    matrix: torch.Tensor,
    config: Frame3DConfig | None = None,
) -> torch.Tensor:
    config = config or Frame3DConfig()
    length_scale = config.workspace_size
    force_scale = config.young_modulus * math.pi * config.radius_max**2
    energy_scale = force_scale * length_scale
    coordinate_scale = torch.tensor(
        [length_scale, length_scale, length_scale, 1.0, 1.0, 1.0],
        device=matrix.device,
        dtype=matrix.dtype,
    )
    scale_matrix = coordinate_scale[:, None] * coordinate_scale[None, :]
    return matrix * scale_matrix / energy_scale


def denormalize_generalized_stiffness(
    matrix: torch.Tensor,
    config: Frame3DConfig | None = None,
) -> torch.Tensor:
    config = config or Frame3DConfig()
    length_scale = config.workspace_size
    force_scale = config.young_modulus * math.pi * config.radius_max**2
    energy_scale = force_scale * length_scale
    coordinate_scale = torch.tensor(
        [length_scale, length_scale, length_scale, 1.0, 1.0, 1.0],
        device=matrix.device,
        dtype=matrix.dtype,
    )
    scale_matrix = coordinate_scale[:, None] * coordinate_scale[None, :]
    return matrix * energy_scale / scale_matrix


def beam_radii(adjacency: torch.Tensor, config: Frame3DConfig) -> torch.Tensor:
    return adjacency.clamp_min(0.0) * config.radius_max


def _section_properties(radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    area = math.pi * radius.square()
    bending_inertia = math.pi * radius.pow(4) / 4.0
    polar_inertia = math.pi * radius.pow(4) / 2.0
    return area, bending_inertia, bending_inertia, polar_inertia


def _choose_reference_axis(x_axis: torch.Tensor) -> torch.Tensor:
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=x_axis.device, dtype=x_axis.dtype)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=x_axis.device, dtype=x_axis.dtype)
    z_ref = z_axis.expand_as(x_axis)
    y_ref = y_axis.expand_as(x_axis)
    use_y = x_axis[..., 2].abs() > 0.9
    return torch.where(use_y.unsqueeze(-1), y_ref, z_ref)


def _local_axes(delta: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    x_axis = delta / length.unsqueeze(-1)
    reference = _choose_reference_axis(x_axis)
    y_axis = torch.linalg.cross(reference, x_axis, dim=-1)
    y_axis = y_axis / torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-2)


def _frame_local_stiffness(
    length: torch.Tensor,
    area: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    polar_inertia: torch.Tensor,
    config: Frame3DConfig,
) -> torch.Tensor:
    axial = config.young_modulus * area / length
    torsion = config.shear_modulus * polar_inertia / length
    by = 12.0 * config.young_modulus * iy / length.pow(3)
    bz = 12.0 * config.young_modulus * iz / length.pow(3)
    cy = 6.0 * config.young_modulus * iy / length.square()
    cz = 6.0 * config.young_modulus * iz / length.square()
    dy = 4.0 * config.young_modulus * iy / length
    dz = 4.0 * config.young_modulus * iz / length
    ey = 2.0 * config.young_modulus * iy / length
    ez = 2.0 * config.young_modulus * iz / length

    k = torch.zeros(length.shape + (12, 12), device=length.device, dtype=length.dtype)
    k[..., 0, 0] = axial
    k[..., 0, 6] = -axial
    k[..., 6, 0] = -axial
    k[..., 6, 6] = axial

    k[..., 1, 1] = bz
    k[..., 1, 5] = cz
    k[..., 1, 7] = -bz
    k[..., 1, 11] = cz

    k[..., 2, 2] = by
    k[..., 2, 4] = -cy
    k[..., 2, 8] = -by
    k[..., 2, 10] = -cy

    k[..., 3, 3] = torsion
    k[..., 3, 9] = -torsion

    k[..., 4, 2] = -cy
    k[..., 4, 4] = dy
    k[..., 4, 8] = cy
    k[..., 4, 10] = ey

    k[..., 5, 1] = cz
    k[..., 5, 5] = dz
    k[..., 5, 7] = -cz
    k[..., 5, 11] = ez

    k[..., 7, 1] = -bz
    k[..., 7, 5] = -cz
    k[..., 7, 7] = bz
    k[..., 7, 11] = -cz

    k[..., 8, 2] = -by
    k[..., 8, 4] = cy
    k[..., 8, 8] = by
    k[..., 8, 10] = cy

    k[..., 9, 3] = -torsion
    k[..., 9, 9] = torsion

    k[..., 10, 2] = -cy
    k[..., 10, 4] = ey
    k[..., 10, 8] = cy
    k[..., 10, 10] = dy

    k[..., 11, 1] = cz
    k[..., 11, 5] = ez
    k[..., 11, 7] = -cz
    k[..., 11, 11] = dz
    return symmetrize_matrix(k)


def _frame_transform(rotation: torch.Tensor) -> torch.Tensor:
    transform = torch.zeros(
        rotation.shape[:-2] + (12, 12),
        device=rotation.device,
        dtype=rotation.dtype,
    )
    for block in range(4):
        start = 3 * block
        transform[..., start : start + 3, start : start + 3] = rotation
    return transform


def assemble_global_stiffness(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    config: Frame3DConfig | None = None,
) -> torch.Tensor:
    config = config or Frame3DConfig()
    adjacency = symmetrize_matrix(adjacency.float().clamp_min(0.0))
    batch_size, num_nodes, spatial_dim = positions.shape
    if spatial_dim != 3:
        raise ValueError("assemble_global_stiffness expects 3D positions")

    edge_i, edge_j = upper_triangle_edge_index(num_nodes, positions.device)
    delta = (positions[:, edge_j] - positions[:, edge_i]) * config.workspace_size
    length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(config.minimum_length)
    radius = beam_radii(adjacency[:, edge_i, edge_j], config)
    area, iy, iz, polar_inertia = _section_properties(radius)
    local = _frame_local_stiffness(length, area, iy, iz, polar_inertia, config)
    rotation = _local_axes(delta, length)
    transform = _frame_transform(rotation)
    element = transform.transpose(-1, -2) @ local @ transform

    stiffness = torch.zeros(
        (batch_size, 6 * num_nodes, 6 * num_nodes),
        device=positions.device,
        dtype=positions.dtype,
    )
    dofs = torch.stack(
        [
            6 * edge_i + 0,
            6 * edge_i + 1,
            6 * edge_i + 2,
            6 * edge_i + 3,
            6 * edge_i + 4,
            6 * edge_i + 5,
            6 * edge_j + 0,
            6 * edge_j + 1,
            6 * edge_j + 2,
            6 * edge_j + 3,
            6 * edge_j + 4,
            6 * edge_j + 5,
        ],
        dim=1,
    )
    batch_index = torch.arange(batch_size, device=positions.device).view(-1, 1, 1, 1)
    row_index = dofs.view(1, -1, 12, 1)
    col_index = dofs.view(1, -1, 1, 12)
    stiffness.index_put_(
        (
            batch_index.expand(-1, dofs.shape[0], 12, 12).reshape(-1),
            row_index.expand(batch_size, -1, 12, 12).reshape(-1),
            col_index.expand(batch_size, -1, 12, 12).reshape(-1),
        ),
        element.reshape(-1),
        accumulate=True,
    )

    diagonal = config.global_regularization * torch.ones(
        (batch_size, 6 * num_nodes),
        device=positions.device,
        dtype=positions.dtype,
    )
    stiffness = stiffness + torch.diag_embed(diagonal)
    return symmetrize_matrix(stiffness)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(vector.shape[:-1], device=vector.device, dtype=vector.dtype)
    return torch.stack(
        [
            torch.stack([zeros, -vector[..., 2], vector[..., 1]], dim=-1),
            torch.stack([vector[..., 2], zeros, -vector[..., 0]], dim=-1),
            torch.stack([-vector[..., 1], vector[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )


def _reduction_transform(positions: torch.Tensor, roles: torch.Tensor) -> tuple[torch.Tensor, int]:
    batch_size, num_nodes, _ = positions.shape
    fixed_mask, mobile_mask, free_mask = role_masks(roles)

    if not torch.equal(fixed_mask, fixed_mask[:1].expand_as(fixed_mask)):
        raise ValueError("all batch items must share the same role layout")
    if not torch.equal(mobile_mask, mobile_mask[:1].expand_as(mobile_mask)):
        raise ValueError("all batch items must share the same role layout")

    free_indices = torch.nonzero(free_mask[0], as_tuple=False).squeeze(-1)
    mobile_indices = torch.nonzero(mobile_mask[0], as_tuple=False).squeeze(-1)
    free_count = int(free_indices.numel())
    reduced_dofs = 6 * free_count + 6

    transform = torch.zeros(
        (batch_size, 6 * num_nodes, reduced_dofs),
        device=positions.device,
        dtype=positions.dtype,
    )

    if free_count > 0:
        free_slots = torch.arange(free_count, device=positions.device)
        for dof in range(6):
            transform[:, 6 * free_indices + dof, 6 * free_slots + dof] = 1.0

    mobile_positions = positions[:, mobile_indices] if mobile_indices.numel() else positions[:, :0]
    centroid = mobile_positions.mean(dim=1, keepdim=True)
    offsets = mobile_positions - centroid
    rigid_base = 6 * free_count
    rigid_map = torch.zeros(
        (batch_size, mobile_indices.numel(), 6, 6),
        device=positions.device,
        dtype=positions.dtype,
    )
    rigid_map[..., :3, :3] = torch.eye(3, device=positions.device, dtype=positions.dtype)
    # A rigid body's nodal translations are the body translation plus the
    # infinitesimal rotation crossed with each node's offset from the centroid.
    rigid_map[..., :3, 3:] = -_skew(offsets)
    rigid_map[..., 3:, 3:] = torch.eye(3, device=positions.device, dtype=positions.dtype)

    for local_node, node_index in enumerate(mobile_indices.tolist()):
        start = 6 * node_index
        transform[:, start : start + 6, rigid_base : rigid_base + 6] = rigid_map[:, local_node]

    return transform, free_count


def effective_output_stiffness(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    config: Frame3DConfig | None = None,
) -> torch.Tensor:
    config = config or Frame3DConfig()
    full = assemble_global_stiffness(positions, adjacency, config=config)
    transform, free_count = _reduction_transform(
        positions * config.workspace_size,
        roles,
    )
    reduced = transform.transpose(-1, -2) @ full @ transform
    rigid_base = 6 * free_count
    output_block = reduced[:, rigid_base:, rigid_base:]
    if free_count == 0:
        return symmetrize_matrix(output_block)

    free_block = reduced[:, :rigid_base, :rigid_base]
    coupling = reduced[:, :rigid_base, rigid_base:]
    regularized_free = free_block + torch.diag_embed(
        config.free_dof_regularization
        * torch.ones(
            (positions.shape[0], rigid_base),
            device=positions.device,
            dtype=positions.dtype,
        )
    )
    relaxed_coupling = torch.linalg.solve(regularized_free, coupling)
    effective = output_block - coupling.transpose(-1, -2) @ relaxed_coupling
    return symmetrize_matrix(effective)


def material_usage(
    positions: torch.Tensor,
    adjacency: torch.Tensor,
    config: Frame3DConfig | None = None,
) -> torch.Tensor:
    config = config or Frame3DConfig()
    adjacency = symmetrize_matrix(adjacency.float().clamp_min(0.0))
    _, num_nodes, _ = positions.shape
    edge_i, edge_j = upper_triangle_edge_index(num_nodes, positions.device)
    delta = (positions[:, edge_j] - positions[:, edge_i]) * config.workspace_size
    length = torch.linalg.vector_norm(delta, dim=-1)
    radius = beam_radii(adjacency[:, edge_i, edge_j], config)
    area, _, _, _ = _section_properties(radius)
    return (length * area).sum(dim=-1)


def geometry_penalties(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    penalty_config: GeometryPenaltyConfig | None = None,
    frame_config: Frame3DConfig | None = None,
) -> dict[str, torch.Tensor]:
    penalty_config = penalty_config or GeometryPenaltyConfig()
    frame_config = frame_config or Frame3DConfig()
    adjacency = symmetrize_matrix(adjacency.float().clamp_min(0.0))
    _, num_nodes, _ = positions.shape
    edge_i, edge_j = upper_triangle_edge_index(num_nodes, positions.device)
    delta = (positions[:, edge_j] - positions[:, edge_i]) * frame_config.workspace_size
    length = torch.linalg.vector_norm(delta, dim=-1)
    edge_activation = adjacency[:, edge_i, edge_j]
    active_mask = edge_activation > 1e-6
    radius = beam_radii(edge_activation, frame_config)

    safe_active_count = active_mask.sum(dim=-1).clamp_min(1)
    short_penalty = torch.where(
        active_mask,
        (penalty_config.min_length - length).clamp_min(0.0).square(),
        torch.zeros_like(length),
    ).sum(dim=-1) / safe_active_count
    long_penalty = torch.where(
        active_mask,
        (length - penalty_config.max_length).clamp_min(0.0).square(),
        torch.zeros_like(length),
    ).sum(dim=-1) / safe_active_count
    thin_penalty = torch.where(
        active_mask,
        (penalty_config.min_radius - radius).clamp_min(0.0).square(),
        torch.zeros_like(radius),
    ).sum(dim=-1) / safe_active_count
    thick_penalty = torch.where(
        active_mask,
        (radius - penalty_config.max_radius).clamp_min(0.0).square(),
        torch.zeros_like(radius),
    ).sum(dim=-1) / safe_active_count

    _, _, free_mask = role_masks(roles)
    free_positions = positions
    pairwise = torch.linalg.vector_norm(
        (free_positions[:, :, None, :] - free_positions[:, None, :, :]) * frame_config.workspace_size,
        dim=-1,
    )
    valid_pairs = free_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    valid_pairs = torch.triu(valid_pairs, diagonal=1)
    spacing_violation = (penalty_config.min_free_spacing - pairwise).clamp_min(0.0).square()
    free_pair_count = valid_pairs.sum(dim=(1, 2)).clamp_min(1)
    spacing_penalty = spacing_violation.masked_fill(~valid_pairs, 0.0).sum(dim=(1, 2)) / free_pair_count

    return {
        "short_beam_penalty": short_penalty,
        "long_beam_penalty": long_penalty,
        "thin_beam_penalty": thin_penalty,
        "thick_beam_penalty": thick_penalty,
        "free_node_spacing_penalty": spacing_penalty,
    }


def mechanical_terms(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    frame_config: Frame3DConfig | None = None,
    penalty_config: GeometryPenaltyConfig | None = None,
) -> dict[str, torch.Tensor]:
    frame_config = frame_config or Frame3DConfig()
    stiffness = effective_output_stiffness(
        positions=positions,
        roles=roles,
        adjacency=adjacency,
        config=frame_config,
    )
    penalties = geometry_penalties(
        positions=positions,
        roles=roles,
        adjacency=adjacency,
        penalty_config=penalty_config,
        frame_config=frame_config,
    )
    return {
        "generalized_stiffness": stiffness,
        "material_usage": material_usage(positions, adjacency, config=frame_config),
        **penalties,
    }
