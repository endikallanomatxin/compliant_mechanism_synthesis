from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from compliant_mechanism_synthesis.dataset import load_offline_dataset
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Structures,
)
from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    mechanical_terms,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import (
    enforce_role_adjacency_constraints,
    symmetrize_matrix,
)


@dataclass(frozen=True)
class TrainingBatch:
    source_structures: Structures
    initial_structures: Structures
    target_stiffness: torch.Tensor
    oracle_structures: Structures
    oracle_analyses: Analyses
    initial_times: torch.Tensor

    @property
    def noisy_structures(self) -> Structures:
        return self.source_structures


@dataclass(frozen=True)
class EvalSplit:
    train_cases: OptimizedCases
    eval_cases: OptimizedCases | None


def load_training_cases(dataset_path: str) -> OptimizedCases:
    optimized_cases, _ = load_offline_dataset(dataset_path)
    return optimized_cases


def _dataset_position_statistics(
    optimized_cases: OptimizedCases,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = optimized_cases.optimized_structures.positions
    _, _, free_mask = role_masks(optimized_cases.optimized_structures.roles)
    free_positions = positions[free_mask]
    mean = free_positions.mean(dim=0, keepdim=True).view(1, 1, 3)
    std = free_positions.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-3)
    std = std.view(1, 1, 3)
    return mean, std


def _dataset_adjacency_statistics(
    optimized_cases: OptimizedCases,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjacency = optimized_cases.optimized_structures.adjacency
    fixed_mask, mobile_mask, free_mask = role_masks(
        optimized_cases.optimized_structures.roles
    )
    diagonal = torch.eye(adjacency.shape[1], device=adjacency.device, dtype=torch.bool)
    diagonal = diagonal.unsqueeze(0)
    free_free = (free_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)) & ~diagonal
    free_fixed = (free_mask.unsqueeze(-1) & fixed_mask.unsqueeze(-2)) | (
        fixed_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    free_mobile = (free_mask.unsqueeze(-1) & mobile_mask.unsqueeze(-2)) | (
        mobile_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    masks = [free_free, free_fixed, free_mobile]
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for mask in masks:
        values = adjacency[mask]
        means.append(values.mean())
        stds.append(values.std(unbiased=False).clamp_min(1e-3))
    mean = torch.stack(means)
    std = torch.stack(stds)
    return mean, std


def dataset_noise_statistics(
    optimized_cases: OptimizedCases,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    position_mean, position_std = _dataset_position_statistics(optimized_cases)
    adjacency_mean, adjacency_std = _dataset_adjacency_statistics(optimized_cases)
    return position_mean, position_std, adjacency_mean, adjacency_std


def _role_pair_adjacency_matrices(
    roles: torch.Tensor,
    adjacency_mean: torch.Tensor,
    adjacency_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_mask, mobile_mask, free_mask = role_masks(roles)
    batch_size, num_nodes = roles.shape
    mean_matrix = torch.zeros(
        batch_size,
        num_nodes,
        num_nodes,
        device=roles.device,
        dtype=adjacency_mean.dtype,
    )
    std_matrix = torch.zeros(
        batch_size, num_nodes, num_nodes, device=roles.device, dtype=adjacency_std.dtype
    )
    free_free = free_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    free_fixed = (free_mask.unsqueeze(-1) & fixed_mask.unsqueeze(-2)) | (
        fixed_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    free_mobile = (free_mask.unsqueeze(-1) & mobile_mask.unsqueeze(-2)) | (
        mobile_mask.unsqueeze(-1) & free_mask.unsqueeze(-2)
    )
    mean_matrix = mean_matrix.masked_fill(free_free, float(adjacency_mean[0].item()))
    mean_matrix = mean_matrix.masked_fill(free_fixed, float(adjacency_mean[1].item()))
    mean_matrix = mean_matrix.masked_fill(free_mobile, float(adjacency_mean[2].item()))
    std_matrix = std_matrix.masked_fill(free_free, float(adjacency_std[0].item()))
    std_matrix = std_matrix.masked_fill(free_fixed, float(adjacency_std[1].item()))
    std_matrix = std_matrix.masked_fill(free_mobile, float(adjacency_std[2].item()))
    diagonal = torch.arange(num_nodes, device=roles.device)
    mean_matrix[:, diagonal, diagonal] = 0.0
    std_matrix[:, diagonal, diagonal] = 0.0
    return mean_matrix, std_matrix


def _minimum_cost_assignment(cost: torch.Tensor) -> torch.Tensor:
    num_items = cost.shape[0]
    if num_items == 0:
        return torch.empty(0, dtype=torch.long)
    _, column_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(column_indices, dtype=torch.long)


def _match_oracle_to_source(
    source_structures: Structures,
    oracle_structures: Structures,
    oracle_analyses: Analyses,
) -> tuple[Structures, Analyses]:
    batch_size, num_nodes, _ = source_structures.positions.shape
    _, _, free_mask = role_masks(source_structures.roles)
    permutations: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        free_indices = torch.nonzero(free_mask[batch_index], as_tuple=False).squeeze(-1)
        permutation = torch.arange(num_nodes, device=source_structures.positions.device)
        if free_indices.numel() > 0:
            source_free = source_structures.positions[batch_index].index_select(
                0, free_indices
            )
            oracle_free = oracle_structures.positions[batch_index].index_select(
                0, free_indices
            )
            cost = torch.cdist(source_free, oracle_free).square()
            assignment = _minimum_cost_assignment(cost).to(
                device=source_structures.positions.device
            )
            permutation[free_indices] = free_indices.index_select(0, assignment)
        permutations.append(permutation)

    stacked_permutations = torch.stack(permutations, dim=0)
    matched_positions = torch.stack(
        [
            oracle_structures.positions[index].index_select(
                0, stacked_permutations[index]
            )
            for index in range(batch_size)
        ],
        dim=0,
    )
    matched_adjacency = torch.stack(
        [
            oracle_structures.adjacency[index]
            .index_select(0, stacked_permutations[index])
            .index_select(1, stacked_permutations[index])
            for index in range(batch_size)
        ],
        dim=0,
    )
    matched_structures = Structures(
        positions=matched_positions,
        roles=oracle_structures.roles,
        adjacency=matched_adjacency,
    )
    matched_analyses = Analyses(
        generalized_stiffness=oracle_analyses.generalized_stiffness,
        material_usage=oracle_analyses.material_usage,
        short_beam_penalty=oracle_analyses.short_beam_penalty,
        long_beam_penalty=oracle_analyses.long_beam_penalty,
        thin_beam_penalty=oracle_analyses.thin_beam_penalty,
        thick_beam_penalty=oracle_analyses.thick_beam_penalty,
        free_node_spacing_penalty=oracle_analyses.free_node_spacing_penalty,
        nodal_displacements=(
            None
            if oracle_analyses.nodal_displacements is None
            else torch.stack(
                [
                    oracle_analyses.nodal_displacements[index].index_select(
                        0, stacked_permutations[index]
                    )
                    for index in range(batch_size)
                ],
                dim=0,
            )
        ),
        edge_von_mises=(
            None
            if oracle_analyses.edge_von_mises is None
            else torch.stack(
                [
                    oracle_analyses.edge_von_mises[index]
                    .index_select(0, stacked_permutations[index])
                    .index_select(1, stacked_permutations[index])
                    for index in range(batch_size)
                ],
                dim=0,
            )
        ),
    )
    matched_structures.validate()
    matched_analyses.validate(batch_size)
    return matched_structures, matched_analyses


def match_oracle_to_source(
    source_structures: Structures,
    oracle_structures: Structures,
    oracle_analyses: Analyses,
) -> tuple[Structures, Analyses]:
    source_structures.validate()
    oracle_structures.validate()
    oracle_analyses.validate(oracle_structures.batch_size)
    if source_structures.positions.shape != oracle_structures.positions.shape:
        raise ValueError("source and oracle positions must have matching shapes")
    if source_structures.adjacency.shape != oracle_structures.adjacency.shape:
        raise ValueError("source and oracle adjacency must have matching shapes")
    if not torch.equal(source_structures.roles, oracle_structures.roles):
        raise ValueError("source and oracle must use the same node roles")
    return _match_oracle_to_source(
        source_structures=source_structures,
        oracle_structures=oracle_structures,
        oracle_analyses=oracle_analyses,
    )


def sample_noisy_structures(
    optimized_cases: OptimizedCases,
    position_mean: torch.Tensor | None = None,
    position_std: torch.Tensor | None = None,
    adjacency_mean: torch.Tensor | None = None,
    adjacency_std: torch.Tensor | None = None,
    seed: int | None = None,
) -> Structures:
    optimized_cases.validate()
    oracle_positions = optimized_cases.optimized_structures.positions
    oracle_adjacency = optimized_cases.optimized_structures.adjacency
    generator = None
    if seed is not None:
        generator = torch.Generator(device=oracle_positions.device).manual_seed(seed)
    if position_mean is None or position_std is None:
        position_mean, position_std = _dataset_position_statistics(optimized_cases)
    if adjacency_mean is None or adjacency_std is None:
        adjacency_mean, adjacency_std = _dataset_adjacency_statistics(optimized_cases)
    adjacency_mean_matrix, adjacency_std_matrix = _role_pair_adjacency_matrices(
        optimized_cases.optimized_structures.roles,
        adjacency_mean.to(device=oracle_adjacency.device, dtype=oracle_adjacency.dtype),
        adjacency_std.to(device=oracle_adjacency.device, dtype=oracle_adjacency.dtype),
    )
    sampled_positions = position_mean + position_std * torch.randn(
        oracle_positions.shape,
        generator=generator,
        device=oracle_positions.device,
        dtype=oracle_positions.dtype,
    )
    sampled_adjacency = adjacency_mean_matrix + adjacency_std_matrix * torch.randn(
        oracle_adjacency.shape,
        generator=generator,
        device=oracle_adjacency.device,
        dtype=oracle_adjacency.dtype,
    )
    noisy_positions = sampled_positions.clamp(0.0, 1.0)
    noisy_adjacency = enforce_role_adjacency_constraints(
        symmetrize_matrix(sampled_adjacency.clamp(0.0, 1.0)),
        optimized_cases.optimized_structures.roles,
    )

    _, _, free_mask = role_masks(optimized_cases.optimized_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=noisy_positions.dtype)
    noisy_positions = (
        optimized_cases.optimized_structures.positions * (1.0 - free_mask)
        + noisy_positions * free_mask
    )
    structures = Structures(
        positions=noisy_positions,
        roles=optimized_cases.optimized_structures.roles,
        adjacency=noisy_adjacency,
    )
    structures.validate()
    return structures


def _analysis_from_terms(terms: dict[str, torch.Tensor]) -> Analyses:
    return Analyses(
        generalized_stiffness=terms["generalized_stiffness"],
        material_usage=terms["material_usage"],
        short_beam_penalty=terms["short_beam_penalty"],
        long_beam_penalty=terms["long_beam_penalty"],
        thin_beam_penalty=terms["thin_beam_penalty"],
        thick_beam_penalty=terms["thick_beam_penalty"],
        free_node_spacing_penalty=terms["free_node_spacing_penalty"],
        nodal_displacements=terms["nodal_displacements"],
        edge_von_mises=terms["edge_von_mises"],
    )


def analyze_structures(
    structures: Structures,
    frame_config: Frame3DConfig | None = None,
    geometry_config: GeometryPenaltyConfig | None = None,
    profile: dict[str, float] | None = None,
) -> Analyses:
    frame_config = frame_config or Frame3DConfig()
    geometry_config = geometry_config or GeometryPenaltyConfig()
    structures.validate()
    return _analysis_from_terms(
        mechanical_terms(
            positions=structures.positions,
            roles=structures.roles,
            adjacency=structures.adjacency,
            frame_config=frame_config,
            penalty_config=geometry_config,
            profile=profile,
        )
    )


def make_training_batch(
    optimized_cases: OptimizedCases,
    position_mean: torch.Tensor | None = None,
    position_std: torch.Tensor | None = None,
    adjacency_mean: torch.Tensor | None = None,
    adjacency_std: torch.Tensor | None = None,
    seed: int | None = None,
    max_initial_time: float | None = None,
) -> TrainingBatch:
    source_structures = sample_noisy_structures(
        optimized_cases=optimized_cases,
        position_mean=position_mean,
        position_std=position_std,
        adjacency_mean=adjacency_mean,
        adjacency_std=adjacency_std,
        seed=seed,
    )
    oracle_structures, oracle_analyses = _match_oracle_to_source(
        source_structures=source_structures,
        oracle_structures=optimized_cases.optimized_structures,
        oracle_analyses=optimized_cases.last_analyses,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator(
            device=source_structures.positions.device
        ).manual_seed(seed + 10_000)
    initial_times = torch.rand(
        (optimized_cases.optimized_structures.batch_size,),
        generator=generator,
        device=source_structures.positions.device,
        dtype=source_structures.positions.dtype,
    )
    if max_initial_time is not None:
        if not 0.0 <= max_initial_time < 1.0:
            raise ValueError("max_initial_time must be in [0.0, 1.0)")
        initial_times = initial_times * max_initial_time
    interpolation = initial_times[:, None, None]
    initial_positions = torch.lerp(
        source_structures.positions,
        oracle_structures.positions,
        interpolation,
    )
    initial_adjacency = torch.lerp(
        source_structures.adjacency,
        oracle_structures.adjacency,
        interpolation,
    )
    initial_structures = Structures(
        positions=initial_positions,
        roles=source_structures.roles,
        adjacency=enforce_role_adjacency_constraints(
            initial_adjacency,
            source_structures.roles,
        ),
    )
    initial_structures.validate()
    return TrainingBatch(
        source_structures=source_structures,
        initial_structures=initial_structures,
        target_stiffness=optimized_cases.target_stiffness,
        oracle_structures=oracle_structures,
        oracle_analyses=oracle_analyses,
        initial_times=initial_times,
    )


def select_batch(
    optimized_cases: OptimizedCases,
    batch_indices: torch.Tensor,
) -> OptimizedCases:
    if batch_indices.ndim != 1:
        raise ValueError("batch_indices must be one-dimensional")
    return optimized_cases.index_select(batch_indices)


def split_train_eval_cases(
    optimized_cases: OptimizedCases,
    eval_fraction: float,
) -> EvalSplit:
    optimized_cases.validate()
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0.0, 1.0)")

    num_cases = optimized_cases.optimized_structures.batch_size
    if num_cases <= 1 or eval_fraction == 0.0:
        return EvalSplit(train_cases=optimized_cases, eval_cases=None)

    eval_count = int(math.ceil(num_cases * eval_fraction))
    eval_count = min(max(eval_count, 1), num_cases - 1)
    split_index = num_cases - eval_count
    all_indices = torch.arange(
        num_cases,
        device=optimized_cases.optimized_structures.positions.device,
        dtype=torch.long,
    )
    train_indices = all_indices[:split_index]
    eval_indices = all_indices[split_index:]
    return EvalSplit(
        train_cases=select_batch(optimized_cases, train_indices),
        eval_cases=select_batch(optimized_cases, eval_indices),
    )


def iter_training_batches(
    optimized_cases: OptimizedCases,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    optimized_cases.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    index_device = optimized_cases.optimized_structures.positions.device
    indices = torch.arange(
        optimized_cases.optimized_structures.batch_size,
        device=index_device,
        dtype=torch.long,
    )
    if shuffle:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=index_device).manual_seed(seed)
        permutation = torch.randperm(
            indices.shape[0], device=index_device, generator=generator
        )
        indices = indices.index_select(0, permutation)
    for start in range(0, indices.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield select_batch(optimized_cases, batch_indices)
