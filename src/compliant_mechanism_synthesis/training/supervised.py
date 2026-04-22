from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.dataset import load_offline_dataset
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Scaffolds,
    Structures,
)
from compliant_mechanism_synthesis.losses import (
    generalized_stiffness_error,
    log_generalized_stiffness_error,
    stress_violation_terms,
)
from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    mechanical_terms,
)
from compliant_mechanism_synthesis.models import (
    FlowPrediction,
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import (
    enforce_role_adjacency_constraints,
    symmetrize_matrix,
)
from compliant_mechanism_synthesis.utils import resolve_torch_device


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    dataset_path: str
    device: str = "auto"
    batch_size: int = 224
    log_every_steps: int = 10
    eval_every_steps: int = 100
    max_grad_norm: float = 1.0
    num_steps: int = 120_000
    learning_rate: float = 4e-5
    warmup_steps: int = 1000
    min_learning_rate: float = 4e-6
    eval_fraction: float = 0.02
    use_style_token: bool = True
    style_sample_dropout: float = 0.15
    style_token_dropout: float = 0.10
    position_loss_weight: float = 1.0
    adjacency_loss_weight: float = 0.5

    # KL DIVERGENCE LOSS FOR STYLE TOKEN
    # It helps the development of the latent representation by limiting its
    # deviation from a standard Gaussian distribution
    style_kl_loss_weight: float = 6e-4
    style_kl_anneal_steps: int = 1_000

    # STIFFNESS LOSS
    # It deviates from the pure supervised learning setup, but it
    # - helps guide the model towards more realistic solutions
    # - reduces over-reliance on the style token
    stiffness_loss_weight: float = 0.000001
    # Delaying the start of the stiffness loss:
    # - avoids the messy gradients overwhelming an inmature model
    # - reduces de computational cost because it avoids simulating the
    #   resulting structure.
    stiffness_loss_delay_steps: int = 2_000
    stiffness_loss_warmup_steps: int = 18_000
    
    # STRESS LOSS
    stress_loss_weight: float = 0.001
    allowable_von_mises: float = 250e6
    stress_activation_threshold: float = 0.15
    stress_loss_delay_steps: int = 2_000
    stress_loss_warmup_steps: int = 18_000

    checkpoint_path: str | None = None
    logdir: str = "runs/supervised"
    seed: int = 7


@dataclass(frozen=True)
class SupervisedBatch:
    source_structures: Structures
    flow_structures: Structures
    target_stiffness: torch.Tensor
    oracle_structures: Structures
    oracle_analyses: Analyses
    current_analyses: Analyses
    flow_times: torch.Tensor
    target_position_velocity: torch.Tensor
    target_adjacency_velocity: torch.Tensor

    @property
    def noisy_structures(self) -> Structures:
        return self.source_structures


@dataclass(frozen=True)
class SupervisedTrainingSummary:
    history: dict[str, list[float]]
    checkpoint_path: Path


@dataclass(frozen=True)
class EvalSplit:
    train_cases: OptimizedCases
    eval_cases: OptimizedCases | None


@dataclass(frozen=True)
class EvalRunResult:
    losses: dict[str, float]
    metrics: dict[str, float]
    num_batches: int
    num_cases: int


def _scheduled_learning_rate(
    step: int,
    config: SupervisedTrainingConfig,
) -> float:
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    warmup_steps = min(config.warmup_steps, config.num_steps)
    if warmup_steps > 0 and step < warmup_steps:
        warmup_fraction = (step + 1) / warmup_steps
        return config.min_learning_rate + warmup_fraction * (
            config.learning_rate - config.min_learning_rate
        )

    if config.num_steps <= warmup_steps + 1:
        return config.learning_rate

    anneal_progress = (step - warmup_steps) / (config.num_steps - warmup_steps - 1)
    anneal_progress = min(max(anneal_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * anneal_progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def _synchronize_device_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _style_kl_weight(step: int, config: SupervisedTrainingConfig) -> float:
    if config.style_kl_loss_weight < 0.0:
        raise ValueError("style_kl_loss_weight must be non-negative")
    if config.style_kl_anneal_steps < 0:
        raise ValueError("style_kl_anneal_steps must be non-negative")
    if config.style_kl_loss_weight == 0.0:
        return 0.0
    if config.style_kl_anneal_steps == 0:
        return config.style_kl_loss_weight
    progress = min(max((step + 1) / config.style_kl_anneal_steps, 0.0), 1.0)
    return config.style_kl_loss_weight * progress


def _stiffness_loss_weight_effective(
    step: int,
    config: SupervisedTrainingConfig,
) -> float:
    return _delayed_warmup_weight(
        step=step,
        final_weight=config.stiffness_loss_weight,
        delay_steps=config.stiffness_loss_delay_steps,
        warmup_steps=config.stiffness_loss_warmup_steps,
        weight_name="stiffness_loss_weight",
    )


def _stress_loss_weight_effective(
    step: int,
    config: SupervisedTrainingConfig,
) -> float:
    return _delayed_warmup_weight(
        step=step,
        final_weight=config.stress_loss_weight,
        delay_steps=config.stress_loss_delay_steps,
        warmup_steps=config.stress_loss_warmup_steps,
        weight_name="stress_loss_weight",
    )


def _delayed_warmup_weight(
    *,
    step: int,
    final_weight: float,
    delay_steps: int,
    warmup_steps: int,
    weight_name: str,
) -> float:
    if final_weight < 0.0:
        raise ValueError(f"{weight_name} must be non-negative")
    if delay_steps < 0:
        raise ValueError(
            f"{weight_name.replace('weight', 'delay_steps')} must be non-negative"
        )
    if warmup_steps < 0:
        raise ValueError(
            f"{weight_name.replace('weight', 'warmup_steps')} must be non-negative"
        )
    if final_weight == 0.0:
        return 0.0
    if step < delay_steps:
        return 0.0
    if warmup_steps == 0:
        return final_weight
    warmup_progress = (step - delay_steps + 1) / warmup_steps
    warmup_progress = min(max(warmup_progress, 0.0), 1.0)
    return final_weight * warmup_progress


def load_supervised_cases(dataset_path: str) -> OptimizedCases:
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


def make_supervised_batch(
    optimized_cases: OptimizedCases,
    position_mean: torch.Tensor | None = None,
    position_std: torch.Tensor | None = None,
    adjacency_mean: torch.Tensor | None = None,
    adjacency_std: torch.Tensor | None = None,
    seed: int | None = None,
    profile: dict[str, float] | None = None,
) -> SupervisedBatch:
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
    flow_times = torch.rand(
        (optimized_cases.optimized_structures.batch_size,),
        generator=generator,
        device=source_structures.positions.device,
        dtype=source_structures.positions.dtype,
    )
    # Sample supervision on points along the straight path from a full Gaussian
    # structure-noise source to the optimized oracle, and train the model to
    # predict the constant flow velocity of that straight transport path.
    interpolation = flow_times[:, None, None]
    flow_positions = torch.lerp(
        source_structures.positions,
        oracle_structures.positions,
        interpolation,
    )
    flow_adjacency = torch.lerp(
        source_structures.adjacency,
        oracle_structures.adjacency,
        interpolation,
    )
    flow_structures = Structures(
        positions=flow_positions,
        roles=source_structures.roles,
        adjacency=enforce_role_adjacency_constraints(
            flow_adjacency, source_structures.roles
        ),
    )
    # These mechanics features condition the model input but do not need a
    # gradient path: the flow batch is sampled from offline data, not from the
    # model's own predictions.
    with torch.no_grad():
        current_analyses = analyze_structures(flow_structures, profile=profile)
    return SupervisedBatch(
        source_structures=source_structures,
        flow_structures=flow_structures,
        target_stiffness=optimized_cases.target_stiffness,
        oracle_structures=oracle_structures,
        oracle_analyses=oracle_analyses,
        current_analyses=current_analyses,
        flow_times=flow_times,
        target_position_velocity=oracle_structures.positions
        - source_structures.positions,
        target_adjacency_velocity=oracle_structures.adjacency
        - source_structures.adjacency,
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


def iter_supervised_batches(
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


def _position_velocity_loss(
    prediction: FlowPrediction, batch: SupervisedBatch
) -> torch.Tensor:
    _, _, free_mask = role_masks(batch.flow_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=prediction.position_velocity.dtype)
    error = (
        (prediction.position_velocity - batch.target_position_velocity).square()
        * free_mask
    ).sum(dim=(1, 2))
    denom = free_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return error / denom


def _adjacency_velocity_loss(
    prediction: FlowPrediction, batch: SupervisedBatch
) -> torch.Tensor:
    return (
        (prediction.adjacency_velocity - batch.target_adjacency_velocity)
        .square()
        .mean(dim=(1, 2))
    )


def _training_losses(
    prediction: FlowPrediction,
    batch: SupervisedBatch,
    config: SupervisedTrainingConfig,
    step: int,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    position_error = _position_velocity_loss(prediction, batch)
    adjacency_error = _adjacency_velocity_loss(prediction, batch)
    stiffness_weight_effective = _stiffness_loss_weight_effective(step, config)
    stress_weight_effective = _stress_loss_weight_effective(step, config)
    endpoint_analyses = None
    if stiffness_weight_effective > 0.0 or stress_weight_effective > 0.0:
        _, _, free_mask = role_masks(batch.flow_structures.roles)
        free_mask = free_mask.unsqueeze(-1).to(dtype=prediction.position_velocity.dtype)
        remaining = (1.0 - batch.flow_times)[:, None, None]
        estimated_positions = (
            batch.flow_structures.positions
            + remaining * prediction.position_velocity * free_mask
        ).clamp(0.0, 1.0)
        estimated_adjacency = enforce_role_adjacency_constraints(
            (
                batch.flow_structures.adjacency
                + remaining * prediction.adjacency_velocity
            ).clamp(0.0, 1.0),
            batch.flow_structures.roles,
        )
        estimated = Structures(
            positions=estimated_positions,
            roles=batch.flow_structures.roles,
            adjacency=estimated_adjacency,
        )
        endpoint_analyses = analyze_structures(estimated, profile=profile)
    if stiffness_weight_effective > 0.0:
        stiffness_error = log_generalized_stiffness_error(
            endpoint_analyses.generalized_stiffness,
            batch.target_stiffness,
        )
    else:
        stiffness_error = position_error.new_zeros(position_error.shape)
    if stress_weight_effective > 0.0:
        if endpoint_analyses is None or endpoint_analyses.edge_von_mises is None:
            raise ValueError("endpoint analyses must provide edge_von_mises")
        stress_violation, mean_stress_ratio, max_stress_ratio = stress_violation_terms(
            edge_von_mises=endpoint_analyses.edge_von_mises,
            adjacency=estimated_adjacency,
            allowable_von_mises=config.allowable_von_mises,
            stress_activation_threshold=config.stress_activation_threshold,
        )
    else:
        stress_violation = position_error.new_zeros(position_error.shape)
        mean_stress_ratio = position_error.new_zeros(position_error.shape)
        max_stress_ratio = position_error.new_zeros(position_error.shape)
    if prediction.style_kl is None or not config.use_style_token:
        style_kl = position_error.new_zeros(position_error.shape)
    else:
        style_kl = prediction.style_kl
    kl_weight = _style_kl_weight(step, config)
    loss_contributions = {
        "position_error_loss_contribution": config.position_loss_weight
        * position_error,
        "adjacency_error_loss_contribution": config.adjacency_loss_weight
        * adjacency_error,
        "stiffness_error_loss_contribution": stiffness_weight_effective
        * stiffness_error,
        "stress_loss_contribution": stress_weight_effective * stress_violation,
        "style_kl_loss_contribution": kl_weight * style_kl,
    }
    total_loss = (
        loss_contributions["position_error_loss_contribution"]
        + loss_contributions["adjacency_error_loss_contribution"]
        + loss_contributions["stiffness_error_loss_contribution"]
        + loss_contributions["style_kl_loss_contribution"]
    )
    metrics = {
        "position_error": position_error.mean(),
        "adjacency_error": adjacency_error.mean(),
        "stiffness_error": stiffness_error.mean(),
        "stiffness_loss_weight_effective": position_error.new_tensor(
            stiffness_weight_effective
        ),
        "stress_violation": stress_violation.mean(),
        "mean_stress_ratio": mean_stress_ratio.mean(),
        "max_stress_ratio": max_stress_ratio.mean(),
        "stress_loss_weight_effective": position_error.new_tensor(
            stress_weight_effective
        ),
        "style_kl": style_kl.mean(),
        "style_kl_weight": position_error.new_tensor(kl_weight),
    }
    loss_summary = {name: value.mean() for name, value in loss_contributions.items()}
    loss_summary["total_loss"] = total_loss.mean()
    return total_loss.mean(), metrics, loss_summary


def _append_prefixed_history(
    history: dict[str, list[float]],
    prefix: str,
    values: dict[str, float],
) -> None:
    for name, value in values.items():
        history.setdefault(f"{prefix}{name}", []).append(value)


def _evaluate_supervised_batches(
    model: SupervisedRefiner,
    optimized_cases: OptimizedCases,
    train_config: SupervisedTrainingConfig,
    position_mean: torch.Tensor,
    position_std: torch.Tensor,
    adjacency_mean: torch.Tensor,
    adjacency_std: torch.Tensor,
    step: int,
    seed: int,
    use_style: bool,
) -> EvalRunResult:
    if optimized_cases.optimized_structures.batch_size == 0:
        raise ValueError("eval requires at least one case")

    total_losses: dict[str, float] = {}
    total_metrics: dict[str, float] = {}
    num_batches = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch_cases in enumerate(
                iter_supervised_batches(
                    optimized_cases,
                    batch_size=train_config.batch_size,
                    shuffle=False,
                )
            ):
                batch_cases = batch_cases.to(position_mean.device)
                batch = make_supervised_batch(
                    optimized_cases=batch_cases,
                    position_mean=position_mean,
                    position_std=position_std,
                    adjacency_mean=adjacency_mean,
                    adjacency_std=adjacency_std,
                    seed=seed + batch_index,
                )
                prediction = model.predict_flow(
                    structures=batch.flow_structures,
                    target_stiffness=batch.target_stiffness,
                    current_stiffness=batch.current_analyses.generalized_stiffness,
                    nodal_displacements=batch.current_analyses.nodal_displacements,
                    edge_von_mises=batch.current_analyses.edge_von_mises,
                    flow_times=batch.flow_times,
                    style_structures=(batch.oracle_structures if use_style else None),
                    style_analyses=(batch.oracle_analyses if use_style else None),
                )
                _, metrics, loss_terms = _training_losses(
                    prediction,
                    batch,
                    train_config,
                    step=step,
                )
                for name, value_tensor in loss_terms.items():
                    total_losses[name] = total_losses.get(name, 0.0) + float(
                        value_tensor.detach().item()
                    )
                for name, value_tensor in metrics.items():
                    total_metrics[name] = total_metrics.get(name, 0.0) + float(
                        value_tensor.detach().item()
                    )
                num_batches += 1
    finally:
        if was_training:
            model.train()

    averaged_losses = {
        name: value / num_batches for name, value in total_losses.items()
    }
    averaged_metrics = {
        name: value / num_batches for name, value in total_metrics.items()
    }
    return EvalRunResult(
        losses=averaged_losses,
        metrics=averaged_metrics,
        num_batches=num_batches,
        num_cases=optimized_cases.optimized_structures.batch_size,
    )


def train_supervised_refiner(
    optimized_cases: OptimizedCases,
    model_config: SupervisedRefinerConfig | None = None,
    train_config: SupervisedTrainingConfig | None = None,
) -> tuple[SupervisedRefiner, SupervisedTrainingSummary]:
    optimized_cases.validate()
    train_config = train_config or SupervisedTrainingConfig(dataset_path="")
    model_config = model_config or SupervisedRefinerConfig(
        use_style_token=train_config.use_style_token,
    )
    device = resolve_torch_device(train_config.device)

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")
    if train_config.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if train_config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if train_config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if train_config.min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if train_config.min_learning_rate > train_config.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate")
    if train_config.style_kl_loss_weight < 0.0:
        raise ValueError("style_kl_loss_weight must be non-negative")
    if train_config.style_kl_anneal_steps < 0:
        raise ValueError("style_kl_anneal_steps must be non-negative")
    if train_config.stiffness_loss_weight < 0.0:
        raise ValueError("stiffness_loss_weight must be non-negative")
    if train_config.stiffness_loss_delay_steps < 0:
        raise ValueError("stiffness_loss_delay_steps must be non-negative")
    if train_config.stiffness_loss_warmup_steps < 0:
        raise ValueError("stiffness_loss_warmup_steps must be non-negative")
    if train_config.stress_loss_weight < 0.0:
        raise ValueError("stress_loss_weight must be non-negative")
    if train_config.allowable_von_mises <= 0.0:
        raise ValueError("allowable_von_mises must be positive")
    if not 0.0 <= train_config.stress_activation_threshold <= 1.0:
        raise ValueError("stress_activation_threshold must be in [0.0, 1.0]")
    if train_config.stress_loss_delay_steps < 0:
        raise ValueError("stress_loss_delay_steps must be non-negative")
    if train_config.stress_loss_warmup_steps < 0:
        raise ValueError("stress_loss_warmup_steps must be non-negative")
    if not 0.0 <= train_config.style_sample_dropout <= 1.0:
        raise ValueError("style_sample_dropout must be in [0.0, 1.0]")
    if not 0.0 <= train_config.style_token_dropout <= 1.0:
        raise ValueError("style_token_dropout must be in [0.0, 1.0]")
    if not 0.0 <= train_config.eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0.0, 1.0)")

    split = split_train_eval_cases(optimized_cases, train_config.eval_fraction)
    train_cases = split.train_cases
    eval_cases = split.eval_cases
    dataset_cases = train_cases.optimized_structures.batch_size
    eval_case_count = (
        0 if eval_cases is None else eval_cases.optimized_structures.batch_size
    )
    steps_per_epoch = max(1, math.ceil(dataset_cases / train_config.batch_size))
    global_position_mean, global_position_std = _dataset_position_statistics(
        train_cases
    )
    global_adjacency_mean, global_adjacency_std = _dataset_adjacency_statistics(
        train_cases
    )
    model = SupervisedRefiner(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history = {
        "total_loss": [],
        "position_error_loss_contribution": [],
        "adjacency_error_loss_contribution": [],
        "stiffness_error_loss_contribution": [],
        "stress_loss_contribution": [],
        "style_kl_loss_contribution": [],
    }
    checkpoint_path = (
        Path(train_config.checkpoint_path)
        if train_config.checkpoint_path is not None
        else Path(train_config.logdir) / "refiner.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=train_config.logdir)
    print(
        f"training started dataset_cases={dataset_cases} batch_size={train_config.batch_size} "
        f"eval_cases={eval_case_count} eval_every_steps={train_config.eval_every_steps} "
        f"eval_fraction={train_config.eval_fraction:.4f} "
        f"steps_per_epoch={steps_per_epoch} device={device} "
        f"use_style_token={'yes' if model.config.use_style_token else 'no'} "
        f"style_local_latent_dim={model.config.style_local_latent_dim} "
        f"style_local_scale={model.config.style_local_scale:.4f} "
        f"style_sample_dropout={train_config.style_sample_dropout:.3f} "
        f"style_token_dropout={train_config.style_token_dropout:.3f} "
        f"learning_rate={train_config.learning_rate} warmup_steps={train_config.warmup_steps} "
        f"min_learning_rate={train_config.min_learning_rate} "
        f"stiffness_loss_weight={train_config.stiffness_loss_weight} "
        f"stiffness_loss_delay_steps={train_config.stiffness_loss_delay_steps} "
        f"stiffness_loss_warmup_steps={train_config.stiffness_loss_warmup_steps} "
        f"stress_loss_weight={train_config.stress_loss_weight} "
        f"allowable_von_mises={train_config.allowable_von_mises:.3e} "
        f"stress_activation_threshold={train_config.stress_activation_threshold:.3f} "
        f"stress_loss_delay_steps={train_config.stress_loss_delay_steps} "
        f"stress_loss_warmup_steps={train_config.stress_loss_warmup_steps} "
        f"style_kl_loss_weight={train_config.style_kl_loss_weight} "
        f"style_kl_anneal_steps={train_config.style_kl_anneal_steps}",
        flush=True,
    )
    try:
        step = 0
        examples_seen = 0
        while step < train_config.num_steps:
            for batch_cases in iter_supervised_batches(
                train_cases,
                batch_size=train_config.batch_size,
                shuffle=True,
                seed=train_config.seed + step,
            ):
                _synchronize_device_if_needed(device)
                step_start_time = time.perf_counter()
                batch_cases = batch_cases.to(device)
                _synchronize_device_if_needed(device)
                batch_transfer_time = time.perf_counter()
                learning_rate = _scheduled_learning_rate(step, train_config)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                batch_analysis_profile: dict[str, float] = {}
                batch = make_supervised_batch(
                    optimized_cases=batch_cases,
                    position_mean=global_position_mean.to(device),
                    position_std=global_position_std.to(device),
                    adjacency_mean=global_adjacency_mean.to(device),
                    adjacency_std=global_adjacency_std.to(device),
                    seed=train_config.seed + step,
                    profile=batch_analysis_profile,
                )
                _synchronize_device_if_needed(device)
                batch_build_time = time.perf_counter()
                style_token_mask = None
                if model.config.use_style_token:
                    style_sample_mask = (
                        torch.rand(
                            batch.flow_structures.batch_size,
                            device=device,
                        )
                        >= train_config.style_sample_dropout
                    )
                    style_token_mask = (
                        torch.rand(
                            batch.flow_structures.batch_size,
                            batch.flow_structures.num_nodes,
                            device=device,
                        )
                        >= train_config.style_token_dropout
                    )
                    style_token_mask = style_token_mask & style_sample_mask[:, None]
                prediction = model.predict_flow(
                    structures=batch.flow_structures,
                    target_stiffness=batch.target_stiffness,
                    current_stiffness=batch.current_analyses.generalized_stiffness,
                    nodal_displacements=batch.current_analyses.nodal_displacements,
                    edge_von_mises=batch.current_analyses.edge_von_mises,
                    flow_times=batch.flow_times,
                    style_structures=(
                        batch.oracle_structures
                        if model.config.use_style_token
                        else None
                    ),
                    style_analyses=(
                        batch.oracle_analyses if model.config.use_style_token else None
                    ),
                    style_token_mask=style_token_mask,
                )
                _synchronize_device_if_needed(device)
                forward_time = time.perf_counter()
                loss_analysis_profile: dict[str, float] = {}
                total_loss, metrics, loss_terms = _training_losses(
                    prediction,
                    batch,
                    train_config,
                    step=step,
                    profile=loss_analysis_profile,
                )
                _synchronize_device_if_needed(device)
                loss_time = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=train_config.max_grad_norm
                    ).item()
                )
                clipped = grad_norm > train_config.max_grad_norm
                clip_ratio = grad_norm / train_config.max_grad_norm
                optimizer.step()
                _synchronize_device_if_needed(device)
                optimizer_time = time.perf_counter()

                for name, value_tensor in loss_terms.items():
                    value = float(value_tensor.detach().item())
                    history[name].append(value)
                    writer.add_scalar(f"train/loss/{name}", value, step)
                for name, value_tensor in metrics.items():
                    writer.add_scalar(
                        f"train/metrics/{name}",
                        float(value_tensor.detach().item()),
                        step,
                    )
                writer.add_scalar(
                    "train/parameters/flow_time_mean",
                    float(batch.flow_times.mean().item()),
                    step,
                )
                writer.add_scalar(
                    "train/parameters/style_sample_available",
                    (
                        0.0
                        if style_token_mask is None
                        else float(style_token_mask.any(dim=1).float().mean().item())
                    ),
                    step,
                )
                writer.add_scalar(
                    "train/parameters/style_token_available",
                    (
                        0.0
                        if style_token_mask is None
                        else float(style_token_mask.float().mean().item())
                    ),
                    step,
                )
                writer.add_scalar(
                    "train/parameters/stiffness_loss_weight_effective",
                    float(metrics["stiffness_loss_weight_effective"].detach().item()),
                    step,
                )
                writer.add_scalar(
                    "train/parameters/stress_loss_weight_effective",
                    float(metrics["stress_loss_weight_effective"].detach().item()),
                    step,
                )
                writer.add_scalar("train/parameters/learning_rate", learning_rate, step)
                writer.add_scalar("train/gradients/grad_norm", grad_norm, step)
                writer.add_scalar("train/gradients/clip_ratio", clip_ratio, step)
                writer.add_scalar("train/gradients/clipped", float(clipped), step)
                examples_seen += batch_cases.optimized_structures.batch_size
                if (
                    step % train_config.log_every_steps == 0
                    or step == train_config.num_steps - 1
                ):
                    print(
                        f"train step={step + 1}/{train_config.num_steps} "
                        f"batch_cases={batch_cases.optimized_structures.batch_size} "
                        f"examples_seen={examples_seen} "
                        f"learning_rate={learning_rate:.8f} "
                        f"loss_total={history['total_loss'][-1]:.6f} "
                        f"loss_position={history['position_error_loss_contribution'][-1]:.6f} "
                        f"loss_adjacency={history['adjacency_error_loss_contribution'][-1]:.6f} "
                        f"loss_stiffness={history['stiffness_error_loss_contribution'][-1]:.6f} "
                        f"metric_position={float(metrics['position_error'].detach().item()):.6f} "
                        f"metric_adjacency={float(metrics['adjacency_error'].detach().item()):.6f} "
                        f"metric_stiffness={float(metrics['stiffness_error'].detach().item()):.6f} "
                        f"grad_norm={grad_norm:.6f} clipped={'yes' if clipped else 'no'} "
                        f"clip_ratio={clip_ratio:.3f} "
                        f"t_transfer={batch_transfer_time - step_start_time:.3f}s "
                        f"t_batch={batch_build_time - batch_transfer_time:.3f}s "
                        f"t_forward={forward_time - batch_build_time:.3f}s "
                        f"t_loss={loss_time - forward_time:.3f}s "
                        f"t_opt={optimizer_time - loss_time:.3f}s "
                        f"t_total={optimizer_time - step_start_time:.3f}s "
                        f"t_batch_stiffness={batch_analysis_profile.get('stiffness', 0.0):.3f}s "
                        f"t_batch_penalties={batch_analysis_profile.get('penalties', 0.0):.3f}s "
                        f"t_batch_material={batch_analysis_profile.get('material_usage', 0.0):.3f}s "
                        f"t_batch_assemble={batch_analysis_profile.get('assemble', 0.0):.3f}s "
                        f"t_batch_transform={batch_analysis_profile.get('transform', 0.0):.3f}s "
                        f"t_batch_reduce={batch_analysis_profile.get('reduce', 0.0):.3f}s "
                        f"t_batch_solve={batch_analysis_profile.get('solve', 0.0):.3f}s "
                        f"t_loss_stiffness={loss_analysis_profile.get('stiffness', 0.0):.3f}s "
                        f"t_loss_penalties={loss_analysis_profile.get('penalties', 0.0):.3f}s "
                        f"t_loss_material={loss_analysis_profile.get('material_usage', 0.0):.3f}s "
                        f"t_loss_assemble={loss_analysis_profile.get('assemble', 0.0):.3f}s "
                        f"t_loss_transform={loss_analysis_profile.get('transform', 0.0):.3f}s "
                        f"t_loss_reduce={loss_analysis_profile.get('reduce', 0.0):.3f}s "
                        f"t_loss_solve={loss_analysis_profile.get('solve', 0.0):.3f}s",
                        flush=True,
                    )
                should_run_eval = eval_cases is not None and (
                    (step + 1) % train_config.eval_every_steps == 0
                    or step == train_config.num_steps - 1
                )
                if should_run_eval:
                    eval_seed = train_config.seed + 100_000 + step
                    eval_runs = {
                        "no_style": _evaluate_supervised_batches(
                            model=model,
                            optimized_cases=eval_cases,
                            train_config=train_config,
                            position_mean=global_position_mean.to(device),
                            position_std=global_position_std.to(device),
                            adjacency_mean=global_adjacency_mean.to(device),
                            adjacency_std=global_adjacency_std.to(device),
                            step=step,
                            seed=eval_seed,
                            use_style=False,
                        )
                    }
                    if model.config.use_style_token:
                        eval_runs["with_style"] = _evaluate_supervised_batches(
                            model=model,
                            optimized_cases=eval_cases,
                            train_config=train_config,
                            position_mean=global_position_mean.to(device),
                            position_std=global_position_std.to(device),
                            adjacency_mean=global_adjacency_mean.to(device),
                            adjacency_std=global_adjacency_std.to(device),
                            step=step,
                            seed=eval_seed,
                            use_style=True,
                        )
                    for mode_name, result in eval_runs.items():
                        _append_prefixed_history(
                            history,
                            prefix=f"eval_{mode_name}_",
                            values=result.losses,
                        )
                        _append_prefixed_history(
                            history,
                            prefix=f"eval_{mode_name}_metric_",
                            values=result.metrics,
                        )
                        for name, value in result.losses.items():
                            writer.add_scalar(
                                f"eval/{mode_name}/loss/{name}",
                                value,
                                step,
                            )
                        for name, value in result.metrics.items():
                            writer.add_scalar(
                                f"eval/{mode_name}/metrics/{name}",
                                value,
                                step,
                            )
                    no_style_total = eval_runs["no_style"].losses["total_loss"]
                    eval_message = (
                        f"eval step={step + 1}/{train_config.num_steps} "
                        f"eval_cases={eval_case_count} "
                        f"loss_total_no_style={no_style_total:.6f}"
                    )
                    if "with_style" in eval_runs:
                        with_style_total = eval_runs["with_style"].losses["total_loss"]
                        eval_message += (
                            f" loss_total_with_style={with_style_total:.6f} "
                            f"delta_with_minus_no={with_style_total - no_style_total:.6f}"
                        )
                    print(eval_message, flush=True)
                step += 1
                if step >= train_config.num_steps:
                    break

        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "history": history,
            },
            checkpoint_path,
        )
    finally:
        writer.close()

    model = model.to("cpu")
    return model, SupervisedTrainingSummary(
        history=history, checkpoint_path=checkpoint_path
    )


def run_supervised_training(config: SupervisedTrainingConfig) -> None:
    optimized_cases = load_supervised_cases(config.dataset_path)
    _, summary = train_supervised_refiner(
        optimized_cases=optimized_cases,
        train_config=config,
    )
    print(f"checkpoint={summary.checkpoint_path}")
    print(f"final_total_loss={summary.history['total_loss'][-1]:.6f}")
