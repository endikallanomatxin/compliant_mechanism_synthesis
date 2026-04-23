from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.dataset.types import OptimizedCases, Structures
from compliant_mechanism_synthesis.losses import (
    StructuralObjectiveWeights,
    structural_objective_terms,
)
from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import enforce_role_adjacency_constraints
from compliant_mechanism_synthesis.training.data import (
    TrainingBatch,
    _dataset_adjacency_statistics,
    _dataset_position_statistics,
    analyze_structures,
    iter_training_batches,
    load_training_cases,
    make_training_batch,
    split_train_eval_cases,
)
from compliant_mechanism_synthesis.utils import resolve_torch_device


@dataclass(frozen=True)
class FlowCurriculumTrainingConfig:
    dataset_path: str
    device: str = "auto"
    batch_size: int = 32
    log_every_steps: int = 10
    eval_every_steps: int = 100
    max_grad_norm: float = 1.0
    num_steps: int = 120_000
    learning_rate: float = 1e-5
    warmup_steps: int = 1000
    min_learning_rate: float = 4e-6
    eval_fraction: float = 0.02
    init_checkpoint_path: str | None = None
    checkpoint_path: str | None = None
    logdir: str = "runs/train"
    seed: int = 7

    num_integration_steps: int = 4
    flow_target_epsilon: float = 1e-4

    use_style_conditioning: bool = True
    style_token_dropout: float = 0.10
    style_kl_loss_weight: float = 6e-4
    style_kl_anneal_steps: int = 1_000

    position_loss_weight: float = 1.0
    adjacency_loss_weight: float = 0.5
    supervised_weight_start: float = 1.0
    supervised_weight_end: float = 0.0
    supervised_transition_start_step: int = 5_000
    supervised_transition_end_step: int = 30_000

    physical_weight_start: float = 0.0
    physical_weight_end: float = 1.0
    physical_transition_start_step: int = 5_000
    physical_transition_end_step: int = 30_000

    stiffness_loss_weight: float = 1.0
    stress_loss_weight: float = 0.01
    material_loss_weight: float = 0.0
    short_beam_penalty_weight: float = 0.0
    long_beam_penalty_weight: float = 0.0
    thin_beam_penalty_weight: float = 0.0
    thick_beam_penalty_weight: float = 0.0
    free_node_spacing_penalty_weight: float = 0.0
    allowable_von_mises: float = 250e6
    stress_activation_threshold: float = 0.15

    log_gradient_diagnostics: bool = False


@dataclass(frozen=True)
class FlowCurriculumTrainingSummary:
    history: dict[str, list[float]]
    checkpoint_path: Path


@dataclass(frozen=True)
class EvalRunResult:
    losses: dict[str, float]
    metrics: dict[str, float]
    num_batches: int
    num_cases: int


def rollout_step_schedule(
    initial_times: torch.Tensor,
    num_integration_steps: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if initial_times.ndim != 1:
        raise ValueError("initial_times must have shape [batch]")
    if num_integration_steps <= 0:
        raise ValueError("num_integration_steps must be positive")
    step_sizes = (1.0 - initial_times).clamp_min(0.0) / num_integration_steps
    step_times = [
        initial_times + index * step_sizes for index in range(num_integration_steps)
    ]
    return step_times, step_sizes


def local_flow_targets(
    current_structures: Structures,
    oracle_structures: Structures,
    flow_times: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_structures.validate()
    oracle_structures.validate()
    if current_structures.positions.shape != oracle_structures.positions.shape:
        raise ValueError("current and oracle positions must have matching shapes")
    if current_structures.adjacency.shape != oracle_structures.adjacency.shape:
        raise ValueError("current and oracle adjacency must have matching shapes")
    if not torch.equal(current_structures.roles, oracle_structures.roles):
        raise ValueError("current and oracle structures must use the same roles")
    if flow_times.shape != (current_structures.batch_size,):
        raise ValueError("flow_times must have shape [batch]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    remaining = (1.0 - flow_times).clamp_min(epsilon)[:, None, None]
    target_position_velocity = (
        oracle_structures.positions - current_structures.positions
    ) / remaining
    target_adjacency_velocity = (
        oracle_structures.adjacency - current_structures.adjacency
    ) / remaining
    return target_position_velocity, target_adjacency_velocity


def _scheduled_learning_rate(
    step: int,
    config: FlowCurriculumTrainingConfig,
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


def _linear_weight_schedule(
    *,
    step: int,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
    weight_name: str,
) -> float:
    if start_weight < 0.0 or end_weight < 0.0:
        raise ValueError(f"{weight_name} weights must be non-negative")
    if start_step < 0 or end_step < 0:
        raise ValueError(f"{weight_name} transition steps must be non-negative")
    if end_step < start_step:
        raise ValueError(f"{weight_name} transition end must be >= start")
    if step <= start_step:
        return start_weight
    if step >= end_step:
        return end_weight
    if end_step == start_step:
        return end_weight
    progress = (step - start_step) / (end_step - start_step)
    return start_weight + progress * (end_weight - start_weight)


def _supervised_weight(step: int, config: FlowCurriculumTrainingConfig) -> float:
    return _linear_weight_schedule(
        step=step,
        start_weight=config.supervised_weight_start,
        end_weight=config.supervised_weight_end,
        start_step=config.supervised_transition_start_step,
        end_step=config.supervised_transition_end_step,
        weight_name="supervised",
    )


def _physical_weight(step: int, config: FlowCurriculumTrainingConfig) -> float:
    return _linear_weight_schedule(
        step=step,
        start_weight=config.physical_weight_start,
        end_weight=config.physical_weight_end,
        start_step=config.physical_transition_start_step,
        end_step=config.physical_transition_end_step,
        weight_name="physical",
    )


def _style_kl_weight(step: int, config: FlowCurriculumTrainingConfig) -> float:
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


def _position_velocity_loss(
    prediction_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    roles: torch.Tensor,
) -> torch.Tensor:
    _, _, free_mask = role_masks(roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=prediction_velocity.dtype)
    error = ((prediction_velocity - target_velocity).square() * free_mask).sum(
        dim=(1, 2)
    )
    denom = free_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return error / denom


def _adjacency_velocity_loss(
    prediction_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    return (prediction_velocity - target_velocity).square().mean(dim=(1, 2))


def _physical_time_weight(flow_times: torch.Tensor) -> torch.Tensor:
    return flow_times.clamp(0.0, 1.0)


def _euler_step(
    current_structures: Structures,
    position_velocity: torch.Tensor,
    adjacency_velocity: torch.Tensor,
    step_sizes: torch.Tensor,
) -> Structures:
    _, _, free_mask = role_masks(current_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=current_structures.positions.dtype)
    step_scale = step_sizes[:, None, None]
    next_positions = torch.nan_to_num(
        current_structures.positions + step_scale * position_velocity * free_mask,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    next_adjacency = enforce_role_adjacency_constraints(
        torch.nan_to_num(
            current_structures.adjacency + step_scale * adjacency_velocity,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0),
        current_structures.roles,
    )
    return Structures(
        positions=next_positions,
        roles=current_structures.roles,
        adjacency=next_adjacency,
    )


def _aggregate_rollout_losses(
    *,
    supervised_step_losses: list[torch.Tensor],
    physical_step_losses: list[torch.Tensor],
    weighted_physical_step_losses: list[torch.Tensor],
    weighted_stiffness_step_losses: list[torch.Tensor],
    weighted_stress_step_losses: list[torch.Tensor],
    style_kl: torch.Tensor,
    lambda_sup: float,
    lambda_phys: float,
    lambda_kl: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    supervised_total = torch.stack(supervised_step_losses, dim=0).sum(dim=0)
    physical_total = torch.stack(physical_step_losses, dim=0).sum(dim=0)
    weighted_physical_total = torch.stack(weighted_physical_step_losses, dim=0).sum(
        dim=0
    )
    weighted_stiffness_total = torch.stack(weighted_stiffness_step_losses, dim=0).sum(
        dim=0
    )
    weighted_stress_total = torch.stack(weighted_stress_step_losses, dim=0).sum(dim=0)
    total_loss = (
        lambda_sup * supervised_total
        + lambda_phys * weighted_physical_total
        + lambda_kl * style_kl
    )
    loss_terms = {
        "supervised_loss_contribution": lambda_sup * supervised_total.mean(),
        "physical_loss_contribution": lambda_phys * weighted_physical_total.mean(),
        "stiffness_loss_contribution": lambda_phys * weighted_stiffness_total.mean(),
        "stress_loss_contribution": lambda_phys * weighted_stress_total.mean(),
        "style_kl_loss_contribution": lambda_kl * style_kl.mean(),
        "supervised_loss": supervised_total.mean(),
        "physical_loss": physical_total.mean(),
        "time_weighted_physical_loss": weighted_physical_total.mean(),
        "total_loss": total_loss.mean(),
    }
    return total_loss.mean(), loss_terms


def _grad_norm_for_loss(loss: torch.Tensor, model: SupervisedRefiner) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.square().sum()
    return torch.sqrt(squared_norm)


def _trajectory_loss_terms(
    model: SupervisedRefiner,
    batch: TrainingBatch,
    config: FlowCurriculumTrainingConfig,
    step: int,
    style_token_mask: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    lambda_sup = _supervised_weight(step, config)
    lambda_phys = _physical_weight(step, config)
    lambda_kl = _style_kl_weight(step, config)
    step_times, step_sizes = rollout_step_schedule(
        batch.initial_times,
        config.num_integration_steps,
    )
    current = batch.initial_structures
    style_kl = batch.initial_times.new_zeros(batch.initial_times.shape)
    supervised_step_losses: list[torch.Tensor] = []
    physical_step_losses: list[torch.Tensor] = []
    weighted_physical_step_losses: list[torch.Tensor] = []
    weighted_stiffness_step_losses: list[torch.Tensor] = []
    weighted_stress_step_losses: list[torch.Tensor] = []
    position_step_losses: list[torch.Tensor] = []
    adjacency_step_losses: list[torch.Tensor] = []
    stiffness_errors: list[torch.Tensor] = []
    stress_violations: list[torch.Tensor] = []
    mean_stress_ratios: list[torch.Tensor] = []
    max_stress_ratios: list[torch.Tensor] = []

    for integration_index, flow_times in enumerate(step_times):
        analyses = analyze_structures(current, profile=profile)
        if analyses.nodal_displacements is None:
            raise ValueError("analyze_structures must provide nodal_displacements")
        if analyses.edge_von_mises is None:
            raise ValueError("analyze_structures must provide edge_von_mises")
        prediction = model.predict_flow(
            structures=current,
            target_stiffness=batch.target_stiffness,
            current_stiffness=analyses.generalized_stiffness,
            nodal_displacements=analyses.nodal_displacements,
            edge_von_mises=analyses.edge_von_mises,
            flow_times=flow_times,
            style_structures=(
                batch.oracle_structures if model.config.use_style_conditioning else None
            ),
            style_analyses=(
                batch.oracle_analyses if model.config.use_style_conditioning else None
            ),
            style_token_mask=style_token_mask,
        )
        if integration_index == 0 and prediction.style_kl is not None:
            style_kl = prediction.style_kl

        target_position_velocity, target_adjacency_velocity = local_flow_targets(
            current,
            batch.oracle_structures,
            flow_times,
            epsilon=config.flow_target_epsilon,
        )
        position_loss = _position_velocity_loss(
            prediction.position_velocity,
            target_position_velocity,
            current.roles,
        )
        adjacency_loss = _adjacency_velocity_loss(
            prediction.adjacency_velocity,
            target_adjacency_velocity,
        )
        supervised_loss = (
            config.position_loss_weight * position_loss
            + config.adjacency_loss_weight * adjacency_loss
        )
        structural_terms, structural_metrics = structural_objective_terms(
            analyses=analyses,
            adjacency=current.adjacency,
            target_stiffness=batch.target_stiffness,
            weights=StructuralObjectiveWeights(
                stiffness=config.stiffness_loss_weight,
                stress=config.stress_loss_weight,
                material=config.material_loss_weight,
                short_beam=config.short_beam_penalty_weight,
                long_beam=config.long_beam_penalty_weight,
                thin_beam=config.thin_beam_penalty_weight,
                thick_beam=config.thick_beam_penalty_weight,
                free_node_spacing=config.free_node_spacing_penalty_weight,
            ),
            allowable_von_mises=config.allowable_von_mises,
            stress_activation_threshold=config.stress_activation_threshold,
        )
        stiffness_loss = structural_terms["stiffness_loss_contribution"]
        stress_loss = structural_terms["stress_loss_contribution"]
        physical_loss = sum(structural_terms.values())
        time_weight = _physical_time_weight(flow_times)
        weighted_physical_loss = time_weight * physical_loss
        weighted_stiffness_loss = time_weight * stiffness_loss
        weighted_stress_loss = time_weight * stress_loss

        supervised_step_losses.append(supervised_loss)
        physical_step_losses.append(physical_loss)
        weighted_physical_step_losses.append(weighted_physical_loss)
        weighted_stiffness_step_losses.append(weighted_stiffness_loss)
        weighted_stress_step_losses.append(weighted_stress_loss)
        position_step_losses.append(position_loss)
        adjacency_step_losses.append(adjacency_loss)
        stiffness_errors.append(structural_metrics["stiffness_error"])
        stress_violations.append(structural_metrics["stress_violation"])
        mean_stress_ratios.append(structural_metrics["mean_stress_ratio"])
        max_stress_ratios.append(structural_metrics["max_stress_ratio"])

        current = _euler_step(
            current_structures=current,
            position_velocity=prediction.position_velocity,
            adjacency_velocity=prediction.adjacency_velocity,
            step_sizes=step_sizes,
        )

    total_loss, loss_terms = _aggregate_rollout_losses(
        supervised_step_losses=supervised_step_losses,
        physical_step_losses=physical_step_losses,
        weighted_physical_step_losses=weighted_physical_step_losses,
        weighted_stiffness_step_losses=weighted_stiffness_step_losses,
        weighted_stress_step_losses=weighted_stress_step_losses,
        style_kl=(
            style_kl
            if model.config.use_style_conditioning
            else batch.initial_times.new_zeros(batch.initial_times.shape)
        ),
        lambda_sup=lambda_sup,
        lambda_phys=lambda_phys,
        lambda_kl=lambda_kl,
    )
    metrics = {
        "position_velocity_loss": torch.stack(position_step_losses, dim=0)
        .sum(dim=0)
        .mean(),
        "adjacency_velocity_loss": torch.stack(adjacency_step_losses, dim=0)
        .sum(dim=0)
        .mean(),
        "stiffness_error": torch.stack(stiffness_errors).mean(),
        "stress_violation": torch.stack(stress_violations).mean(),
        "mean_stress_ratio": torch.stack(mean_stress_ratios).mean(),
        "max_stress_ratio": torch.stack(max_stress_ratios).mean(),
        "style_kl": style_kl.mean(),
        "lambda_sup": batch.initial_times.new_tensor(lambda_sup),
        "lambda_phys": batch.initial_times.new_tensor(lambda_phys),
        "lambda_kl": batch.initial_times.new_tensor(lambda_kl),
        "integration_end_time": (
            batch.initial_times + config.num_integration_steps * step_sizes
        ).mean(),
        "physical_time_weight_mean": torch.stack(
            [_physical_time_weight(time) for time in step_times]
        ).mean(),
    }
    return total_loss, metrics, loss_terms


def _append_prefixed_history(
    history: dict[str, list[float]],
    prefix: str,
    values: dict[str, float],
) -> None:
    for name, value in values.items():
        history.setdefault(f"{prefix}{name}", []).append(value)


def _evaluate_batches(
    model: SupervisedRefiner,
    optimized_cases: OptimizedCases,
    train_config: FlowCurriculumTrainingConfig,
    position_mean: torch.Tensor,
    position_std: torch.Tensor,
    adjacency_mean: torch.Tensor,
    adjacency_std: torch.Tensor,
    step: int,
    seed: int,
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
                iter_training_batches(
                    optimized_cases,
                    batch_size=train_config.batch_size,
                    shuffle=False,
                )
            ):
                batch_cases = batch_cases.to(position_mean.device)
                batch = make_training_batch(
                    optimized_cases=batch_cases,
                    position_mean=position_mean,
                    position_std=position_std,
                    adjacency_mean=adjacency_mean,
                    adjacency_std=adjacency_std,
                    seed=seed + batch_index,
                )
                _, metrics, loss_terms = _trajectory_loss_terms(
                    model=model,
                    batch=batch,
                    config=train_config,
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


def _load_initial_model(
    device: torch.device,
    train_config: FlowCurriculumTrainingConfig,
    model_config: SupervisedRefinerConfig | None,
) -> tuple[SupervisedRefiner, SupervisedRefinerConfig]:
    effective_config = model_config or SupervisedRefinerConfig(
        use_style_conditioning=train_config.use_style_conditioning,
        num_integration_steps=train_config.num_integration_steps,
    )
    model = SupervisedRefiner(effective_config).to(device)
    if train_config.init_checkpoint_path is None:
        return model, effective_config

    checkpoint = torch.load(train_config.init_checkpoint_path, map_location=device)
    if model_config is None:
        effective_config = SupervisedRefinerConfig(**checkpoint["model_config"])
        model = SupervisedRefiner(effective_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, effective_config


def train_flow_refiner(
    optimized_cases: OptimizedCases,
    model_config: SupervisedRefinerConfig | None = None,
    train_config: FlowCurriculumTrainingConfig | None = None,
) -> tuple[SupervisedRefiner, FlowCurriculumTrainingSummary]:
    optimized_cases.validate()
    train_config = train_config or FlowCurriculumTrainingConfig(dataset_path="")
    device = resolve_torch_device(train_config.device)

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")
    if train_config.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if train_config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if train_config.num_integration_steps <= 0:
        raise ValueError("num_integration_steps must be positive")
    if train_config.flow_target_epsilon <= 0.0:
        raise ValueError("flow_target_epsilon must be positive")
    if not 0.0 <= train_config.style_token_dropout <= 1.0:
        raise ValueError("style_token_dropout must be in [0.0, 1.0]")
    if not 0.0 <= train_config.eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0.0, 1.0)")
    if train_config.allowable_von_mises <= 0.0:
        raise ValueError("allowable_von_mises must be positive")
    if not 0.0 <= train_config.stress_activation_threshold <= 1.0:
        raise ValueError("stress_activation_threshold must be in [0.0, 1.0]")

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
    global_position_mean = global_position_mean.to(device)
    global_position_std = global_position_std.to(device)
    global_adjacency_mean = global_adjacency_mean.to(device)
    global_adjacency_std = global_adjacency_std.to(device)

    model, effective_model_config = _load_initial_model(
        device, train_config, model_config
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: dict[str, list[float]] = {
        "total_loss": [],
        "supervised_loss_contribution": [],
        "physical_loss_contribution": [],
        "stiffness_loss_contribution": [],
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
        f"eval_fraction={train_config.eval_fraction:.4f} steps_per_epoch={steps_per_epoch} "
        f"device={device} init_checkpoint={train_config.init_checkpoint_path or 'none'} "
        f"num_integration_steps={train_config.num_integration_steps} "
        f"use_style_conditioning={'yes' if effective_model_config.use_style_conditioning else 'no'} "
        f"style_token_dropout={train_config.style_token_dropout:.3f} "
        f"learning_rate={train_config.learning_rate} warmup_steps={train_config.warmup_steps} "
        f"min_learning_rate={train_config.min_learning_rate} "
        f"supervised_weight_start={train_config.supervised_weight_start:.4f} "
        f"supervised_weight_end={train_config.supervised_weight_end:.4f} "
        f"supervised_transition_start_step={train_config.supervised_transition_start_step} "
        f"supervised_transition_end_step={train_config.supervised_transition_end_step} "
        f"physical_weight_start={train_config.physical_weight_start:.4f} "
        f"physical_weight_end={train_config.physical_weight_end:.4f} "
        f"physical_transition_start_step={train_config.physical_transition_start_step} "
        f"physical_transition_end_step={train_config.physical_transition_end_step} "
        f"stiffness_loss_weight={train_config.stiffness_loss_weight} "
        f"stress_loss_weight={train_config.stress_loss_weight} "
        f"allowable_von_mises={train_config.allowable_von_mises:.3e} "
        f"stress_activation_threshold={train_config.stress_activation_threshold:.3f} "
        f"style_kl_loss_weight={train_config.style_kl_loss_weight} "
        f"style_kl_anneal_steps={train_config.style_kl_anneal_steps}",
        flush=True,
    )

    try:
        step = 0
        examples_seen = 0
        while step < train_config.num_steps:
            for batch_cases in iter_training_batches(
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

                batch = make_training_batch(
                    optimized_cases=batch_cases,
                    position_mean=global_position_mean,
                    position_std=global_position_std,
                    adjacency_mean=global_adjacency_mean,
                    adjacency_std=global_adjacency_std,
                    seed=train_config.seed + step,
                )
                _synchronize_device_if_needed(device)
                batch_build_time = time.perf_counter()

                style_token_mask = None
                if effective_model_config.use_style_conditioning:
                    style_token_mask = (
                        torch.rand(
                            batch.initial_structures.batch_size,
                            batch.initial_structures.num_nodes,
                            device=device,
                        )
                        >= train_config.style_token_dropout
                    )

                loss_profile: dict[str, float] = {}
                total_loss, metrics, loss_terms = _trajectory_loss_terms(
                    model=model,
                    batch=batch,
                    config=train_config,
                    step=step,
                    style_token_mask=style_token_mask,
                    profile=loss_profile,
                )
                _synchronize_device_if_needed(device)
                loss_time = time.perf_counter()

                gradient_metrics: dict[str, float] = {}
                if train_config.log_gradient_diagnostics:
                    supervised_grad_norm = _grad_norm_for_loss(
                        loss_terms["supervised_loss_contribution"],
                        model,
                    )
                    physical_grad_norm = _grad_norm_for_loss(
                        loss_terms["physical_loss_contribution"],
                        model,
                    )
                    gradient_metrics = {
                        "supervised_grad_norm": float(
                            supervised_grad_norm.detach().item()
                        ),
                        "physical_grad_norm": float(physical_grad_norm.detach().item()),
                    }

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=train_config.max_grad_norm,
                    ).item()
                )
                clipped = grad_norm > train_config.max_grad_norm
                clip_ratio = grad_norm / train_config.max_grad_norm
                optimizer.step()
                _synchronize_device_if_needed(device)
                optimizer_time = time.perf_counter()

                for name, value_tensor in loss_terms.items():
                    value = float(value_tensor.detach().item())
                    history.setdefault(name, []).append(value)
                    writer.add_scalar(f"train/loss/{name}", value, step)
                for name, value_tensor in metrics.items():
                    writer.add_scalar(
                        f"train/metrics/{name}",
                        float(value_tensor.detach().item()),
                        step,
                    )
                writer.add_scalar("train/parameters/learning_rate", learning_rate, step)
                writer.add_scalar(
                    "train/parameters/style_token_available",
                    (
                        0.0
                        if style_token_mask is None
                        else float(style_token_mask.float().mean().item())
                    ),
                    step,
                )
                writer.add_scalar("train/gradients/grad_norm", grad_norm, step)
                writer.add_scalar("train/gradients/clip_ratio", clip_ratio, step)
                writer.add_scalar("train/gradients/clipped", float(clipped), step)
                for name, value in gradient_metrics.items():
                    writer.add_scalar(f"train/gradients/{name}", value, step)

                examples_seen += batch_cases.optimized_structures.batch_size
                if (
                    step % train_config.log_every_steps == 0
                    or step == train_config.num_steps - 1
                ):
                    diagnostics_suffix = "".join(
                        f" {name}={value:.6f}"
                        for name, value in gradient_metrics.items()
                    )
                    print(
                        f"train step={step + 1}/{train_config.num_steps} "
                        f"batch_cases={batch_cases.optimized_structures.batch_size} "
                        f"examples_seen={examples_seen} learning_rate={learning_rate:.8f} "
                        f"loss_total={history['total_loss'][-1]:.6f} "
                        f"loss_sup={history['supervised_loss_contribution'][-1]:.6f} "
                        f"loss_phys={history['physical_loss_contribution'][-1]:.6f} "
                        f"stiffness={float(metrics['stiffness_error'].detach().item()):.6f} "
                        f"stress={float(metrics['stress_violation'].detach().item()):.6f} "
                        f"lambda_sup={float(metrics['lambda_sup'].detach().item()):.6f} "
                        f"lambda_phys={float(metrics['lambda_phys'].detach().item()):.6f} "
                        f"grad_norm={grad_norm:.6f} clipped={'yes' if clipped else 'no'} "
                        f"clip_ratio={clip_ratio:.3f} "
                        f"t_transfer={batch_transfer_time - step_start_time:.3f}s "
                        f"t_batch={batch_build_time - batch_transfer_time:.3f}s "
                        f"t_loss={loss_time - batch_build_time:.3f}s "
                        f"t_opt={optimizer_time - loss_time:.3f}s "
                        f"t_total={optimizer_time - step_start_time:.3f}s"
                        f"{diagnostics_suffix}",
                        flush=True,
                    )

                should_run_eval = eval_cases is not None and (
                    (step + 1) % train_config.eval_every_steps == 0
                    or step == train_config.num_steps - 1
                )
                if should_run_eval:
                    eval_seed = train_config.seed + 100_000 + step
                    result = _evaluate_batches(
                        model=model,
                        optimized_cases=eval_cases,
                        train_config=train_config,
                        position_mean=global_position_mean,
                        position_std=global_position_std,
                        adjacency_mean=global_adjacency_mean,
                        adjacency_std=global_adjacency_std,
                        step=step,
                        seed=eval_seed,
                    )
                    _append_prefixed_history(
                        history, prefix="eval_", values=result.losses
                    )
                    _append_prefixed_history(
                        history, prefix="eval_metric_", values=result.metrics
                    )
                    for name, value in result.losses.items():
                        writer.add_scalar(f"eval/loss/{name}", value, step)
                    for name, value in result.metrics.items():
                        writer.add_scalar(f"eval/metrics/{name}", value, step)
                    print(
                        f"eval step={step + 1}/{train_config.num_steps} eval_cases={result.num_cases} "
                        f"loss_total={result.losses['total_loss']:.6f} "
                        f"loss_sup={result.losses['supervised_loss_contribution']:.6f} "
                        f"loss_phys={result.losses['physical_loss_contribution']:.6f}",
                        flush=True,
                    )

                step += 1
                if step >= train_config.num_steps:
                    break
    finally:
        checkpoint = {
            "model_config": asdict(effective_model_config),
            "train_config": asdict(train_config),
            "model_state_dict": model.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, checkpoint_path)
        writer.close()

    print(f"checkpoint={checkpoint_path}")
    print(f"final_total_loss={history['total_loss'][-1]:.6f}")
    return model, FlowCurriculumTrainingSummary(
        history=history,
        checkpoint_path=checkpoint_path,
    )


def run_flow_training(config: FlowCurriculumTrainingConfig) -> None:
    optimized_cases = load_training_cases(config.dataset_path)
    _, summary = train_flow_refiner(
        optimized_cases=optimized_cases,
        train_config=config,
    )
    print(f"logs={Path(config.logdir)}")
    print(f"checkpoint={summary.checkpoint_path}")
