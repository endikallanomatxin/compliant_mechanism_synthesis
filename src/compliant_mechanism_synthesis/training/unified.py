from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.adjacency import logits_from_adjacency
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Structures,
)
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
    main_grad_clip_norm: float = 1.0
    physical_grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    num_steps: int = 120_000
    learning_rate: float = 1e-5
    warmup_steps: int = 1000
    min_learning_rate: float = 1e-6
    eval_fraction: float = 0.02
    init_checkpoint_path: str | None = None
    checkpoint_path: str | None = None
    logdir: str = "runs/train"
    seed: int = 7

    num_integration_steps: int = 3
    flow_target_epsilon: float = 0.02
    max_position_step: float = 0.05
    max_adjacency_logit_step: float = 1.0

    use_style_conditioning: bool = True
    style_token_dropout: float = 0.50
    style_kl_loss_weight: float = 1e-4
    style_kl_anneal_steps: int = 20_000

    position_loss_weight: float = 1.0
    adjacency_loss_weight: float = 0.001  # Loss occurs in logit space
    position_huber_beta: float = 0.02
    adjacency_huber_beta: float = 0.05
    supervised_weight_start: float = 1.0
    supervised_weight_end: float = 0.01
    supervised_transition_start_step: int = 20_000
    supervised_transition_end_step: int = 60_000

    physical_weight_start: float = 0
    physical_weight_end: float = 1
    physical_transition_start_step: int = 0
    physical_transition_end_step: int = 80_000

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
    simulation_position_grad_clip_norm: float = 1.0
    simulation_adjacency_grad_clip_norm: float = 0.25
    absolute_physical_loss_weight: float = 0.2
    relative_physical_loss_weight: float = 0.8

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
    current_adjacency_logits = logits_from_adjacency(current_structures.adjacency)
    oracle_adjacency_logits = logits_from_adjacency(oracle_structures.adjacency)
    target_adjacency_logit_velocity = (
        oracle_adjacency_logits - current_adjacency_logits
    ) / remaining
    return target_position_velocity, target_adjacency_logit_velocity


def flow_step_predictions(
    position_step: torch.Tensor,
    adjacency_logit_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_step.ndim != 3:
        raise ValueError("position_step must have shape [batch, nodes, dims]")
    if adjacency_logit_step.ndim != 3:
        raise ValueError("adjacency_logit_step must have shape [batch, nodes, nodes]")
    if position_step.shape[:2] != adjacency_logit_step.shape[:2]:
        raise ValueError(
            "position_step and adjacency_logit_step batch/node dims must match"
        )
    return position_step, adjacency_logit_step


def local_flow_step_targets(
    current_structures: Structures,
    oracle_structures: Structures,
    flow_times: torch.Tensor,
    step_sizes: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_position_velocity, target_adjacency_logit_velocity = local_flow_targets(
        current_structures=current_structures,
        oracle_structures=oracle_structures,
        flow_times=flow_times,
        epsilon=epsilon,
    )
    step_scale = step_sizes[:, None, None]
    return (
        step_scale * target_position_velocity,
        step_scale * target_adjacency_logit_velocity,
    )


def _max_initial_time(config: FlowCurriculumTrainingConfig) -> float:
    max_initial_time = 1.0 - config.num_integration_steps * config.flow_target_epsilon
    if max_initial_time <= 0.0:
        raise ValueError("num_integration_steps * flow_target_epsilon must be < 1.0")
    return max_initial_time


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


def _position_step_loss(
    predicted_step: torch.Tensor,
    target_step: torch.Tensor,
    roles: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta <= 0.0:
        raise ValueError("position_huber_beta must be positive")
    _, _, free_mask = role_masks(roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=predicted_step.dtype)
    error = (
        F.smooth_l1_loss(
            predicted_step,
            target_step,
            reduction="none",
            beta=beta,
        )
        * free_mask
    ).sum(dim=(1, 2))
    denom = free_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return error / denom


def _adjacency_step_loss(
    predicted_step: torch.Tensor,
    target_step: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta <= 0.0:
        raise ValueError("adjacency_huber_beta must be positive")
    return F.smooth_l1_loss(
        predicted_step,
        target_step,
        reduction="none",
        beta=beta,
    ).mean(dim=(1, 2))


def _euler_step(
    current_structures: Structures,
    position_step: torch.Tensor,
    adjacency_logit_step: torch.Tensor,
) -> Structures:
    _, _, free_mask = role_masks(current_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=current_structures.positions.dtype)
    next_positions = torch.nan_to_num(
        current_structures.positions + position_step * free_mask,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    current_adjacency_logits = logits_from_adjacency(current_structures.adjacency)
    next_adjacency_logits = torch.nan_to_num(
        current_adjacency_logits + adjacency_logit_step,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    next_adjacency = enforce_role_adjacency_constraints(
        torch.sigmoid(next_adjacency_logits),
        current_structures.roles,
    )
    return Structures(
        positions=next_positions,
        roles=current_structures.roles,
        adjacency=next_adjacency,
    )


def _safe_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        velocity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _range_violation(values: torch.Tensor) -> torch.Tensor:
    below = torch.relu(-values)
    above = torch.relu(values - 1.0)
    return (below + above).mean(dim=tuple(range(1, values.ndim)))


def _clip_grad_norm_per_sample(max_norm: float):
    if max_norm <= 0.0:
        raise ValueError("per-sample gradient clip norm must be positive")

    def hook(grad: torch.Tensor) -> torch.Tensor:
        flat = grad.flatten(start_dim=1)
        norm = flat.norm(dim=1).clamp_min(1e-12)
        scale = (max_norm / norm).clamp_max(1.0)
        return grad * scale.view(-1, *([1] * (grad.ndim - 1)))

    return hook


def _register_simulation_gradient_hooks(
    structures: Structures,
    config: FlowCurriculumTrainingConfig,
) -> None:
    if structures.positions.requires_grad:
        structures.positions.register_hook(
            _clip_grad_norm_per_sample(config.simulation_position_grad_clip_norm)
        )
    if structures.adjacency.requires_grad:
        structures.adjacency.register_hook(
            _clip_grad_norm_per_sample(config.simulation_adjacency_grad_clip_norm)
        )


def _structural_terms_and_metrics(
    analyses: Analyses,
    adjacency: torch.Tensor,
    target_stiffness: torch.Tensor,
    config: FlowCurriculumTrainingConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return structural_objective_terms(
        analyses=analyses,
        adjacency=adjacency,
        target_stiffness=target_stiffness,
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


def _optimizer_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    parameter_lookup = dict(model.named_parameters())
    no_decay_names: set[str] = set()
    for module_name, module in model.named_modules():
        for parameter_name, _parameter in module.named_parameters(recurse=False):
            full_name = (
                parameter_name if not module_name else f"{module_name}.{parameter_name}"
            )
            if parameter_name.endswith("bias"):
                no_decay_names.add(full_name)
            if isinstance(module, (nn.LayerNorm, nn.Embedding)):
                no_decay_names.add(full_name)
    for name in parameter_lookup:
        if name.endswith("_raw_scale"):
            no_decay_names.add(name)

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, parameter in parameter_lookup.items():
        if not parameter.requires_grad:
            continue
        if name in no_decay_names:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _all_finite_tensors(values: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def _physical_loss_enabled(
    config: FlowCurriculumTrainingConfig,
    lambda_phys: float,
) -> bool:
    if lambda_phys <= 0.0:
        return False
    if (
        config.absolute_physical_loss_weight <= 0.0
        and config.relative_physical_loss_weight <= 0.0
    ):
        return False
    return any(
        weight > 0.0
        for weight in (
            config.stiffness_loss_weight,
            config.stress_loss_weight,
            config.material_loss_weight,
            config.short_beam_penalty_weight,
            config.long_beam_penalty_weight,
            config.thin_beam_penalty_weight,
            config.thick_beam_penalty_weight,
            config.free_node_spacing_penalty_weight,
        )
    )


def _zero_conditioning_inputs(
    current: Structures,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_nodes, _ = current.positions.shape
    dtype = current.positions.dtype
    device = current.positions.device
    return (
        torch.zeros(batch_size, 6, 6, device=device, dtype=dtype),
        torch.zeros(batch_size, num_nodes, 18, device=device, dtype=dtype),
        torch.zeros(batch_size, num_nodes, num_nodes, 6, device=device, dtype=dtype),
    )


def _conditioning_inputs(
    current: Structures,
    *,
    use_analysis: bool,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_analysis:
        return _zero_conditioning_inputs(current)

    with torch.no_grad():
        analyses = analyze_structures(current, profile=profile)
    if analyses.nodal_displacements is None:
        raise ValueError("analyze_structures must provide nodal_displacements")
    if analyses.edge_von_mises is None:
        raise ValueError("analyze_structures must provide edge_von_mises")
    return (
        analyses.generalized_stiffness,
        analyses.nodal_displacements,
        analyses.edge_von_mises,
    )


def _endpoint_analyses(
    current: Structures,
    *,
    physical_loss_enabled: bool,
    profile: dict[str, float] | None = None,
) -> Analyses | None:
    if not physical_loss_enabled:
        return None
    analyses = analyze_structures(current, profile=profile)
    if analyses.nodal_displacements is None:
        raise ValueError("analyze_structures must provide nodal_displacements")
    if analyses.edge_von_mises is None:
        raise ValueError("analyze_structures must provide edge_von_mises")
    return analyses


def _aggregate_rollout_losses(
    *,
    supervised_step_losses: list[torch.Tensor],
    endpoint_physical_loss: torch.Tensor,
    endpoint_stiffness_loss: torch.Tensor,
    endpoint_stress_loss: torch.Tensor,
    style_kl: torch.Tensor,
    lambda_sup: float,
    lambda_phys: float,
    lambda_kl: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    supervised_total = torch.stack(supervised_step_losses, dim=0).mean(dim=0)
    total_loss = (
        lambda_sup * supervised_total
        + lambda_phys * endpoint_physical_loss
        + lambda_kl * style_kl
    )
    loss_terms = {
        "supervised_loss_contribution": lambda_sup * supervised_total.mean(),
        "physical_loss_contribution": lambda_phys * endpoint_physical_loss.mean(),
        "stiffness_loss_contribution": lambda_phys * endpoint_stiffness_loss.mean(),
        "stress_loss_contribution": lambda_phys * endpoint_stress_loss.mean(),
        "style_kl_loss_contribution": lambda_kl * style_kl.mean(),
        "supervised_loss": supervised_total.mean(),
        "physical_loss": endpoint_physical_loss.mean(),
        "total_loss": total_loss.mean(),
    }
    return total_loss.mean(), loss_terms


def _zero_endpoint_physical_terms(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    zeros = torch.zeros_like(reference)
    metrics = {
        "stiffness_error": zeros,
        "stress_violation": zeros,
        "mean_stress_ratio": zeros,
        "max_stress_ratio": zeros,
    }
    return zeros, zeros, zeros, metrics


def _loss_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    if not loss.requires_grad:
        return [None] * len(parameters)
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    )


def _gradient_list_norm(
    gradients: list[torch.Tensor | None],
    reference: torch.Tensor,
) -> torch.Tensor:
    squared_norm = reference.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.square().sum()
    return torch.sqrt(squared_norm)


def _gradient_lists_cosine(
    lhs_gradients: list[torch.Tensor | None],
    rhs_gradients: list[torch.Tensor | None],
    reference: torch.Tensor,
) -> torch.Tensor:
    lhs_norm = _gradient_list_norm(lhs_gradients, reference)
    rhs_norm = _gradient_list_norm(rhs_gradients, reference)
    denom = lhs_norm * rhs_norm
    if float(denom.detach().item()) == 0.0:
        return reference.new_zeros(())
    dot_product = reference.new_zeros(())
    for lhs_gradient, rhs_gradient in zip(lhs_gradients, rhs_gradients, strict=True):
        if lhs_gradient is not None and rhs_gradient is not None:
            dot_product = dot_product + (lhs_gradient * rhs_gradient).sum()
    return torch.nan_to_num(
        dot_product / denom,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _clip_gradient_list(
    gradients: list[torch.Tensor | None],
    max_norm: float,
    reference: torch.Tensor,
    prefix: str,
) -> tuple[list[torch.Tensor | None], dict[str, torch.Tensor]]:
    if max_norm <= 0.0:
        raise ValueError(f"{prefix} max_norm must be positive")
    pre_clip_norm = _gradient_list_norm(gradients, reference)
    pre_clip_value = float(pre_clip_norm.detach().item())
    metrics = {
        f"{prefix}_grad_norm_pre_clip": pre_clip_norm,
        f"{prefix}_grad_norm_post_clip": pre_clip_norm,
        f"{prefix}_grad_clip_ratio": reference.new_tensor(pre_clip_value / max_norm),
        f"{prefix}_grad_clipped": reference.new_tensor(
            float(pre_clip_value > max_norm)
        ),
    }
    if (
        pre_clip_value == 0.0
        or not math.isfinite(pre_clip_value)
        or pre_clip_value <= max_norm
    ):
        return gradients, metrics

    scale = max_norm / pre_clip_value
    clipped_gradients = [
        None if gradient is None else gradient * scale for gradient in gradients
    ]
    metrics[f"{prefix}_grad_norm_post_clip"] = _gradient_list_norm(
        clipped_gradients,
        reference,
    )
    return clipped_gradients, metrics


def _combine_gradient_lists(
    gradient_groups: list[list[torch.Tensor | None]],
) -> list[torch.Tensor | None]:
    combined_gradients: list[torch.Tensor | None] = []
    for grouped_gradients in zip(*gradient_groups, strict=True):
        combined_gradient = None
        for gradient in grouped_gradients:
            if gradient is not None:
                combined_gradient = (
                    gradient
                    if combined_gradient is None
                    else combined_gradient + gradient
                )
        combined_gradients.append(combined_gradient)
    return combined_gradients


def _assign_gradients(
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor | None],
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient


def _objective_clipped_gradients(
    *,
    main_loss: torch.Tensor,
    physical_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    config: FlowCurriculumTrainingConfig,
) -> tuple[list[torch.Tensor | None], dict[str, torch.Tensor]]:
    reference = main_loss
    main_gradients = _loss_gradients(main_loss, parameters)
    physical_gradients = _loss_gradients(physical_loss, parameters)

    clipped_main_gradients, main_metrics = _clip_gradient_list(
        main_gradients,
        config.main_grad_clip_norm,
        reference,
        prefix="main",
    )
    clipped_physical_gradients, physical_metrics = _clip_gradient_list(
        physical_gradients,
        config.physical_grad_clip_norm,
        reference,
        prefix="physical",
    )
    combined_gradients = _combine_gradient_lists(
        [
            clipped_main_gradients,
            clipped_physical_gradients,
        ]
    )
    metrics = {
        **main_metrics,
        **physical_metrics,
        "combined_grad_norm_pre_global_clip": _gradient_list_norm(
            combined_gradients,
            reference,
        ),
    }
    metrics["main_physical_grad_cosine"] = _gradient_lists_cosine(
        main_gradients,
        physical_gradients,
        reference,
    )
    return combined_gradients, metrics


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
    physical_loss_enabled = _physical_loss_enabled(config, lambda_phys)
    step_times, step_sizes = rollout_step_schedule(
        batch.initial_times,
        config.num_integration_steps,
    )
    current = batch.initial_structures
    style_kl = batch.initial_times.new_zeros(batch.initial_times.shape)
    supervised_step_losses: list[torch.Tensor] = []
    position_step_losses: list[torch.Tensor] = []
    adjacency_step_losses: list[torch.Tensor] = []
    position_range_violations: list[torch.Tensor] = []
    adjacency_range_violations: list[torch.Tensor] = []

    for integration_index, flow_times in enumerate(step_times):
        current_stiffness, nodal_displacements, edge_von_mises = _conditioning_inputs(
            current=current,
            use_analysis=True,
            profile=profile,
        )
        prediction = model.predict_flow(
            structures=current,
            target_stiffness=batch.target_stiffness,
            current_stiffness=current_stiffness,
            nodal_displacements=nodal_displacements,
            edge_von_mises=edge_von_mises,
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
            style_kl = torch.nan_to_num(
                prediction.style_kl,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        safe_position_step = _safe_velocity(prediction.position_step)
        safe_adjacency_logit_step = _safe_velocity(prediction.adjacency_logit_step)

        predicted_position_step, predicted_adjacency_step = flow_step_predictions(
            position_step=safe_position_step,
            adjacency_logit_step=safe_adjacency_logit_step,
        )
        target_position_step, target_adjacency_step = local_flow_step_targets(
            current_structures=current,
            oracle_structures=batch.oracle_structures,
            flow_times=flow_times,
            step_sizes=step_sizes,
            epsilon=config.flow_target_epsilon,
        )
        position_loss = _position_step_loss(
            predicted_position_step,
            target_position_step,
            current.roles,
            beta=config.position_huber_beta,
        )
        adjacency_loss = _adjacency_step_loss(
            predicted_adjacency_step,
            target_adjacency_step,
            beta=config.adjacency_huber_beta,
        )
        supervised_loss = (
            config.position_loss_weight * position_loss
            + config.adjacency_loss_weight * adjacency_loss
        )

        supervised_step_losses.append(supervised_loss)
        position_step_losses.append(position_loss)
        adjacency_step_losses.append(adjacency_loss)
        position_range_violations.append(
            _range_violation(current.positions + predicted_position_step)
        )
        current_logits = logits_from_adjacency(current.adjacency)
        adjacency_range_violations.append(
            _range_violation(torch.sigmoid(current_logits + predicted_adjacency_step))
        )

        current = _euler_step(
            current_structures=current,
            position_step=safe_position_step,
            adjacency_logit_step=safe_adjacency_logit_step,
        )

    initial_physical_loss = None
    if physical_loss_enabled and config.relative_physical_loss_weight > 0.0:
        with torch.no_grad():
            initial_analyses = analyze_structures(
                batch.initial_structures, profile=profile
            )
            initial_structural_terms, _ = _structural_terms_and_metrics(
                analyses=initial_analyses,
                adjacency=batch.initial_structures.adjacency,
                target_stiffness=batch.target_stiffness,
                config=config,
            )
            initial_physical_loss = sum(initial_structural_terms.values()).detach()

    if physical_loss_enabled:
        _register_simulation_gradient_hooks(current, config)
    endpoint_analyses = _endpoint_analyses(
        current,
        physical_loss_enabled=physical_loss_enabled,
        profile=profile,
    )
    if endpoint_analyses is None:
        (
            endpoint_physical_loss,
            endpoint_stiffness_loss,
            endpoint_stress_loss,
            endpoint_metrics,
        ) = _zero_endpoint_physical_terms(batch.initial_times)
    else:
        structural_terms, endpoint_metrics = _structural_terms_and_metrics(
            analyses=endpoint_analyses,
            adjacency=current.adjacency,
            target_stiffness=batch.target_stiffness,
            config=config,
        )
        endpoint_stiffness_loss = structural_terms["stiffness_loss_contribution"]
        endpoint_stress_loss = structural_terms["stress_loss_contribution"]
        endpoint_absolute_physical_loss = sum(structural_terms.values())
        endpoint_relative_physical_loss = (
            endpoint_absolute_physical_loss.new_zeros(
                endpoint_absolute_physical_loss.shape
            )
            if initial_physical_loss is None
            else torch.log1p(endpoint_absolute_physical_loss)
            - torch.log1p(initial_physical_loss)
        )
        endpoint_physical_loss = (
            config.absolute_physical_loss_weight * endpoint_absolute_physical_loss
            + config.relative_physical_loss_weight * endpoint_relative_physical_loss
        )

    total_loss, loss_terms = _aggregate_rollout_losses(
        supervised_step_losses=supervised_step_losses,
        endpoint_physical_loss=endpoint_physical_loss,
        endpoint_stiffness_loss=endpoint_stiffness_loss,
        endpoint_stress_loss=endpoint_stress_loss,
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
        "position_step_loss": torch.stack(position_step_losses, dim=0)
        .mean(dim=0)
        .mean(),
        "adjacency_step_loss": torch.stack(adjacency_step_losses, dim=0)
        .mean(dim=0)
        .mean(),
        "position_range_violation": torch.stack(position_range_violations, dim=0)
        .mean(dim=0)
        .mean(),
        "adjacency_range_violation": torch.stack(adjacency_range_violations, dim=0)
        .mean(dim=0)
        .mean(),
        "stiffness_error": endpoint_metrics["stiffness_error"].mean(),
        "stress_violation": endpoint_metrics["stress_violation"].mean(),
        "mean_stress_ratio": endpoint_metrics["mean_stress_ratio"].mean(),
        "max_stress_ratio": endpoint_metrics["max_stress_ratio"].mean(),
        "style_kl": style_kl.mean(),
        "lambda_sup": batch.initial_times.new_tensor(lambda_sup),
        "lambda_phys": batch.initial_times.new_tensor(lambda_phys),
        "lambda_kl": batch.initial_times.new_tensor(lambda_kl),
        "integration_end_time": (
            batch.initial_times + config.num_integration_steps * step_sizes
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
        max_position_step=train_config.max_position_step,
        max_adjacency_logit_step=train_config.max_adjacency_logit_step,
    )
    model = SupervisedRefiner(effective_config).to(device)
    if train_config.init_checkpoint_path is None:
        return model, effective_config

    checkpoint = torch.load(train_config.init_checkpoint_path, map_location=device)
    if model_config is None:
        checkpoint_model_config = dict(checkpoint["model_config"])
        checkpoint_model_config["max_position_step"] = train_config.max_position_step
        checkpoint_model_config["max_adjacency_logit_step"] = (
            train_config.max_adjacency_logit_step
        )
        effective_config = SupervisedRefinerConfig(**checkpoint_model_config)
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
    if train_config.main_grad_clip_norm <= 0.0:
        raise ValueError("main_grad_clip_norm must be positive")
    if train_config.physical_grad_clip_norm <= 0.0:
        raise ValueError("physical_grad_clip_norm must be positive")
    if train_config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if train_config.num_integration_steps <= 0:
        raise ValueError("num_integration_steps must be positive")
    if train_config.flow_target_epsilon <= 0.0:
        raise ValueError("flow_target_epsilon must be positive")
    if train_config.max_position_step <= 0.0:
        raise ValueError("max_position_step must be positive")
    if train_config.max_adjacency_logit_step <= 0.0:
        raise ValueError("max_adjacency_logit_step must be positive")
    if train_config.position_huber_beta <= 0.0:
        raise ValueError("position_huber_beta must be positive")
    if train_config.adjacency_huber_beta <= 0.0:
        raise ValueError("adjacency_huber_beta must be positive")
    if not 0.0 <= train_config.style_token_dropout <= 1.0:
        raise ValueError("style_token_dropout must be in [0.0, 1.0]")
    if not 0.0 <= train_config.eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0.0, 1.0)")
    if train_config.allowable_von_mises <= 0.0:
        raise ValueError("allowable_von_mises must be positive")
    if not 0.0 <= train_config.stress_activation_threshold <= 1.0:
        raise ValueError("stress_activation_threshold must be in [0.0, 1.0]")
    if train_config.simulation_position_grad_clip_norm <= 0.0:
        raise ValueError("simulation_position_grad_clip_norm must be positive")
    if train_config.simulation_adjacency_grad_clip_norm <= 0.0:
        raise ValueError("simulation_adjacency_grad_clip_norm must be positive")
    if train_config.absolute_physical_loss_weight < 0.0:
        raise ValueError("absolute_physical_loss_weight must be non-negative")
    if train_config.relative_physical_loss_weight < 0.0:
        raise ValueError("relative_physical_loss_weight must be non-negative")
    _max_initial_time(train_config)

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
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(model, train_config.weight_decay),
        lr=train_config.learning_rate,
    )
    history: dict[str, list[float]] = {
        "total_loss": [],
        "supervised_loss_contribution": [],
        "physical_loss_contribution": [],
        "stiffness_loss_contribution": [],
        "stress_loss_contribution": [],
        "style_kl_loss_contribution": [],
        "nonfinite_step_skips": [],
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
        f"max_position_step={train_config.max_position_step} "
        f"max_adjacency_logit_step={train_config.max_adjacency_logit_step} "
        f"use_style_conditioning={'yes' if effective_model_config.use_style_conditioning else 'no'} "
        f"style_token_dropout={train_config.style_token_dropout:.3f} "
        f"learning_rate={train_config.learning_rate} warmup_steps={train_config.warmup_steps} "
        f"min_learning_rate={train_config.min_learning_rate} "
        f"weight_decay={train_config.weight_decay} "
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
        f"style_kl_anneal_steps={train_config.style_kl_anneal_steps} "
        f"position_huber_beta={train_config.position_huber_beta} "
        f"adjacency_huber_beta={train_config.adjacency_huber_beta} "
        f"simulation_position_grad_clip_norm={train_config.simulation_position_grad_clip_norm} "
        f"simulation_adjacency_grad_clip_norm={train_config.simulation_adjacency_grad_clip_norm} "
        f"absolute_physical_loss_weight={train_config.absolute_physical_loss_weight} "
        f"relative_physical_loss_weight={train_config.relative_physical_loss_weight}",
        flush=True,
    )

    try:
        step = 0
        examples_seen = 0
        nonfinite_step_skips = 0
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
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
                    max_initial_time=_max_initial_time(train_config),
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
                if not _all_finite_tensors(
                    [total_loss, *metrics.values(), *loss_terms.values()]
                ):
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_step_skips += 1
                    history["nonfinite_step_skips"].append(float(nonfinite_step_skips))
                    writer.add_scalar(
                        "train/gradients/nonfinite_step_skips",
                        float(nonfinite_step_skips),
                        step,
                    )
                    print(
                        f"train step={step + 1}/{train_config.num_steps} skipped_update=yes reason=nonfinite_loss "
                        f"nonfinite_step_skips={nonfinite_step_skips}",
                        flush=True,
                    )
                    step += 1
                    if step >= train_config.num_steps:
                        break
                    continue

                gradient_metrics: dict[str, float] = {}
                optimizer.zero_grad(set_to_none=True)
                combined_gradients, gradient_metric_tensors = (
                    _objective_clipped_gradients(
                        main_loss=(
                            loss_terms["supervised_loss_contribution"]
                            + loss_terms["style_kl_loss_contribution"]
                        ),
                        physical_loss=loss_terms["physical_loss_contribution"],
                        parameters=trainable_parameters,
                        config=train_config,
                    )
                )
                if train_config.log_gradient_diagnostics:
                    gradient_metrics["main_grad_norm"] = float(
                        gradient_metric_tensors["main_grad_norm_pre_clip"]
                        .detach()
                        .item()
                    )
                    gradient_metrics["physical_grad_norm"] = float(
                        gradient_metric_tensors["physical_grad_norm_pre_clip"]
                        .detach()
                        .item()
                    )
                    gradient_metrics["main_physical_grad_cosine"] = float(
                        gradient_metric_tensors["main_physical_grad_cosine"]
                        .detach()
                        .item()
                    )

                if not _all_finite_tensors(list(gradient_metric_tensors.values())):
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_step_skips += 1
                    history["nonfinite_step_skips"].append(float(nonfinite_step_skips))
                    writer.add_scalar(
                        "train/gradients/nonfinite_step_skips",
                        float(nonfinite_step_skips),
                        step,
                    )
                    print(
                        f"train step={step + 1}/{train_config.num_steps} skipped_update=yes reason=nonfinite_objective_grad "
                        f"nonfinite_step_skips={nonfinite_step_skips}",
                        flush=True,
                    )
                    step += 1
                    if step >= train_config.num_steps:
                        break
                    continue

                _assign_gradients(trainable_parameters, combined_gradients)
                global_pre_clip_norm = float(
                    gradient_metric_tensors["combined_grad_norm_pre_global_clip"]
                    .detach()
                    .item()
                )
                if not math.isfinite(global_pre_clip_norm):
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_step_skips += 1
                    history["nonfinite_step_skips"].append(float(nonfinite_step_skips))
                    writer.add_scalar(
                        "train/gradients/nonfinite_step_skips",
                        float(nonfinite_step_skips),
                        step,
                    )
                    print(
                        f"train step={step + 1}/{train_config.num_steps} skipped_update=yes reason=nonfinite_combined_grad "
                        f"nonfinite_step_skips={nonfinite_step_skips}",
                        flush=True,
                    )
                    step += 1
                    if step >= train_config.num_steps:
                        break
                    continue

                pre_global_clip_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=train_config.max_grad_norm,
                    ).item()
                )
                post_global_clip_norm = float(
                    _gradient_list_norm(
                        [parameter.grad for parameter in trainable_parameters],
                        loss_terms["total_loss"],
                    )
                    .detach()
                    .item()
                )
                global_clipped = pre_global_clip_norm > train_config.max_grad_norm
                global_clip_ratio = pre_global_clip_norm / train_config.max_grad_norm

                if train_config.log_gradient_diagnostics:
                    gradient_metrics.update(
                        {
                            "combined_grad_norm_pre_global_clip": pre_global_clip_norm,
                            "combined_grad_norm": post_global_clip_norm,
                        }
                    )

                if not math.isfinite(post_global_clip_norm):
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_step_skips += 1
                    history["nonfinite_step_skips"].append(float(nonfinite_step_skips))
                    writer.add_scalar(
                        "train/gradients/nonfinite_step_skips",
                        float(nonfinite_step_skips),
                        step,
                    )
                    print(
                        f"train step={step + 1}/{train_config.num_steps} skipped_update=yes reason=nonfinite_grad "
                        f"nonfinite_step_skips={nonfinite_step_skips}",
                        flush=True,
                    )
                    step += 1
                    if step >= train_config.num_steps:
                        break
                    continue

                for name, value_tensor in gradient_metric_tensors.items():
                    writer.add_scalar(
                        f"train/gradients/{name}",
                        float(value_tensor.detach().item()),
                        step,
                    )

                writer.add_scalar(
                    "train/gradients/combined_grad_norm",
                    post_global_clip_norm,
                    step,
                )
                writer.add_scalar(
                    "train/gradients/combined_global_clip_ratio",
                    global_clip_ratio,
                    step,
                )
                writer.add_scalar(
                    "train/gradients/combined_global_clipped",
                    float(global_clipped),
                    step,
                )

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
                writer.add_scalar(
                    "train/gradients/grad_norm", post_global_clip_norm, step
                )
                writer.add_scalar(
                    "train/gradients/clip_ratio",
                    global_clip_ratio,
                    step,
                )
                writer.add_scalar(
                    "train/gradients/clipped",
                    float(global_clipped),
                    step,
                )
                writer.add_scalar(
                    "train/gradients/nonfinite_step_skips",
                    float(nonfinite_step_skips),
                    step,
                )
                for name, value in gradient_metrics.items():
                    writer.add_scalar(f"train/gradients/{name}", value, step)
                history["nonfinite_step_skips"].append(float(nonfinite_step_skips))

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
                        f"grad_norm={post_global_clip_norm:.6f} clipped={'yes' if global_clipped else 'no'} "
                        f"nonfinite_step_skips={nonfinite_step_skips} "
                        f"clip_ratio={global_clip_ratio:.3f} "
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
