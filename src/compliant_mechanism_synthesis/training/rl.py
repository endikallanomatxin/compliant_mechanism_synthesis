from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Structures,
)
from compliant_mechanism_synthesis.losses import (
    log_generalized_stiffness_error,
    StructuralObjectiveWeights,
    structural_objective_terms,
)
from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import (
    enforce_role_adjacency_constraints,
    symmetrize_matrix,
)
from compliant_mechanism_synthesis.training.supervised import (
    _dataset_adjacency_statistics,
    _dataset_position_statistics,
    _scheduled_learning_rate,
    _synchronize_device_if_needed,
    iter_supervised_batches,
    load_supervised_cases,
    sample_noisy_structures,
)
from compliant_mechanism_synthesis.training.supervised import analyze_structures
from compliant_mechanism_synthesis.utils import resolve_torch_device


@dataclass(frozen=True)
class RLTrainingConfig:
    dataset_path: str
    device: str = "auto"
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    log_every_steps: int = 10
    max_grad_norm: float = 1.0
    num_steps: int = 50_000
    rollout_steps: int = 2
    learning_rate: float = 1e-7
    warmup_steps: int = 1000
    min_learning_rate: float = 1e-8
    loss_scale: float = 1e-8
    stiffness_loss_weight: float = 1.0
    stress_loss_weight: float = 0.1
    allowable_von_mises: float = 250e6
    stress_activation_threshold: float = 0.15
    rollout_monotonicity_loss_weight: float = 0.01
    material_loss_weight: float = 250.0
    short_beam_penalty_weight: float = 5.0
    long_beam_penalty_weight: float = 5.0
    thin_beam_penalty_weight: float = 5.0
    thick_beam_penalty_weight: float = 5.0
    free_node_spacing_penalty_weight: float = 5.0
    init_checkpoint_path: str | None = None
    checkpoint_path: str | None = None
    logdir: str = "runs/rl"
    seed: int = 7


@dataclass(frozen=True)
class RLTrainingSummary:
    history: dict[str, list[float]]
    checkpoint_path: Path


@dataclass(frozen=True)
class RolloutOutcome:
    final_structures: Structures
    step_analyses: list[Analyses]


def _rollout_refiner_final(
    model: SupervisedRefiner,
    source_structures: Structures,
    target_stiffness: torch.Tensor,
    num_steps: int,
) -> RolloutOutcome:
    source_structures.validate()
    if target_stiffness.shape != (source_structures.batch_size, 6, 6):
        raise ValueError("target_stiffness must have shape [batch, 6, 6]")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    current = source_structures
    _, _, free_mask = role_masks(current.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=current.positions.dtype)
    step_analyses: list[Analyses] = []
    for step in range(num_steps):
        analyses = analyze_structures(current)
        step_analyses.append(analyses)
        if analyses.nodal_displacements is None:
            raise ValueError("analyze_structures must provide nodal_displacements")
        if analyses.edge_von_mises is None:
            raise ValueError("analyze_structures must provide edge_von_mises")
        flow_time = current.positions.new_full(
            (current.batch_size,),
            (step + 0.5) / num_steps,
        )
        prediction = model.predict_flow(
            structures=current,
            target_stiffness=target_stiffness,
            current_stiffness=analyses.generalized_stiffness,
            nodal_displacements=analyses.nodal_displacements,
            edge_von_mises=analyses.edge_von_mises,
            flow_times=flow_time,
        )
        safe_position_velocity = torch.nan_to_num(
            prediction.position_velocity,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        safe_adjacency_velocity = torch.nan_to_num(
            prediction.adjacency_velocity,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        step_size = 1.0 / num_steps
        positions = torch.nan_to_num(
            current.positions + step_size * safe_position_velocity * free_mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        adjacency = enforce_role_adjacency_constraints(
            symmetrize_matrix(
                torch.nan_to_num(
                    current.adjacency + step_size * safe_adjacency_velocity,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
            ),
            current.roles,
        )
        current = Structures(
            positions=positions,
            roles=current.roles,
            adjacency=adjacency,
        )
    current.validate()
    final_analyses = analyze_structures(current)
    step_analyses.append(final_analyses)
    return RolloutOutcome(final_structures=current, step_analyses=step_analyses)


def _rl_losses(
    final_analyses: Analyses,
    final_adjacency: torch.Tensor,
    target_stiffness: torch.Tensor,
    config: RLTrainingConfig,
    step_analyses: list[Analyses] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stiffness_error = log_generalized_stiffness_error(
        final_analyses.generalized_stiffness,
        target_stiffness,
    )
    base_loss_contributions, metrics = structural_objective_terms(
        analyses=final_analyses,
        adjacency=final_adjacency,
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
    if step_analyses is None or len(step_analyses) < 2:
        rollout_monotonicity = stiffness_error.new_zeros(stiffness_error.shape)
    else:
        per_step_stiffness_losses = [
            log_generalized_stiffness_error(
                analysis.generalized_stiffness, target_stiffness
            )
            for analysis in step_analyses
        ]
        monotonicity_terms = [
            torch.relu(current - previous)
            for previous, current in zip(
                per_step_stiffness_losses[:-1],
                per_step_stiffness_losses[1:],
                strict=False,
            )
        ]
        rollout_monotonicity = torch.stack(monotonicity_terms, dim=0).mean(dim=0)

    loss_contributions = {
        **base_loss_contributions,
        "rollout_monotonicity_loss_contribution": config.rollout_monotonicity_loss_weight
        * rollout_monotonicity,
    }
    total_loss = sum(loss_contributions.values())
    metrics = {
        **metrics,
        "rollout_monotonicity": rollout_monotonicity.mean(),
        "reward": -total_loss.mean(),
    }
    loss_summary = {name: value.mean() for name, value in loss_contributions.items()}
    loss_summary["total_loss"] = total_loss.mean()
    return total_loss.mean(), metrics, loss_summary


def _load_initial_model(
    device: torch.device,
    train_config: RLTrainingConfig,
    model_config: SupervisedRefinerConfig | None,
) -> tuple[SupervisedRefiner, SupervisedRefinerConfig]:
    effective_config = replace(
        model_config or SupervisedRefinerConfig(),
        use_style_conditioning=False,
    )
    model = SupervisedRefiner(effective_config).to(device)
    if train_config.init_checkpoint_path is None:
        return model, effective_config

    checkpoint = torch.load(train_config.init_checkpoint_path, map_location=device)
    if model_config is None:
        effective_config = replace(
            SupervisedRefinerConfig(**checkpoint["model_config"]),
            use_style_conditioning=False,
        )
        model = SupervisedRefiner(effective_config).to(device)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            "RL initialization left missing model weights after disabling style conditioning: "
            + ", ".join(incompatible.missing_keys)
        )
    unexpected_keys = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("style_local_encoder.")
    ]
    if unexpected_keys:
        raise RuntimeError(
            "RL initialization found unexpected non-style weights: "
            + ", ".join(unexpected_keys)
        )
    return model, effective_config


def train_rl_refiner(
    optimized_cases: OptimizedCases,
    model_config: SupervisedRefinerConfig | None = None,
    train_config: RLTrainingConfig | None = None,
) -> tuple[SupervisedRefiner, RLTrainingSummary]:
    optimized_cases.validate()
    train_config = train_config or RLTrainingConfig(dataset_path="")
    device = resolve_torch_device(train_config.device)

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")
    if train_config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if train_config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if train_config.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if train_config.loss_scale <= 0.0:
        raise ValueError("loss_scale must be positive")
    if train_config.rollout_monotonicity_loss_weight < 0.0:
        raise ValueError("rollout_monotonicity_loss_weight must be non-negative")
    if train_config.stress_loss_weight < 0.0:
        raise ValueError("stress_loss_weight must be non-negative")
    if train_config.allowable_von_mises <= 0.0:
        raise ValueError("allowable_von_mises must be positive")
    if not 0.0 <= train_config.stress_activation_threshold <= 1.0:
        raise ValueError("stress_activation_threshold must be in [0.0, 1.0]")
    if train_config.material_loss_weight < 0.0:
        raise ValueError("material_loss_weight must be non-negative")
    if train_config.short_beam_penalty_weight < 0.0:
        raise ValueError("short_beam_penalty_weight must be non-negative")
    if train_config.long_beam_penalty_weight < 0.0:
        raise ValueError("long_beam_penalty_weight must be non-negative")
    if train_config.thin_beam_penalty_weight < 0.0:
        raise ValueError("thin_beam_penalty_weight must be non-negative")
    if train_config.thick_beam_penalty_weight < 0.0:
        raise ValueError("thick_beam_penalty_weight must be non-negative")
    if train_config.free_node_spacing_penalty_weight < 0.0:
        raise ValueError("free_node_spacing_penalty_weight must be non-negative")
    train_cases = optimized_cases
    dataset_cases = train_cases.optimized_structures.batch_size
    steps_per_epoch = max(1, math.ceil(dataset_cases / train_config.batch_size))
    global_position_mean, global_position_std = _dataset_position_statistics(
        train_cases
    )
    global_adjacency_mean, global_adjacency_std = _dataset_adjacency_statistics(
        train_cases
    )
    model, effective_model_config = _load_initial_model(
        device, train_config, model_config
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history = {
        "total_loss": [],
        "stiffness_loss_contribution": [],
        "stress_loss_contribution": [],
        "rollout_monotonicity_loss_contribution": [],
        "material_loss_contribution": [],
        "short_beam_loss_contribution": [],
        "long_beam_loss_contribution": [],
        "thin_beam_loss_contribution": [],
        "thick_beam_loss_contribution": [],
        "free_node_spacing_loss_contribution": [],
    }
    checkpoint_path = (
        Path(train_config.checkpoint_path)
        if train_config.checkpoint_path is not None
        else Path(train_config.logdir) / "refiner.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=train_config.logdir)
    print(
        f"rl training started dataset_cases={dataset_cases} batch_size={train_config.batch_size} "
        f"effective_batch_size={train_config.batch_size * train_config.gradient_accumulation_steps} "
        f"rollout_steps={train_config.rollout_steps} "
        f"steps_per_epoch={steps_per_epoch} device={device} "
        f"init_checkpoint={train_config.init_checkpoint_path or 'none'} "
        f"use_style_conditioning={'yes' if effective_model_config.use_style_conditioning else 'no'} "
        f"style_local_latent_dim={effective_model_config.style_local_latent_dim} "
        f"learning_rate={train_config.learning_rate} warmup_steps={train_config.warmup_steps} "
        f"min_learning_rate={train_config.min_learning_rate} "
        f"loss_scale={train_config.loss_scale:.3e} "
        f"stiffness_loss_weight={train_config.stiffness_loss_weight} "
        f"stress_loss_weight={train_config.stress_loss_weight} "
        f"allowable_von_mises={train_config.allowable_von_mises:.3e} "
        f"stress_activation_threshold={train_config.stress_activation_threshold:.3f} "
        f"rollout_monotonicity_loss_weight={train_config.rollout_monotonicity_loss_weight}",
        flush=True,
    )
    try:
        step = 0
        examples_seen = 0
        accumulation_counter = 0
        accumulation_examples = 0
        accumulation_learning_rate = 0.0
        accumulation_step_start_time = 0.0
        accumulation_transfer_time = 0.0
        accumulation_batch_time = 0.0
        accumulation_rollout_time = 0.0
        accumulation_loss_time = 0.0
        accumulated_loss_terms: dict[str, float] = {}
        accumulated_metrics: dict[str, float] = {}
        nonfinite_loss_skips = 0
        optimizer.zero_grad(set_to_none=True)

        def finalize_accumulation() -> None:
            nonlocal step
            nonlocal accumulation_counter
            nonlocal accumulation_examples
            nonlocal accumulation_learning_rate
            nonlocal accumulation_step_start_time
            nonlocal accumulation_transfer_time
            nonlocal accumulation_batch_time
            nonlocal accumulation_rollout_time
            nonlocal accumulation_loss_time
            nonlocal accumulated_loss_terms
            nonlocal accumulated_metrics
            nonlocal examples_seen
            nonlocal nonfinite_loss_skips
            if accumulation_counter == 0 or step >= train_config.num_steps:
                return

            raw_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config.max_grad_norm
                ).item()
            )
            skipped_update = not math.isfinite(raw_grad_norm)
            grad_norm = 0.0 if skipped_update else raw_grad_norm
            clipped = (grad_norm > train_config.max_grad_norm) and not skipped_update
            clip_ratio = (
                0.0 if skipped_update else grad_norm / train_config.max_grad_norm
            )
            if skipped_update:
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            _synchronize_device_if_needed(device)
            optimizer_time = time.perf_counter()

            averaged_loss_terms = {
                name: value / accumulation_counter
                for name, value in accumulated_loss_terms.items()
            }
            averaged_metrics = {
                name: value / accumulation_counter
                for name, value in accumulated_metrics.items()
            }
            for name, value in averaged_loss_terms.items():
                history[name].append(value)
                writer.add_scalar(f"train/loss/{name}", value, step)
            for name, value in averaged_metrics.items():
                writer.add_scalar(f"train/metrics/{name}", value, step)
            writer.add_scalar(
                "train/parameters/learning_rate", accumulation_learning_rate, step
            )
            writer.add_scalar(
                "train/parameters/effective_batch_size",
                train_config.batch_size * accumulation_counter,
                step,
            )
            writer.add_scalar("train/gradients/grad_norm", grad_norm, step)
            writer.add_scalar("train/gradients/clip_ratio", clip_ratio, step)
            writer.add_scalar("train/gradients/clipped", float(clipped), step)
            writer.add_scalar(
                "train/gradients/skipped_update",
                float(skipped_update),
                step,
            )
            writer.add_scalar(
                "train/gradients/nonfinite_loss_skips",
                float(nonfinite_loss_skips),
                step,
            )

            examples_seen += accumulation_examples
            if (
                step % train_config.log_every_steps == 0
                or step == train_config.num_steps - 1
            ):
                optimizer_duration = max(
                    0.0,
                    optimizer_time
                    - accumulation_step_start_time
                    - accumulation_transfer_time
                    - accumulation_batch_time
                    - accumulation_rollout_time
                    - accumulation_loss_time,
                )
                print(
                    f"rl train step={step + 1}/{train_config.num_steps} "
                    f"microbatches={accumulation_counter} "
                    f"batch_cases={train_config.batch_size} "
                    f"effective_batch_cases={accumulation_examples} "
                    f"examples_seen={examples_seen} "
                    f"learning_rate={accumulation_learning_rate:.8f} "
                    f"loss_total={averaged_loss_terms['total_loss']:.6f} "
                    f"loss_stiffness={averaged_loss_terms['stiffness_loss_contribution']:.6f} "
                    f"metric_stiffness={averaged_metrics['stiffness_error']:.6f} "
                    f"reward={averaged_metrics['reward']:.6f} "
                    f"grad_norm={grad_norm:.6f} clipped={'yes' if clipped else 'no'} "
                    f"skipped_update={'yes' if skipped_update else 'no'} "
                    f"nonfinite_loss_skips={nonfinite_loss_skips} "
                    f"clip_ratio={clip_ratio:.3f} "
                    f"t_transfer={accumulation_transfer_time:.3f}s "
                    f"t_batch={accumulation_batch_time:.3f}s "
                    f"t_rollout={accumulation_rollout_time:.3f}s "
                    f"t_loss={accumulation_loss_time:.3f}s "
                    f"t_opt={optimizer_duration:.3f}s "
                    f"t_total={optimizer_time - accumulation_step_start_time:.3f}s",
                    flush=True,
                )

            step += 1
            accumulation_counter = 0
            accumulation_examples = 0
            accumulation_learning_rate = 0.0
            accumulation_step_start_time = 0.0
            accumulation_transfer_time = 0.0
            accumulation_batch_time = 0.0
            accumulation_rollout_time = 0.0
            accumulation_loss_time = 0.0
            accumulated_loss_terms = {}
            accumulated_metrics = {}
            nonfinite_loss_skips = 0

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
                if accumulation_counter == 0:
                    accumulation_step_start_time = step_start_time
                    accumulation_learning_rate = _scheduled_learning_rate(
                        step, train_config
                    )
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = accumulation_learning_rate
                source_structures = sample_noisy_structures(
                    optimized_cases=batch_cases,
                    position_mean=global_position_mean.to(device),
                    position_std=global_position_std.to(device),
                    adjacency_mean=global_adjacency_mean.to(device),
                    adjacency_std=global_adjacency_std.to(device),
                    seed=train_config.seed + step,
                )
                _synchronize_device_if_needed(device)
                batch_build_time = time.perf_counter()
                rollout_outcome = _rollout_refiner_final(
                    model=model,
                    source_structures=source_structures,
                    target_stiffness=batch_cases.target_stiffness,
                    num_steps=train_config.rollout_steps,
                )
                _synchronize_device_if_needed(device)
                forward_time = time.perf_counter()
                final_structures = rollout_outcome.final_structures
                final_analyses = rollout_outcome.step_analyses[-1]
                total_loss, metrics, loss_terms = _rl_losses(
                    final_analyses=final_analyses,
                    final_adjacency=final_structures.adjacency,
                    target_stiffness=batch_cases.target_stiffness,
                    config=train_config,
                    step_analyses=rollout_outcome.step_analyses,
                )
                _synchronize_device_if_needed(device)
                loss_time = time.perf_counter()
                if not torch.isfinite(total_loss):
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                    accumulation_examples = 0
                    accumulation_learning_rate = 0.0
                    accumulation_step_start_time = 0.0
                    accumulation_transfer_time = 0.0
                    accumulation_batch_time = 0.0
                    accumulation_rollout_time = 0.0
                    accumulation_loss_time = 0.0
                    accumulated_loss_terms = {}
                    accumulated_metrics = {}
                    nonfinite_loss_skips += 1
                    continue
                scaled_total_loss = total_loss * train_config.loss_scale
                (
                    scaled_total_loss / train_config.gradient_accumulation_steps
                ).backward()

                accumulation_counter += 1
                accumulation_examples += batch_cases.optimized_structures.batch_size
                accumulation_transfer_time += batch_transfer_time - step_start_time
                accumulation_batch_time += batch_build_time - batch_transfer_time
                accumulation_rollout_time += forward_time - batch_build_time
                accumulation_loss_time += loss_time - forward_time
                for name, value_tensor in loss_terms.items():
                    accumulated_loss_terms[name] = accumulated_loss_terms.get(
                        name, 0.0
                    ) + float(value_tensor.detach().item())
                writer.add_scalar(
                    "train/parameters/mean_final_adjacency",
                    float(final_structures.adjacency.mean().detach().item()),
                    step,
                )
                writer.add_scalar(
                    "train/parameters/mean_final_position",
                    float(final_structures.positions.mean().detach().item()),
                    step,
                )
                for name, value_tensor in metrics.items():
                    accumulated_metrics[name] = accumulated_metrics.get(
                        name, 0.0
                    ) + float(value_tensor.detach().item())

                if accumulation_counter >= train_config.gradient_accumulation_steps:
                    finalize_accumulation()
                if step >= train_config.num_steps:
                    break
            finalize_accumulation()

        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "model_config": asdict(effective_model_config),
                "train_config": asdict(train_config),
                "history": history,
            },
            checkpoint_path,
        )
    finally:
        writer.close()

    model = model.to("cpu")
    return model, RLTrainingSummary(history=history, checkpoint_path=checkpoint_path)


def run_rl_training(config: RLTrainingConfig) -> None:
    optimized_cases = load_supervised_cases(config.dataset_path)
    _, summary = train_rl_refiner(
        optimized_cases=optimized_cases,
        train_config=config,
    )
    print(f"checkpoint={summary.checkpoint_path}")
    print(f"final_total_loss={summary.history['total_loss'][-1]:.6f}")
