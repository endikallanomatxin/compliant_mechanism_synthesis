from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.dataset import load_offline_dataset
from compliant_mechanism_synthesis.dataset.types import (
    Analyses,
    OptimizedCases,
    Scaffolds,
    Structures,
)
from compliant_mechanism_synthesis.mechanics import (
    Frame3DConfig,
    GeometryPenaltyConfig,
    mechanical_terms,
    normalize_generalized_stiffness,
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
class CurriculumConfig:
    initial_mix: float = 0.15
    final_mix: float = 1.0
    position_noise: float = 0.008
    adjacency_noise: float = 0.04


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    dataset_path: str
    device: str = "auto"
    batch_size: int = 64
    log_every_steps: int = 10
    max_grad_norm: float = 1.0
    num_steps: int = 4_096
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    min_learning_rate: float = 1e-6
    use_style_token: bool = True
    position_loss_weight: float = 1.0
    adjacency_loss_weight: float = 0.35
    endpoint_loss_weight: float = 0.03
    stiffness_loss_weight: float = 0.0
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
    position_noise_levels: torch.Tensor
    adjacency_noise_levels: torch.Tensor
    target_position_velocity: torch.Tensor
    target_adjacency_velocity: torch.Tensor

    @property
    def noisy_structures(self) -> Structures:
        return self.source_structures


@dataclass(frozen=True)
class SupervisedTrainingSummary:
    history: dict[str, list[float]]
    checkpoint_path: Path


def _difficulty_fraction(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return min(max(step / (total_steps - 1), 0.0), 1.0)


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


def load_supervised_cases(dataset_path: str) -> OptimizedCases:
    optimized_cases, _ = load_offline_dataset(dataset_path)
    return optimized_cases


def sample_noisy_structures(
    optimized_cases: OptimizedCases,
    curriculum: CurriculumConfig,
    difficulty: float,
    seed: int | None = None,
) -> Structures:
    optimized_cases.validate()
    difficulty = float(min(max(difficulty, 0.0), 1.0))
    mix = curriculum.initial_mix + difficulty * (
        curriculum.final_mix - curriculum.initial_mix
    )

    base_positions = optimized_cases.optimized_structures.positions + mix * (
        optimized_cases.raw_structures.positions
        - optimized_cases.optimized_structures.positions
    )
    base_adjacency = optimized_cases.optimized_structures.adjacency + mix * (
        optimized_cases.raw_structures.adjacency
        - optimized_cases.optimized_structures.adjacency
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=base_positions.device).manual_seed(seed)

    position_noise = (
        curriculum.position_noise
        * difficulty
        * torch.randn(
            base_positions.shape,
            generator=generator,
            device=base_positions.device,
            dtype=base_positions.dtype,
        )
    )
    adjacency_noise = (
        curriculum.adjacency_noise
        * difficulty
        * torch.randn(
            base_adjacency.shape,
            generator=generator,
            device=base_adjacency.device,
            dtype=base_adjacency.dtype,
        )
    )

    noisy_positions = (base_positions + position_noise).clamp(0.0, 1.0)
    noisy_adjacency = enforce_role_adjacency_constraints(
        symmetrize_matrix((base_adjacency + adjacency_noise).clamp(0.0, 1.0)),
        optimized_cases.raw_structures.roles,
    )

    _, _, free_mask = role_masks(optimized_cases.raw_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=noisy_positions.dtype)
    noisy_positions = (
        optimized_cases.raw_structures.positions * (1.0 - free_mask)
        + noisy_positions * free_mask
    )
    structures = Structures(
        positions=noisy_positions,
        roles=optimized_cases.raw_structures.roles,
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
        nodal_mechanics=terms["nodal_mechanics"],
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


def _flow_noise_levels(
    source_structures: Structures,
    oracle_structures: Structures,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, free_mask = role_masks(source_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=source_structures.positions.dtype)
    free_count = free_mask.sum(dim=(1, 2)).clamp_min(1.0)
    position_gap = (
        (
            (oracle_structures.positions - source_structures.positions).square()
            * free_mask
        ).sum(dim=(1, 2))
        / free_count
    ).sqrt()
    adjacency_gap = (
        (oracle_structures.adjacency - source_structures.adjacency)
        .square()
        .mean(dim=(1, 2))
    ).sqrt()
    return position_gap, adjacency_gap


def make_supervised_batch(
    optimized_cases: OptimizedCases,
    curriculum: CurriculumConfig,
    difficulty: float,
    seed: int | None = None,
    profile: dict[str, float] | None = None,
) -> SupervisedBatch:
    source_structures = sample_noisy_structures(
        optimized_cases=optimized_cases,
        curriculum=curriculum,
        difficulty=difficulty,
        seed=seed,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator(
            device=source_structures.positions.device
        ).manual_seed(seed + 10_000)
    flow_times = torch.rand(
        (optimized_cases.raw_structures.batch_size,),
        generator=generator,
        device=source_structures.positions.device,
        dtype=source_structures.positions.dtype,
    )
    # Sample supervision on points along the straight path from the noisy
    # source structure to the optimized oracle, but train the model to predict
    # the residual correction from the current flow state to the oracle.
    interpolation = flow_times[:, None, None]
    flow_positions = torch.lerp(
        source_structures.positions,
        optimized_cases.optimized_structures.positions,
        interpolation,
    )
    flow_adjacency = torch.lerp(
        source_structures.adjacency,
        optimized_cases.optimized_structures.adjacency,
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
    position_noise_levels, adjacency_noise_levels = _flow_noise_levels(
        source_structures=source_structures,
        oracle_structures=optimized_cases.optimized_structures,
    )
    return SupervisedBatch(
        source_structures=source_structures,
        flow_structures=flow_structures,
        target_stiffness=optimized_cases.target_stiffness,
        oracle_structures=optimized_cases.optimized_structures,
        oracle_analyses=optimized_cases.last_analyses,
        current_analyses=current_analyses,
        flow_times=flow_times,
        position_noise_levels=position_noise_levels,
        adjacency_noise_levels=adjacency_noise_levels,
        target_position_velocity=optimized_cases.optimized_structures.positions
        - flow_structures.positions,
        target_adjacency_velocity=optimized_cases.optimized_structures.adjacency
        - flow_structures.adjacency,
    )


def select_batch(
    optimized_cases: OptimizedCases,
    batch_indices: torch.Tensor,
) -> OptimizedCases:
    if batch_indices.ndim != 1:
        raise ValueError("batch_indices must be one-dimensional")
    return optimized_cases.index_select(batch_indices)


def iter_supervised_batches(
    optimized_cases: OptimizedCases,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    optimized_cases.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    index_device = optimized_cases.raw_structures.positions.device
    indices = torch.arange(
        optimized_cases.raw_structures.batch_size,
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


def generalized_stiffness_error(
    generalized_stiffness: torch.Tensor,
    target_stiffness: torch.Tensor,
) -> torch.Tensor:
    normalized_current = normalize_generalized_stiffness(generalized_stiffness)
    normalized_target = normalize_generalized_stiffness(target_stiffness)
    return (normalized_current - normalized_target).square().mean(dim=(1, 2))


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


def _endpoint_loss(
    prediction: FlowPrediction, batch: SupervisedBatch
) -> tuple[torch.Tensor, Structures]:
    _, _, free_mask = role_masks(batch.flow_structures.roles)
    free_mask = free_mask.unsqueeze(-1).to(dtype=prediction.position_velocity.dtype)
    estimated_positions = (
        batch.flow_structures.positions + prediction.position_velocity * free_mask
    ).clamp(0.0, 1.0)
    estimated_adjacency = enforce_role_adjacency_constraints(
        (batch.flow_structures.adjacency + prediction.adjacency_velocity).clamp(
            0.0, 1.0
        ),
        batch.flow_structures.roles,
    )
    # The supervised target is the residual from the current flow state to the
    # optimized oracle, so a locally correct prediction should land on the
    # oracle in one step.
    estimated = Structures(
        positions=estimated_positions,
        roles=batch.flow_structures.roles,
        adjacency=estimated_adjacency,
    )
    position_error = (
        (estimated.positions - batch.oracle_structures.positions).square() * free_mask
    ).sum(dim=(1, 2))
    position_error = position_error / free_mask.sum(dim=(1, 2)).clamp_min(1.0)
    adjacency_error = (
        (estimated.adjacency - batch.oracle_structures.adjacency)
        .square()
        .mean(dim=(1, 2))
    )
    return position_error + adjacency_error, estimated


def _training_losses(
    prediction: FlowPrediction,
    batch: SupervisedBatch,
    config: SupervisedTrainingConfig,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    position_error = _position_velocity_loss(prediction, batch)
    adjacency_error = _adjacency_velocity_loss(prediction, batch)
    endpoint_error, estimated = _endpoint_loss(prediction, batch)
    if config.stiffness_loss_weight > 0.0:
        endpoint_analyses = analyze_structures(estimated, profile=profile)
        stiffness_error = torch.log1p(
            generalized_stiffness_error(
                endpoint_analyses.generalized_stiffness, batch.target_stiffness
            )
        )
    else:
        stiffness_error = endpoint_error.new_zeros(endpoint_error.shape)
    loss_contributions = {
        "position_error_loss_contribution": config.position_loss_weight
        * position_error,
        "adjacency_error_loss_contribution": config.adjacency_loss_weight
        * adjacency_error,
        "endpoint_error_loss_contribution": config.endpoint_loss_weight
        * endpoint_error,
        "stiffness_error_loss_contribution": config.stiffness_loss_weight
        * stiffness_error,
    }
    total_loss = (
        loss_contributions["position_error_loss_contribution"]
        + loss_contributions["adjacency_error_loss_contribution"]
        + loss_contributions["endpoint_error_loss_contribution"]
        + loss_contributions["stiffness_error_loss_contribution"]
    )
    metrics = {
        "position_error": position_error.mean(),
        "adjacency_error": adjacency_error.mean(),
        "endpoint_error": endpoint_error.mean(),
        "stiffness_error": stiffness_error.mean(),
    }
    loss_summary = {name: value.mean() for name, value in loss_contributions.items()}
    loss_summary["total_loss"] = total_loss.mean()
    return total_loss.mean(), metrics, loss_summary


def train_supervised_refiner(
    optimized_cases: OptimizedCases,
    model_config: SupervisedRefinerConfig | None = None,
    train_config: SupervisedTrainingConfig | None = None,
    curriculum: CurriculumConfig | None = None,
) -> tuple[SupervisedRefiner, SupervisedTrainingSummary]:
    optimized_cases.validate()
    train_config = train_config or SupervisedTrainingConfig(dataset_path="")
    model_config = model_config or SupervisedRefinerConfig(
        use_style_token=train_config.use_style_token
    )
    curriculum = curriculum or CurriculumConfig()
    device = resolve_torch_device(train_config.device)

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")
    if train_config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if train_config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if train_config.min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if train_config.min_learning_rate > train_config.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate")

    dataset_cases = optimized_cases.raw_structures.batch_size
    steps_per_epoch = max(1, math.ceil(dataset_cases / train_config.batch_size))
    model = SupervisedRefiner(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history = {
        "total_loss": [],
        "position_error_loss_contribution": [],
        "adjacency_error_loss_contribution": [],
        "endpoint_error_loss_contribution": [],
        "stiffness_error_loss_contribution": [],
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
        f"steps_per_epoch={steps_per_epoch} device={device} "
        f"use_style_token={'yes' if model.config.use_style_token else 'no'} "
        f"learning_rate={train_config.learning_rate} warmup_steps={train_config.warmup_steps} "
        f"min_learning_rate={train_config.min_learning_rate}",
        flush=True,
    )
    try:
        step = 0
        examples_seen = 0
        while step < train_config.num_steps:
            for batch_cases in iter_supervised_batches(
                optimized_cases,
                batch_size=train_config.batch_size,
                shuffle=True,
                seed=train_config.seed + step,
            ):
                _synchronize_device_if_needed(device)
                step_start_time = time.perf_counter()
                batch_cases = batch_cases.to(device)
                _synchronize_device_if_needed(device)
                batch_transfer_time = time.perf_counter()
                difficulty = _difficulty_fraction(step, train_config.num_steps)
                learning_rate = _scheduled_learning_rate(step, train_config)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                batch_analysis_profile: dict[str, float] = {}
                batch = make_supervised_batch(
                    optimized_cases=batch_cases,
                    curriculum=curriculum,
                    difficulty=difficulty,
                    seed=train_config.seed + step,
                    profile=batch_analysis_profile,
                )
                _synchronize_device_if_needed(device)
                batch_build_time = time.perf_counter()
                prediction = model.predict_flow(
                    structures=batch.flow_structures,
                    target_stiffness=batch.target_stiffness,
                    current_stiffness=batch.current_analyses.generalized_stiffness,
                    nodal_mechanics=batch.current_analyses.nodal_mechanics,
                    flow_times=batch.flow_times,
                    position_noise_levels=batch.position_noise_levels,
                    adjacency_noise_levels=batch.adjacency_noise_levels,
                    style_structures=(
                        batch.oracle_structures
                        if model.config.use_style_token
                        else None
                    ),
                )
                _synchronize_device_if_needed(device)
                forward_time = time.perf_counter()
                loss_analysis_profile: dict[str, float] = {}
                total_loss, metrics, loss_terms = _training_losses(
                    prediction, batch, train_config, profile=loss_analysis_profile
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
                writer.add_scalar("train/curriculum/difficulty", difficulty, step)
                writer.add_scalar(
                    "train/curriculum/flow_time_mean",
                    float(batch.flow_times.mean().item()),
                    step,
                )
                writer.add_scalar("train/parameters/learning_rate", learning_rate, step)
                writer.add_scalar("train/gradients/grad_norm", grad_norm, step)
                writer.add_scalar("train/gradients/clip_ratio", clip_ratio, step)
                writer.add_scalar("train/gradients/clipped", float(clipped), step)
                examples_seen += batch_cases.raw_structures.batch_size
                if (
                    step % train_config.log_every_steps == 0
                    or step == train_config.num_steps - 1
                ):
                    print(
                        f"train step={step + 1}/{train_config.num_steps} "
                        f"batch_cases={batch_cases.raw_structures.batch_size} "
                        f"examples_seen={examples_seen} difficulty={difficulty:.3f} "
                        f"learning_rate={learning_rate:.8f} "
                        f"loss_total={history['total_loss'][-1]:.6f} "
                        f"loss_position={history['position_error_loss_contribution'][-1]:.6f} "
                        f"loss_adjacency={history['adjacency_error_loss_contribution'][-1]:.6f} "
                        f"loss_endpoint={history['endpoint_error_loss_contribution'][-1]:.6f} "
                        f"loss_stiffness={history['stiffness_error_loss_contribution'][-1]:.6f} "
                        f"metric_position={float(metrics['position_error'].detach().item()):.6f} "
                        f"metric_adjacency={float(metrics['adjacency_error'].detach().item()):.6f} "
                        f"metric_endpoint={float(metrics['endpoint_error'].detach().item()):.6f} "
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
                "curriculum_config": asdict(curriculum),
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
