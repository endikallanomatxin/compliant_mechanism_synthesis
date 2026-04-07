from __future__ import annotations

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
    position_noise: float = 0.01
    adjacency_noise: float = 0.04


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    dataset_path: str
    device: str = "auto"
    batch_size: int = 256
    num_steps: int = 20_000
    learning_rate: float = 1e-4
    position_loss_weight: float = 1.0
    adjacency_loss_weight: float = 0.5
    endpoint_loss_weight: float = 0.1
    stiffness_loss_weight: float = 0.02
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
    )


def analyze_structures(
    structures: Structures,
    frame_config: Frame3DConfig | None = None,
    geometry_config: GeometryPenaltyConfig | None = None,
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
    # Conditional flow matching trains on points along the straight transport
    # path from the noisy source structure to the optimized oracle.
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
    current_analyses = analyze_structures(flow_structures)
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
        - source_structures.positions,
        target_adjacency_velocity=optimized_cases.optimized_structures.adjacency
        - source_structures.adjacency,
    )


def select_batch(
    optimized_cases: OptimizedCases,
    batch_indices: torch.Tensor,
) -> OptimizedCases:
    if batch_indices.ndim != 1:
        raise ValueError("batch_indices must be one-dimensional")
    return OptimizedCases(
        raw_structures=Structures(
            positions=optimized_cases.raw_structures.positions.index_select(
                0, batch_indices
            ),
            roles=optimized_cases.raw_structures.roles.index_select(0, batch_indices),
            adjacency=optimized_cases.raw_structures.adjacency.index_select(
                0, batch_indices
            ),
        ),
        target_stiffness=optimized_cases.target_stiffness.index_select(
            0, batch_indices
        ),
        optimized_structures=Structures(
            positions=optimized_cases.optimized_structures.positions.index_select(
                0, batch_indices
            ),
            roles=optimized_cases.optimized_structures.roles.index_select(
                0, batch_indices
            ),
            adjacency=optimized_cases.optimized_structures.adjacency.index_select(
                0, batch_indices
            ),
        ),
        initial_loss=optimized_cases.initial_loss.index_select(0, batch_indices),
        best_loss=optimized_cases.best_loss.index_select(0, batch_indices),
        last_analyses=Analyses(
            generalized_stiffness=optimized_cases.last_analyses.generalized_stiffness.index_select(
                0, batch_indices
            ),
            material_usage=optimized_cases.last_analyses.material_usage.index_select(
                0, batch_indices
            ),
            short_beam_penalty=optimized_cases.last_analyses.short_beam_penalty.index_select(
                0, batch_indices
            ),
            long_beam_penalty=optimized_cases.last_analyses.long_beam_penalty.index_select(
                0, batch_indices
            ),
            thin_beam_penalty=optimized_cases.last_analyses.thin_beam_penalty.index_select(
                0, batch_indices
            ),
            thick_beam_penalty=optimized_cases.last_analyses.thick_beam_penalty.index_select(
                0, batch_indices
            ),
            free_node_spacing_penalty=optimized_cases.last_analyses.free_node_spacing_penalty.index_select(
                0, batch_indices
            ),
        ),
        scaffolds=(
            None
            if optimized_cases.scaffolds is None
            else Scaffolds(
                positions=optimized_cases.scaffolds.positions.index_select(
                    0, batch_indices
                ),
                roles=optimized_cases.scaffolds.roles.index_select(0, batch_indices),
                adjacency=optimized_cases.scaffolds.adjacency.index_select(
                    0, batch_indices
                ),
                edge_primitive_types=optimized_cases.scaffolds.edge_primitive_types.index_select(
                    0, batch_indices
                ),
            )
        ),
    )


def iter_supervised_minibatches(
    optimized_cases: OptimizedCases,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    optimized_cases.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(optimized_cases.raw_structures.batch_size))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    index_device = optimized_cases.raw_structures.positions.device
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(
            indices[start : start + batch_size], dtype=torch.long, device=index_device
        )
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
    remaining = (1.0 - batch.flow_times)[:, None, None]
    estimated_positions = (
        batch.flow_structures.positions
        + remaining * prediction.position_velocity * free_mask
    ).clamp(0.0, 1.0)
    estimated_adjacency = enforce_role_adjacency_constraints(
        (
            batch.flow_structures.adjacency + remaining * prediction.adjacency_velocity
        ).clamp(0.0, 1.0),
        batch.flow_structures.roles,
    )
    # This endpoint estimate is an inexpensive consistency term: if the
    # predicted flow is locally correct, integrating the remaining segment of
    # the path should land near the optimized structure.
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    position_loss = _position_velocity_loss(prediction, batch)
    adjacency_loss = _adjacency_velocity_loss(prediction, batch)
    endpoint_loss, estimated = _endpoint_loss(prediction, batch)
    endpoint_analyses = analyze_structures(estimated)
    stiffness_loss = torch.log1p(
        generalized_stiffness_error(
            endpoint_analyses.generalized_stiffness, batch.target_stiffness
        )
    )
    total = (
        config.position_loss_weight * position_loss
        + config.adjacency_loss_weight * adjacency_loss
        + config.endpoint_loss_weight * endpoint_loss
        + config.stiffness_loss_weight * stiffness_loss
    )
    return total.mean(), {
        "position": position_loss.mean(),
        "adjacency": adjacency_loss.mean(),
        "endpoint": endpoint_loss.mean(),
        "stiffness": stiffness_loss.mean(),
        "total": total.mean(),
    }


def train_supervised_refiner(
    optimized_cases: OptimizedCases,
    model_config: SupervisedRefinerConfig | None = None,
    train_config: SupervisedTrainingConfig | None = None,
    curriculum: CurriculumConfig | None = None,
) -> tuple[SupervisedRefiner, SupervisedTrainingSummary]:
    optimized_cases.validate()
    model_config = model_config or SupervisedRefinerConfig()
    train_config = train_config or SupervisedTrainingConfig(dataset_path="")
    curriculum = curriculum or CurriculumConfig()
    device = resolve_torch_device(train_config.device)

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    model = SupervisedRefiner(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history = {
        "total": [],
        "position": [],
        "adjacency": [],
        "endpoint": [],
        "stiffness": [],
    }
    checkpoint_path = (
        Path(train_config.checkpoint_path)
        if train_config.checkpoint_path is not None
        else Path(train_config.logdir) / "refiner.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=train_config.logdir)
    try:
        step = 0
        while step < train_config.num_steps:
            for minibatch_cases in iter_supervised_minibatches(
                optimized_cases,
                batch_size=train_config.batch_size,
                shuffle=True,
                seed=train_config.seed + step,
            ):
                minibatch_cases = minibatch_cases.to(device)
                difficulty = _difficulty_fraction(step, train_config.num_steps)
                batch = make_supervised_batch(
                    optimized_cases=minibatch_cases,
                    curriculum=curriculum,
                    difficulty=difficulty,
                    seed=train_config.seed + step,
                )
                prediction = model.predict_flow(
                    structures=batch.flow_structures,
                    target_stiffness=batch.target_stiffness,
                    current_stiffness=batch.current_analyses.generalized_stiffness,
                    flow_times=batch.flow_times,
                    position_noise_levels=batch.position_noise_levels,
                    adjacency_noise_levels=batch.adjacency_noise_levels,
                )
                total_loss, loss_terms = _training_losses(
                    prediction, batch, train_config
                )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                for name in history:
                    value = float(loss_terms[name].detach().item())
                    history[name].append(value)
                    writer.add_scalar(f"train/{name}", value, step)
                writer.add_scalar("train/difficulty", difficulty, step)
                writer.add_scalar(
                    "train/flow_time_mean", float(batch.flow_times.mean().item()), step
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
    print(f"final_total_loss={summary.history['total'][-1]:.6f}")
