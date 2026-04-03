from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.common import (
    enforce_role_adjacency_constraints,
    ROLE_FREE,
    apply_free_node_update,
)
from compliant_mechanism_synthesis.data import generate_noise_sample
from compliant_mechanism_synthesis.mechanics import (
    FrameFEMConfig,
    GeometryRegularizationConfig,
    characteristic_scales,
    mechanical_terms,
    mechanical_response_fields,
    refine_connectivity,
    threshold_connectivity,
)
from compliant_mechanism_synthesis.model import GraphRefinementModel
from compliant_mechanism_synthesis.repertoire import SimulationRepertoire
from compliant_mechanism_synthesis.scaling import (
    normalize_generalized_stiffness_matrix,
)
from compliant_mechanism_synthesis.viz import (
    export_canonical_animation,
    export_rollout_animation,
    plot_graph_design,
)


@dataclass
class TrainConfig:
    num_nodes: int = 64
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    node_effect_dim: int = 64
    batch_size: int = 128
    gradient_accumulation_steps: int = 8
    train_steps: int = 20_000
    learning_rate: float = 1e-4
    learning_rate_warmup_steps: int = 100
    learning_rate_min_scale: float = 0.1
    rollout_steps: int = 8
    position_step_size: float = 0.1
    connectivity_step_size: float = 0.1
    rollout_position_noise: float = 0.04
    rollout_connectivity_noise: float = 0.08
    supervised_denoising_weight: float = 0.5
    supervised_position_weight: float = 1.0
    supervised_adjacency_weight: float = 1.0
    supervised_position_noise: float = 0.08
    supervised_connectivity_noise: float = 0.16
    supervised_every_steps: int = 1
    supervised_priority_start: int = 3
    supervised_priority_end: int = 1
    supervised_priority_duration: int = 2_000
    repertoire_bootstrap_cases: int = 512
    repertoire_max_cases: int = 4_096
    canonical_case_count: int = 6
    property_weight: float = 1.0
    stress_weight: float = 0.3
    monotonic_improvement_weight: float = 0.2
    material_weight: float = 0.0
    sparsity_weight: float = 0.0
    connectivity_weight: float = 0.0
    fixed_mobile_connectivity_weight: float = 0.0
    short_beam_weight: float = 0.0
    long_beam_weight: float = 0.0
    thin_diameter_weight: float = 0.0
    thick_diameter_weight: float = 0.0
    node_spacing_weight: float = 0.0
    free_repulsion_weight: float = 0.0
    rigid_attachment_weight: float = 0.0
    centroid_weight: float = 0.0
    spread_weight: float = 0.0
    soft_domain_weight: float = 0.0
    min_beam_length: float = 5e-3
    max_beam_length: float = 4e-2
    min_beam_diameter: float = 2e-4
    max_beam_diameter: float = 3e-3
    min_free_node_spacing: float = 1.2e-2
    display_animation_scale: float = 4.0
    animation_every_steps: int = 500
    log_every_steps: int = 1
    canonical_eval_every_steps: int = 100
    sample_threshold: float = 0.5
    device: str = "auto"
    name: str = "prototype"
    checkpoint_path: str = "artifacts/prototype.pt"
    seed: int = 7


def _device(device_spec: str = "auto") -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress(message: str) -> None:
    print(message, flush=True)


def _resolve_sample_seed(seed_override: int | None) -> int:
    if seed_override is not None:
        return seed_override
    return random.SystemRandom().randrange(0, 2**31)


def _scheduled_learning_rate(
    step: int,
    total_steps: int,
    base_learning_rate: float,
    warmup_steps: int,
    min_scale: float,
) -> float:
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps), 0)
    min_scale = float(min(max(min_scale, 0.0), 1.0))
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * (step / warmup_steps)
    if step >= total_steps:
        return base_learning_rate * min_scale
    anneal_span = max(total_steps - warmup_steps, 1)
    anneal_progress = (step - warmup_steps) / anneal_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * anneal_progress))
    scale = min_scale + (1.0 - min_scale) * cosine
    return base_learning_rate * scale


def _load_train_config(config_dict: dict[str, object]) -> TrainConfig:
    return TrainConfig(**dict(config_dict))


def _timestamped_run_dir(name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-{name}"


def _pure_noise_batch(
    batch_size: int,
    num_nodes: int,
    device: torch.device,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = []
    roles = []
    adjacency = []
    if seed is not None:
        state = random.getstate()
        torch_state = torch.random.get_rng_state()
        random.seed(seed)
        torch.manual_seed(seed)
    for _ in range(batch_size):
        x, r, a = generate_noise_sample(num_nodes)
        positions.append(x)
        roles.append(r)
        adjacency.append(a)
    if seed is not None:
        random.setstate(state)
        torch.random.set_rng_state(torch_state)
    return (
        torch.stack(positions, dim=0).to(device),
        torch.stack(roles, dim=0).to(device),
        torch.stack(adjacency, dim=0).to(device),
    )


def _sample_target_stiffnesses(
    batch_size: int,
    device: torch.device,
    repertoire: SimulationRepertoire,
) -> torch.Tensor:
    return repertoire.sample_target_stiffness(batch_size, device, FrameFEMConfig())


def _bootstrap_repertoire(
    config: TrainConfig,
    device: torch.device,
) -> SimulationRepertoire:
    repertoire = SimulationRepertoire.empty(
        num_nodes=config.num_nodes,
        max_cases=config.repertoire_max_cases,
    )
    remaining = max(config.repertoire_bootstrap_cases, config.batch_size)
    while remaining > 0:
        batch_size = min(config.batch_size, remaining)
        positions, roles, adjacency = _pure_noise_batch(
            batch_size,
            config.num_nodes,
            device,
            seed=config.seed + remaining + 17,
        )
        with torch.no_grad():
            terms = mechanical_terms(positions, roles, adjacency)
        repertoire.add(
            positions,
            roles,
            adjacency,
            terms["stiffness_matrix"],
            source_code=0,
        )
        remaining -= batch_size
    return repertoire


def _observe_cases(
    repertoire: SimulationRepertoire,
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    source_code: int,
) -> None:
    with torch.no_grad():
        terms = mechanical_terms(positions, roles, adjacency)
    finite_mask = torch.isfinite(terms["stiffness_matrix"]).all(dim=(1, 2))
    if not finite_mask.any():
        return
    repertoire.add(
        positions[finite_mask],
        roles[finite_mask],
        adjacency[finite_mask],
        terms["stiffness_matrix"][finite_mask],
        source_code=source_code,
    )


def _scheduled_supervised_priority(
    step: int,
    duration: int,
    start: int,
    end: int,
) -> int:
    if duration <= 1:
        return max(int(end), 1)
    progress = min(max((step - 1) / max(duration - 1, 1), 0.0), 1.0)
    value = start + progress * (end - start)
    return max(int(round(value)), 1)


def _scheduled_training_phase(
    step: int,
    supervised_priority_start: int,
    supervised_priority_end: int,
    supervised_priority_duration: int,
) -> str:
    cycle_start = 1
    while True:
        supervised_span = _scheduled_supervised_priority(
            cycle_start,
            supervised_priority_duration,
            supervised_priority_start,
            supervised_priority_end,
        )
        rl_step = cycle_start + supervised_span
        next_cycle_start = rl_step + 1
        if step < rl_step:
            return "supervised"
        if step == rl_step:
            return "rl"
        cycle_start = next_cycle_start


def _step_weights(steps: int, device: torch.device) -> torch.Tensor:
    weights = torch.linspace(1.0, float(steps), steps=steps, device=device)
    return weights / weights.sum()


def _visualization_threshold() -> float:
    return 0.08


def _format_matrix(matrix: torch.Tensor | list[list[float]]) -> str:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().tolist()
    rows = ["[" + ",".join(f"{value:.4e}" for value in row) + "]" for row in matrix]
    return "[" + ";".join(rows) + "]"


def _parse_target_stiffness(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 9:
        raise ValueError(
            "target stiffness must contain exactly 9 comma-separated values"
        )
    return values


def _mechanics_condition_matrices(
    target_stiffness: torch.Tensor,
    current_stiffness: torch.Tensor,
    frame_config: FrameFEMConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = characteristic_scales(frame_config)
    residual_stiffness = target_stiffness - current_stiffness
    return (
        normalize_generalized_stiffness_matrix(target_stiffness, scales),
        normalize_generalized_stiffness_matrix(current_stiffness, scales),
        normalize_generalized_stiffness_matrix(residual_stiffness, scales),
    )


def _nodal_mechanics_features(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        response_fields = mechanical_response_fields(positions, roles, adjacency)
    return _nodal_mechanics_features_from_fields(positions, response_fields)


def _nodal_mechanics_features_from_fields(
    positions: torch.Tensor,
    response_fields: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch_size = positions.shape[0]
    deformation_features = (
        response_fields["normalized_translations"]
        .permute(0, 2, 1, 3)
        .reshape(batch_size, positions.shape[1], 6)
    )
    stress_feature = response_fields["normalized_nodal_stress"].unsqueeze(-1)
    return torch.cat([deformation_features, stress_feature], dim=-1)


def _sample_supervised_denoising_batch(
    teacher_positions: torch.Tensor,
    roles: torch.Tensor,
    teacher_adjacency: torch.Tensor,
    position_noise_scale: float,
    connectivity_noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _inject_rollout_noise(
        teacher_positions,
        roles,
        teacher_adjacency,
        torch.ones((teacher_positions.shape[0],), device=teacher_positions.device),
        position_noise_scale=position_noise_scale,
        connectivity_noise_scale=connectivity_noise_scale,
    )


def _supervised_training_step(
    model: GraphRefinementModel,
    teacher_positions: torch.Tensor,
    teacher_roles: torch.Tensor,
    teacher_adjacency: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    geometry_config: GeometryRegularizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_terms = mechanical_terms(
            teacher_positions,
            teacher_roles,
            teacher_adjacency,
        )
        teacher_stiffness = teacher_terms["stiffness_matrix"]
    noisy_positions, noisy_adjacency = _sample_supervised_denoising_batch(
        teacher_positions,
        teacher_roles,
        teacher_adjacency,
        position_noise_scale=config.supervised_position_noise,
        connectivity_noise_scale=config.supervised_connectivity_noise,
    )
    with torch.no_grad():
        noisy_terms = mechanical_terms(noisy_positions, teacher_roles, noisy_adjacency)
    rollout = rollout_refinement(
        model,
        noisy_positions,
        teacher_roles,
        noisy_adjacency,
        teacher_stiffness,
        steps=config.rollout_steps,
        position_step_size=config.position_step_size,
        connectivity_step_size=config.connectivity_step_size,
        base_time=torch.ones((config.batch_size,), device=device),
        position_noise_scale=config.rollout_position_noise,
        connectivity_noise_scale=config.rollout_connectivity_noise,
        geometry_config=geometry_config,
        initial_stiffness=noisy_terms["stiffness_matrix"],
    )
    final_state = rollout[-1]
    position_loss, adjacency_loss = _supervised_reconstruction_losses(
        final_state["refined_positions"],
        teacher_positions,
        teacher_roles,
        final_state["refined_adjacency"],
        teacher_adjacency,
    )
    total = config.supervised_denoising_weight * (
        config.supervised_position_weight * position_loss
        + config.supervised_adjacency_weight * adjacency_loss
    )
    return total, position_loss, adjacency_loss


def _supervised_reconstruction_losses(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    roles: torch.Tensor,
    predicted_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    free_mask = (roles == ROLE_FREE).to(dtype=predicted_positions.dtype)
    position_sq_error = (predicted_positions - target_positions).square().sum(dim=-1)
    position_loss = (position_sq_error * free_mask).sum() / free_mask.sum().clamp_min(
        1.0
    )
    adjacency_loss = F.mse_loss(predicted_adjacency, target_adjacency)
    return position_loss, adjacency_loss


def _matrix_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    frame_config: FrameFEMConfig | None = None,
) -> torch.Tensor:
    scales = characteristic_scales(frame_config)
    normalized_predicted = normalize_generalized_stiffness_matrix(predicted, scales)
    normalized_target = normalize_generalized_stiffness_matrix(target, scales)
    return F.mse_loss(normalized_predicted, normalized_target)


def _stiffness_to_response(stiffness_matrix: torch.Tensor) -> torch.Tensor:
    stabilized = 0.5 * (stiffness_matrix + stiffness_matrix.transpose(1, 2))
    eye = torch.eye(3, device=stabilized.device, dtype=stabilized.dtype)
    return torch.linalg.solve(stabilized, eye.unsqueeze(0).expand_as(stabilized))


def _inject_rollout_noise(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    time_fraction: torch.Tensor,
    position_noise_scale: float,
    connectivity_noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_noise_scale <= 0.0 and connectivity_noise_scale <= 0.0:
        return positions, adjacency

    free_mask = (roles == ROLE_FREE).unsqueeze(-1).to(dtype=positions.dtype)
    if position_noise_scale > 0.0:
        position_noise = torch.randn_like(positions) * (
            position_noise_scale * time_fraction[:, None, None]
        )
        positions = positions + free_mask * position_noise

    if connectivity_noise_scale > 0.0:
        adjacency_noise = torch.randn_like(adjacency) * (
            connectivity_noise_scale * time_fraction[:, None, None]
        )
        adjacency = enforce_role_adjacency_constraints(
            (adjacency + adjacency_noise).clamp_min(0.0),
            roles,
        )
    return positions, adjacency


def _monotonic_improvement_loss(step_errors: list[torch.Tensor]) -> torch.Tensor:
    if len(step_errors) < 2:
        return torch.zeros((), device=step_errors[0].device)
    penalties = [
        (current - previous).clamp_min(0.0)
        for previous, current in zip(step_errors[:-1], step_errors[1:])
    ]
    return torch.stack(penalties, dim=0).mean()


def _geometry_regularization_config(
    config: TrainConfig,
) -> GeometryRegularizationConfig:
    return GeometryRegularizationConfig(
        min_length=config.min_beam_length,
        max_length=config.max_beam_length,
        min_diameter=config.min_beam_diameter,
        max_diameter=config.max_beam_diameter,
        min_free_node_spacing=config.min_free_node_spacing,
    )


def _log_matrix(
    writer: SummaryWriter,
    prefix: str,
    matrix: torch.Tensor,
    step: int,
) -> None:
    names = [
        ["ux_ux", "ux_uy", "ux_theta"],
        ["uy_ux", "uy_uy", "uy_theta"],
        ["theta_ux", "theta_uy", "theta_theta"],
    ]
    for row_idx in range(3):
        for col_idx in range(3):
            writer.add_scalar(
                f"{prefix}/{names[row_idx][col_idx]}",
                matrix[row_idx, col_idx].item(),
                step,
            )


def rollout_refinement(
    model: GraphRefinementModel,
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    target_stiffness: torch.Tensor,
    steps: int,
    position_step_size: float,
    connectivity_step_size: float,
    base_time: torch.Tensor,
    position_noise_scale: float = 0.0,
    connectivity_noise_scale: float = 0.0,
    geometry_config: GeometryRegularizationConfig | None = None,
    initial_stiffness: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor]]:
    current_positions = positions
    current_adjacency = adjacency
    states: list[dict[str, torch.Tensor]] = []
    current_stiffness = initial_stiffness

    for step_idx in range(steps):
        time_fraction = 1.0 - step_idx / max(steps - 1, 1)
        timestep = base_time * time_fraction
        position_noise_levels = position_noise_scale * timestep
        connectivity_noise_levels = connectivity_noise_scale * timestep
        if current_stiffness is None:
            with torch.no_grad():
                current_response_fields = mechanical_response_fields(
                    current_positions,
                    roles,
                    current_adjacency,
                )
            current_stiffness = current_response_fields["stiffness_matrix"]
            current_nodal_mechanics = _nodal_mechanics_features_from_fields(
                current_positions,
                current_response_fields,
            )
        else:
            current_nodal_mechanics = _nodal_mechanics_features(
                current_positions,
                roles,
                current_adjacency,
            )
        target_features, current_features, residual_features = (
            _mechanics_condition_matrices(
                target_stiffness,
                current_stiffness,
            )
        )
        outputs = model(
            current_positions,
            roles,
            current_adjacency,
            target_features,
            current_features,
            residual_features,
            current_nodal_mechanics,
            timestep,
            position_noise_levels,
            connectivity_noise_levels,
        )
        outputs["displacements"] = torch.nan_to_num(
            outputs["displacements"], nan=0.0, posinf=0.0, neginf=0.0
        )
        outputs["delta_scores"] = torch.nan_to_num(
            outputs["delta_scores"], nan=0.0, posinf=0.0, neginf=0.0
        )
        outputs["node_latents"] = torch.nan_to_num(
            outputs["node_latents"], nan=0.0, posinf=0.0, neginf=0.0
        )
        refined_positions = apply_free_node_update(
            current_positions,
            outputs["displacements"],
            roles,
            position_step_size,
        )
        refined_adjacency = refine_connectivity(
            current_adjacency,
            refined_positions,
            roles,
            outputs["delta_scores"],
            connectivity_step_size,
        )
        current_positions, current_adjacency = _inject_rollout_noise(
            refined_positions,
            roles,
            refined_adjacency,
            base_time * time_fraction,
            position_noise_scale,
            connectivity_noise_scale,
        )
        state = {
            "positions": current_positions,
            "roles": roles,
            "adjacency": current_adjacency,
            "refined_positions": refined_positions,
            "refined_adjacency": refined_adjacency,
            "displacements": outputs["displacements"],
            "node_latents": outputs["node_latents"],
            "delta_scores": outputs["delta_scores"],
        }
        if geometry_config is not None:
            step_terms = mechanical_terms(
                current_positions,
                roles,
                current_adjacency,
                geometry_config=geometry_config,
            )
            finite_stiffness = torch.isfinite(step_terms["stiffness_matrix"]).all(
                dim=(1, 2)
            )
            if not finite_stiffness.all():
                fallback_stiffness = (
                    current_stiffness
                    if current_stiffness is not None
                    else target_stiffness.detach()
                )
                sanitized_stiffness = step_terms["stiffness_matrix"].detach().clone()
                sanitized_stiffness[~finite_stiffness] = fallback_stiffness[
                    ~finite_stiffness
                ]
                step_terms["stiffness_matrix"] = sanitized_stiffness
            state["terms"] = step_terms
            current_stiffness = step_terms["stiffness_matrix"].detach()
        else:
            current_stiffness = None
        states.append(state)
    return states


def _log_canonical_evaluation(
    writer: SummaryWriter,
    model: GraphRefinementModel,
    config: TrainConfig,
    step: int,
    device: torch.device,
    repertoire: SimulationRepertoire,
    animation_output_dir: Path | None = None,
) -> None:
    started_at = time.perf_counter()
    canonical_specs = repertoire.canonical_specs(
        device,
        FrameFEMConfig(),
        max_specs=config.canonical_case_count,
    )
    if not canonical_specs:
        return
    _progress(
        f"train:canonical start step={step} cases={len(canonical_specs)} animation={animation_output_dir is not None}"
    )
    geometry_config = _geometry_regularization_config(config)
    stage_started_at = time.perf_counter()
    positions, roles, adjacency = _pure_noise_batch(
        len(canonical_specs),
        config.num_nodes,
        device,
        seed=config.seed + 101,
    )
    raw_targets = torch.stack([values for _, values in canonical_specs], dim=0).to(
        device
    )
    target_responses = _stiffness_to_response(raw_targets)
    base_time = torch.ones((len(canonical_specs),), device=device)
    _progress(
        f"train:canonical prepare step={step} dt={time.perf_counter() - stage_started_at:.2f}s"
    )
    stage_started_at = time.perf_counter()
    rollout = rollout_refinement(
        model,
        positions,
        roles,
        adjacency,
        raw_targets,
        steps=config.rollout_steps,
        position_step_size=config.position_step_size,
        connectivity_step_size=config.connectivity_step_size,
        base_time=base_time,
        position_noise_scale=0.0,
        connectivity_noise_scale=0.0,
        geometry_config=geometry_config,
    )
    _progress(
        f"train:canonical rollout step={step} dt={time.perf_counter() - stage_started_at:.2f}s"
    )
    stage_started_at = time.perf_counter()
    final_state = rollout[-1]
    final_terms = mechanical_terms(
        final_state["positions"],
        roles,
        final_state["adjacency"],
        geometry_config=geometry_config,
    )
    _progress(
        f"train:canonical final_terms step={step} dt={time.perf_counter() - stage_started_at:.2f}s"
    )

    for idx, (name, _) in enumerate(canonical_specs):
        case_started_at = time.perf_counter()
        figure = plot_graph_design(
            final_state["positions"][idx],
            roles[idx],
            final_state["adjacency"][idx],
            threshold=_visualization_threshold(),
            title=name,
        )
        writer.add_figure(f"canonical/00_designs/{name}", figure, global_step=step)
        _log_matrix(
            writer,
            f"canonical/10_achieved_stiffness/{name}",
            final_terms["stiffness_matrix"][idx],
            step,
        )
        _log_matrix(
            writer,
            f"canonical/20_achieved_response/{name}",
            final_terms["response_matrix"][idx],
            step,
        )
        for rollout_idx, state in enumerate(rollout, start=1):
            step_error = _matrix_loss(
                state["terms"]["stiffness_matrix"][idx : idx + 1],
                raw_targets[idx : idx + 1],
            )
            writer.add_scalar(
                f"canonical/30_property_error/{name}/step_{rollout_idx}",
                step_error.item(),
                step,
            )
        plt = figure
        plt.clf()
        _progress(
            f"train:canonical case step={step} idx={idx} name={name} dt={time.perf_counter() - case_started_at:.2f}s"
        )

    if animation_output_dir is not None and canonical_specs:
        animation_started_at = time.perf_counter()
        animation_name = "canonical_grid"
        _progress(f"train:animation start step={step} name={animation_name}")
        animation_rollout = []
        for state in rollout:
            animation_rollout.append(
                {
                    key: value
                    for key, value in state.items()
                    if isinstance(value, torch.Tensor)
                }
            )
        animation_path = export_canonical_animation(
            animation_output_dir / f"step_{step:05d}_{animation_name}.gif",
            positions,
            roles,
            adjacency,
            animation_rollout,
            target_responses,
            [name for name, _ in canonical_specs],
            display_scale=config.display_animation_scale,
            threshold=_visualization_threshold(),
            frame_config=FrameFEMConfig(),
            final_positions=final_state["positions"],
            final_adjacency=final_state["adjacency"],
        )
        _progress(
            f"train:animation done step={step} dt={time.perf_counter() - animation_started_at:.2f}s path={animation_path}"
        )
    _progress(
        f"train:canonical done step={step} dt={time.perf_counter() - started_at:.2f}s"
    )


def train(config: TrainConfig) -> tuple[Path, Path]:
    _seed_everything(config.seed)
    device = _device(config.device)
    geometry_config = _geometry_regularization_config(config)
    train_steps = max(config.train_steps, 1)
    _progress(
        f"train:start device={device} train_steps={train_steps} batch_size={config.batch_size} num_nodes={config.num_nodes}"
    )

    repertoire = _bootstrap_repertoire(config, device)
    model = GraphRefinementModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        node_effect_dim=config.node_effect_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(log_dir=str(_timestamped_run_dir(config.name)))
    log_dir = Path(writer.log_dir)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0
    running_totals = {
        "total": 0.0,
        "property": 0.0,
        "stress": 0.0,
        "monotonic": 0.0,
        "supervised": 0.0,
        "supervised_position": 0.0,
        "supervised_adjacency": 0.0,
        "max_von_mises_stress": 0.0,
        "max_stress_ratio": 0.0,
        "mean_free_position_update": 0.0,
        "max_free_position_update": 0.0,
        "mean_connectivity_update": 0.0,
    }
    running_counts = {
        "total": 0,
        "property": 0,
        "stress": 0,
        "monotonic": 0,
        "supervised": 0,
        "supervised_position": 0,
        "supervised_adjacency": 0,
        "max_von_mises_stress": 0,
        "max_stress_ratio": 0,
        "mean_free_position_update": 0,
        "max_free_position_update": 0,
        "mean_connectivity_update": 0,
    }
    step_weights = _step_weights(config.rollout_steps, device)
    gradient_accumulation_steps = max(config.gradient_accumulation_steps, 1)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, train_steps + 1):
        scheduled_lr = _scheduled_learning_rate(
            step=step,
            total_steps=train_steps,
            base_learning_rate=config.learning_rate,
            warmup_steps=config.learning_rate_warmup_steps,
            min_scale=config.learning_rate_min_scale,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduled_lr
        property_loss = torch.zeros((), device=device)
        supervised_position_loss = torch.zeros((), device=device)
        supervised_adjacency_loss = torch.zeros((), device=device)
        supervised_loss = torch.zeros((), device=device)
        stress_loss = torch.zeros((), device=device)
        monotonic_loss = torch.zeros((), device=device)
        phase = _scheduled_training_phase(
            step,
            config.supervised_priority_start,
            config.supervised_priority_end,
            config.supervised_priority_duration,
        )

        current_seed = config.seed + step + 1000
        fresh_positions, fresh_roles, fresh_adjacency = _pure_noise_batch(
            config.batch_size,
            config.num_nodes,
            device,
            seed=current_seed,
        )
        _observe_cases(
            repertoire,
            fresh_positions,
            fresh_roles,
            fresh_adjacency,
            source_code=0,
        )

        if phase == "supervised":
            supervised_loss, supervised_position_loss, supervised_adjacency_loss = (
                _supervised_training_step(
                    model,
                    fresh_positions,
                    fresh_roles,
                    fresh_adjacency,
                    config,
                    device,
                    geometry_config,
                )
            )
            total = supervised_loss
        else:
            positions, roles, adjacency = fresh_positions, fresh_roles, fresh_adjacency
            goal_targets = _sample_target_stiffnesses(
                config.batch_size,
                device,
                repertoire,
            )
            raw_targets = goal_targets
            base_time = torch.rand((positions.shape[0],), device=device)
            rollout = rollout_refinement(
                model,
                positions,
                roles,
                adjacency,
                raw_targets,
                steps=config.rollout_steps,
                position_step_size=config.position_step_size,
                connectivity_step_size=config.connectivity_step_size,
                base_time=base_time,
                position_noise_scale=config.rollout_position_noise,
                connectivity_noise_scale=config.rollout_connectivity_noise,
                geometry_config=geometry_config,
                initial_stiffness=None,
            )
            material_loss = torch.zeros((), device=device)
            sparsity_loss = torch.zeros((), device=device)
            connectivity_loss = torch.zeros((), device=device)
            fixed_mobile_connectivity_loss = torch.zeros((), device=device)
            short_beam_loss = torch.zeros((), device=device)
            long_beam_loss = torch.zeros((), device=device)
            thin_diameter_loss = torch.zeros((), device=device)
            thick_diameter_loss = torch.zeros((), device=device)
            node_spacing_loss = torch.zeros((), device=device)
            free_repulsion_loss = torch.zeros((), device=device)
            rigid_attachment_loss = torch.zeros((), device=device)
            centroid_loss = torch.zeros((), device=device)
            spread_loss = torch.zeros((), device=device)
            soft_domain_loss = torch.zeros((), device=device)
            step_objectives: list[torch.Tensor] = []
            for step_idx, state in enumerate(rollout):
                step_terms = state["terms"]
                step_error = _matrix_loss(
                    step_terms["stiffness_matrix"],
                    raw_targets,
                )
                property_loss = property_loss + step_weights[step_idx] * step_error
                step_stress_loss = step_terms["stress_loss"].mean()
                stress_loss = stress_loss + step_weights[step_idx] * step_stress_loss
                step_material = step_terms["normalized_material"].mean()
                material_loss = (
                    material_loss + step_weights[step_idx] * step_material
                )
                step_sparsity = step_terms["sparsity"].mean()
                sparsity_loss = (
                    sparsity_loss + step_weights[step_idx] * step_sparsity
                )
                step_connectivity = step_terms["connectivity_penalty"].mean()
                connectivity_loss = (
                    connectivity_loss + step_weights[step_idx] * step_connectivity
                )
                step_fixed_mobile_connectivity = step_terms[
                    "fixed_mobile_connectivity_penalty"
                ].mean()
                fixed_mobile_connectivity_loss = (
                    fixed_mobile_connectivity_loss
                    + step_weights[step_idx] * step_fixed_mobile_connectivity
                )
                step_short_beam = step_terms["short_beam_penalty"].mean()
                short_beam_loss = (
                    short_beam_loss + step_weights[step_idx] * step_short_beam
                )
                step_long_beam = step_terms["long_beam_penalty"].mean()
                long_beam_loss = (
                    long_beam_loss + step_weights[step_idx] * step_long_beam
                )
                step_node_spacing = step_terms["node_spacing_penalty"].mean()
                node_spacing_loss = (
                    node_spacing_loss + step_weights[step_idx] * step_node_spacing
                )
                step_free_repulsion = step_terms["free_repulsion_penalty"].mean()
                free_repulsion_loss = (
                    free_repulsion_loss
                    + step_weights[step_idx] * step_free_repulsion
                )
                step_rigid_attachment = step_terms["rigid_attachment_penalty"].mean()
                rigid_attachment_loss = (
                    rigid_attachment_loss
                    + step_weights[step_idx] * step_rigid_attachment
                )
                step_centroid = step_terms["centroid_penalty"].mean()
                centroid_loss = (
                    centroid_loss + step_weights[step_idx] * step_centroid
                )
                step_spread = step_terms["spread_penalty"].mean()
                spread_loss = spread_loss + step_weights[step_idx] * step_spread
                step_soft_domain = step_terms["soft_domain_penalty"].mean()
                soft_domain_loss = (
                    soft_domain_loss + step_weights[step_idx] * step_soft_domain
                )
                step_objectives.append(
                    config.property_weight * step_error
                    + config.stress_weight * step_stress_loss
                    + config.material_weight * step_material
                    + config.sparsity_weight * step_sparsity
                    + config.connectivity_weight * step_connectivity
                    + config.fixed_mobile_connectivity_weight
                    * step_fixed_mobile_connectivity
                    + config.short_beam_weight * step_short_beam
                    + config.long_beam_weight * step_long_beam
                    + config.node_spacing_weight * step_node_spacing
                    + config.free_repulsion_weight * step_free_repulsion
                    + config.rigid_attachment_weight * step_rigid_attachment
                    + config.centroid_weight * step_centroid
                    + config.spread_weight * step_spread
                    + config.soft_domain_weight * step_soft_domain
                )
            monotonic_loss = _monotonic_improvement_loss(step_objectives)
            final_step_terms = rollout[-1]["terms"]
            free_mask = roles == ROLE_FREE
            free_update_means = []
            free_update_maxes = []
            connectivity_updates = []
            edge_i, edge_j = torch.triu_indices(
                roles.shape[1],
                roles.shape[1],
                offset=1,
                device=device,
            )
            for state in rollout:
                free_displacement_norm = torch.linalg.vector_norm(
                    state["displacements"], dim=-1
                )
                masked_free_displacement = free_displacement_norm * free_mask.to(
                    dtype=free_displacement_norm.dtype
                )
                free_counts = free_mask.to(dtype=free_displacement_norm.dtype).sum(
                    dim=1
                ).clamp_min(1.0)
                free_update_means.append(
                    masked_free_displacement.sum(dim=1) / free_counts
                )
                free_update_maxes.append(masked_free_displacement.max(dim=1).values)
                connectivity_updates.append(
                    state["delta_scores"].abs()[:, edge_i, edge_j].mean(dim=1)
                )
            mean_free_position_update = torch.stack(free_update_means, dim=0).mean()
            max_free_position_update = torch.stack(free_update_maxes, dim=0).max()
            mean_connectivity_update = torch.stack(connectivity_updates, dim=0).mean()
            thin_diameter_loss = final_step_terms["thin_diameter_penalty"].mean()
            thick_diameter_loss = final_step_terms["thick_diameter_penalty"].mean()
            total = (
                config.property_weight * property_loss
                + config.stress_weight * stress_loss
                + config.monotonic_improvement_weight * monotonic_loss
                + config.material_weight * material_loss
                + config.sparsity_weight * sparsity_loss
                + config.connectivity_weight * connectivity_loss
                + config.fixed_mobile_connectivity_weight
                * fixed_mobile_connectivity_loss
                + config.short_beam_weight * short_beam_loss
                + config.long_beam_weight * long_beam_loss
                + config.thin_diameter_weight * thin_diameter_loss
                + config.thick_diameter_weight * thick_diameter_loss
                + config.node_spacing_weight * node_spacing_loss
                + config.free_repulsion_weight * free_repulsion_loss
                + config.rigid_attachment_weight * rigid_attachment_loss
                + config.centroid_weight * centroid_loss
                + config.spread_weight * spread_loss
                + config.soft_domain_weight * soft_domain_loss
            )
            final_state = rollout[-1]
            _observe_cases(
                repertoire,
                final_state["refined_positions"],
                roles,
                final_state["refined_adjacency"],
                source_code=1,
            )
        if not torch.isfinite(total):
            _progress(
                f"train:skip_nonfinite step={step} phase={phase} total={total.item()}"
            )
            continue
        (total / gradient_accumulation_steps).backward()
        if step % gradient_accumulation_steps == 0 or step == train_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_totals["total"] += total.item()
        running_counts["total"] += 1
        if phase == "supervised":
            running_totals["supervised"] += supervised_loss.item()
            running_totals["supervised_position"] += supervised_position_loss.item()
            running_totals["supervised_adjacency"] += supervised_adjacency_loss.item()
            running_counts["supervised"] += 1
            running_counts["supervised_position"] += 1
            running_counts["supervised_adjacency"] += 1
        else:
            running_totals["property"] += property_loss.item()
            running_totals["stress"] += stress_loss.item()
            running_totals["monotonic"] += monotonic_loss.item()
            running_totals["max_von_mises_stress"] += final_step_terms[
                "max_von_mises_stress"
            ].mean().item()
            running_totals["max_stress_ratio"] += final_step_terms[
                "max_stress_ratio"
            ].mean().item()
            running_totals["mean_free_position_update"] += (
                mean_free_position_update.item()
            )
            running_totals["max_free_position_update"] += (
                max_free_position_update.item()
            )
            running_totals["mean_connectivity_update"] += (
                mean_connectivity_update.item()
            )
            running_counts["property"] += 1
            running_counts["stress"] += 1
            running_counts["monotonic"] += 1
            running_counts["max_von_mises_stress"] += 1
            running_counts["max_stress_ratio"] += 1
            running_counts["mean_free_position_update"] += 1
            running_counts["max_free_position_update"] += 1
            running_counts["mean_connectivity_update"] += 1

        if config.log_every_steps > 0 and (
            step % config.log_every_steps == 0 or step == train_steps
        ):
            total_mean = running_totals["total"] / max(running_counts["total"], 1)
            writer.add_scalar("train/total_loss", total_mean, global_step)
            writer.add_scalar("train/learning_rate", scheduled_lr, global_step)
            writer.add_text("train/phase", phase, global_step)
            writer.add_scalar("train/repertoire_size", len(repertoire), global_step)
            if running_counts["supervised"] > 0:
                writer.add_scalar(
                    "train/supervised_loss",
                    running_totals["supervised"] / running_counts["supervised"],
                    global_step,
                )
                writer.add_scalar(
                    "train/supervised_position_loss",
                    running_totals["supervised_position"]
                    / running_counts["supervised_position"],
                    global_step,
                )
                writer.add_scalar(
                    "train/supervised_adjacency_loss",
                    running_totals["supervised_adjacency"]
                    / running_counts["supervised_adjacency"],
                    global_step,
                )
            if running_counts["property"] > 0:
                writer.add_scalar(
                    "train/property_loss",
                    running_totals["property"] / running_counts["property"],
                    global_step,
                )
                writer.add_scalar(
                    "train/monotonic_improvement_loss",
                    running_totals["monotonic"] / running_counts["monotonic"],
                    global_step,
                )
                writer.add_scalar(
                    "train/stress_loss",
                    running_totals["stress"] / running_counts["stress"],
                    global_step,
                )
                writer.add_scalar(
                    "metrics/max_von_mises_stress",
                    running_totals["max_von_mises_stress"]
                    / running_counts["max_von_mises_stress"],
                    global_step,
                )
                writer.add_scalar(
                    "metrics/max_stress_ratio",
                    running_totals["max_stress_ratio"]
                    / running_counts["max_stress_ratio"],
                    global_step,
                )
                writer.add_scalar(
                    "metrics/mean_free_position_update",
                    running_totals["mean_free_position_update"]
                    / running_counts["mean_free_position_update"],
                    global_step,
                )
                writer.add_scalar(
                    "metrics/max_free_position_update",
                    running_totals["max_free_position_update"]
                    / running_counts["max_free_position_update"],
                    global_step,
                )
                writer.add_scalar(
                    "metrics/mean_connectivity_update",
                    running_totals["mean_connectivity_update"]
                    / running_counts["mean_connectivity_update"],
                    global_step,
                )
        global_step += 1

        if config.log_every_steps > 0 and (
            step % config.log_every_steps == 0 or step == train_steps
        ):
            window = (
                config.log_every_steps
                if step % config.log_every_steps == 0
                else step % config.log_every_steps
            )
            if window == 0:
                window = config.log_every_steps
            _progress(
                f"train:step {step}/{train_steps} phase={phase} total={total.item():.4f} prop={property_loss.item():.4f} stress={stress_loss.item():.4f} sup={supervised_loss.item():.4f} rep={len(repertoire)}"
            )
            for key in running_totals:
                running_totals[key] = 0.0
                running_counts[key] = 0

        if (
            config.canonical_eval_every_steps > 0
            and global_step % config.canonical_eval_every_steps == 0
        ):
            model.eval()
            with torch.no_grad():
                _log_canonical_evaluation(
                    writer,
                    model,
                    config,
                    global_step,
                    device,
                    repertoire,
                    (
                        log_dir / "animations"
                        if config.animation_every_steps > 0
                        and global_step % config.animation_every_steps == 0
                        else None
                    ),
                )
            model.train()

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "repertoire": repertoire.payload(),
    }
    torch.save(payload, checkpoint_path)
    torch.save(repertoire.payload(), log_dir / "repertoire.pt")
    writer.close()
    _progress(f"train:done checkpoint={checkpoint_path} log_dir={log_dir}")
    return checkpoint_path, log_dir


def refine_sample_state(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    raw_target_stiffness: torch.Tensor,
    config: TrainConfig,
    steps: int = 12,
    lr: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry_config = _geometry_regularization_config(config)
    free_mask = (roles == ROLE_FREE).unsqueeze(-1).to(dtype=positions.dtype)
    position_param = torch.nn.Parameter(positions.clone())
    adjacency_param = torch.nn.Parameter(adjacency.clone())
    optimizer = torch.optim.Adam([position_param, adjacency_param], lr=lr)

    for _ in range(steps):
        current_positions = positions + free_mask * (position_param - positions)
        current_adjacency = enforce_role_adjacency_constraints(
            adjacency_param.clamp_min(0.0),
            roles,
        )
        terms = mechanical_terms(
            current_positions,
            roles,
            current_adjacency,
            geometry_config=geometry_config,
        )
        property_loss = _matrix_loss(
            terms["stiffness_matrix"],
            raw_target_stiffness,
        )
        loss = (
            config.property_weight * property_loss
            + config.stress_weight * terms["stress_loss"].mean()
            + config.material_weight * terms["normalized_material"].mean()
            + config.sparsity_weight * terms["sparsity"].mean()
            + config.connectivity_weight * terms["connectivity_penalty"].mean()
            + config.fixed_mobile_connectivity_weight
            * terms["fixed_mobile_connectivity_penalty"].mean()
            + config.short_beam_weight * terms["short_beam_penalty"].mean()
            + config.long_beam_weight * terms["long_beam_penalty"].mean()
            + config.node_spacing_weight * terms["node_spacing_penalty"].mean()
            + config.free_repulsion_weight * terms["free_repulsion_penalty"].mean()
            + config.rigid_attachment_weight
            * terms["rigid_attachment_penalty"].mean()
            + config.centroid_weight * terms["centroid_penalty"].mean()
            + config.spread_weight * terms["spread_penalty"].mean()
            + config.soft_domain_weight * terms["soft_domain_penalty"].mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    refined_positions = (positions + free_mask * (position_param - positions)).detach()
    refined_adjacency = enforce_role_adjacency_constraints(
        adjacency_param.detach().clamp_min(0.0),
        roles,
    )
    return refined_positions, refined_adjacency


def sample(
    checkpoint_path: str,
    target_stiffness: list[float],
    name: str,
    output_path: str,
    steps: int,
    sample_threshold: float,
    device_override: str | None = None,
    seed_override: int | None = None,
) -> dict[str, object]:
    device = _device(device_override or "auto")
    payload = torch.load(checkpoint_path, map_location=device)
    config = _load_train_config(payload["config"])
    if device_override is not None:
        config.device = device_override
    geometry_config = _geometry_regularization_config(config)
    sample_seed = _resolve_sample_seed(seed_override)
    _seed_everything(sample_seed)

    model = GraphRefinementModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        node_effect_dim=config.node_effect_dim,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    positions, roles, adjacency = _pure_noise_batch(
        1,
        config.num_nodes,
        device,
        seed=sample_seed + 500,
    )
    raw_targets = torch.tensor(
        target_stiffness, dtype=torch.float32, device=device
    ).reshape(1, 3, 3)
    raw_targets = 0.5 * (raw_targets + raw_targets.transpose(1, 2))
    target_responses = _stiffness_to_response(raw_targets)
    base_time = torch.ones((1,), device=device)
    with torch.no_grad():
        rollout = rollout_refinement(
            model,
            positions,
            roles,
            adjacency,
            raw_targets,
            steps=steps,
            position_step_size=config.position_step_size,
            connectivity_step_size=config.connectivity_step_size,
            base_time=base_time,
            position_noise_scale=0.0,
            connectivity_noise_scale=0.0,
            geometry_config=geometry_config,
        )
    final_state = rollout[-1]
    refined_positions, refined_adjacency = refine_sample_state(
        final_state["positions"],
        roles,
        final_state["adjacency"],
        raw_targets,
        config,
    )
    terms = mechanical_terms(
        refined_positions,
        roles,
        refined_adjacency,
        geometry_config=geometry_config,
    )
    thresholded_adjacency = threshold_connectivity(
        refined_adjacency, roles, threshold=sample_threshold
    )

    log_dir = _timestamped_run_dir(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    figure = plot_graph_design(
        refined_positions[0],
        roles[0],
        refined_adjacency[0],
        threshold=_visualization_threshold(),
        title=name,
    )
    writer.add_figure("sample/00_design/final_graph", figure, global_step=0)
    _log_matrix(writer, "sample/10_target_stiffness", raw_targets[0], 0)
    _log_matrix(writer, "sample/20_achieved_stiffness", terms["stiffness_matrix"][0], 0)
    _log_matrix(writer, "sample/30_achieved_response", terms["response_matrix"][0], 0)
    writer.add_scalar("sample/40_sparsity_loss", terms["sparsity"][0].item(), 0)
    writer.add_scalar(
        "sample/40_short_beam_penalty", terms["short_beam_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_long_beam_penalty", terms["long_beam_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_thin_diameter_penalty", terms["thin_diameter_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_thick_diameter_penalty", terms["thick_diameter_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_node_spacing_penalty", terms["node_spacing_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_free_repulsion_penalty",
        terms["free_repulsion_penalty"][0].item(),
        0,
    )
    writer.add_scalar(
        "sample/40_rigid_attachment_penalty",
        terms["rigid_attachment_penalty"][0].item(),
        0,
    )
    writer.add_scalar(
        "sample/40_centroid_penalty", terms["centroid_penalty"][0].item(), 0
    )
    writer.add_scalar("sample/40_spread_penalty", terms["spread_penalty"][0].item(), 0)
    writer.add_scalar(
        "sample/40_soft_domain_penalty", terms["soft_domain_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_stress_loss",
        terms["stress_loss"][0].item(),
        0,
    )
    writer.add_scalar(
        "metrics/max_von_mises_stress",
        terms["max_von_mises_stress"][0].item(),
        0,
    )
    writer.add_scalar(
        "metrics/max_stress_ratio",
        terms["max_stress_ratio"][0].item(),
        0,
    )
    animation_rollout = []
    for state in rollout:
        animation_rollout.append(
            {
                key: value[0]
                for key, value in state.items()
                if isinstance(value, torch.Tensor)
            }
        )
    animation_path = export_rollout_animation(
        log_dir / "sample_rollout.gif",
        positions[0],
        roles[0],
        adjacency[0],
        animation_rollout,
        target_responses[0],
        display_scale=config.display_animation_scale,
        threshold=_visualization_threshold(),
        frame_config=FrameFEMConfig(),
        title=name,
        final_positions=refined_positions[0],
        final_adjacency=refined_adjacency[0],
    )
    writer.close()

    result = {
        "animation_path": str(animation_path),
        "seed": sample_seed,
        "positions": refined_positions.cpu(),
        "roles": roles.cpu(),
        "adjacency": refined_adjacency.cpu(),
        "thresholded_adjacency": thresholded_adjacency.cpu(),
        "response_matrix": terms["response_matrix"].cpu(),
        "stiffness_matrix": terms["stiffness_matrix"].cpu(),
        "sparsity": terms["sparsity"].cpu(),
        "fixed_mobile_connectivity_penalty": terms[
            "fixed_mobile_connectivity_penalty"
        ].cpu(),
        "nodal_stress": terms["nodal_stress"].cpu(),
        "short_beam_penalty": terms["short_beam_penalty"].cpu(),
        "long_beam_penalty": terms["long_beam_penalty"].cpu(),
        "thin_diameter_penalty": terms["thin_diameter_penalty"].cpu(),
        "thick_diameter_penalty": terms["thick_diameter_penalty"].cpu(),
        "node_spacing_penalty": terms["node_spacing_penalty"].cpu(),
        "free_repulsion_penalty": terms["free_repulsion_penalty"].cpu(),
        "rigid_attachment_penalty": terms["rigid_attachment_penalty"].cpu(),
        "centroid_penalty": terms["centroid_penalty"].cpu(),
        "spread_penalty": terms["spread_penalty"].cpu(),
        "soft_domain_penalty": terms["soft_domain_penalty"].cpu(),
        "stress_loss": terms["stress_loss"].cpu(),
        "max_von_mises_stress": terms["max_von_mises_stress"].cpu(),
        "max_stress_ratio": terms["max_stress_ratio"].cpu(),
        "log_dir": str(log_dir),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    _progress(
        "sample:done "
        f"seed={sample_seed} response={_format_matrix(terms['response_matrix'][0])} animation={animation_path} log_dir={log_dir}"
    )
    return result


def _train_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the point-and-beam compliant mechanism prototype"
    )
    parser.add_argument("--num-nodes", type=int, default=defaults.num_nodes)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--nhead", type=int, default=defaults.nhead)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument(
        "--node-effect-dim",
        type=int,
        default=defaults.node_effect_dim,
    )
    parser.add_argument("--train-steps", type=int, default=defaults.train_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--learning-rate-warmup-steps",
        type=int,
        default=defaults.learning_rate_warmup_steps,
    )
    parser.add_argument(
        "--learning-rate-min-scale",
        type=float,
        default=defaults.learning_rate_min_scale,
    )
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument(
        "--position-step-size", type=float, default=defaults.position_step_size
    )
    parser.add_argument(
        "--connectivity-step-size",
        type=float,
        default=defaults.connectivity_step_size,
    )
    parser.add_argument(
        "--rollout-position-noise",
        type=float,
        default=defaults.rollout_position_noise,
    )
    parser.add_argument(
        "--rollout-connectivity-noise",
        type=float,
        default=defaults.rollout_connectivity_noise,
    )
    parser.add_argument(
        "--supervised-denoising-weight",
        type=float,
        default=defaults.supervised_denoising_weight,
    )
    parser.add_argument(
        "--supervised-position-weight",
        type=float,
        default=defaults.supervised_position_weight,
    )
    parser.add_argument(
        "--supervised-adjacency-weight",
        type=float,
        default=defaults.supervised_adjacency_weight,
    )
    parser.add_argument(
        "--supervised-position-noise",
        type=float,
        default=defaults.supervised_position_noise,
    )
    parser.add_argument(
        "--supervised-connectivity-noise",
        type=float,
        default=defaults.supervised_connectivity_noise,
    )
    parser.add_argument(
        "--supervised-every-steps",
        type=int,
        default=defaults.supervised_every_steps,
    )
    parser.add_argument(
        "--supervised-priority-start",
        type=int,
        default=defaults.supervised_priority_start,
    )
    parser.add_argument(
        "--supervised-priority-end",
        type=int,
        default=defaults.supervised_priority_end,
    )
    parser.add_argument(
        "--supervised-priority-duration",
        type=int,
        default=defaults.supervised_priority_duration,
    )
    parser.add_argument(
        "--property-weight", type=float, default=defaults.property_weight
    )
    parser.add_argument(
        "--stress-weight",
        type=float,
        default=defaults.stress_weight,
    )
    parser.add_argument(
        "--monotonic-improvement-weight",
        type=float,
        default=defaults.monotonic_improvement_weight,
    )
    parser.add_argument(
        "--material-weight", type=float, default=defaults.material_weight
    )
    parser.add_argument(
        "--sparsity-weight", type=float, default=defaults.sparsity_weight
    )
    parser.add_argument(
        "--connectivity-weight", type=float, default=defaults.connectivity_weight
    )
    parser.add_argument(
        "--fixed-mobile-connectivity-weight",
        type=float,
        default=defaults.fixed_mobile_connectivity_weight,
    )
    parser.add_argument(
        "--short-beam-weight", type=float, default=defaults.short_beam_weight
    )
    parser.add_argument(
        "--long-beam-weight", type=float, default=defaults.long_beam_weight
    )
    parser.add_argument(
        "--thin-diameter-weight", type=float, default=defaults.thin_diameter_weight
    )
    parser.add_argument(
        "--thick-diameter-weight", type=float, default=defaults.thick_diameter_weight
    )
    parser.add_argument(
        "--node-spacing-weight", type=float, default=defaults.node_spacing_weight
    )
    parser.add_argument(
        "--free-repulsion-weight",
        type=float,
        default=defaults.free_repulsion_weight,
    )
    parser.add_argument(
        "--rigid-attachment-weight",
        type=float,
        default=defaults.rigid_attachment_weight,
    )
    parser.add_argument(
        "--centroid-weight", type=float, default=defaults.centroid_weight
    )
    parser.add_argument("--spread-weight", type=float, default=defaults.spread_weight)
    parser.add_argument(
        "--soft-domain-weight", type=float, default=defaults.soft_domain_weight
    )
    parser.add_argument(
        "--repertoire-bootstrap-cases",
        type=int,
        default=defaults.repertoire_bootstrap_cases,
    )
    parser.add_argument(
        "--repertoire-max-cases",
        type=int,
        default=defaults.repertoire_max_cases,
    )
    parser.add_argument(
        "--canonical-case-count",
        type=int,
        default=defaults.canonical_case_count,
    )
    parser.add_argument(
        "--min-beam-length", type=float, default=defaults.min_beam_length
    )
    parser.add_argument(
        "--max-beam-length", type=float, default=defaults.max_beam_length
    )
    parser.add_argument(
        "--min-beam-diameter", type=float, default=defaults.min_beam_diameter
    )
    parser.add_argument(
        "--max-beam-diameter", type=float, default=defaults.max_beam_diameter
    )
    parser.add_argument(
        "--min-free-node-spacing",
        type=float,
        default=defaults.min_free_node_spacing,
    )
    parser.add_argument(
        "--animation-every-steps",
        type=int,
        default=defaults.animation_every_steps,
    )
    parser.add_argument("--log-every-steps", type=int, default=defaults.log_every_steps)
    parser.add_argument(
        "--canonical-eval-every-steps",
        type=int,
        default=defaults.canonical_eval_every_steps,
    )
    parser.add_argument(
        "--sample-threshold", type=float, default=defaults.sample_threshold
    )
    parser.add_argument(
        "--display-animation-scale",
        type=float,
        default=defaults.display_animation_scale,
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--name", default=defaults.name)
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _sample_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Sample a point-and-beam design from a trained prototype"
    )
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path)
    parser.add_argument(
        "--target-stiffness",
        required=True,
        help="Nine comma-separated row-major values for the 3x3 target stiffness matrix",
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--sample-threshold", type=float, default=defaults.sample_threshold
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default="sample")
    parser.add_argument("--output-path", default="artifacts/sample.pt")
    return parser


def train_main() -> None:
    args = _train_parser().parse_args()
    checkpoint_path, log_dir = train(TrainConfig(**vars(args)))
    print(f"checkpoint={checkpoint_path}")
    print(f"log_dir={log_dir}")


def sample_main() -> None:
    args = _sample_parser().parse_args()
    result = sample(
        checkpoint_path=args.checkpoint_path,
        target_stiffness=_parse_target_stiffness(args.target_stiffness),
        name=args.name,
        output_path=args.output_path,
        steps=args.steps,
        sample_threshold=args.sample_threshold,
        device_override=args.device,
        seed_override=args.seed,
    )
    print(f"seed={result['seed']}")
    print(f"log_dir={result['log_dir']}")
    print(f"animation_path={result['animation_path']}")
    print(f"achieved_response={_format_matrix(result['response_matrix'][0])}")
    print(f"achieved_stiffness={_format_matrix(result['stiffness_matrix'][0])}")
