from __future__ import annotations

import argparse
import random
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
    unique_values_to_symmetric_matrix,
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
from compliant_mechanism_synthesis.scaling import (
    denormalize_generalized_stiffness_matrix,
    normalize_generalized_stiffness_matrix,
)
from compliant_mechanism_synthesis.viz import (
    export_rollout_animation,
    plot_graph_design,
)


@dataclass
class TrainConfig:
    num_nodes: int = 32
    d_model: int = 128
    nhead: int = 16
    num_layers: int = 12
    latent_dim: int = 128
    batch_size: int = 128
    train_steps: int = 20_000
    learning_rate: float = 1e-4
    rollout_steps: int = 8
    position_step_size: float = 0.2
    connectivity_step_size: float = 0.1
    rollout_position_noise: float = 0.01
    rollout_connectivity_noise: float = 0.05
    supervised_denoising_weight: float = 0.5
    supervised_position_weight: float = 1.0
    supervised_adjacency_weight: float = 1.0
    supervised_position_noise: float = 0.02
    supervised_connectivity_noise: float = 0.08
    supervised_every_steps: int = 1
    training_goal_blend_start: float = 0.0
    training_goal_blend_end: float = 1.0
    property_weight: float = 1.0
    monotonic_improvement_weight: float = 0.25
    material_weight: float = 0.1
    sparsity_weight: float = 0.2
    connectivity_weight: float = 0.25
    fixed_mobile_connectivity_weight: float = 2.0
    short_beam_weight: float = 1.0
    long_beam_weight: float = 2.0
    thin_diameter_weight: float = 0.20
    thick_diameter_weight: float = 0.10
    node_spacing_weight: float = 0.20
    spread_weight: float = 0.15
    soft_domain_weight: float = 40.0
    yield_stress_weight: float = 0.10
    min_beam_length: float = 5e-3
    max_beam_length: float = 3e-2
    min_beam_diameter: float = 2e-4
    max_beam_diameter: float = 3e-3
    min_free_node_spacing: float = 6e-3
    animation_every_steps: int = 1_000
    log_every_steps: int = 5
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


def _load_train_config(config_dict: dict[str, object]) -> TrainConfig:
    data = dict(config_dict)
    if "training_goal_blend" in data:
        blend = float(data.pop("training_goal_blend"))
        data.setdefault("training_goal_blend_start", blend)
        data.setdefault("training_goal_blend_end", blend)
    return TrainConfig(**data)


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


def _fixed_stiffness_target_specs(
    device: torch.device,
) -> list[tuple[str, torch.Tensor]]:
    scales = characteristic_scales(FrameFEMConfig())
    specs = [
        (
            "01_flex_x",
            [4.2, 0.0, 0.0, 6.7, 0.0, 2.8],
        ),
        (
            "02_flex_y",
            [6.7, 0.0, 0.0, 4.2, 0.0, 2.8],
        ),
        (
            "03_flex_theta",
            [6.1, 0.0, 0.0, 6.1, 0.0, 2.1],
        ),
        (
            "04_balanced",
            [5.2, 0.0, 0.0, 5.2, 0.0, 3.0],
        ),
        (
            "05_couple_xy_pos",
            [5.9, 0.6, -0.3, 4.6, 0.2, 2.9],
        ),
        (
            "06_couple_xy_neg",
            [4.6, -0.6, 0.3, 5.9, -0.2, 2.9],
        ),
    ]
    return [
        (
            name,
            denormalize_generalized_stiffness_matrix(
                unique_values_to_symmetric_matrix(
                    torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0),
                    size=3,
                ),
                scales,
            )[0],
        )
        for name, values in specs
    ]


def _sample_stiffness_targets(
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    library = torch.stack(
        [matrix for _, matrix in _fixed_stiffness_target_specs(device)], dim=0
    )
    indices = torch.randint(library.shape[0], (batch_size,), device=device)
    return library.index_select(0, indices)


def _blend_training_targets(
    start_stiffness: torch.Tensor,
    goal_stiffness: torch.Tensor,
    goal_blend: float,
) -> torch.Tensor:
    blend = float(min(max(goal_blend, 0.0), 1.0))
    return (1.0 - blend) * start_stiffness + blend * goal_stiffness


def _scheduled_goal_blend(
    step: int,
    train_steps: int,
    blend_start: float,
    blend_end: float,
) -> float:
    if train_steps <= 1:
        return float(min(max(blend_end, 0.0), 1.0))
    progress = (step - 1) / max(train_steps - 1, 1)
    blend = blend_start + progress * (blend_end - blend_start)
    return float(min(max(blend, 0.0), 1.0))


def _step_weights(steps: int, device: torch.device) -> torch.Tensor:
    weights = torch.linspace(1.0, float(steps), steps=steps, device=device)
    return weights / weights.sum()


def _visualization_threshold() -> float:
    return 0.0


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
            (adjacency + adjacency_noise).clamp(0.0, 1.0),
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
    canonical_specs: list[tuple[str, torch.Tensor]],
    animation_output_dir: Path | None = None,
) -> None:
    geometry_config = _geometry_regularization_config(config)
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
    )
    final_state = rollout[-1]
    final_terms = mechanical_terms(
        final_state["positions"],
        roles,
        final_state["adjacency"],
        geometry_config=geometry_config,
    )

    for idx, (name, target_values) in enumerate(canonical_specs):
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
            f"canonical/10_target_stiffness/{name}",
            target_values,
            step,
        )
        _log_matrix(
            writer,
            f"canonical/20_achieved_stiffness/{name}",
            final_terms["stiffness_matrix"][idx],
            step,
        )
        _log_matrix(
            writer,
            f"canonical/25_achieved_response/{name}",
            final_terms["response_matrix"][idx],
            step,
        )
        for rollout_idx, state in enumerate(rollout, start=1):
            step_terms = mechanical_terms(
                state["positions"][idx : idx + 1],
                roles[idx : idx + 1],
                state["adjacency"][idx : idx + 1],
                geometry_config=geometry_config,
            )
            step_error = _matrix_loss(
                step_terms["stiffness_matrix"],
                raw_targets[idx : idx + 1],
            )
            writer.add_scalar(
                f"canonical/30_property_error/{name}/step_{rollout_idx}",
                step_error.item(),
                step,
            )
        plt = figure
        plt.clf()

    if animation_output_dir is not None and canonical_specs:
        name, _ = canonical_specs[0]
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
            animation_output_dir / f"step_{step:05d}_{name}.gif",
            positions[0],
            roles[0],
            adjacency[0],
            animation_rollout,
            target_responses[0],
            threshold=_visualization_threshold(),
            frame_config=FrameFEMConfig(),
            title=name,
        )
        _progress(f"train:animation path={animation_path}")


def train(config: TrainConfig) -> tuple[Path, Path]:
    _seed_everything(config.seed)
    device = _device(config.device)
    geometry_config = _geometry_regularization_config(config)
    train_steps = max(config.train_steps, 1)
    _progress(
        f"train:start device={device} train_steps={train_steps} batch_size={config.batch_size} num_nodes={config.num_nodes}"
    )

    canonical_specs = _fixed_stiffness_target_specs(device)[:3]
    _progress(
        "train:canonical_stiffness_targets "
        + " ".join(
            f"{name}={_format_matrix(values)}" for name, values in canonical_specs
        )
    )

    model = GraphRefinementModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        latent_dim=config.latent_dim,
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
        "monotonic": 0.0,
        "supervised": 0.0,
        "supervised_position": 0.0,
        "supervised_adjacency": 0.0,
        "material": 0.0,
        "sparsity": 0.0,
        "connectivity": 0.0,
        "fixed_mobile_connectivity": 0.0,
        "short_beam": 0.0,
        "long_beam": 0.0,
        "thin_diameter": 0.0,
        "thick_diameter": 0.0,
        "node_spacing": 0.0,
        "spread": 0.0,
        "soft_domain": 0.0,
        "yield_stress": 0.0,
    }
    step_weights = _step_weights(config.rollout_steps, device)

    model.train()
    for step in range(1, train_steps + 1):
        current_seed = config.seed + step + 1000
        positions, roles, adjacency = _pure_noise_batch(
            config.batch_size,
            config.num_nodes,
            device,
            seed=current_seed,
        )
        goal_targets = _sample_stiffness_targets(config.batch_size, device)
        with torch.no_grad():
            start_terms = mechanical_terms(positions, roles, adjacency)
        goal_blend = _scheduled_goal_blend(
            step,
            train_steps,
            blend_start=config.training_goal_blend_start,
            blend_end=config.training_goal_blend_end,
        )
        raw_targets = _blend_training_targets(
            start_terms["stiffness_matrix"],
            goal_targets,
            goal_blend=goal_blend,
        )
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
            initial_stiffness=start_terms["stiffness_matrix"],
        )
        property_loss = torch.zeros((), device=device)
        material_loss = torch.zeros((), device=device)
        sparsity_loss = torch.zeros((), device=device)
        connectivity_loss = torch.zeros((), device=device)
        fixed_mobile_connectivity_loss = torch.zeros((), device=device)
        short_beam_loss = torch.zeros((), device=device)
        long_beam_loss = torch.zeros((), device=device)
        thin_diameter_loss = torch.zeros((), device=device)
        thick_diameter_loss = torch.zeros((), device=device)
        node_spacing_loss = torch.zeros((), device=device)
        spread_loss = torch.zeros((), device=device)
        soft_domain_loss = torch.zeros((), device=device)
        yield_stress_loss = torch.zeros((), device=device)
        step_errors: list[torch.Tensor] = []
        for step_idx, state in enumerate(rollout):
            step_terms = state["terms"]
            step_error = _matrix_loss(
                step_terms["stiffness_matrix"],
                raw_targets,
            )
            step_errors.append(step_error)
            property_loss = property_loss + step_weights[step_idx] * step_error
            material_loss = (
                material_loss
                + step_weights[step_idx] * step_terms["normalized_material"].mean()
            )
            sparsity_loss = (
                sparsity_loss + step_weights[step_idx] * step_terms["sparsity"].mean()
            )
            connectivity_loss = (
                connectivity_loss
                + step_weights[step_idx] * step_terms["connectivity_penalty"].mean()
            )
            fixed_mobile_connectivity_loss = (
                fixed_mobile_connectivity_loss
                + step_weights[step_idx]
                * step_terms["fixed_mobile_connectivity_penalty"].mean()
            )
            short_beam_loss = (
                short_beam_loss
                + step_weights[step_idx] * step_terms["short_beam_penalty"].mean()
            )
            long_beam_loss = (
                long_beam_loss
                + step_weights[step_idx] * step_terms["long_beam_penalty"].mean()
            )
            thin_diameter_loss = (
                thin_diameter_loss
                + step_weights[step_idx] * step_terms["thin_diameter_penalty"].mean()
            )
            thick_diameter_loss = (
                thick_diameter_loss
                + step_weights[step_idx] * step_terms["thick_diameter_penalty"].mean()
            )
            node_spacing_loss = (
                node_spacing_loss
                + step_weights[step_idx] * step_terms["node_spacing_penalty"].mean()
            )
            spread_loss = (
                spread_loss
                + step_weights[step_idx] * step_terms["spread_penalty"].mean()
            )
            soft_domain_loss = (
                soft_domain_loss
                + step_weights[step_idx] * step_terms["soft_domain_penalty"].mean()
            )
            yield_stress_loss = (
                yield_stress_loss
                + step_weights[step_idx] * step_terms["yield_stress_penalty"].mean()
            )
        monotonic_loss = _monotonic_improvement_loss(step_errors)

        final_state = rollout[-1]
        free_loss = (
            config.property_weight * property_loss
            + config.monotonic_improvement_weight * monotonic_loss
            + config.material_weight * material_loss
            + config.sparsity_weight * sparsity_loss
            + config.connectivity_weight * connectivity_loss
            + config.fixed_mobile_connectivity_weight * fixed_mobile_connectivity_loss
            + config.short_beam_weight * short_beam_loss
            + config.long_beam_weight * long_beam_loss
            + config.thin_diameter_weight * thin_diameter_loss
            + config.thick_diameter_weight * thick_diameter_loss
            + config.node_spacing_weight * node_spacing_loss
            + config.spread_weight * spread_loss
            + config.soft_domain_weight * soft_domain_loss
            + config.yield_stress_weight * yield_stress_loss
        )

        supervised_position_loss = torch.zeros((), device=device)
        supervised_adjacency_loss = torch.zeros((), device=device)
        supervised_loss = torch.zeros((), device=device)
        if (
            config.supervised_denoising_weight > 0.0
            and step % max(config.supervised_every_steps, 1) == 0
        ):
            teacher_positions = final_state["refined_positions"].detach()
            teacher_adjacency = final_state["refined_adjacency"].detach()
            noisy_positions, noisy_adjacency = _sample_supervised_denoising_batch(
                teacher_positions,
                roles,
                teacher_adjacency,
                position_noise_scale=config.supervised_position_noise,
                connectivity_noise_scale=config.supervised_connectivity_noise,
            )
            with torch.no_grad():
                noisy_terms = mechanical_terms(noisy_positions, roles, noisy_adjacency)
                noisy_nodal_mechanics = _nodal_mechanics_features_from_fields(
                    noisy_positions,
                    noisy_terms,
                )
            target_features, current_features, residual_features = (
                _mechanics_condition_matrices(
                    raw_targets,
                    noisy_terms["stiffness_matrix"],
                )
            )
            denoising_outputs = model(
                noisy_positions,
                roles,
                noisy_adjacency,
                target_features,
                current_features,
                residual_features,
                noisy_nodal_mechanics,
                torch.zeros((positions.shape[0],), device=device),
                torch.full(
                    (positions.shape[0],),
                    config.supervised_position_noise,
                    device=device,
                ),
                torch.full(
                    (positions.shape[0],),
                    config.supervised_connectivity_noise,
                    device=device,
                ),
            )
            predicted_positions = apply_free_node_update(
                noisy_positions,
                denoising_outputs["displacements"],
                roles,
                config.position_step_size,
            )
            predicted_adjacency = denoising_outputs["predicted_adjacency"]
            supervised_position_loss, supervised_adjacency_loss = (
                _supervised_reconstruction_losses(
                    predicted_positions,
                    teacher_positions,
                    roles,
                    predicted_adjacency,
                    teacher_adjacency,
                )
            )
            supervised_loss = config.supervised_denoising_weight * (
                config.supervised_position_weight * supervised_position_loss
                + config.supervised_adjacency_weight * supervised_adjacency_loss
            )

        total = free_loss + supervised_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        running_totals["total"] += total.item()
        running_totals["property"] += property_loss.item()
        running_totals["monotonic"] += monotonic_loss.item()
        running_totals["supervised"] += supervised_loss.item()
        running_totals["supervised_position"] += supervised_position_loss.item()
        running_totals["supervised_adjacency"] += supervised_adjacency_loss.item()
        running_totals["material"] += material_loss.item()
        running_totals["sparsity"] += sparsity_loss.item()
        running_totals["connectivity"] += connectivity_loss.item()
        running_totals["fixed_mobile_connectivity"] += (
            fixed_mobile_connectivity_loss.item()
        )
        running_totals["short_beam"] += short_beam_loss.item()
        running_totals["long_beam"] += long_beam_loss.item()
        running_totals["thin_diameter"] += thin_diameter_loss.item()
        running_totals["thick_diameter"] += thick_diameter_loss.item()
        running_totals["node_spacing"] += node_spacing_loss.item()
        running_totals["spread"] += spread_loss.item()
        running_totals["soft_domain"] += soft_domain_loss.item()
        running_totals["yield_stress"] += yield_stress_loss.item()

        if config.log_every_steps > 0 and step % config.log_every_steps == 0:
            writer.add_scalar("train/total_loss", total.item(), global_step)
            writer.add_scalar("train/goal_blend", goal_blend, global_step)
            writer.add_scalar("train/property_loss", property_loss.item(), global_step)
            writer.add_scalar(
                "train/monotonic_improvement_loss", monotonic_loss.item(), global_step
            )
            writer.add_scalar(
                "train/supervised_loss", supervised_loss.item(), global_step
            )
            writer.add_scalar(
                "train/supervised_position_loss",
                supervised_position_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "train/supervised_adjacency_loss",
                supervised_adjacency_loss.item(),
                global_step,
            )
            writer.add_scalar("train/material_loss", material_loss.item(), global_step)
            writer.add_scalar("train/sparsity_loss", sparsity_loss.item(), global_step)
            writer.add_scalar(
                "train/connectivity_penalty", connectivity_loss.item(), global_step
            )
            writer.add_scalar(
                "train/fixed_mobile_connectivity_penalty",
                fixed_mobile_connectivity_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "train/short_beam_penalty", short_beam_loss.item(), global_step
            )
            writer.add_scalar(
                "train/long_beam_penalty", long_beam_loss.item(), global_step
            )
            writer.add_scalar(
                "train/thin_diameter_penalty", thin_diameter_loss.item(), global_step
            )
            writer.add_scalar(
                "train/thick_diameter_penalty", thick_diameter_loss.item(), global_step
            )
            writer.add_scalar(
                "train/node_spacing_penalty", node_spacing_loss.item(), global_step
            )
            writer.add_scalar(
                "train/spread_penalty", spread_loss.item(), global_step
            )
            writer.add_scalar(
                "train/soft_domain_penalty", soft_domain_loss.item(), global_step
            )
            writer.add_scalar(
                "train/yield_stress_penalty",
                yield_stress_loss.item(),
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
                f"train:step {step}/{train_steps} total={total.item():.4f} prop={property_loss.item():.4f} sup={supervised_loss.item():.4f} conn={connectivity_loss.item():.4f} material={material_loss.item():.4f}"
            )
            for key, value in running_totals.items():
                writer.add_scalar(f"window/{key}", value / window, global_step)
                running_totals[key] = 0.0

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
                    canonical_specs,
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
    }
    torch.save(payload, checkpoint_path)
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
            adjacency_param.clamp(0.0, 1.0),
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
            + config.material_weight * terms["normalized_material"].mean()
            + config.sparsity_weight * terms["sparsity"].mean()
            + config.connectivity_weight * terms["connectivity_penalty"].mean()
            + config.fixed_mobile_connectivity_weight
            * terms["fixed_mobile_connectivity_penalty"].mean()
            + config.short_beam_weight * terms["short_beam_penalty"].mean()
            + config.long_beam_weight * terms["long_beam_penalty"].mean()
            + config.thin_diameter_weight * terms["thin_diameter_penalty"].mean()
            + config.thick_diameter_weight * terms["thick_diameter_penalty"].mean()
            + config.node_spacing_weight * terms["node_spacing_penalty"].mean()
            + config.spread_weight * terms["spread_penalty"].mean()
            + config.soft_domain_weight * terms["soft_domain_penalty"].mean()
            + config.yield_stress_weight * terms["yield_stress_penalty"].mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    refined_positions = (positions + free_mask * (position_param - positions)).detach()
    refined_adjacency = enforce_role_adjacency_constraints(
        adjacency_param.detach().clamp(0.0, 1.0),
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
        latent_dim=config.latent_dim,
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
    writer.add_scalar("sample/40_spread_penalty", terms["spread_penalty"][0].item(), 0)
    writer.add_scalar(
        "sample/40_soft_domain_penalty", terms["soft_domain_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/40_yield_stress_penalty",
        terms["yield_stress_penalty"][0].item(),
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
        "spread_penalty": terms["spread_penalty"].cpu(),
        "soft_domain_penalty": terms["soft_domain_penalty"].cpu(),
        "yield_stress_penalty": terms["yield_stress_penalty"].cpu(),
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
    parser.add_argument("--latent-dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--train-steps", type=int, default=defaults.train_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
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
        "--training-goal-blend-start",
        type=float,
        default=defaults.training_goal_blend_start,
    )
    parser.add_argument(
        "--training-goal-blend-end",
        type=float,
        default=defaults.training_goal_blend_end,
    )
    parser.add_argument(
        "--property-weight", type=float, default=defaults.property_weight
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
    parser.add_argument("--spread-weight", type=float, default=defaults.spread_weight)
    parser.add_argument(
        "--soft-domain-weight", type=float, default=defaults.soft_domain_weight
    )
    parser.add_argument(
        "--yield-stress-weight",
        type=float,
        default=defaults.yield_stress_weight,
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
