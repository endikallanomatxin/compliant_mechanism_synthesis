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
    ROLE_FREE,
    adjacency_logits,
    apply_free_node_update,
    logits_to_adjacency,
    symmetric_matrix_unique_values,
    unique_values_to_symmetric_matrix,
)
from compliant_mechanism_synthesis.data import generate_noise_sample
from compliant_mechanism_synthesis.mechanics import (
    GeometryRegularizationConfig,
    mechanical_terms,
    threshold_connectivity,
)
from compliant_mechanism_synthesis.model import GraphRefinementModel
from compliant_mechanism_synthesis.viz import plot_graph_design


@dataclass
class TrainConfig:
    num_nodes: int = 64
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    latent_dim: int = 256
    batch_size: int = 2048
    train_steps: int = 20000
    learning_rate: float = 1e-5
    rollout_steps: int = 8
    position_step_size: float = 0.2
    connectivity_step_size: float = 1.0
    property_weight: float = 2.0
    material_weight: float = 0.02
    connectivity_weight: float = 0.10
    short_beam_weight: float = 0.20
    long_beam_weight: float = 0.10
    thin_diameter_weight: float = 0.20
    thick_diameter_weight: float = 0.10
    min_beam_length: float = 0.01
    max_beam_length: float = 0.1
    min_beam_diameter: float = 0.001
    max_beam_diameter: float = 0.01
    log_every_steps: int = 5
    canonical_eval_every_steps: int = 20
    sample_threshold: float = 0.5
    device: str = "auto"
    name: str = "prototype"
    checkpoint_path: str = "artifacts/prototype.pt"
    seed: int = 7


@dataclass
class ResponseStatistics:
    count: int
    sum: torch.Tensor
    sumsq: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> "ResponseStatistics":
        zeros = torch.zeros(6, device=device, dtype=torch.float32)
        return cls(count=0, sum=zeros.clone(), sumsq=zeros.clone())

    def update(self, response_matrix: torch.Tensor) -> None:
        features = symmetric_matrix_unique_values(response_matrix.detach())
        self.count += int(features.shape[0])
        self.sum = self.sum + features.sum(dim=0)
        self.sumsq = self.sumsq + features.square().sum(dim=0)

    def normalization(self) -> dict[str, list[float]]:
        count = max(self.count, 1)
        mean = self.sum / count
        variance = (self.sumsq / count) - mean.square()
        std = variance.clamp_min(1e-6).sqrt()
        return {"mean": mean.tolist(), "std": std.tolist()}

    def sample_targets(self, batch_size: int, device: torch.device) -> torch.Tensor:
        normalization = self.normalization()
        mean = torch.tensor(normalization["mean"], device=device, dtype=torch.float32)
        std = torch.tensor(normalization["std"], device=device, dtype=torch.float32)
        sampled = mean.unsqueeze(0) + torch.randn(
            (batch_size, mean.shape[0]), device=device, dtype=torch.float32
        ) * std.unsqueeze(0)
        return unique_values_to_symmetric_matrix(sampled, size=3)


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


def _timestamped_run_dir(name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-{name}"


def _pure_noise_batch(
    batch_size: int, num_nodes: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = []
    roles = []
    adjacency = []
    state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    for _ in range(batch_size):
        x, r, a = generate_noise_sample(num_nodes)
        positions.append(x)
        roles.append(r)
        adjacency.append(a)
    random.setstate(state)
    torch.random.set_rng_state(torch_state)
    return (
        torch.stack(positions, dim=0).to(device),
        torch.stack(roles, dim=0).to(device),
        torch.stack(adjacency, dim=0).to(device),
    )


def _canonical_target_specs(
    normalization: dict[str, list[float]],
    device: torch.device,
) -> list[tuple[str, torch.Tensor]]:
    mean = torch.tensor(normalization["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(normalization["std"], device=device, dtype=torch.float32)
    low = mean - std
    high = mean + std
    specs = []
    first = mean.clone()
    first[0] = low[0]
    first[3] = high[3]
    first[5] = high[5]
    specs.append(
        (
            "low-cxx_high-cyy-ctheta",
            unique_values_to_symmetric_matrix(first.unsqueeze(0), size=3)[0],
        )
    )

    second = mean.clone()
    second[0] = high[0]
    second[3] = low[3]
    second[5] = high[5]
    specs.append(
        (
            "high-cxx_low-cyy-high-ctheta",
            unique_values_to_symmetric_matrix(second.unsqueeze(0), size=3)[0],
        )
    )

    third = mean.clone()
    third[0] = high[0]
    third[3] = high[3]
    third[5] = low[5]
    specs.append(
        (
            "high-cxx-cyy_low-ctheta",
            unique_values_to_symmetric_matrix(third.unsqueeze(0), size=3)[0],
        )
    )
    return specs


def _step_weights(steps: int, device: torch.device) -> torch.Tensor:
    weights = torch.linspace(1.0, float(steps), steps=steps, device=device)
    return weights / weights.sum()


def _format_matrix(matrix: torch.Tensor | list[list[float]]) -> str:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().tolist()
    rows = ["[" + ",".join(f"{value:.4e}" for value in row) + "]" for row in matrix]
    return "[" + ";".join(rows) + "]"


def _parse_target_response(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 9:
        raise ValueError(
            "target response must contain exactly 9 comma-separated values"
        )
    return values


def _normalize_target_response(
    raw_targets: torch.Tensor,
    normalization: dict[str, list[float]],
) -> torch.Tensor:
    flattened = symmetric_matrix_unique_values(raw_targets)
    mean = raw_targets.new_tensor(normalization["mean"])
    std = raw_targets.new_tensor(normalization["std"])
    normalized = (flattened - mean) / std
    return unique_values_to_symmetric_matrix(normalized, size=3)


def _response_matrix_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    normalization: dict[str, list[float]],
) -> torch.Tensor:
    pred_unique = symmetric_matrix_unique_values(predicted)
    target_unique = symmetric_matrix_unique_values(target)
    mean = predicted.new_tensor(normalization["mean"])
    std = predicted.new_tensor(normalization["std"])
    return F.mse_loss((pred_unique - mean) / std, (target_unique - mean) / std)


def _geometry_regularization_config(
    config: TrainConfig,
) -> GeometryRegularizationConfig:
    return GeometryRegularizationConfig(
        min_length=config.min_beam_length,
        max_length=config.max_beam_length,
        min_diameter=config.min_beam_diameter,
        max_diameter=config.max_beam_diameter,
    )


def _log_response_matrix(
    writer: SummaryWriter,
    prefix: str,
    matrix: torch.Tensor,
    step: int,
) -> None:
    names = [
        ["ux_fx", "ux_fy", "ux_m"],
        ["uy_fx", "uy_fy", "uy_m"],
        ["theta_fx", "theta_fy", "theta_m"],
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
    targets: torch.Tensor,
    steps: int,
    position_step_size: float,
    connectivity_step_size: float,
    base_time: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    current_positions = positions
    current_adjacency = adjacency
    states: list[dict[str, torch.Tensor]] = []

    for step_idx in range(steps):
        time_fraction = 1.0 - step_idx / max(steps - 1, 1)
        timestep = base_time * time_fraction
        outputs = model(current_positions, roles, current_adjacency, targets, timestep)
        current_positions = apply_free_node_update(
            current_positions,
            outputs["displacements"],
            roles,
            position_step_size,
        )
        current_adjacency = logits_to_adjacency(
            adjacency_logits(current_adjacency)
            + connectivity_step_size * outputs["delta_scores"]
        )
        states.append(
            {
                "positions": current_positions,
                "roles": roles,
                "adjacency": current_adjacency,
                "displacements": outputs["displacements"],
                "node_latents": outputs["node_latents"],
                "delta_scores": outputs["delta_scores"],
            }
        )
    return states


def _log_canonical_evaluation(
    writer: SummaryWriter,
    model: GraphRefinementModel,
    config: TrainConfig,
    step: int,
    device: torch.device,
    canonical_specs: list[tuple[str, torch.Tensor]],
    target_normalization: dict[str, list[float]],
) -> None:
    geometry_config = _geometry_regularization_config(config)
    positions, roles, adjacency = _pure_noise_batch(
        len(canonical_specs), config.num_nodes, config.seed + 101, device
    )
    raw_targets = torch.stack([values for _, values in canonical_specs], dim=0).to(
        device
    )
    targets = _normalize_target_response(raw_targets, target_normalization)
    base_time = torch.ones((len(canonical_specs),), device=device)
    rollout = rollout_refinement(
        model,
        positions,
        roles,
        adjacency,
        targets,
        steps=config.rollout_steps,
        position_step_size=config.position_step_size,
        connectivity_step_size=config.connectivity_step_size,
        base_time=base_time,
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
            threshold_connectivity(
                final_state["adjacency"][idx : idx + 1],
                threshold=config.sample_threshold,
            )[0],
            threshold=0.05,
            title=name,
        )
        writer.add_figure(f"canonical/{name}/design", figure, global_step=step)
        _log_response_matrix(writer, f"canonical/{name}/target", target_values, step)
        _log_response_matrix(
            writer,
            f"canonical/{name}/achieved",
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
            step_error = _response_matrix_loss(
                step_terms["response_matrix"],
                raw_targets[idx : idx + 1],
                target_normalization,
            )
            writer.add_scalar(
                f"canonical/{name}/property_error_step_{rollout_idx}",
                step_error.item(),
                step,
            )
        plt = figure
        plt.clf()


def train(config: TrainConfig) -> tuple[Path, Path]:
    _seed_everything(config.seed)
    device = _device(config.device)
    geometry_config = _geometry_regularization_config(config)
    train_steps = max(config.train_steps, 1)
    _progress(
        f"train:start device={device} train_steps={train_steps} batch_size={config.batch_size} num_nodes={config.num_nodes}"
    )

    response_stats = ResponseStatistics.empty(device)
    bootstrap_positions, bootstrap_roles, bootstrap_adjacency = _pure_noise_batch(
        max(config.batch_size * 4, 64), config.num_nodes, config.seed + 17, device
    )
    bootstrap_terms = mechanical_terms(
        bootstrap_positions,
        bootstrap_roles,
        bootstrap_adjacency,
        geometry_config=geometry_config,
    )
    response_stats.update(bootstrap_terms["response_matrix"])
    target_normalization = response_stats.normalization()
    canonical_specs = _canonical_target_specs(target_normalization, device)
    _progress(
        "train:canonical_targets "
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
        "material": 0.0,
        "connectivity": 0.0,
        "short_beam": 0.0,
        "long_beam": 0.0,
        "thin_diameter": 0.0,
        "thick_diameter": 0.0,
    }

    model.train()
    for step in range(1, train_steps + 1):
        current_seed = config.seed + step + 1000
        positions, roles, adjacency = _pure_noise_batch(
            config.batch_size, config.num_nodes, current_seed, device
        )
        target_normalization = response_stats.normalization()
        raw_targets = response_stats.sample_targets(config.batch_size, device)
        target_features = _normalize_target_response(raw_targets, target_normalization)
        base_time = torch.rand((positions.shape[0],), device=device)

        rollout = rollout_refinement(
            model,
            positions,
            roles,
            adjacency,
            target_features,
            steps=config.rollout_steps,
            position_step_size=config.position_step_size,
            connectivity_step_size=config.connectivity_step_size,
            base_time=base_time,
        )
        weights = _step_weights(config.rollout_steps, device)
        property_loss = torch.zeros((), device=device)
        for step_idx, state in enumerate(rollout):
            step_terms = mechanical_terms(
                state["positions"],
                roles,
                state["adjacency"],
                geometry_config=geometry_config,
            )
            property_loss = property_loss + weights[step_idx] * _response_matrix_loss(
                step_terms["response_matrix"],
                raw_targets,
                target_normalization,
            )

        final_state = rollout[-1]
        final_terms = mechanical_terms(
            final_state["positions"],
            roles,
            final_state["adjacency"],
            geometry_config=geometry_config,
        )
        material_loss = final_terms["material"].mean()
        connectivity_loss = final_terms["connectivity_penalty"].mean()
        short_beam_loss = final_terms["short_beam_penalty"].mean()
        long_beam_loss = final_terms["long_beam_penalty"].mean()
        thin_diameter_loss = final_terms["thin_diameter_penalty"].mean()
        thick_diameter_loss = final_terms["thick_diameter_penalty"].mean()

        total = (
            config.property_weight * property_loss
            + config.material_weight * material_loss
            + config.connectivity_weight * connectivity_loss
            + config.short_beam_weight * short_beam_loss
            + config.long_beam_weight * long_beam_loss
            + config.thin_diameter_weight * thin_diameter_loss
            + config.thick_diameter_weight * thick_diameter_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        running_totals["total"] += total.item()
        running_totals["property"] += property_loss.item()
        running_totals["material"] += material_loss.item()
        running_totals["connectivity"] += connectivity_loss.item()
        running_totals["short_beam"] += short_beam_loss.item()
        running_totals["long_beam"] += long_beam_loss.item()
        running_totals["thin_diameter"] += thin_diameter_loss.item()
        running_totals["thick_diameter"] += thick_diameter_loss.item()

        response_stats.update(final_terms["response_matrix"])
        target_normalization = response_stats.normalization()

        writer.add_scalar("train/total_loss", total.item(), global_step)
        writer.add_scalar("train/property_loss", property_loss.item(), global_step)
        writer.add_scalar("train/material_loss", material_loss.item(), global_step)
        writer.add_scalar(
            "train/connectivity_penalty", connectivity_loss.item(), global_step
        )
        writer.add_scalar(
            "train/short_beam_penalty", short_beam_loss.item(), global_step
        )
        writer.add_scalar("train/long_beam_penalty", long_beam_loss.item(), global_step)
        writer.add_scalar(
            "train/thin_diameter_penalty", thin_diameter_loss.item(), global_step
        )
        writer.add_scalar(
            "train/thick_diameter_penalty", thick_diameter_loss.item(), global_step
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
                f"train:step {step}/{train_steps} total={total.item():.4f} prop={property_loss.item():.4f} conn={connectivity_loss.item():.4f} material={material_loss.item():.4f}"
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
                    _canonical_target_specs(target_normalization, device),
                    target_normalization,
                )
            model.train()

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "target_normalization": response_stats.normalization(),
        "response_stats_count": response_stats.count,
    }
    torch.save(payload, checkpoint_path)
    writer.close()
    _progress(f"train:done checkpoint={checkpoint_path} log_dir={log_dir}")
    return checkpoint_path, log_dir


def refine_sample_state(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    raw_target_response: torch.Tensor,
    target_normalization: dict[str, list[float]],
    config: TrainConfig,
    steps: int = 12,
    lr: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry_config = _geometry_regularization_config(config)
    free_mask = (roles == ROLE_FREE).unsqueeze(-1).to(dtype=positions.dtype)
    position_param = torch.nn.Parameter(positions.clone())
    adjacency_param = torch.nn.Parameter(adjacency_logits(adjacency))
    optimizer = torch.optim.Adam([position_param, adjacency_param], lr=lr)

    for _ in range(steps):
        current_positions = positions + free_mask * (position_param - positions)
        current_positions = current_positions.clamp(0.0, 1.0)
        current_adjacency = logits_to_adjacency(adjacency_param)
        terms = mechanical_terms(
            current_positions,
            roles,
            current_adjacency,
            geometry_config=geometry_config,
        )
        property_loss = _response_matrix_loss(
            terms["response_matrix"],
            raw_target_response,
            target_normalization,
        )
        loss = (
            config.property_weight * property_loss
            + config.material_weight * terms["material"].mean()
            + config.connectivity_weight * terms["connectivity_penalty"].mean()
            + config.short_beam_weight * terms["short_beam_penalty"].mean()
            + config.long_beam_weight * terms["long_beam_penalty"].mean()
            + config.thin_diameter_weight * terms["thin_diameter_penalty"].mean()
            + config.thick_diameter_weight * terms["thick_diameter_penalty"].mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    refined_positions = (
        (positions + free_mask * (position_param - positions)).detach().clamp(0.0, 1.0)
    )
    refined_adjacency = logits_to_adjacency(adjacency_param.detach())
    return refined_positions, refined_adjacency


def sample(
    checkpoint_path: str,
    target_response: list[float],
    name: str,
    output_path: str,
    steps: int,
    sample_threshold: float,
    device_override: str | None = None,
) -> dict[str, object]:
    device = _device(device_override or "auto")
    payload = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**payload["config"])
    if device_override is not None:
        config.device = device_override
    geometry_config = _geometry_regularization_config(config)
    target_normalization = payload["target_normalization"]
    _seed_everything(config.seed)

    model = GraphRefinementModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        latent_dim=config.latent_dim,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    positions, roles, adjacency = _pure_noise_batch(
        1, config.num_nodes, config.seed + 500, device
    )
    raw_targets = torch.tensor(
        target_response, dtype=torch.float32, device=device
    ).reshape(1, 3, 3)
    raw_targets = 0.5 * (raw_targets + raw_targets.transpose(1, 2))
    targets = _normalize_target_response(raw_targets, target_normalization)
    base_time = torch.ones((1,), device=device)
    with torch.no_grad():
        rollout = rollout_refinement(
            model,
            positions,
            roles,
            adjacency,
            targets,
            steps=steps,
            position_step_size=config.position_step_size,
            connectivity_step_size=config.connectivity_step_size,
            base_time=base_time,
        )
    final_state = rollout[-1]
    refined_positions, refined_adjacency = refine_sample_state(
        final_state["positions"],
        roles,
        final_state["adjacency"],
        raw_targets,
        target_normalization,
        config,
    )
    thresholded_adjacency = threshold_connectivity(
        refined_adjacency, threshold=sample_threshold
    )
    terms = mechanical_terms(
        refined_positions,
        roles,
        thresholded_adjacency,
        geometry_config=geometry_config,
    )

    log_dir = _timestamped_run_dir(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    figure = plot_graph_design(
        refined_positions[0],
        roles[0],
        thresholded_adjacency[0],
        threshold=0.05,
        title=name,
    )
    writer.add_figure("sample/final_graph", figure, global_step=0)
    _log_response_matrix(writer, "sample/target_response", raw_targets[0], 0)
    _log_response_matrix(
        writer, "sample/achieved_response", terms["response_matrix"][0], 0
    )
    _log_response_matrix(
        writer, "sample/achieved_stiffness", terms["stiffness_matrix"][0], 0
    )
    writer.add_scalar(
        "sample/short_beam_penalty", terms["short_beam_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/long_beam_penalty", terms["long_beam_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/thin_diameter_penalty", terms["thin_diameter_penalty"][0].item(), 0
    )
    writer.add_scalar(
        "sample/thick_diameter_penalty", terms["thick_diameter_penalty"][0].item(), 0
    )
    writer.close()

    result = {
        "positions": refined_positions.cpu(),
        "roles": roles.cpu(),
        "adjacency": refined_adjacency.cpu(),
        "thresholded_adjacency": thresholded_adjacency.cpu(),
        "response_matrix": terms["response_matrix"].cpu(),
        "stiffness_matrix": terms["stiffness_matrix"].cpu(),
        "short_beam_penalty": terms["short_beam_penalty"].cpu(),
        "long_beam_penalty": terms["long_beam_penalty"].cpu(),
        "thin_diameter_penalty": terms["thin_diameter_penalty"].cpu(),
        "thick_diameter_penalty": terms["thick_diameter_penalty"].cpu(),
        "log_dir": str(log_dir),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    _progress(
        "sample:done "
        f"response={_format_matrix(terms['response_matrix'][0])} log_dir={log_dir}"
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
        "--property-weight", type=float, default=defaults.property_weight
    )
    parser.add_argument(
        "--material-weight", type=float, default=defaults.material_weight
    )
    parser.add_argument(
        "--connectivity-weight", type=float, default=defaults.connectivity_weight
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
        "--target-response",
        required=True,
        help="Nine comma-separated row-major values for the 3x3 target response matrix",
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--sample-threshold", type=float, default=defaults.sample_threshold
    )
    parser.add_argument("--device", default=defaults.device)
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
        target_response=_parse_target_response(args.target_response),
        name=args.name,
        output_path=args.output_path,
        steps=args.steps,
        sample_threshold=args.sample_threshold,
        device_override=args.device,
    )
    print(f"log_dir={result['log_dir']}")
    print(f"achieved_response={_format_matrix(result['response_matrix'][0])}")
    print(f"achieved_stiffness={_format_matrix(result['stiffness_matrix'][0])}")
