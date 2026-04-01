from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.data import generate_dataset, generate_design
from compliant_mechanism_synthesis.mechanics import (
    binarization_penalty,
    mechanical_terms,
    threshold_occupancy,
)
from compliant_mechanism_synthesis.model import ConditionedDenoiser


@dataclass
class TrainConfig:
    grid_size: int = 24
    patch_size: int = 2
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 6
    dataset_size: int = 512
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 3e-4
    rollout_steps: int = 8
    rollout_step_size: float = 0.5
    rollout_noise_scale: float = 0.03
    property_weight: float = 2.0
    improvement_weight: float = 0.25
    diversity_weight: float = 0.05
    diversity_scale: float = 0.15
    surface_weight: float = 0.02
    connectivity_weight: float = 0.12
    mass_weight: float = 0.15
    binarization_weight: float = 0.1
    train_samples_per_target: int = 4
    train_softmin_temperature: float = 0.25
    log_every_steps: int = 5
    canonical_eval_every_steps: int = 20
    name: str = "prototype"
    checkpoint_path: str = "artifacts/prototype.pt"
    seed: int = 7


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fixed_noise(
    batch_size: int, grid_size: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.rand((batch_size, 1, grid_size, grid_size), generator=generator)
    return noise.to(device)


def _force_plates(grids: torch.Tensor) -> torch.Tensor:
    grids = grids.clone()
    grids[:, :, 0, :] = 1.0
    grids[:, :, -1, :] = 1.0
    return grids.clamp(0.0, 1.0)


def _write_samples(
    writer: SummaryWriter, tag: str, grids: torch.Tensor, step: int
) -> None:
    writer.add_images(tag, grids[:8], global_step=step, dataformats="NCHW")


def _timestamped_run_dir(name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-{name}"


def _progress(message: str) -> None:
    print(message, flush=True)


def _canonical_target_specs(
    target_pool: torch.Tensor,
) -> list[tuple[str, tuple[float, float, float]]]:
    mean = target_pool.mean(dim=0)
    std = target_pool.std(dim=0, unbiased=False)
    low = (mean - std).clamp(0.0, 1.0)
    high = (mean + std).clamp(0.0, 1.0)

    return [
        (
            "low-kx_high-ky-ktheta",
            (float(low[0]), float(high[1]), float(high[2])),
        ),
        (
            "high-kx_low-ky-high-ktheta",
            (float(high[0]), float(low[1]), float(high[2])),
        ),
        (
            "high-kx-ky_low-ktheta",
            (float(high[0]), float(high[1]), float(low[2])),
        ),
    ]


def _step_weights(steps: int, device: torch.device) -> torch.Tensor:
    weights = torch.linspace(1.0, float(steps), steps=steps, device=device)
    return weights / weights.sum()


def _target_sampling_weights(
    targets: torch.Tensor, bins_per_dim: int = 6
) -> torch.Tensor:
    weights = torch.zeros(targets.shape[0], dtype=torch.float32)
    quantiles = torch.linspace(0.0, 1.0, steps=bins_per_dim + 1)
    for dim in range(targets.shape[1]):
        values = targets[:, dim].contiguous()
        edges = torch.quantile(values, quantiles)
        inner_edges = edges[1:-1].contiguous()
        bin_ids = torch.bucketize(values, inner_edges)
        counts = torch.bincount(bin_ids, minlength=bins_per_dim).clamp_min(1)
        weights = weights + counts[bin_ids].float().reciprocal()

    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def _softmin_weights(losses: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = -losses / max(temperature, 1e-6)
    return torch.softmax(scaled, dim=1)


def _monotonic_improvement_penalty(step_errors: list[torch.Tensor]) -> torch.Tensor:
    penalties = []
    for previous, current in zip(step_errors, step_errors[1:]):
        penalties.append(F.relu(current - previous))
    if not penalties:
        return torch.zeros_like(step_errors[0])
    return torch.stack(penalties, dim=0).mean(dim=0)


def _diversity_penalty(grouped_probs: torch.Tensor, scale: float) -> torch.Tensor:
    sample_count = grouped_probs.shape[1]
    if sample_count < 2:
        return torch.zeros(grouped_probs.shape[0], device=grouped_probs.device)

    flattened = grouped_probs.reshape(grouped_probs.shape[0], sample_count, -1)
    pair_penalties = []
    for left in range(sample_count):
        for right in range(left + 1, sample_count):
            difference = (flattened[:, left] - flattened[:, right]).abs().mean(dim=1)
            pair_penalties.append(torch.exp(-difference / max(scale, 1e-6)))

    return torch.stack(pair_penalties, dim=0).mean(dim=0)


def candidate_scores(
    designs: torch.Tensor, target_props: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    terms = mechanical_terms(designs)
    property_error = (terms["properties"] - target_props).square().mean(dim=1)
    score = (
        property_error
        + 0.10 * terms["connectivity_penalty"]
        + 0.02 * terms["occupancy_mass"]
        + 0.005 * terms["surface"]
    )
    return score, property_error, terms


def _log_canonical_evaluation(
    writer: SummaryWriter,
    model: ConditionedDenoiser,
    config: TrainConfig,
    step: int,
    device: torch.device,
    canonical_specs: list[tuple[str, tuple[float, float, float]]],
) -> None:
    targets = torch.tensor(
        [values for _, values in canonical_specs], dtype=torch.float32, device=device
    )
    noise = _fixed_noise(
        len(canonical_specs), config.grid_size, config.seed + 101, device
    )
    rollout = rollout_model(
        model,
        targets,
        noise,
        steps=config.rollout_steps,
        step_size=config.rollout_step_size,
    )
    probs = rollout[-1]
    design = binarize_design(probs)
    search_scores, property_error, terms = candidate_scores(design, targets)

    step_errors = []
    for probs_step in rollout:
        step_terms = mechanical_terms(probs_step)
        step_errors.append(
            ((step_terms["properties"] - targets).square().mean(dim=1)).cpu()
        )

    for idx, (name, target_values) in enumerate(canonical_specs):
        design_image = design[idx : idx + 1].cpu()
        achieved = terms["properties"][idx]
        error = property_error[idx].item()
        writer.add_images(
            f"canonical/{name}/design",
            design_image,
            global_step=step,
            dataformats="NCHW",
        )
        writer.add_scalar(f"canonical/{name}/target_kx", target_values[0], step)
        writer.add_scalar(f"canonical/{name}/target_ky", target_values[1], step)
        writer.add_scalar(f"canonical/{name}/target_ktheta", target_values[2], step)
        writer.add_scalar(f"canonical/{name}/achieved_kx", achieved[0].item(), step)
        writer.add_scalar(f"canonical/{name}/achieved_ky", achieved[1].item(), step)
        writer.add_scalar(f"canonical/{name}/achieved_ktheta", achieved[2].item(), step)
        writer.add_scalar(f"canonical/{name}/property_error", error, step)
        writer.add_scalar(f"canonical/{name}/score", search_scores[idx].item(), step)
        for rollout_idx, errors in enumerate(step_errors, start=1):
            writer.add_scalar(
                f"canonical/{name}/property_error_step_{rollout_idx}",
                errors[idx].item(),
                step,
            )

    _progress(
        "train:canonical_eval "
        f"step={step} {canonical_specs[0][0]}={property_error[0].item():.4f} "
        f"{canonical_specs[1][0]}={property_error[1].item():.4f} "
        f"{canonical_specs[2][0]}={property_error[2].item():.4f}"
    )


def train(config: TrainConfig) -> tuple[Path, Path]:
    _seed_everything(config.seed)
    device = _device()
    _progress(
        "train:start "
        f"device={device} epochs={config.epochs} dataset_size={config.dataset_size} "
        f"batch_size={config.batch_size} rollout_steps={config.rollout_steps} "
        f"samples_per_target={config.train_samples_per_target}"
    )

    designs = generate_dataset(config.dataset_size, config.grid_size, config.seed)
    targets = mechanical_terms(threshold_occupancy(designs))["properties"]
    canonical_reference_size = max(128, config.dataset_size)
    canonical_reference_designs = generate_dataset(
        canonical_reference_size, config.grid_size, config.seed + 1
    )
    canonical_reference_targets = mechanical_terms(
        threshold_occupancy(canonical_reference_designs)
    )["properties"]
    canonical_specs = _canonical_target_specs(canonical_reference_targets)
    dataset = TensorDataset(targets)
    sampling_weights = _target_sampling_weights(targets)
    sampler = WeightedRandomSampler(
        weights=sampling_weights,
        num_samples=len(targets),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    _progress(
        "train:canonical_stats "
        f"mu=({canonical_reference_targets[:, 0].mean().item():.3f},"
        f"{canonical_reference_targets[:, 1].mean().item():.3f},"
        f"{canonical_reference_targets[:, 2].mean().item():.3f}) "
        f"sigma=({canonical_reference_targets[:, 0].std(unbiased=False).item():.3f},"
        f"{canonical_reference_targets[:, 1].std(unbiased=False).item():.3f},"
        f"{canonical_reference_targets[:, 2].std(unbiased=False).item():.3f})"
    )
    _progress(
        "train:canonical_targets "
        + " ".join(
            f"{name}=({values[0]:.3f},{values[1]:.3f},{values[2]:.3f})"
            for name, values in canonical_specs
        )
    )

    model = ConditionedDenoiser(
        grid_size=config.grid_size,
        patch_size=config.patch_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    log_dir = _timestamped_run_dir(config.name)
    checkpoint_path = Path(config.checkpoint_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        _progress(f"train:epoch_start epoch={epoch + 1}/{config.epochs}")
        epoch_totals = {
            "total": 0.0,
            "property": 0.0,
            "surface": 0.0,
            "connectivity": 0.0,
            "mass": 0.0,
            "binarization": 0.0,
            "improvement": 0.0,
            "diversity": 0.0,
            "binary_property_error": 0.0,
            "best_binary_property_error": 0.0,
            "candidate_spread": 0.0,
        }

        for batch_idx, (target_props,) in enumerate(loader, start=1):
            target_props = target_props.to(device)
            expanded_targets = target_props.repeat_interleave(
                config.train_samples_per_target, dim=0
            )
            noise = torch.rand(
                (
                    expanded_targets.shape[0],
                    1,
                    config.grid_size,
                    config.grid_size,
                ),
                device=device,
            )
            rollout = rollout_model(
                model,
                expanded_targets,
                noise,
                steps=config.rollout_steps,
                step_size=config.rollout_step_size,
                noise_scale=config.rollout_noise_scale,
            )
            final_probs = rollout[-1]
            final_terms = mechanical_terms(final_probs)
            binary_predictions = _force_plates(threshold_occupancy(final_probs))
            _, binary_property_error, _ = candidate_scores(
                binary_predictions, expanded_targets
            )

            step_weights = _step_weights(config.rollout_steps, device)
            per_candidate_property = torch.zeros(
                expanded_targets.shape[0], device=device
            )
            per_step_errors: list[torch.Tensor] = []
            for step_idx, probs in enumerate(rollout):
                step_terms = mechanical_terms(probs)
                step_error = (
                    (step_terms["properties"] - expanded_targets).square().mean(dim=1)
                )
                per_step_errors.append(step_error)
                per_candidate_property = (
                    per_candidate_property + step_weights[step_idx] * step_error
                )

            improvement_penalty = _monotonic_improvement_penalty(per_step_errors)

            per_candidate_total = (
                config.property_weight * per_candidate_property
                + config.improvement_weight * improvement_penalty
                + config.surface_weight * final_terms["surface"]
                + config.connectivity_weight * final_terms["connectivity_penalty"]
                + config.mass_weight * final_terms["occupancy_mass"]
                + config.binarization_weight * binarization_penalty(final_probs)
            )
            per_target_total = per_candidate_total.view(
                target_props.shape[0], config.train_samples_per_target
            )
            soft_weights = _softmin_weights(
                per_target_total, config.train_softmin_temperature
            )
            total = (soft_weights * per_target_total).sum(dim=1).mean()

            per_target_property = per_candidate_property.view(
                target_props.shape[0], config.train_samples_per_target
            )
            per_target_surface = final_terms["surface"].view(
                target_props.shape[0], config.train_samples_per_target
            )
            per_target_connectivity = final_terms["connectivity_penalty"].view(
                target_props.shape[0], config.train_samples_per_target
            )
            per_target_mass = final_terms["occupancy_mass"].view(
                target_props.shape[0], config.train_samples_per_target
            )
            per_target_binarization = binarization_penalty(final_probs).view(
                target_props.shape[0], config.train_samples_per_target
            )
            grouped_final_probs = final_probs.view(
                target_props.shape[0],
                config.train_samples_per_target,
                1,
                config.grid_size,
                config.grid_size,
            )
            per_target_binary_property = binary_property_error.view(
                target_props.shape[0], config.train_samples_per_target
            )
            diversity_penalty = _diversity_penalty(
                grouped_final_probs, config.diversity_scale
            )

            property_loss = (soft_weights * per_target_property).sum(dim=1).mean()
            per_target_improvement = improvement_penalty.view(
                target_props.shape[0], config.train_samples_per_target
            )
            surface = (soft_weights * per_target_surface).sum(dim=1).mean()
            connectivity = (soft_weights * per_target_connectivity).sum(dim=1).mean()
            mass = (soft_weights * per_target_mass).sum(dim=1).mean()
            binarization = (soft_weights * per_target_binarization).sum(dim=1).mean()
            improvement = (soft_weights * per_target_improvement).sum(dim=1).mean()
            diversity = diversity_penalty.mean()
            binary_property_error_mean = (
                (soft_weights * per_target_binary_property).sum(dim=1).mean()
            )
            best_binary_property_error = per_target_binary_property.min(
                dim=1
            ).values.mean()
            candidate_spread = per_target_total.std(dim=1).mean()

            total = total + config.diversity_weight * diversity_penalty.mean()

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            epoch_totals["total"] += total.item()
            epoch_totals["property"] += property_loss.item()
            epoch_totals["surface"] += surface.item()
            epoch_totals["connectivity"] += connectivity.item()
            epoch_totals["mass"] += mass.item()
            epoch_totals["binarization"] += binarization.item()
            epoch_totals["improvement"] += improvement.item()
            epoch_totals["diversity"] += diversity.item()
            epoch_totals["binary_property_error"] += binary_property_error_mean.item()
            epoch_totals["best_binary_property_error"] += (
                best_binary_property_error.item()
            )
            epoch_totals["candidate_spread"] += candidate_spread.item()

            writer.add_scalar("train/total_loss", total.item(), global_step)
            writer.add_scalar("train/property_loss", property_loss.item(), global_step)
            writer.add_scalar("train/surface_loss", surface.item(), global_step)
            writer.add_scalar(
                "train/connectivity_penalty", connectivity.item(), global_step
            )
            writer.add_scalar("train/occupancy_mass", mass.item(), global_step)
            writer.add_scalar(
                "train/binarization_penalty", binarization.item(), global_step
            )
            writer.add_scalar(
                "train/improvement_penalty", improvement.item(), global_step
            )
            writer.add_scalar("train/diversity_penalty", diversity.item(), global_step)
            writer.add_scalar(
                "train/binary_property_error",
                binary_property_error_mean.item(),
                global_step,
            )
            writer.add_scalar(
                "train/best_binary_property_error",
                best_binary_property_error.item(),
                global_step,
            )
            writer.add_scalar(
                "train/candidate_spread", candidate_spread.item(), global_step
            )
            global_step += 1

            if (
                config.canonical_eval_every_steps > 0
                and global_step % config.canonical_eval_every_steps == 0
            ):
                _log_canonical_evaluation(
                    writer, model, config, global_step, device, canonical_specs
                )

            if config.log_every_steps > 0 and (
                batch_idx % config.log_every_steps == 0 or batch_idx == len(loader)
            ):
                _progress(
                    "train:step "
                    f"epoch={epoch + 1}/{config.epochs} batch={batch_idx}/{len(loader)} "
                    f"global_step={global_step} total={total.item():.4f} "
                    f"prop={property_loss.item():.4f} bin_prop={binary_property_error_mean.item():.4f} "
                    f"best_bin={best_binary_property_error.item():.4f} improve={improvement.item():.4f} "
                    f"div={diversity.item():.4f} spread={candidate_spread.item():.4f}"
                )

        num_batches = max(len(loader), 1)
        for name, value in epoch_totals.items():
            writer.add_scalar(f"epoch/{name}", value / num_batches, epoch)

        _progress(
            "train:epoch_end "
            f"epoch={epoch + 1}/{config.epochs} "
            f"total={epoch_totals['total'] / num_batches:.4f} "
            f"prop={epoch_totals['property'] / num_batches:.4f} "
            f"bin_prop={epoch_totals['binary_property_error'] / num_batches:.4f} "
            f"improve={epoch_totals['improvement'] / num_batches:.4f} "
            f"div={epoch_totals['diversity'] / num_batches:.4f} "
            f"best_bin={epoch_totals['best_binary_property_error'] / num_batches:.4f} "
            f"spread={epoch_totals['candidate_spread'] / num_batches:.4f}"
        )

        model.eval()
        with torch.no_grad():
            preview_targets = targets[:8].to(device)
            preview_noise = _fixed_noise(
                preview_targets.shape[0], config.grid_size, config.seed + 202, device
            )
            preview_probs = sample_from_model(
                model,
                preview_targets,
                preview_noise,
                steps=config.rollout_steps,
                step_size=config.rollout_step_size,
            )
            preview_binary = binarize_design(preview_probs)
            preview_terms = mechanical_terms(preview_binary)
            property_error = F.mse_loss(preview_terms["properties"], preview_targets)
            _write_samples(writer, "samples/generated", preview_binary.cpu(), epoch)
            writer.add_scalar("samples/property_error", property_error.item(), epoch)
            writer.add_scalar(
                "samples/kx_mean",
                preview_terms["properties"][:, 0].mean().item(),
                epoch,
            )
            writer.add_scalar(
                "samples/ky_mean",
                preview_terms["properties"][:, 1].mean().item(),
                epoch,
            )
            writer.add_scalar(
                "samples/ktheta_mean",
                preview_terms["properties"][:, 2].mean().item(),
                epoch,
            )

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, checkpoint_path)
    writer.close()
    _progress(f"train:done checkpoint={checkpoint_path} log_dir={log_dir}")
    return checkpoint_path, log_dir


def rollout_model(
    model: ConditionedDenoiser,
    target_props: torch.Tensor,
    initial_noise: torch.Tensor,
    steps: int,
    step_size: float,
    noise_scale: float = 0.0,
) -> list[torch.Tensor]:
    current_logits = torch.logit(initial_noise.clamp(1e-4, 1.0 - 1e-4))
    states: list[torch.Tensor] = []

    for step_idx in range(steps):
        current_probs = _force_plates(torch.sigmoid(current_logits))
        timestep = torch.full(
            (initial_noise.shape[0],),
            (step_idx + 1) / steps,
            device=initial_noise.device,
        )
        delta_logits = model(current_probs, target_props, timestep)
        if noise_scale > 0.0:
            step_noise_scale = noise_scale * (1.0 - step_idx / max(steps - 1, 1))
            delta_logits = delta_logits + step_noise_scale * torch.randn_like(
                delta_logits
            )
        current_logits = current_logits + step_size * delta_logits
        states.append(_force_plates(torch.sigmoid(current_logits)))

    return states


def sample_from_model(
    model: ConditionedDenoiser,
    target_props: torch.Tensor,
    initial_noise: torch.Tensor,
    steps: int,
    step_size: float = 1.0,
) -> torch.Tensor:
    return rollout_model(
        model,
        target_props,
        initial_noise,
        steps=steps,
        step_size=step_size,
    )[-1]


def binarize_design(probs: torch.Tensor) -> torch.Tensor:
    return _force_plates(threshold_occupancy(probs))


def candidate_score(design: torch.Tensor, target_props: torch.Tensor) -> float:
    score, _, _ = candidate_scores(design, target_props)
    return score.mean().item()


def search_elite_batch(
    model: ConditionedDenoiser,
    target_props: torch.Tensor,
    grid_size: int,
    model_candidates: int,
    random_candidates: int,
    steps: int,
    step_size: float,
) -> dict[str, torch.Tensor | list[str]]:
    if model_candidates <= 0 and random_candidates <= 0:
        raise ValueError("at least one candidate source must be enabled")

    device = target_props.device
    was_training = model.training
    model.eval()
    elite_designs: list[torch.Tensor] = []
    elite_reference_probs: list[torch.Tensor] = []
    elite_sources: list[str] = []

    with torch.no_grad():
        for index in range(target_props.shape[0]):
            target = target_props[index : index + 1]
            candidate_designs: list[torch.Tensor] = []
            candidate_references: list[torch.Tensor] = []
            candidate_sources: list[str] = []
            candidate_scores_list: list[float] = []

            if model_candidates > 0:
                noises = torch.rand(
                    (model_candidates, 1, grid_size, grid_size), device=device
                )
                repeated_target = target.repeat(model_candidates, 1)
                model_probs = sample_from_model(
                    model,
                    repeated_target,
                    noises,
                    steps=steps,
                    step_size=step_size,
                )
                model_designs = binarize_design(model_probs)
                scores, _, _ = candidate_scores(model_designs, repeated_target)
                for candidate_idx in range(model_candidates):
                    candidate_designs.append(
                        model_designs[candidate_idx : candidate_idx + 1]
                    )
                    candidate_references.append(
                        model_probs[candidate_idx : candidate_idx + 1]
                    )
                    candidate_sources.append("model")
                    candidate_scores_list.append(scores[candidate_idx].item())

            if random_candidates > 0:
                random_probs = torch.stack(
                    [generate_design(grid_size) for _ in range(random_candidates)],
                    dim=0,
                ).unsqueeze(1)
                random_probs = _force_plates(random_probs.to(device))
                random_designs = binarize_design(random_probs)
                repeated_target = target.repeat(random_candidates, 1)
                scores, _, _ = candidate_scores(random_designs, repeated_target)
                for candidate_idx in range(random_candidates):
                    candidate_designs.append(
                        random_designs[candidate_idx : candidate_idx + 1]
                    )
                    candidate_references.append(
                        random_probs[candidate_idx : candidate_idx + 1]
                    )
                    candidate_sources.append("random")
                    candidate_scores_list.append(scores[candidate_idx].item())

            best_index = min(
                range(len(candidate_scores_list)),
                key=lambda idx: candidate_scores_list[idx],
            )
            elite_designs.append(candidate_designs[best_index])
            elite_reference_probs.append(candidate_references[best_index])
            elite_sources.append(candidate_sources[best_index])

    if was_training:
        model.train()

    stacked_designs = torch.cat(elite_designs, dim=0)
    stacked_reference_probs = torch.cat(elite_reference_probs, dim=0)
    scores, property_error, terms = candidate_scores(stacked_designs, target_props)
    return {
        "designs": stacked_designs,
        "reference_probs": stacked_reference_probs,
        "scores": scores,
        "property_error": property_error,
        "terms": terms,
        "sources": elite_sources,
    }


def _proposal_coordinates(
    design: torch.Tensor, reference_probs: torch.Tensor, proposal_count: int
) -> list[tuple[int, int]]:
    height = design.shape[-2]
    width = design.shape[-1]
    uncertainty = (reference_probs[0, 0, 1:-1, :] - 0.5).abs().flatten().cpu()
    order = torch.argsort(uncertainty)

    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for flat_index in order.tolist():
        row = flat_index // width + 1
        col = flat_index % width
        coord = (row, col)
        if coord in seen:
            continue
        seen.add(coord)
        coords.append(coord)
        if len(coords) >= proposal_count:
            return coords

    random_order = torch.randperm((height - 2) * width)
    for flat_index in random_order.tolist():
        row = flat_index // width + 1
        col = flat_index % width
        coord = (row, col)
        if coord in seen:
            continue
        seen.add(coord)
        coords.append(coord)
        if len(coords) >= proposal_count:
            break
    return coords


def local_search_refine(
    initial_design: torch.Tensor,
    target_props: torch.Tensor,
    reference_probs: torch.Tensor,
    iterations: int,
    proposal_count: int,
    log_every: int = 0,
) -> torch.Tensor:
    current = initial_design.clone()
    current_score = candidate_score(current, target_props)

    for iteration in range(iterations):
        best_candidate = None
        best_score = current_score
        for row, col in _proposal_coordinates(current, reference_probs, proposal_count):
            candidate = current.clone()
            candidate[0, 0, row, col] = 1.0 - candidate[0, 0, row, col]
            candidate = _force_plates(candidate)
            score = candidate_score(candidate, target_props)
            if score < best_score - 1e-6:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            if log_every > 0:
                _progress(
                    "sample:local_search_stop "
                    f"iteration={iteration + 1}/{iterations} score={current_score:.4f}"
                )
            break

        current = best_candidate
        current_score = best_score

        if log_every > 0 and (
            (iteration + 1) % log_every == 0 or iteration + 1 == iterations
        ):
            _progress(
                "sample:local_search "
                f"iteration={iteration + 1}/{iterations} score={current_score:.4f}"
            )

    return current


def sample(
    checkpoint_path: str,
    target_props: list[float],
    steps: int,
    name: str,
    output_path: str,
    model_candidates: int,
    random_candidates: int,
    search_iterations: int,
    proposal_count: int,
    log_every_steps: int,
) -> dict[str, object]:
    device = _device()
    _progress(
        "sample:start "
        f"target={target_props} steps={steps} model_candidates={model_candidates} "
        f"random_candidates={random_candidates} search_iterations={search_iterations}"
    )
    payload = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**payload["config"])
    _seed_everything(config.seed)

    model = ConditionedDenoiser(
        grid_size=config.grid_size,
        patch_size=config.patch_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    target = torch.tensor([target_props], dtype=torch.float32, device=device)
    with torch.no_grad():
        elite_search = search_elite_batch(
            model=model,
            target_props=target,
            grid_size=config.grid_size,
            model_candidates=model_candidates,
            random_candidates=random_candidates,
            steps=steps,
            step_size=config.rollout_step_size,
        )
    _progress(
        "sample:elite "
        f"source={elite_search['sources'][0]} score={elite_search['scores'][0].item():.4f}"
    )

    design = local_search_refine(
        elite_search["designs"],
        target,
        elite_search["reference_probs"],
        iterations=search_iterations,
        proposal_count=proposal_count,
        log_every=log_every_steps,
    )
    with torch.no_grad():
        terms = mechanical_terms(design)

    log_path = _timestamped_run_dir(name)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))
    _write_samples(writer, "sample/final_design", design.cpu(), 0)
    _write_samples(
        writer, "sample/best_candidate_probs", elite_search["reference_probs"].cpu(), 0
    )
    writer.add_scalar("sample/elite_score", elite_search["scores"][0].item(), 0)
    writer.add_scalar("sample/target_kx", target[0, 0].item(), 0)
    writer.add_scalar("sample/target_ky", target[0, 1].item(), 0)
    writer.add_scalar("sample/target_ktheta", target[0, 2].item(), 0)
    writer.add_scalar("sample/achieved_kx", terms["properties"][0, 0].item(), 0)
    writer.add_scalar("sample/achieved_ky", terms["properties"][0, 1].item(), 0)
    writer.add_scalar("sample/achieved_ktheta", terms["properties"][0, 2].item(), 0)
    writer.close()

    result = {
        "design": design.cpu(),
        "properties": terms["properties"].cpu(),
        "surface": terms["surface"].cpu(),
        "connectivity_penalty": terms["connectivity_penalty"].cpu(),
        "source": elite_search["sources"][0],
        "log_dir": str(log_path),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    _progress(
        "sample:done "
        f"source={result['source']} kx={terms['properties'][0, 0].item():.3f} "
        f"ky={terms['properties'][0, 1].item():.3f} "
        f"ktheta={terms['properties'][0, 2].item():.3f} log_dir={log_path}"
    )
    return result


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the first compliant mechanism prototype"
    )
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--rollout-step-size", type=float, default=0.5)
    parser.add_argument("--rollout-noise-scale", type=float, default=0.03)
    parser.add_argument("--property-weight", type=float, default=2.0)
    parser.add_argument("--improvement-weight", type=float, default=0.25)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--diversity-scale", type=float, default=0.15)
    parser.add_argument("--surface-weight", type=float, default=0.02)
    parser.add_argument("--connectivity-weight", type=float, default=0.12)
    parser.add_argument("--mass-weight", type=float, default=0.15)
    parser.add_argument("--binarization-weight", type=float, default=0.1)
    parser.add_argument("--train-samples-per-target", type=int, default=4)
    parser.add_argument("--train-softmin-temperature", type=float, default=0.25)
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--canonical-eval-every-steps", type=int, default=20)
    parser.add_argument("--name", default="prototype")
    parser.add_argument("--checkpoint-path", default="artifacts/prototype.pt")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _sample_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a design from a trained prototype"
    )
    parser.add_argument("--checkpoint-path", default="artifacts/prototype.pt")
    parser.add_argument("--target-kx", type=float, required=True)
    parser.add_argument("--target-ky", type=float, required=True)
    parser.add_argument("--target-ktheta", type=float, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--model-candidates", type=int, default=3)
    parser.add_argument("--random-candidates", type=int, default=10)
    parser.add_argument("--search-iterations", type=int, default=10)
    parser.add_argument("--proposal-count", type=int, default=12)
    parser.add_argument("--log-every-steps", type=int, default=2)
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
        target_props=[args.target_kx, args.target_ky, args.target_ktheta],
        steps=args.steps,
        name=args.name,
        output_path=args.output_path,
        model_candidates=args.model_candidates,
        random_candidates=args.random_candidates,
        search_iterations=args.search_iterations,
        proposal_count=args.proposal_count,
        log_every_steps=args.log_every_steps,
    )
    print(f"log_dir={result['log_dir']}")
    properties = result["properties"][0].tolist()
    print(
        "achieved_properties="
        f"kx:{properties[0]:.3f},ky:{properties[1]:.3f},ktheta:{properties[2]:.3f}"
    )
