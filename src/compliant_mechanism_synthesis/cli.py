from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.data import generate_dataset, generate_design
from compliant_mechanism_synthesis.mechanics import (
    binarization_penalty,
    mechanical_terms,
    threshold_occupancy,
    topology_regularizers,
)
from compliant_mechanism_synthesis.model import ConditionedDenoiser


@dataclass
class TrainConfig:
    grid_size: int = 24
    patch_size: int = 4
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dataset_size: int = 256
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 3e-4
    surface_weight: float = 0.05
    connectivity_weight: float = 0.2
    mass_weight: float = 0.15
    binarization_weight: float = 0.1
    train_model_candidates: int = 2
    train_random_candidates: int = 6
    train_sample_steps: int = 6
    log_dir: str = "runs/prototype"
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


def _add_noise(grids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    noise = torch.rand_like(grids)
    mixing = timesteps[:, None, None, None]
    return grids * (1.0 - mixing) + noise * mixing


def _force_plates(grids: torch.Tensor) -> torch.Tensor:
    grids = grids.clone()
    grids[:, :, 0, :] = 1.0
    grids[:, :, -1, :] = 1.0
    return grids.clamp(0.0, 1.0)


def _write_samples(
    writer: SummaryWriter, tag: str, grids: torch.Tensor, step: int
) -> None:
    writer.add_images(tag, grids[:8], global_step=step, dataformats="NCHW")


def _timestamped_run_dir(log_dir: str) -> Path:
    base = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base.parent / f"{timestamp}-{base.name}"


def candidate_scores(
    designs: torch.Tensor, target_props: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    terms = mechanical_terms(designs)
    property_error = (terms["properties"] - target_props).square().mean(dim=1)
    score = (
        property_error
        + 0.10 * terms["connectivity_penalty"]
        + 0.02 * terms["occupancy_mass"]
        + 0.02 * terms["surface"]
    )
    return score, property_error, terms


def train(config: TrainConfig) -> tuple[Path, Path]:
    _seed_everything(config.seed)
    device = _device()

    designs = generate_dataset(config.dataset_size, config.grid_size, config.seed)
    targets = mechanical_terms(threshold_occupancy(designs))["properties"]
    dataset = TensorDataset(targets)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = ConditionedDenoiser(
        grid_size=config.grid_size,
        patch_size=config.patch_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    reconstruction_loss = nn.BCEWithLogitsLoss()

    log_dir = _timestamped_run_dir(config.log_dir)
    checkpoint_path = Path(config.checkpoint_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_totals = {
            "total": 0.0,
            "recon": 0.0,
            "surface": 0.0,
            "connectivity": 0.0,
            "mass": 0.0,
            "binarization": 0.0,
            "property_error": 0.0,
            "elite_score": 0.0,
            "model_elite_fraction": 0.0,
        }

        for (target_props,) in loader:
            target_props = target_props.to(device)

            with torch.no_grad():
                elite_search = search_elite_batch(
                    model=model,
                    target_props=target_props,
                    grid_size=config.grid_size,
                    model_candidates=config.train_model_candidates,
                    random_candidates=config.train_random_candidates,
                    steps=config.train_sample_steps,
                )
                elite_designs = elite_search["designs"]
                elite_sources = elite_search["sources"]
                elite_scores = elite_search["scores"]

            timesteps = torch.rand(elite_designs.shape[0], device=device)
            noisy_grids = _add_noise(elite_designs, timesteps)

            logits = model(noisy_grids, target_props, timesteps)
            probs = _force_plates(torch.sigmoid(logits))
            regularizers = topology_regularizers(probs)
            binary_predictions = _force_plates(threshold_occupancy(probs))
            _, property_error, _ = candidate_scores(binary_predictions, target_props)

            recon = reconstruction_loss(logits, elite_designs)
            surface = regularizers["surface"].mean()
            connectivity = regularizers["connectivity_penalty"].mean()
            mass = regularizers["occupancy_mass"].mean()
            binarization = binarization_penalty(probs).mean()
            property_error_mean = property_error.mean()
            model_elite_fraction = sum(
                source == "model" for source in elite_sources
            ) / len(elite_sources)
            total = (
                recon
                + config.surface_weight * surface
                + config.connectivity_weight * connectivity
                + config.mass_weight * mass
                + config.binarization_weight * binarization
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            epoch_totals["total"] += total.item()
            epoch_totals["recon"] += recon.item()
            epoch_totals["surface"] += surface.item()
            epoch_totals["connectivity"] += connectivity.item()
            epoch_totals["mass"] += mass.item()
            epoch_totals["binarization"] += binarization.item()
            epoch_totals["property_error"] += property_error_mean.item()
            epoch_totals["elite_score"] += elite_scores.mean().item()
            epoch_totals["model_elite_fraction"] += model_elite_fraction

            writer.add_scalar("train/total_loss", total.item(), global_step)
            writer.add_scalar("train/reconstruction_loss", recon.item(), global_step)
            writer.add_scalar("train/surface_loss", surface.item(), global_step)
            writer.add_scalar(
                "train/connectivity_penalty", connectivity.item(), global_step
            )
            writer.add_scalar("train/occupancy_mass", mass.item(), global_step)
            writer.add_scalar(
                "train/binarization_penalty", binarization.item(), global_step
            )
            writer.add_scalar(
                "train/property_error", property_error_mean.item(), global_step
            )
            writer.add_scalar(
                "train/elite_score", elite_scores.mean().item(), global_step
            )
            writer.add_scalar(
                "train/model_elite_fraction", model_elite_fraction, global_step
            )
            global_step += 1

        num_batches = max(len(loader), 1)
        for name, value in epoch_totals.items():
            writer.add_scalar(f"epoch/{name}", value / num_batches, epoch)

        model.eval()
        with torch.no_grad():
            preview_targets = targets[:8].to(device)
            preview_search = search_elite_batch(
                model=model,
                target_props=preview_targets,
                grid_size=config.grid_size,
                model_candidates=config.train_model_candidates,
                random_candidates=config.train_random_candidates,
                steps=config.train_sample_steps,
            )
            preview_binary = preview_search["designs"]
            preview_terms = preview_search["terms"]
            property_error = preview_search["property_error"].mean()
            _write_samples(writer, "samples/elites", preview_binary.cpu(), epoch)
            _write_samples(writer, "samples/target_pool_reference", designs[:8], epoch)
            writer.add_scalar("samples/property_error", property_error.item(), epoch)
            writer.add_scalar(
                "samples/elite_score", preview_search["scores"].mean().item(), epoch
            )
            writer.add_scalar(
                "samples/model_elite_fraction",
                sum(source == "model" for source in preview_search["sources"])
                / len(preview_search["sources"]),
                epoch,
            )
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
    return checkpoint_path, log_dir


def sample_from_model(
    model: ConditionedDenoiser,
    target_props: torch.Tensor,
    initial_noise: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    current = initial_noise
    device = current.device
    for step in reversed(range(steps)):
        t_value = torch.full((current.shape[0],), (step + 1) / steps, device=device)
        logits = model(current, target_props, t_value)
        probs = _force_plates(torch.sigmoid(logits))
        blend = 0.6 if step > 0 else 1.0
        current = blend * probs + (1.0 - blend) * current
    return _force_plates(current)


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
                    model, repeated_target, noises, steps=steps
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
) -> torch.Tensor:
    current = initial_design.clone()
    current_score = candidate_score(current, target_props)

    for _ in range(iterations):
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
            break

        current = best_candidate
        current_score = best_score

    return current


def sample(
    checkpoint_path: str,
    target_props: list[float],
    steps: int,
    log_dir: str,
    output_path: str,
    model_candidates: int,
    random_candidates: int,
    search_iterations: int,
    proposal_count: int,
) -> dict[str, object]:
    device = _device()
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
        )

    design = local_search_refine(
        elite_search["designs"],
        target,
        elite_search["reference_probs"],
        iterations=search_iterations,
        proposal_count=proposal_count,
    )
    with torch.no_grad():
        terms = mechanical_terms(design)

    log_path = _timestamped_run_dir(log_dir)
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
    return result


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the first compliant mechanism prototype"
    )
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--surface-weight", type=float, default=0.05)
    parser.add_argument("--connectivity-weight", type=float, default=0.2)
    parser.add_argument("--mass-weight", type=float, default=0.15)
    parser.add_argument("--binarization-weight", type=float, default=0.1)
    parser.add_argument("--train-model-candidates", type=int, default=2)
    parser.add_argument("--train-random-candidates", type=int, default=6)
    parser.add_argument("--train-sample-steps", type=int, default=6)
    parser.add_argument("--log-dir", default="runs/prototype")
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
    parser.add_argument("--log-dir", default="runs/sample")
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
        log_dir=args.log_dir,
        output_path=args.output_path,
        model_candidates=args.model_candidates,
        random_candidates=args.random_candidates,
        search_iterations=args.search_iterations,
        proposal_count=args.proposal_count,
    )
    print(f"log_dir={result['log_dir']}")
    properties = result["properties"][0].tolist()
    print(
        "achieved_properties="
        f"kx:{properties[0]:.3f},ky:{properties[1]:.3f},ktheta:{properties[2]:.3f}"
    )
