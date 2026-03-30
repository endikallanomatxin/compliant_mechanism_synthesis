from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.data import generate_dataset
from compliant_mechanism_synthesis.mechanics import mechanical_terms
from compliant_mechanism_synthesis.model import ConditionedDenoiser


@dataclass
class TrainConfig:
    grid_size: int = 16
    patch_size: int = 4
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dataset_size: int = 512
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 3e-4
    property_weight: float = 3.0
    surface_weight: float = 0.05
    connectivity_weight: float = 0.2
    mass_weight: float = 0.15
    log_dir: str = "runs/prototype"
    checkpoint_path: str = "artifacts/prototype.pt"
    seed: int = 7


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
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


def train(config: TrainConfig) -> Path:
    _seed_everything(config.seed)
    device = _device()

    designs = generate_dataset(config.dataset_size, config.grid_size, config.seed)
    targets = mechanical_terms(designs)["properties"]
    dataset = TensorDataset(designs, targets)
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

    log_dir = Path(config.log_dir)
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
            "props": 0.0,
            "surface": 0.0,
            "connectivity": 0.0,
            "mass": 0.0,
        }

        for clean_grids, target_props in loader:
            clean_grids = clean_grids.to(device)
            target_props = target_props.to(device)
            timesteps = torch.rand(clean_grids.shape[0], device=device)
            noisy_grids = _add_noise(clean_grids, timesteps)

            logits = model(noisy_grids, target_props, timesteps)
            probs = _force_plates(torch.sigmoid(logits))
            terms = mechanical_terms(probs)

            recon = reconstruction_loss(logits, clean_grids)
            props = nn.functional.mse_loss(terms["properties"], target_props)
            surface = terms["surface"].mean()
            connectivity = terms["connectivity_penalty"].mean()
            mass = terms["occupancy_mass"].mean()
            total = (
                recon
                + config.property_weight * props
                + config.surface_weight * surface
                + config.connectivity_weight * connectivity
                + config.mass_weight * mass
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            epoch_totals["total"] += total.item()
            epoch_totals["recon"] += recon.item()
            epoch_totals["props"] += props.item()
            epoch_totals["surface"] += surface.item()
            epoch_totals["connectivity"] += connectivity.item()
            epoch_totals["mass"] += mass.item()

            writer.add_scalar("train/total_loss", total.item(), global_step)
            writer.add_scalar("train/reconstruction_loss", recon.item(), global_step)
            writer.add_scalar("train/property_loss", props.item(), global_step)
            writer.add_scalar("train/surface_loss", surface.item(), global_step)
            writer.add_scalar(
                "train/connectivity_penalty", connectivity.item(), global_step
            )
            writer.add_scalar("train/occupancy_mass", mass.item(), global_step)
            global_step += 1

        num_batches = max(len(loader), 1)
        for name, value in epoch_totals.items():
            writer.add_scalar(f"epoch/{name}", value / num_batches, epoch)

        model.eval()
        with torch.no_grad():
            preview_targets = targets[:8].to(device)
            preview_noise = torch.rand(
                (preview_targets.shape[0], 1, config.grid_size, config.grid_size),
                device=device,
            )
            sample = sample_from_model(model, preview_targets, preview_noise, steps=8)
            preview_terms = mechanical_terms(sample)
            _write_samples(writer, "samples/generated", sample.cpu(), epoch)
            _write_samples(writer, "samples/reference", designs[:8], epoch)
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
    return checkpoint_path


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
        blend = 0.55 if step > 0 else 1.0
        current = blend * probs + (1.0 - blend) * current
    return _force_plates(current)


def refine_design(
    initial_design: torch.Tensor,
    target_props: torch.Tensor,
    iterations: int = 40,
    lr: float = 0.2,
) -> torch.Tensor:
    logits = nn.Parameter(torch.logit(initial_design.clamp(0.05, 0.95)))
    optimizer = torch.optim.Adam([logits], lr=lr)

    for _ in range(iterations):
        probs = _force_plates(torch.sigmoid(logits))
        terms = mechanical_terms(probs)
        property_loss = nn.functional.mse_loss(terms["properties"], target_props)
        regularization = (
            0.08 * terms["surface"].mean()
            + 0.12 * terms["occupancy_mass"].mean()
            + 0.25 * terms["connectivity_penalty"].mean()
        )
        loss = 6.0 * property_loss + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return _force_plates(torch.sigmoid(logits).detach())


def binarize_design(probs: torch.Tensor, target_props: torch.Tensor) -> torch.Tensor:
    best_design = None
    best_error = None
    for threshold in torch.linspace(0.35, 0.7, steps=8, device=probs.device):
        candidate = _force_plates((probs > threshold).float())
        terms = mechanical_terms(candidate)
        error = nn.functional.mse_loss(terms["properties"], target_props)
        score = error + 0.1 * terms["connectivity_penalty"].mean()
        if best_error is None or score.item() < best_error:
            best_error = score.item()
            best_design = candidate

    if best_design is None:
        raise RuntimeError("failed to binarize sample")

    return best_design


def candidate_score(design: torch.Tensor, target_props: torch.Tensor) -> float:
    terms = mechanical_terms(design)
    score = nn.functional.mse_loss(terms["properties"], target_props)
    score = score + 0.1 * terms["connectivity_penalty"].mean()
    score = score + 0.02 * terms["occupancy_mass"].mean()
    return score.item()


def sample(
    checkpoint_path: str,
    target_props: list[float],
    steps: int,
    log_dir: str,
    output_path: str,
) -> dict[str, torch.Tensor]:
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
    noise = torch.rand((1, 1, config.grid_size, config.grid_size), device=device)

    with torch.no_grad():
        coarse = sample_from_model(model, target, noise, steps=steps)

    initial_candidates = [
        coarse,
        _force_plates(torch.full_like(noise, 0.35)),
        _force_plates(torch.rand_like(noise) * 0.55),
    ]
    refined_candidates = [
        refine_design(candidate, target, iterations=60)
        for candidate in initial_candidates
    ]
    binary_candidates = [
        binarize_design(candidate, target) for candidate in refined_candidates
    ]
    best_index = min(
        range(len(binary_candidates)),
        key=lambda idx: candidate_score(binary_candidates[idx], target),
    )
    refined = refined_candidates[best_index]
    design = binary_candidates[best_index]
    with torch.no_grad():
        terms = mechanical_terms(design)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))
    _write_samples(writer, "sample/final_design", design.cpu(), 0)
    _write_samples(writer, "sample/refined_probs", refined.cpu(), 0)
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
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return result


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the first compliant mechanism prototype"
    )
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--property-weight", type=float, default=3.0)
    parser.add_argument("--surface-weight", type=float, default=0.05)
    parser.add_argument("--connectivity-weight", type=float, default=0.2)
    parser.add_argument("--mass-weight", type=float, default=0.15)
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
    parser.add_argument("--log-dir", default="runs/sample")
    parser.add_argument("--output-path", default="artifacts/sample.pt")
    return parser


def train_main() -> None:
    args = _train_parser().parse_args()
    checkpoint_path = train(TrainConfig(**vars(args)))
    print(f"checkpoint={checkpoint_path}")


def sample_main() -> None:
    args = _sample_parser().parse_args()
    result = sample(
        checkpoint_path=args.checkpoint_path,
        target_props=[args.target_kx, args.target_ky, args.target_ktheta],
        steps=args.steps,
        log_dir=args.log_dir,
        output_path=args.output_path,
    )
    properties = result["properties"][0].tolist()
    print(
        "achieved_properties="
        f"kx:{properties[0]:.3f},ky:{properties[1]:.3f},ktheta:{properties[2]:.3f}"
    )
