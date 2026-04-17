from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from compliant_mechanism_synthesis.models import SupervisedRefiner, SupervisedRefinerConfig
from compliant_mechanism_synthesis.training import SupervisedTrainingConfig


def _backup_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}-original{checkpoint_path.suffix}"
    )


def _merge_model_config(existing: dict[str, object]) -> dict[str, object]:
    merged = asdict(SupervisedRefinerConfig())
    merged.update(existing)
    return merged


def _merge_train_config(existing: dict[str, object]) -> dict[str, object]:
    dataset_path = str(existing.get("dataset_path", ""))
    merged = asdict(SupervisedTrainingConfig(dataset_path=dataset_path))
    merged.update(existing)
    return merged


def upgrade_supervised_checkpoint(checkpoint_path: str | Path) -> tuple[Path, Path]:
    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved_path}")
    backup_path = _backup_path(resolved_path)
    if backup_path.exists():
        raise FileExistsError(f"backup checkpoint already exists: {backup_path}")

    checkpoint = torch.load(resolved_path, map_location="cpu")
    model_config = _merge_model_config(checkpoint["model_config"])
    train_config = _merge_train_config(checkpoint["train_config"])

    model = SupervisedRefiner(SupervisedRefinerConfig(**model_config))
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    allowed_missing = (
        {"style_token_encoder.token_seed"} if model.config.use_style_token else set()
    )
    if missing_keys - allowed_missing:
        raise RuntimeError(
            f"checkpoint is missing unsupported parameters: {sorted(missing_keys)}"
        )
    if unexpected_keys:
        raise RuntimeError(
            f"checkpoint contains unexpected parameters: {sorted(unexpected_keys)}"
        )

    shutil.copy2(resolved_path, backup_path)
    upgraded_checkpoint = dict(checkpoint)
    upgraded_checkpoint["model_state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    upgraded_checkpoint["model_config"] = model_config
    upgraded_checkpoint["train_config"] = train_config
    torch.save(upgraded_checkpoint, resolved_path)
    return resolved_path, backup_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-upgrade-supervised-checkpoint",
        description=(
            "Rewrite an older supervised refiner checkpoint to the current checkpoint "
            "schema, saving a sibling *-original backup first."
        ),
    )
    parser.add_argument("--checkpoint-path", required=True)
    return parser


def upgrade_supervised_checkpoint_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    checkpoint_path, backup_path = upgrade_supervised_checkpoint(args.checkpoint_path)
    print(f"backup={backup_path}")
    print(f"checkpoint={checkpoint_path}")
