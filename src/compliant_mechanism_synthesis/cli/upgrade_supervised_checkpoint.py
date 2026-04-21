from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)


def _backup_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}-original{checkpoint_path.suffix}"
    )


def upgrade_supervised_checkpoint(checkpoint_path: str | Path) -> tuple[Path, Path]:
    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved_path}")
    backup_path = _backup_path(resolved_path)
    if backup_path.exists():
        raise FileExistsError(f"backup checkpoint already exists: {backup_path}")

    checkpoint = torch.load(resolved_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    train_config = checkpoint["train_config"]

    model = SupervisedRefiner(SupervisedRefinerConfig(**model_config))
    model.load_state_dict(checkpoint["model_state_dict"])

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
