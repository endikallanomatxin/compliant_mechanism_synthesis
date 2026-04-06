from __future__ import annotations

import argparse

from compliant_mechanism_synthesis.utils import timestamped_run_dir
from compliant_mechanism_synthesis.training import (
    SupervisedTrainingConfig,
    run_supervised_training,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-train-supervised",
        description="Train the supervised refinement model with flow matching over the offline dataset.",
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-path", default="artifacts/supervised_refiner.pt")
    parser.add_argument("--logdir", default="runs/supervised")
    parser.add_argument("--name", "-n", default="supervised")
    return parser


def train_supervised_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logdir_path = timestamped_run_dir(args.logdir, args.name)
    run_supervised_training(
        SupervisedTrainingConfig(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            checkpoint_path=args.checkpoint_path,
            logdir=str(logdir_path),
        )
    )
    print(f"logs={logdir_path}")
