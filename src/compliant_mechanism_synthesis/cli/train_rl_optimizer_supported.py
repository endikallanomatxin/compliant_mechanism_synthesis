from __future__ import annotations

import argparse

from compliant_mechanism_synthesis.training import (
    ExploreOptimizeTrainingConfig,
    run_explore_optimize_training,
)
from compliant_mechanism_synthesis.utils import timestamped_run_dir


def _build_parser() -> argparse.ArgumentParser:
    defaults = ExploreOptimizeTrainingConfig(dataset_path="")
    parser = argparse.ArgumentParser(
        prog="cms-train-rl-optimizer-supported",
        description=(
            "Train a hybrid explore/optimize refiner that interleaves model proposal "
            "steps with short differential optimization refinements."
        ),
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--log-every-steps", type=int, default=defaults.log_every_steps)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--explore-steps", type=int, default=defaults.explore_steps)
    parser.add_argument("--optimize-steps", type=int, default=defaults.optimize_steps)
    parser.add_argument(
        "--optimize-learning-rate",
        type=float,
        default=defaults.optimize_learning_rate,
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument(
        "--min-learning-rate", type=float, default=defaults.min_learning_rate
    )
    parser.add_argument("--loss-scale", type=float, default=defaults.loss_scale)
    parser.add_argument("--init-checkpoint-path", default=defaults.init_checkpoint_path)
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path)
    parser.add_argument("--logdir", default=defaults.logdir)
    parser.add_argument("--name", "-n", default="rl-optimizer-supported")
    return parser


def train_rl_optimizer_supported_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logdir_path = timestamped_run_dir(args.logdir, args.name)
    run_explore_optimize_training(
        ExploreOptimizeTrainingConfig(
            dataset_path=args.dataset_path,
            device=args.device,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_every_steps=args.log_every_steps,
            num_steps=args.num_steps,
            explore_steps=args.explore_steps,
            optimize_steps=args.optimize_steps,
            optimize_learning_rate=args.optimize_learning_rate,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_learning_rate=args.min_learning_rate,
            loss_scale=args.loss_scale,
            init_checkpoint_path=args.init_checkpoint_path,
            checkpoint_path=args.checkpoint_path,
            logdir=str(logdir_path),
        )
    )
    print(f"logs={logdir_path}")
