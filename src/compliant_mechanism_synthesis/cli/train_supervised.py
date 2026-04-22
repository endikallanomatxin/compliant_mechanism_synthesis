from __future__ import annotations

import argparse

from compliant_mechanism_synthesis.utils import timestamped_run_dir
from compliant_mechanism_synthesis.training import (
    SupervisedTrainingConfig,
    run_supervised_training,
)


def _build_parser() -> argparse.ArgumentParser:
    defaults = SupervisedTrainingConfig(dataset_path="")
    parser = argparse.ArgumentParser(
        prog="cms-train-supervised",
        description="Train the supervised refinement model with flow matching over the offline dataset.",
    )
    parser.set_defaults(use_style_token=defaults.use_style_token)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--log-every-steps", type=int, default=defaults.log_every_steps)
    parser.add_argument(
        "--eval-every-steps", type=int, default=defaults.eval_every_steps
    )
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument(
        "--min-learning-rate", type=float, default=defaults.min_learning_rate
    )
    parser.add_argument("--eval-fraction", type=float, default=defaults.eval_fraction)
    parser.add_argument(
        "--stiffness-loss-weight",
        type=float,
        default=defaults.stiffness_loss_weight,
    )
    parser.add_argument(
        "--stiffness-loss-delay-steps",
        type=int,
        default=defaults.stiffness_loss_delay_steps,
    )
    parser.add_argument(
        "--stiffness-loss-warmup-steps",
        type=int,
        default=defaults.stiffness_loss_warmup_steps,
    )
    parser.add_argument(
        "--stress-loss-weight",
        type=float,
        default=defaults.stress_loss_weight,
    )
    parser.add_argument(
        "--allowable-von-mises",
        type=float,
        default=defaults.allowable_von_mises,
    )
    parser.add_argument(
        "--stress-activation-threshold",
        type=float,
        default=defaults.stress_activation_threshold,
    )
    parser.add_argument(
        "--stress-loss-delay-steps",
        type=int,
        default=defaults.stress_loss_delay_steps,
    )
    parser.add_argument(
        "--stress-loss-warmup-steps",
        type=int,
        default=defaults.stress_loss_warmup_steps,
    )
    parser.add_argument(
        "--style-sample-dropout",
        type=float,
        default=defaults.style_sample_dropout,
    )
    parser.add_argument(
        "--style-token-dropout",
        type=float,
        default=defaults.style_token_dropout,
    )
    parser.add_argument(
        "--no-style-token",
        dest="use_style_token",
        action="store_false",
        help="Disable oracle style-token conditioning during supervised training.",
    )
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path)
    parser.add_argument("--logdir", default=defaults.logdir)
    parser.add_argument("--name", "-n", default="supervised")
    return parser


def train_supervised_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logdir_path = timestamped_run_dir(args.logdir, args.name)
    run_supervised_training(
        SupervisedTrainingConfig(
            dataset_path=args.dataset_path,
            device=args.device,
            batch_size=args.batch_size,
            log_every_steps=args.log_every_steps,
            eval_every_steps=args.eval_every_steps,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_learning_rate=args.min_learning_rate,
            eval_fraction=args.eval_fraction,
            stiffness_loss_weight=args.stiffness_loss_weight,
            stiffness_loss_delay_steps=args.stiffness_loss_delay_steps,
            stiffness_loss_warmup_steps=args.stiffness_loss_warmup_steps,
            stress_loss_weight=args.stress_loss_weight,
            allowable_von_mises=args.allowable_von_mises,
            stress_activation_threshold=args.stress_activation_threshold,
            stress_loss_delay_steps=args.stress_loss_delay_steps,
            stress_loss_warmup_steps=args.stress_loss_warmup_steps,
            use_style_token=args.use_style_token,
            style_sample_dropout=args.style_sample_dropout,
            style_token_dropout=args.style_token_dropout,
            checkpoint_path=args.checkpoint_path,
            logdir=str(logdir_path),
        )
    )
    print(f"logs={logdir_path}")
