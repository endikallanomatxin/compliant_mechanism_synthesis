from __future__ import annotations

import argparse

from compliant_mechanism_synthesis.training.supervised import (
    SupervisedTrainingConfig,
    run_supervised_training,
)


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-train",
        description=(
            "Top-level training entrypoint. The refactor now separates offline "
            "dataset generation from supervised training so RL is not coupled "
            "to the first phase anymore."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="artifacts/offline_dataset.pt",
        help="Path to the offline dataset consumed by the supervised stage.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=10_000)
    return parser


def _build_sample_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-sample",
        description=(
            "Inspection entrypoint for single designs and dataset samples. "
            "Sampling is reintroduced after the offline dataset pipeline lands."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="artifacts/offline_dataset.pt",
        help="Path that will later be used to inspect generated offline cases.",
    )
    return parser


def train_main(argv: list[str] | None = None) -> None:
    parser = _build_train_parser()
    args = parser.parse_args(argv)
    config = SupervisedTrainingConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
    )
    run_supervised_training(config)


def sample_main(argv: list[str] | None = None) -> None:
    parser = _build_sample_parser()
    parser.parse_args(argv)
    raise NotImplementedError(
        "Sampling is intentionally paused while the new 3D offline pipeline is "
        "implemented."
    )
