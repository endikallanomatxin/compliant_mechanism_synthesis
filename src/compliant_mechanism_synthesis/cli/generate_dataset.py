from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.cli.common import timestamped_run_dir
from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-generate-dataset",
        description="Generate and refine an offline dataset of 3D compliant-mechanism cases.",
    )
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--num-free-nodes", type=int, default=18)
    parser.add_argument("--optimization-steps", type=int, default=120)
    parser.add_argument("--output-path", default="artifacts/offline_dataset.pt")
    parser.add_argument("--logdir", default="runs/offline_dataset")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--preview-cases", type=int, default=6)
    parser.add_argument("--name", "-n", default="generated_dataset")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def generate_dataset_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logdir_path = timestamped_run_dir(args.logdir, args.name)
    preview_path = (
        args.preview_dir
        if args.preview_dir is not None
        else str(logdir_path / "preview")
    )
    config = OfflineDatasetConfig(
        num_cases=args.num_cases,
        seed=args.seed,
        output_path=args.output_path,
        logdir=str(logdir_path),
        preview_dir=preview_path,
        preview_cases=args.preview_cases,
        primitive=PrimitiveConfig(num_free_nodes=args.num_free_nodes),
        optimization=CaseOptimizationConfig(num_steps=args.optimization_steps),
    )
    generate_offline_dataset(config)
    print(f"dataset={args.output_path}")
    print(f"visualizations={Path(preview_path)}")
    print(f"logs={logdir_path}")
