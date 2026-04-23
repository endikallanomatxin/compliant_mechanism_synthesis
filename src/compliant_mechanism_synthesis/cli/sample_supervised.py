from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.training import load_training_cases
from compliant_mechanism_synthesis.visualization.supervised import (
    write_supervised_sampling_visualizations,
)


def _parse_case_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-sample",
        description=(
            "Render supervised-refiner samples from training-style noise, with oracle overlays "
            "and rollout GIFs."
        ),
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--case-indices", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser


def sample_supervised_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset_path = Path(args.dataset_path)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else checkpoint_path.parent
        / f"{checkpoint_path.stem}_{dataset_path.stem}_samples"
    )
    optimized_cases = load_training_cases(str(dataset_path))
    write_supervised_sampling_visualizations(
        optimized_cases=optimized_cases,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=args.device,
        max_cases=args.max_cases,
        case_indices=_parse_case_indices(args.case_indices),
        seed=args.seed,
        num_steps=args.num_steps,
        threshold=args.threshold,
    )
    print(f"dataset={dataset_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"samples={output_dir}")
