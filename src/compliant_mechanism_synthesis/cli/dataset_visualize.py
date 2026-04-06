from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.dataset import load_offline_dataset
from compliant_mechanism_synthesis.visualization import write_dataset_visualizations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-dataset-visualize",
        description="Render dataset previews and a summary from an existing offline dataset.",
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-cases", type=int, default=6)
    return parser


def visualize_dataset_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset_path = Path(args.dataset_path)
    optimized_cases, primitive_kinds, _ = load_offline_dataset(dataset_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else dataset_path.parent / f"{dataset_path.stem}_preview"
    )
    write_dataset_visualizations(
        optimized_cases=optimized_cases,
        primitive_kinds=primitive_kinds,
        output_dir=output_dir,
        max_cases=args.max_cases,
    )
    print(f"dataset={dataset_path}")
    print(f"visualizations={output_dir}")
