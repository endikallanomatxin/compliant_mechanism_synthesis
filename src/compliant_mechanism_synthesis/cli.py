from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
    optimize_case,
    sample_primitive_design,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.visualization import plot_design_3d


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-train",
        description=(
            "Generate and refine an offline dataset of 3D compliant-mechanism "
            "cases. This is the first stage of the refactor and replaces the old RL-first loop."
        ),
    )
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--num-free-nodes", type=int, default=18)
    parser.add_argument("--optimization-steps", type=int, default=120)
    parser.add_argument("--output-path", default="artifacts/offline_dataset.pt")
    parser.add_argument("--logdir", default="runs/offline_dataset")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _build_sample_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-sample",
        description=(
            "Inspect and optimize a single starting point from the offline case generator."
        ),
    )
    parser.add_argument("--primitive", default="curved_lattice_sheet")
    parser.add_argument("--num-free-nodes", type=int, default=18)
    parser.add_argument("--optimization-steps", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/sample_case")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def train_main(argv: list[str] | None = None) -> None:
    parser = _build_train_parser()
    args = parser.parse_args(argv)
    config = OfflineDatasetConfig(
        num_cases=args.num_cases,
        seed=args.seed,
        output_path=args.output_path,
        logdir=args.logdir,
        primitive=PrimitiveConfig(num_free_nodes=args.num_free_nodes),
        optimization=CaseOptimizationConfig(num_steps=args.optimization_steps),
    )
    generate_offline_dataset(config)


def sample_main(argv: list[str] | None = None) -> None:
    parser = _build_sample_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive = PrimitiveConfig(num_free_nodes=args.num_free_nodes)
    optimization = CaseOptimizationConfig(num_steps=args.optimization_steps)
    initial_design = sample_primitive_design(args.primitive, config=primitive, seed=args.seed)
    target = sample_target_stiffness(initial_design, config=optimization, seed=args.seed + 1)
    result = optimize_case(
        primitive_kind=args.primitive,
        initial_design=initial_design,
        target_stiffness=target,
        config=optimization,
        logdir=output_dir / "tensorboard",
    )

    initial_figure = plot_design_3d(
        result.initial_design.positions,
        result.initial_design.roles,
        result.initial_design.adjacency,
        title=f"initial-{args.primitive}",
    )
    optimized_figure = plot_design_3d(
        result.optimized_design.positions,
        result.optimized_design.roles,
        result.optimized_design.adjacency,
        title=f"optimized-{args.primitive}",
    )
    initial_figure.savefig(output_dir / "initial.png", dpi=160, bbox_inches="tight")
    optimized_figure.savefig(output_dir / "optimized.png", dpi=160, bbox_inches="tight")

    print(f"primitive={result.primitive_kind}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"best_loss={result.best_loss:.6f}")
