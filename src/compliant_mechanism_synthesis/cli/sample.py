from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    PrimitiveConfig,
    optimize_cases,
    sample_primitive_design,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.visualization import plot_design_3d


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-sample",
        description="Inspect and optimize a single starting point from the offline case generator.",
    )
    parser.add_argument("--primitive", default="curved_lattice_sheet")
    parser.add_argument("--num-free-nodes", type=int, default=18)
    parser.add_argument("--optimization-steps", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/sample_case")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def sample_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive = PrimitiveConfig(num_free_nodes=args.num_free_nodes)
    optimization = CaseOptimizationConfig(num_steps=args.optimization_steps)
    initial_structures = sample_primitive_design(args.primitive, config=primitive, seed=args.seed)
    target = sample_target_stiffness(initial_structures, config=optimization, seed=args.seed + 1)
    result = optimize_cases(
        structures=initial_structures,
        target_stiffness=target.unsqueeze(0),
        config=optimization,
        logdir=output_dir / "tensorboard_cases",
    )

    initial_figure = plot_design_3d(
        result.raw_structures.positions[0],
        result.raw_structures.roles[0],
        result.raw_structures.adjacency[0],
        title=f"initial-{args.primitive}",
    )
    optimized_figure = plot_design_3d(
        result.optimized_structures.positions[0],
        result.optimized_structures.roles[0],
        result.optimized_structures.adjacency[0],
        title=f"optimized-{args.primitive}",
    )
    initial_figure.savefig(output_dir / "initial.png", dpi=160, bbox_inches="tight")
    optimized_figure.savefig(output_dir / "optimized.png", dpi=160, bbox_inches="tight")

    print(f"primitive={args.primitive}")
    print(f"initial_loss={float(result.initial_loss[0].item()):.6f}")
    print(f"best_loss={float(result.best_loss[0].item()):.6f}")
