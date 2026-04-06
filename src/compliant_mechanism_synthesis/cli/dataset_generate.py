from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.utils import timestamped_run_dir
from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
    optimize_cases,
    sample_primitive_design,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.visualization import plot_design_3d


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cms-dataset-generate",
        description="Generate and refine an offline dataset of 3D compliant-mechanism cases.",
    )
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--num-free-nodes", type=int, default=18)
    parser.add_argument("--optimization-steps", type=int, default=120)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--logdir", default="datasets")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--preview-cases", type=int, default=6)
    parser.add_argument("--name", "-n", default="generated_dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--just-check-sample", action="store_true")
    parser.add_argument("--sample-primitive", default="curved_lattice_sheet")
    parser.add_argument("--sample-num-free-nodes", type=int, default=18)
    parser.add_argument("--sample-optimization-steps", type=int, default=120)
    parser.add_argument("--sample-output-dir", default="artifacts/sample_case")
    parser.add_argument("--sample-seed", type=int, default=7)
    return parser


def dataset_generate_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.just_check_sample:
        _run_sample_check(args)
        return

    logdir_path = timestamped_run_dir(args.logdir, args.name)
    resolved_output_path = Path(args.output_path) if args.output_path is not None else logdir_path / "offline_dataset.pt"
    preview_path = (
        args.preview_dir
        if args.preview_dir is not None
        else str(logdir_path / "preview")
    )
    config = OfflineDatasetConfig(
        num_cases=args.num_cases,
        seed=args.seed,
        output_path=str(resolved_output_path),
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


def _run_sample_check(args: argparse.Namespace) -> None:
    output_dir = Path(args.sample_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive_cfg = PrimitiveConfig(num_free_nodes=args.sample_num_free_nodes)
    optimization = CaseOptimizationConfig(num_steps=args.sample_optimization_steps)
    initial_structures = sample_primitive_design(
        args.sample_primitive, config=primitive_cfg, seed=args.sample_seed
    )
    target = sample_target_stiffness(
        initial_structures, config=optimization, seed=args.sample_seed + 1
    )
    result = optimize_cases(
        structures=initial_structures,
        target_stiffness=target.unsqueeze(0),
        config=optimization,
        logdir=output_dir / "tensorboard_cases",
    )
    _dump_sample_figures(result, output_dir, args.sample_primitive)
    print(f"primitive={args.sample_primitive}")
    print(f"initial_loss={float(result.initial_loss[0].item()):.6f}")
    print(f"best_loss={float(result.best_loss[0].item()):.6f}")


def _dump_sample_figures(result, output_dir: Path, primitive: str) -> None:
    initial_figure = plot_design_3d(
        result.raw_structures.positions[0],
        result.raw_structures.roles[0],
        result.raw_structures.adjacency[0],
        title=f"initial-{primitive}",
    )
    optimized_figure = plot_design_3d(
        result.optimized_structures.positions[0],
        result.optimized_structures.roles[0],
        result.optimized_structures.adjacency[0],
        title=f"optimized-{primitive}",
    )
    initial_figure.savefig(output_dir / "initial.png", dpi=160, bbox_inches="tight")
    optimized_figure.savefig(output_dir / "optimized.png", dpi=160, bbox_inches="tight")
