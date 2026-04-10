from __future__ import annotations

import argparse
from pathlib import Path

from compliant_mechanism_synthesis.utils import (
    resolve_torch_device,
    timestamped_run_dir,
)
from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
    optimize_cases,
    optimize_scaffolds,
    sample_random_primitive,
)
from compliant_mechanism_synthesis.visualization import plot_design_3d


def _build_parser() -> argparse.ArgumentParser:
    dataset_defaults = OfflineDatasetConfig()
    primitive_defaults = PrimitiveConfig()
    optimization_defaults = CaseOptimizationConfig()
    parser = argparse.ArgumentParser(
        prog="cms-dataset-generate",
        description="Generate and refine an offline dataset of 3D compliant-mechanism cases.",
    )
    parser.add_argument("--num-cases", type=int, default=dataset_defaults.num_cases)
    parser.add_argument("--batch-size", type=int, default=dataset_defaults.batch_size)
    parser.add_argument("--device", default=dataset_defaults.device)
    parser.add_argument(
        "--num-free-nodes", type=int, default=primitive_defaults.num_free_nodes
    )
    parser.add_argument(
        "--scaffold-optimization-steps",
        type=int,
        default=optimization_defaults.scaffold_num_steps,
    )
    parser.add_argument(
        "--optimization-steps", type=int, default=optimization_defaults.num_steps
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--logdir", default="datasets")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument(
        "--preview-case-number", type=int, default=dataset_defaults.preview_case_number
    )
    parser.add_argument("--name", "-n", default="generated_dataset")
    parser.add_argument("--seed", type=int, default=dataset_defaults.seed)
    parser.add_argument("--just-check-sample", action="store_true")
    parser.add_argument(
        "--sample-num-free-nodes", type=int, default=primitive_defaults.num_free_nodes
    )
    parser.add_argument(
        "--sample-scaffold-optimization-steps",
        type=int,
        default=optimization_defaults.scaffold_num_steps,
    )
    parser.add_argument(
        "--sample-optimization-steps", type=int, default=optimization_defaults.num_steps
    )
    parser.add_argument("--sample-output-dir", default="artifacts/sample_case")
    parser.add_argument("--sample-seed", type=int, default=dataset_defaults.seed)
    return parser


def dataset_generate_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.just_check_sample:
        _run_sample_check(args)
        return

    logdir_path = timestamped_run_dir(args.logdir, args.name)
    resolved_output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else logdir_path / "dataset.pt"
    )
    preview_path = (
        args.preview_dir
        if args.preview_dir is not None
        else str(logdir_path / "preview")
    )
    config = OfflineDatasetConfig(
        num_cases=args.num_cases,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        output_path=str(resolved_output_path),
        logdir=str(logdir_path),
        preview_dir=preview_path,
        preview_case_number=args.preview_case_number,
        primitive=PrimitiveConfig(num_free_nodes=args.num_free_nodes),
        optimization=CaseOptimizationConfig(
            scaffold_num_steps=args.scaffold_optimization_steps,
            num_steps=args.optimization_steps,
        ),
    )
    generate_offline_dataset(config)
    print(f"dataset={resolved_output_path}")
    print(f"visualizations={Path(preview_path)}")
    print(f"logs={logdir_path}")


def _run_sample_check(args: argparse.Namespace) -> None:
    device = resolve_torch_device(args.device)
    output_dir = Path(args.sample_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive_cfg = PrimitiveConfig(num_free_nodes=args.sample_num_free_nodes)
    initial_structures, initial_scaffold = sample_random_primitive(
        config=primitive_cfg,
        seed=args.sample_seed,
    )
    initial_structures = initial_structures.to(device)
    initial_scaffold = initial_scaffold.to(device)
    if args.sample_scaffold_optimization_steps > 0:
        _, scaffold_structures = optimize_scaffolds(
            scaffolds=initial_scaffold,
            primitive_config=primitive_cfg,
            config=CaseOptimizationConfig(
                scaffold_num_steps=args.sample_scaffold_optimization_steps,
                num_steps=args.sample_optimization_steps,
            ),
            logdir=output_dir / "tensorboard_scaffold",
        )
    else:
        scaffold_structures = initial_structures
    result = optimize_cases(
        structures=scaffold_structures,
        config=CaseOptimizationConfig(
            scaffold_num_steps=args.sample_scaffold_optimization_steps,
            num_steps=args.sample_optimization_steps,
        ),
        logdir=output_dir / "tensorboard_cases",
    )
    _dump_sample_figures(initial_structures, result, output_dir)
    print(f"initial_loss={float(result.initial_loss[0].item()):.6f}")
    print(f"best_loss={float(result.best_loss[0].item()):.6f}")


def _dump_sample_figures(
    initial_structures,
    result,
    output_dir: Path,
) -> None:
    initial_figure = plot_design_3d(
        initial_structures.positions[0],
        initial_structures.roles[0],
        initial_structures.adjacency[0],
        title="initial",
    )
    optimized_figure = plot_design_3d(
        result.optimized_structures.positions[0],
        result.optimized_structures.roles[0],
        result.optimized_structures.adjacency[0],
        title="optimized",
    )
    initial_figure.savefig(output_dir / "initial.png", dpi=160, bbox_inches="tight")
    optimized_figure.savefig(output_dir / "optimized.png", dpi=160, bbox_inches="tight")
