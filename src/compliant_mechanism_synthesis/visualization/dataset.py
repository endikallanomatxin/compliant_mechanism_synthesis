from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from compliant_mechanism_synthesis.dataset.primitives import CHAIN_PRIMITIVE_LIBRARY
from compliant_mechanism_synthesis.dataset.types import OptimizedCases
from compliant_mechanism_synthesis.visualization.plots import (
    plot_design_3d,
    plot_scaffold_primitives_3d,
)


def _loss_improvement(
    initial_loss: torch.Tensor, best_loss: torch.Tensor
) -> torch.Tensor:
    safe_scale = initial_loss.abs().clamp_min(1e-6)
    return (initial_loss - best_loss) / safe_scale


def write_dataset_visualizations(
    optimized_cases: OptimizedCases,
    output_dir: str | Path,
    max_cases: int = 6,
    case_indices: list[int] | None = None,
) -> Path:
    optimized_cases.validate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    initial_positions = optimized_cases.raw_structures.positions
    initial_roles = optimized_cases.raw_structures.roles
    initial_adjacency = optimized_cases.raw_structures.adjacency
    optimized_positions = optimized_cases.optimized_structures.positions
    optimized_roles = optimized_cases.optimized_structures.roles
    optimized_adjacency = optimized_cases.optimized_structures.adjacency
    scaffolds = optimized_cases.scaffolds
    initial_loss = optimized_cases.initial_loss
    best_loss = optimized_cases.best_loss
    target_stiffness = optimized_cases.target_stiffness
    if scaffolds is None:
        raise ValueError("dataset visualizations require scaffold metadata")

    improvement = _loss_improvement(initial_loss, best_loss)
    total_cases = int(initial_positions.shape[0])
    if case_indices is None:
        case_indices = list(range(min(total_cases, max_cases)))
    else:
        case_indices = list(case_indices)
        for case_index in case_indices:
            if case_index < 0 or case_index >= total_cases:
                raise ValueError("case_indices must refer to valid dataset cases")
    case_count = len(case_indices)
    summary_lines = [
        f"cases={total_cases}",
        f"preview_cases={case_count}",
        f"preview_case_indices={','.join(str(index) for index in case_indices)}",
        f"mean_initial_loss={float(initial_loss.mean().item()):.6f}",
        f"mean_best_loss={float(best_loss.mean().item()):.6f}",
        f"mean_relative_improvement={float(improvement.mean().item()):.6f}",
    ]

    for case_index in case_indices:
        title_prefix = f"case_{case_index:04d}"
        primitives_figure = plot_scaffold_primitives_3d(
            scaffolds.positions[case_index],
            scaffolds.roles[case_index],
            scaffolds.adjacency[case_index],
            scaffolds.edge_primitive_types[case_index],
            primitive_labels=CHAIN_PRIMITIVE_LIBRARY,
            title=f"{title_prefix}_primitives",
        )
        initial_figure = plot_design_3d(
            initial_positions[case_index],
            initial_roles[case_index],
            initial_adjacency[case_index],
            title=f"{title_prefix}_initial",
        )
        optimized_figure = plot_design_3d(
            optimized_positions[case_index],
            optimized_roles[case_index],
            optimized_adjacency[case_index],
            title=f"{title_prefix}_optimized",
        )
        primitives_figure.savefig(
            output_path / f"{title_prefix}_primitives.png", dpi=160, bbox_inches="tight"
        )
        initial_figure.savefig(
            output_path / f"{title_prefix}_initial.png", dpi=160, bbox_inches="tight"
        )
        optimized_figure.savefig(
            output_path / f"{title_prefix}_optimized.png", dpi=160, bbox_inches="tight"
        )
        plt.close(primitives_figure)
        plt.close(initial_figure)
        plt.close(optimized_figure)

        summary_lines.extend(
            [
                f"[case_{case_index:04d}]",
                f"initial_loss={float(initial_loss[case_index].item()):.6f}",
                f"best_loss={float(best_loss[case_index].item()):.6f}",
                f"relative_improvement={float(improvement[case_index].item()):.6f}",
                f"target_trace={float(torch.trace(target_stiffness[case_index]).item()):.6f}",
            ]
        )

    (output_path / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    return output_path


def load_visualizable_dataset(
    dataset_path: str | Path,
) -> OptimizedCases:
    from compliant_mechanism_synthesis.dataset.offline import load_offline_dataset

    optimized_cases, _ = load_offline_dataset(dataset_path)
    return optimized_cases
