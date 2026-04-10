from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import torch

from compliant_mechanism_synthesis.dataset.types import OptimizedCases, Structures
from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.roles import NodeRole
from compliant_mechanism_synthesis.training import (
    analyze_structures,
    dataset_noise_statistics,
    generalized_stiffness_error,
    match_oracle_to_source,
    sample_noisy_structures,
)
from compliant_mechanism_synthesis.utils import resolve_torch_device
from compliant_mechanism_synthesis.visualization.plots import ROLE_STYLE


def load_supervised_refiner_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> SupervisedRefiner:
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = SupervisedRefiner(SupervisedRefinerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()
    return model


def _case_indices_to_render(
    optimized_cases: OptimizedCases,
    max_cases: int,
    case_indices: list[int] | None,
) -> list[int]:
    total_cases = optimized_cases.optimized_structures.batch_size
    if case_indices is None:
        return list(range(min(total_cases, max_cases)))
    selected = []
    for case_index in case_indices:
        if case_index < 0 or case_index >= total_cases:
            raise ValueError("case_indices must refer to valid dataset cases")
        selected.append(case_index)
    return selected


def _case_error(structures: Structures, target_stiffness: torch.Tensor) -> float:
    analyses = analyze_structures(structures)
    return float(
        generalized_stiffness_error(
            analyses.generalized_stiffness,
            target_stiffness,
        )
        .mean()
        .detach()
        .item()
    )


def _stacked_positions(structures_list: list[Structures]) -> torch.Tensor:
    return torch.cat(
        [structures.positions[0].detach().cpu() for structures in structures_list],
        dim=0,
    )


def _axis_limits(structures_list: list[Structures]) -> tuple[torch.Tensor, float]:
    stacked = _stacked_positions(structures_list)
    minimum = stacked.min(dim=0).values
    maximum = stacked.max(dim=0).values
    center = 0.5 * (minimum + maximum)
    radius = 0.55 * float((maximum - minimum).max().item())
    return center, max(radius, 0.1)


def _set_3d_axes_limits(ax: plt.Axes, center: torch.Tensor, radius: float) -> None:
    ax.set_xlim(float(center[0].item() - radius), float(center[0].item() + radius))
    ax.set_ylim(float(center[1].item() - radius), float(center[1].item() + radius))
    ax.set_zlim(float(center[2].item() - radius), float(center[2].item() + radius))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _draw_structure(
    ax: plt.Axes,
    structures: Structures,
    threshold: float,
    edge_color: str,
    edge_alpha: float,
    edge_linestyle: str,
    node_alpha: float,
) -> None:
    positions = structures.positions[0].detach().cpu()
    roles = structures.roles[0].detach().cpu()
    adjacency = structures.adjacency[0].detach().cpu()
    num_nodes = positions.shape[0]

    for node_index in range(num_nodes):
        for neighbor_index in range(node_index + 1, num_nodes):
            activation = float(adjacency[node_index, neighbor_index].item())
            if activation < threshold:
                continue
            ax.plot(
                [positions[node_index, 0].item(), positions[neighbor_index, 0].item()],
                [positions[node_index, 1].item(), positions[neighbor_index, 1].item()],
                [positions[node_index, 2].item(), positions[neighbor_index, 2].item()],
                color=edge_color,
                linewidth=0.5 + 2.0 * activation,
                alpha=edge_alpha,
                linestyle=edge_linestyle,
            )

    for role, style in ROLE_STYLE.items():
        mask = roles == int(role)
        if not mask.any():
            continue
        ax.scatter(
            positions[mask, 0].numpy(),
            positions[mask, 1].numpy(),
            positions[mask, 2].numpy(),
            color=style["color"],
            marker=style["marker"],
            s=26 if role != NodeRole.FREE else 18,
            alpha=node_alpha,
            depthshade=False,
        )


def _draw_overlay_panel(
    ax: plt.Axes,
    predicted: Structures,
    oracle: Structures,
    threshold: float,
    center: torch.Tensor,
    radius: float,
    title: str,
    predicted_edge_color: str,
) -> None:
    _draw_structure(
        ax=ax,
        structures=oracle,
        threshold=threshold,
        edge_color="#222222",
        edge_alpha=0.35,
        edge_linestyle="--",
        node_alpha=0.2,
    )
    _draw_structure(
        ax=ax,
        structures=predicted,
        threshold=threshold,
        edge_color=predicted_edge_color,
        edge_alpha=0.95,
        edge_linestyle="-",
        node_alpha=0.95,
    )
    _set_3d_axes_limits(ax, center=center, radius=radius)
    ax.set_title(title)


def _comparison_legend_handles(include_style: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [0], [0], color="#222222", linestyle="--", linewidth=2.0, label="oracle"
        ),
        Line2D([0], [0], color="#7b61ff", linestyle="-", linewidth=2.0, label="noise"),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linestyle="-",
            linewidth=2.0,
            label="generated no style",
        ),
    ]
    if include_style:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#17a398",
                linestyle="-",
                linewidth=2.0,
                label="generated with style",
            )
        )
    handles.extend(
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="None",
            markersize=5,
            label=style["label"],
        )
        for style in ROLE_STYLE.values()
    )
    return handles


def _write_case_summary_figure(
    output_path: Path,
    case_index: int,
    source_structures: Structures,
    oracle_structures: Structures,
    no_style_structures: Structures,
    no_style_error: float,
    with_style_structures: Structures | None,
    with_style_error: float | None,
    threshold: float,
) -> None:
    panels: list[tuple[str, Structures, str]] = [
        ("noise vs oracle", source_structures, "#7b61ff"),
        (
            f"generated no style\nerror={no_style_error:.4f}",
            no_style_structures,
            "#d62728",
        ),
    ]
    if with_style_structures is not None and with_style_error is not None:
        panels.append(
            (
                f"generated with style\nerror={with_style_error:.4f}",
                with_style_structures,
                "#17a398",
            )
        )
    center, radius = _axis_limits(
        [source_structures, oracle_structures, no_style_structures]
        + ([] if with_style_structures is None else [with_style_structures])
    )
    figure = plt.figure(figsize=(6 * len(panels), 5.5))
    for panel_index, (title, structures, edge_color) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), panel_index, projection="3d")
        _draw_overlay_panel(
            axis,
            predicted=structures,
            oracle=oracle_structures,
            threshold=threshold,
            center=center,
            radius=radius,
            title=title,
            predicted_edge_color=edge_color,
        )
    figure.suptitle(f"supervised sample case {case_index:04d}")
    figure.legend(
        handles=_comparison_legend_handles(with_style_structures is not None),
        loc="lower center",
        ncols=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_case_rollout_gif(
    output_path: Path,
    case_index: int,
    oracle_structures: Structures,
    no_style_trajectory: list[Structures],
    with_style_trajectory: list[Structures] | None,
    threshold: float,
) -> None:
    panels: list[tuple[str, list[Structures], str]] = [
        ("no style", no_style_trajectory, "#d62728"),
    ]
    if with_style_trajectory is not None:
        panels.append(("with style", with_style_trajectory, "#17a398"))
    center, radius = _axis_limits(
        [oracle_structures]
        + no_style_trajectory
        + ([] if with_style_trajectory is None else with_style_trajectory)
    )
    frame_count = max(len(trajectory) for _, trajectory, _ in panels)
    figure = plt.figure(figsize=(6 * len(panels), 5.0))
    axes = [
        figure.add_subplot(1, len(panels), index + 1, projection="3d")
        for index in range(len(panels))
    ]

    def update(frame_index: int) -> list[plt.Axes]:
        for axis, (label, trajectory, edge_color) in zip(axes, panels, strict=True):
            axis.cla()
            structures = trajectory[min(frame_index, len(trajectory) - 1)]
            _draw_overlay_panel(
                axis,
                predicted=structures,
                oracle=oracle_structures,
                threshold=threshold,
                center=center,
                radius=radius,
                title=f"{label} step {frame_index}/{len(trajectory) - 1}",
                predicted_edge_color=edge_color,
            )
        figure.suptitle(f"supervised rollout case {case_index:04d}")
        return axes

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=350,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=3))
    plt.close(figure)


def write_supervised_sampling_visualizations(
    optimized_cases: OptimizedCases,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    max_cases: int = 4,
    case_indices: list[int] | None = None,
    seed: int = 7,
    num_steps: int | None = None,
    threshold: float = 0.05,
) -> Path:
    optimized_cases.validate()
    selected_case_indices = _case_indices_to_render(
        optimized_cases=optimized_cases,
        max_cases=max_cases,
        case_indices=case_indices,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model = load_supervised_refiner_checkpoint(
        checkpoint_path=checkpoint_path, device=device
    )
    resolved_device = next(model.parameters()).device
    position_mean, position_std, adjacency_mean, adjacency_std = (
        dataset_noise_statistics(optimized_cases)
    )
    position_mean = position_mean.to(resolved_device)
    position_std = position_std.to(resolved_device)
    adjacency_mean = adjacency_mean.to(resolved_device)
    adjacency_std = adjacency_std.to(resolved_device)
    summary_lines = [
        f"dataset_cases={optimized_cases.optimized_structures.batch_size}",
        f"rendered_cases={len(selected_case_indices)}",
        f"case_indices={','.join(str(index) for index in selected_case_indices)}",
        f"checkpoint={Path(checkpoint_path)}",
        f"use_style_token={model.config.use_style_token}",
        f"seed={seed}",
        f"num_steps={num_steps if num_steps is not None else model.config.num_integration_steps}",
    ]

    for render_order, case_index in enumerate(selected_case_indices):
        case = optimized_cases.slice(case_index).to(resolved_device)
        source_structures = sample_noisy_structures(
            optimized_cases=case,
            position_mean=position_mean,
            position_std=position_std,
            adjacency_mean=adjacency_mean,
            adjacency_std=adjacency_std,
            seed=seed + render_order,
        )
        oracle_structures, oracle_analyses = match_oracle_to_source(
            source_structures=source_structures,
            oracle_structures=case.optimized_structures,
            oracle_analyses=case.last_analyses,
        )
        with torch.no_grad():
            no_style_trajectory = model.rollout_trajectory(
                source_structures=source_structures,
                target_stiffness=case.target_stiffness,
                analysis_fn=analyze_structures,
                num_steps=num_steps,
            )
            with_style_trajectory = (
                None
                if not model.config.use_style_token
                else model.rollout_trajectory(
                    source_structures=source_structures,
                    target_stiffness=case.target_stiffness,
                    analysis_fn=analyze_structures,
                    num_steps=num_steps,
                    style_structures=oracle_structures,
                    style_analyses=oracle_analyses,
                )
            )

        no_style_structures = no_style_trajectory[-1]
        with_style_structures = (
            None if with_style_trajectory is None else with_style_trajectory[-1]
        )
        noisy_error = _case_error(source_structures, case.target_stiffness)
        oracle_error = _case_error(oracle_structures, case.target_stiffness)
        no_style_error = _case_error(no_style_structures, case.target_stiffness)
        with_style_error = (
            None
            if with_style_structures is None
            else _case_error(with_style_structures, case.target_stiffness)
        )
        source_cpu = source_structures.to("cpu")
        oracle_cpu = oracle_structures.to("cpu")
        no_style_trajectory_cpu = [
            structures.to("cpu") for structures in no_style_trajectory
        ]
        with_style_trajectory_cpu = (
            None
            if with_style_trajectory is None
            else [structures.to("cpu") for structures in with_style_trajectory]
        )
        _write_case_summary_figure(
            output_path=output_path / f"case_{case_index:04d}_comparison.png",
            case_index=case_index,
            source_structures=source_cpu,
            oracle_structures=oracle_cpu,
            no_style_structures=no_style_trajectory_cpu[-1],
            no_style_error=no_style_error,
            with_style_structures=(
                None
                if with_style_trajectory_cpu is None
                else with_style_trajectory_cpu[-1]
            ),
            with_style_error=with_style_error,
            threshold=threshold,
        )
        _write_case_rollout_gif(
            output_path=output_path / f"case_{case_index:04d}_rollout.gif",
            case_index=case_index,
            oracle_structures=oracle_cpu,
            no_style_trajectory=no_style_trajectory_cpu,
            with_style_trajectory=with_style_trajectory_cpu,
            threshold=threshold,
        )
        summary_lines.extend(
            [
                f"[case_{case_index:04d}]",
                f"noisy_target_error={noisy_error:.6f}",
                f"oracle_target_error={oracle_error:.6f}",
                f"generated_no_style_target_error={no_style_error:.6f}",
                (
                    f"generated_with_style_target_error={with_style_error:.6f}"
                    if with_style_error is not None
                    else "generated_with_style_target_error=not_available"
                ),
            ]
        )

    (output_path / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return output_path
