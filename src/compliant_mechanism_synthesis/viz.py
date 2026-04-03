from __future__ import annotations

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from PIL import Image

from compliant_mechanism_synthesis.common import ROLE_FIXED, ROLE_FREE, ROLE_MOBILE
from compliant_mechanism_synthesis.mechanics import FrameFEMConfig, load_case_deformations


ROLE_STYLE = {
    ROLE_FIXED: {"label": "fixed", "color": "tab:blue", "marker": "s"},
    ROLE_MOBILE: {"label": "mobile", "color": "tab:orange", "marker": "^"},
    ROLE_FREE: {"label": "free", "color": "tab:green", "marker": "o"},
}

LOAD_CASES = ["Fx", "Fy", "M"]
UNDEFORMED_EDGE_COLOR = "0.45"
DEFORMED_EDGE_COLOR = "0.10"
TARGET_MOBILE_COLOR = "tab:red"
ACHIEVED_MOBILE_COLOR = ROLE_STYLE[ROLE_MOBILE]["color"]
MAX_RENDERED_EDGES = 256


def _setup_axis(
    ax: plt.Axes,
    x_limits: tuple[float, float] = (-0.15, 1.15),
    y_limits: tuple[float, float] = (-0.15, 1.15),
) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def _draw_rigid_pair(
    ax: plt.Axes,
    positions: torch.Tensor,
    roles: torch.Tensor,
    *,
    linestyle: str = "-",
) -> None:
    for role in (ROLE_FIXED, ROLE_MOBILE):
        indices = torch.where(roles == role)[0]
        if indices.numel() != 2:
            continue
        style = ROLE_STYLE[role]
        pair = positions[indices]
        ax.plot(
            pair[:, 0].numpy(),
            pair[:, 1].numpy(),
            color=style["color"],
            linewidth=2.4,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=2,
        )


def _draw_graph(
    ax: plt.Axes,
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    threshold: float,
    frame_config: FrameFEMConfig,
    title: str | None = None,
    legend: bool = False,
    edge_color: str = "black",
    edge_linestyle: str = "-",
    fill_nodes: bool = True,
    axis_width_in: float | None = None,
    x_limits: tuple[float, float] = (-0.15, 1.15),
    y_limits: tuple[float, float] = (-0.15, 1.15),
) -> None:
    _setup_axis(ax, x_limits=x_limits, y_limits=y_limits)
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()

    if axis_width_in is None:
        ax.figure.canvas.draw()
        axis_width_in = (
            ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted()).width
        )

    segments = []
    linewidths = []
    edge_records: list[tuple[float, list[list[float]], float]] = []
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            activation = adjacency[i, j].item()
            if activation < threshold:
                continue
            diameter = 2.0 * frame_config.r_max * activation
            linewidth = (diameter / frame_config.workspace_size) * axis_width_in * 72.0
            if linewidth <= 0.0 or activation <= 0.0:
                continue
            edge_records.append(
                (
                    activation,
                    [
                        [positions[i, 0].item(), positions[i, 1].item()],
                        [positions[j, 0].item(), positions[j, 1].item()],
                    ],
                    linewidth,
                )
            )

    if len(edge_records) > MAX_RENDERED_EDGES:
        edge_records.sort(key=lambda item: item[0], reverse=True)
        edge_records = edge_records[:MAX_RENDERED_EDGES]

    for _, segment, linewidth in edge_records:
        segments.append(segment)
        linewidths.append(linewidth)

    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=edge_color,
                linewidths=linewidths,
                linestyles=edge_linestyle,
                zorder=1,
            )
        )

    _draw_rigid_pair(ax, positions, roles, linestyle=edge_linestyle)

    for role, style in ROLE_STYLE.items():
        mask = roles == role
        if mask.any():
            ax.scatter(
                positions[mask, 0].numpy(),
                positions[mask, 1].numpy(),
                label=style["label"],
                facecolors=style["color"] if fill_nodes else "white",
                edgecolors=style["color"],
                marker=style["marker"],
                s=55,
                linewidths=1.0,
                zorder=3,
            )

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper right")


def plot_graph_design(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    threshold: float = 0.5,
    title: str | None = None,
    frame_config: FrameFEMConfig | None = None,
) -> plt.Figure:
    frame_config = frame_config or FrameFEMConfig()
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_graph(
        ax,
        positions,
        roles,
        adjacency,
        threshold=threshold,
        frame_config=frame_config,
        title=title,
        legend=True,
    )
    return fig


def _mobile_pair_positions(
    positions: torch.Tensor, roles: torch.Tensor
) -> torch.Tensor | None:
    indices = torch.where(roles == ROLE_MOBILE)[0]
    if indices.numel() != 2:
        return None
    return positions[indices]


def _target_mobile_pair(
    positions: torch.Tensor,
    roles: torch.Tensor,
    generalized_displacement: torch.Tensor,
    frame_config: FrameFEMConfig,
    response_scale: float = 1.0,
) -> torch.Tensor | None:
    pair = _mobile_pair_positions(positions, roles)
    if pair is None:
        return None
    physical_pair = pair * frame_config.workspace_size
    centroid = physical_pair.mean(dim=0, keepdim=True)
    rel = physical_pair - centroid
    theta = response_scale * generalized_displacement[2].item()
    rotation = torch.tensor(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=physical_pair.dtype,
    )
    translation = response_scale * generalized_displacement[:2].to(
        dtype=physical_pair.dtype
    )
    transformed = centroid + rel @ rotation.transpose(0, 1) + translation.unsqueeze(0)
    return transformed / frame_config.workspace_size


def _draw_mobile_pair_overlay(
    ax: plt.Axes,
    pair: torch.Tensor | None,
    color: str,
    linestyle: str,
    label: str,
    marker: str,
) -> None:
    if pair is None:
        return
    pair = pair.detach().cpu()
    ax.plot(
        pair[:, 0].numpy(),
        pair[:, 1].numpy(),
        color=color,
        linestyle=linestyle,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=5,
    )
    scatter_kwargs = {
        "color": color,
        "marker": marker,
        "s": 90,
        "linewidths": 1.0,
        "zorder": 6,
    }
    if marker != "x":
        scatter_kwargs["edgecolors"] = "white"
    ax.scatter(
        pair[:, 0].numpy(),
        pair[:, 1].numpy(),
        **scatter_kwargs,
    )
    center = pair.mean(dim=0)
    ax.text(
        center[0].item(),
        center[1].item(),
        label,
        fontsize=7,
        color=color,
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        zorder=7,
    )


def _display_scale(
    display_scale: float,
) -> float:
    return display_scale


def _draw_load_case_panel(
    ax: plt.Axes,
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    deformation: torch.Tensor,
    achieved_response: torch.Tensor,
    target_response: torch.Tensor,
    load_name: str,
    threshold: float,
    frame_config: FrameFEMConfig,
    display_scale: float,
    axis_width_in: float,
) -> None:
    case_scale = _display_scale(display_scale)
    normalized_deformation = deformation / frame_config.workspace_size
    deformed_positions = positions + case_scale * normalized_deformation
    achieved_mobile_pair = _mobile_pair_positions(deformed_positions, roles)
    load_magnitude = (
        frame_config.reference_moment if load_name == "M" else frame_config.reference_force
    )
    target_pair = _target_mobile_pair(
        positions,
        roles,
        load_magnitude * target_response,
        frame_config=frame_config,
        response_scale=case_scale,
    )
    _draw_graph(
        ax,
        positions,
        roles,
        adjacency,
        threshold=threshold,
        frame_config=frame_config,
        title=f"{load_name} eval",
        legend=False,
        edge_color=UNDEFORMED_EDGE_COLOR,
        edge_linestyle="--",
        fill_nodes=False,
        axis_width_in=axis_width_in,
    )
    _draw_graph(
        ax,
        deformed_positions,
        roles,
        adjacency,
        threshold=threshold,
        frame_config=frame_config,
        legend=False,
        edge_color=DEFORMED_EDGE_COLOR,
        edge_linestyle="-",
        fill_nodes=True,
        axis_width_in=axis_width_in,
    )
    if target_pair is not None:
        _draw_mobile_pair_overlay(
            ax,
            target_pair,
            color=TARGET_MOBILE_COLOR,
            linestyle="--",
            label="target mobile",
            marker="x",
        )
    _draw_mobile_pair_overlay(
        ax,
        achieved_mobile_pair,
        color=ACHIEVED_MOBILE_COLOR,
        linestyle="-",
        label="achieved mobile",
        marker="^",
    )
    ax.text(
        0.02,
        0.02,
        (
            f"target=({target_response[0].item():.2e}, {target_response[1].item():.2e}, {target_response[2].item():.2e})\n"
            f"done=({achieved_response[0].item():.2e}, {achieved_response[1].item():.2e}, {achieved_response[2].item():.2e})\n"
            f"mobile: solid=achieved  dashed=target\n"
            f"visual deformation scale={case_scale:.1f}x"
        ),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def _render_rollout_frame(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    deformations: torch.Tensor,
    achieved: torch.Tensor,
    target_response: torch.Tensor,
    phase_label: str,
    threshold: float,
    frame_config: FrameFEMConfig,
    display_scale: float,
    title: str | None,
) -> Image.Image:
    started_at = time.perf_counter()
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()
    deformations = deformations.detach().cpu()
    achieved = achieved.detach().cpu()
    target_response = target_response.detach().cpu()

    fig, axes = plt.subplots(4, 1, figsize=(8, 14))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.04,
        top=0.97,
        hspace=0.28,
    )
    stage_started_at = time.perf_counter()
    fig.canvas.draw()
    axis_width_in = (
        axes[0]
        .get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .width
    )
    print(
        f"viz:frame phase={phase_label} layout_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )
    main_title = phase_label if title is None else f"{title} | {phase_label}"
    stage_started_at = time.perf_counter()
    _draw_graph(
        axes[0],
        positions,
        roles,
        adjacency,
        threshold=threshold,
        frame_config=frame_config,
        title=main_title,
        legend=True,
        axis_width_in=axis_width_in,
    )
    for idx, (load_name, ax) in enumerate(
        zip(LOAD_CASES, axes[1:])
    ):
        _draw_load_case_panel(
            ax,
            positions,
            roles,
            adjacency,
            deformations[idx],
            achieved[:, idx],
            target_response[:, idx],
            load_name,
            threshold,
            frame_config,
            display_scale,
            axis_width_in,
        )
    print(
        f"viz:frame phase={phase_label} plot_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )
    stage_started_at = time.perf_counter()
    fig.canvas.draw()
    print(
        f"viz:frame phase={phase_label} draw_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )
    stage_started_at = time.perf_counter()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    print(
        f"viz:frame phase={phase_label} buffer_dt={time.perf_counter() - stage_started_at:.2f}s total_dt={time.perf_counter() - started_at:.2f}s",
        flush=True,
    )
    plt.close(fig)
    return image


def export_rollout_animation(
    output_path: str | Path,
    initial_positions: torch.Tensor,
    roles: torch.Tensor,
    initial_adjacency: torch.Tensor,
    rollout: list[dict[str, torch.Tensor]],
    target_response: torch.Tensor,
    threshold: float = 0.5,
    frame_config: FrameFEMConfig | None = None,
    display_scale: float = 10.0,
    title: str | None = None,
    final_positions: torch.Tensor | None = None,
    final_adjacency: torch.Tensor | None = None,
) -> Path:
    started_at = time.perf_counter()
    frame_config = frame_config or FrameFEMConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        (initial_positions, initial_adjacency, "00 init_noise"),
    ]
    for idx, state in enumerate(rollout, start=1):
        refined_positions = state.get("refined_positions", state["positions"])
        refined_adjacency = state.get("refined_adjacency", state["adjacency"])
        frames.append((refined_positions, refined_adjacency, f"{idx:02d} refine"))
    if final_positions is not None and final_adjacency is not None:
        frames.append((final_positions, final_adjacency, "99 final_eval"))
    print(
        f"viz:animation frames={len(frames)} collect_dt={time.perf_counter() - started_at:.2f}s",
        flush=True,
    )

    stage_started_at = time.perf_counter()
    frame_positions = torch.stack(
        [positions.detach().cpu() for positions, _, _ in frames],
        dim=0,
    )
    frame_roles = roles.detach().cpu().unsqueeze(0).expand(len(frames), -1)
    frame_adjacency = torch.stack(
        [adjacency.detach().cpu() for _, adjacency, _ in frames],
        dim=0,
    )
    frame_deformations, frame_achieved = load_case_deformations(
        frame_positions,
        frame_roles,
        frame_adjacency,
        config=frame_config,
    )
    print(
        f"viz:animation mechanics_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )

    stage_started_at = time.perf_counter()
    images = [
        _render_rollout_frame(
            positions,
            roles,
            adjacency,
            frame_deformations[idx],
            frame_achieved[idx],
            target_response,
            phase_label,
            threshold,
            frame_config,
            display_scale,
            title,
        )
        for idx, (positions, adjacency, phase_label) in enumerate(frames)
    ]
    print(
        f"viz:animation render_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )
    stage_started_at = time.perf_counter()
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=550,
        loop=0,
    )
    print(
        f"viz:animation save_dt={time.perf_counter() - stage_started_at:.2f}s total_dt={time.perf_counter() - started_at:.2f}s",
        flush=True,
    )
    return output_path


def _render_canonical_frame(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    deformations: torch.Tensor,
    achieved: torch.Tensor,
    target_responses: torch.Tensor,
    names: list[str],
    phase_label: str,
    threshold: float,
    frame_config: FrameFEMConfig,
    display_scale: float,
) -> Image.Image:
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()
    deformations = deformations.detach().cpu()
    achieved = achieved.detach().cpu()
    target_responses = target_responses.detach().cpu()
    num_cases = positions.shape[0]

    fig, axes = plt.subplots(4, num_cases, figsize=(4.5 * num_cases, 14))
    if num_cases == 1:
        axes = np.asarray(axes).reshape(4, 1)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.95, wspace=0.18, hspace=0.28)
    fig.suptitle(phase_label, fontsize=12)
    fig.canvas.draw()
    axis_width_in = (
        axes[0, 0]
        .get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .width
    )

    for case_idx in range(num_cases):
        _draw_graph(
            axes[0, case_idx],
            positions[case_idx],
            roles[case_idx],
            adjacency[case_idx],
            threshold=threshold,
            frame_config=frame_config,
            title=names[case_idx],
            legend=(case_idx == 0),
            axis_width_in=axis_width_in,
        )
        for load_idx, ax in enumerate(axes[1:, case_idx]):
            _draw_load_case_panel(
                ax,
                positions[case_idx],
                roles[case_idx],
                adjacency[case_idx],
                deformations[case_idx, load_idx],
                achieved[case_idx, :, load_idx],
                target_responses[case_idx, :, load_idx],
                LOAD_CASES[load_idx],
                threshold,
                frame_config,
                display_scale,
                axis_width_in,
            )

    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return image


def export_canonical_animation(
    output_path: str | Path,
    initial_positions: torch.Tensor,
    roles: torch.Tensor,
    initial_adjacency: torch.Tensor,
    rollout: list[dict[str, torch.Tensor]],
    target_responses: torch.Tensor,
    names: list[str],
    threshold: float = 0.5,
    frame_config: FrameFEMConfig | None = None,
    display_scale: float = 10.0,
    final_positions: torch.Tensor | None = None,
    final_adjacency: torch.Tensor | None = None,
) -> Path:
    started_at = time.perf_counter()
    frame_config = frame_config or FrameFEMConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [(initial_positions, initial_adjacency, "00 init_noise")]
    for idx, state in enumerate(rollout, start=1):
        refined_positions = state.get("refined_positions", state["positions"])
        refined_adjacency = state.get("refined_adjacency", state["adjacency"])
        frames.append((refined_positions, refined_adjacency, f"{idx:02d} refine"))
    if final_positions is not None and final_adjacency is not None:
        frames.append((final_positions, final_adjacency, "99 final_eval"))
    print(
        f"viz:canonical_animation frames={len(frames)} cases={initial_positions.shape[0]} collect_dt={time.perf_counter() - started_at:.2f}s",
        flush=True,
    )

    stage_started_at = time.perf_counter()
    frame_positions = torch.stack(
        [frame_positions.detach().cpu() for frame_positions, _, _ in frames],
        dim=0,
    )
    frame_roles = roles.detach().cpu().unsqueeze(0).expand(len(frames), -1, -1)
    frame_adjacency = torch.stack(
        [frame_adjacency.detach().cpu() for _, frame_adjacency, _ in frames],
        dim=0,
    )
    batch_frames, num_cases, num_nodes, _ = frame_positions.shape
    flat_positions = frame_positions.reshape(batch_frames * num_cases, num_nodes, 2)
    flat_roles = frame_roles.reshape(batch_frames * num_cases, num_nodes)
    flat_adjacency = frame_adjacency.reshape(batch_frames * num_cases, num_nodes, num_nodes)
    flat_deformations, flat_achieved = load_case_deformations(
        flat_positions,
        flat_roles,
        flat_adjacency,
        config=frame_config,
    )
    frame_deformations = flat_deformations.reshape(batch_frames, num_cases, 3, num_nodes, 2)
    frame_achieved = flat_achieved.reshape(batch_frames, num_cases, 3, 3)
    print(
        f"viz:canonical_animation mechanics_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )

    stage_started_at = time.perf_counter()
    images = [
        _render_canonical_frame(
            frame_positions[idx],
            frame_roles[idx],
            frame_adjacency[idx],
            frame_deformations[idx],
            frame_achieved[idx],
            target_responses,
            names,
            phase_label,
            threshold,
            frame_config,
            display_scale,
        )
        for idx, (_, _, phase_label) in enumerate(frames)
    ]
    print(
        f"viz:canonical_animation render_dt={time.perf_counter() - stage_started_at:.2f}s",
        flush=True,
    )
    stage_started_at = time.perf_counter()
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=550,
        loop=0,
    )
    print(
        f"viz:canonical_animation save_dt={time.perf_counter() - stage_started_at:.2f}s total_dt={time.perf_counter() - started_at:.2f}s",
        flush=True,
    )
    return output_path
