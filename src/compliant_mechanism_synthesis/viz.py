from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from PIL import Image

from compliant_mechanism_synthesis.common import ROLE_FIXED, ROLE_FREE, ROLE_MOBILE
from compliant_mechanism_synthesis.mechanics import (
    FrameFEMConfig,
    load_case_deformations,
)


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
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            activation = adjacency[i, j].item()
            if activation < threshold:
                continue
            diameter = 2.0 * frame_config.r_max * activation
            linewidth = (diameter / frame_config.workspace_size) * axis_width_in * 72.0
            if linewidth <= 0.0 or activation <= 0.0:
                continue
            segments.append(
                [
                    [positions[i, 0].item(), positions[i, 1].item()],
                    [positions[j, 0].item(), positions[j, 1].item()],
                ]
            )
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
    deformation: torch.Tensor,
    frame_config: FrameFEMConfig,
) -> float:
    norms = (
        torch.linalg.vector_norm(deformation.detach().cpu(), dim=-1)
        / frame_config.workspace_size
    )
    reference = torch.quantile(norms, 0.95).item()
    if reference <= 1e-8:
        return 1.0
    return min(max(0.08 / reference, 1.0), 2_000.0)


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
    axis_width_in: float,
) -> None:
    case_scale = _display_scale(deformation, frame_config)
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
            f"disp x{case_scale:.2e}"
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
    title: str | None,
) -> Image.Image:
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()
    deformations = deformations.detach().cpu()
    achieved = achieved.detach().cpu()
    target_response = target_response.detach().cpu()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.05,
        top=0.94,
        wspace=0.16,
        hspace=0.20,
    )
    fig.canvas.draw()
    axis_width_in = (
        axes[0, 0]
        .get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .width
    )
    main_title = phase_label if title is None else f"{title} | {phase_label}"
    _draw_graph(
        axes[0, 0],
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
        zip(LOAD_CASES, [axes[0, 1], axes[1, 0], axes[1, 1]])
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
            axis_width_in,
        )
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
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
    title: str | None = None,
    final_positions: torch.Tensor | None = None,
    final_adjacency: torch.Tensor | None = None,
) -> Path:
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
        if (
            "positions" in state
            and "adjacency" in state
            and not (
                torch.allclose(state["positions"], refined_positions)
                and torch.allclose(state["adjacency"], refined_adjacency)
            )
        ):
            frames.append((state["positions"], state["adjacency"], f"{idx:02d} noise"))
    if final_positions is not None and final_adjacency is not None:
        frames.append((final_positions, final_adjacency, "99 final_eval"))

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
            title,
        )
        for idx, (positions, adjacency, phase_label) in enumerate(frames)
    ]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=550,
        loop=0,
    )
    return output_path
