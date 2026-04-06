from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch

from compliant_mechanism_synthesis.roles import NodeRole


ROLE_STYLE = {
    NodeRole.FIXED: {"label": "fixed", "color": "#1f77b4", "marker": "s"},
    NodeRole.MOBILE: {"label": "mobile", "color": "#ff7f0e", "marker": "^"},
    NodeRole.FREE: {"label": "free", "color": "#2ca02c", "marker": "o"},
}

PRIMITIVE_TYPE_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
)


def plot_design_3d(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    threshold: float = 0.05,
    title: str | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()

    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            activation = float(adjacency[i, j].item())
            if activation < threshold:
                continue
            ax.plot(
                [positions[i, 0].item(), positions[j, 0].item()],
                [positions[i, 1].item(), positions[j, 1].item()],
                [positions[i, 2].item(), positions[j, 2].item()],
                color="0.25",
                linewidth=0.5 + 2.0 * activation,
                alpha=0.9,
            )

    for role, style in ROLE_STYLE.items():
        mask = roles == int(role)
        if not mask.any():
            continue
        ax.scatter(
            positions[mask, 0].numpy(),
            positions[mask, 1].numpy(),
            positions[mask, 2].numpy(),
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            s=50,
            depthshade=False,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    return fig


def plot_scaffold_primitives_3d(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    edge_primitive_types: torch.Tensor,
    primitive_labels: tuple[str, ...],
    title: str | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()
    edge_primitive_types = edge_primitive_types.detach().cpu()

    used_labels: set[int] = set()
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            if float(adjacency[i, j].item()) <= 0.0:
                continue
            primitive_index = int(edge_primitive_types[i, j].item())
            color = "#666666" if primitive_index < 0 else PRIMITIVE_TYPE_COLORS[primitive_index % len(PRIMITIVE_TYPE_COLORS)]
            if primitive_index >= 0:
                used_labels.add(primitive_index)
            ax.plot(
                [positions[i, 0].item(), positions[j, 0].item()],
                [positions[i, 1].item(), positions[j, 1].item()],
                [positions[i, 2].item(), positions[j, 2].item()],
                color=color,
                linewidth=2.0,
                alpha=0.95,
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
            s=70 if role != NodeRole.FREE else 45,
            depthshade=False,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PRIMITIVE_TYPE_COLORS[index % len(PRIMITIVE_TYPE_COLORS)],
            linewidth=2.5,
            label=primitive_labels[index],
        )
        for index in sorted(used_labels)
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="None",
            markersize=7,
            label=style["label"],
        )
        for _, style in ROLE_STYLE.items()
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    if title:
        ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper right")
    return fig
