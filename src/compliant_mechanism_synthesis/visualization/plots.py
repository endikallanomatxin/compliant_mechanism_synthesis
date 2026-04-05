from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch

from compliant_mechanism_synthesis.roles import NodeRole


ROLE_STYLE = {
    NodeRole.FIXED: {"label": "fixed", "color": "#1f77b4", "marker": "s"},
    NodeRole.MOBILE: {"label": "mobile", "color": "#ff7f0e", "marker": "^"},
    NodeRole.FREE: {"label": "free", "color": "#2ca02c", "marker": "o"},
}


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
