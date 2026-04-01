from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from compliant_mechanism_synthesis.common import ROLE_FIXED, ROLE_FREE, ROLE_MOBILE


ROLE_STYLE = {
    ROLE_FIXED: {"label": "fixed", "color": "tab:blue", "marker": "s"},
    ROLE_MOBILE: {"label": "mobile", "color": "tab:orange", "marker": "^"},
    ROLE_FREE: {"label": "free", "color": "tab:green", "marker": "o"},
}


def plot_graph_design(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
    threshold: float = 0.5,
    title: str | None = None,
) -> plt.Figure:
    positions = positions.detach().cpu()
    roles = roles.detach().cpu()
    adjacency = adjacency.detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            activation = adjacency[i, j].item()
            if activation < threshold:
                continue
            ax.plot(
                [positions[i, 0].item(), positions[j, 0].item()],
                [positions[i, 1].item(), positions[j, 1].item()],
                color="black",
                alpha=min(0.2 + activation, 1.0),
                linewidth=0.5 + 3.0 * activation,
            )

    for role, style in ROLE_STYLE.items():
        mask = roles == role
        if mask.any():
            ax.scatter(
                positions[mask, 0].numpy(),
                positions[mask, 1].numpy(),
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                s=55,
                edgecolors="white",
                linewidths=0.6,
            )

    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    return fig
