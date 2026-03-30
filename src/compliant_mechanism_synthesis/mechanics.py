from __future__ import annotations

import torch
import torch.nn.functional as F


def _reachability(occupancy: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    reach = occupancy * seeds
    for _ in range(occupancy.shape[-2]):
        reach = occupancy * F.max_pool2d(reach, kernel_size=3, stride=1, padding=1)
    return reach


def interface_length(occupancy: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(occupancy[:, :, :, 1:] - occupancy[:, :, :, :-1]).mean(dim=(1, 2, 3))
    dy = torch.abs(occupancy[:, :, 1:, :] - occupancy[:, :, :-1, :]).mean(dim=(1, 2, 3))
    return dx + dy


def mechanical_terms(occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, _, height, width = occupancy.shape
    device = occupancy.device

    bottom = torch.zeros((batch, 1, height, width), device=device)
    top = torch.zeros((batch, 1, height, width), device=device)
    bottom[:, :, -1, :] = 1.0
    top[:, :, 0, :] = 1.0

    reach_bottom = _reachability(occupancy, bottom)
    reach_top = _reachability(occupancy, top)
    bridge = torch.minimum(reach_bottom, reach_top)

    connected_mass = bridge.mean(dim=(1, 2, 3))
    top_connectivity = bridge[:, :, 0, :].mean(dim=(1, 2))
    x_coverage = 1.0 - torch.prod(1.0 - bridge.squeeze(1) + 1e-6, dim=1)
    x_coverage = x_coverage.mean(dim=1)

    xs = torch.linspace(-1.0, 1.0, width, device=device)
    ys = torch.linspace(1.0, -1.0, height, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    radius_sq = xx.square() + (yy - 1.0).square()
    polar = (bridge.squeeze(1) * radius_sq).sum(dim=(1, 2)) / (
        bridge.squeeze(1).sum(dim=(1, 2)) + 1e-6
    )
    polar = (polar / 5.0).clamp(0.0, 1.0)

    kx = (top_connectivity * x_coverage).clamp(0.0, 1.0)
    ky = (top_connectivity * (0.5 + connected_mass)).clamp(0.0, 1.0)
    ktheta = (top_connectivity * polar).clamp(0.0, 1.0)

    occupancy_mass = occupancy.mean(dim=(1, 2, 3))
    connectivity_penalty = 1.0 - (
        bridge.sum(dim=(1, 2, 3)) / (occupancy.sum(dim=(1, 2, 3)) + 1e-6)
    )
    connectivity_penalty = connectivity_penalty.clamp(0.0, 1.0)

    return {
        "properties": torch.stack([kx, ky, ktheta], dim=1),
        "surface": interface_length(occupancy),
        "connectivity_penalty": connectivity_penalty,
        "connected_mass": connected_mass,
        "occupancy_mass": occupancy_mass,
    }
