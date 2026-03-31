from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DatasetBundle:
    designs: torch.Tensor
    targets: torch.Tensor


def _draw_segment(
    grid: torch.Tensor, start: tuple[int, int], end: tuple[int, int]
) -> None:
    r0, c0 = start
    r1, c1 = end
    row_step = 1 if r1 >= r0 else -1
    col_step = 1 if c1 >= c0 else -1

    for row in range(r0, r1 + row_step, row_step):
        grid[row, c0] = 1.0
    for col in range(c0, c1 + col_step, col_step):
        grid[r1, col] = 1.0


def _random_polyline(grid_size: int) -> list[tuple[int, int]]:
    n_midpoints = random.randint(2, max(2, min(5, grid_size // 4)))
    interior_rows = sorted(
        random.sample(range(2, grid_size - 2), k=n_midpoints), reverse=True
    )
    points = [(grid_size - 1, random.randrange(grid_size))]
    points.extend((row, random.randrange(grid_size)) for row in interior_rows)
    points.append((0, random.randrange(grid_size)))
    return points


def generate_design(
    grid_size: int, min_paths: int = 2, max_paths: int = 5
) -> torch.Tensor:
    grid = torch.zeros((grid_size, grid_size), dtype=torch.float32)

    for _ in range(random.randint(min_paths, max_paths)):
        points = _random_polyline(grid_size)
        for start, end in zip(points, points[1:]):
            _draw_segment(grid, start, end)

    thickness = random.randint(0, 1)
    for _ in range(thickness):
        grid = F.max_pool2d(grid[None, None], kernel_size=3, stride=1, padding=1)[0, 0]

    if random.random() < 0.85:
        holes = torch.rand((grid_size - 2, grid_size - 2)) > 0.84
        grid[1:-1, 1:-1] = grid[1:-1, 1:-1] * (~holes)

    if random.random() < 0.4:
        # A light erosion broadens the synthetic stiffness range without a full solver.
        grid = (
            1.0
            - F.max_pool2d(
                (1.0 - grid)[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
        )

    if random.random() < 0.35:
        branch_row = random.randrange(1, grid_size - 1)
        branch_col = random.randrange(grid_size)
        _draw_segment(grid, (branch_row, branch_col), (0, random.randrange(grid_size)))

    # The plates are always present in this first prototype.
    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    return grid.clamp_(0.0, 1.0)


def generate_dataset(num_samples: int, grid_size: int, seed: int) -> torch.Tensor:
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.stack(
        [generate_design(grid_size) for _ in range(num_samples)], dim=0
    ).unsqueeze(1)
