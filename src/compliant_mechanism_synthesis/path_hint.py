"""Detached parent-path guesses; never a substitute for verified equilibrium."""

from dataclasses import dataclass

import torch

from .geometry_losses import interpolate_poses


@dataclass
class PathHint:
    positions: torch.Tensor
    presence: torch.Tensor
    records: list
    segments: int
    radii: torch.Tensor | None = None

    def compatible(
        self, positions, presence, segments, distance, radii=None, radius_fraction=0.05
    ):
        if segments != self.segments or not torch.equal(presence, self.presence):
            return False
        if (positions - self.positions).norm(dim=-1).max() > distance:
            return False
        if radii is not None and self.radii is not None:
            present = presence != 0
            if (
                (radii[present] - self.radii[present]).abs() / self.radii[present]
            ).max() > radius_fraction:
                return False
        return True

    def to(self, device):
        records = [
            (r[0], r[1].to(device), *r[2:6], None if r[6] is None else r[6].to(device))
            for r in self.records
        ]
        return PathHint(
            self.positions.to(device),
            self.presence.to(device),
            records,
            self.segments,
            None if self.radii is None else self.radii.to(device),
        )

    def at(self, factor, positions, edges):
        if (
            not self.records
            or factor < self.records[0][0]
            or factor > self.records[-1][0]
        ):
            return None, None
        upper = next(
            (i for i, r in enumerate(self.records) if r[0] >= factor),
            len(self.records) - 1,
        )
        lower = max(0, upper - 1)
        a, b = self.records[lower], self.records[upper]
        fraction = 0.0 if a[0] == b[0] else (factor - a[0]) / (b[0] - a[0])
        q = interpolate_poses(a[1][None], b[1][None], fraction)[0]
        if a[6] is None:
            return q, None
        inner = interpolate_poses(
            a[6].reshape(1, -1), b[6].reshape(1, -1), fraction
        ).reshape_as(a[6])
        ids = torch.nonzero(self.presence != 0).flatten()
        delta = positions - self.positions
        fraction_mesh = (
            torch.arange(
                1, self.segments, dtype=positions.dtype, device=positions.device
            )
            / self.segments
        )
        shift = (
            delta[edges[ids, 0], None]
            + fraction_mesh[None, :, None]
            * (delta[edges[ids, 1]] - delta[edges[ids, 0]])[:, None]
        )
        inner = torch.cat((inner[..., :3] + shift, inner[..., 3:]), -1)
        full = positions.new_zeros(len(edges), self.segments - 1, 6).index_copy(
            0, ids, inner
        )
        return q, full
