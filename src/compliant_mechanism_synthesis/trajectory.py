"""Cubic Hermite interpolation of generalized rigid-body pose waypoints."""

from dataclasses import dataclass

import torch


@dataclass
class Trajectory:
    factors: torch.Tensor
    poses: torch.Tensor
    rest_factor: float = 0.0

    def validate(self):
        if (
            self.factors.ndim != 1
            or self.poses.ndim != 2
            or len(self.factors) != len(self.poses)
            or len(self.factors) < 2
        ):
            raise ValueError("Provide at least two factor/pose waypoints")
        if (
            not torch.isfinite(self.factors).all()
            or not torch.isfinite(self.poses).all()
        ):
            raise ValueError("Nonfinite trajectory")
        if (
            self.factors[0] != 0
            or self.factors[-1] != 1
            or not (self.factors.diff() > 0).all()
        ):
            raise ValueError("Factors must strictly increase from 0 to 1")
        if not 0 <= self.rest_factor <= 1:
            raise ValueError("Rest progress must lie between 0 and 1")
        rest = self.factors.new_tensor(self.rest_factor)
        if not torch.isclose(self.factors, rest, atol=1e-12, rtol=0).any():
            raise ValueError("Add a waypoint at the rest progress")
        if self.at(rest).abs().max() > 1e-12:
            raise ValueError(
                "The rest waypoint must have zero translation and rotation"
            )

    def load_at(self, t):
        return (t - self.rest_factor) / max(self.rest_factor, 1 - self.rest_factor)

    def at(self, t):
        slopes = self.poses.diff(dim=0) / self.factors.diff()[:, None]
        tangents = torch.cat(
            (slopes[:1], (slopes[:-1] + slopes[1:]) / 2, slopes[-1:]), dim=0
        )
        i = (
            torch.searchsorted(self.factors, t.contiguous(), right=True)
            .sub(1)
            .clamp(0, len(self.factors) - 2)
        )
        h = self.factors[i + 1] - self.factors[i]
        u = ((t - self.factors[i]) / h).clamp(0, 1)
        u = u[..., None]
        h = h[..., None]
        return (
            (2 * u**3 - 3 * u**2 + 1) * self.poses[i]
            + (u**3 - 2 * u**2 + u) * h * tangents[i]
            + (-2 * u**3 + 3 * u**2) * self.poses[i + 1]
            + (u**3 - u**2) * h * tangents[i + 1]
        )


def canonical(end):
    factors = end.new_tensor([0, 0.15, 0.35, 0.55, 0.75, 1])
    poses = factors[:, None] * end
    poses[:, 0] = end[0] * end.new_tensor([0, -0.35, -0.65, -0.3, 0.35, 1])
    poses[:, -1] = end[-1] * end.new_tensor([0, -0.15, 0.1, 0.65, 0.45, 1])
    poses[:, 1] = end[1] * end.new_tensor([0, 0.2, 0.42, 0.6, 0.8, 1])
    return Trajectory(factors, poses)
