from __future__ import annotations

from dataclasses import dataclass

import torch

from compliant_mechanism_synthesis.roles import NodeRole


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


@dataclass(frozen=True)
class DesignTargets:
    generalized_stiffness: torch.Tensor

    def validate(self) -> None:
        _require_rank("generalized_stiffness", self.generalized_stiffness, 2)
        rows, cols = self.generalized_stiffness.shape
        if rows != cols:
            raise ValueError("generalized_stiffness must be square")
        if rows not in {3, 6}:
            raise ValueError("generalized_stiffness must be either 3x3 or 6x6")


@dataclass(frozen=True)
class GraphDesign:
    positions: torch.Tensor
    roles: torch.Tensor
    adjacency: torch.Tensor

    def validate(self) -> None:
        _require_rank("positions", self.positions, 2)
        _require_rank("roles", self.roles, 1)
        _require_rank("adjacency", self.adjacency, 2)
        num_nodes, spatial_dim = self.positions.shape
        if spatial_dim != 3:
            raise ValueError("positions must contain 3D coordinates")
        if self.roles.shape[0] != num_nodes:
            raise ValueError("roles must have one entry per node")
        if self.adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adjacency must be a square node-by-node matrix")
        if not torch.allclose(self.adjacency, self.adjacency.transpose(0, 1)):
            raise ValueError("adjacency must be symmetric")
        diagonal = torch.diagonal(self.adjacency)
        if not torch.allclose(diagonal, torch.zeros_like(diagonal)):
            raise ValueError("adjacency diagonal must be zero")

        role_values = {int(role) for role in self.roles.tolist()}
        valid_roles = {int(NodeRole.FIXED), int(NodeRole.MOBILE), int(NodeRole.FREE)}
        if not role_values.issubset(valid_roles):
            raise ValueError("roles contain unknown values")

        fixed_count = int((self.roles == int(NodeRole.FIXED)).sum().item())
        mobile_count = int((self.roles == int(NodeRole.MOBILE)).sum().item())
        if fixed_count < 3:
            raise ValueError("3D rigid clamping needs at least 3 fixed anchor nodes")
        if mobile_count < 3:
            raise ValueError("3D rigid output needs at least 3 mobile anchor nodes")
