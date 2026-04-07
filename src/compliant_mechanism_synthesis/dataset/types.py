from __future__ import annotations

from dataclasses import dataclass

import torch

from compliant_mechanism_synthesis.roles import NodeRole


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(
            f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}"
        )


@dataclass(frozen=True)
class Structures:
    positions: torch.Tensor
    roles: torch.Tensor
    adjacency: torch.Tensor

    def to(self, device: torch.device | str) -> Structures:
        return Structures(
            positions=self.positions.to(device),
            roles=self.roles.to(device),
            adjacency=self.adjacency.to(device),
        )

    def index_select(self, batch_indices: torch.Tensor) -> Structures:
        return Structures(
            positions=self.positions.index_select(0, batch_indices),
            roles=self.roles.index_select(0, batch_indices),
            adjacency=self.adjacency.index_select(0, batch_indices),
        )

    def validate(self) -> None:
        _require_rank("positions", self.positions, 3)
        _require_rank("roles", self.roles, 2)
        _require_rank("adjacency", self.adjacency, 3)

        batch_size, num_nodes, spatial_dim = self.positions.shape
        if spatial_dim != 3:
            raise ValueError("positions must contain 3D coordinates")
        if self.roles.shape != (batch_size, num_nodes):
            raise ValueError("roles must have shape [batch, nodes]")
        if self.adjacency.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError("adjacency must have shape [batch, nodes, nodes]")
        if not torch.allclose(self.adjacency, self.adjacency.transpose(1, 2)):
            raise ValueError("adjacency must be symmetric")

        diagonal = torch.diagonal(self.adjacency, dim1=1, dim2=2)
        if not torch.allclose(diagonal, torch.zeros_like(diagonal)):
            raise ValueError("adjacency diagonal must be zero")

        role_values = {int(role) for role in self.roles.unique().tolist()}
        valid_roles = {int(NodeRole.FIXED), int(NodeRole.MOBILE), int(NodeRole.FREE)}
        if not role_values.issubset(valid_roles):
            raise ValueError("roles contain unknown values")

        fixed_count = (self.roles == int(NodeRole.FIXED)).sum(dim=1)
        mobile_count = (self.roles == int(NodeRole.MOBILE)).sum(dim=1)
        if torch.any(fixed_count < 3):
            raise ValueError(
                "3D rigid clamping needs at least 3 fixed anchor nodes per structure"
            )
        if torch.any(mobile_count < 3):
            raise ValueError(
                "3D rigid output needs at least 3 mobile anchor nodes per structure"
            )

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.positions.shape[1])

    def slice(self, index: int) -> Structures:
        return Structures(
            positions=self.positions[index : index + 1],
            roles=self.roles[index : index + 1],
            adjacency=self.adjacency[index : index + 1],
        )


@dataclass(frozen=True)
class Scaffolds:
    positions: torch.Tensor
    roles: torch.Tensor
    adjacency: torch.Tensor
    edge_primitive_types: torch.Tensor

    def to(self, device: torch.device | str) -> Scaffolds:
        return Scaffolds(
            positions=self.positions.to(device),
            roles=self.roles.to(device),
            adjacency=self.adjacency.to(device),
            edge_primitive_types=self.edge_primitive_types.to(device),
        )

    def index_select(self, batch_indices: torch.Tensor) -> Scaffolds:
        return Scaffolds(
            positions=self.positions.index_select(0, batch_indices),
            roles=self.roles.index_select(0, batch_indices),
            adjacency=self.adjacency.index_select(0, batch_indices),
            edge_primitive_types=self.edge_primitive_types.index_select(
                0, batch_indices
            ),
        )

    def validate(self) -> None:
        _require_rank("positions", self.positions, 3)
        _require_rank("roles", self.roles, 2)
        _require_rank("adjacency", self.adjacency, 3)
        _require_rank("edge_primitive_types", self.edge_primitive_types, 3)

        batch_size, num_nodes, spatial_dim = self.positions.shape
        if spatial_dim != 3:
            raise ValueError("scaffold positions must contain 3D coordinates")
        if self.roles.shape != (batch_size, num_nodes):
            raise ValueError("scaffold roles must have shape [batch, nodes]")
        if self.adjacency.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError("scaffold adjacency must have shape [batch, nodes, nodes]")
        if self.edge_primitive_types.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError(
                "edge_primitive_types must have shape [batch, nodes, nodes]"
            )
        if not torch.allclose(self.adjacency, self.adjacency.transpose(1, 2)):
            raise ValueError("scaffold adjacency must be symmetric")
        if not torch.equal(
            self.edge_primitive_types, self.edge_primitive_types.transpose(1, 2)
        ):
            raise ValueError("edge_primitive_types must be symmetric")

        adjacency_diagonal = torch.diagonal(self.adjacency, dim1=1, dim2=2)
        if not torch.allclose(adjacency_diagonal, torch.zeros_like(adjacency_diagonal)):
            raise ValueError("scaffold adjacency diagonal must be zero")
        primitive_diagonal = torch.diagonal(self.edge_primitive_types, dim1=1, dim2=2)
        if not torch.equal(primitive_diagonal, -torch.ones_like(primitive_diagonal)):
            raise ValueError("edge_primitive_types diagonal must be -1")

        role_values = {int(role) for role in self.roles.unique().tolist()}
        valid_roles = {int(NodeRole.FIXED), int(NodeRole.MOBILE), int(NodeRole.FREE)}
        if not role_values.issubset(valid_roles):
            raise ValueError("scaffold roles contain unknown values")

        fixed_count = (self.roles == int(NodeRole.FIXED)).sum(dim=1)
        mobile_count = (self.roles == int(NodeRole.MOBILE)).sum(dim=1)
        if torch.any(fixed_count < 1):
            raise ValueError("scaffolds need at least one fixed node per structure")
        if torch.any(mobile_count < 1):
            raise ValueError("scaffolds need at least one mobile node per structure")

        edge_mask = self.adjacency > 0.0
        if torch.any(self.edge_primitive_types[edge_mask] < 0):
            raise ValueError("every scaffold edge must have an assigned primitive type")
        if torch.any(self.edge_primitive_types[~edge_mask] != -1):
            raise ValueError("non-edges must use primitive type -1")

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])

    def slice(self, index: int) -> Scaffolds:
        return Scaffolds(
            positions=self.positions[index : index + 1],
            roles=self.roles[index : index + 1],
            adjacency=self.adjacency[index : index + 1],
            edge_primitive_types=self.edge_primitive_types[index : index + 1],
        )


@dataclass(frozen=True)
class Analyses:
    generalized_stiffness: torch.Tensor
    material_usage: torch.Tensor
    short_beam_penalty: torch.Tensor
    long_beam_penalty: torch.Tensor
    thin_beam_penalty: torch.Tensor
    thick_beam_penalty: torch.Tensor
    free_node_spacing_penalty: torch.Tensor

    def to(self, device: torch.device | str) -> Analyses:
        return Analyses(
            generalized_stiffness=self.generalized_stiffness.to(device),
            material_usage=self.material_usage.to(device),
            short_beam_penalty=self.short_beam_penalty.to(device),
            long_beam_penalty=self.long_beam_penalty.to(device),
            thin_beam_penalty=self.thin_beam_penalty.to(device),
            thick_beam_penalty=self.thick_beam_penalty.to(device),
            free_node_spacing_penalty=self.free_node_spacing_penalty.to(device),
        )

    def index_select(self, batch_indices: torch.Tensor) -> Analyses:
        return Analyses(
            generalized_stiffness=self.generalized_stiffness.index_select(
                0, batch_indices
            ),
            material_usage=self.material_usage.index_select(0, batch_indices),
            short_beam_penalty=self.short_beam_penalty.index_select(0, batch_indices),
            long_beam_penalty=self.long_beam_penalty.index_select(0, batch_indices),
            thin_beam_penalty=self.thin_beam_penalty.index_select(0, batch_indices),
            thick_beam_penalty=self.thick_beam_penalty.index_select(0, batch_indices),
            free_node_spacing_penalty=self.free_node_spacing_penalty.index_select(
                0, batch_indices
            ),
        )

    def validate(self, batch_size: int) -> None:
        _require_rank("generalized_stiffness", self.generalized_stiffness, 3)
        if self.generalized_stiffness.shape != (batch_size, 6, 6):
            raise ValueError("generalized_stiffness must have shape [batch, 6, 6]")
        for name in (
            "material_usage",
            "short_beam_penalty",
            "long_beam_penalty",
            "thin_beam_penalty",
            "thick_beam_penalty",
            "free_node_spacing_penalty",
        ):
            value = getattr(self, name)
            _require_rank(name, value, 1)
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [batch]")


@dataclass(frozen=True)
class OptimizedCases:
    raw_structures: Structures
    target_stiffness: torch.Tensor
    optimized_structures: Structures
    initial_loss: torch.Tensor
    best_loss: torch.Tensor
    last_analyses: Analyses
    scaffolds: Scaffolds | None = None

    def to(self, device: torch.device | str) -> OptimizedCases:
        return OptimizedCases(
            raw_structures=self.raw_structures.to(device),
            target_stiffness=self.target_stiffness.to(device),
            optimized_structures=self.optimized_structures.to(device),
            initial_loss=self.initial_loss.to(device),
            best_loss=self.best_loss.to(device),
            last_analyses=self.last_analyses.to(device),
            scaffolds=None if self.scaffolds is None else self.scaffolds.to(device),
        )

    def index_select(self, batch_indices: torch.Tensor) -> OptimizedCases:
        return OptimizedCases(
            raw_structures=self.raw_structures.index_select(batch_indices),
            target_stiffness=self.target_stiffness.index_select(0, batch_indices),
            optimized_structures=self.optimized_structures.index_select(batch_indices),
            initial_loss=self.initial_loss.index_select(0, batch_indices),
            best_loss=self.best_loss.index_select(0, batch_indices),
            last_analyses=self.last_analyses.index_select(batch_indices),
            scaffolds=(
                None
                if self.scaffolds is None
                else self.scaffolds.index_select(batch_indices)
            ),
        )

    def validate(self) -> None:
        self.raw_structures.validate()
        self.optimized_structures.validate()
        batch_size = self.raw_structures.batch_size
        _require_rank("target_stiffness", self.target_stiffness, 3)
        if self.target_stiffness.shape != (batch_size, 6, 6):
            raise ValueError("target_stiffness must have shape [batch, 6, 6]")
        for name in ("initial_loss", "best_loss"):
            value = getattr(self, name)
            _require_rank(name, value, 1)
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [batch]")
        self.last_analyses.validate(batch_size=batch_size)
        if self.scaffolds is not None:
            self.scaffolds.validate()
            if self.scaffolds.batch_size != batch_size:
                raise ValueError("scaffolds must match optimized case batch size")

    def slice(self, index: int) -> OptimizedCases:
        return OptimizedCases(
            raw_structures=self.raw_structures.slice(index),
            target_stiffness=self.target_stiffness[index : index + 1],
            optimized_structures=self.optimized_structures.slice(index),
            initial_loss=self.initial_loss[index : index + 1],
            best_loss=self.best_loss[index : index + 1],
            last_analyses=Analyses(
                generalized_stiffness=self.last_analyses.generalized_stiffness[
                    index : index + 1
                ],
                material_usage=self.last_analyses.material_usage[index : index + 1],
                short_beam_penalty=self.last_analyses.short_beam_penalty[
                    index : index + 1
                ],
                long_beam_penalty=self.last_analyses.long_beam_penalty[
                    index : index + 1
                ],
                thin_beam_penalty=self.last_analyses.thin_beam_penalty[
                    index : index + 1
                ],
                thick_beam_penalty=self.last_analyses.thick_beam_penalty[
                    index : index + 1
                ],
                free_node_spacing_penalty=self.last_analyses.free_node_spacing_penalty[
                    index : index + 1
                ],
            ),
            scaffolds=None if self.scaffolds is None else self.scaffolds.slice(index),
        )
