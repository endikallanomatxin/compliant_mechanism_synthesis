"""Persistent population search against batched nonlinear deformation paths."""

import hashlib
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import ClassVar

import torch

from .batching import DesignBatch
from .mechanics import DEFAULT_MATERIAL, Design
from .mechanics_work import measured
from .nonlinear import (
    DEFAULT_CONTINUATION,
    PathResult,
    PathState,
    configuration,
    energy,
    rotation,
    rotation_vector,
)
from .physical_diagnostics import path_diagnostics
from .prescribed_motion import trace_motion
from .trajectory import Trajectory, canonical


@dataclass(frozen=True)
class SearchConfig:
    dimension: ClassVar[int] = 3
    device: str = "cpu"
    seed: int = 7
    initial_neighbours: int = 4
    input_y: float = 0.003
    input_rotation: float = 0.0
    target_y: float = 0.006
    target_x: float = -0.002
    target_rotation: float = 0.12
    target_z: float = 0.004
    target_rx: float = 0.08
    learning_rate: float = 0.02
    adaptive_learning_rate: int = 1
    gradient_monotonic_target: float = 0.9
    learning_rate_adaptation: float = 1.0
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 1.0
    gradient_backtracks: int = 4
    gradient_steps_per_evolution: int = 3
    offspring_gradient_steps: int = 1
    gradient_geometry_rate: float = 1.0
    gradient_radius_rate: float = 1.0
    population: int = 32
    evaluation_batch_size: int = 16
    gradient_batch_size: int = 4
    topology_batch_union_ratio: float = 2.0
    linearization_cache_mb: float = 256.0
    reuse_parent_paths: int = 1
    parent_path_position_limit: float = 0.0005
    parent_path_radius_limit: float = 0.05
    elites: int = 4
    tournament_size: int = 3
    focused_fraction: float = 0.8
    parent_pool_fraction: float = 0.25
    local_mutation_factor: float = 0.5
    crossover_probability: float = 0.8
    mutation_scale: float = 0.08
    geometry_mutation_probability: float = 0.8
    radius_mutation_probability: float = 0.8
    geometry_scale: float = 0.01
    topology_mutation_probability: float = 0.6
    topology_mutation_edits: int = 2
    topology_add_probability: float = 0.5
    topology_distance_scale: float = 0.04
    topology_distance_power: float = 2.0
    initial_radius: float = 0.001
    radius_min: float = 0.0003
    radius_max: float = 0.002
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    position_scale: float = 0.01
    orientation_scale: float = 0.1
    diversity_weight: float = 0.2
    diversity_radius: float = 0.25
    diversity_geometry_weight: float = 1.0
    diversity_topology_weight: float = 1.0
    diversity_radius_weight: float = 0.5
    passive_support_weight: float = 0.1
    minimum_passive_support: float = 1000.0
    material_weight: float = 0.01
    collision_weight: float = 0.1
    collision_clearance: float = 0.0
    collision_subdivisions: int = 4
    collision_path_substeps: int = 2
    collision_joint_allowance: float = 2.0
    beam_length_weight: float = 0.1
    beam_length_min: float = 0.01
    beam_length_max: float = 0.05
    log_dir: str = "runs/"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "population",
            "initial_neighbours",
            "collision_subdivisions",
            "collision_path_substeps",
            "evaluation_batch_size",
            "gradient_batch_size",
            "topology_mutation_edits",
            "elites",
            "tournament_size",
            "gradient_backtracks",
            "adaptive_learning_rate",
            "reuse_parent_paths",
            "gradient_steps_per_evolution",
            "offspring_gradient_steps",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.gradient_steps_per_evolution < 0 or self.offspring_gradient_steps < 1:
            raise ValueError("Invalid hybrid cycle counts")
        if (
            min(
                self.gradient_geometry_rate,
                self.gradient_radius_rate,
            )
            < 0
        ):
            raise ValueError("Gradient rates must be nonnegative")
        if (
            min(self.position_weight, self.orientation_weight) < 0
            or self.position_weight + self.orientation_weight <= 0
        ):
            raise ValueError("At least one pose weight must be positive")
        if (
            min(self.position_scale, self.orientation_scale, self.diversity_radius) <= 0
            or self.diversity_weight < 0
        ):
            raise ValueError("Invalid objective scale or diversity weight")
        if (
            min(
                self.diversity_geometry_weight,
                self.diversity_topology_weight,
                self.diversity_radius_weight,
            )
            < 0
            or self.diversity_geometry_weight
            + self.diversity_topology_weight
            + self.diversity_radius_weight
            <= 0
        ):
            raise ValueError("At least one similarity component must be positive")

        if (
            not 0 <= self.focused_fraction <= 1
            or not 0 < self.parent_pool_fraction <= 1
            or not 0 < self.local_mutation_factor <= 1
        ):
            raise ValueError("Invalid focused search fractions")
        if (
            not all(
                0 <= v <= 1
                for v in (
                    self.geometry_mutation_probability,
                    self.radius_mutation_probability,
                    self.topology_mutation_probability,
                    self.topology_add_probability,
                )
            )
            or self.geometry_scale <= 0
        ):
            raise ValueError("Invalid mutation probability or geometry scale")
        if self.topology_distance_scale <= 0 or self.topology_distance_power < 0:
            raise ValueError("Invalid topology distance parameters")
        if self.linearization_cache_mb <= 0:
            raise ValueError("Linearization cache budget must be positive")
        if (
            self.reuse_parent_paths not in (0, 1)
            or min(self.parent_path_position_limit, self.parent_path_radius_limit) <= 0
        ):
            raise ValueError("Invalid parent path reuse settings")
        if self.topology_mutation_edits < 1 or self.topology_batch_union_ratio < 1:
            raise ValueError("Invalid topology edit budget or batch union ratio")
        if min(self.evaluation_batch_size, self.gradient_batch_size) < 1:
            raise ValueError("Evaluation batch size must be positive")
        if self.device not in ("auto", "cpu", "xpu"):
            raise ValueError("Device must be auto, cpu or xpu")
        if self.population < 2 or not 1 <= self.elites < self.population:
            raise ValueError(
                "Use at least two individuals and fewer elites than population size"
            )
        if not 0 < self.radius_min < self.initial_radius < self.radius_max:
            raise ValueError("Invalid radius bounds")
        if (
            self.adaptive_learning_rate not in (0, 1)
            or not 0 < self.gradient_monotonic_target <= 1
            or self.learning_rate_adaptation <= 0
            or not 0
            < self.learning_rate_min
            <= self.learning_rate
            <= self.learning_rate_max
        ):
            raise ValueError("Invalid adaptive learning-rate settings")
        if (
            not 1 <= self.initial_neighbours < 33
            or self.collision_weight < 0
            or self.collision_clearance < 0
            or self.collision_subdivisions < 1
            or self.collision_path_substeps < 1
            or self.collision_joint_allowance < 1
            or self.beam_length_weight < 0
            or not 0 < self.beam_length_min < self.beam_length_max
        ):
            raise ValueError("Invalid geometry-loss or nearest-neighbour settings")
        if self.passive_support_weight < 0 or self.minimum_passive_support <= 0:
            raise ValueError("Invalid stiffness objective parameters")
        if (
            self.learning_rate <= 0
            or self.mutation_scale <= 0
            or self.material_weight < 0
            or self.gradient_backtracks < 1
            or self.tournament_size < 1
        ):
            raise ValueError("Invalid search parameters")
        if (
            not 0 <= self.crossover_probability <= 1
            or not 0 <= self.topology_mutation_probability <= 1
        ):
            raise ValueError("Evolution probabilities must be in [0,1]")


DEFAULT_SEARCH = SearchConfig()


def example(
    dimension,
    device=None,
    radius=DEFAULT_SEARCH.initial_radius,
    seed=DEFAULT_SEARCH.seed,
    neighbours=DEFAULT_SEARCH.initial_neighbours,
):
    """Irregular nearest-neighbour seed, reproducible across devices."""
    if dimension not in (2, 3):
        raise ValueError("Dimension must be 2 or 3")
    if type(neighbours) is not int or not 1 <= neighbours < (
        33 if dimension == 3 else 30
    ):
        raise ValueError(
            "Choose an initial neighbour count smaller than the node count"
        )
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.tensor(
        [[0.0, -0.04], [0.0, 0.04], [0.12, -0.02], [0.12, 0.02]],
        dtype=torch.float64,
    )
    active = anchors.new_tensor([[0.032, -0.023], [0.032, 0.023]])
    if dimension == 3:
        anchors = anchors.new_tensor(
            [
                [0, -0.04],
                [0, 0.04],
                [-0.012, 0],
                [0.12, -0.02],
                [0.12, 0.02],
                [0.132, 0],
            ]
        )
        active = anchors.new_tensor([[0.032, -0.023], [0.032, 0.023], [0.046, 0]])
    points = []
    # Rejection spacing avoids nearly coincident nodes and tiny beam lengths.
    for _ in range(4096):
        candidate = anchors.new_tensor([0.014, -0.036]) + torch.rand(
            2, generator=generator, dtype=anchors.dtype
        ) * anchors.new_tensor([0.092, 0.072])
        existing = (
            torch.cat((anchors, active, torch.stack(points)))
            if points
            else torch.cat((anchors, active))
        )
        if (existing - candidate).norm(dim=-1).min() >= 0.009:
            points.append(candidate)
        if len(points) == 24:
            break
    if len(points) != 24:
        raise RuntimeError("Could not place the initial free nodes")
    p = torch.cat((anchors, torch.stack(points), active))
    if dimension == 3:
        z = torch.zeros(len(p), 1, dtype=p.dtype)
        z[6:30] = (torch.rand(24, 1, generator=generator, dtype=p.dtype) - 0.5) * 0.028
        z[[2, 5], 0] = 0.018
        z[-3:, 0] = p.new_tensor([-0.006, 0.006, 0.012])
        p = torch.cat((p, z), 1)
    attachments = 3 if dimension == 3 else 2
    roles = (
        ("fixed",) * attachments
        + ("passive",) * attachments
        + ("free",) * 24
        + ("active",) * attachments
    )
    pairs = torch.tensor(
        [
            (i, j)
            for i in range(len(p))
            for j in range(i + 1, len(p))
            if not (roles[i] == roles[j] and roles[i] != "free")
        ],
        dtype=torch.long,
    )
    lengths = (p[pairs[:, 1]] - p[pairs[:, 0]]).norm(dim=-1)
    chosen = torch.zeros(len(pairs), dtype=torch.bool)
    # A short-edge spanning tree ensures every attachment is connected.
    visited = torch.zeros(len(p), dtype=torch.bool)
    visited[0] = True
    while not bool(visited.all()):
        boundary = visited[pairs[:, 0]] ^ visited[pairs[:, 1]]
        edge = lengths.masked_fill(~boundary, torch.inf).argmin()
        chosen[edge] = True
        visited[pairs[edge]] = True
    # Local redundancy supplies moment-carrying routes, without a row/column grid.
    for node in range(len(p)):
        adjacent = (pairs == node).any(-1)
        nearest = lengths.masked_fill(~adjacent, torch.inf).argsort()[:neighbours]
        chosen[nearest] = True
    edges = pairs[chosen]
    return Design(
        p.to(device),
        roles,
        edges.to(device),
        p.new_full((len(edges),), radius).to(device),
        p.new_ones(len(edges)).to(device),
    )


@dataclass
class Population:
    positions: torch.Tensor
    radius_logits: torch.Tensor
    topology: torch.Tensor

    @property
    def size(self):
        return len(self.positions)

    def take(self, indices):
        return Population(
            *(getattr(self, name)[indices] for name in self.__dataclass_fields__)
        )

    def detached(self):
        return Population(
            *(
                getattr(self, name).detach().clone()
                for name in self.__dataclass_fields__
            )
        )

    @staticmethod
    def concatenate(*populations):
        return Population(
            *(
                torch.cat([getattr(p, name) for p in populations])
                for name in Population.__dataclass_fields__
            )
        )


class Search:
    def __init__(
        self,
        config=DEFAULT_SEARCH,
        continuation=DEFAULT_CONTINUATION,
        initial_design=None,
    ):
        self.config, self.continuation = config, continuation
        self.device_timings = {}
        if config.device == "auto":
            from .devices import choose_device

            reference = (
                initial_design
                if initial_design is not None
                else example(
                    config.dimension,
                    radius=config.initial_radius,
                    seed=config.seed,
                    neighbours=config.initial_neighbours,
                )
            )
            self.device, self.device_timings = choose_device(
                config.dimension,
                config.population,
                len(reference.positions),
                tuple(tuple(row) for row in reference.positions.tolist()),
                reference.roles,
                tuple(tuple(edge) for edge in reference.edges.tolist()),
                config.initial_radius,
                config.evaluation_batch_size,
                continuation.beam_segments,
            )
        else:
            self.device = config.device
        if self.device == "xpu" and not torch.xpu.is_available():
            raise ValueError("Intel XPU unavailable")
        if self.device == "xpu" and not torch.xpu.get_device_properties(0).has_fp64:
            raise ValueError("FEM requires float64")
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        seed = (
            example(
                config.dimension,
                self.device,
                config.initial_radius,
                seed=config.seed,
                neighbours=config.initial_neighbours,
            )
            if initial_design is None
            else replace(
                initial_design,
                positions=initial_design.positions.to(self.device),
                edges=initial_design.edges.to(self.device),
                radii=initial_design.radii.to(self.device),
                presence=initial_design.presence.to(self.device),
            )
        )
        seed.validate()
        if seed.dimension != config.dimension:
            raise ValueError("Design dimension does not match configuration")
        for role in ("fixed", "active", "passive"):
            points = seed.positions[[i for i, r in enumerate(seed.roles) if r == role]]
            if len(points) < 3:
                raise ValueError(
                    f"Provide at least three non-collinear nodes for the {role} body"
                )
            offsets = points - points[0]
            area = (
                torch.linalg.cross(offsets[:, None], offsets[None, :], dim=-1)
                .norm(dim=-1)
                .amax()
            )
            if area <= 1e-8 * offsets.square().sum(-1).amax():
                raise ValueError(
                    f"The {role} body's attachment nodes must be non-collinear"
                )
        pairs = [
            (i, j)
            for i in range(len(seed.positions))
            for j in range(i + 1, len(seed.positions))
            if not (
                seed.roles[i] == seed.roles[j]
                and seed.roles[i] in ("fixed", "passive", "active")
            )
        ]
        edges = torch.tensor(pairs, device=self.device)
        seed_edges = {
            tuple(sorted(e))
            for e, present in zip(
                seed.edges.cpu().tolist(), seed.presence.cpu().tolist(), strict=True
            )
            if present
        }
        topology = torch.tensor(
            [tuple(e) in seed_edges for e in pairs], device=self.device
        )
        self.specification = seed
        self.initial = replace(
            seed,
            edges=edges,
            radii=seed.positions.new_full((len(edges),), config.initial_radius),
            presence=topology.to(seed.positions.dtype),
        )
        self.free_mask = seed.positions.new_tensor([r == "free" for r in seed.roles])[
            :, None
        ]
        edge_radii = seed.positions.new_full((len(edges),), config.initial_radius)
        for edge, value in zip(seed.edges.cpu().tolist(), seed.radii, strict=True):
            edge_radii[pairs.index(tuple(sorted(edge)))] = value
        fractions = (edge_radii - config.radius_min) / (
            config.radius_max - config.radius_min
        )
        if not ((fractions > 0) & (fractions < 1)).all():
            raise ValueError("Beam radii must lie strictly inside optimization bounds")
        radius = torch.logit(fractions)
        population = Population(
            seed.positions.expand(config.population, -1, -1).clone(),
            radius.expand(config.population, -1).clone(),
            topology.expand(config.population, -1).clone(),
        )
        self.population = self._mutate(population)
        for name in Population.__dataclass_fields__:
            getattr(self.population, name)[0] = getattr(population, name)[0]
        self.target = seed.positions.new_zeros(6)
        self.target[1] = config.target_y
        self.target[0] = config.target_x
        self.target[-1] = config.target_rotation
        self.target[2] = config.target_z
        self.target[3] = config.target_rx
        self.trajectory = canonical(self.target)
        input_end = torch.zeros_like(self.target)
        input_end[1] = config.input_y
        input_end[-1] = config.input_rotation
        self.input_trajectory = Trajectory(
            self.trajectory.factors.clone(),
            self.trajectory.factors[:, None] * input_end,
        )

        self.evaluated_input = None
        self.evaluated_input_factors = None
        self.evaluated_poses = None
        self.evaluated_rest_factor = None
        self.hybrid_phase = 0
        self.iteration = 0
        self.scores = None
        self.snapshot_cache = {}
        self.path_cache = {}
        self.parent_paths = {}
        self.linearization_cache = {}
        self.linearization_cache_bytes = 0
        self.evaluated_factors = None
        self.evaluated_target = None
        self.last_diagnostics = {}
        self.gradient_history = []
        self.gradient_updates = 0
        self.current_learning_rate = config.learning_rate
        self.stagnation = 0
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(config.log_dir)

    @property
    def positions(self):
        return self.population.positions[0]

    def _rand(self, shape):
        return torch.rand(
            shape,
            dtype=self.initial.positions.dtype,
            device=self.device,
            generator=self.generator,
        )

    def _noise(self, shape):
        return torch.randn(
            shape,
            dtype=self.initial.positions.dtype,
            device=self.device,
            generator=self.generator,
        )

    def _bounded(self, p):
        return torch.where(
            self.free_mask.to(torch.bool), p.clamp(-0.15, 0.2), self.initial.positions
        )

    def topology_weights(self, positions):
        i, j = self.initial.edges.T
        distance = (positions[:, i] - positions[:, j]).norm(dim=-1)
        near = 2 / (
            1
            + (distance / self.config.topology_distance_scale).pow(
                self.config.topology_distance_power
            )
        )
        return near, 2 - near

    def _mutate_topology(self, topology, positions, factor):
        """Bounded distinct edits, balanced by operation rather than edge count."""
        add, remove = self.topology_weights(positions)
        result = topology.clone()
        touched = torch.zeros_like(topology)
        rows = torch.arange(len(topology), device=self.device)
        for _ in range(min(self.config.topology_mutation_edits, topology.shape[1])):
            available_add, available_remove = ~result & ~touched, result & ~touched
            adding = self._rand((len(topology),)) < self.config.topology_add_probability
            adding = torch.where(~available_remove.any(-1), True, adding)
            adding &= available_add.any(-1)
            available = torch.where(adding[:, None], available_add, available_remove)
            weights = torch.where(adding[:, None], add, remove)
            # Gumbel-max draws proportional to distance weights without CPU queries.
            uniform = self._rand(topology.shape).clamp(1e-12, 1 - 1e-12)
            logits = weights.clamp_min(1e-12).log() - (-uniform.log()).log()
            edge = logits.masked_fill(~available, -torch.inf).argmax(-1)
            apply = (
                self._rand((len(topology),))
                < self.config.topology_mutation_probability * factor
            ) & available.any(-1)
            result[rows, edge] ^= apply
            touched[rows, edge] |= apply
        return result

    def _mutate(self, population, factor=1.0):
        scale = (
            self.config.mutation_scale * min(3.0, 1 + self.stagnation * 0.1)
            if hasattr(self, "stagnation")
            else self.config.mutation_scale
        )
        scale = scale * factor
        positions = self._bounded(
            population.positions
            + self._noise(population.positions.shape)
            * scale
            * self.config.geometry_scale
            * (
                self._rand((population.size, len(self.initial.positions), 1))
                < self.config.geometry_mutation_probability
            )
            * self.free_mask
        )
        return Population(
            positions,
            population.radius_logits
            + self._noise(population.radius_logits.shape)
            * scale
            * (
                self._rand(population.radius_logits.shape)
                < self.config.radius_mutation_probability
            ),
            self._mutate_topology(population.topology, positions, factor),
        )

    def batch_design(self, population):
        c = self.config
        return DesignBatch(
            population.positions,
            self.initial.roles,
            self.initial.edges,
            c.radius_min
            + (c.radius_max - c.radius_min) * torch.sigmoid(population.radius_logits),
            population.topology.to(population.positions.dtype),
        )

    def design(self, index=0):
        return self.batch_design(self.population).item(index)

    def _valid(self, population):
        # Discrete graph validity is checked once per candidate, outside FEM.
        edges = self.initial.edges.cpu().tolist()
        roles = self.initial.roles
        n = len(roles)
        valid = []
        for topology in population.topology.cpu().tolist():
            parent = list(range(n))

            def root(i, parent=parent):
                while parent[i] != i:
                    i = parent[i]
                return i

            def union(i, j, parent=parent, root=root):
                parent[root(i)] = root(j)

            for role in ("fixed", "passive", "active"):
                nodes = [i for i, r in enumerate(roles) if r == role]
                for i in nodes[1:]:
                    union(nodes[0], i)
            for on, (i, j) in zip(topology, edges, strict=True):
                if on:
                    union(i, j)
            anchor = next(i for i, r in enumerate(roles) if r == "fixed")
            valid.append(all(root(i) == root(anchor) for i in range(n)))
        mask = torch.tensor(valid, dtype=torch.bool, device=self.device)
        p = population.positions
        delta = p[:, self.initial.edges[:, 1]] - p[:, self.initial.edges[:, 0]]
        return (
            mask
            & torch.isfinite(p).all(dim=(-2, -1))
            & (delta.norm(dim=-1) > 1e-7).all(-1)
        )

    @staticmethod
    def _path_key(design):
        digest = hashlib.sha256()
        effective_radii = design.radii * (design.presence != 0)
        for tensor in (design.positions, effective_radii, design.presence):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.digest()

    @classmethod
    def _path_keys(cls, batch):
        cpu = DesignBatch(
            batch.positions.detach().cpu(),
            batch.roles,
            batch.edges,
            batch.radii.detach().cpu(),
            batch.presence.detach().cpu(),
        )
        return [cls._path_key(cpu.item(i)) for i in range(cpu.size)]

    def _store_paths(self, design, path):
        if any(t.requires_grad for t in (design.positions, design.radii)):
            return
        # One transfer per batched tensor, outside the mechanics hot path.
        cpu = DesignBatch(
            design.positions.cpu(),
            design.roles,
            design.edges.cpu(),
            design.radii.cpu(),
            design.presence.cpu(),
        )
        completed = path.completed.cpu().tolist()
        records = [[] for _ in range(design.size)]
        linearizations = [[] for _ in range(design.size)]
        for state in path.states:
            local_linearizations = {}
            if state.linearization is not None:
                context, selected = state.linearization
                for local, index in enumerate(selected.cpu().tolist()):
                    local_linearizations[index] = context.take(
                        torch.tensor([local], device=design.positions.device)
                    )
            q = state.q.detach().cpu()
            factors = state.factors.cpu().tolist()
            accepted = state.accepted.cpu().tolist()
            residuals = state.residual.cpu().tolist()
            iterations = state.iterations.cpu().tolist()
            eigenvalues = state.minimum_eigenvalue.cpu().tolist()
            spectra = state.stiffness.detach().cpu()
            beam_meshes = (
                None if state.beam_nodes is None else state.beam_nodes.detach().cpu()
            )
            for i, ok in enumerate(accepted):
                if ok:
                    linearizations[i].append(local_linearizations.get(i))
                    records[i].append(
                        (
                            factors[i],
                            q[i].clone(),
                            residuals[i],
                            iterations[i],
                            eigenvalues[i],
                            spectra[i].clone(),
                            None
                            if beam_meshes is None
                            else beam_meshes[i, cpu.presence[i] != 0, 1:-1, :].clone(),
                        )
                    )
        for i in range(design.size):
            self.path_cache[self._path_key(cpu.item(i))] = (completed[i], records[i])
            # Contexts use the union graph solved for this batch; keep that
            # metadata to repack private systems on later regrouping.
            union_ids = torch.nonzero((design.presence != 0).any(0)).flatten()
            self.linearization_cache[self._path_key(cpu.item(i))] = (
                union_ids,
                linearizations[i],
            )

        self._trim_linearizations()

    def _trim_linearizations(self):
        def size(entry):
            total = entry[0].numel() * entry[0].element_size()
            for c in entry[1]:
                if c is None:
                    continue
                tensors = [
                    c.response,
                    c.system.factor,
                    c.system.valid,
                    c.system.scale,
                ]
                if c.system.pivots is not None:
                    tensors.append(c.system.pivots)
                if c.private_system is not None:
                    tensors.extend(
                        [
                            c.private_system.factor,
                            c.private_system.valid,
                            c.private_system.scale,
                        ]
                    )
                    if c.private_system.pivots is not None:
                        tensors.append(c.private_system.pivots)
                tensors.extend(
                    c.edge_map if isinstance(c.edge_map, tuple) else [c.edge_map]
                )
                total += sum(t.numel() * t.element_size() for t in tensors)
            return total

        sizes = {key: size(entry) for key, entry in self.linearization_cache.items()}
        total = sum(sizes.values())
        while (
            total > self.config.linearization_cache_mb * 1024**2
            and self.linearization_cache
        ):
            key = next(iter(self.linearization_cache))
            del self.linearization_cache[key]
            total -= sizes[key]
        self.linearization_cache_bytes = total

    def _cached_path(self, design):
        completed, records = self.path_cache[self._path_key(design)]
        states = []
        for (
            factor,
            stored_q,
            residual,
            iterations,
            eigen,
            spectrum,
            stored_beams,
        ) in records:
            q = stored_q.to(design.positions.device).detach().requires_grad_(True)
            positions, rotations = configuration(design, q)
            beam_nodes = None
            if stored_beams is not None:
                present_ids = torch.nonzero(design.presence != 0).flatten()
                compact = stored_beams.to(q.device)
                interior = compact.new_zeros(
                    len(design.edges),
                    compact.shape[-2],
                    6,
                ).index_copy(0, present_ids, compact)
                nodal = torch.cat((positions, rotations), -1).detach()
                beam_nodes = torch.cat(
                    (
                        nodal[design.edges[:, 0], None],
                        interior,
                        nodal[design.edges[:, 1], None],
                    ),
                    1,
                )
            if beam_nodes is None:
                value = energy(design, q)
            else:
                from .condensed_mechanics import trial_energy

                single = DesignBatch(
                    design.positions[None],
                    design.roles,
                    design.edges[design.presence != 0],
                    design.radii[None, design.presence != 0],
                    design.presence[None, design.presence != 0],
                )
                value = trial_energy(
                    single,
                    q[None],
                    beam_nodes[None, design.presence != 0, 1:-1, :],
                    self.continuation.beam_segments,
                    DEFAULT_MATERIAL,
                )[0]
            reaction = torch.autograd.grad(value, q)[0][-len(self.target) :]
            states.append(
                PathState(
                    factor,
                    q.detach(),
                    positions.detach(),
                    rotations.detach(),
                    reaction.detach(),
                    value.detach(),
                    residual,
                    iterations,
                    eigen,
                    beam_nodes,
                )
            )
        message = (
            "Nonlinear input-output path completed."
            if completed
            else "Input-output path stopped: nonlinear equilibrium or stability failed."
        )
        return PathResult(states, completed, message)

    def _register_parent_paths(self, children, parents):
        from .path_hint import PathHint

        if not self.config.reuse_parent_paths:
            return
        child = self.batch_design(children).detached()
        parent = self.batch_design(parents).detached()
        # Transfer metadata once outside mechanics, retaining only detached parent records.
        child_keys = self._path_keys(child)
        parent_keys = self._path_keys(parent)
        for i, (key, parent_key) in enumerate(
            zip(child_keys, parent_keys, strict=True)
        ):
            entry = self.path_cache.get(parent_key)
            if entry is None or not entry[0]:
                continue
            hint = PathHint(
                parent.positions[i].cpu(),
                parent.presence[i].cpu(),
                entry[1],
                self.continuation.beam_segments,
                parent.radii[i].cpu(),
            )
            if hint.compatible(
                child.positions[i].cpu(),
                child.presence[i].cpu(),
                self.continuation.beam_segments,
                self.config.parent_path_position_limit,
                child.radii[i].cpu(),
                self.config.parent_path_radius_limit,
            ):
                self.parent_paths[key] = hint

    def _prune_paths(self, extra=None):
        keys = set(self._path_keys(self.batch_design(self.population)))
        if extra is not None:
            keys.update(self._path_keys(self.batch_design(extra)))
        self.path_cache = {
            key: value for key, value in self.path_cache.items() if key in keys
        }
        self.linearization_cache = {
            key: value for key, value in self.linearization_cache.items() if key in keys
        }
        self.parent_paths = {
            key: value for key, value in self.parent_paths.items() if key in keys
        }
        self._trim_linearizations()

    def _evaluation_groups(self, population, limit):
        """Group similar binary graphs; limit union padding outside mechanics."""
        masks = [
            sum(1 << i for i, on in enumerate(row) if on)
            for row in population.topology.detach().cpu().tolist()
        ]
        entries = [
            self.path_cache.get(key)
            for key in self._path_keys(self.batch_design(population))
        ]
        signatures = [
            tuple(r[0] for r in entry[1]) if entry is not None and entry[0] else None
            for entry in entries
        ]
        remaining = set(range(population.size))
        groups = []
        while remaining:
            first = min(remaining)
            group, union, total = [first], masks[first], masks[first].bit_count()
            remaining.remove(first)
            while remaining and len(group) < limit:
                compatible = [
                    i for i in remaining if signatures[i] == signatures[first]
                ]
                if not compatible:
                    break
                candidate = min(
                    compatible, key=lambda i: ((union | masks[i]).bit_count(), i)
                )
                expanded = union | masks[candidate]
                next_total = total + masks[candidate].bit_count()
                ratio = expanded.bit_count() * (len(group) + 1) / max(next_total, 1)
                if ratio > self.config.topology_batch_union_ratio:
                    break
                group.append(candidate)
                remaining.remove(candidate)
                union, total = expanded, next_total
            groups.append(torch.tensor(group, dtype=torch.long, device=self.device))
        return groups

    @measured
    def _score(self, population):
        chunks = []
        diagnostics = []
        beam_slots = present_slots = 0
        started = time.perf_counter()
        groups = self._evaluation_groups(population, self.config.evaluation_batch_size)
        for indices in groups:
            chunk = population.take(indices)
            beam_slots += int(chunk.topology.any(0).sum()) * chunk.size
            present_slots += int(chunk.topology.sum())
            chunks.append(self._score_chunk(chunk))
            diagnostics.append(dict(self.last_diagnostics))
        self.last_diagnostics = {
            "batch_size": population.size,
            "peak_batch_size": max(len(ids) for ids in groups),
            "converged": sum(d["converged"] for d in diagnostics),
            "rejected": sum(d["rejected"] for d in diagnostics),
            "equilibrium_batches": sum(d["equilibrium_batches"] for d in diagnostics),
            "seconds": time.perf_counter() - started,
            "residual": max(d["residual"] for d in diagnostics),
            "evaluation_groups": len(groups),
            "beam_slots": beam_slots,
            "present_beam_slots": present_slots,
            "topology_padding_ratio": beam_slots / max(present_slots, 1),
        }
        self.writer.add_scalar(
            "batch/topology_padding_ratio",
            self.last_diagnostics["topology_padding_ratio"],
            self.iteration,
        )
        order = torch.cat(groups).argsort()
        return torch.cat(chunks)[order]

    @staticmethod
    def _batch_work(diagnostics):
        slots = sum(d.get("beam_slots", 0) for d in diagnostics)
        present = sum(d.get("present_beam_slots", 0) for d in diagnostics)
        mechanical = {
            key: sum(d.get(key, 0) for d in diagnostics)
            for key in (
                "factorizations",
                "cached_factorizations_reused",
                "adjoint_solves",
                "tangent_assemblies",
                "parent_hint_trials",
                "parent_hint_used",
                "parent_hint_fallbacks",
            )
        }
        return {
            **mechanical,
            "evaluation_groups": sum(
                d.get("evaluation_groups", 0) for d in diagnostics
            ),
            "beam_slots": slots,
            "present_beam_slots": present,
            "topology_padding_ratio": slots / max(present, 1),
        }

    def _score_chunk(self, population):
        started = time.perf_counter()
        valid = self._valid(population)
        indices = torch.nonzero(valid).flatten()
        scores = population.positions.new_full((population.size,), float("inf"))
        completed = torch.zeros_like(valid)
        evaluations = 0
        residual = 0.0
        if len(indices):
            design = self.batch_design(population.take(indices))
            differentiating = any(
                t.requires_grad for t in (design.positions, design.radii)
            )
            design_keys = self._path_keys(design)
            cached = [self.path_cache.get(key) for key in design_keys]
            hints = (
                [self.parent_paths.get(key) for key in design_keys]
                if self.config.reuse_parent_paths
                else []
            )
            complete = all(entry is not None and entry[0] for entry in cached)
            aligned = complete and all(
                tuple(r[0] for r in entry[1]) == tuple(r[0] for r in cached[0][1])
                for entry in cached
            )
            differentiate_stiffness = self.config.passive_support_weight > 0
            if differentiating and complete:
                # An inactive support hinge is identically zero locally. Avoid
                # retaining third-order energy graphs solely to multiply by zero.
                differentiate_stiffness &= any(
                    record[5][0] < self.config.minimum_passive_support
                    for entry in cached
                    for record in entry[1]
                )

            stored_linearizations = [
                self.linearization_cache.get(key) for key in design_keys
            ]
            linearized = (
                aligned
                and differentiating
                and all(
                    entry is not None
                    and len(entry[1]) == len(cached[i][1])
                    and all(c is not None for c in entry[1])
                    for i, entry in enumerate(stored_linearizations)
                )
            )
            if aligned and (not differentiating or linearized):
                from .cached_equilibrium import replay_path

                path = replay_path(
                    design,
                    [entry[1] for entry in cached],
                    stored_linearizations if linearized else None,
                    self.continuation,
                    self.input_trajectory,
                    differentiate_stiffness,
                    self.config.collision_weight > 0
                    or self.config.beam_length_weight > 0,
                )
            else:
                # Binary topology excludes absent beams from design derivatives.
                # Keep the full graph for keys/scoring; differentiate only its union
                # of present beams, rather than allocating graphs for absent edges.
                path = trace_motion(
                    design,
                    self.input_trajectory,
                    self.continuation,
                    cached=[entry[1] for entry in cached] if aligned else None,
                    differentiate_stiffness=differentiate_stiffness,
                    initial_path=hints
                    if any(hint is not None for hint in hints)
                    else None,
                )
                self._store_paths(design, path)
            evaluations = path.evaluations
            if not path.states:
                self.last_diagnostics = {
                    "batch_size": population.size,
                    "converged": 0,
                    "rejected": population.size,
                    "equilibrium_batches": evaluations,
                    "seconds": time.perf_counter() - started,
                    "residual": 0.0,
                }
                return scores
            dof = len(self.target)
            losses = torch.stack(
                [
                    self.pose_loss(s.q[:, -2 * dof : -dof], self.target_at(s.factors))
                    for j, s in enumerate(path.states)
                ]
            )
            factors = torch.stack([s.factors for s in path.states])
            objective = torch.trapezoid(losses, factors, dim=0)
            spectra = torch.stack([s.stiffness for s in path.states])
            weakest_support = spectra[..., 0].clamp_min(1e-12)
            support = (
                (1 - weakest_support / self.config.minimum_passive_support)
                .clamp_min(0)
                .square()
            )
            objective = objective + torch.trapezoid(
                self.config.passive_support_weight * support,
                factors,
                dim=0,
            )
            length = (
                design.positions[:, design.edges[:, 1]]
                - design.positions[:, design.edges[:, 0]]
            ).norm(dim=-1)
            material = (length * design.radii.square() * design.presence).sum(-1) / 1e-6
            i, j = torch.triu_indices(
                len(self.initial.positions),
                len(self.initial.positions),
                1,
                device=self.device,
            )
            distances = (design.positions[:, i] - design.positions[:, j]).norm(dim=-1)
            geometry = ((0.008 - distances).clamp_min(0) / 0.008).square().sum(-1)
            objective = objective + self.config.material_weight * material + geometry
            if self.config.collision_weight > 0 or self.config.beam_length_weight > 0:
                from .geometry_losses import collision_loss, length_loss, path_curves

                # Geometry sampling must not build rotations/interpolation graphs
                # for all admissible connections when most are absent.
                edge_ids = torch.nonzero((design.presence != 0).any(0)).flatten()
                geometry_design = replace(
                    design,
                    edges=design.edges[edge_ids],
                    radii=design.radii[:, edge_ids],
                    presence=design.presence[:, edge_ids],
                )
                curves = path_curves(
                    geometry_design,
                    [s.q for s in path.states],
                    [
                        None if s.beam_nodes is None else s.beam_nodes[:, edge_ids]
                        for s in path.states
                    ],
                    factors,
                    self.config,
                    self.input_trajectory,
                )
                if self.config.collision_weight > 0:
                    collisions, _, _ = collision_loss(
                        curves, geometry_design, self.config
                    )
                    objective = objective + self.config.collision_weight * collisions
                if self.config.beam_length_weight > 0:
                    lengths, _, _ = length_loss(curves, geometry_design, self.config)
                    objective = objective + self.config.beam_length_weight * lengths

            objective = torch.where(
                path.completed, objective, torch.full_like(objective, float("inf"))
            )
            scores = scores.scatter(0, indices, objective)
            completed[indices] = path.completed
            if bool(path.completed.any()):
                residual = float(path.states[-1].residual[path.completed].max())
        self.last_diagnostics = {
            "batch_size": population.size,
            "converged": int(completed.sum()),
            "rejected": int((~completed).sum()),
            "equilibrium_batches": evaluations,
            "seconds": time.perf_counter() - started,
            "residual": residual,
        }
        return scores

    def pose_loss(self, actual, target):
        c = self.config
        d = c.dimension
        position = (
            ((actual[..., :d] - target[..., :d]) / c.position_scale).square().mean(-1)
        )
        angles = rotation_vector(
            rotation(target[..., 3:]) @ rotation(actual[..., 3:]).transpose(-1, -2)
        )
        orientation = (angles / c.orientation_scale).square().mean(-1)
        return (c.position_weight * position + c.orientation_weight * orientation) / (
            c.position_weight + c.orientation_weight
        )

    def similarity(self, a, b=None):
        """Pairwise similarity with O(N² + N·features) memory, no N²·features arrays."""
        b = a if b is None else b
        c = self.config
        free = self.free_mask[:, 0].to(torch.bool)
        if bool(free.any()):
            reference = self.initial.positions[free]
            x = ((a.positions[:, free] - reference) / c.geometry_scale).flatten(1)
            y = ((b.positions[:, free] - reference) / c.geometry_scale).flatten(1)
            geometry = (
                x.square().sum(-1)[:, None] + y.square().sum(-1)[None] - 2 * x @ y.T
            ).clamp_min(0) / x.shape[-1]
        else:
            geometry = a.positions.new_zeros(a.size, b.size)
        mask_a = a.topology.to(a.positions.dtype)
        mask_b = b.topology.to(b.positions.dtype)
        overlap = mask_a @ mask_b.T
        count_a = mask_a.sum(-1)[:, None]
        count_b = mask_b.sum(-1)[None]
        topology = (count_a + count_b - 2 * overlap) / mask_a.shape[-1]
        ra = torch.sigmoid(a.radius_logits) * mask_a
        rb = torch.sigmoid(b.radius_logits) * mask_b
        # Compare radii only on common beams; latent radii on absent beams have
        # no physical meaning and must not manufacture structural novelty.
        radii = (
            ra.square() @ mask_b.T + mask_a @ rb.square().T - 2 * ra @ rb.T
        ).clamp_min(0) / overlap.clamp_min(1)
        total = (
            c.diversity_geometry_weight
            + c.diversity_topology_weight
            + c.diversity_radius_weight
        )
        distance = (
            c.diversity_geometry_weight * geometry
            + c.diversity_topology_weight * topology
            + c.diversity_radius_weight * radii
        ) / total
        return torch.exp(-distance / (2 * c.diversity_radius**2))

    def selection_metrics(self):
        similarity = self.similarity(self.population)
        similarity.fill_diagonal_(0)
        crowding = similarity.sum(-1) / max(1, self.population.size - 1)
        normalized = self.scores / self.scores.abs().median().clamp_min(1e-6)
        return normalized + self.config.diversity_weight * crowding, crowding

    def _select(self, population, scores, count, seed_population=None):
        finite = torch.nonzero(torch.isfinite(scores)).flatten()
        if not len(finite):
            raise ValueError("No candidate completed the input-output path")
        ranked = finite[torch.argsort(scores[finite])]
        if self.config.diversity_weight > 0 and count < len(finite):
            viable = population.take(finite)
            similarity = self.similarity(viable)
            penalty = scores.new_zeros(len(finite))
            if seed_population is not None:
                penalty = self.similarity(viable, seed_population).amax(-1)
            quality = (scores[finite] - scores[finite].min()) / scores[
                finite
            ].abs().median().clamp_min(1e-6)
            available = torch.ones(len(finite), dtype=torch.bool, device=scores.device)
            chosen = []
            for _ in range(min(count, len(finite))):
                merit = quality + self.config.diversity_weight * penalty
                index = torch.argmin(torch.where(available, merit, float("inf")))
                chosen.append(finite[index])
                available[index] = False
                penalty = torch.maximum(penalty, similarity[index])
            ranked = torch.stack(chosen)
        if len(ranked) < count:
            copies = ranked[
                torch.randint(
                    len(ranked),
                    (count - len(ranked),),
                    device=self.device,
                    generator=self.generator,
                )
            ]
            ranked = torch.cat((ranked, copies))
        chosen = ranked[:count]
        return population.take(chosen).detached(), scores[chosen].detach()

    def ensure_evaluated(self):
        if (
            self.scores is None
            or not torch.equal(self.input_trajectory.poses, self.evaluated_input)
            or not torch.equal(
                self.input_trajectory.factors, self.evaluated_input_factors
            )
            or not torch.equal(self.trajectory.factors, self.evaluated_factors)
            or not torch.equal(self.target, self.evaluated_target)
            or not torch.equal(self.trajectory.poses, self.evaluated_poses)
            or self.input_trajectory.rest_factor != self.evaluated_rest_factor
        ):
            if self.evaluated_target is not None and not torch.equal(
                self.target, self.evaluated_target
            ):
                if self.trajectory.rest_factor == 0:
                    self.trajectory = canonical(self.target)
                else:
                    self.trajectory.poses[-1] = self.target
                    self.trajectory.validate()
            self.snapshot_cache.clear()
            if (
                self.evaluated_input is None
                or not torch.equal(self.input_trajectory.poses, self.evaluated_input)
                or not torch.equal(
                    self.input_trajectory.factors, self.evaluated_input_factors
                )
                or self.evaluated_rest_factor != self.input_trajectory.rest_factor
            ):
                self.path_cache.clear()
                self.parent_paths.clear()
                self.linearization_cache.clear()
            scores = self._score(self.population)
            if bool(torch.isfinite(scores).any()):
                self.population, self.scores = self._select(
                    self.population, scores, self.config.population
                )
            else:
                self.scores = scores.detach()
            self._prune_paths()
            self.evaluated_input = self.input_trajectory.poses.detach().clone()
            self.evaluated_input_factors = (
                self.input_trajectory.factors.detach().clone()
            )
            self.evaluated_factors = self.trajectory.factors.detach().clone()
            self.evaluated_target = self.target.detach().clone()
            self.evaluated_poses = self.trajectory.poses.detach().clone()
            self.evaluated_rest_factor = self.input_trajectory.rest_factor

    def _offspring(self):
        total = self.config.population - self.config.elites
        fitness, _ = self.selection_metrics()
        parent_order = torch.argsort(fitness)
        focused = min(total, round(total * self.config.focused_fraction))
        count = total - focused
        pool = max(1, round(self.population.size * self.config.parent_pool_fraction))
        weights = (
            1
            / torch.arange(
                1, pool + 1, dtype=self.positions.dtype, device=self.device
            ).square()
        )
        parents = (
            torch.multinomial(
                weights, focused, replacement=True, generator=self.generator
            )
            if focused
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        parents = parent_order[parents]
        local = self._mutate(
            self.population.take(parents), self.config.local_mutation_factor
        )
        self._register_parent_paths(local, self.population.take(parents))
        self.parent_diagnostics = {
            "focused_children": focused,
            "exploratory_children": count,
            "parent_pool": pool,
            "parent_counts": torch.bincount(parents, minlength=self.population.size)
            .cpu()
            .tolist(),
        }
        if count == 0:
            return local

        def tournament():
            options = torch.randint(
                self.population.size,
                (count, self.config.tournament_size),
                device=self.device,
                generator=self.generator,
            )
            winners = fitness[options].argmin(-1)
            return self.population.take(
                options[torch.arange(count, device=self.device), winners]
            )

        a, b = tournament(), tournament()
        cross = self._rand((count, 1)) < self.config.crossover_probability
        node_mix = self._rand((count, len(self.initial.positions), 1))
        nodes = torch.where(
            cross[:, :, None],
            node_mix * a.positions + (1 - node_mix) * b.positions,
            a.positions,
        )
        edge_mix = self._rand(a.radius_logits.shape) < 0.5
        choose = cross & edge_mix
        child = Population(
            nodes,
            torch.where(choose, b.radius_logits, a.radius_logits),
            torch.where(choose, b.topology, a.topology),
        )
        return Population.concatenate(local, self._mutate(child))

    def _gradient_proposals(self, population=None):
        population = self.population if population is None else population
        gradients = []
        size = min(self.config.evaluation_batch_size, self.config.gradient_batch_size)
        groups = self._evaluation_groups(population, size)
        for indices in groups:
            part = population.take(indices)
            p = part.positions.detach().requires_grad_(True)
            r = part.radius_logits.detach().requires_grad_(True)
            scores = self._score(Population(p, r, part.topology))
            if bool(torch.isfinite(scores).any()):
                gradients.append(
                    tuple(
                        g.detach()
                        for g in torch.autograd.grad(
                            scores[torch.isfinite(scores)].sum(), (p, r)
                        )
                    )
                )
            else:
                gradients.append(tuple(torch.zeros_like(x) for x in (p, r)))
            del scores, p, r
        order = torch.cat(groups).argsort()
        gp, gr = (torch.cat([item[k] for item in gradients])[order] for k in range(2))
        p = population.positions
        r = population.radius_logits
        gp = gp * self.free_mask
        gp = (
            gp
            / gp.flatten(1).norm(dim=-1).clamp_min(1e-12)[:, None, None]
            * self.config.geometry_scale
        )
        gr = gr / gr.norm(dim=-1).clamp_min(1e-12)[:, None]
        proposals = []
        for k in range(self.config.gradient_backtracks):
            rate = self.current_learning_rate * 0.5**k
            proposals.append(
                Population(
                    self._bounded(
                        p.detach()
                        - rate * self.config.gradient_geometry_rate * gp.detach()
                    ),
                    r.detach() - rate * self.config.gradient_radius_rate * gr.detach(),
                    population.topology,
                )
            )
        combined = Population.concatenate(*proposals)
        self._register_parent_paths(
            combined, Population.concatenate(*([population] * len(proposals)))
        )
        return combined

    @measured
    def _adaptive_gradient(
        self, population=None, baseline_scores=None, phase="population"
    ):
        started = time.perf_counter()
        requested_rate = self.current_learning_rate
        initial_monotonic = 0
        proposals = (
            self._gradient_proposals()
            if population is None
            else self._gradient_proposals(population)
        )
        population = self.population if population is None else population
        baseline = self.scores if baseline_scores is None else baseline_scores
        size = population.size
        pending = torch.nonzero(torch.isfinite(baseline)).flatten()
        chosen = population.detached()
        scores = baseline.clone()
        diagnostics = []
        trials = []
        accepted_counts = []
        improved_counts = []
        accepted_rates = baseline.new_zeros(size)
        for k in range(self.config.gradient_backtracks):
            if not len(pending):
                break
            trial = proposals.take(k * size + pending)
            values = self._score(trial)
            diagnostics.append(dict(self.last_diagnostics))
            trials.append(len(pending))
            accepted = torch.isfinite(values) & (values <= baseline[pending])
            accepted_counts.append(int(accepted.sum()))
            improved_counts.append(int((accepted & (values < baseline[pending])).sum()))
            indices = pending[accepted]
            accepted_rates[indices] = requested_rate * 0.5**k
            if k == 0:
                margin = (
                    16
                    * torch.finfo(values.dtype).eps
                    * baseline[pending].abs().clamp_min(1)
                )
                initial_monotonic = int(
                    (
                        torch.isfinite(values) & (values < baseline[pending] - margin)
                    ).sum()
                )
            for name in Population.__dataclass_fields__:
                getattr(chosen, name)[indices] = getattr(trial, name)[accepted]
            scores[indices] = values[accepted]
            pending = pending[~accepted]
        self.last_diagnostics = {
            "batch_size": sum(d["batch_size"] for d in diagnostics),
            "peak_batch_size": max(d["peak_batch_size"] for d in diagnostics),
            "converged": sum(d["converged"] for d in diagnostics),
            "rejected": sum(d["rejected"] for d in diagnostics),
            "equilibrium_batches": sum(d["equilibrium_batches"] for d in diagnostics),
            "seconds": time.perf_counter() - started,
            "residual": max(d["residual"] for d in diagnostics),
            "gradient_trials": trials,
            "gradient_accepted": sum(accepted_counts),
            "gradient_accepted_counts": accepted_counts,
            "gradient_improved_counts": improved_counts,
        }
        self.last_diagnostics.update(self._batch_work(diagnostics))
        finite = torch.isfinite(baseline)
        before = float(baseline[finite].mean())
        after = float(scores[finite].mean())
        eligible = int(finite.sum())
        monotonic_fraction = initial_monotonic / eligible if eligible else 0.0
        next_rate = requested_rate
        if self.config.adaptive_learning_rate and eligible:
            exponent = self.config.learning_rate_adaptation * (
                monotonic_fraction - self.config.gradient_monotonic_target
            )
            factor = math.exp(max(math.log(0.5), min(math.log(1.25), exponent)))
            next_rate = min(
                self.config.learning_rate_max,
                max(self.config.learning_rate_min, requested_rate * factor),
            )
        self.current_learning_rate = next_rate
        record = {
            "iteration": self.iteration + 1,
            "phase": phase,
            "sequence": self.gradient_updates + 1,
            "requested_rate": requested_rate,
            "next_rate": next_rate,
            "initial_monotonic": initial_monotonic,
            "monotonic_fraction": monotonic_fraction,
            "monotonic_target": self.config.gradient_monotonic_target,
            "adaptive_rate": bool(self.config.adaptive_learning_rate),
            "mean_accepted_rate": float(accepted_rates[accepted_rates > 0].mean())
            if bool((accepted_rates > 0).any())
            else 0.0,
            "population": size,
            "eligible": eligible,
            "trials": trials,
            "accepted": accepted_counts,
            "improved": improved_counts,
            "unchanged": size - sum(accepted_counts),
            "objective_before": before,
            "objective_after": after,
            "improvement_percent": 100 * (before - after) / max(abs(before), 1e-12),
        }
        self.gradient_updates += 1
        self.gradient_history.append(record)
        self.gradient_history = self.gradient_history[-100:]
        for name in (
            "requested_rate",
            "mean_accepted_rate",
            "improvement_percent",
            "next_rate",
            "monotonic_fraction",
            "monotonic_target",
        ):
            self.writer.add_scalar("gradient/" + name, record[name], record["sequence"])
        self.writer.add_scalar(
            "gradient/initial_acceptance",
            accepted_counts[0] / max(eligible, 1),
            record["sequence"],
        )
        return chosen, scores

    def step(self, method):
        scheduled = method == "hybrid"
        if scheduled:
            cycle = self.config.gradient_steps_per_evolution + 1
            self.hybrid_phase %= cycle
            method = (
                "gradient"
                if self.hybrid_phase < self.config.gradient_steps_per_evolution
                else "evolution"
            )
            next_phase = (self.hybrid_phase + 1) % cycle
        if method not in ("gradient", "evolution"):
            raise ValueError("Unknown search method")
        self.ensure_evaluated()
        if not bool(torch.isfinite(self.scores).any()):
            raise ValueError("No converged population")
        before = float(self.scores.min())
        if method == "evolution":
            proposal = self._offspring()
            proposal_scores = self._score(proposal)
            stages = [dict(self.last_diagnostics)]
            valid_count = int(torch.isfinite(proposal_scores).sum())
            before_refinement = proposal_scores.detach().clone()
            refinement_steps = 0
            if valid_count:
                for _ in range(self.config.offspring_gradient_steps):
                    proposal, proposal_scores = self._adaptive_gradient(
                        proposal, proposal_scores, phase="offspring"
                    )
                    stages.append(dict(self.last_diagnostics))
                    self._prune_paths(proposal)
                    refinement_steps += 1
            self.last_diagnostics = {
                name: sum(d[name] for d in stages)
                for name in (
                    "batch_size",
                    "converged",
                    "rejected",
                    "equilibrium_batches",
                    "seconds",
                )
            } | {
                "peak_batch_size": max(d["peak_batch_size"] for d in stages),
                "residual": max(d["residual"] for d in stages),
                "offspring_gradient_steps": refinement_steps,
                "offspring_refined": valid_count,
                "offspring_improved": int((proposal_scores < before_refinement).sum()),
            }
            self.last_diagnostics.update(self._batch_work(stages))

        else:
            proposal, proposal_scores = self._adaptive_gradient()
        diagnostics = dict(self.last_diagnostics)
        if method == "evolution":
            diagnostics.update(self.parent_diagnostics)
        # Preserve explicit elites; fill the remainder from viable children
        # and surviving parents. Failed equilibria cannot enter the population.
        elites = self.population.take(
            torch.arange(self.config.elites, device=self.device)
        )
        pool = Population.concatenate(
            self.population.take(
                torch.arange(
                    self.config.elites, self.population.size, device=self.device
                )
            ),
            proposal,
        )
        pool_scores = torch.cat((self.scores[self.config.elites :], proposal_scores))
        rest, rest_scores = self._select(
            pool,
            pool_scores,
            self.config.population - self.config.elites,
            seed_population=elites,
        )
        combined = Population.concatenate(elites, rest)
        self.population, self.scores = self._select(
            combined,
            torch.cat((self.scores[: self.config.elites], rest_scores)),
            self.config.population,
        )
        self.snapshot_cache.clear()
        self._prune_paths()
        if scheduled:
            self.hybrid_phase = next_phase
        self.iteration += 1
        best = float(self.scores[0])
        self.stagnation = self.stagnation + 1 if best >= before else 0
        diversity = float(
            self.population.positions[:, self.free_mask[:, 0].to(torch.bool)]
            .std(dim=0)
            .mean()
        )
        self.last_diagnostics = diagnostics | {
            "diversity": diversity,
            "elites": self.config.elites,
            "method": method,
            "hybrid_phase": self.hybrid_phase,
        }
        self.writer.add_scalar("search/objective", best, self.iteration)
        self.writer.add_scalar("search/accepted", best < before, self.iteration)
        self.writer.add_scalar(
            "population/median", float(self.scores.median()), self.iteration
        )
        self.writer.add_scalar("population/diversity", diversity, self.iteration)
        _, crowding = self.selection_metrics()
        self.writer.add_scalar(
            "population/similarity", float(crowding.mean()), self.iteration
        )
        self.writer.add_scalar("search/hybrid_phase", self.hybrid_phase, self.iteration)
        self.writer.add_histogram("population/objectives", self.scores, self.iteration)
        for name in (
            "batch_size",
            "converged",
            "rejected",
            "seconds",
            "equilibrium_batches",
        ):
            self.writer.add_scalar("batch/" + name, diagnostics[name], self.iteration)
        self.writer.add_scalar("path/residual", diagnostics["residual"], self.iteration)
        if diagnostics.get("present_beam_slots", 0):
            self.writer.add_scalar(
                "batch/topology_padding_ratio",
                diagnostics["topology_padding_ratio"],
                self.iteration,
            )
        self.writer.flush()
        return self.snapshot()

    def tune(self, parameters, continuation=None):
        forbidden = {
            "dimension",
            "device",
            "seed",
            "initial_neighbours",
            "log_dir",
            "target_y",
            "target_x",
            "target_rotation",
            "target_z",
            "target_rx",
            "input_y",
            "input_rotation",
            "initial_radius",
            "radius_min",
            "radius_max",
        }
        if set(parameters) & forbidden or set(parameters) - set(
            self.config.__dataclass_fields__
        ):
            raise ValueError("These settings require restarting the search")
        config = replace(self.config, **parameters)
        mechanics = replace(self.continuation, **(continuation or {}))
        mechanics_changed = mechanics != self.continuation
        previous = self.config
        self.config = config
        self._trim_linearizations()
        if (
            config.learning_rate != previous.learning_rate
            or config.adaptive_learning_rate != previous.adaptive_learning_rate
        ):
            self.current_learning_rate = config.learning_rate
        self.current_learning_rate = min(
            config.learning_rate_max,
            max(config.learning_rate_min, self.current_learning_rate),
        )
        if config.population > self.population.size:
            count = config.population - self.population.size
            parents = torch.randint(
                self.population.size,
                (count,),
                device=self.device,
                generator=self.generator,
            )
            self.population = Population.concatenate(
                self.population, self._mutate(self.population.take(parents))
            )
        elif config.population < self.population.size:
            self.population, self.scores = self._select(
                self.population, self.scores, config.population
            )
        if mechanics_changed:
            self.path_cache.clear()
            self.parent_paths.clear()
            self.linearization_cache.clear()
        self.continuation = mechanics
        objective_fields = (
            "position_weight",
            "orientation_weight",
            "position_scale",
            "orientation_scale",
            "material_weight",
            "collision_weight",
            "collision_clearance",
            "collision_subdivisions",
            "collision_path_substeps",
            "collision_joint_allowance",
            "beam_length_weight",
            "beam_length_min",
            "beam_length_max",
            "passive_support_weight",
            "minimum_passive_support",
        )
        if (
            config.population != previous.population
            or mechanics_changed
            or any(
                getattr(config, name) != getattr(previous, name)
                for name in objective_fields
            )
        ):
            self.scores = None
        self.snapshot_cache.clear()
        self.hybrid_phase = 0
        self.writer.add_text("search/parameters", str(asdict(config)), self.iteration)
        self.ensure_evaluated()
        self._prune_paths()

    def target_at(self, factors):
        return self.trajectory.at(factors)

    def set_trajectory(self, factors, poses, rest_factor=0.0):
        trajectory = Trajectory(
            self.target.new_tensor(factors), self.target.new_tensor(poses), rest_factor
        )
        trajectory.validate()
        if trajectory.poses.shape[1] != len(self.target):
            raise ValueError("Incorrect pose dimension")
        self.trajectory = trajectory
        self.target = trajectory.poses[-1].clone()
        self.evaluated_target = self.target.clone()
        self.scores = None

    def set_input(self, factors, poses, rest_factor=0.0):
        trajectory = Trajectory(
            self.target.new_tensor(factors), self.target.new_tensor(poses), rest_factor
        )
        trajectory.validate()
        if trajectory.poses.shape[1] != len(self.target):
            raise ValueError("Incorrect active pose dimension")
        self.input_trajectory = trajectory
        self.scores = None

    def snapshot(self, index=0):
        self.ensure_evaluated()
        if not 0 <= index < self.population.size:
            raise ValueError("Invalid population index")
        if index in self.snapshot_cache:
            return self.snapshot_cache[index]
        design = self.design(index)
        key = self._path_key(design)
        if key not in self.path_cache:
            # Only disconnected, unevaluated designs have no stored FEM path.
            path = PathResult([], False, "Disconnected structure; no equilibrium path.")
        else:
            path = self._cached_path(design)
        batch = self.batch_design(self.population)
        length = (
            batch.positions[:, batch.edges[:, 1]]
            - batch.positions[:, batch.edges[:, 0]]
        ).norm(dim=-1)
        volume = (torch.pi * length * batch.radii.square() * batch.presence).sum(-1)
        selection_scores, crowding = self.selection_metrics()
        geometry_report = None
        if path.states:
            from .geometry_losses import collision_loss, length_loss, path_curves

            single = DesignBatch(
                design.positions[None],
                design.roles,
                design.edges,
                design.radii[None],
                design.presence[None],
            )
            with torch.no_grad():
                curves = path_curves(
                    single,
                    [s.q[None] for s in path.states],
                    [
                        None if s.beam_nodes is None else s.beam_nodes[None]
                        for s in path.states
                    ],
                    design.positions.new_tensor([[s.factor] for s in path.states]),
                    self.config,
                    self.input_trajectory,
                )
                collisions, depth, pairs = collision_loss(curves, single, self.config)
                lengths, shortest, longest = length_loss(curves, single, self.config)
                present = design.presence != 0
                geometry_report = {
                    "collision_loss": float(collisions[0]),
                    "collision_cost": float(
                        self.config.collision_weight * collisions[0]
                    ),
                    "colliding_pairs": int(pairs[0]),
                    "maximum_penetration_mm": float(depth[0] * 1000),
                    "length_loss": float(lengths[0]),
                    "length_cost": float(self.config.beam_length_weight * lengths[0]),
                    "minimum_beam_length_mm": float(shortest[0, present].min() * 1000),
                    "maximum_beam_length_mm": float(longest[0, present].max() * 1000),
                    "samples": len(curves),
                }
            if index == 0:
                for name in (
                    "collision_loss",
                    "collision_cost",
                    "colliding_pairs",
                    "maximum_penetration_mm",
                    "length_loss",
                    "length_cost",
                ):
                    self.writer.add_scalar(
                        "geometry/" + name, geometry_report[name], self.iteration
                    )
        result = {
            "search_config": asdict(self.config),
            "continuation_config": asdict(self.continuation),
            "hybrid_phase": self.hybrid_phase,
            "specification": {
                "positions": self.specification.positions.tolist(),
                "roles": self.specification.roles,
                "edges": self.specification.edges.tolist(),
                "radii": self.specification.radii.tolist(),
            },
            "analysis": "active_passive",
            "dimension": design.dimension,
            "device": self.device,
            "device_timings": self.device_timings,
            "device_name": torch.xpu.get_device_properties(0).name
            if self.device == "xpu"
            else "CPU",
            "positions": design.positions.tolist(),
            "roles": design.roles,
            "edges": design.edges.tolist(),
            "radii": design.radii.tolist(),
            "presence": design.presence.tolist(),
            "target": self.target.tolist(),
            "input_waypoints": {
                "factors": self.input_trajectory.factors.tolist(),
                "poses": self.input_trajectory.poses.tolist(),
                "rest_factor": self.input_trajectory.rest_factor,
            },
            "waypoints": {
                "factors": self.trajectory.factors.tolist(),
                "poses": self.trajectory.poses.tolist(),
                "rest_factor": self.trajectory.rest_factor,
            },
            "input_path": [
                {"factor": float(t), "output": pose.tolist()}
                for t, pose in zip(
                    torch.linspace(0, 1, 101, device=self.device),
                    self.input_trajectory.at(
                        torch.linspace(0, 1, 101, device=self.device)
                    ),
                    strict=True,
                )
            ],
            "target_path": [
                {"factor": float(t), "output": pose.tolist()}
                for t, pose in zip(
                    torch.linspace(0, 1, 101, device=self.device),
                    self.target_at(torch.linspace(0, 1, 101, device=self.device)),
                    strict=True,
                )
            ],
            "iteration": self.iteration,
            "objective": float(self.scores[index])
            if torch.isfinite(self.scores[index])
            else None,
            "selected_index": index,
            "physical_diagnostics": path_diagnostics(design, path.states),
            "geometry_diagnostics": geometry_report,
            "converged": path.converged,
            "message": path.message,
            "batch": dict(
                self.last_diagnostics,
                factor_cache_mb=self.linearization_cache_bytes / 1024**2,
            ),
            "gradient_history": self.gradient_history,
            "current_learning_rate": self.current_learning_rate,
            "population": [
                {
                    "index": i,
                    "objective": float(self.scores[i])
                    if torch.isfinite(self.scores[i])
                    else None,
                    "topology_changes": int(
                        (
                            self.population.topology[i] != self.population.topology[0]
                        ).sum()
                    ),
                    "gap_percent": float(
                        (self.scores[i] - self.scores[0])
                        / self.scores[0].abs().clamp_min(1e-12)
                        * 100
                    )
                    if torch.isfinite(self.scores[i])
                    else None,
                    "beams": int(self.population.topology[i].sum()),
                    "similarity": float(crowding[i]),
                    "selection_score": float(selection_scores[i])
                    if torch.isfinite(selection_scores[i])
                    else None,
                    "volume": float(volume[i]),
                    "positions": batch.positions[i].tolist(),
                    "presence": batch.presence[i].tolist(),
                }
                for i in range(self.population.size)
            ],
            "states": [
                {
                    "factor": s.factor,
                    "positions": s.positions.detach().tolist(),
                    "rotations": s.rotations.detach().tolist(),
                    "output": s.q[-2 * len(self.target) : -len(self.target)]
                    .detach()
                    .tolist(),
                    "input": s.q[-len(self.target) :].detach().tolist(),
                    "reaction": s.reaction.detach().tolist(),
                    "energy": float(s.energy.detach()),
                    "residual": s.residual,
                    "iterations": s.iterations,
                    "minimum_eigenvalue": s.minimum_eigenvalue,
                    "stiffness": self.path_cache[key][1][j][5].tolist(),
                    "beam_nodes": None
                    if s.beam_nodes is None
                    else s.beam_nodes.detach().tolist(),
                }
                for j, s in enumerate(path.states)
            ],
        }

        self.snapshot_cache[index] = result
        return result
