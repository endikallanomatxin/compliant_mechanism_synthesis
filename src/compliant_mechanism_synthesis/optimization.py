"""Persistent population search against batched nonlinear deformation paths."""

import hashlib
import time
from dataclasses import asdict, dataclass, replace

import torch

from .batching import DesignBatch
from .mechanics import Design
from .nonlinear import (
    DEFAULT_CONTINUATION,
    PathResult,
    PathState,
    configuration,
    energy,
)
from .soft_modes import pose_delta, trace_soft
from .trajectory import Trajectory, canonical


@dataclass(frozen=True)
class SearchConfig:
    dimension: int = 2
    device: str = "cpu"
    seed: int = 7
    target_y: float = 0.006
    target_x: float = -0.002
    target_rotation: float = 0.12
    target_z: float = 0.004
    target_rx: float = 0.08
    learning_rate: float = 0.02
    gradient_backtracks: int = 4
    gradient_steps_per_evolution: int = 3
    evolution_steps_per_cycle: int = 1
    gradient_geometry_rate: float = 1.0
    gradient_radius_rate: float = 1.0
    gradient_presence_rate: float = 1.0
    population: int = 32
    evaluation_batch_size: int = 16
    elites: int = 4
    tournament_size: int = 3
    focused_fraction: float = 0.8
    parent_pool_fraction: float = 0.25
    local_mutation_factor: float = 0.5
    crossover_probability: float = 0.8
    mutation_scale: float = 0.08
    geometry_mutation_probability: float = 0.8
    radius_mutation_probability: float = 0.8
    presence_mutation_probability: float = 0.8
    geometry_scale: float = 0.01
    topology_mutation_probability: float = 0.06
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
    diversity_presence_weight: float = 0.5
    mode_separation_weight: float = 0.1
    transverse_stiffness_weight: float = 0.1
    minimum_transverse_stiffness: float = 1000.0
    material_weight: float = 0.01
    log_dir: str = "runs/"

    def __post_init__(self):
        import math

        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "population",
            "evaluation_batch_size",
            "elites",
            "tournament_size",
            "gradient_backtracks",
            "gradient_steps_per_evolution",
            "evolution_steps_per_cycle",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.gradient_steps_per_evolution < 0 or self.evolution_steps_per_cycle < 1:
            raise ValueError("Invalid hybrid cycle counts")
        if (
            min(
                self.gradient_geometry_rate,
                self.gradient_radius_rate,
                self.gradient_presence_rate,
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
                self.diversity_presence_weight,
            )
            < 0
            or self.diversity_geometry_weight
            + self.diversity_topology_weight
            + self.diversity_radius_weight
            + self.diversity_presence_weight
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
                    self.presence_mutation_probability,
                )
            )
            or self.geometry_scale <= 0
        ):
            raise ValueError("Invalid mutation probability or geometry scale")
        if self.evaluation_batch_size < 1:
            raise ValueError("Evaluation batch size must be positive")
        if self.device not in ("auto", "cpu", "xpu"):
            raise ValueError("Device must be auto, cpu or xpu")
        if (
            self.dimension not in (2, 3)
            or self.population < 2
            or not 1 <= self.elites < self.population
        ):
            raise ValueError(
                "Use at least two individuals and fewer elites than population size"
            )
        if not 0 < self.radius_min < self.initial_radius < self.radius_max:
            raise ValueError("Invalid radius bounds")
        if (
            min(self.mode_separation_weight, self.transverse_stiffness_weight) < 0
            or self.minimum_transverse_stiffness <= 0
        ):
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


def example(dimension, device=None, radius=DEFAULT_SEARCH.initial_radius):
    if dimension not in (2, 3):
        raise ValueError("Dimension must be 2 or 3")
    # Four longitudinal stations and three transverse rows provide many
    # independent bending paths, rather than a small chain of two flexures.
    points = [[0.0, -0.04], [0.0, 0.04], [0.12, -0.02], [0.12, 0.02]]
    for column, x in enumerate((0.022, 0.047, 0.072, 0.097)):
        for row, y in enumerate((-0.036, 0.0, 0.036)):
            points.append([x, y])
    p = torch.tensor(points, dtype=torch.float64, device=device)
    if dimension == 3:
        z = [0.0] * 4 + [
            0.009 * (-1 if (column + row) % 2 else 1)
            for column in range(4)
            for row in range(3)
        ]
        p = torch.cat((p, p.new_tensor(z)[:, None]), 1)

    def node(column, row):
        return 4 + column * 3 + row

    links = []
    for column in range(4):
        for row in range(2):
            links.append([node(column, row), node(column, row + 1)])
    for column in range(3):
        for row in range(3):
            links.append([node(column, row), node(column + 1, row)])
        for row in range(2):
            links.extend(
                [
                    [node(column, row), node(column + 1, row + 1)],
                    [node(column, row + 1), node(column + 1, row)],
                ]
            )
    links.extend([[0, 4], [0, 5], [1, 5], [1, 6], [13, 2], [14, 2], [14, 3], [15, 3]])
    edges = torch.tensor(links, device=p.device)
    return Design(
        p,
        ("fixed", "fixed", "mobile", "mobile") + ("free",) * 12,
        edges,
        p.new_full((len(edges),), radius),
        p.new_ones(len(edges)),
    )


@dataclass
class Population:
    positions: torch.Tensor
    radius_logits: torch.Tensor
    presence_logits: torch.Tensor
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

            reference = example(config.dimension, radius=config.initial_radius)
            self.device, self.device_timings = choose_device(
                config.dimension,
                config.population,
                len(reference.positions),
                tuple(tuple(row) for row in reference.positions.tolist()),
                reference.roles,
                tuple(tuple(edge) for edge in reference.edges.tolist()),
                config.initial_radius,
                config.evaluation_batch_size,
            )
        else:
            self.device = config.device
        if self.device == "xpu" and not torch.xpu.is_available():
            raise ValueError("Intel XPU unavailable")
        if self.device == "xpu" and not torch.xpu.get_device_properties(0).has_fp64:
            raise ValueError("FEM requires float64")
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        seed = (
            example(config.dimension, self.device, config.initial_radius)
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
        pairs = [
            (i, j)
            for i in range(len(seed.positions))
            for j in range(i + 1, len(seed.positions))
            if not (
                seed.roles[i] == seed.roles[j] and seed.roles[i] in ("fixed", "mobile")
            )
        ]
        edges = torch.tensor(pairs, device=self.device)
        seed_edges = {tuple(sorted(e)) for e in seed.edges.cpu().tolist()}
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
            radius.new_full((config.population, len(edges)), 3.0),
            topology.expand(config.population, -1).clone(),
        )
        self.population = self._mutate(population)
        for name in Population.__dataclass_fields__:
            getattr(self.population, name)[0] = getattr(population, name)[0]
        self.target = seed.positions.new_zeros(3 if config.dimension == 2 else 6)
        self.target[1] = config.target_y
        self.target[0] = config.target_x
        self.target[-1] = config.target_rotation
        if config.dimension == 3:
            self.target[2] = config.target_z
            self.target[3] = config.target_rx
        self.trajectory = canonical(self.target)
        self.evaluated_poses = None
        self.evaluated_rest_factor = None
        self.hybrid_phase = 0
        self.iteration = 0
        self.scores = None
        self.snapshot_cache = {}
        self.path_cache = {}
        self.evaluated_factors = None
        self.evaluated_target = None
        self.last_diagnostics = {}
        self.gradient_history = []
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

    def _mutate(self, population, factor=1.0):
        scale = (
            self.config.mutation_scale * min(3.0, 1 + self.stagnation * 0.1)
            if hasattr(self, "stagnation")
            else self.config.mutation_scale
        )
        scale = scale * factor
        return Population(
            self._bounded(
                population.positions
                + self._noise(population.positions.shape)
                * scale
                * self.config.geometry_scale
                * (
                    self._rand((population.size, len(self.initial.positions), 1))
                    < self.config.geometry_mutation_probability
                )
                * self.free_mask
            ),
            population.radius_logits
            + self._noise(population.radius_logits.shape)
            * scale
            * (
                self._rand(population.radius_logits.shape)
                < self.config.radius_mutation_probability
            ),
            population.presence_logits
            + self._noise(population.presence_logits.shape)
            * scale
            * (
                self._rand(population.presence_logits.shape)
                < self.config.presence_mutation_probability
            ),
            population.topology
            ^ (
                self._rand(population.topology.shape)
                < self.config.topology_mutation_probability * factor
            ),
        )

    def batch_design(self, population):
        c = self.config
        return DesignBatch(
            population.positions,
            self.initial.roles,
            self.initial.edges,
            c.radius_min
            + (c.radius_max - c.radius_min) * torch.sigmoid(population.radius_logits),
            population.topology * torch.sigmoid(population.presence_logits),
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

            for role in ("fixed", "mobile"):
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
        if any(
            t.requires_grad for t in (design.positions, design.radii, design.presence)
        ):
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
        for state in path.states:
            q = state.q.detach().cpu()
            factors = state.factors.cpu().tolist()
            accepted = state.accepted.cpu().tolist()
            residuals = state.residual.cpu().tolist()
            iterations = state.iterations.cpu().tolist()
            eigenvalues = state.minimum_eigenvalue.cpu().tolist()
            for i, ok in enumerate(accepted):
                if ok:
                    records[i].append(
                        (
                            factors[i],
                            q[i].clone(),
                            residuals[i],
                            iterations[i],
                            eigenvalues[i],
                            state.stiffness[i].detach().cpu().clone(),
                            state.mode[i].detach().cpu().clone(),
                        )
                    )
        for i in range(design.size):
            self.path_cache[self._path_key(cpu.item(i))] = (completed[i], records[i])

    def _cached_path(self, design):
        completed, records = self.path_cache[self._path_key(design)]
        states = []
        for factor, stored_q, residual, iterations, eigen, spectrum, mode in records:
            q = stored_q.to(design.positions.device).detach().requires_grad_(True)
            positions, rotations = configuration(design, q)
            value = energy(design, q)
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
                )
            )
        message = (
            "Natural compliant path completed."
            if completed
            else "Natural path stopped: free equilibrium, stability or mode separation failed."
        )
        return PathResult(states, completed, message)

    def _prune_paths(self):
        keys = set(self._path_keys(self.batch_design(self.population)))
        self.path_cache = {
            key: value for key, value in self.path_cache.items() if key in keys
        }

    def _score(self, population):
        chunks = []
        diagnostics = []
        started = time.perf_counter()
        for start in range(0, population.size, self.config.evaluation_batch_size):
            chunk = population.take(
                slice(start, start + self.config.evaluation_batch_size)
            )
            chunks.append(self._score_chunk(chunk))
            diagnostics.append(dict(self.last_diagnostics))
        self.last_diagnostics = {
            "batch_size": population.size,
            "peak_batch_size": min(population.size, self.config.evaluation_batch_size),
            "converged": sum(d["converged"] for d in diagnostics),
            "rejected": sum(d["rejected"] for d in diagnostics),
            "equilibrium_batches": sum(d["equilibrium_batches"] for d in diagnostics),
            "seconds": time.perf_counter() - started,
            "residual": max(d["residual"] for d in diagnostics),
        }
        return torch.cat(chunks)

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
                t.requires_grad
                for t in (design.positions, design.radii, design.presence)
            )
            cached = [self.path_cache.get(key) for key in self._path_keys(design)]
            if (
                all(entry is not None and entry[0] for entry in cached)
                and not differentiating
            ):
                # Detached cached states need no tangent or Newton evaluation.
                from .soft_modes import SoftPath, SoftState

                records = [entry[1] for entry in cached]
                path = SoftPath(
                    [
                        SoftState(
                            design.positions.new_tensor([r[j][0] for r in records]),
                            torch.stack([r[j][1] for r in records]).to(self.device),
                            design.positions.new_tensor([r[j][2] for r in records]),
                            torch.tensor(
                                [r[j][3] for r in records], device=self.device
                            ),
                            design.positions.new_tensor([r[j][4] for r in records]),
                            torch.ones(
                                design.size, dtype=torch.bool, device=self.device
                            ),
                            torch.stack([r[j][5] for r in records]).to(self.device),
                            torch.stack([r[j][6] for r in records]).to(self.device),
                        )
                        for j in range(len(records[0]))
                    ],
                    torch.ones(design.size, dtype=torch.bool, device=self.device),
                    0,
                )
            else:
                path = trace_soft(
                    design,
                    self.trajectory,
                    self.continuation,
                    cached=[entry[1] for entry in cached]
                    if all(entry is not None and entry[0] for entry in cached)
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
                    self.pose_loss(s.q[:, -dof:], self.target_at(s.factors))
                    for j, s in enumerate(path.states)
                ]
            )
            factors = torch.stack([s.factors for s in path.states])
            objective = torch.trapezoid(losses, factors, dim=0)
            spectra = torch.stack([s.stiffness for s in path.states])
            second = spectra[..., 1].clamp_min(1e-12)
            separation = (spectra[..., 0].clamp_min(0) / second).square()
            support = (
                (1 - second / self.config.minimum_transverse_stiffness)
                .clamp_min(0)
                .square()
            )
            objective = objective + torch.trapezoid(
                self.config.mode_separation_weight * separation
                + self.config.transverse_stiffness_weight * support,
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
        angles = pose_delta(actual, target, d)[..., d:]
        if d == 2:
            angles = torch.atan2(angles.sin(), angles.cos())
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
        pa = torch.sigmoid(a.presence_logits) * mask_a
        pb = torch.sigmoid(b.presence_logits) * mask_b
        presence = (
            pa.square().sum(-1)[:, None] + pb.square().sum(-1)[None] - 2 * pa @ pb.T
        ).clamp_min(0) / (count_a + count_b - overlap).clamp_min(1)
        total = (
            c.diversity_geometry_weight
            + c.diversity_topology_weight
            + c.diversity_radius_weight
            + c.diversity_presence_weight
        )
        distance = (
            c.diversity_geometry_weight * geometry
            + c.diversity_topology_weight * topology
            + c.diversity_radius_weight * radii
            + c.diversity_presence_weight * presence
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
            raise ValueError("No candidate completed the natural compliant path")
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
            or not torch.equal(self.trajectory.factors, self.evaluated_factors)
            or not torch.equal(self.target, self.evaluated_target)
            or not torch.equal(self.trajectory.poses, self.evaluated_poses)
            or self.trajectory.rest_factor != self.evaluated_rest_factor
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
                self.evaluated_factors is None
                or not torch.equal(self.trajectory.factors, self.evaluated_factors)
                or not torch.equal(self.trajectory.poses, self.evaluated_poses)
                or self.evaluated_rest_factor != self.trajectory.rest_factor
            ):
                self.path_cache.clear()
            scores = self._score(self.population)
            if bool(torch.isfinite(scores).any()):
                self.population, self.scores = self._select(
                    self.population, scores, self.config.population
                )
            else:
                self.scores = scores.detach()
            self._prune_paths()
            self.evaluated_factors = self.trajectory.factors.detach().clone()
            self.evaluated_target = self.target.detach().clone()
            self.evaluated_poses = self.trajectory.poses.detach().clone()
            self.evaluated_rest_factor = self.trajectory.rest_factor

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
            torch.where(choose, b.presence_logits, a.presence_logits),
            torch.where(choose, b.topology, a.topology),
        )
        return Population.concatenate(local, self._mutate(child))

    def _gradient_proposals(self):
        gradients = []
        for start in range(0, self.population.size, self.config.evaluation_batch_size):
            part = self.population.take(
                slice(start, start + self.config.evaluation_batch_size)
            )
            p = part.positions.detach().requires_grad_(True)
            r = part.radius_logits.detach().requires_grad_(True)
            a = part.presence_logits.detach().requires_grad_(True)
            scores = self._score(Population(p, r, a, part.topology))
            if bool(torch.isfinite(scores).any()):
                gradients.append(
                    tuple(
                        g.detach()
                        for g in torch.autograd.grad(
                            scores[torch.isfinite(scores)].sum(), (p, r, a)
                        )
                    )
                )
            else:
                gradients.append(tuple(torch.zeros_like(x) for x in (p, r, a)))
            del scores, p, r, a
        gp, gr, ga = (torch.cat([item[k] for item in gradients]) for k in range(3))
        p = self.population.positions
        r = self.population.radius_logits
        a = self.population.presence_logits
        gp = gp * self.free_mask
        gp = (
            gp
            / gp.flatten(1).norm(dim=-1).clamp_min(1e-12)[:, None, None]
            * self.config.geometry_scale
        )
        gr = gr / gr.norm(dim=-1).clamp_min(1e-12)[:, None]
        ga = ga / ga.norm(dim=-1).clamp_min(1e-12)[:, None]
        proposals = []
        for k in range(self.config.gradient_backtracks):
            rate = self.config.learning_rate * 0.5**k
            proposals.append(
                Population(
                    self._bounded(
                        p.detach()
                        - rate * self.config.gradient_geometry_rate * gp.detach()
                    ),
                    r.detach() - rate * self.config.gradient_radius_rate * gr.detach(),
                    a.detach()
                    - rate * self.config.gradient_presence_rate * ga.detach(),
                    self.population.topology,
                )
            )
        return Population.concatenate(*proposals)

    def _adaptive_gradient(self):
        proposals = self._gradient_proposals()
        size = self.population.size
        pending = torch.arange(size, device=self.device)
        chosen = self.population.detached()
        scores = self.scores.clone()
        diagnostics = []
        trials = []
        accepted_counts = []
        improved_counts = []
        accepted_rates = self.scores.new_zeros(size)
        started = time.perf_counter()
        for k in range(self.config.gradient_backtracks):
            if not len(pending):
                break
            trial = proposals.take(k * size + pending)
            values = self._score(trial)
            diagnostics.append(dict(self.last_diagnostics))
            trials.append(len(pending))
            accepted = torch.isfinite(values) & (values <= self.scores[pending])
            accepted_counts.append(int(accepted.sum()))
            improved_counts.append(
                int((accepted & (values < self.scores[pending])).sum())
            )
            indices = pending[accepted]
            accepted_rates[indices] = self.config.learning_rate * 0.5**k
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
            "gradient_accepted": size - len(pending),
            "gradient_accepted_counts": accepted_counts,
            "gradient_improved_counts": improved_counts,
        }
        finite = torch.isfinite(self.scores)
        before = float(self.scores[finite].mean())
        after = float(scores[finite].mean())
        record = {
            "iteration": self.iteration + 1,
            "requested_rate": self.config.learning_rate,
            "mean_accepted_rate": float(accepted_rates[accepted_rates > 0].mean())
            if bool((accepted_rates > 0).any())
            else 0.0,
            "population": size,
            "trials": trials,
            "accepted": accepted_counts,
            "improved": improved_counts,
            "unchanged": len(pending),
            "objective_before": before,
            "objective_after": after,
            "improvement_percent": 100 * (before - after) / max(abs(before), 1e-12),
        }
        self.gradient_history.append(record)
        self.gradient_history = self.gradient_history[-100:]
        for name in ("requested_rate", "mean_accepted_rate", "improvement_percent"):
            self.writer.add_scalar(
                "gradient/" + name, record[name], record["iteration"]
            )
        self.writer.add_scalar(
            "gradient/initial_acceptance",
            accepted_counts[0] / size,
            record["iteration"],
        )
        return chosen, scores

    def step(self, method):
        scheduled = method == "hybrid"
        if scheduled:
            cycle = (
                self.config.gradient_steps_per_evolution
                + self.config.evolution_steps_per_cycle
            )
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
        self.writer.flush()
        return self.snapshot()

    def tune(self, parameters, continuation=None):
        forbidden = {
            "dimension",
            "device",
            "seed",
            "log_dir",
            "target_y",
            "target_x",
            "target_rotation",
            "target_z",
            "target_rx",
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
        self.continuation = mechanics
        objective_fields = (
            "position_weight",
            "orientation_weight",
            "position_scale",
            "orientation_scale",
            "material_weight",
            "mode_separation_weight",
            "transverse_stiffness_weight",
            "minimum_transverse_stiffness",
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
            "analysis": "minimum_stiffness",
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
            "waypoints": {
                "factors": self.trajectory.factors.tolist(),
                "poses": self.trajectory.poses.tolist(),
                "rest_factor": self.trajectory.rest_factor,
            },
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
            "converged": path.converged,
            "message": path.message,
            "batch": self.last_diagnostics,
            "gradient_history": self.gradient_history,
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
                    "output": s.q[-len(self.target) :].detach().tolist(),
                    "reaction": s.reaction.detach().tolist(),
                    "energy": float(s.energy.detach()),
                    "residual": s.residual,
                    "iterations": s.iterations,
                    "minimum_eigenvalue": s.minimum_eigenvalue,
                    "stiffness": self.path_cache[key][1][j][5].tolist(),
                    "soft_mode": self.path_cache[key][1][j][6].tolist(),
                    "stiffness_ratio": float(
                        self.path_cache[key][1][j][5][0]
                        / self.path_cache[key][1][j][5][1].clamp_min(1e-12)
                    ),
                }
                for j, s in enumerate(path.states)
            ],
        }

        self.snapshot_cache[index] = result
        return result
