"""Choose the compute device by timing this workload, rather than its label."""

import time
from functools import lru_cache

import torch

from .batching import DesignBatch, kernels
from .mechanics import DEFAULT_MATERIAL


@lru_cache(maxsize=8)
def choose_device(
    dimension, population, nodes, positions, roles, edges, radius, batch_size
):
    if not torch.xpu.is_available() or not torch.xpu.get_device_properties(0).has_fp64:
        return "cpu", {"cpu": None}
    # Benchmark the actual bounded batch, not an unbounded full population.
    population = min(population, batch_size)
    timings = {}
    for device in ("cpu", "xpu"):
        p = (
            torch.tensor(positions, dtype=torch.float64, device=device)
            .expand(population, -1, -1)
            .clone()
        )
        e = torch.tensor(edges, device=device)
        r = p.new_full((population, len(edges)), radius)
        batch = DesignBatch(p, roles, e, r, torch.ones_like(r))
        _, gradient, hessian = kernels(batch, DEFAULT_MATERIAL)
        dof = 3 if dimension == 2 else 6
        q = p.new_zeros(population, (sum(role == "free" for role in roles) + 1) * dof)
        eye = torch.eye(q.shape[1], dtype=q.dtype, device=device)

        def workload(
            gradient=gradient,
            hessian=hessian,
            p=p,
            r=r,
            batch=batch,
            q=q,
            eye=eye,
            device=device,
        ):
            g = gradient(p, r, batch.presence, q)
            h = hessian(p, r, batch.presence, q)
            torch.linalg.solve(h + 1e-9 * eye, g)
            if device == "xpu":
                torch.xpu.synchronize()

        workload()
        started = time.perf_counter()
        for _ in range(3):
            workload()
        timings[device] = (time.perf_counter() - started) / 3
    return min(timings, key=timings.get), timings
