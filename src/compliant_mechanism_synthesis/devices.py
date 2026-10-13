"""Choose a device using actual two-body nonlinear input-output continuation."""

import time
from functools import lru_cache

import torch

from .batching import DesignBatch
from .nonlinear import DEFAULT_CONTINUATION, ContinuationConfig
from .prescribed_motion import trace_motion
from .trajectory import Trajectory


@lru_cache(maxsize=8)
def choose_device(
    dimension,
    population,
    nodes,
    positions,
    roles,
    edges,
    radius,
    batch_size,
    beam_segments=DEFAULT_CONTINUATION.beam_segments,
):
    if not torch.xpu.is_available() or not torch.xpu.get_device_properties(0).has_fp64:
        return "cpu", {"cpu": None}
    count = min(population, batch_size, 4)
    timings = {}
    for device in ("cpu", "xpu"):
        p = (
            torch.tensor(positions, dtype=torch.float64, device=device)
            .expand(count, -1, -1)
            .clone()
        )
        e = torch.tensor(edges, dtype=torch.long, device=device)
        r = p.new_full((count, len(edges)), radius)
        batch = DesignBatch(p, roles, e, r, torch.ones_like(r))
        dof = 3 if dimension == 2 else 6
        end = p.new_zeros(dof)
        end[1] = 0.001
        end[-1] = 0.02
        trajectory = Trajectory(p.new_tensor([0, 1]), torch.stack((end * 0, end)))
        config = ContinuationConfig(steps=2, beam_segments=beam_segments)

        def workload(batch=batch, trajectory=trajectory, config=config, device=device):
            path = trace_motion(batch, trajectory, config)
            if device == "xpu":
                torch.xpu.synchronize()
            if not bool(path.completed.all()):
                raise RuntimeError("Device benchmark did not converge")

        workload()
        started = time.perf_counter()
        workload()
        timings[device] = time.perf_counter() - started
    return min(timings, key=timings.get), timings
