"""Capture a warmed nonlinear equilibrium with CPU/XPU operation timelines."""

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class TraceConfig:
    device: str = "cpu"
    dimension: ClassVar[int] = 3
    batch_size: int = 1
    input_displacement: float = 0.0003
    output: str = "runs/diagnostics/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TraceConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(value), default=value
        )
    config = TraceConfig(**vars(parser.parse_args()))
    if config.device not in ("cpu", "xpu") or config.batch_size < 1:
        parser.error("Use cpu/xpu, a positive batch size")
    import torch

    from .batching import DesignBatch
    from .nonlinear import ContinuationConfig
    from .optimization import example
    from .prescribed_motion import relax_equilibrium

    torch.set_num_threads(1)
    design = example(config.dimension, config.device)
    batch = DesignBatch(
        design.positions.expand(config.batch_size, -1, -1),
        design.roles,
        design.edges,
        design.radii.expand(config.batch_size, -1),
        design.presence.expand(config.batch_size, -1),
    )
    dof = 6
    q = design.positions.new_zeros(
        config.batch_size, (sum(r == "free" for r in design.roles) + 2) * dof
    )
    body = q.new_zeros(config.batch_size, dof)
    body[:, 1] = config.input_displacement
    mechanics = ContinuationConfig()
    relax_equilibrium(batch, body, q, mechanics)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if config.device == "xpu":
        torch.xpu.synchronize()
        activities.append(torch.profiler.ProfilerActivity.XPU)
    with torch.profiler.profile(activities=activities) as profile:
        with torch.profiler.record_function("nonlinear_equilibrium"):
            _, converged, *_ = relax_equilibrium(batch, body, q, mechanics)
        if config.device == "xpu":
            torch.xpu.synchronize()
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{config.device}-{config.dimension}d-equilibrium"
    profile.export_chrome_trace(str(output / f"{stem}-trace.json"))
    events = profile.key_averages()
    table = events.table(
        sort_by="self_device_time_total"
        if config.device == "xpu"
        else "self_cpu_time_total",
        row_limit=30,
    )
    (output / f"{stem}-profile.txt").write_text(table)
    summary = {
        "config": vars(config),
        "converged": converged.cpu().tolist(),
        "control_operations": [
            {"name": e.key, "count": e.count, "cpu_ms": e.cpu_time_total / 1000}
            for e in events
            if e.key in ("aten::nonzero", "aten::item", "aten::_local_scalar_dense")
        ],
    }
    (output / f"{stem}-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    print(table)


if __name__ == "__main__":
    main()
