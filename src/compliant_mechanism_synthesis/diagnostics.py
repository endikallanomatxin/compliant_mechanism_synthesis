"""Reproducible backend time/RAM/XPU-memory measurements, without a server.

Run with uv run python -m compliant_mechanism_synthesis.diagnostics.
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class DiagnosticsConfig:
    device: str = "cpu"
    dimension: ClassVar[int] = 3
    repetitions: int = 6
    steps: int = 4
    output: str = "runs/diagnostics/"


def process_memory():
    rows = Path("/proc/self/smaps_rollup").read_text().splitlines()[1:]
    return {
        key + "_kb": int(value.split()[0])
        for row in rows
        if ":" in row
        for key, value in [row.split(":", 1)]
        if key in ("Rss", "Pss", "Anonymous", "Private_Dirty")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DiagnosticsConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument("--" + field.name, type=type(value), default=value)
    config = DiagnosticsConfig(**vars(parser.parse_args()))
    if config.device not in ("cpu", "xpu") or config.repetitions < 1:
        parser.error("Use cpu/xpu, positive repetitions")
    records = [{"stage": "before_torch", **process_memory()}]
    import torch

    from . import batching, prescribed_motion
    from .nonlinear import ContinuationConfig
    from .optimization import Search, SearchConfig

    torch.set_num_threads(1)
    records.append({"stage": "after_import", **process_memory()})
    phases = {}
    original = batching.kernels

    def synchronize():
        if config.device == "xpu":
            torch.xpu.synchronize()

    def instrument(fn, label):
        def call(*args):
            synchronize()
            start = time.perf_counter()
            result = fn(*args)
            synchronize()
            phases[label] = phases.get(label, 0) + time.perf_counter() - start
            return result

        return call

    def kernels(*args, **kwargs):
        return tuple(
            instrument(fn, label)
            for fn, label in zip(
                original(*args, **kwargs),
                ("energy", "residual", "hessian"),
                strict=True,
            )
        )

    batching.kernels = kernels
    prescribed_motion.kernels = kernels
    search = None
    try:
        search = Search(
            SearchConfig(device=config.device),
            ContinuationConfig(steps=config.steps),
        )
        operations = [("initial", search.snapshot), ("cached", search.snapshot)]
        operations += [
            (f"evolution_{i}", lambda: search.step("evolution"))
            for i in range(config.repetitions)
        ]
        operations.append(("gradient", lambda: search.step("gradient")))
        for label, fn in operations:
            phases.clear()
            synchronize()
            if config.device == "xpu":
                torch.xpu.reset_peak_memory_stats()
            start = time.perf_counter()
            result = fn()
            synchronize()
            elapsed = time.perf_counter() - start
            del result
            gc.collect()
            row = {
                "stage": label,
                "seconds": elapsed,
                "phases": dict(phases),
                **process_memory(),
                "cache_entries": len(search.snapshot_cache),
                "path_cache_entries": len(search.path_cache),
                "path_cache_tensor_bytes": sum(
                    q.numel() * q.element_size()
                    for _, records in search.path_cache.values()
                    for _, q, *_ in records
                ),
            }
            if config.device == "xpu":
                row.update(
                    gpu_allocated=torch.xpu.memory_allocated(),
                    gpu_reserved=torch.xpu.memory_reserved(),
                    gpu_peak_allocated=torch.xpu.max_memory_allocated(),
                )
            records.append(row)
            print(json.dumps(row), flush=True)
        search.writer.close()
        search = None
        gc.collect()
        if config.device == "xpu":
            torch.xpu.empty_cache()
        records.append({"stage": "after_delete_and_gc", **process_memory()})
    finally:
        batching.kernels = original
        prescribed_motion.kernels = original
        if search is not None:
            search.writer.close()
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{config.device}-profile.json").write_text(
        json.dumps(
            {"config": asdict(config), "torch": torch.__version__, "records": records},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
