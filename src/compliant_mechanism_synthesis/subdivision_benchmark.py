"""Compare condensed and full nonlinear continuation at equal beam resolution."""

import argparse
import faulthandler
import json
import math
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from .batching import DesignBatch
from .mechanics import Design
from .nonlinear import ContinuationConfig
from .optimization import example
from .prescribed_motion import trace_motion
from .subdivided import BeamMesh, reference_nodes
from .trajectory import Trajectory


def expand_design(design, segments):
    """Independent uncondensed reference: every private node is a free FEM node."""
    ids = torch.nonzero(design.presence != 0).flatten()
    edges = design.edges[ids]
    ref = reference_nodes(design.positions[edges], BeamMesh(segments))
    interior = ref[:, 1:-1, :].reshape(-1, design.dimension)
    positions = torch.cat((design.positions, interior), 0)
    private = torch.arange(
        len(design.positions), len(positions), device=positions.device
    ).reshape(len(edges), segments - 1)
    mesh_ids = torch.cat((edges[:, :1], private, edges[:, 1:]), -1)
    links = torch.stack((mesh_ids[:, :-1], mesh_ids[:, 1:]), -1).reshape(-1, 2)
    expanded = Design(
        positions,
        design.roles + ("free",) * len(interior),
        links,
        design.radii[ids].repeat_interleave(segments),
        positions.new_ones(len(links)),
    )
    return expanded, mesh_ids


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str = "cpu"
    batch_size: int = 1
    segments: int = 4
    steps: int = 4
    repeats: int = 3
    displacement: float = 0.0006
    output: str = "runs/diagnostics/condensation-benchmark.json"
    mode: str = "compare"
    timeout_seconds: float = 180.0


def save_results(config, results):
    from torch.utils.tensorboard import SummaryWriter

    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    writer = SummaryWriter(str(output.with_suffix("")))
    for name in ("condensed", "full", "single"):
        if name in results and "seconds" in results[name]:
            for k, seconds in enumerate(results[name]["seconds"]):
                writer.add_scalar(f"benchmark/{name}_seconds", seconds, k)
    if results.get("speedup") is not None:
        writer.add_scalar("benchmark/speedup", results["speedup"], 0)
    writer.close()


def run(config):
    if (
        config.device not in ("cpu", "xpu")
        or min(config.batch_size, config.repeats) < 1
    ):
        raise ValueError("Invalid benchmark settings")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("Benchmark timeout must be positive and finite")
    if config.mode not in ("compare", "condensed", "full", "single"):
        raise ValueError("Mode must be compare, condensed, full or single")
    if config.mode == "compare":
        # Each backend runs in a fresh process: RSS is not a cumulative high-water
        # mark contaminated by the preceding comparison's retained allocations.
        results = {"configuration": asdict(config)}
        for mode in ("condensed", "full", "single"):
            output = Path(config.output).with_name(
                Path(config.output).stem + "-" + mode + ".json"
            )
            worker = replace(config, mode=mode, output=str(output))
            command = [
                sys.executable,
                "-m",
                "compliant_mechanism_synthesis.subdivision_benchmark",
            ]
            for key, value in asdict(worker).items():
                command.extend(("--" + key.replace("_", "-"), str(value)))
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                results[mode] = {
                    "status": "timeout",
                    "timeout_seconds": config.timeout_seconds,
                }
                continue
            result = json.loads(output.read_text())
            results[mode] = result[mode]
            results["macro_unknowns"] = result["macro_unknowns"]
            results["full_unknowns"] = result["full_unknowns"]
        a, b = results["condensed"], results["full"]
        if a.get("status") == "timeout" or b.get("status") == "timeout":
            results["speedup"] = None
            results["same_factors"] = None
            results["endpoint_error"] = None
            save_results(config, results)
            return results
        if not all(a["completed"]) or not all(b["completed"]):
            raise RuntimeError("Comparison did not converge")
        torch.testing.assert_close(
            torch.tensor(a["body_path"], dtype=torch.float64),
            torch.tensor(b["body_path"], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-9,
        )
        torch.testing.assert_close(
            torch.tensor(a["final_beams"], dtype=torch.float64),
            torch.tensor(b["final_beams"], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-9,
        )
        results["same_factors"] = a["factors"] == b["factors"]
        results["speedup"] = sum(b["seconds"]) / sum(a["seconds"])
        results["endpoint_error"] = float(
            (
                torch.tensor(a["body_path"], dtype=torch.float64)
                - torch.tensor(b["body_path"], dtype=torch.float64)
            )
            .abs()
            .max()
        )
        save_results(config, results)
        return results
    torch.set_num_threads(1)
    design = example(3, config.device)
    expanded, mesh_ids = expand_design(design, config.segments)
    selected = expanded if config.mode == "full" else design
    settings = ContinuationConfig(
        steps=config.steps,
        beam_segments=config.segments if config.mode == "condensed" else 1,
    )
    batch = DesignBatch(
        selected.positions.expand(config.batch_size, -1, -1),
        selected.roles,
        selected.edges,
        selected.radii.expand(config.batch_size, -1),
        selected.presence.expand(config.batch_size, -1),
    )
    trajectory = Trajectory(
        design.positions.new_tensor([0, 1]),
        design.positions.new_tensor(
            [[0, 0, 0, 0, 0, 0], [0, config.displacement, 0, 0, 0, 0]]
        ),
    )
    sync = torch.xpu.synchronize if config.device == "xpu" else lambda: None

    def stage(name, **values):
        destination = Path(config.output).with_suffix(".phase.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"stage": name, "time": time.time(), **values}) + "\n"
        )

    stage("warmup")
    diagnostic_path = Path(config.output).with_suffix(".stack.txt")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_path.open("w") as diagnostic:
        faulthandler.dump_traceback_later(
            min(60, config.timeout_seconds / 2), repeat=True, file=diagnostic
        )
        try:
            trace_motion(batch, trajectory, settings)
            sync()
            stage("warmup_complete")
            if config.device == "xpu":
                torch.xpu.reset_peak_memory_stats()
            samples = []
            for _ in range(config.repeats):
                sync()
                start = time.perf_counter()
                path = trace_motion(batch, trajectory, settings)
                sync()
                samples.append(time.perf_counter() - start)
                stage("timing", seconds=samples)
        finally:
            faulthandler.cancel_dump_traceback_later()
    stage("export", seconds=samples)
    if config.mode == "condensed":
        beam_positions = path.states[-1].beam_nodes[..., :3]
    elif config.mode == "full":
        from .condensed_mechanics import nodal_nodes

        beam_positions = nodal_nodes(batch, path.states[-1].q)[..., :3][:, mesh_ids]
    else:
        beam_positions = design.positions.new_zeros(config.batch_size, 0, 3)
    measurement = {
        "seconds": samples,
        "completed": path.completed.detach().cpu().tolist(),
        "states": len(path.states),
        "equilibrium_batches": path.evaluations,
        "newton_iterations": sum(int(s.iterations.sum()) for s in path.states),
        "factors": [float(s.factors[0]) for s in path.states],
        "body_path": torch.stack([s.q[:, -12:] for s in path.states])
        .detach()
        .cpu()
        .tolist(),
        "final_beams": beam_positions.detach().cpu().tolist(),
        "peak_tensor_bytes": torch.xpu.max_memory_allocated()
        if config.device == "xpu"
        else None,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    results = {
        "configuration": asdict(config),
        config.mode: measurement,
        "macro_unknowns": sum(r == "free" for r in design.roles) * 6 + 6,
        "full_unknowns": sum(r == "free" for r in expanded.roles) * 6 + 6,
    }
    save_results(config, results)
    stage("complete", seconds=samples)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in asdict(BenchmarkConfig()).items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=value
        )
    result = run(BenchmarkConfig(**vars(parser.parse_args())))
    summary = {
        key: (
            {k: v for k, v in value.items() if k not in ("body_path", "final_beams")}
            if key in ("condensed", "full", "single")
            else value
        )
        for key, value in result.items()
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
