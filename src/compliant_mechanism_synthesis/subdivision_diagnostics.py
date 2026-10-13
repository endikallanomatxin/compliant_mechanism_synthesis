"""Reproducible nonlinear 2D beam-network condensation experiment."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .subdivided import NetworkConfig, solve_network


@dataclass(frozen=True)
class DiagnosticConfig:
    device: str = "cpu"
    batch_size: int = 16
    segments: int = 4
    repeats: int = 3
    output: str = "runs/diagnostics/subdivision.json"


def run(config):
    if config.batch_size < 1 or config.repeats < 1:
        raise ValueError("Batch size and repeats must be positive")
    if config.device not in ("cpu", "xpu"):
        raise ValueError("Device must be cpu or xpu")
    torch.set_num_threads(1)
    p = (
        torch.tensor(
            [[[0.0, 0], [0.05, 0.015], [0.1, 0]]],
            dtype=torch.float64,
            device=config.device,
        )
        .expand(config.batch_size, -1, -1)
        .clone()
    )
    edges = torch.tensor([[0, 1], [1, 2]], device=config.device)
    radii = p.new_tensor([[0.001, 0.0008]]).expand(config.batch_size, -1).clone()
    mask = torch.tensor([[True] * 3, [False] * 3, [True] * 3], device=config.device)
    boundary = torch.cat((p, p.new_zeros(config.batch_size, 3, 1)), -1)
    boundary[:, 2, 1] = torch.linspace(
        0.001, 0.004, config.batch_size, dtype=p.dtype, device=p.device
    )
    boundary[:, 2, 2] = 0.04
    settings = NetworkConfig(segments=config.segments)
    synchronize = torch.xpu.synchronize if config.device == "xpu" else lambda: None
    solve_network(p, edges, radii, mask, boundary, settings)
    timings = []
    for _ in range(config.repeats):
        synchronize()
        started = time.perf_counter()
        state = solve_network(p, edges, radii, mask, boundary, settings)
        synchronize()
        timings.append(time.perf_counter() - started)
    result = {
        "configuration": asdict(config),
        "seconds": timings,
        "converged": state.converged.detach().cpu().tolist(),
        "residual": state.residual.detach().cpu().tolist(),
        "iterations": state.iterations.detach().cpu().tolist(),
        "global_dofs": p.shape[1] * 3,
        "internal_dofs_per_design": len(edges) * 3 * (config.segments - 1),
        "nodes": state.nodes.detach().cpu().tolist(),
        "beam_nodes": state.beam_nodes.detach().cpu().tolist(),
    }
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DiagnosticConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), default=value, type=type(value)
        )
    result = run(DiagnosticConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "seconds",
                    "converged",
                    "global_dofs",
                    "internal_dofs_per_design",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
