"""Reproducible complete-search timings; snapshots are excluded explicitly."""

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .nonlinear import DEFAULT_CONTINUATION, ContinuationConfig
from .optimization import DEFAULT_SEARCH, Search, SearchConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    device: str = DEFAULT_SEARCH.device
    population: int = 4
    elites: int = 1
    repeats: int = 3
    warmups: int = 1
    steps: int = DEFAULT_CONTINUATION.steps
    staged_analysis: int = 0
    adaptive_beam_resolution: int = 0
    beam_segments: int = DEFAULT_CONTINUATION.beam_segments
    gradient_steps: int = 1
    gradient_backtracks: int = 1
    output: str = "runs/diagnostics/search-benchmark.json"


def run(config):
    if config.repeats < 1 or config.warmups < 0 or config.gradient_steps < 1:
        raise ValueError("Invalid benchmark counts")
    torch.set_num_threads(1)
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    sync = torch.xpu.synchronize if config.device == "xpu" else lambda: None
    for repetition in range(-config.warmups, config.repeats):
        search = Search(
            SearchConfig(
                device=config.device,
                staged_analysis=config.staged_analysis,
                adaptive_beam_resolution=config.adaptive_beam_resolution,
                population=config.population,
                elites=config.elites,
                gradient_steps_per_evolution=config.gradient_steps,
                gradient_backtracks=config.gradient_backtracks,
                log_dir=str(destination.with_suffix("")) + "/" + str(repetition),
            ),
            ContinuationConfig(steps=config.steps, beam_segments=config.beam_segments),
        )
        try:
            # Benchmark mechanics/search, excluding selected-viewer diagnostics
            # consistently for evaluation, gradient and evolution.
            search.snapshot = lambda *args, **kwargs: {}
            sync = (
                torch.xpu.synchronize
                if torch.device(search.device).type == "xpu"
                else lambda: None
            )
            row = {"repeat": repetition, "device": str(search.device)}
            sync()
            start = time.perf_counter()
            search.ensure_evaluated()
            sync()
            row["evaluation_seconds"] = time.perf_counter() - start
            row["initial_loss"] = float(search.scores.min())
            start = time.perf_counter()
            for _ in range(config.gradient_steps):
                search.step("gradient")
            sync()
            row["gradient_seconds"] = time.perf_counter() - start
            row["gradient_work"] = dict(search.last_diagnostics)
            start = time.perf_counter()
            search.step("evolution")
            sync()
            row["evolution_seconds"] = time.perf_counter() - start
            row["cycle_seconds"] = row["gradient_seconds"] + row["evolution_seconds"]
            row["best_loss"] = float(search.scores.min())
            row["analysis_level"] = search.analysis_level
            row["factor_cache_bytes"] = search.linearization_cache_bytes
            if repetition >= 0:
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            search.writer.close()
        del search
        gc.collect()
    result = {
        "configuration": asdict(config),
        "results": rows,
        "medians": {
            key: statistics.median(row[key] for row in rows)
            for key in (
                "evaluation_seconds",
                "gradient_seconds",
                "evolution_seconds",
                "cycle_seconds",
                "best_loss",
            )
        },
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["medians"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in asdict(BenchmarkConfig()).items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=value
        )
    run(BenchmarkConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
