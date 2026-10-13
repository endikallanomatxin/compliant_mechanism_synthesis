"""Measure local intrabeam throughput separately from global continuation."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .optimization import example
from .subdivided import BeamMesh, chain_derivatives, condense, reference_nodes


@dataclass(frozen=True)
class ScalingConfig:
    batches: str = "1,4,16"
    segments: int = 4
    repeats: int = 3
    output: str = "runs/diagnostics/beam-scaling.json"


def run(config):
    from torch.utils.tensorboard import SummaryWriter

    batches = [int(value) for value in config.batches.split(",")]
    if min(batches) < 1 or config.repeats < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(1)
    design = example(3)
    ref = reference_nodes(design.positions[design.edges], BeamMesh(config.segments))
    delta = design.positions[design.edges[:, 1]] - design.positions[design.edges[:, 0]]
    length = delta.norm(dim=-1)
    normal = torch.stack(
        (-delta[:, 1], delta[:, 0], torch.zeros_like(delta[:, 0])), -1
    ) / delta[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fraction = torch.linspace(0, 1, config.segments + 1, dtype=ref.dtype)
    nodes = torch.cat((ref, torch.zeros_like(ref)), -1)
    nodes[..., :3] += (
        normal[:, None, :]
        * (length[:, None, None] * 0.001)
        * torch.sin(torch.pi * fraction)[None, :, None]
    )
    results = []
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(destination.with_suffix("")))
    for device in ("cpu", "xpu"):
        if device == "xpu" and not torch.xpu.is_available():
            continue
        for count in batches:
            reference = ref.to(device)[None].expand(count, -1, -1, -1).contiguous()
            state = nodes.to(device)[None].expand(count, -1, -1, -1).contiguous()
            radii = design.radii.to(device)[None].expand(count, -1).contiguous()
            sync = torch.xpu.synchronize if device == "xpu" else lambda: None

            def kernel(reference=reference, state=state, radii=radii):
                return chain_derivatives(reference, state, radii)

            def condensed(kernel=kernel):
                _, g, h = kernel()
                return condense(g, h, 6)

            row = {
                "device": device,
                "batch": count,
                "beams": count * len(design.edges),
                "segments": count * len(design.edges) * config.segments,
            }
            for name, fn in [
                ("kernel", kernel),
                ("kernel_and_condensation", condensed),
            ]:
                fn()
                sync()
                samples = []
                for _ in range(config.repeats):
                    sync()
                    start = time.perf_counter()
                    value = fn()
                    sync()
                    samples.append(time.perf_counter() - start)
                row[name + "_seconds"] = samples
                writer.add_scalar(
                    "scaling/" + device + "/" + name,
                    sum(samples) / len(samples),
                    row["segments"],
                )
                if name == "kernel":
                    expected = chain_derivatives(
                        ref[None], nodes[None], design.radii[None]
                    )
                    for measured, target in zip(value, expected, strict=True):
                        torch.testing.assert_close(
                            measured[:1].cpu(), target, rtol=1e-8, atol=1e-5
                        )
                else:
                    if not bool(value.info.eq(0).all()):
                        raise RuntimeError("Local condensation failed")
            results.append(row)
            print(json.dumps(row), flush=True)
            del reference, state, radii, value
    writer.close()
    destination.write_text(
        json.dumps({"configuration": asdict(config), "results": results}, indent=2)
        + "\n"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in asdict(ScalingConfig()).items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=value
        )
    run(ScalingConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
