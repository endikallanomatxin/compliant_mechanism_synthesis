from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random

import torch

from compliant_mechanism_synthesis.dataset.optimization import (
    CaseOptimizationConfig,
    optimize_case,
    sample_target_stiffness,
)
from compliant_mechanism_synthesis.dataset.primitives import (
    PRIMITIVE_LIBRARY,
    PrimitiveConfig,
    sample_primitive_design,
)


@dataclass(frozen=True)
class OfflineDatasetConfig:
    num_cases: int = 32
    seed: int = 7
    output_path: str = "artifacts/offline_dataset.pt"
    logdir: str = "runs/offline_dataset"
    primitive: PrimitiveConfig = PrimitiveConfig()
    optimization: CaseOptimizationConfig = field(default_factory=CaseOptimizationConfig)


def generate_offline_dataset(config: OfflineDatasetConfig | None = None) -> dict[str, object]:
    config = config or OfflineDatasetConfig()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.logdir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    cases = []
    for case_index in range(config.num_cases):
        primitive_kind = PRIMITIVE_LIBRARY[case_index % len(PRIMITIVE_LIBRARY)]
        primitive_seed = rng.randrange(0, 2**31)
        target_seed = rng.randrange(0, 2**31)
        initial_design = sample_primitive_design(
            primitive_kind,
            config=config.primitive,
            seed=primitive_seed,
        )
        target = sample_target_stiffness(
            initial_design,
            config=config.optimization,
            seed=target_seed,
        )
        result = optimize_case(
            primitive_kind=primitive_kind,
            initial_design=initial_design,
            target_stiffness=target,
            config=config.optimization,
            logdir=Path(config.logdir) / f"case_{case_index:04d}",
        )
        cases.append(result)

    payload = {
        "primitive_kind": [case.primitive_kind for case in cases],
        "target_stiffness": torch.stack([case.target_stiffness for case in cases], dim=0),
        "initial_positions": torch.stack([case.initial_design.positions for case in cases], dim=0),
        "initial_roles": torch.stack([case.initial_design.roles for case in cases], dim=0),
        "initial_adjacency": torch.stack([case.initial_design.adjacency for case in cases], dim=0),
        "optimized_positions": torch.stack([case.optimized_design.positions for case in cases], dim=0),
        "optimized_roles": torch.stack([case.optimized_design.roles for case in cases], dim=0),
        "optimized_adjacency": torch.stack([case.optimized_design.adjacency for case in cases], dim=0),
        "initial_loss": torch.tensor([case.initial_loss for case in cases], dtype=torch.float32),
        "best_loss": torch.tensor([case.best_loss for case in cases], dtype=torch.float32),
        "config": asdict(config),
    }
    torch.save(payload, output_path)
    return payload
