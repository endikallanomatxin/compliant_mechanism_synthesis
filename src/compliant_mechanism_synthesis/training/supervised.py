from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    dataset_path: str
    batch_size: int = 16
    num_steps: int = 10_000


def run_supervised_training(config: SupervisedTrainingConfig) -> None:
    raise NotImplementedError(
        "Supervised training is still intentionally deferred until we validate "
        "that the offline dataset produced by the explicit case optimizer is good enough."
    )
