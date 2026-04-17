from __future__ import annotations

from compliant_mechanism_synthesis.cli.dataset_generate import dataset_generate_main
from compliant_mechanism_synthesis.cli.sample_supervised import sample_supervised_main
from compliant_mechanism_synthesis.cli.train_rl import train_rl_main
from compliant_mechanism_synthesis.cli.train_supervised import train_supervised_main
from compliant_mechanism_synthesis.cli.upgrade_supervised_checkpoint import (
    upgrade_supervised_checkpoint_main,
)
from compliant_mechanism_synthesis.cli.dataset_visualize import visualize_dataset_main

__all__ = [
    "dataset_generate_main",
    "sample_supervised_main",
    "train_rl_main",
    "train_supervised_main",
    "upgrade_supervised_checkpoint_main",
    "visualize_dataset_main",
]
