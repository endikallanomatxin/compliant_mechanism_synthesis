from __future__ import annotations

import pytest
import torch

from compliant_mechanism_synthesis.dataset.types import Structures
from compliant_mechanism_synthesis.roles import NodeRole


def test_structures_accept_valid_batched_state() -> None:
    structures = Structures(
        positions=torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.5, 0.5, 0.5],
                ]
            ],
            dtype=torch.float32,
        ),
        roles=torch.tensor(
            [
                [
                    NodeRole.FIXED,
                    NodeRole.FIXED,
                    NodeRole.FIXED,
                    NodeRole.MOBILE,
                    NodeRole.MOBILE,
                    NodeRole.MOBILE,
                    NodeRole.FREE,
                ]
            ],
            dtype=torch.long,
        ),
        adjacency=torch.zeros((1, 7, 7), dtype=torch.float32),
    )

    structures.validate()


def test_structures_reject_non_batched_positions() -> None:
    structures = Structures(
        positions=torch.zeros((7, 3), dtype=torch.float32),
        roles=torch.zeros((1, 7), dtype=torch.long),
        adjacency=torch.zeros((1, 7, 7), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="rank 3"):
        structures.validate()
