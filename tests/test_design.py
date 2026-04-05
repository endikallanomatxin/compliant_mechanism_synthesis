from __future__ import annotations

import pytest
import torch

from compliant_mechanism_synthesis.design import GraphDesign
from compliant_mechanism_synthesis.roles import NodeRole


def test_graph_design_accepts_valid_3d_state() -> None:
    design = GraphDesign(
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        roles=torch.tensor(
            [
                NodeRole.FIXED,
                NodeRole.FIXED,
                NodeRole.FIXED,
                NodeRole.MOBILE,
                NodeRole.MOBILE,
                NodeRole.MOBILE,
                NodeRole.FREE,
            ],
            dtype=torch.long,
        ),
        adjacency=torch.zeros((7, 7), dtype=torch.float32),
    )

    design.validate()


def test_graph_design_rejects_2d_positions() -> None:
    design = GraphDesign(
        positions=torch.zeros((7, 2), dtype=torch.float32),
        roles=torch.tensor(
            [
                NodeRole.FIXED,
                NodeRole.FIXED,
                NodeRole.FIXED,
                NodeRole.MOBILE,
                NodeRole.MOBILE,
                NodeRole.MOBILE,
                NodeRole.FREE,
            ],
            dtype=torch.long,
        ),
        adjacency=torch.zeros((7, 7), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="3D coordinates"):
        design.validate()
