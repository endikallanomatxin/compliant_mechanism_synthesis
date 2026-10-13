import pytest
import torch

from compliant_mechanism_synthesis.mechanics import Design
from compliant_mechanism_synthesis.physical_diagnostics import beam_strains


@pytest.mark.parametrize("dimension", [2, 3])
def test_axial_and_bending_strain_match_beam_formulas(dimension):
    positions = torch.zeros(2, dimension, dtype=torch.float64)
    positions[1, 0] = 0.1
    design = Design(
        positions,
        ("fixed", "passive"),
        torch.tensor([[0, 1]]),
        torch.tensor([0.001], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    dof = 3 if dimension == 2 else 6
    q = positions.new_zeros(dof)
    normal, shear = beam_strains(design, q)
    torch.testing.assert_close(normal, torch.zeros_like(normal), atol=1e-14, rtol=0)
    q[0] = 0.001
    normal, shear = beam_strains(design, q)
    torch.testing.assert_close(normal, normal.new_tensor([0.01]))
    q.zero_()
    q[-1] = 0.02
    normal, shear = beam_strains(design, q)
    torch.testing.assert_close(
        normal, normal.new_tensor([4 * 0.02 * 0.001 / 0.1]), atol=1e-12, rtol=1e-8
    )
    torch.testing.assert_close(shear, torch.zeros_like(shear), atol=1e-12, rtol=0)
    design.presence.zero_()
    normal, shear = beam_strains(design, q)
    assert normal.item() == 0 and shear.item() == 0


def test_spatial_torsional_engineering_shear():
    p = torch.tensor([[0.0, 0, 0], [0.1, 0, 0]], dtype=torch.float64)
    design = Design(
        p,
        ("fixed", "passive"),
        torch.tensor([[0, 1]]),
        p.new_tensor([0.001]),
        p.new_ones(1),
    )
    q = p.new_tensor([0, 0, 0, 0.02, 0, 0])
    normal, shear = beam_strains(design, q)
    torch.testing.assert_close(normal, p.new_zeros(1), atol=1e-12, rtol=0)
    torch.testing.assert_close(shear, p.new_tensor([0.001 * 0.02 / 0.1]))
