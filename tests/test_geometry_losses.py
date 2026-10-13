from dataclasses import replace

import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.geometry_losses import (
    collision_loss,
    length_loss,
    path_curves,
)
from compliant_mechanism_synthesis.optimization import SearchConfig
from compliant_mechanism_synthesis.trajectory import Trajectory


def fixture(points, edges, device="cpu", radius=0.001):
    positions = torch.tensor(points, dtype=torch.float64, device=device)
    links = torch.tensor(edges, device=device)
    return DesignBatch(
        positions[None],
        ("free",) * len(points),
        links,
        positions.new_full((1, len(edges)), radius),
        positions.new_ones(1, len(edges)),
    )


def chords(batch, count=4):
    t = torch.linspace(
        0, 1, count + 1, dtype=batch.positions.dtype, device=batch.positions.device
    )
    ends = batch.positions[:, batch.edges]
    return (ends[..., :1, :] + t[:, None] * (ends[..., 1:2, :] - ends[..., :1, :]))[
        None
    ]


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_projected_crossing_has_no_spatial_collision(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    b = fixture(
        [[-0.02, 0, 0], [0.02, 0, 0], [0, -0.02, 0.003], [0, 0.02, 0.003]],
        [[0, 1], [2, 3]],
        device,
    )
    loss, depth, count = collision_loss(chords(b), b, SearchConfig())
    assert loss.item() == 0 and depth.item() == 0 and count.item() == 0


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_true_crossing_has_finite_separating_gradient(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    b = fixture(
        [[-0.02, 0, 0], [0.02, 0, 0], [0, -0.02, 0], [0, 0.02, 0]],
        [[0, 1], [2, 3]],
        device,
    )
    b.positions.requires_grad_()
    loss, depth, count = collision_loss(chords(b), b, SearchConfig())
    assert count.item() == 1 and depth.item() == pytest.approx(0.002)
    gradient = torch.autograd.grad(loss.sum(), b.positions)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    moved = replace(b, positions=b.positions - 0.0000001 * gradient)
    assert collision_loss(chords(moved), moved, SearchConfig())[0].item() < loss.item()


def test_shared_joint_is_allowed_but_overlapping_adjacent_beams_are_not():
    b = fixture([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0]], [[0, 1], [0, 2]])
    assert collision_loss(chords(b), b, SearchConfig())[0].item() == 0
    b.positions[0, 2] = b.positions.new_tensor([0.04, 0, 0])
    assert collision_loss(chords(b), b, SearchConfig())[0].item() > 0


def test_near_contact_gradient_matches_finite_differences():
    b = fixture(
        [[-0.02, 0, 0], [0.02, 0, 0], [0, -0.02, 0.001], [0, 0.02, 0.001]],
        [[0, 1], [2, 3]],
    )
    b.radii.requires_grad_()
    b.positions.requires_grad_()
    value = collision_loss(chords(b), b, SearchConfig())[0].sum()
    gp, gr = torch.autograd.grad(value, (b.positions, b.radii))
    step = 1e-7
    for name, index, expected in [
        ("positions", (0, 2, 2), gp[0, 2, 2]),
        ("radii", (0, 0), gr[0, 0]),
    ]:
        values = []
        for sign in (-1, 1):
            changed = getattr(b, name).detach().clone()
            changed[index] += sign * step
            candidate = replace(b, **{name: changed})
            values.append(
                collision_loss(chords(candidate), candidate, SearchConfig())[0].sum()
            )
        torch.testing.assert_close(
            (values[1] - values[0]) / (2 * step), expected, rtol=1e-5, atol=1e-5
        )


def test_absent_beam_has_zero_loss_and_zero_radius_gradient():
    b = fixture(
        [[-0.02, 0, 0], [0.02, 0, 0], [0, -0.02, 0], [0, 0.02, 0]], [[0, 1], [2, 3]]
    )
    b.presence[0, 1] = 0
    b.radii.requires_grad_()
    value = collision_loss(chords(b), b, SearchConfig())[0].sum()
    assert value.item() == 0
    torch.testing.assert_close(
        torch.autograd.grad(value, b.radii)[0], torch.zeros_like(b.radii)
    )


def test_collision_between_clear_endpoint_states_is_penalized():
    b = fixture(
        [
            [-0.02, 0, 0],
            [0.02, 0, 0],
            [0, -0.02, -0.004],
            [0, 0.02, -0.004],
            [0.1, 0, 0],
            [0.1, 0.02, 0],
            [0.15, 0, 0],
            [0.15, 0.02, 0],
        ],
        [[0, 1], [2, 3]],
    )
    b.roles = ("free",) * 4 + ("passive",) * 2 + ("active",) * 2
    a = torch.zeros(1, 36, dtype=torch.float64)
    z = a.clone()
    z[0, 14] = z[0, 20] = 0.008
    factors = a.new_tensor([[0], [1]])
    trajectory = Trajectory(a.new_tensor([0, 1]), a.new_zeros(2, 6))
    sparse = path_curves(
        b,
        [a, z],
        [None, None],
        factors,
        SearchConfig(collision_path_substeps=1),
        trajectory,
    )
    refined = path_curves(
        b,
        [a, z],
        [None, None],
        factors,
        SearchConfig(collision_path_substeps=2),
        trajectory,
    )
    assert collision_loss(sparse, b, SearchConfig())[0].item() == 0
    assert collision_loss(refined, b, SearchConfig())[0].item() > 0


def test_full_beam_length_bounds_do_not_penalize_internal_discretization():
    b = fixture(
        [[0, 0, 0], [0.006, 0, 0], [0, 0.02, 0], [0.08, 0.02, 0]], [[0, 1], [2, 3]]
    )
    b.positions.requires_grad_()
    values = []
    for subdivisions in (1, 8):
        loss, _, _ = length_loss(chords(b, subdivisions), b, SearchConfig())
        values.append(loss)
    torch.testing.assert_close(*values)
    assert values[0].item() > 0
    gradient = torch.autograd.grad(values[0].sum(), b.positions)[0]
    moved = replace(b, positions=b.positions - 0.0001 * gradient)
    assert (
        length_loss(chords(moved), moved, SearchConfig())[0].item() < values[0].item()
    )
