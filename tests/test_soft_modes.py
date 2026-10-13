"""Analytical condensation, mode sensitivities and natural nonlinear motion."""

import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig, energy
from compliant_mechanism_synthesis.optimization import example
from compliant_mechanism_synthesis.soft_modes import (
    MinimumMode,
    body_step,
    condensed_stiffness,
    pose_delta,
    relax_free,
    symmetric_eigh,
    trace_soft,
)
from compliant_mechanism_synthesis.trajectory import Trajectory


def batch_of(design, radii=None):
    return DesignBatch(
        design.positions[None],
        design.roles,
        design.edges,
        (design.radii if radii is None else radii)[None],
        design.presence[None],
    )


def test_condensation_relaxes_internal_nodes_and_scales_rotations():
    h = torch.tensor(
        [
            [
                [4.0, 2.0, 0.0, 0.0],
                [2.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ]
        ],
        dtype=torch.float64,
    )
    k = condensed_stiffness(h, h.new_zeros(1, 4), h.new_zeros(1, 3), 2, 0.5)
    torch.testing.assert_close(k, torch.diag(h.new_tensor([4, 8, 8]))[None])
    assert k[0, 0, 0] < h[0, 1, 1]


def test_minimum_mode_derivative_with_repeated_higher_eigenvalues():
    a = torch.diag(torch.tensor([1.0, 4.0, 4.0], dtype=torch.float64)).requires_grad_()
    values, v = MinimumMode.apply(a)
    g = torch.autograd.grad(v[1] + values[0] * 0.2, a)[0]
    assert torch.isfinite(g).all()
    assert g[0, 0] == pytest.approx(0.2)
    assert g[0, 1] == pytest.approx(-1 / 6)
    direction = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.4, 0.1], [0.3, 0.1, 0.2]], dtype=a.dtype
    )

    def objective(x):
        values, v = MinimumMode.apply(x)
        v = v * torch.sign(v[0])
        return v[1] + values[0] * 0.2

    eps = 1e-5
    finite = (
        objective(a.detach() + eps * direction)
        - objective(a.detach() - eps * direction)
    ) / (2 * eps)
    assert finite == pytest.approx(float((g * direction).sum()), rel=1e-6)


@pytest.mark.parametrize("dimension", [2, 3])
def test_spatial_chart_tangent_matches_direct_energy_hessian(dimension):
    design = example(dimension)
    dof = 3 if dimension == 2 else 6
    config = ContinuationConfig(steps=2)
    body = design.positions.new_zeros(1, dof)
    body[0, 1] = 0.0003
    body[0, -1] = 0.05
    initial = body.new_zeros(1, (sum(r == "free" for r in design.roles) + 1) * dof)
    q, ok, _, _, g, h = relax_free(batch_of(design), body, initial, config)
    assert ok.all()
    k = condensed_stiffness(h, g, body, dimension, 0.05)
    scale = body.new_ones(dof)
    scale[dimension:] = 1 / 0.05
    base = q.shape[-1] - dof

    def chart_energy(z):
        free = q[0, :base].detach() + z[:base]
        b = body_step(body[0], z[base:] * scale, dimension)
        return energy(design, torch.cat((free, b)))

    zero = q.new_zeros(q.shape[-1])
    full = torch.func.hessian(chart_energy)(zero)
    reference = full[base:, base:] - full[base:, :base] @ torch.linalg.solve(
        full[:base, :base], full[:base, base:]
    )
    torch.testing.assert_close(k[0], reference, rtol=1e-7, atol=1e-5)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("field", ["radii", "positions", "presence"])
def test_full_natural_path_derivative_matches_finite_difference(dimension, field):
    design = example(dimension)
    dof = 3 if dimension == 2 else 6
    end = design.positions.new_zeros(dof)
    end[1] = 0.001
    end[-1] = 0.02
    if dimension == 3:
        end[2] = 0.001
        end[3] = 0.02
    target = Trajectory(
        end.new_tensor([0, 0.5, 1]), torch.stack((end * 0, end * 0.5, end))
    )
    config = ContinuationConfig(steps=2)
    parameter = getattr(design, field).detach().clone().requires_grad_()

    def objective(value):
        batch = batch_of(design)
        setattr(batch, field, value[None])
        path = trace_soft(batch, target, config)
        assert path.completed.all()
        return (path.states[-1].q[0, -dof:] * end).sum()

    gradient = torch.autograd.grad(objective(parameter), parameter)[0]
    assert torch.isfinite(gradient).all()
    if field == "positions":
        mask = torch.tensor([r == "free" for r in design.roles])[:, None]
        gradient = gradient * mask
    i = int(gradient.abs().argmax())
    epsilon = 1e-5 if field == "presence" else 1e-7
    plus = parameter.detach().clone()
    minus = plus.clone()
    plus.flatten()[i] += epsilon
    minus.flatten()[i] -= epsilon
    finite = (objective(plus) - objective(minus)) / (2 * epsilon)
    torch.testing.assert_close(gradient.flatten()[i], finite, rtol=2e-3, atol=1e-7)


def test_target_selects_sign_without_imposing_direction_and_rest_is_in_middle():
    design = example(2)
    config = ContinuationConfig(steps=4)
    end = design.positions.new_tensor([0.001, 0.002, 0.02])
    target = Trajectory(
        end.new_tensor([0, 0.5, 1]), torch.stack((-end, end * 0, end)), 0.5
    )
    path = trace_soft(batch_of(design), target, config)
    assert path.completed.all()
    rest = next(s for s in path.states if s.factors[0] == 0.5)
    torch.testing.assert_close(rest.q, torch.zeros_like(rest.q))
    assert path.states[0].q[0, -2] < 0 and path.states[-1].q[0, -2] > 0
    assert not torch.allclose(path.states[-1].q[0, -3:], end)
    scale = end.new_tensor([1, 1, 0.05])
    i = path.states.index(rest)
    increment = pose_delta(rest.q[0, -3:], path.states[i + 1].q[0, -3:], 2) * scale
    direction = increment / increment.norm()
    assert abs(torch.dot(direction, rest.mode[0])) > 1 - 1e-10
    assert torch.dot(direction, end * scale) > 0


def test_degenerate_lowest_mode_is_rejected(monkeypatch):
    from compliant_mechanism_synthesis import soft_modes

    design = example(2)
    monkeypatch.setattr(
        soft_modes,
        "condensed_stiffness",
        lambda h, *args: torch.eye(3, dtype=h.dtype, device=h.device)[None],
    )
    end = design.positions.new_tensor([0, 0.001, 0])
    target = Trajectory(end.new_tensor([0, 1]), torch.stack((end * 0, end)))
    path = trace_soft(batch_of(design), target, ContinuationConfig(steps=2))
    assert not path.completed.any()
    assert not path.states


@pytest.mark.parametrize("dimension", [3, 6])
def test_xpu_spectrum_remains_on_device_and_matches_cpu(dimension):
    if not torch.xpu.is_available():
        pytest.skip("No Intel XPU")
    generator = torch.Generator().manual_seed(9)
    a = torch.randn(2, dimension, dimension, dtype=torch.float64, generator=generator)
    a = a @ a.transpose(-1, -2) + torch.eye(dimension)
    expected = torch.linalg.eigvalsh(a)
    values, vectors = symmetric_eigh(a.to("xpu"))
    assert values.device.type == "xpu"
    torch.testing.assert_close(values.cpu(), expected, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        (a.to("xpu") @ vectors).cpu(),
        (vectors * values[:, None, :]).cpu(),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_cpu_xpu_natural_path_and_gradient_agree(dimension):
    if not torch.xpu.is_available():
        pytest.skip("No Intel XPU")
    outputs = []
    gradients = []
    for device in ["cpu", "xpu"]:
        design = example(dimension, device)
        dof = 3 if dimension == 2 else 6
        end = design.positions.new_zeros(dof)
        end[1] = 0.001
        end[-1] = 0.02
        if dimension == 3:
            end[2] = 0.001
            end[3] = 0.02
        target = Trajectory(
            end.new_tensor([0, 0.5, 1]), torch.stack((end * 0, end * 0.5, end))
        )
        r = design.radii.detach().requires_grad_()
        path = trace_soft(batch_of(design, r), target, ContinuationConfig(steps=2))
        assert path.completed.all()
        output = path.states[-1].q[0, -dof:]
        gradient = torch.autograd.grad((output * end).sum(), r)[0]
        outputs.append(output.detach().cpu())
        gradients.append(gradient.detach().cpu())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-4, atol=1e-7)
