from dataclasses import replace

import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import example
from compliant_mechanism_synthesis.prescribed_motion import (
    predict_equilibrium,
    trace_motion,
)
from compliant_mechanism_synthesis.trajectory import Trajectory


@pytest.mark.parametrize("dimension", [2, 3])
def test_predictor_preserves_equilibrium_and_design_derivatives(dimension):
    torch.set_num_threads(1)
    design = example(dimension)
    dof = 3 if dimension == 2 else 6
    poses = design.positions.new_zeros(2, dof)
    poses[1, 1] = 0.0006
    trajectory = Trajectory(poses.new_tensor([0, 1]), poses)
    results = []
    gradients = []
    for enabled in (0, 1):
        radii = design.radii.detach().clone().requires_grad_()
        batch = DesignBatch(
            design.positions[None],
            design.roles,
            design.edges,
            radii[None],
            design.presence[None],
        )
        path = trace_motion(
            batch, trajectory, ContinuationConfig(steps=3, predictor=enabled)
        )
        assert path.completed.all()
        results.append(path.states[-1].q.detach())
        gradients.append(
            torch.autograd.grad(path.states[-1].q[0, -2 * dof + 1], radii)[0]
        )
    torch.testing.assert_close(results[0], results[1], atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-4)


def test_prediction_refines_even_when_newton_converges():
    design = example(2)
    batch = DesignBatch(
        design.positions[None],
        design.roles,
        design.edges,
        design.radii[None],
        design.presence[None],
    )
    trajectory = Trajectory(
        design.positions.new_tensor([0, 1]),
        design.positions.new_tensor([[0, 0, 0], [0, 0.001, 0.02]]),
    )
    coarse = ContinuationConfig(
        steps=1, path_position_tolerance=1, path_orientation_tolerance=1
    )
    a = trace_motion(batch, trajectory, coarse)
    b = trace_motion(
        batch,
        trajectory,
        replace(coarse, path_position_tolerance=1e-6, path_orientation_tolerance=1e-4),
    )
    assert a.completed.all() and b.completed.all()
    assert len(b.states) > len(a.states)
    torch.testing.assert_close(a.states[-1].q, b.states[-1].q, atol=1e-8, rtol=1e-5)


def test_predictor_is_only_a_detached_initial_guess():
    q = torch.zeros(1, 9, dtype=torch.float64, requires_grad=True)
    sensitivity = torch.ones(1, 6, 3, dtype=torch.float64, requires_grad=True)
    body = torch.tensor([[0, 0.01, 0]], dtype=torch.float64, requires_grad=True)
    predicted = predict_equilibrium(q, sensitivity, body)
    assert not predicted.requires_grad
    torch.testing.assert_close(
        predicted[:, :6], torch.full((1, 6), 0.01, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"predictor": 2},
        {"path_position_tolerance": 0},
        {"path_orientation_tolerance": float("nan")},
    ],
)
def test_predictor_settings_are_validated(parameters):
    with pytest.raises(ValueError):
        ContinuationConfig(**parameters)
