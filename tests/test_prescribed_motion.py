"""Two rigid bodies coupled through finite-deformation equilibrium."""

import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.mechanics import assemble, reduction
from compliant_mechanism_synthesis.nonlinear import (
    ContinuationConfig,
    configuration,
    energy,
)
from compliant_mechanism_synthesis.optimization import Search, SearchConfig, example
from compliant_mechanism_synthesis.prescribed_motion import trace_motion
from compliant_mechanism_synthesis.spectral import positive_definite
from compliant_mechanism_synthesis.trajectory import Trajectory


def make_batch(design, field=None, value=None):
    batch = DesignBatch(
        design.positions[None],
        design.roles,
        design.edges,
        design.radii[None],
        design.presence[None],
    )
    if field:
        setattr(batch, field, value[None])
    return batch


def input_path(design, rest=0.0):
    dof = 3 if design.dimension == 2 else 6
    end = design.positions.new_zeros(dof)
    end[1] = 0.003
    end[-1] = 0.04
    if rest == 0:
        return Trajectory(
            end.new_tensor([0, 0.5, 1]), torch.stack((end * 0, end * 0.5, end))
        )
    return Trajectory(
        end.new_tensor([0, rest, 1]), torch.stack((-end, end * 0, end)), rest
    )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_nonlinear_path_exactly_prescribes_active_and_equilibrates_passive(
    dimension, device
):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    design = example(dimension, device)
    trajectory = input_path(design)
    dof = trajectory.poses.shape[-1]
    path = trace_motion(make_batch(design), trajectory, ContinuationConfig(steps=3))
    assert path.completed.all()
    for state in path.states:
        torch.testing.assert_close(
            state.q[0, -dof:], trajectory.at(state.factors[0]), atol=1e-13, rtol=1e-13
        )
        q = state.q[0].detach().requires_grad_()
        gradient = torch.autograd.grad(energy(design, q), q)[0]
        assert float(gradient[:-dof].norm()) <= 1e-7 * max(
            1, float(gradient[-dof:].norm())
        )
        points, _ = configuration(design, q)
        fixed = [i for i, r in enumerate(design.roles) if r == "fixed"]
        torch.testing.assert_close(
            points[fixed], design.positions[fixed], atol=0, rtol=0
        )
        for role in ["active", "passive"]:
            ids = [i for i, r in enumerate(design.roles) if r == role]
            torch.testing.assert_close(
                (points[ids[0]] - points[ids[1]]).norm(),
                (design.positions[ids[0]] - design.positions[ids[1]]).norm(),
                atol=1e-12,
                rtol=1e-12,
            )


@pytest.mark.parametrize("dimension", [2, 3])
def test_nonlinear_output_is_not_a_fixed_linear_transfer(dimension):
    design = example(dimension)
    trajectory = input_path(design)
    path = trace_motion(make_batch(design), trajectory, ContinuationConfig(steps=4))
    assert path.completed.all()
    transform, base = reduction(design)
    stiffness = transform.T @ assemble(design) @ transform
    imposed = trajectory.poses[-1]
    linear = -torch.linalg.solve(
        stiffness[:base, :base], stiffness[:base, base:] @ imposed
    )
    dof = len(imposed)
    actual = path.states[-1].q[0, -2 * dof : -dof]
    assert (actual - linear[-dof:]).norm() > 1e-5


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("field", ["radii", "positions"])
def test_output_sensitivity_matches_finite_differences(dimension, field):
    design = example(dimension)
    trajectory = input_path(design)
    parameter = getattr(design, field).detach().clone().requires_grad_()
    dof = trajectory.poses.shape[-1]
    weights = parameter.new_tensor(
        [0.3, 1, 0.05] if dimension == 2 else [0.3, 1, 0.2, 0.03, 0.04, 0.05]
    )

    def objective(value):
        path = trace_motion(
            make_batch(design, field, value), trajectory, ContinuationConfig(steps=2)
        )
        assert path.completed.all()
        return (path.states[-1].q[0, -2 * dof : -dof] * weights).sum()

    gradient = torch.autograd.grad(objective(parameter), parameter)[0]
    if field == "positions":
        gradient = (
            gradient
            * parameter.new_tensor([r == "free" for r in design.roles])[:, None]
        )
    i = int(gradient.abs().argmax())
    epsilon = 1e-7
    plus = parameter.detach().clone()
    minus = plus.clone()
    plus.flatten()[i] += epsilon
    minus.flatten()[i] -= epsilon
    finite = (objective(plus) - objective(minus)) / (2 * epsilon)
    torch.testing.assert_close(gradient.flatten()[i], finite, rtol=2e-3, atol=1e-6)


def test_passive_target_does_not_impose_passive_motion_and_input_invalidates(tmp_path):
    search = Search(
        SearchConfig(staged_analysis=0, population=2, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
    )
    try:
        before = search.snapshot()
        actual = [s["output"] for s in before["states"]]
        target = search.trajectory.poses.clone()
        target[:, 1] *= 2
        search.set_trajectory(search.trajectory.factors.tolist(), target.tolist())
        after = search.snapshot()
        assert [s["output"] for s in after["states"]] == actual
        assert after["objective"] != before["objective"]
        assert after["batch"]["equilibrium_batches"] == 0
        imposed = search.input_trajectory.poses.clone()
        imposed[:, 1] *= 2
        search.set_input(search.input_trajectory.factors.tolist(), imposed.tolist())
        changed = search.snapshot()
        assert changed["batch"]["equilibrium_batches"] > 0
        assert [s["output"] for s in changed["states"]] != actual
    finally:
        search.writer.close()


def test_bidirectional_input_crosses_rest_and_records_nonlinear_path():
    design = example(2)
    trajectory = input_path(design, 0.5)
    path = trace_motion(make_batch(design), trajectory, ContinuationConfig(steps=4))
    assert path.completed.all()
    rest = next(s for s in path.states if s.factors[0] == 0.5)
    torch.testing.assert_close(rest.q, torch.zeros_like(rest.q), atol=1e-9, rtol=0)
    assert path.states[0].q[0, -2] < 0 and path.states[-1].q[0, -2] > 0


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_device_local_stability_check_rejects_two_negative_eigenvalues(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    a = torch.diag(torch.tensor([-1.0, -2.0, 3.0], dtype=torch.float64, device=device))[
        None
    ]
    assert not positive_definite(a).any()
    assert positive_definite(a.abs()).all()
