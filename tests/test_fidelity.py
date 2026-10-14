from dataclasses import replace

import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.linear_motion import trace_linear
from compliant_mechanism_synthesis.mechanics import (
    assemble,
    reduction,
)
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import (
    Population,
    Search,
    SearchConfig,
    example,
)
from compliant_mechanism_synthesis.prescribed_motion import trace_motion
from compliant_mechanism_synthesis.trajectory import Trajectory


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_linear_matches_analytical_stiffness_and_small_nonlinear_motion(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    d = example(3, device)
    radii = d.radii.clone().requires_grad_()
    b = DesignBatch(d.positions[None], d.roles, d.edges, radii[None], d.presence[None])
    poses = d.positions.new_zeros(3, 6)
    poses[:, 1] = poses.new_tensor([-0.000001, 0, 0.000001])
    trajectory = Trajectory(poses.new_tensor([0, 0.5, 1]), poses, 0.5)
    config = ContinuationConfig(steps=2)
    path = trace_linear(b, trajectory, config)
    assert path.completed.all() and path.evaluations == 1
    t, base = reduction(d)
    h = t.T @ assemble(d) @ t
    for state in path.states:
        active = trajectory.at(state.factors)
        expected = torch.linalg.solve(h[:base, :base], -h[:base, base:] @ active[0])
        torch.testing.assert_close(state.q[0, :base], expected, rtol=1e-7, atol=1e-10)
    value = path.states[-1].q[0, -11]
    grad = torch.autograd.grad(value, radii)[0]
    eps = 1e-7
    values = []
    for sign in [-1, 1]:
        r = d.radii.clone()
        r[3] += sign * eps
        bb = DesignBatch(b.positions, b.roles, b.edges, r[None], b.presence)
        values.append(trace_linear(bb, trajectory, config).states[-1].q[0, -11])
    torch.testing.assert_close(
        grad[3], (values[1] - values[0]) / (2 * eps), rtol=5e-4, atol=1e-8
    )
    fine = trace_motion(b.detached(), trajectory, config)
    assert fine.completed.all()
    torch.testing.assert_close(
        path.states[-1].q[0, -12:-6], fine.states[-1].q[0, -12:-6], rtol=0.01, atol=1e-7
    )


@pytest.mark.parametrize(
    "reason", ["iteration_budget", "motion_loss", "stagnation", "model_discrepancy"]
)
def test_stage_transition_replaces_all_scores_and_caches(tmp_path, reason):
    torch.set_num_threads(1)
    cfg = SearchConfig(
        population=2,
        elites=1,
        input_y=0.00005,
        linear_max_iterations=100,
        fidelity_check_interval=100,
        linear_motion_loss_target=1e-9,
        linear_stagnation_iterations=100,
        fidelity_pose_error_limit=1e6,
        log_dir=str(tmp_path),
    )
    s = Search(cfg, ContinuationConfig(steps=2, beam_segments=3))
    try:
        s.ensure_evaluated()
        assert s.analysis_level == "linear"
        assert not s.linearization_cache
        if reason == "iteration_budget":
            s.iteration = 100
        elif reason == "motion_loss":
            s.config = replace(s.config, linear_motion_loss_target=1e6)
        elif reason == "stagnation":
            s.linear_best_motion = 0
            s.linear_motion_stagnation = 100
        else:
            s.iteration = 100
            s.config = replace(
                s.config, linear_max_iterations=1000, fidelity_pose_error_limit=1e-12
            )
        s._advance_fidelity()
        assert s.analysis_level == "nonlinear" and s.continuation.beam_segments == 1
        assert s.fidelity_report["transition_reason"] == reason
        assert s.scores.isfinite().all()
        assert s.linearization_cache
        current = s._score(s.population)
        # The transition ranks the entire population under the new model.
        torch.testing.assert_close(current, s.scores, rtol=1e-8, atol=1e-9)
        assert s.snapshot()["trajectory_validated"]
    finally:
        s.writer.close()


def test_linear_search_is_differentiable_and_labels_approximation(tmp_path):
    torch.set_num_threads(1)
    s = Search(
        SearchConfig(
            population=2, elites=1, gradient_backtracks=1, log_dir=str(tmp_path)
        ),
        ContinuationConfig(steps=2),
    )
    try:
        state = s.snapshot()
        assert state["analysis_level"] == "linear" and not state["trajectory_validated"]
        p = s.population.positions.detach().requires_grad_()
        r = s.population.radius_logits.detach().requires_grad_()
        loss = s._score(Population(p, r, s.population.topology))
        grads = torch.autograd.grad(loss.sum(), (p, r))
        assert all(torch.isfinite(g).all() for g in grads)
        assert s.last_diagnostics["equilibrium_batches"] == 1
    finally:
        s.writer.close()
