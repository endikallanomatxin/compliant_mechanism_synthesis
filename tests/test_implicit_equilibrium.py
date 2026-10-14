import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.implicit_equilibrium import attach_equilibrium
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import example
from compliant_mechanism_synthesis.prescribed_motion import relax_equilibrium


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_adjoint_pose_derivative_matches_finite_difference(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    d = example(3, device)
    b = DesignBatch(
        d.positions[None], d.roles, d.edges, d.radii[None], d.presence[None]
    )
    q = d.positions.new_zeros(1, 156)
    body = q.new_tensor([[0, 0.0003, 0, 0, 0, 0.005]])
    config = ContinuationConfig()
    solved, ok, *_ = relax_equilibrium(b, body, q, config)
    assert ok.all()
    positions = b.positions.clone().requires_grad_()
    radii = b.radii.clone().requires_grad_()
    active = body.clone().requires_grad_()
    differentiable = DesignBatch(positions, b.roles, b.edges, radii, b.presence)
    result, _ = attach_equilibrium(differentiable, active, solved, None, config)
    weights = q.new_tensor([0.2, 1.0, 0.1, 0.03, -0.02, 0.04])
    value = (result[:, -12:-6] * weights).sum() + result[:, -5].sum() * 0.3
    grads = torch.autograd.grad(value, (positions, radii, active))
    for field, index, epsilon in [
        ("positions", (0, 8, 0), 1e-6),
        ("radii", (0, 3), 1e-7),
        ("body", (0, 1), 1e-7),
    ]:
        values = []
        for sign in [-1, 1]:
            pos, rad, act = b.positions.clone(), b.radii.clone(), body.clone()
            data = {"positions": pos, "radii": rad, "body": act}[field]
            data[index] += sign * epsilon
            trial = DesignBatch(pos, b.roles, b.edges, rad, b.presence)
            state, good, *_ = relax_equilibrium(trial, act, solved, config)
            assert good.all()
            values.append(
                float((state[:, -12:-6] * weights).sum() + state[:, -5].sum() * 0.3)
            )
        expected = (values[1] - values[0]) / (2 * epsilon)
        actual = grads[{"positions": 0, "radii": 1, "body": 2}[field]][index]
        torch.testing.assert_close(
            actual, actual.new_tensor(expected), rtol=3e-4, atol=2e-6
        )


def test_pose_backward_reuses_factorization_without_tangent(monkeypatch):
    from compliant_mechanism_synthesis import condensed_mechanics

    torch.set_num_threads(1)
    d = example(3)
    radii = d.radii[None].clone().requires_grad_()
    b = DesignBatch(d.positions[None], d.roles, d.edges, radii, d.presence[None])
    q = d.positions.new_zeros(1, 156)
    body = q.new_tensor([[0, 0.0003, 0, 0, 0, 0]])
    result, ok, *_ = relax_equilibrium(
        b, body, q, ContinuationConfig(), differentiate_stiffness=False
    )
    assert ok.all()

    def forbidden(*args, **kwargs):
        raise AssertionError("Tangent rebuilt during adjoint")

    monkeypatch.setattr(condensed_mechanics, "evaluate", forbidden)
    grad = torch.autograd.grad(result[:, -11].sum(), radii)[0]
    assert torch.isfinite(grad).all() and grad.abs().max() > 0


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_private_coordinate_adjoint_matches_finite_difference(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    from compliant_mechanism_synthesis.prescribed_motion import trace_motion
    from compliant_mechanism_synthesis.trajectory import Trajectory

    torch.set_num_threads(1)
    d = example(3, device)
    poses = d.positions.new_zeros(2, 6)
    poses[1, 1] = 0.00005
    trajectory = Trajectory(poses.new_tensor([0, 1]), poses)
    config = ContinuationConfig(steps=2, beam_segments=2)

    def value(radii):
        b = DesignBatch(
            d.positions[None], d.roles, d.edges, radii[None], d.presence[None]
        )
        path = trace_motion(b, trajectory, config, differentiate_stiffness=False)
        assert path.completed.all()
        return (
            path.states[-1].beam_nodes[0, :, 1, :3]
            * d.positions.new_tensor([0.2, 0.5, -0.3])
        ).sum()

    radii = d.radii.clone().requires_grad_()
    derivative = torch.autograd.grad(value(radii), radii)[0]
    eps = 1e-7
    lo, hi = d.radii.clone(), d.radii.clone()
    lo[3] -= eps
    hi[3] += eps
    expected = (value(hi) - value(lo)) / (2 * eps)
    torch.testing.assert_close(derivative[3], expected, rtol=5e-4, atol=2e-6)


def test_exact_cache_reuses_factors_and_enforces_memory_budget(tmp_path, monkeypatch):
    from dataclasses import replace

    from compliant_mechanism_synthesis import equilibrium_system
    from compliant_mechanism_synthesis.optimization import (
        Population,
        Search,
        SearchConfig,
    )

    torch.set_num_threads(1)
    s = Search(
        SearchConfig(population=2, elites=1, input_y=0.00005, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
    )
    try:
        s.ensure_evaluated()
        assert s.linearization_cache and 0 < s.linearization_cache_bytes < 256 * 1024**2

        def forbidden(*args, **kwargs):
            raise AssertionError("Exact cached gradient refactored equilibrium")

        p = s.population.positions.detach().requires_grad_()
        r = s.population.radius_logits.detach().requires_grad_()
        with monkeypatch.context() as patch:
            patch.setattr(equilibrium_system, "factor_system", forbidden)
            scores = s._score(Population(p, r, s.population.topology))
            gp, gr = torch.autograd.grad(scores.sum(), (p, r))
        assert torch.isfinite(gp).all() and torch.isfinite(gr).all()
        assert s.last_diagnostics["cached_factorizations_reused"] > 0
        for _, contexts in s.linearization_cache.values():
            for c in contexts:
                assert not c.system.factor.requires_grad
                assert c.work is None
        s.config = replace(s.config, linearization_cache_mb=0.001)
        s._trim_linearizations()
        assert s.linearization_cache_bytes <= 0.001 * 1024**2
        assert not s.linearization_cache
        s.scores = None
        s.ensure_evaluated()
        assert s.scores.isfinite().all()
    finally:
        s.writer.close()


def test_work_caches_remain_bounded_across_iterations(tmp_path, monkeypatch):
    from compliant_mechanism_synthesis.optimization import Search, SearchConfig

    torch.set_num_threads(1)
    s = Search(
        SearchConfig(
            population=2,
            elites=1,
            input_y=0.00005,
            gradient_backtracks=1,
            linearization_cache_mb=8,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )
    monkeypatch.setattr(s, "snapshot", lambda *args, **kwargs: {})
    try:
        for method in ["gradient", "evolution", "gradient"]:
            s.step(method)
            live = set(s._path_keys(s.batch_design(s.population)))
            assert set(s.linearization_cache) <= live
            assert set(s.parent_paths) <= live
            assert s.linearization_cache_bytes <= 8 * 1024**2
            for _, contexts in s.linearization_cache.values():
                for c in contexts:
                    assert c.work is None
                    assert not c.system.factor.requires_grad
                    assert not c.response.requires_grad
                    if c.private_system is not None:
                        assert not c.private_system.factor.requires_grad
    finally:
        s.writer.close()
