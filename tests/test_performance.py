"""Mechanical equivalence and checks that eliminated work stays eliminated."""

import pytest
import torch

from compliant_mechanism_synthesis import batching, nonlinear
from compliant_mechanism_synthesis.batching import DesignBatch, equilibria
from compliant_mechanism_synthesis.mechanics import DEFAULT_MATERIAL
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig, energy
from compliant_mechanism_synthesis.optimization import (
    Population,
    Search,
    SearchConfig,
    example,
)
from compliant_mechanism_synthesis.tangents import assembled_tangent, descent_step


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_local_tangents_include_rigid_body_curvature(dimension, device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    design = example(dimension, device)
    dof = 3 if dimension == 2 else 6
    q = design.positions.new_zeros(26 * dof)
    q[:dof] = 0.0002
    q[-dof:] = torch.linspace(0.0001, 0.2, dof, dtype=q.dtype, device=q.device)
    tangent = assembled_tangent(design, q, DEFAULT_MATERIAL, chunk_size=8)
    reference = torch.func.jacrev(
        torch.func.grad(lambda x: energy(design, x)), chunk_size=8
    )(q)
    torch.testing.assert_close(tangent, reference, rtol=1e-8, atol=1e-6)
    torch.testing.assert_close(tangent, tangent.T, rtol=1e-10, atol=1e-7)


def test_newton_excludes_finished_and_disabled_members(monkeypatch):
    design = example(2)
    batch = DesignBatch(
        design.positions.expand(3, -1, -1),
        design.roles,
        design.edges,
        design.radii.expand(3, -1),
        design.presence.expand(3, -1),
    )
    sizes = []
    original = batching.kernels

    def instrument(*args, **kwargs):
        value, gradient, hessian = original(*args, **kwargs)

        def tangent(*inputs):
            sizes.append(len(inputs[-1]))
            return hessian(*inputs)

        return value, gradient, tangent

    monkeypatch.setattr(batching, "kernels", instrument)
    load = design.positions.new_tensor([[0, 0, 0], [0, 1, 0], [0, 50, 0]])
    result, completed, *_ = equilibria(
        batch,
        load,
        load.new_zeros(3, 78),
        torch.tensor([True, True, False]),
        ContinuationConfig(),
        DEFAULT_MATERIAL,
    )
    assert completed.tolist() == [True, True, False]
    assert sizes[0] == 2
    assert all(size == 1 for size in sizes[1:])
    assert len(sizes) > 1
    assert torch.equal(result[0], torch.zeros_like(result[0]))


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_stabilization_is_device_local_and_descending(monkeypatch, device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Eigensolve inside Newton")

    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    h = torch.tensor(
        [[[-1.0, 0], [0, 2]], [[3.0, 1], [1, 2]], [[-1e-9, 0], [0, 2]]],
        dtype=torch.float64,
        device=device,
    )
    g = torch.tensor([[1.0, 0], [2.0, 1], [1.0, 0]], dtype=torch.float64, device=device)
    step = descent_step(h, g, 1e-9)
    assert ((step * g).sum(-1) < 0).all()
    torch.testing.assert_close(
        step[1],
        torch.linalg.solve(
            h[1] + 1e-9 * torch.eye(2, dtype=h.dtype, device=device), -g[1]
        ),
    )


def test_viewer_reuses_all_retained_paths_and_cache_is_bounded(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(population=4, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
    )
    try:
        search.ensure_evaluated()

        def forbidden(*args, **kwargs):
            raise AssertionError("Viewer repeated equilibrium")

        with monkeypatch.context() as patch:
            patch.setattr(nonlinear, "equilibrium", forbidden)
            patch.setattr(batching, "equilibria", forbidden)
            for i in range(4):
                view = search.snapshot(i)
                assert view["converged"]
                assert view["states"][0]["factor"] == 0
                assert view["states"][-1]["factor"] == 1
                assert all(
                    a["factor"] < b["factor"]
                    for a, b in zip(view["states"], view["states"][1:])
                )
        previous_scores = search.scores.clone()
        search.target[1] += 0.001
        search.snapshot()
        # The passive target changes scoring without imposing passive motion.
        assert search.last_diagnostics["equilibrium_batches"] == 0
        assert not torch.equal(search.scores, previous_scores)
        for _ in range(2):
            search.step("evolution")
            assert len(search.path_cache) <= search.population.size
            assert torch.isfinite(torch.tensor(search.last_diagnostics["residual"]))
        for _, records in search.path_cache.values():
            assert all(
                not q.requires_grad and q.device.type == "cpu" for _, q, *_ in records
            )
    finally:
        search.writer.close()


def test_gradient_backtracking_stops_per_parent(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(population=4, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=1),
    )
    try:
        search.scores = search.positions.new_tensor([1.0, 1.0, 1.0, 1.0])
        proposals = Population.concatenate(*([search.population] * 4))
        monkeypatch.setattr(search, "_gradient_proposals", lambda: proposals)
        responses = [[0.5, 2.0, 0.6, 3.0], [0.7, 2.0], [0.8]]
        seen = []

        def score(population):
            seen.append(population.size)
            search.last_diagnostics = {
                "batch_size": population.size,
                "peak_batch_size": population.size,
                "converged": population.size,
                "rejected": 0,
                "equilibrium_batches": 1,
                "residual": 0,
            }
            return search.positions.new_tensor(responses[len(seen) - 1])

        monkeypatch.setattr(search, "_score", score)
        _, scores = search._adaptive_gradient()
        assert seen == [4, 2, 1]
        torch.testing.assert_close(
            scores, search.positions.new_tensor([0.5, 0.7, 0.6, 0.8])
        )
        assert search.last_diagnostics["gradient_trials"] == seen
    finally:
        search.writer.close()


@pytest.mark.parametrize("dimension", [3])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_cached_implicit_derivatives_match_fresh_equilibria(
    dimension, device, tmp_path, monkeypatch
):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    search = Search(
        SearchConfig(
            device=device,
            population=2,
            elites=1,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )
    try:
        search.ensure_evaluated()

        def evaluate():
            p = search.population.positions.detach().requires_grad_(True)
            r = search.population.radius_logits.detach().requires_grad_(True)
            scores = search._score(Population(p, r, search.population.topology))
            return scores.detach(), torch.autograd.grad(scores.sum(), (p, r))

        def forbidden(*args, **kwargs):
            raise AssertionError("Cached gradient repeated Newton")

        with monkeypatch.context() as patch:
            patch.setattr(batching, "equilibria", forbidden)
            cached, cached_gradients = evaluate()
            assert search.last_diagnostics["equilibrium_batches"] == 0
        search.path_cache.clear()
        fresh, fresh_gradients = evaluate()
        torch.testing.assert_close(cached, fresh, rtol=1e-7, atol=1e-9)
        for cached_gradient, fresh_gradient in zip(cached_gradients, fresh_gradients):
            torch.testing.assert_close(
                cached_gradient, fresh_gradient, rtol=1e-5, atol=1e-7
            )
    finally:
        search.writer.close()


@pytest.mark.parametrize("dimension", [3])
def test_synthesis_prunes_only_masked_edges_without_changing_gradients(
    tmp_path, monkeypatch, dimension
):
    from compliant_mechanism_synthesis import prescribed_motion

    search = Search(
        SearchConfig(population=2, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=1),
    )
    try:

        def evaluate():
            p = search.population.positions.detach().requires_grad_()
            r = search.population.radius_logits.detach().requires_grad_()
            score = search._score(Population(p, r, search.population.topology))
            return score.detach(), torch.autograd.grad(score.sum(), (p, r))

        pruned, gradients = evaluate()
        with monkeypatch.context() as patch:
            patch.setattr(prescribed_motion, "active_beams", lambda batch: batch)
            full, reference = evaluate()
        torch.testing.assert_close(pruned, full, rtol=1e-7, atol=1e-9)
        for actual, expected in zip(gradients, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    finally:
        search.writer.close()


def test_prescribed_newton_skips_tangent_for_converged_residual(monkeypatch):
    from compliant_mechanism_synthesis import prescribed_motion

    design = example(2)
    batch = DesignBatch(
        design.positions[None],
        design.roles,
        design.edges,
        design.radii[None],
        design.presence[None],
    )
    original = prescribed_motion.kernels
    calls = []

    def tracked(*args, **kwargs):
        index = len(calls)
        calls.append(0)
        value, gradient, tangent = original(*args, **kwargs)

        def count(*inputs):
            calls[index] += 1
            return tangent(*inputs)

        return value, gradient, count

    monkeypatch.setattr(prescribed_motion, "kernels", tracked)
    q = design.positions.new_zeros(1, 78)
    _, ok, *_ = prescribed_motion.relax_equilibrium(
        batch, q[:, -3:], q, ContinuationConfig()
    )
    assert ok.all()
    assert calls[0] == 0
    assert calls[1] == 2  # Final stability/implicit tangent remains available.
