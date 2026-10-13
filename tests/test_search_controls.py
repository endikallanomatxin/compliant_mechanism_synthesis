import pytest
import torch

from compliant_mechanism_synthesis import batching
from compliant_mechanism_synthesis.batching import DesignBatch, trace_batch
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig, trace
from compliant_mechanism_synthesis.optimization import (
    Population,
    Search,
    SearchConfig,
    example,
)
from compliant_mechanism_synthesis.trajectory import Trajectory


def test_pose_weights_and_scales_are_independent(tmp_path):
    from dataclasses import replace

    search = Search(
        SearchConfig(population=2, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=1),
    )
    try:
        actual = torch.tensor([[0.01, 0, 0.1]], dtype=torch.float64)
        target = torch.zeros_like(actual)
        search.config = replace(search.config, position_weight=1, orientation_weight=0)
        torch.testing.assert_close(
            search.pose_loss(actual, target), actual.new_tensor([0.5])
        )
        search.config = replace(search.config, position_weight=0, orientation_weight=1)
        torch.testing.assert_close(
            search.pose_loss(actual, target), actual.new_tensor([1])
        )
        search.config = replace(search.config, orientation_scale=0.2)
        torch.testing.assert_close(
            search.pose_loss(actual, target), actual.new_tensor([0.25])
        )
    finally:
        search.writer.close()


def test_selection_preserves_a_different_structure_over_a_clone(tmp_path):
    search = Search(
        SearchConfig(
            population=2,
            elites=1,
            diversity_weight=1,
            diversity_radius=0.1,
            log_dir=str(tmp_path),
        )
    )
    try:
        parent = search.population.take(slice(0, 1)).detached()
        novel = parent.detached()
        novel.positions[:, 4:] += 0.01
        population = Population.concatenate(parent, parent, novel)
        scores = torch.tensor([1.0, 1.001, 1.05], dtype=torch.float64)
        chosen, _ = search._select(population, scores, 2)
        assert torch.equal(chosen.positions[0], parent.positions[0])
        assert torch.equal(chosen.positions[1], novel.positions[0])
        similarity = search.similarity(population)
        torch.testing.assert_close(similarity, similarity.T)
        torch.testing.assert_close(
            similarity[0, 1], scores.new_tensor(1), atol=1e-12, rtol=1e-12
        )
        assert similarity[0, 2] < 0.1
    finally:
        search.writer.close()


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_signed_path_starts_preloaded_and_passes_through_rest(dimension, device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    design = example(dimension, device)
    radii = design.radii.detach().clone().requires_grad_(True)
    batch = DesignBatch(
        design.positions[None],
        design.roles,
        design.edges,
        radii[None],
        design.presence[None],
    )
    dof = 3 if dimension == 2 else 6
    load = design.positions.new_zeros(dof)
    load[1] = 2
    config = ContinuationConfig(steps=3)
    path = trace_batch(batch, load, config, rest_factor=0.4)
    assert path.completed.all()
    assert path.states[0].q[0, -dof + 1] < 0
    assert path.states[-1].q[0, -dof + 1] > 0
    rest = [
        state
        for state in path.states
        if state.accepted[0] and abs(state.factors[0].item() - 0.4) < 1e-12
    ]
    assert len(rest) == 1
    assert rest[0].q.abs().max() < 1e-7
    individual = trace(batch.item(0), load, config, rest_factor=0.4)
    assert individual.converged
    torch.testing.assert_close(
        path.states[0].q[0], individual.states[0].q, rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        path.states[-1].q[0], individual.states[-1].q, rtol=1e-6, atol=1e-8
    )
    value = path.states[0].q[0, -dof + 1] + path.states[-1].q[0, -dof + 1]
    derivative = torch.autograd.grad(value, radii)[0]
    reference = torch.autograd.grad(
        individual.states[0].q[-dof + 1] + individual.states[-1].q[-dof + 1], radii
    )[0]
    torch.testing.assert_close(derivative, reference, rtol=1e-5, atol=1e-7)


def test_rest_waypoint_can_be_in_the_middle():
    factors = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    poses = torch.tensor(
        [[0.0, -0.002, -0.1], [0.0, 0, 0], [0.0, 0.003, 0.2]], dtype=torch.float64
    )
    path = Trajectory(factors, poses, 0.5)
    path.validate()
    torch.testing.assert_close(path.load_at(factors), factors.new_tensor([-1, 0, 1]))
    assert path.at(factors)[0, 1] < 0
    with pytest.raises(ValueError, match="zero"):
        Trajectory(factors, poses + 0.1, 0.5).validate()
    with pytest.raises(ValueError, match="waypoint"):
        Trajectory(factors, poses, 0.4).validate()


def test_live_tuning_reuses_physics_and_resizes_population(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(population=4, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
    )
    try:
        search.ensure_evaluated()
        paths = dict(search.path_cache)
        before = search.scores.clone()

        def forbidden(*args, **kwargs):
            raise AssertionError("Changing objective repeated FEM")

        with monkeypatch.context() as patch:
            patch.setattr(batching, "equilibria", forbidden)
            search.tune(
                {
                    "position_weight": 4,
                    "orientation_weight": 0.25,
                    "learning_rate": 0.01,
                }
            )
        assert search.path_cache.keys() == paths.keys()
        assert not torch.equal(search.scores, before)
        search.tune({"population": 8, "evaluation_batch_size": 2})
        assert search.population.size == 8
        assert search.last_diagnostics["peak_batch_size"] == 2
        assert len(search.path_cache) <= 8
        search.tune({"population": 3})
        assert search.population.size == 3
        assert len(search.path_cache) <= 3
        with pytest.raises(ValueError):
            search.tune({"position_weight": 0, "orientation_weight": 0})
        assert search.config.position_weight == 4
    finally:
        search.writer.close()


def test_hybrid_schedule_uses_configured_cycle(tmp_path):
    search = Search(
        SearchConfig(
            population=2,
            elites=1,
            gradient_steps_per_evolution=2,
            evolution_steps_per_cycle=1,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=1),
    )
    try:
        methods = [search.step("hybrid")["batch"]["method"] for _ in range(4)]
        assert methods == ["gradient", "gradient", "evolution", "gradient"]
        assert search.hybrid_phase == 1
    finally:
        search.writer.close()


def test_signed_cached_derivatives_and_rest_invalidation(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(population=2, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=3),
    )
    try:
        search.set_trajectory(
            [0, 0.4, 1], [[0, -0.001, -0.05], [0, 0, 0], [0, 0.002, 0.1]], 0.4
        )
        search.ensure_evaluated()

        def evaluate():
            p = search.population.positions.detach().requires_grad_(True)
            r = search.population.radius_logits.detach().requires_grad_(True)
            a = search.population.presence_logits.detach().requires_grad_(True)
            scores = search._score(Population(p, r, a, search.population.topology))
            return scores.detach(), torch.autograd.grad(scores.sum(), (p, r, a))

        def forbidden(*args, **kwargs):
            raise AssertionError("Cached signed path repeated Newton")

        with monkeypatch.context() as patch:
            patch.setattr(batching, "equilibria", forbidden)
            cached, grad = evaluate()
        search.path_cache.clear()
        fresh, ref = evaluate()
        torch.testing.assert_close(cached, fresh, rtol=1e-7, atol=1e-8)
        for a, b in zip(grad, ref):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    finally:
        search.writer.close()


def test_inactive_beam_parameters_do_not_create_novelty_or_repeat_physics(tmp_path):
    from compliant_mechanism_synthesis.nonlinear import energy

    search = Search(SearchConfig(population=2, elites=1, log_dir=str(tmp_path)))
    try:
        parent = search.population.take(slice(0, 1)).detached()
        altered = parent.detached()
        altered.radius_logits[~altered.topology] += 4
        altered.presence_logits[~altered.topology] -= 4
        a = search.batch_design(parent).item(0)
        b = search.batch_design(altered).item(0)
        q = a.positions.new_zeros(39)
        q[-2] = 0.001
        torch.testing.assert_close(energy(a, q), energy(b, q))
        assert search._path_key(a) == search._path_key(b)
        torch.testing.assert_close(
            search.similarity(parent, altered), q.new_ones(1, 1), atol=1e-12, rtol=1e-12
        )
    finally:
        search.writer.close()
