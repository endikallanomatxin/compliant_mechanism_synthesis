import pytest
import torch

from compliant_mechanism_synthesis.optimization import Search, SearchConfig


def test_distance_prior_favors_short_additions_and_long_removals(tmp_path):
    search = Search(SearchConfig(population=2, elites=1, log_dir=str(tmp_path)))
    try:
        add, remove = search.topology_weights(search.population.positions)
        edges = search.initial.edges
        distance = (
            search.population.positions[:, edges[:, 0]]
            - search.population.positions[:, edges[:, 1]]
        ).norm(dim=-1)
        order = distance[0].argsort()
        assert (add[0, order].diff() <= 1e-12).all()
        assert (remove[0, order].diff() >= -1e-12).all()
        assert add[0, order[0]] > add[0, order[-1]]
        assert remove[0, order[-1]] > remove[0, order[0]]
        short = order[0]
        long = order[-1]
        assert add[0, short] > remove[0, short]
        assert remove[0, long] > add[0, long]
    finally:
        search.writer.close()


def test_children_are_refined_before_survival_and_parents_stay_unchanged(
    tmp_path, monkeypatch
):
    search = Search(
        SearchConfig(
            population=2, elites=1, offspring_gradient_steps=2, log_dir=str(tmp_path)
        )
    )
    try:
        search.scores = torch.tensor([1.0, 2.0], dtype=torch.float64)
        parent = search.population.detached()
        child = parent.take(torch.tensor([1]))
        child.positions[:, 4, 0] = 0.05
        monkeypatch.setattr(search, "ensure_evaluated", lambda: None)
        monkeypatch.setattr(search, "_offspring", lambda: child)
        search.parent_diagnostics = {}
        diagnostics = {
            "batch_size": 1,
            "peak_batch_size": 1,
            "converged": 1,
            "rejected": 0,
            "equilibrium_batches": 1,
            "seconds": 0.0,
            "residual": 0.0,
        }

        def score(population):
            search.last_diagnostics = dict(diagnostics)
            return population.positions.new_tensor([10.0])

        monkeypatch.setattr(search, "_score", score)
        calls = []

        def refine(population, scores, phase):
            assert phase == "offspring"
            torch.testing.assert_close(search.population.positions, parent.positions)
            calls.append(float(scores[0]))
            refined = population.detached()
            refined.positions[:, 4, 0] += 0.001
            search.last_diagnostics = dict(diagnostics)
            return refined, scores * 0.5

        monkeypatch.setattr(search, "_adaptive_gradient", refine)
        original_select = search._select

        def select(population, scores, *args, **kwargs):
            assert calls == [10.0, 5.0]
            assert float(scores.max()) < 10
            return original_select(population, scores, *args, **kwargs)

        monkeypatch.setattr(search, "_select", select)
        monkeypatch.setattr(
            search, "snapshot", lambda: {"batch": search.last_diagnostics}
        )
        result = search.step("evolution")
        assert calls == [10.0, 5.0]
        assert result["batch"]["offspring_gradient_steps"] == 2
        assert result["batch"]["offspring_improved"] == 1
        assert result["batch"]["offspring_refined"] == 1
        assert result["batch"]["batch_size"] == 3
    finally:
        search.writer.close()


@pytest.mark.parametrize(
    "values",
    [
        {"offspring_gradient_steps": 0},
        {"topology_distance_scale": 0},
        {"topology_distance_power": -1},
    ],
)
def test_new_controls_are_validated(values):
    with pytest.raises(ValueError):
        SearchConfig(**values)


def test_bounded_topology_edits_do_not_drift_toward_dense_graphs(tmp_path):
    from dataclasses import replace

    search = Search(
        SearchConfig(
            population=2,
            elites=1,
            topology_mutation_probability=1,
            log_dir=str(tmp_path),
        )
    )
    try:
        topology = search.population.topology[:1].expand(2048, -1)
        positions = search.population.positions[:1].expand(2048, -1, -1)
        changed = search._mutate_topology(topology, positions, 1)
        edits = changed ^ topology
        assert edits.sum(-1).eq(2).all()
        additions = int((edits & ~topology).sum())
        removals = int((edits & topology).sum())
        assert abs(additions - removals) < 0.08 * int(edits.sum())
        for bias, adding in [(1.0, True), (0.0, False)]:
            search.config = replace(search.config, topology_add_probability=bias)
            child = search._mutate_topology(topology[:16], positions[:16], 1)
            assert (
                (child ^ topology[:16]) & (topology[:16] if adding else ~topology[:16])
            ).sum() == 0
    finally:
        search.writer.close()


def test_topology_batches_limit_padding_and_restore_score_gradient_order(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    search = Search(
        SearchConfig(
            population=4,
            elites=1,
            topology_batch_union_ratio=1,
            evaluation_batch_size=4,
            log_dir=str(tmp_path),
        )
    )
    try:
        p = search.population.detached()
        p.topology[:] = False
        p.topology[[0, 2], :3] = True
        p.topology[[1, 3], 10:13] = True
        p.positions = p.positions.requires_grad_()
        groups = search._evaluation_groups(p, 4)
        assert [g.tolist() for g in groups] == [[0, 2], [1, 3]]

        def score(part):
            search.last_diagnostics = {
                "converged": part.size,
                "rejected": 0,
                "equilibrium_batches": 0,
                "residual": 0.0,
            }
            return part.positions.square().sum((1, 2))

        monkeypatch.setattr(search, "_score_chunk", score)
        actual = search._score(p)
        torch.testing.assert_close(actual, p.positions.square().sum((1, 2)))
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), p.positions)[0], 2 * p.positions
        )
        assert search.last_diagnostics["topology_padding_ratio"] == 1
        assert search.last_diagnostics["evaluation_groups"] == 2
        # An uncached candidate must not invalidate another candidate's replay.
        search.config = replace(search.config, topology_batch_union_ratio=8)
        keys = search._path_keys(search.batch_design(p))
        search.path_cache[keys[0]] = (True, [(0.0,), (1.0,)])
        search.path_cache[keys[1]] = (True, [(0.0,), (0.5,), (1.0,)])
        groups = search._evaluation_groups(p, 4)
        assert [g.tolist() for g in groups] == [[0], [1], [2, 3]]
    finally:
        search.writer.close()
