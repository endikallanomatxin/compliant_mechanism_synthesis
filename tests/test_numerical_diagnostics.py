import json

import pytest
import torch

from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import Population, Search, SearchConfig
from compliant_mechanism_synthesis.resolution import compare_resolution, interpolate


def test_interpolation_detects_unresolved_curvature():
    factors = torch.tensor([0.0, 1.0], dtype=torch.float64)
    fine = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    samples = torch.linspace(0, 1, 101, dtype=torch.float64)
    coarse = interpolate(factors, factors[:, None].square(), samples)
    refined = interpolate(fine, fine[:, None].square(), samples)
    assert (coarse - refined).abs().max() == pytest.approx(0.25)
    with pytest.raises(ValueError, match="complete path"):
        interpolate(factors, factors[:, None], torch.tensor([-0.1]))


def test_gradient_reports_actual_backtracking(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(
            population=2, elites=1, gradient_backtracks=2, log_dir=str(tmp_path)
        )
    )
    try:
        search.scores = torch.tensor([1.0, 2.0], dtype=torch.float64)
        monkeypatch.setattr(
            search,
            "_gradient_proposals",
            lambda: Population.concatenate(search.population, search.population),
        )
        values = iter([[1.5, 1.0], [0.5]])

        def score(population):
            search.last_diagnostics = {
                "batch_size": population.size,
                "peak_batch_size": population.size,
                "converged": population.size,
                "rejected": 0,
                "equilibrium_batches": 1,
                "seconds": 0,
                "residual": 0,
            }
            return search.scores.new_tensor(next(values))

        monkeypatch.setattr(search, "_score", score)
        _, scores = search._adaptive_gradient()
        torch.testing.assert_close(scores, scores.new_tensor([0.5, 1.0]))
        record = search.gradient_history[-1]
        assert record["trials"] == [2, 1]
        assert record["accepted"] == [1, 1]
        assert record["improved"] == [1, 1]
        assert record["unchanged"] == 0
        assert record["mean_accepted_rate"] == pytest.approx(
            search.config.learning_rate * 0.75
        )
        assert record["improvement_percent"] == pytest.approx(50)
        json.dumps(record, allow_nan=False)
    finally:
        search.writer.close()


def test_resolution_check_is_finite_and_does_not_change_search(tmp_path):
    search = Search(
        SearchConfig(
            population=2,
            elites=1,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )
    try:
        before = search.snapshot()
        population = search.population.detached()
        scores = search.scores.clone()
        cache_keys = set(search.path_cache)
        result = compare_resolution(search)
        assert result["converged"]
        assert result["fine_steps"] == 4
        assert result["samples"][0] == 0 and result["samples"][-1] == 1
        assert result["position_max_mm"] >= 0
        assert result["orientation_max_deg"] >= 0
        json.dumps(result, allow_nan=False)
        assert search.snapshot() is before
        assert set(search.path_cache) == cache_keys
        assert search.continuation.steps == 2
        assert search.iteration == 0
        torch.testing.assert_close(search.scores, scores)
        torch.testing.assert_close(search.population.positions, population.positions)
        with pytest.raises(ValueError, match="index"):
            compare_resolution(search, -1)
    finally:
        search.writer.close()
