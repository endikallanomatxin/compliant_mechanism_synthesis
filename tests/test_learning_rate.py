"""The population controller must measure first attempts, not rescued steps."""

import pytest
import torch

from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import Population, Search, SearchConfig


@pytest.mark.parametrize("improving", [0, 8, 9, 10])
def test_rate_targets_first_attempt_improvement(tmp_path, monkeypatch, improving):
    search = Search(
        SearchConfig(population=10, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=1),
    )
    try:
        search.scores = search.positions.new_ones(10)
        proposals = Population.concatenate(*([search.population] * 4))
        monkeypatch.setattr(search, "_gradient_proposals", lambda: proposals)
        calls = []

        def score(population):
            search.last_diagnostics = {
                "batch_size": population.size,
                "peak_batch_size": population.size,
                "converged": population.size,
                "rejected": 0,
                "equilibrium_batches": 1,
                "residual": 0,
            }
            calls.append(population.size)
            return search.positions.new_tensor(
                [0.8] * improving + [1.2] * (10 - improving)
                if len(calls) == 1
                else [0.9] * population.size
            )

        monkeypatch.setattr(search, "_score", score)
        _, scores = search._adaptive_gradient()
        assert (scores < search.scores).all()  # Backtracking rescues every parent.
        record = search.gradient_history[-1]
        assert record["monotonic_fraction"] == improving / 10
        assert record["initial_monotonic"] == improving
        assert sum(record["accepted"]) == 10
        before, after = record["requested_rate"], search.current_learning_rate
        if improving > 9:
            assert after > before
        elif improving < 9:
            assert after < before
        else:
            assert after == before
        assert record["next_rate"] == after
        assert sum(calls) == 10 + (10 - improving)  # No additional FEM evaluations.
    finally:
        search.writer.close()


@pytest.mark.parametrize("adaptive", [0, 1])
def test_flat_cost_does_not_grow_rate_and_manual_mode_is_fixed(
    tmp_path, monkeypatch, adaptive
):
    search = Search(
        SearchConfig(
            population=2,
            elites=1,
            adaptive_learning_rate=adaptive,
            learning_rate_min=0.019,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=1),
    )
    try:
        # Exclude invalid baselines from the proportion being controlled.
        search.scores = search.positions.new_tensor([1.0, torch.inf])
        proposals = Population.concatenate(*([search.population] * 4))
        monkeypatch.setattr(search, "_gradient_proposals", lambda: proposals)

        def score(population):
            search.last_diagnostics = {
                "batch_size": population.size,
                "peak_batch_size": population.size,
                "converged": population.size,
                "rejected": 0,
                "equilibrium_batches": 1,
                "residual": 0,
            }
            return search.positions.new_ones(population.size)

        monkeypatch.setattr(search, "_score", score)
        search._adaptive_gradient()
        record = search.gradient_history[-1]
        assert record["eligible"] == 1
        assert record["accepted"] == [1]
        assert record["monotonic_fraction"] == 0
        assert search.current_learning_rate == (0.019 if adaptive else 0.02)
    finally:
        search.writer.close()


def test_upper_rate_limit_and_next_proposal_use_adapted_rate(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(
            population=2, elites=1, learning_rate_max=0.0201, log_dir=str(tmp_path)
        ),
        ContinuationConfig(steps=1),
    )
    try:
        search.scores = search.positions.new_ones(2)
        proposals = Population.concatenate(*([search.population] * 4))
        monkeypatch.setattr(search, "_gradient_proposals", lambda: proposals)

        def score(population):
            search.last_diagnostics = {
                "batch_size": population.size,
                "peak_batch_size": population.size,
                "converged": population.size,
                "rejected": 0,
                "equilibrium_batches": 1,
                "residual": 0,
            }
            return search.positions.new_full((population.size,), 0.8)

        monkeypatch.setattr(search, "_score", score)
        search._adaptive_gradient()
        assert search.current_learning_rate == 0.0201
        search._adaptive_gradient()
        assert search.gradient_history[-1]["requested_rate"] == 0.0201
    finally:
        search.writer.close()
