from dataclasses import replace

import pytest
import torch

from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import Search, SearchConfig, example


def test_only_evolution_changes_binary_beam_presence(tmp_path):
    search = Search(
        SearchConfig(
            population=2,
            elites=1,
            gradient_backtracks=1,
            geometry_mutation_probability=0,
            radius_mutation_probability=0,
            topology_mutation_probability=0,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=1),
    )
    try:
        parent = search.population.detached()
        design = search.batch_design(parent)
        torch.testing.assert_close(design.presence, parent.topology.to(torch.float64))
        assert not design.presence.requires_grad
        assert design.presence[0].sum() == len(example(3).edges)
        proposals = search._gradient_proposals(parent)
        assert torch.equal(proposals.topology, parent.topology)
        assert not torch.equal(proposals.radius_logits, parent.radius_logits)
        assert set(parent.__dataclass_fields__) == {
            "positions",
            "radius_logits",
            "topology",
        }
        search.config = replace(
            search.config, topology_mutation_probability=1, topology_distance_power=0
        )
        mutated = search._mutate(parent)
        assert (
            (mutated.topology ^ parent.topology).sum(-1)
            == search.config.topology_mutation_edits
        ).all()
        torch.testing.assert_close(mutated.positions, parent.positions)
        torch.testing.assert_close(mutated.radius_logits, parent.radius_logits)
        assert set(search.batch_design(mutated).presence.unique().tolist()) <= {0, 1}
    finally:
        search.writer.close()


def test_mechanics_rejects_fractional_or_differentiable_presence():
    design = example(2)
    with pytest.raises(ValueError, match="binary"):
        replace(design, presence=design.presence * 0.3).validate()
    with pytest.raises(ValueError, match="non-differentiable"):
        replace(design, presence=design.presence.clone().requires_grad_()).validate()


def test_binary_topology_is_pruned_once_for_the_whole_path(tmp_path, monkeypatch):
    from compliant_mechanism_synthesis import prescribed_motion
    from compliant_mechanism_synthesis.trajectory import Trajectory

    search = Search(
        SearchConfig(population=2, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=3),
    )
    try:
        batch = search.batch_design(search.population.take(slice(0, 1)))
        calls = []
        original = prescribed_motion.active_beams

        def prune(design):
            result = original(design)
            calls.append((len(design.edges), len(result.edges)))
            return result

        monkeypatch.setattr(prescribed_motion, "active_beams", prune)
        trajectory = Trajectory(
            batch.positions.new_tensor([0, 1]),
            batch.positions.new_tensor([[0, 0, 0, 0, 0, 0], [0, 0.0003, 0, 0, 0, 0]]),
        )
        path = prescribed_motion.trace_motion(batch, trajectory, search.continuation)
        assert path.completed.all()
        assert len(path.states) >= 4
        assert calls == [(519, len(example(3).edges))]
    finally:
        search.writer.close()


def test_cached_equilibrium_skips_newton_batch_preparation(monkeypatch):
    from compliant_mechanism_synthesis.batching import DesignBatch
    from compliant_mechanism_synthesis.prescribed_motion import relax_equilibrium

    design = example(2)
    batch = DesignBatch(
        design.positions[None],
        design.roles,
        design.edges,
        design.radii[None],
        design.presence[None],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached equilibrium prepared a detached Newton batch")

    monkeypatch.setattr(DesignBatch, "detached", forbidden)
    initial = design.positions.new_zeros(1, 78)
    q, ok, *_ = relax_equilibrium(
        batch, initial[:, -3:], initial, ContinuationConfig(), cached=True
    )
    assert ok.all()
    torch.testing.assert_close(q, initial)
