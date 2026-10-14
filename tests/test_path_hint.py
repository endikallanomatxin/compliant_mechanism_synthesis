import pytest
import torch

from compliant_mechanism_synthesis.optimization import example
from compliant_mechanism_synthesis.path_hint import PathHint


def test_hint_uses_relative_private_deformation_and_rejects_large_changes():
    d = example(3)
    q = d.positions.new_zeros(156)
    q[:3] = q.new_tensor([0.0001, 0, 0])
    records = [
        (0.0, q, 0.0, 0, 1.0, q.new_ones(6), None),
        (1.0, q * 2, 0.0, 0, 1.0, q.new_ones(6), None),
    ]
    hint = PathHint(d.positions, d.presence, records, 1)
    child = d.positions.clone()
    child[8, 0] += 0.00001
    assert hint.compatible(child, d.presence, 1, 0.001)
    assert not hint.compatible(child + 0.1, d.presence, 1, 0.001)
    assert not hint.compatible(child, 1 - d.presence, 1, 0.001)
    guess, private = hint.at(0.5, child, d.edges)
    torch.testing.assert_close(guess, q * 1.5)
    assert private is None


def test_private_hint_shifts_reference_geometry():
    d = example(3)
    ids = d.edges
    original = (d.positions[ids[:, 0]] + d.positions[ids[:, 1]]) / 2
    inner = torch.cat((original[:, None], original.new_zeros(len(ids), 1, 3)), -1)
    q = d.positions.new_zeros(156)
    records = [
        (0.0, q, 0.0, 0, 1.0, q.new_ones(6), inner),
        (1.0, q, 0.0, 0, 1.0, q.new_ones(6), inner),
    ]
    hint = PathHint(d.positions, d.presence, records, 2)
    child = d.positions.clone()
    child[8] += 0.0001
    _, private = hint.at(0.5, child, ids)
    expected = (child[ids[:, 0]] + child[ids[:, 1]]) / 2
    torch.testing.assert_close(private[:, 0, :3], expected)


def test_hint_does_not_replace_exact_equilibrium_or_newton(monkeypatch):
    from compliant_mechanism_synthesis import prescribed_motion
    from compliant_mechanism_synthesis.batching import DesignBatch
    from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
    from compliant_mechanism_synthesis.trajectory import Trajectory

    torch.set_num_threads(1)
    d = example(3)
    b = DesignBatch(
        d.positions[None], d.roles, d.edges, d.radii[None], d.presence[None]
    )
    poses = d.positions.new_zeros(2, 6)
    poses[1, 1] = 0.0003
    t = Trajectory(poses.new_tensor([0, 1]), poses)
    config = ContinuationConfig(steps=2)
    cold = prescribed_motion.trace_motion(b, t, config)
    records = [
        (float(s.factors[0]), s.q[0], 0.0, 0, 1.0, s.stiffness[0], None)
        for s in cold.states
    ]
    hint = PathHint(d.positions, d.presence, records, 1)
    positions = d.positions.clone()
    positions[8, 0] += 0.00001
    child = DesignBatch(
        positions[None], d.roles, d.edges, d.radii[None], d.presence[None]
    )
    original = prescribed_motion.relax_equilibrium
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args[4] if len(args) > 4 else kwargs.get("cached"))
        return original(*args, **kwargs)

    monkeypatch.setattr(prescribed_motion, "relax_equilibrium", tracked)
    warm = prescribed_motion.trace_motion(child, t, config, initial_path=[hint])
    assert calls and all(x is None for x in calls)
    reference = prescribed_motion.trace_motion(child, t, config)
    assert warm.completed.all() and reference.completed.all()
    torch.testing.assert_close(
        warm.states[-1].q, reference.states[-1].q, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("segments", [1, 2])
def test_parent_hint_preserves_preloading_and_signed_path(segments):
    from compliant_mechanism_synthesis.batching import DesignBatch
    from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
    from compliant_mechanism_synthesis.prescribed_motion import trace_motion
    from compliant_mechanism_synthesis.trajectory import Trajectory

    torch.set_num_threads(1)
    d = example(3)
    poses = d.positions.new_zeros(3, 6)
    poses[0, 1] = -0.00005
    poses[2, 1] = 0.00005
    t = Trajectory(poses.new_tensor([0, 0.5, 1]), poses, 0.5)
    c = ContinuationConfig(steps=2, beam_segments=segments)
    b = DesignBatch(
        d.positions[None], d.roles, d.edges, d.radii[None], d.presence[None]
    )
    cold = trace_motion(b, t, c)
    records = [
        (
            float(s.factors[0]),
            s.q[0].detach(),
            0.0,
            0,
            1.0,
            s.stiffness[0].detach(),
            None if s.beam_nodes is None else s.beam_nodes[0, :, 1:-1].detach(),
        )
        for s in cold.states
    ]
    h = PathHint(d.positions, d.presence, records, segments)
    warm = trace_motion(b, t, c, initial_path=[h])
    assert warm.completed.all() and cold.completed.all()
    for a in warm.states:
        reference = next(
            (r for r in cold.states if torch.equal(r.factors, a.factors)), None
        )
        if reference is not None:
            torch.testing.assert_close(a.q, reference.q, rtol=1e-6, atol=1e-9)
    assert warm.states[0].q[0, -5] < 0
    assert warm.states[-1].q[0, -5] > 0
