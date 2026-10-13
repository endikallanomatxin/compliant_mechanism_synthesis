import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import example
from compliant_mechanism_synthesis.prescribed_motion import (
    newton_masked,
    relax_equilibrium,
)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_masked_newton_matches_compaction_and_preserves_derivatives(dimension, device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    design = example(dimension, device)
    dof = 3 if dimension == 2 else 6
    initial = design.positions.new_zeros(3, 26 * dof)
    body = initial.new_zeros(3, dof)
    body[:, 1] = body.new_tensor([0, 0.0003, 0.0006])
    body[2, -1] = 0.02
    results = []
    gradients = []
    for mode in ["compact", "masked"]:
        radii = design.radii.expand(3, -1).detach().clone().requires_grad_()
        batch = DesignBatch(
            design.positions.expand(3, -1, -1),
            design.roles,
            design.edges,
            radii,
            design.presence.expand(3, -1),
        )
        q, ok, residual, counts, *_ = relax_equilibrium(
            batch, body, initial, ContinuationConfig(newton_batch_mode=mode)
        )
        assert ok.all()
        assert torch.isfinite(residual).all()
        results.append((q.detach(), counts))
        gradients.append(torch.autograd.grad(q[:, -2 * dof + 1].sum(), radii)[0])
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-7, atol=1e-10)
    torch.testing.assert_close(results[0][1], results[1][1])
    assert results[1][1][0] == 0
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-5, atol=1e-7)


def test_masked_newton_does_not_compact_inside_iteration(monkeypatch):
    design = example(2)
    batch = DesignBatch(
        design.positions.expand(2, -1, -1),
        design.roles,
        design.edges,
        design.radii.expand(2, -1),
        design.presence.expand(2, -1),
    )
    body = design.positions.new_tensor([[0, 0, 0], [0, 0.0003, 0]])
    initial = body.new_zeros(2, 78)
    initial[:, -3:] = body

    def forbidden(*args, **kwargs):
        raise AssertionError("Masked Newton compacted its batch")

    monkeypatch.setattr(torch, "nonzero", forbidden)
    q, ok, counts = newton_masked(batch, body, initial, ContinuationConfig())
    assert ok.all()
    assert counts[0] == 0 and counts[1] > 0
    torch.testing.assert_close(q[0], initial[0])


def test_masked_newton_keeps_finished_members_when_another_fails(monkeypatch):
    from compliant_mechanism_synthesis import prescribed_motion

    design = example(2)
    batch = DesignBatch(
        design.positions.expand(2, -1, -1),
        design.roles,
        design.edges,
        design.radii.expand(2, -1),
        design.presence.expand(2, -1),
    )
    original = prescribed_motion.kernels

    def kernels(*args, **kwargs):
        value, gradient, tangent = original(*args, **kwargs)

        def failed(*inputs):
            g = gradient(*inputs)
            g[1] = float("nan")
            return g

        return value, failed, tangent

    monkeypatch.setattr(prescribed_motion, "kernels", kernels)
    q = batch.positions.new_zeros(2, 78)
    result, ok, counts = newton_masked(batch, q[:, -3:], q, ContinuationConfig())
    assert ok.tolist() == [True, False]
    torch.testing.assert_close(result[0], q[0])
    assert counts[0] == 0


def test_batch_mode_is_validated():
    with pytest.raises(ValueError):
        ContinuationConfig(newton_batch_mode="invalid")
