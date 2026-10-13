import pytest
import torch

from compliant_mechanism_synthesis.devices import choose_device
from compliant_mechanism_synthesis.optimization import example
from compliant_mechanism_synthesis.spectral import Spectrum, symmetric_eigh


@pytest.mark.parametrize("size", [3, 6])
def test_parallel_xpu_spectrum_and_derivatives_match_cpu(size):
    if not torch.xpu.is_available():
        pytest.skip("No XPU")
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(4, size, size, dtype=torch.float64, generator=generator)
    a = raw @ raw.transpose(-1, -2) + torch.eye(size, dtype=torch.float64)
    values, vectors = symmetric_eigh(a.to("xpu"))
    torch.testing.assert_close(
        values.cpu(), torch.linalg.eigvalsh(a), rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        (a.to("xpu") @ vectors).cpu(),
        (vectors * values[:, None, :]).cpu(),
        rtol=1e-8,
        atol=1e-8,
    )
    results = []
    for device in ["cpu", "xpu"]:
        x = a.to(device).detach().requires_grad_()
        loss = (Spectrum.apply(x) * torch.arange(1, size + 1, device=device)).sum()
        results.append(torch.autograd.grad(loss, x)[0].cpu())
    torch.testing.assert_close(*results, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("segments", [1, 2])
def test_auto_benchmarks_two_body_continuation(monkeypatch, segments):
    from compliant_mechanism_synthesis import devices

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

    class Properties:
        has_fp64 = True

    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda index: Properties())
    # Exercise construction and actual prescribed-motion benchmark on CPU,
    # while verifying the auto selector supplies both rigid bodies and DOFs.
    original_tensor = torch.tensor

    def tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor)
    monkeypatch.setattr(torch.xpu, "synchronize", lambda: None)
    observed = []
    original_trace = devices.trace_motion

    def trace(batch, trajectory, config):
        observed.append(batch.roles)
        assert config.beam_segments == segments
        assert "active" in batch.roles and "passive" in batch.roles
        return original_trace(batch, trajectory, config)

    monkeypatch.setattr(devices, "trace_motion", trace)
    design = example(2)
    choose_device.cache_clear()
    device, timings = choose_device(
        2,
        2,
        len(design.positions),
        tuple(map(tuple, design.positions.tolist())),
        design.roles,
        tuple(map(tuple, design.edges.tolist())),
        0.001,
        2,
        beam_segments=segments,
    )
    assert device in ["cpu", "xpu"]
    assert set(timings) == {"cpu", "xpu"}
    assert len(observed) == 4
    choose_device.cache_clear()
