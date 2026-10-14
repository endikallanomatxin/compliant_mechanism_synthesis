import pytest
import torch

from compliant_mechanism_synthesis.equilibrium_system import (
    coordinate_scale,
    factor_system,
    scaled_descent,
)


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_scaled_factorization_matches_physical_solve(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 12, 12, generator=g, dtype=torch.float64).to(device)
    h = x @ x.transpose(-1, -2) + torch.eye(12, dtype=x.dtype, device=device)
    rhs = torch.randn(2, 12, 3, generator=g, dtype=x.dtype).to(device)
    scale = coordinate_scale(h, 6, 0.05)
    context = factor_system(h, scale)
    assert context.valid.all()
    torch.testing.assert_close(
        context.solve(rhs), torch.linalg.solve(h, rhs), rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(
        context.solve(rhs[..., 0]),
        torch.linalg.solve(h, rhs[..., 0]),
        rtol=1e-10,
        atol=1e-10,
    )
    step = scaled_descent(h, rhs[..., 0], 6, 0.05, 0)
    torch.testing.assert_close(
        step, -torch.linalg.solve(h, rhs[..., 0]), rtol=1e-10, atol=1e-10
    )


def test_invalid_systems_remain_invalid_and_safe():
    h = torch.eye(6, dtype=torch.float64).expand(4, -1, -1).clone()
    h[1, 0, 0] = -1
    h[2, 0, 0] = 0
    h[3, 0, 0] = torch.nan
    c = factor_system(h, coordinate_scale(h, 6, 0.05))
    assert c.valid.tolist() == [True, False, False, False]
    out = c.solve(torch.ones(4, 6, dtype=h.dtype))
    assert torch.isfinite(out).all()
    assert out[1:].eq(0).all()


def test_cpu_positivity_uses_native_factorization(monkeypatch):
    from compliant_mechanism_synthesis.spectral import positive_definite

    original = torch.linalg.cholesky_ex
    calls = []

    def checked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", checked)
    assert positive_definite(torch.eye(150, dtype=torch.float64)[None]).all()
    assert calls
