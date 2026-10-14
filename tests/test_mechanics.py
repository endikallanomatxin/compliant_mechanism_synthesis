import math
from dataclasses import replace

import pytest
import torch

from compliant_mechanism_synthesis.mechanics import (
    Design,
    Material,
    assemble,
    reduction,
    solve,
)
from compliant_mechanism_synthesis.nonlinear import (
    ContinuationConfig,
    energy,
    rotation,
    trace,
)
from compliant_mechanism_synthesis.optimization import Search, SearchConfig, example

torch.set_num_threads(1)


def cantilever(dim):
    p = torch.zeros((2, dim), dtype=torch.float64)
    p[1, 0] = 0.1
    return Design(
        p,
        ("fixed", "passive"),
        torch.tensor([[0, 1]]),
        p.new_tensor([0.001]),
        p.new_ones(1),
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_cantilever_linear_analytical(dim):
    d = cantilever(dim)
    load = d.positions.new_zeros(3 if dim == 2 else 6)
    load[1] = 0.01
    response = solve(d, load, Material(regularization=0))
    inertia = math.pi * 0.001**4 / 4
    expected = 0.01 * 0.1**3 / (3 * Material().young_modulus * inertia)
    assert float(response.output[1]) == pytest.approx(expected, rel=1e-10)
    assert response.residual < 1e-10
    assert torch.allclose(assemble(d), assemble(d).T)


@pytest.mark.parametrize("dim", [2, 3])
def test_nonlinear_tangent_matches_linear_at_origin(dim):
    d = example(dim)
    t, _ = reduction(d)
    q = d.positions.new_zeros(t.shape[1])
    h = torch.autograd.functional.hessian(lambda x: energy(d, x), q)
    assert torch.allclose(h, t.T @ assemble(d) @ t, atol=1e-7, rtol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_rigid_rotation_has_zero_energy(dim):
    d = cantilever(dim)
    d.roles = ("free", "free")
    angle = 0.7
    if dim == 2:
        r = d.positions.new_tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        rotations = d.positions.new_tensor([[angle], [angle]])
    else:
        vector = d.positions.new_tensor([0.2, 0.3, 0.7])
        r = rotation(vector)
        rotations = vector.expand(2, -1)
    delta = (
        d.positions @ r.T - d.positions + d.positions.new_full(d.positions.shape, 0.02)
    )
    q = torch.cat((delta, rotations), 1).reshape(-1)
    # No mobile node: append the unused body coordinates to configuration.
    q = torch.cat((q, q.new_zeros(3 if dim == 2 else 6)))
    assert energy(d, q).abs() < 1e-18


@pytest.mark.parametrize("dim", [2, 3])
def test_continuation_equilibrium_and_small_load_limit(dim):
    d = cantilever(dim)
    load = d.positions.new_zeros(3 if dim == 2 else 6)
    load[1] = 0.0001
    path = trace(d, load, ContinuationConfig(steps=3))
    assert path.converged
    assert [s.factor for s in path.states] == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert torch.allclose(
        path.states[-1].q, solve(d, load).output, rtol=1e-4, atol=1e-8
    )
    assert max(s.residual for s in path.states) < 1e-7
    assert torch.allclose(path.states[-1].reaction, load, atol=1e-7)


def test_large_deflection_is_nonlinear():
    d = cantilever(2)
    load = d.positions.new_tensor([0.0, 10.0, 0.0])
    path = trace(d, load, ContinuationConfig(steps=10, max_iterations=80))
    assert path.converged
    assert abs(float(path.states[-1].q[0].detach())) > 0.001
    assert not torch.allclose(
        path.states[-1].q, solve(d, load).output, rtol=0.05, atol=1e-6
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_equilibrium_radius_gradient_matches_finite_difference(dim):
    d = cantilever(dim)
    d.radii = d.radii.clone().requires_grad_(True)
    load = d.positions.new_zeros(3 if dim == 2 else 6)
    load[1] = 0.1
    config = ContinuationConfig(steps=2)
    y = trace(d, load, config).states[-1].q[1]
    gradient = torch.autograd.grad(y, d.radii)[0][0]
    eps = 1e-7
    plus = trace(replace(d, radii=d.radii.detach() + eps), load, config).states[-1].q[1]
    minus = (
        trace(replace(d, radii=d.radii.detach() - eps), load, config).states[-1].q[1]
    )
    assert float(gradient.detach()) == pytest.approx(
        float(((plus - minus) / (2 * eps)).detach()), rel=2e-3
    )


def test_nonconvergence_is_explicit():
    d = cantilever(2)
    load = d.positions.new_tensor([0.0, 10.0, 0.0])
    path = trace(
        d, load, ContinuationConfig(steps=1, max_iterations=1, minimum_increment=0.2)
    )
    assert not path.converged
    assert path.states[-1].factor < 1


@pytest.mark.parametrize("method", ["gradient", "evolution"])
def test_search_is_elitist_and_keeps_anchors(tmp_path, method):
    search = Search(
        SearchConfig(
            staged_analysis=0,
            log_dir=str(tmp_path),
            population=4,
            elites=1,
            device="cpu",
        ),
        ContinuationConfig(steps=2),
    )
    before = search.snapshot()["objective"]
    after = search.step(method)
    assert after["objective"] <= before + 1e-12
    assert torch.equal(search.positions[:4], search.initial.positions[:4])
    assert list(tmp_path.glob("events.out.tfevents.*"))
    search.writer.close()


@pytest.mark.parametrize("variable", ["positions"])
@pytest.mark.parametrize("dim", [2, 3])
def test_geometry_equilibrium_gradients(dim, variable):
    d = cantilever(dim)
    values = getattr(d, variable).detach().clone()
    values.requires_grad_(True)
    d = replace(d, **{variable: values})
    load = d.positions.new_zeros(3 if dim == 2 else 6)
    load[1] = 0.05
    config = ContinuationConfig(steps=2)
    value = trace(d, load, config).states[-1].q[1]
    grad = torch.autograd.grad(value, values)[0]
    index = (1, 0) if variable == "positions" else (0,)
    eps = 1e-6
    delta = torch.zeros_like(values)
    delta[index] = eps
    plus = (
        trace(replace(d, **{variable: values.detach() + delta}), load, config)
        .states[-1]
        .q[1]
    )
    minus = (
        trace(replace(d, **{variable: values.detach() - delta}), load, config)
        .states[-1]
        .q[1]
    )
    assert float(grad[index].detach()) == pytest.approx(
        float(((plus - minus) / (2 * eps)).detach()), rel=2e-3
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_beam_ends_transmit_clamping_moments(dim):
    d = cantilever(dim)
    dof = 3 if dim == 2 else 6
    u = d.positions.new_zeros(2 * dof)
    u[dof + 1] = 0.001
    reactions = assemble(d) @ u
    inertia = math.pi * 0.001**4 / 4
    expected_force = 12 * Material().young_modulus * inertia / 0.1**3 * 0.001
    expected_moment = -6 * Material().young_modulus * inertia / 0.1**2 * 0.001
    moment_axis = 2 if dim == 2 else 5
    assert float(reactions[dof + 1]) == pytest.approx(expected_force)
    assert float(reactions[moment_axis]) == pytest.approx(expected_moment)
    assert float(reactions[dof + moment_axis]) == pytest.approx(expected_moment)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU not available")
@pytest.mark.parametrize("dim", [2, 3])
def test_intel_nonlinear_equilibrium_and_gradients_match_cpu(dim):
    d = cantilever(dim)
    config = ContinuationConfig(steps=2)
    results = []
    for device in ("cpu", "xpu"):
        dd = replace(
            d,
            positions=d.positions.to(device),
            edges=d.edges.to(device),
            radii=d.radii.to(device).detach().requires_grad_(True),
            presence=d.presence.to(device),
        )
        load = dd.positions.new_zeros(3 if dim == 2 else 6)
        load[1] = 0.1
        path = trace(dd, load, config)
        assert path.converged
        out = path.states[-1].q
        gradient = torch.autograd.grad(out[1], dd.radii)[0]
        results.append((out.detach().cpu(), gradient.detach().cpu()))
    assert torch.allclose(results[0][0], results[1][0], rtol=1e-6, atol=1e-9)
    assert torch.allclose(results[0][1], results[1][1], rtol=1e-5, atol=1e-8)
