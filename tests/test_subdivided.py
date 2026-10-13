import pytest
import torch

from compliant_mechanism_synthesis.mechanics import DEFAULT_MATERIAL
from compliant_mechanism_synthesis.subdivided import (
    BeamMesh,
    NetworkConfig,
    assemble_condensed,
    chain_derivatives,
    condense,
    initial_nodes,
    reference_nodes,
    solve_network,
)
from compliant_mechanism_synthesis.tangents import planar_element_derivatives


def problem(device="cpu"):
    p = torch.tensor(
        [[[0.0, 0], [0.05, 0.015], [0.1, 0]]], dtype=torch.float64, device=device
    )
    edges = torch.tensor([[0, 1], [1, 2]], device=device)
    radii = p.new_tensor([[0.001, 0.0008]])
    prescribed = torch.tensor([[True] * 3, [False] * 3, [True] * 3], device=device)
    boundary = torch.cat((p, p.new_zeros(1, 3, 1)), -1)
    boundary[0, 2, 1] = 0.003
    boundary[0, 2, 2] = 0.04
    return p, edges, radii, prescribed, boundary


def test_one_segment_is_the_original_kernel():
    p, e, r, _, b = problem()
    ref = reference_nodes(p[:, e], BeamMesh(1))
    nodes = b[:, e]
    _, g, h = chain_derivatives(ref, nodes, r)
    expected = planar_element_derivatives(
        (p[:, e[:, 1]] - p[:, e[:, 0]]).flatten(0, 1),
        (nodes[..., 1, :2] - nodes[..., 0, :2]).flatten(0, 1),
        nodes[..., 2].flatten(0, 1),
        r.flatten(),
        r.new_ones(2),
        DEFAULT_MATERIAL,
    )
    torch.testing.assert_close(g.flatten(0, 1), expected[0])
    torch.testing.assert_close(h.flatten(0, 1), expected[1])
    blocks = condense(g, h)
    torch.testing.assert_close(blocks.tangent, h)
    assert blocks.interior_offset.shape[-1] == 0


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_condensed_newton_step_equals_full_coupled_system(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    p, e, r, mask, b = problem(device)
    mesh = BeamMesh(3)
    ref = reference_nodes(p[:, e], mesh)
    nodes = initial_nodes(ref, b[:, e])
    nodes[..., 1:-1, 1] += 0.0002
    _, g, h = chain_derivatives(ref, nodes, r)
    blocks = condense(g, h)
    gg, hh = assemble_condensed(blocks, e, 3)
    free = torch.nonzero(~mask.flatten()).flatten()
    step = torch.linalg.solve(hh[:, free[:, None], free[None, :]], -gg[:, free])
    global_step = torch.zeros_like(b).flatten(1).index_copy(1, free, step).reshape_as(b)
    endpoint_step = global_step[:, e].flatten(-2)
    internal_step = blocks.recover(endpoint_step)
    # Independent full assembly: three graph nodes, then six private DOFs/beam.
    total = 9 + 2 * mesh.internal_dofs
    ids = torch.stack(
        [
            torch.cat(
                (
                    e[k, :1] * 3 + torch.arange(3, device=p.device),
                    torch.arange(9 + k * 6, 9 + (k + 1) * 6, device=p.device),
                    e[k, 1:] * 3 + torch.arange(3, device=p.device),
                )
            )
            for k in range(2)
        ]
    )
    full_g = g.new_zeros(1, total).index_add(-1, ids.flatten(), g.flatten(-2))
    matrix_ids = (ids[:, :, None] * total + ids[:, None, :]).flatten()
    full_h = (
        h.new_zeros(1, total * total)
        .index_add(-1, matrix_ids, h.flatten(-3))
        .reshape(1, total, total)
    )
    unknown = torch.cat((free, torch.arange(9, total, device=p.device)))
    direct = torch.linalg.solve(
        full_h[:, unknown[:, None], unknown[None, :]], -full_g[:, unknown]
    )
    torch.testing.assert_close(step, direct[:, : len(free)], rtol=1e-7, atol=1e-10)
    torch.testing.assert_close(
        internal_step.flatten(1), direct[:, len(free) :], rtol=1e-7, atol=1e-10
    )
    assert blocks.info.eq(0).all()


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_nonlinear_network_continuation_and_radius_derivative(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    p, e, r, mask, b = problem(device)
    radii = r.clone().requires_grad_()
    state = solve_network(p, e, radii, mask, b, NetworkConfig(segments=3))
    assert state.converged.all()
    assert state.residual.max() < 1e-7
    torch.testing.assert_close(state.nodes[:, mask], b[:, mask])
    torch.testing.assert_close(state.beam_nodes[:, :, 0], state.nodes[:, e[:, 0]])
    torch.testing.assert_close(state.beam_nodes[:, :, -1], state.nodes[:, e[:, 1]])
    grad = torch.autograd.grad(state.nodes[0, 1, 1], radii)[0]
    eps = 1e-7
    delta = torch.zeros_like(r)
    delta[0, 0] = eps
    plus = solve_network(p, e, r + delta, mask, b, NetworkConfig(segments=3))
    minus = solve_network(p, e, r - delta, mask, b, NetworkConfig(segments=3))
    assert plus.converged.all() and minus.converged.all()
    fd = (plus.nodes[0, 1, 1] - minus.nodes[0, 1, 1]) / (2 * eps)
    torch.testing.assert_close(grad[0, 0], fd, rtol=1e-3, atol=1e-5)
    next_boundary = b.clone()
    next_boundary[0, 2, 1] += 0.0003
    warm = solve_network(
        p, e, r, mask, next_boundary, NetworkConfig(segments=3), initial=state
    )
    cold = solve_network(p, e, r, mask, next_boundary, NetworkConfig(segments=3))
    assert warm.converged.all() and cold.converged.all()
    torch.testing.assert_close(warm.nodes, cold.nodes, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(warm.beam_nodes, cold.beam_nodes, rtol=1e-7, atol=1e-9)


def test_failed_internal_solve_is_exposed():
    g = torch.zeros(1, 9, dtype=torch.float64)
    h = torch.zeros(1, 9, 9, dtype=torch.float64)
    assert condense(g, h).info.ne(0).all()


@pytest.mark.parametrize("segments", [0, -1, 1.5, True])
def test_invalid_mesh_is_rejected(segments):
    with pytest.raises(ValueError):
        BeamMesh(segments)


def test_batched_network_matches_individual_networks():
    p, e, r, mask, b = problem()
    positions = p.expand(3, -1, -1).clone()
    radii = r.expand(3, -1).clone()
    boundary = b.expand(3, -1, -1).clone()
    boundary[:, 2, 1] = boundary.new_tensor([0, 0.001, 0.004])
    boundary[:, 2, 2] = boundary.new_tensor([0, 0.02, 0.06])
    config = NetworkConfig(segments=4)
    batch = solve_network(positions, e, radii, mask, boundary, config)
    assert batch.converged.all()
    for k in range(3):
        single = solve_network(
            positions[k : k + 1], e, radii[k : k + 1], mask, boundary[k : k + 1], config
        )
        assert single.converged.all()
        torch.testing.assert_close(
            batch.nodes[k], single.nodes[0], rtol=1e-7, atol=1e-9
        )
        torch.testing.assert_close(
            batch.beam_nodes[k], single.beam_nodes[0], rtol=1e-7, atol=1e-9
        )


def test_internal_mesh_refinement_converges():
    p, e, r, mask, b = problem()
    states = [
        solve_network(p, e, r, mask, b, NetworkConfig(segments=s)) for s in [2, 4, 8]
    ]
    assert all(state.converged.all() for state in states)
    coarse = (states[0].nodes - states[1].nodes).norm()
    fine = (states[1].nodes - states[2].nodes).norm()
    assert fine < coarse


def test_rest_condensation_recovers_analytical_single_beam_stiffness():
    p, e, r, _, b = problem()
    b[:, :, 1] = p[:, :, 1]
    b[:, :, 2] = 0
    ref = reference_nodes(p[:, e], BeamMesh(5))
    _, g, h = chain_derivatives(ref, initial_nodes(ref, b[:, e]), r)
    blocks = condense(g, h)
    _, original_h = planar_element_derivatives(
        (p[:, e[:, 1]] - p[:, e[:, 0]]).flatten(0, 1),
        (p[:, e[:, 1]] - p[:, e[:, 0]]).flatten(0, 1),
        r.new_zeros(2, 2),
        r.flatten(),
        r.new_ones(2),
        DEFAULT_MATERIAL,
    )
    torch.testing.assert_close(
        blocks.tangent.flatten(0, 1), original_h, rtol=1e-9, atol=1e-5
    )


def test_chain_residual_and_tangent_match_energy_derivatives():
    from compliant_mechanism_synthesis.subdivided import chain_energy

    p, e, r, _, b = problem()
    ref = reference_nodes(p[:, e], BeamMesh(3))
    nodes = initial_nodes(ref, b[:, e])
    _, g, h = chain_derivatives(ref, nodes, r)
    # Independent energy differentiation, including the nonzero internal residual.
    x = nodes[0, 0].flatten()
    value = lambda z: chain_energy(ref[:, :1], z.reshape(1, 1, 4, 3), r[:, :1]).sum()
    torch.testing.assert_close(g[0, 0], torch.func.grad(value)(x), rtol=1e-9, atol=1e-6)
    torch.testing.assert_close(
        h[0, 0], torch.func.hessian(value)(x), rtol=1e-9, atol=1e-5
    )


def test_nonlinear_condensed_solution_matches_uncondensed_network():
    from compliant_mechanism_synthesis.subdivided import chain_energy

    p, e, r, mask, b = problem()
    mesh = BeamMesh(3)
    ref = reference_nodes(p[:, e], mesh)
    initial = initial_nodes(ref, b[:, e])
    x = torch.cat((b.flatten(), initial[..., 1:-1, :].flatten()))
    # The uncondensed solve retains graph junctions and every private mesh DOF.
    free = torch.cat(
        (torch.nonzero(~mask.flatten()).flatten(), torch.arange(9, len(x)))
    )

    def chains(value):
        graph = value[:9].reshape(1, 3, 3)
        internal = value[9:].reshape(1, 2, 2, 3)
        return torch.cat(
            (graph[:, e[:, 0], None], internal, graph[:, e[:, 1], None]), -2
        )

    def energy(value):
        return chain_energy(ref, chains(value), r).sum()

    for _ in range(30):
        g = torch.func.grad(energy)(x)
        if g[free].norm() < 1e-7:
            break
        h = torch.func.hessian(energy)(x)
        step = torch.linalg.solve(h[free[:, None], free[None, :]], -g[free])
        accepted = False
        for k in range(18):
            trial = x.index_add(0, free, 0.5**k * step)
            if (
                energy(trial)
                <= energy(x) + 1e-4 * 0.5**k * (g[free] * step).sum() + 1e-14
            ):
                x = trial
                accepted = True
                break
        assert accepted
    assert torch.func.grad(energy)(x)[free].norm() < 1e-7
    condensed = solve_network(p, e, r, mask, b, NetworkConfig(segments=3))
    assert condensed.converged.all()
    torch.testing.assert_close(
        condensed.nodes, x[:9].reshape(1, 3, 3), rtol=1e-7, atol=1e-9
    )
    torch.testing.assert_close(condensed.beam_nodes, chains(x), rtol=1e-7, atol=1e-9)
