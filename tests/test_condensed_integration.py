from dataclasses import replace

import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch
from compliant_mechanism_synthesis.mechanics import Design
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig, configuration
from compliant_mechanism_synthesis.optimization import (
    Population,
    Search,
    SearchConfig,
    example,
)
from compliant_mechanism_synthesis.prescribed_motion import trace_motion
from compliant_mechanism_synthesis.subdivision_benchmark import expand_design
from compliant_mechanism_synthesis.trajectory import Trajectory


def seed(device="cpu"):
    old = example(3, device)
    ids = [0, 1, 2, 3, 4, 5, 8, 23, 30, 31, 32]
    p = old.positions[ids]
    roles = tuple(old.roles[i] for i in ids)
    edges = torch.tensor(
        [
            (i, j)
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
            if not (roles[i] == roles[j] and roles[i] != "free")
        ],
        device=p.device,
    )
    return Design(
        p, roles, edges, p.new_full((len(edges),), 0.001), p.new_ones(len(edges))
    )


def batch(d):
    return DesignBatch(
        d.positions[None], d.roles, d.edges, d.radii[None], d.presence[None]
    )


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_rigid_body_path_and_derivatives_match_uncondensed(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    torch.set_num_threads(1)
    original = seed(device)
    config = ContinuationConfig(beam_segments=2, steps=2)
    poses = original.positions.new_tensor(
        [[0, 0, 0, 0, 0, 0], [0, 0.0002, 0, 0, 0, 0.01]]
    )
    trajectory = Trajectory(poses.new_tensor([0, 1]), poses)
    paths = []
    gradients = []
    designs = []
    mesh_ids = None
    for condensed in (True, False):
        radii = original.radii.detach().clone().requires_grad_()
        positions = original.positions.detach().clone().requires_grad_()
        d = replace(original, radii=radii, positions=positions)
        if not condensed:
            d, mesh_ids = expand_design(d, 2)
        path = trace_motion(
            batch(d),
            trajectory,
            config if condensed else replace(config, beam_segments=1),
        )
        assert path.completed.all()
        value = (
            path.states[-1].q[0, -12:-6]
            * poses.new_tensor([0.3, 1, 0.05, 0.02, 0.03, 0.04])
        ).sum()
        gradients.append(torch.autograd.grad(value, (radii, positions)))
        paths.append(path)
        designs.append(d)
    # Condensation changes predictor error estimates, hence adaptive step sizes.
    # Compare equilibria at common progress; both paths must reach the endpoint.
    shared = [
        (a, b)
        for a in paths[0].states
        for b in paths[1].states
        if torch.equal(a.factors, b.factors)
    ]
    assert len(shared) >= 2
    assert torch.equal(paths[0].states[-1].factors, paths[1].states[-1].factors)
    for a, b in shared:
        torch.testing.assert_close(a.q[:, -12:], b.q[:, -12:], rtol=1e-6, atol=1e-9)
        full, _ = configuration(designs[1], b.q[0])
        torch.testing.assert_close(
            a.beam_nodes[0, ..., :3], full[mesh_ids], rtol=1e-6, atol=1e-9
        )
        current, _ = configuration(original, a.q[0])
        torch.testing.assert_close(current[:3], original.positions[:3], rtol=0, atol=0)
    for a, b in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_cached_subdivision_keeps_interiors_and_skips_newton(
    tmp_path, monkeypatch, device
):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    from compliant_mechanism_synthesis import condensed_mechanics

    search = Search(
        SearchConfig(
            staged_analysis=0,
            device=device,
            population=2,
            elites=1,
            input_y=0.00005,
            target_y=0.0003,
            geometry_mutation_probability=0,
            radius_mutation_probability=0,
            topology_mutation_probability=0,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(beam_segments=2, steps=2),
        initial_design=seed(device),
    )
    try:
        search.ensure_evaluated()

        def evaluate():
            p = search.population.positions.detach().requires_grad_()
            r = search.population.radius_logits.detach().requires_grad_()
            scores = search._score(Population(p, r, search.population.topology))
            return scores.detach(), torch.autograd.grad(scores.sum(), (p, r))

        def forbidden(*args, **kwargs):
            raise AssertionError("Exact cached gradient repeated equilibrium solve")

        with monkeypatch.context() as patch:
            patch.setattr(condensed_mechanics, "relax_condensed", forbidden)
            cached, gc = evaluate()
        assert search.last_diagnostics["cached_factorizations_reused"] > 0
        for _, records in search.path_cache.values():
            for record in records:
                assert record[6].shape[-2:] == (1, 6)
                assert record[6].shape[0] <= len(search.initial.edges)
                assert record[6].numel() < len(search.initial.edges) * 3 * 6
        assert all(
            record[6] is not None and not record[6].requires_grad
            for _, records in search.path_cache.values()
            for record in records
        )
        search.path_cache.clear()
        fresh, gf = evaluate()
        torch.testing.assert_close(cached, fresh, rtol=1e-7, atol=1e-8)
        for a, b in zip(gc, gf, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
        search.scores = None
        search.ensure_evaluated()
        snapshot = search.snapshot()
        assert snapshot["states"][-1]["beam_nodes"] is not None
        assert snapshot["physical_diagnostics"]["maximum_normal_strain"] >= 0
        before = search.population.topology.clone()
        search.step("gradient")
        assert torch.equal(before, search.population.topology)
    finally:
        search.writer.close()


def test_subdivided_path_can_start_before_rest():
    d = seed()
    trajectory = Trajectory(
        d.positions.new_tensor([0, 0.5, 1]),
        d.positions.new_tensor(
            [[0, -0.00005, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0.00005, 0, 0, 0, 0]]
        ),
        0.5,
    )
    path = trace_motion(
        batch(d), trajectory, ContinuationConfig(steps=2, beam_segments=2)
    )
    assert path.completed.all()
    assert path.states[0].beam_nodes is not None
    rest = next(s for s in path.states if float(s.factors[0]) == 0.5)
    assert rest.q[0, -12:].abs().max() < 1e-8


def test_synthesis_exposes_spatial_designs_only(tmp_path):
    with pytest.raises(TypeError):
        SearchConfig(staged_analysis=0, dimension=2)
    with pytest.raises(ValueError, match="dimension"):
        Search(
            SearchConfig(
                staged_analysis=0, population=2, elites=1, log_dir=str(tmp_path)
            ),
            initial_design=example(2),
        )


def test_mixed_topology_batch_preserves_private_nodes_and_zero_absent_gradients():
    d = seed()
    radii = d.radii.expand(2, -1).clone().requires_grad_()
    presence = d.presence.expand(2, -1).clone()
    presence[1, 0] = 0
    designs = DesignBatch(
        d.positions.expand(2, -1, -1), d.roles, d.edges, radii, presence
    )
    trajectory = Trajectory(
        d.positions.new_tensor([0, 1]),
        d.positions.new_tensor([[0, 0, 0, 0, 0, 0], [0, 0.00005, 0, 0, 0, 0]]),
    )
    config = ContinuationConfig(steps=2, beam_segments=2)
    path = trace_motion(designs, trajectory, config)
    assert path.completed.all()
    gradient = torch.autograd.grad(path.states[-1].q[:, -11].sum(), radii)[0]
    assert gradient[1, 0] == 0
    for i in range(2):
        part = DesignBatch(
            designs.positions[i : i + 1],
            d.roles,
            d.edges,
            radii[i : i + 1],
            presence[i : i + 1],
        )
        individual = trace_motion(part, trajectory, config)
        assert individual.completed.all()
        torch.testing.assert_close(
            path.states[-1].q[i], individual.states[-1].q[0], rtol=1e-6, atol=1e-9
        )
        actual = path.states[-1].beam_nodes[i, presence[i] != 0]
        reference = individual.states[-1].beam_nodes[0, presence[i] != 0]
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("role", ["fixed", "active", "passive"])
@pytest.mark.parametrize("invalid", ["two", "collinear"])
def test_synthesis_requires_spatial_body_geometry(tmp_path, role, invalid):
    d = example(3)
    ids = [i for i, r in enumerate(d.roles) if r == role]
    if invalid == "two":
        roles = list(d.roles)
        roles[ids[-1]] = "free"
        d = replace(d, roles=tuple(roles))
    else:
        positions = d.positions.clone()
        positions[ids[-1]] = (positions[ids[0]] + positions[ids[1]]) / 2
        d = replace(d, positions=positions)
    with pytest.raises(ValueError, match="non-collinear"):
        Search(
            SearchConfig(
                staged_analysis=0, population=2, elites=1, log_dir=str(tmp_path)
            ),
            initial_design=d,
        )


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_spatial_chain_force_and_tangent_match_independent_energy(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    from compliant_mechanism_synthesis.subdivided import (
        BeamMesh,
        chain_derivatives,
        chain_energy,
        reference_nodes,
    )

    p = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.04, 0.02, 0.01]]]], dtype=torch.float64, device=device
    )
    reference = reference_nodes(p, BeamMesh(2))
    nodes = torch.cat((reference, torch.zeros_like(reference)), -1)
    nodes[..., 1, :3] += nodes.new_tensor([0.0003, -0.0002, 0.0001])
    nodes[..., 1, 3:] = nodes.new_tensor([0.02, -0.01, 0.03])
    radii = nodes.new_full((1, 1), 0.001)
    energy, force, tangent = chain_derivatives(reference, nodes, radii)
    value = lambda z: chain_energy(reference, z.reshape_as(nodes), radii).sum()
    x = nodes.flatten()
    torch.testing.assert_close(energy.sum(), value(x), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        force.flatten(), torch.func.grad(value)(x), rtol=1e-9, atol=1e-8
    )
    torch.testing.assert_close(
        tangent[0, 0], torch.func.hessian(value)(x), rtol=1e-9, atol=1e-6
    )


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_single_segment_batched_tangent_matches_independent_energy(device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    from compliant_mechanism_synthesis.batching import kernels
    from compliant_mechanism_synthesis.mechanics import DEFAULT_MATERIAL
    from compliant_mechanism_synthesis.nonlinear import energy

    d = seed(device)
    q = d.positions.new_zeros((1, (d.roles.count("free") + 2) * 6))
    q[:, :6] = q.new_tensor([0.0001, -0.0002, 0.0001, 0.02, -0.01, 0.03])
    q[:, -12:] = q.new_tensor(
        [0, 0.0001, 0.0001, 0.01, 0.02, 0, 0, 0.0002, 0, 0, 0, 0.03]
    )
    _value, _force, tangent = kernels(batch(d), DEFAULT_MATERIAL)
    actual = tangent(d.positions[None], d.radii[None], d.presence[None], q)[0]
    reference = torch.func.jacrev(
        torch.func.grad(lambda x: energy(d, x, DEFAULT_MATERIAL))
    )(q[0])
    torch.testing.assert_close(actual, reference, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("floor", [1000.0, 1e12])
def test_support_gradient_guard_preserves_scores_and_derivatives(
    tmp_path, monkeypatch, floor
):
    from compliant_mechanism_synthesis import optimization

    torch.set_num_threads(1)
    s = Search(
        SearchConfig(
            staged_analysis=0,
            population=2,
            elites=1,
            input_y=0.00005,
            minimum_passive_support=floor,
            geometry_mutation_probability=0,
            radius_mutation_probability=0,
            topology_mutation_probability=0,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
        initial_design=seed(),
    )
    try:
        s.ensure_evaluated()
        requested = []
        original = optimization.trace_motion

        def checked(*args, **kwargs):
            requested.append(kwargs["differentiate_stiffness"])
            return original(*args, **kwargs)

        def evaluate():
            p = s.population.positions.detach().requires_grad_()
            r = s.population.radius_logits.detach().requires_grad_()
            scores = s._score(Population(p, r, s.population.topology))
            return scores.detach(), torch.autograd.grad(scores.sum(), (p, r))

        monkeypatch.setattr(optimization, "trace_motion", checked)
        fast, gfast = evaluate()
        assert not requested
        assert s.last_diagnostics["cached_factorizations_reused"] > 0
        s.linearization_cache.clear()

        def full(*args, **kwargs):
            kwargs["differentiate_stiffness"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(optimization, "trace_motion", full)
        reference, gref = evaluate()
        torch.testing.assert_close(fast, reference, rtol=1e-9, atol=1e-10)
        for a, b in zip(gfast, gref, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)
    finally:
        s.writer.close()
