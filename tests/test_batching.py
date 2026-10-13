import pytest
import torch

from compliant_mechanism_synthesis.batching import DesignBatch, trace_batch
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig, trace
from compliant_mechanism_synthesis.optimization import Search, SearchConfig, example


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_batched_equilibria_and_gradients_match_individual(dim, device):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    d = example(dim, device)
    radii = torch.stack((d.radii, d.radii * 1.1)).detach().requires_grad_(True)
    batch = DesignBatch(
        d.positions.expand(2, -1, -1).clone(),
        d.roles,
        d.edges,
        radii,
        d.presence.expand(2, -1).clone(),
    )
    load = d.positions.new_zeros(3 if dim == 2 else 6)
    load[1] = 10
    config = ContinuationConfig(steps=2)
    path = trace_batch(batch, load, config)
    assert path.completed.all()
    values = path.states[-1].q[:, -len(load) + 1]
    gradients = torch.autograd.grad(values.sum(), radii)[0]
    for i in range(2):
        individual = trace(batch.item(i), load, config).states[-1].q
        gradient = torch.autograd.grad(
            individual[-len(load) + 1], radii, retain_graph=True
        )[0][i]
        assert torch.allclose(path.states[-1].q[i], individual, rtol=1e-6, atol=1e-8)
        assert torch.allclose(gradients[i], gradient, rtol=1e-5, atol=1e-6)


def test_independent_failure_does_not_poison_batch():
    d = example(2)
    batch = DesignBatch(
        d.positions.expand(2, -1, -1).clone(),
        d.roles,
        d.edges,
        d.radii.expand(2, -1).clone(),
        d.presence.expand(2, -1).clone(),
    )
    loads = d.positions.new_tensor([[0.0, 0.0001, 0.0], [0.0, 100.0, 0.0]])
    path = trace_batch(
        batch,
        loads,
        ContinuationConfig(steps=1, max_iterations=2, minimum_increment=0.2),
    )
    assert path.completed.tolist() == [True, False]


def test_population_persists_and_evolves_in_batches(tmp_path):
    s = Search(
        SearchConfig(device="cpu", population=8, elites=2, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
    )
    s.ensure_evaluated()
    before = float(s.scores[0])
    elite = s.population.take(torch.arange(2)).detached()
    result = s.step("evolution")
    assert s.population.size == 8
    assert result["batch"]["batch_size"] >= 6
    assert (
        result["batch"]["offspring_gradient_steps"] == s.config.offspring_gradient_steps
    )
    assert result["objective"] <= before
    assert any(torch.equal(elite.positions[0], p) for p in s.population.positions)
    assert len(result["population"]) == 8
    assert torch.isfinite(s.scores).all()
    result = s.step("gradient")
    assert 8 <= result["batch"]["batch_size"] <= 8 * s.config.gradient_backtracks
    assert sum(result["batch"]["gradient_trials"]) == result["batch"]["batch_size"]
    assert result["objective"] <= before
    assert torch.equal(
        s.population.positions[:, :4], s.initial.positions[:4].expand(8, -1, -1)
    )
    s.writer.close()


def test_richer_example_and_cached_view(tmp_path, monkeypatch):
    search = Search(
        SearchConfig(device="cpu", population=4, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=3),
    )
    assert len(search.initial.positions) == 33
    assert int(search.population.topology[0].sum()) == len(example(3).edges)
    assert search.target[0] != 0 and search.target[-1] != 0
    before = search.snapshot()

    def unexpected(*args, **kwargs):
        raise AssertionError("Cached view repeated FEM")

    from compliant_mechanism_synthesis import optimization

    with monkeypatch.context() as patch:
        patch.setattr(optimization, "trace_motion", unexpected)
        assert search.snapshot() is before
    search.target[1] += 0.001
    assert search.snapshot() is not before
    search.writer.close()


def test_evolution_mutates_topology_but_gradient_keeps_it(tmp_path):
    search = Search(
        SearchConfig(
            device="cpu",
            population=4,
            elites=1,
            topology_mutation_probability=1,
            topology_distance_power=0,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )
    population = search.population.detached()
    changed = search._mutate(population)
    assert (
        (changed.topology ^ population.topology).sum(-1)
        == search.config.topology_mutation_edits
    ).all()
    # Start from the connected seed topology for differentiable proposals.
    search.population.topology[:] = population.topology[0]
    search.scores = None
    search.ensure_evaluated()
    proposals = search._gradient_proposals()
    assert torch.equal(
        proposals.topology,
        search.population.topology.repeat(search.config.gradient_backtracks, 1),
    )
    search.writer.close()


def test_custom_seed_keeps_geometry_topology_and_radii(tmp_path):
    from dataclasses import replace

    seed = example(3)
    seed = replace(seed, positions=seed.positions.clone(), radii=seed.radii * 0.85)
    seed.positions[4, 0] += 0.002
    search = Search(
        SearchConfig(device="cpu", population=4, elites=1, log_dir=str(tmp_path)),
        ContinuationConfig(steps=2),
        seed,
    )
    assert torch.equal(search.population.positions[0], seed.positions)
    active = search.population.topology[0]
    assert int(active.sum()) == len(seed.edges)
    assert torch.allclose(
        search.batch_design(search.population).radii[0, active], seed.radii
    )
    assert search.snapshot()["converged"]
    search.writer.close()


def test_focused_offspring_uses_only_leading_parent_pool(tmp_path):
    search = Search(
        SearchConfig(
            device="cpu",
            population=16,
            elites=4,
            focused_fraction=1,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )
    # Ranked parents have identifiable geometry; zero stochastic mutation so
    # membership can be checked without depending on a random score outcome.
    search.scores = torch.arange(16, dtype=torch.float64)
    from dataclasses import replace

    search.config = replace(search.config, topology_mutation_probability=0)
    children = search._offspring()
    counts = search.parent_diagnostics["parent_counts"]
    assert children.size == 12
    assert sum(counts) == 12
    assert sum(counts[4:]) == 0
    assert search.parent_diagnostics["exploratory_children"] == 0
    search.writer.close()


def test_mutation_probabilities_can_disable_each_operator(tmp_path):
    search = Search(
        SearchConfig(
            device="cpu",
            population=4,
            elites=1,
            geometry_mutation_probability=0,
            radius_mutation_probability=0,
            topology_mutation_probability=0,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )
    before = search.population.detached()
    after = search._mutate(before)
    for name in before.__dataclass_fields__:
        assert torch.equal(getattr(before, name), getattr(after, name))
    assert len(search.initial.positions) == 33
    assert len(search.initial.edges) == 519
    search.writer.close()


def test_bounded_batches_preserve_objectives_and_gradients(tmp_path):
    from dataclasses import replace

    search = Search(
        SearchConfig(
            device="cpu",
            population=4,
            elites=1,
            evaluation_batch_size=4,
            log_dir=str(tmp_path),
        ),
        ContinuationConfig(steps=2),
    )

    # Compare identical equilibria: adaptive primal grids can differ between
    # independently solved batches, affecting sampled worst-case collisions.
    search.ensure_evaluated()

    def evaluate(size):
        search.config = replace(search.config, evaluation_batch_size=size)
        p = search.population.positions.detach().requires_grad_(True)
        r = search.population.radius_logits.detach().requires_grad_(True)
        from compliant_mechanism_synthesis.optimization import Population

        score = search._score(Population(p, r, search.population.topology))
        gradients = torch.autograd.grad(score.sum(), (p, r))
        assert search.last_diagnostics["peak_batch_size"] <= size
        return score.detach(), gradients

    full, full_grads = evaluate(4)
    bounded, bounded_grads = evaluate(2)
    assert torch.allclose(full, bounded, rtol=1e-6, atol=1e-8)
    for a, b in zip(full_grads, bounded_grads, strict=True):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
    search.writer.close()


@pytest.mark.parametrize("dim", [2, 3])
def test_pruned_newton_beams_preserve_tangent_and_radius_derivatives(dim):
    from compliant_mechanism_synthesis.batching import active_beams, kernels
    from compliant_mechanism_synthesis.mechanics import DEFAULT_MATERIAL

    d = example(dim)
    edges = torch.cat((d.edges, torch.tensor([[0, 1]])))
    radii = torch.cat((d.radii, d.radii[:1])).expand(2, -1).clone()
    presence = torch.cat((d.presence, d.presence.new_zeros(1))).expand(2, -1).clone()
    presence[1, 0] = 0  # A beam active in only one member must survive pruning.
    batch = DesignBatch(d.positions.expand(2, -1, -1), d.roles, edges, radii, presence)
    pruned = active_beams(batch.detached())
    assert len(pruned.edges) == len(d.edges)
    dof = 3 if dim == 2 else 6
    n = (sum(role == "free" for role in d.roles) + 2) * dof
    q = d.positions.new_zeros(2, n)
    q[:, -dof + 1] = 0.0001
    for full_kernel, pruned_kernel in zip(
        kernels(batch, DEFAULT_MATERIAL), kernels(pruned, DEFAULT_MATERIAL)
    ):
        full = full_kernel(batch.positions, batch.radii, batch.presence, q)
        reduced = pruned_kernel(pruned.positions, pruned.radii, pruned.presence, q)
        torch.testing.assert_close(full, reduced, rtol=1e-10, atol=1e-8)

    radii.requires_grad_(True)
    load = q.new_zeros(dof)
    load[1] = 1
    config = ContinuationConfig(steps=1)
    path = trace_batch(batch, load, config)
    assert path.completed.all()
    gradient = torch.autograd.grad(path.states[-1].q[:, -dof + 1].sum(), radii)[0]
    for index in range(2):
        individual = trace(batch.item(index), load, config)
        reference = torch.autograd.grad(
            individual.states[-1].q[-dof + 1], radii, retain_graph=True
        )[0][index]
        torch.testing.assert_close(gradient[index], reference, rtol=1e-5, atol=1e-6)
