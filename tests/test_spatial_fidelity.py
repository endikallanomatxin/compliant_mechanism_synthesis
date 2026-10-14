from types import SimpleNamespace

import pytest
import torch

from compliant_mechanism_synthesis.fidelity import mesh_pose_error
from compliant_mechanism_synthesis.mechanics import Design
from compliant_mechanism_synthesis.nonlinear import ContinuationConfig
from compliant_mechanism_synthesis.optimization import (
    Population,
    Search,
    SearchConfig,
    example,
)
from compliant_mechanism_synthesis.prescribed_motion import MotionPath


def small_design(device):
    source = example(3, device)
    ids = [0, 1, 2, 3, 4, 5, 8, 23, 30, 31, 32]
    positions = source.positions[ids]
    roles = tuple(source.roles[i] for i in ids)
    edges = torch.tensor(
        [
            (i, j)
            for i in range(11)
            for j in range(i + 1, 11)
            if roles[i] != roles[j] or roles[i] == "free"
        ],
        device=device,
    )
    return Design(
        positions,
        roles,
        edges,
        positions.new_full((len(edges),), 0.001),
        positions.new_ones(len(edges)),
    )


def search(tmp_path, device="cpu", **settings):
    torch.set_num_threads(1)
    config = SearchConfig(
        device=device,
        staged_analysis=0,
        population=2,
        elites=1,
        input_y=0.00001,
        mesh_min_iterations=1,
        mesh_check_interval=1,
        mesh_max_iterations=1,
        fidelity_check_candidates=1,
        log_dir=str(tmp_path),
        **settings,
    )
    return Search(config, ContinuationConfig(steps=2), small_design(device))


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_refinement_reevaluates_population_and_retains_design_gradients(
    tmp_path, device
):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("No XPU")
    s = search(
        tmp_path,
        device,
        mesh_max_segments=2,
        mesh_position_tolerance=1e-12,
        mesh_orientation_tolerance=1e-12,
    )
    try:
        s.ensure_evaluated()
        s.iteration = 1
        s._refine_beams()
        assert s.continuation.beam_segments == 2
        assert s.mesh_report["promoted"] and s.mesh_report["common_samples"] >= 3
        assert s.scores.isfinite().all()
        torch.testing.assert_close(s._score(s.population), s.scores)
        p = s.population.positions.detach().requires_grad_()
        r = s.population.radius_logits.detach().requires_grad_()
        loss = s._score(Population(p, r, s.population.topology)).sum()
        gradients = torch.autograd.grad(loss, (p, r))
        assert all(torch.isfinite(g).all() for g in gradients)
        # The configured cap is a budget, not another refinement to perform.
        s.iteration += 1
        report = dict(s.mesh_report)
        s._refine_beams()
        assert s.mesh_report == report
        if device == "cpu":
            s.tune({"mesh_max_segments": 4})
            s._refine_beams()
            assert s.continuation.beam_segments == 4
            assert s.mesh_report["coarse_segments"] == 2
            assert s.scores.isfinite().all()
            torch.testing.assert_close(s._score(s.population), s.scores)
        s.tune({}, {"beam_segments": 1})
        assert s.mesh_report == {} and s.mesh_level_iteration == s.iteration
    finally:
        s.writer.close()


def test_accurate_coarse_mesh_is_kept_and_failed_fine_solve_does_not_promote(
    tmp_path, monkeypatch
):
    s = search(tmp_path, mesh_position_tolerance=1, mesh_orientation_tolerance=1)
    try:
        s.ensure_evaluated()
        s.iteration = 1
        coarse_scores = s.scores.clone()
        s._refine_beams()
        assert not s.mesh_report["promoted"] and s.continuation.beam_segments == 1
        assert s.mesh_report["position_error"] is not None
        torch.testing.assert_close(s.scores, coarse_scores)
        s.iteration = 2
        monkeypatch.setattr(
            "compliant_mechanism_synthesis.optimization.trace_motion",
            lambda batch, *args, **kwargs: MotionPath(
                [],
                torch.zeros(
                    batch.size, device=batch.positions.device, dtype=torch.bool
                ),
                0,
            ),
        )
        s._refine_beams()
        assert not s.mesh_report["promoted"] and s.mesh_report["fine_failures"] == 1
        assert s.continuation.beam_segments == 1
        torch.testing.assert_close(s.scores, coarse_scores)
    finally:
        s.writer.close()


def test_spatial_comparison_excludes_different_input_poses():
    q = torch.zeros(12, dtype=torch.float64)
    records = [(0.0, q), (1.0, q)]
    states = []
    for f, displacement in [(0.0, 0.0001), (0.5, 100.0), (1.0, 0.0001)]:
        current = q.clone()
        current[0] = displacement
        current[3] = 0.001
        states.append(
            SimpleNamespace(
                accepted=torch.tensor([True]),
                factors=torch.tensor([f]),
                q=current[None],
            )
        )
    position, orientation, samples = mesh_pose_error(records, states, 0)
    assert position == pytest.approx(0.0001)
    assert orientation == pytest.approx(0.001) and samples == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"mesh_max_segments": 0},
        {"mesh_check_interval": 0},
        {"mesh_min_iterations": 5, "mesh_max_iterations": 4},
        {"mesh_position_tolerance": 0},
        {"adaptive_beam_resolution": 2},
    ],
)
def test_invalid_spatial_settings(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)


def test_expensive_checks_wait_for_quality_or_budget_and_respect_manual_mode(
    tmp_path, monkeypatch
):
    s = search(tmp_path)
    try:
        s.ensure_evaluated()
        s.tune(
            {
                "mesh_max_iterations": 20,
                "mesh_motion_loss_target": 1e-12,
                "fidelity_check_candidates": 2,
            }
        )
        # Selection can clone a sole viable survivor; do not solve it twice.
        s.population = s.population.take(
            torch.zeros(2, dtype=torch.long, device=s.device)
        )
        s.scores = s.scores[:1].expand(2).clone()
        calls = []

        def failed_check(batch, *args, **kwargs):
            calls.append(batch.size)
            return MotionPath(
                [],
                torch.zeros(
                    batch.size, dtype=torch.bool, device=batch.positions.device
                ),
                0,
            )

        monkeypatch.setattr(
            "compliant_mechanism_synthesis.optimization.trace_motion", failed_check
        )
        s._refine_beams()  # Minimum dwell time.
        s.iteration = 1
        s._refine_beams()  # Low-quality, nonstagnating candidates.
        assert not calls
        s.iteration = 20
        s._refine_beams()  # Budget prevents indefinite avoidance of validation.
        assert calls == [1] and s.mesh_report["fine_failures"] == 1
        s.tune({"adaptive_beam_resolution": 0})
        s.iteration += 10
        s._refine_beams()
        assert calls == [1]
    finally:
        s.writer.close()
