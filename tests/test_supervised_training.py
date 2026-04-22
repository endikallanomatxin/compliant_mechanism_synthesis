from __future__ import annotations

from pathlib import Path

import pytest
import torch
import compliant_mechanism_synthesis.training.supervised as supervised_training_module

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
)
from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.training import (
    SupervisedTrainingConfig,
    analyze_structures,
    make_supervised_batch,
    load_supervised_cases,
    train_supervised_refiner,
)
from compliant_mechanism_synthesis.training.supervised import (
    _scheduled_learning_rate,
    _stiffness_loss_weight_effective,
    _stress_loss_weight_effective,
    _style_kl_weight,
    _training_losses,
    split_train_eval_cases,
)


def _build_cases(tmp_path: Path):
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=4,
            device="cpu",
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(
                num_free_nodes=6,
                sheet_width_nodes=2,
                sample_sheet_helix_width_nodes=False,
                sheet_helix_width_nodes_min=2,
                sheet_helix_width_nodes_max=2,
                sheet_helix_offset_distance_min=0.05,
                sheet_helix_offset_distance_max=0.07,
                sheet_helix_pitch_distance=0.30,
                sheet_helix_max_longitudinal_points=32,
            ),
            optimization=CaseOptimizationConfig(num_steps=4),
        )
    )
    return load_supervised_cases(str(path))


def test_supervised_refiner_preserves_structure_shapes(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    prediction = model(
        cases.optimized_structures,
        cases.target_stiffness,
        analysis_fn=analyze_structures,
        num_steps=2,
        style_structures=cases.optimized_structures,
        style_analyses=cases.last_analyses,
    )

    assert prediction.positions.shape == cases.optimized_structures.positions.shape
    assert prediction.adjacency.shape == cases.optimized_structures.adjacency.shape


def test_supervised_refiner_forward_supports_missing_style(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    prediction = model(
        cases.optimized_structures,
        cases.target_stiffness,
        analysis_fn=analyze_structures,
        num_steps=2,
    )

    assert prediction.positions.shape == cases.optimized_structures.positions.shape
    assert prediction.adjacency.shape == cases.optimized_structures.adjacency.shape


def test_scheduled_learning_rate_warms_up_then_cosine_decays() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        num_steps=10,
        learning_rate=1e-3,
        warmup_steps=2,
        min_learning_rate=1e-4,
    )

    learning_rates = [_scheduled_learning_rate(step, config) for step in range(10)]

    assert learning_rates[0] < learning_rates[1]
    assert learning_rates[1] == config.learning_rate
    assert learning_rates[-1] == config.min_learning_rate
    assert learning_rates[2] > learning_rates[5] > learning_rates[8]


def test_style_kl_weight_anneals_to_target() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        style_kl_loss_weight=2e-3,
        style_kl_anneal_steps=4,
    )

    weights = [_style_kl_weight(step, config) for step in range(6)]

    assert weights[0] == pytest.approx(5e-4)
    assert weights[1] == pytest.approx(1e-3)
    assert weights[2] == pytest.approx(1.5e-3)
    assert weights[3] == pytest.approx(2e-3)
    assert weights[4] == pytest.approx(2e-3)


def test_stiffness_loss_weight_effective_is_zero_before_delay() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stiffness_loss_weight=0.05,
        stiffness_loss_delay_steps=3,
        stiffness_loss_warmup_steps=4,
    )

    assert _stiffness_loss_weight_effective(0, config) == pytest.approx(0.0)
    assert _stiffness_loss_weight_effective(1, config) == pytest.approx(0.0)
    assert _stiffness_loss_weight_effective(2, config) == pytest.approx(0.0)


def test_stiffness_loss_weight_effective_warms_up_linearly() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stiffness_loss_weight=0.05,
        stiffness_loss_delay_steps=2,
        stiffness_loss_warmup_steps=4,
    )

    assert _stiffness_loss_weight_effective(3, config) == pytest.approx(0.025)


def test_stiffness_loss_weight_effective_reaches_plateau() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stiffness_loss_weight=0.05,
        stiffness_loss_delay_steps=2,
        stiffness_loss_warmup_steps=4,
    )

    assert _stiffness_loss_weight_effective(6, config) == pytest.approx(0.05)
    assert _stiffness_loss_weight_effective(20, config) == pytest.approx(0.05)


def test_stiffness_loss_weight_effective_jumps_after_delay_when_no_warmup() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stiffness_loss_weight=0.05,
        stiffness_loss_delay_steps=2,
        stiffness_loss_warmup_steps=0,
    )

    assert _stiffness_loss_weight_effective(1, config) == pytest.approx(0.0)
    assert _stiffness_loss_weight_effective(2, config) == pytest.approx(0.05)


def test_stress_loss_weight_effective_is_zero_before_delay() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stress_loss_weight=0.01,
        stress_loss_delay_steps=3,
        stress_loss_warmup_steps=4,
    )

    assert _stress_loss_weight_effective(0, config) == pytest.approx(0.0)
    assert _stress_loss_weight_effective(1, config) == pytest.approx(0.0)
    assert _stress_loss_weight_effective(2, config) == pytest.approx(0.0)


def test_stress_loss_weight_effective_warms_up_linearly() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stress_loss_weight=0.01,
        stress_loss_delay_steps=2,
        stress_loss_warmup_steps=4,
    )

    assert _stress_loss_weight_effective(3, config) == pytest.approx(0.005)


def test_stress_loss_weight_effective_reaches_plateau() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stress_loss_weight=0.01,
        stress_loss_delay_steps=2,
        stress_loss_warmup_steps=4,
    )

    assert _stress_loss_weight_effective(6, config) == pytest.approx(0.01)
    assert _stress_loss_weight_effective(20, config) == pytest.approx(0.01)


def test_stress_loss_weight_effective_jumps_after_delay_when_no_warmup() -> None:
    config = SupervisedTrainingConfig(
        dataset_path="dataset.pt",
        stress_loss_weight=0.01,
        stress_loss_delay_steps=2,
        stress_loss_warmup_steps=0,
    )

    assert _stress_loss_weight_effective(1, config) == pytest.approx(0.0)
    assert _stress_loss_weight_effective(2, config) == pytest.approx(0.01)


def test_split_train_eval_cases_uses_deterministic_tail_subset(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)

    split = split_train_eval_cases(cases, eval_fraction=0.26)

    assert split.eval_cases is not None
    assert split.train_cases.optimized_structures.batch_size == 2
    assert split.eval_cases.optimized_structures.batch_size == 2
    assert torch.equal(
        split.train_cases.target_stiffness,
        cases.target_stiffness[:2],
    )
    assert torch.equal(
        split.eval_cases.target_stiffness,
        cases.target_stiffness[2:],
    )


def test_flow_times_cover_unit_interval_without_curriculum(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    batch = make_supervised_batch(optimized_cases=cases, seed=3)

    assert float(batch.flow_times.min().item()) >= 0.0
    assert float(batch.flow_times.max().item()) <= 1.0


def test_train_supervised_refiner_writes_checkpoint_and_reduces_training_loss(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    _, summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=16,
            eval_every_steps=4,
            eval_fraction=0.25,
            learning_rate=5e-4,
            checkpoint_path=str(tmp_path / "refiner.pt"),
            logdir=str(tmp_path / "runs"),
            seed=3,
        ),
    )

    assert summary.checkpoint_path.exists()
    assert "style_kl_loss_contribution" in summary.history
    assert "eval_no_style_total_loss" in summary.history
    assert "eval_with_style_total_loss" in summary.history
    first_window = sum(summary.history["total_loss"][:4]) / 4.0
    best_observed = min(summary.history["total_loss"])
    assert best_observed <= first_window


def test_train_supervised_refiner_supports_style_dropouts(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    _, summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=4,
            eval_every_steps=2,
            eval_fraction=0.25,
            style_sample_dropout=1.0,
            style_token_dropout=0.0,
            checkpoint_path=str(tmp_path / "refiner_dropout.pt"),
            logdir=str(tmp_path / "runs_dropout"),
            seed=31,
        ),
    )

    assert summary.checkpoint_path.exists()
    assert all(
        value == pytest.approx(0.0)
        for value in summary.history["style_kl_loss_contribution"]
    )


def test_training_losses_skip_endpoint_analysis_when_stiffness_weight_effective_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(optimized_cases=cases, seed=33)
    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
    )

    def _fail_analyze(*args, **kwargs):
        raise AssertionError("endpoint stiffness analysis should not run")

    monkeypatch.setattr(supervised_training_module, "analyze_structures", _fail_analyze)
    total_loss, metrics, loss_terms = _training_losses(
        prediction,
        batch,
        SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            stiffness_loss_weight=0.05,
            stiffness_loss_delay_steps=10,
            stiffness_loss_warmup_steps=5,
        ),
        step=0,
    )

    assert torch.isfinite(total_loss)
    assert float(metrics["stiffness_loss_weight_effective"].item()) == pytest.approx(
        0.0
    )
    assert float(
        loss_terms["stiffness_error_loss_contribution"].item()
    ) == pytest.approx(0.0)


def test_training_losses_skip_endpoint_analysis_when_stiffness_and_stress_weights_effective_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(optimized_cases=cases, seed=34)
    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
    )

    def _fail_analyze(*args, **kwargs):
        raise AssertionError("endpoint analysis should not run")

    monkeypatch.setattr(supervised_training_module, "analyze_structures", _fail_analyze)
    total_loss, metrics, loss_terms = _training_losses(
        prediction,
        batch,
        SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            stiffness_loss_weight=0.05,
            stiffness_loss_delay_steps=10,
            stiffness_loss_warmup_steps=5,
            stress_loss_weight=0.01,
            stress_loss_delay_steps=10,
            stress_loss_warmup_steps=5,
        ),
        step=0,
    )

    assert torch.isfinite(total_loss)
    assert float(metrics["stiffness_loss_weight_effective"].item()) == pytest.approx(
        0.0
    )
    assert float(metrics["stress_loss_weight_effective"].item()) == pytest.approx(0.0)
    assert float(
        loss_terms["stiffness_error_loss_contribution"].item()
    ) == pytest.approx(0.0)
    assert float(loss_terms["stress_loss_contribution"].item()) == pytest.approx(0.0)


def test_training_losses_reuse_single_endpoint_analysis_for_stiffness_and_stress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(optimized_cases=cases, seed=35)
    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
    )

    analyze_calls = 0
    original_analyze = supervised_training_module.analyze_structures

    def _counting_analyze(*args, **kwargs):
        nonlocal analyze_calls
        analyze_calls += 1
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(
        supervised_training_module, "analyze_structures", _counting_analyze
    )
    total_loss, metrics, loss_terms = _training_losses(
        prediction,
        batch,
        SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            stiffness_loss_weight=0.05,
            stiffness_loss_delay_steps=0,
            stiffness_loss_warmup_steps=0,
            stress_loss_weight=0.01,
            stress_loss_delay_steps=0,
            stress_loss_warmup_steps=0,
        ),
        step=0,
    )

    assert torch.isfinite(total_loss)
    assert analyze_calls == 1
    assert float(metrics["stress_loss_weight_effective"].item()) == pytest.approx(0.01)
    assert "stress_loss_contribution" in loss_terms
    assert "stress_violation" in metrics
    assert "mean_stress_ratio" in metrics
    assert "max_stress_ratio" in metrics


def test_train_supervised_refiner_supports_stiffness_delay_and_warmup(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    _, summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=5,
            eval_every_steps=5,
            eval_fraction=0.25,
            stiffness_loss_weight=0.1,
            stiffness_loss_delay_steps=2,
            stiffness_loss_warmup_steps=2,
            checkpoint_path=str(tmp_path / "refiner_stiffness_schedule.pt"),
            logdir=str(tmp_path / "runs_stiffness_schedule"),
            seed=37,
        ),
    )

    assert summary.checkpoint_path.exists()
    assert summary.history["stiffness_error_loss_contribution"][0] == pytest.approx(0.0)
    assert summary.history["stiffness_error_loss_contribution"][1] == pytest.approx(0.0)
    assert any(
        value > 0.0
        for value in summary.history["stiffness_error_loss_contribution"][2:]
    )


def test_train_supervised_refiner_supports_stress_delay_and_warmup(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    _, summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=5,
            eval_every_steps=5,
            eval_fraction=0.25,
            stress_loss_weight=0.02,
            stress_loss_delay_steps=2,
            stress_loss_warmup_steps=2,
            checkpoint_path=str(tmp_path / "refiner_stress_schedule.pt"),
            logdir=str(tmp_path / "runs_stress_schedule"),
            seed=39,
        ),
    )

    assert summary.checkpoint_path.exists()
    assert "stress_loss_contribution" in summary.history
    assert summary.history["stress_loss_contribution"][0] == pytest.approx(0.0)
    assert summary.history["stress_loss_contribution"][1] == pytest.approx(0.0)
    assert any(value > 0.0 for value in summary.history["stress_loss_contribution"][2:])


def test_trained_refiner_beats_untrained_baseline_on_seen_batch(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    config = SupervisedRefinerConfig(
        hidden_dim=64,
        connectivity_latent_dim=32,
        num_attention_layers=3,
        num_heads=16,
    )
    _, summary = train_supervised_refiner(
        optimized_cases=cases,
        model_config=config,
        train_config=SupervisedTrainingConfig(
            dataset_path=str(tmp_path / "dataset.pt"),
            device="cpu",
            batch_size=2,
            num_steps=40,
            learning_rate=5e-4,
            checkpoint_path=str(tmp_path / "refiner.pt"),
            logdir=str(tmp_path / "runs"),
            seed=5,
        ),
    )
    loss_history = summary.history["total_loss"]

    first_window = sum(loss_history[:4]) / 4.0
    best_observed = min(loss_history)
    assert best_observed <= first_window


def test_predict_flow_rejects_mismatched_style_roles(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=11,
    )
    mismatched_roles = batch.oracle_structures.roles.clone()
    mismatched_roles[:, 0] = mismatched_roles[:, 1]

    with pytest.raises(ValueError, match="same node roles"):
        model.predict_flow(
            structures=batch.flow_structures,
            target_stiffness=batch.target_stiffness,
            current_stiffness=batch.current_analyses.generalized_stiffness,
            nodal_displacements=batch.current_analyses.nodal_displacements,
            edge_von_mises=batch.current_analyses.edge_von_mises,
            flow_times=batch.flow_times,
            style_structures=type(batch.oracle_structures)(
                positions=batch.oracle_structures.positions,
                roles=mismatched_roles,
                adjacency=batch.oracle_structures.adjacency,
            ),
            style_analyses=batch.oracle_analyses,
        )


def test_predict_flow_ignores_style_inputs_when_style_conditioning_disabled(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
            use_style_conditioning=False,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=13,
    )
    mismatched_roles = batch.oracle_structures.roles.clone()
    mismatched_roles[:, 0] = mismatched_roles[:, 1]

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        style_structures=type(batch.oracle_structures)(
            positions=batch.oracle_structures.positions,
            roles=mismatched_roles,
            adjacency=batch.oracle_structures.adjacency,
        ),
    )

    assert prediction.position_velocity.shape == batch.target_position_velocity.shape
    assert prediction.adjacency_velocity.shape == batch.target_adjacency_velocity.shape
    assert prediction.style_latent is None
    assert prediction.style_residual is None
    assert prediction.style_token_mask is None
    assert prediction.style_mean is None
    assert prediction.style_logvar is None
    assert prediction.style_kl is None


def test_predict_flow_without_style_uses_base_context_and_zero_kl(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=14,
    )

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
    )

    assert prediction.style_latent is not None
    assert prediction.style_residual is not None
    assert prediction.style_token_mask is not None
    assert prediction.style_mean is not None
    assert prediction.style_logvar is not None
    assert prediction.style_kl is not None
    assert prediction.style_latent.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
        model.config.style_local_latent_dim,
    )
    assert prediction.style_mean.shape == prediction.style_latent.shape
    assert prediction.style_logvar.shape == prediction.style_latent.shape
    assert prediction.style_residual.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
        64,
    )
    assert prediction.style_token_mask.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
    )
    assert not prediction.style_token_mask.any()
    assert torch.allclose(
        prediction.style_latent,
        torch.zeros_like(prediction.style_latent),
    )
    assert torch.allclose(
        prediction.style_residual,
        torch.zeros_like(prediction.style_residual),
    )
    assert torch.allclose(prediction.style_kl, torch.zeros_like(prediction.style_kl))
    assert torch.isfinite(prediction.style_residual).all()


def test_predict_flow_returns_variational_style_statistics(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=15,
    )

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )

    assert prediction.style_mean is not None
    assert prediction.style_logvar is not None
    assert prediction.style_kl is not None
    assert prediction.style_latent is not None
    assert prediction.style_residual is not None
    assert prediction.style_token_mask is not None
    assert prediction.style_mean.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
        model.config.style_local_latent_dim,
    )
    assert prediction.style_logvar.shape == prediction.style_mean.shape
    assert prediction.style_kl.shape == (cases.optimized_structures.batch_size,)
    assert prediction.style_latent.shape == prediction.style_mean.shape
    assert prediction.style_residual.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
        64,
    )
    assert prediction.style_token_mask.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
    )
    assert prediction.style_token_mask.all()
    assert torch.isfinite(prediction.style_mean).all()
    assert torch.isfinite(prediction.style_logvar).all()
    assert torch.isfinite(prediction.style_kl).all()
    assert (prediction.style_kl >= 0.0).all()


def test_predict_flow_supports_mixed_style_dropout_within_batch(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=16,
    )
    style_token_mask = torch.zeros(
        batch.flow_structures.batch_size,
        batch.flow_structures.num_nodes,
        dtype=torch.bool,
    )
    style_token_mask[0, 0] = True

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
        style_token_mask=style_token_mask,
    )

    assert prediction.style_token_mask is not None
    assert torch.equal(
        prediction.style_token_mask,
        style_token_mask,
    )
    assert prediction.style_latent is not None
    assert prediction.style_residual is not None
    assert prediction.style_kl is not None
    assert torch.allclose(
        prediction.style_residual[1],
        torch.zeros_like(prediction.style_residual[1]),
    )
    assert float(prediction.style_kl[1].detach().item()) == pytest.approx(0.0)
    assert torch.isfinite(prediction.style_latent[0, 0]).all()
    assert torch.allclose(
        prediction.style_latent[0, 1:],
        torch.zeros_like(prediction.style_latent[0, 1:]),
    )
    assert prediction.style_kl[0] >= 0.0


def test_predict_flow_zero_token_mask_matches_no_style(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=18,
    )
    style_token_mask = torch.zeros(
        batch.flow_structures.batch_size,
        batch.flow_structures.num_nodes,
        dtype=torch.bool,
    )

    base_prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
    )
    masked_prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
        style_token_mask=style_token_mask,
    )

    assert base_prediction.style_latent is not None
    assert masked_prediction.style_latent is not None
    assert torch.allclose(masked_prediction.style_latent, base_prediction.style_latent)
    assert masked_prediction.style_residual is not None
    assert torch.allclose(
        masked_prediction.style_residual,
        torch.zeros_like(masked_prediction.style_residual),
    )
    assert masked_prediction.style_kl is not None
    assert torch.allclose(
        masked_prediction.style_kl,
        torch.zeros_like(masked_prediction.style_kl),
    )


def test_predict_flow_uses_configured_style_local_latent_dim(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
            style_local_latent_dim=8,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=20,
    )

    styled_prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )

    assert styled_prediction.style_latent is not None
    assert styled_prediction.style_residual is not None
    assert styled_prediction.style_latent.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
        8,
    )
    assert styled_prediction.style_mean is not None
    assert styled_prediction.style_mean.shape == styled_prediction.style_latent.shape
    assert styled_prediction.style_residual.shape == (
        cases.optimized_structures.batch_size,
        cases.optimized_structures.num_nodes,
        64,
    )


def test_predict_flow_stabilizes_large_mechanics_inputs(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=17,
    )
    huge_displacements = torch.full_like(
        batch.current_analyses.nodal_displacements, 1e25
    )
    huge_stresses = torch.full_like(batch.current_analyses.edge_von_mises, 1e30)

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=huge_displacements,
        edge_von_mises=huge_stresses,
        flow_times=batch.flow_times,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )

    assert torch.isfinite(prediction.position_velocity).all()
    assert torch.isfinite(prediction.adjacency_velocity).all()


def test_predict_flow_produces_symmetric_bounded_adjacency(tmp_path: Path) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=23,
    )

    prediction = model.predict_flow(
        structures=batch.flow_structures,
        target_stiffness=batch.target_stiffness,
        current_stiffness=batch.current_analyses.generalized_stiffness,
        nodal_displacements=batch.current_analyses.nodal_displacements,
        edge_von_mises=batch.current_analyses.edge_von_mises,
        flow_times=batch.flow_times,
        style_structures=batch.oracle_structures,
        style_analyses=batch.oracle_analyses,
    )

    assert torch.allclose(
        prediction.predicted_adjacency,
        prediction.predicted_adjacency.transpose(1, 2),
    )
    assert torch.all(prediction.predicted_adjacency >= 0.0)
    assert torch.all(prediction.predicted_adjacency <= 1.0)


def test_predict_flow_requires_style_analyses_with_style_structures(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=19,
    )

    with pytest.raises(ValueError, match="style_analyses"):
        model.predict_flow(
            structures=batch.flow_structures,
            target_stiffness=batch.target_stiffness,
            current_stiffness=batch.current_analyses.generalized_stiffness,
            nodal_displacements=batch.current_analyses.nodal_displacements,
            edge_von_mises=batch.current_analyses.edge_von_mises,
            flow_times=batch.flow_times,
            style_structures=batch.oracle_structures,
        )


def test_predict_flow_requires_style_structures_with_style_analyses(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=21,
    )

    with pytest.raises(ValueError, match="style_structures"):
        model.predict_flow(
            structures=batch.flow_structures,
            target_stiffness=batch.target_stiffness,
            current_stiffness=batch.current_analyses.generalized_stiffness,
            nodal_displacements=batch.current_analyses.nodal_displacements,
            edge_von_mises=batch.current_analyses.edge_von_mises,
            flow_times=batch.flow_times,
            style_analyses=batch.oracle_analyses,
        )


def test_predict_flow_requires_style_inputs_when_mask_enables_style(
    tmp_path: Path,
) -> None:
    cases = _build_cases(tmp_path)
    model = SupervisedRefiner(
        SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        )
    )
    batch = make_supervised_batch(
        optimized_cases=cases,
        seed=25,
    )

    with pytest.raises(ValueError, match="style_structures"):
        model.predict_flow(
            structures=batch.flow_structures,
            target_stiffness=batch.target_stiffness,
            current_stiffness=batch.current_analyses.generalized_stiffness,
            nodal_displacements=batch.current_analyses.nodal_displacements,
            edge_von_mises=batch.current_analyses.edge_von_mises,
            flow_times=batch.flow_times,
            style_token_mask=torch.ones(
                batch.flow_structures.batch_size,
                batch.flow_structures.num_nodes,
            ),
        )
