from __future__ import annotations

from pathlib import Path

import torch

from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
)
from compliant_mechanism_synthesis.dataset.types import Analyses, Structures
from compliant_mechanism_synthesis.models import SupervisedRefinerConfig
from compliant_mechanism_synthesis.training import (
    FlowCurriculumTrainingConfig,
    flow_step_predictions,
    load_training_cases,
    local_flow_step_targets,
    local_flow_targets,
    make_training_batch,
    rollout_step_schedule,
    train_flow_refiner,
)
from compliant_mechanism_synthesis.training.unified import (
    _aggregate_rollout_losses,
    _conditioning_inputs,
    _trajectory_loss_terms,
)


def _build_cases(tmp_path: Path):
    path = tmp_path / "dataset.pt"
    generate_offline_dataset(
        OfflineDatasetConfig(
            num_cases=4,
            device="cpu",
            output_path=str(path),
            logdir=str(tmp_path / "runs"),
            primitive=PrimitiveConfig(num_free_nodes=6),
            optimization=CaseOptimizationConfig(num_steps=3),
        )
    )
    return path, load_training_cases(str(path))


def _fake_analyses_from_structures(structures: Structures) -> Analyses:
    batch_size, num_nodes, _ = structures.positions.shape
    base = structures.positions.sum(dim=(1, 2)) + structures.adjacency.sum(dim=(1, 2))
    generalized_stiffness = base[:, None, None].expand(batch_size, 6, 6)
    nodal_displacements = base[:, None, None].expand(batch_size, num_nodes, 18)
    edge_von_mises = base[:, None, None, None].expand(
        batch_size, num_nodes, num_nodes, 6
    )
    return Analyses(
        generalized_stiffness=generalized_stiffness,
        material_usage=base,
        short_beam_penalty=base,
        long_beam_penalty=base,
        thin_beam_penalty=base,
        thick_beam_penalty=base,
        free_node_spacing_penalty=base,
        nodal_displacements=nodal_displacements,
        edge_von_mises=edge_von_mises,
    )


def test_rollout_step_schedule_reaches_final_time_one() -> None:
    initial_times = torch.tensor([0.0, 0.25, 0.9])
    step_times, step_sizes = rollout_step_schedule(
        initial_times, num_integration_steps=4
    )

    assert len(step_times) == 4
    assert torch.allclose(step_times[0], initial_times)
    assert torch.allclose(step_times[-1] + step_sizes, torch.ones_like(initial_times))


def test_local_flow_targets_use_current_state_and_remaining_time(
    tmp_path: Path,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)

    target_position_velocity, target_adjacency_velocity = local_flow_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        epsilon=1e-4,
    )
    remaining = (1.0 - batch.initial_times).clamp_min(1e-4)[:, None, None]
    assert torch.allclose(
        batch.initial_structures.positions + remaining * target_position_velocity,
        batch.oracle_structures.positions,
        atol=1e-5,
    )
    assert torch.allclose(
        batch.initial_structures.adjacency + remaining * target_adjacency_velocity,
        batch.oracle_structures.adjacency,
        atol=1e-5,
    )


def test_local_flow_step_targets_use_dt_correctly(tmp_path: Path) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    _, step_sizes = rollout_step_schedule(batch.initial_times, num_integration_steps=4)

    target_position_velocity, target_adjacency_velocity = local_flow_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        epsilon=1e-4,
    )
    target_position_step, target_adjacency_step = local_flow_step_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        step_sizes,
        epsilon=1e-4,
    )

    step_scale = step_sizes[:, None, None]
    assert torch.allclose(target_position_step, step_scale * target_position_velocity)
    assert torch.allclose(target_adjacency_step, step_scale * target_adjacency_velocity)


def test_flow_step_predictions_supervise_effective_step_not_raw_velocity() -> None:
    position_velocity = torch.tensor([[[10.0, 0.0, 0.0]]])
    adjacency_velocity = torch.tensor([[[0.0]]])
    step_sizes = torch.tensor([0.1])

    predicted_position_step, predicted_adjacency_step = flow_step_predictions(
        position_velocity=position_velocity,
        adjacency_velocity=adjacency_velocity,
        step_sizes=step_sizes,
    )

    assert torch.allclose(predicted_position_step, torch.tensor([[[1.0, 0.0, 0.0]]]))
    assert torch.allclose(predicted_adjacency_step, torch.tensor([[[0.0]]]))


def test_local_flow_step_targets_are_coherent_for_single_integration_step(
    tmp_path: Path,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    _, step_sizes = rollout_step_schedule(batch.initial_times, num_integration_steps=1)

    target_position_step, target_adjacency_step = local_flow_step_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        step_sizes,
        epsilon=1e-4,
    )

    assert torch.allclose(
        batch.initial_structures.positions + target_position_step,
        batch.oracle_structures.positions,
        atol=1e-5,
    )
    assert torch.allclose(
        batch.initial_structures.adjacency + target_adjacency_step,
        batch.oracle_structures.adjacency,
        atol=1e-5,
    )


def test_aggregate_rollout_losses_uses_step_mean_and_endpoint_physics() -> None:
    total_loss, loss_terms = _aggregate_rollout_losses(
        supervised_step_losses=[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        endpoint_physical_loss=torch.tensor([5.0, 6.0]),
        endpoint_stiffness_loss=torch.tensor([0.2, 0.3]),
        endpoint_stress_loss=torch.tensor([0.3, 0.4]),
        style_kl=torch.tensor([0.1, 0.2]),
        lambda_sup=0.5,
        lambda_phys=2.0,
        lambda_kl=3.0,
    )

    expected = torch.tensor(
        [
            0.5 * ((1.0 + 3.0) / 2.0) + 2.0 * 5.0 + 3.0 * 0.1,
            0.5 * ((2.0 + 4.0) / 2.0) + 2.0 * 6.0 + 3.0 * 0.2,
        ]
    ).mean()
    assert torch.allclose(total_loss, expected)
    assert torch.allclose(loss_terms["supervised_loss"], torch.tensor(2.5))
    assert torch.allclose(loss_terms["physical_loss"], torch.tensor(5.5))


def test_conditioning_inputs_use_no_grad_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    grad_enabled_states: list[bool] = []

    def fake_analyze_structures(current: Structures, profile=None) -> Analyses:
        del profile
        grad_enabled_states.append(torch.is_grad_enabled())
        return _fake_analyses_from_structures(current)

    monkeypatch.setattr(
        "compliant_mechanism_synthesis.training.unified.analyze_structures",
        fake_analyze_structures,
    )

    current_stiffness, nodal_displacements, edge_von_mises = _conditioning_inputs(
        current=batch.initial_structures,
        use_analysis=True,
    )

    assert grad_enabled_states == [False]
    assert not current_stiffness.requires_grad
    assert not nodal_displacements.requires_grad
    assert not edge_von_mises.requires_grad


def test_conditioning_inputs_skip_analysis_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)

    def fake_analyze_structures(current: Structures, profile=None) -> Analyses:
        del profile
        raise AssertionError("analyze_structures should not be called")

    monkeypatch.setattr(
        "compliant_mechanism_synthesis.training.unified.analyze_structures",
        fake_analyze_structures,
    )

    current_stiffness, nodal_displacements, edge_von_mises = _conditioning_inputs(
        current=batch.initial_structures,
        use_analysis=False,
    )

    assert torch.allclose(current_stiffness, torch.zeros_like(current_stiffness))
    assert torch.allclose(nodal_displacements, torch.zeros_like(nodal_displacements))
    assert torch.allclose(edge_von_mises, torch.zeros_like(edge_von_mises))


def test_trajectory_loss_terms_uses_endpoint_only_physics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    analysis_calls: list[bool] = []
    structural_calls = 0

    class FakeModel:
        def __init__(self) -> None:
            self.config = SupervisedRefinerConfig(use_style_conditioning=False)

        def predict_flow(
            self,
            *,
            structures: Structures,
            target_stiffness: torch.Tensor,
            current_stiffness: torch.Tensor,
            nodal_displacements: torch.Tensor,
            edge_von_mises: torch.Tensor,
            flow_times: torch.Tensor,
            style_structures,
            style_analyses,
            style_token_mask,
        ):
            del (
                target_stiffness,
                flow_times,
                style_structures,
                style_analyses,
                style_token_mask,
            )
            assert current_stiffness.shape == (structures.batch_size, 6, 6)
            assert nodal_displacements.shape == (
                structures.batch_size,
                structures.num_nodes,
                18,
            )
            assert edge_von_mises.shape == (
                structures.batch_size,
                structures.num_nodes,
                structures.num_nodes,
                6,
            )
            return type(
                "Prediction",
                (),
                {
                    "position_velocity": torch.zeros_like(structures.positions),
                    "adjacency_velocity": torch.zeros_like(structures.adjacency),
                    "style_kl": None,
                },
            )()

    def fake_analyze_structures(current: Structures, profile=None) -> Analyses:
        del profile
        analysis_calls.append(torch.is_grad_enabled())
        return _fake_analyses_from_structures(current)

    def fake_structural_objective_terms(**kwargs):
        nonlocal structural_calls
        structural_calls += 1
        analyses = kwargs["analyses"]
        batch_size = analyses.material_usage.shape[0]
        zeros = torch.zeros(batch_size, dtype=analyses.material_usage.dtype)
        ones = torch.ones(batch_size, dtype=analyses.material_usage.dtype)
        return (
            {
                "stiffness_loss_contribution": ones,
                "stress_loss_contribution": 2.0 * ones,
            },
            {
                "stiffness_error": 3.0 * ones,
                "stress_violation": 4.0 * ones,
                "mean_stress_ratio": 5.0 * ones,
                "max_stress_ratio": 6.0 * ones,
            },
        )

    monkeypatch.setattr(
        "compliant_mechanism_synthesis.training.unified.analyze_structures",
        fake_analyze_structures,
    )
    monkeypatch.setattr(
        "compliant_mechanism_synthesis.training.unified.structural_objective_terms",
        fake_structural_objective_terms,
    )

    total_loss, metrics, loss_terms = _trajectory_loss_terms(
        model=FakeModel(),
        batch=batch,
        config=FlowCurriculumTrainingConfig(
            dataset_path="dataset.pt",
            num_integration_steps=3,
            physical_weight_start=1.0,
            physical_weight_end=1.0,
            physical_transition_start_step=0,
            physical_transition_end_step=0,
        ),
        step=0,
    )

    assert len(analysis_calls) == 4
    assert analysis_calls[:-1] == [False, False, False]
    assert analysis_calls[-1] is True
    assert structural_calls == 1
    assert torch.isfinite(total_loss)
    assert torch.allclose(loss_terms["physical_loss"], torch.tensor(3.0))
    assert torch.allclose(loss_terms["stiffness_loss_contribution"], torch.tensor(1.0))
    assert torch.allclose(loss_terms["stress_loss_contribution"], torch.tensor(2.0))
    assert torch.allclose(metrics["stiffness_error"], torch.tensor(3.0))


def test_trajectory_loss_terms_skips_all_analysis_when_physics_is_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)

    class FakeModel:
        def __init__(self) -> None:
            self.config = SupervisedRefinerConfig(use_style_conditioning=False)

        def predict_flow(
            self,
            *,
            structures: Structures,
            target_stiffness: torch.Tensor,
            current_stiffness: torch.Tensor,
            nodal_displacements: torch.Tensor,
            edge_von_mises: torch.Tensor,
            flow_times: torch.Tensor,
            style_structures,
            style_analyses,
            style_token_mask,
        ):
            del (
                target_stiffness,
                flow_times,
                style_structures,
                style_analyses,
                style_token_mask,
            )
            assert torch.allclose(
                current_stiffness, torch.zeros_like(current_stiffness)
            )
            assert torch.allclose(
                nodal_displacements, torch.zeros_like(nodal_displacements)
            )
            assert torch.allclose(edge_von_mises, torch.zeros_like(edge_von_mises))
            return type(
                "Prediction",
                (),
                {
                    "position_velocity": torch.zeros_like(structures.positions),
                    "adjacency_velocity": torch.zeros_like(structures.adjacency),
                    "style_kl": None,
                },
            )()

    def fake_analyze_structures(current: Structures, profile=None) -> Analyses:
        del current, profile
        raise AssertionError(
            "analyze_structures should not be called when lambda_phys=0"
        )

    def fake_structural_objective_terms(**kwargs):
        raise AssertionError(
            "structural_objective_terms should not be called when lambda_phys=0"
        )

    monkeypatch.setattr(
        "compliant_mechanism_synthesis.training.unified.analyze_structures",
        fake_analyze_structures,
    )
    monkeypatch.setattr(
        "compliant_mechanism_synthesis.training.unified.structural_objective_terms",
        fake_structural_objective_terms,
    )

    total_loss, metrics, loss_terms = _trajectory_loss_terms(
        model=FakeModel(),
        batch=batch,
        config=FlowCurriculumTrainingConfig(
            dataset_path="dataset.pt",
            num_integration_steps=3,
            physical_weight_start=0.0,
            physical_weight_end=0.0,
            physical_transition_start_step=0,
            physical_transition_end_step=0,
        ),
        step=0,
    )

    assert torch.isfinite(total_loss)
    assert torch.allclose(loss_terms["physical_loss"], torch.tensor(0.0))
    assert torch.allclose(loss_terms["physical_loss_contribution"], torch.tensor(0.0))
    assert torch.allclose(metrics["stiffness_error"], torch.tensor(0.0))


def test_train_flow_refiner_writes_checkpoint_and_history(tmp_path: Path) -> None:
    dataset_path, cases = _build_cases(tmp_path)

    _, summary = train_flow_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=FlowCurriculumTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=3,
            eval_fraction=0.0,
            checkpoint_path=str(tmp_path / "refiner.pt"),
            logdir=str(tmp_path / "runs_train"),
            seed=5,
        ),
    )

    checkpoint = torch.load(summary.checkpoint_path, map_location="cpu")
    assert summary.checkpoint_path.exists()
    assert "total_loss" in summary.history
    assert "supervised_loss_contribution" in summary.history
    assert "physical_loss_contribution" in summary.history
    assert checkpoint["train_config"]["num_integration_steps"] == 3
    assert "style_sample_dropout" not in checkpoint["train_config"]


def test_train_flow_refiner_supports_warm_start_from_checkpoint(tmp_path: Path) -> None:
    dataset_path, cases = _build_cases(tmp_path)
    initial_checkpoint = tmp_path / "initial.pt"
    _, initial_summary = train_flow_refiner(
        optimized_cases=cases,
        model_config=SupervisedRefinerConfig(
            hidden_dim=64,
            connectivity_latent_dim=32,
            num_attention_layers=3,
            num_heads=16,
        ),
        train_config=FlowCurriculumTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=2,
            eval_fraction=0.0,
            checkpoint_path=str(initial_checkpoint),
            logdir=str(tmp_path / "runs_initial"),
            seed=7,
        ),
    )

    _, summary = train_flow_refiner(
        optimized_cases=cases,
        train_config=FlowCurriculumTrainingConfig(
            dataset_path=str(dataset_path),
            device="cpu",
            batch_size=2,
            num_steps=2,
            eval_fraction=0.0,
            init_checkpoint_path=str(initial_summary.checkpoint_path),
            checkpoint_path=str(tmp_path / "warm_started.pt"),
            logdir=str(tmp_path / "runs_warm_started"),
            seed=11,
        ),
    )

    checkpoint = torch.load(summary.checkpoint_path, map_location="cpu")
    assert checkpoint["train_config"]["init_checkpoint_path"] == str(
        initial_summary.checkpoint_path
    )


def test_flow_curriculum_config_has_no_sample_style_dropout() -> None:
    config = FlowCurriculumTrainingConfig(dataset_path="dataset.pt")
    assert not hasattr(config, "style_sample_dropout")
