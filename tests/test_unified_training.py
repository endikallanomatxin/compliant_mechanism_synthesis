from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from compliant_mechanism_synthesis.adjacency import (
    logits_from_adjacency,
    logits_from_unit_interval,
)
from compliant_mechanism_synthesis.dataset import (
    CaseOptimizationConfig,
    OfflineDatasetConfig,
    PrimitiveConfig,
    generate_offline_dataset,
)
from compliant_mechanism_synthesis.dataset.types import Analyses, Structures
from compliant_mechanism_synthesis.models import (
    SupervisedRefiner,
    SupervisedRefinerConfig,
)
from compliant_mechanism_synthesis.models.refiner import (
    _normalized_edge_von_mises_features,
)
from compliant_mechanism_synthesis.roles import role_masks
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
    _edge_radius_step_loss,
    _clip_grad_norm_per_sample,
    _conditioning_inputs,
    _euler_step,
    _adjacency_step_loss,
    _objective_clipped_gradients,
    _optimizer_parameter_groups,
    _position_step_loss,
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

    (
        target_position_velocity,
        target_adjacency_logit_velocity,
        target_edge_radius_logit_velocity,
    ) = local_flow_targets(
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
    current_logits = logits_from_adjacency(batch.initial_structures.adjacency)
    oracle_logits = logits_from_adjacency(batch.oracle_structures.adjacency)
    assert torch.allclose(
        current_logits + remaining * target_adjacency_logit_velocity,
        oracle_logits,
        atol=1e-5,
    )
    current_edge_radius_logits = logits_from_unit_interval(
        batch.initial_structures.edge_radius
    )
    oracle_edge_radius_logits = logits_from_unit_interval(
        batch.oracle_structures.edge_radius
    )
    assert torch.allclose(
        current_edge_radius_logits + remaining * target_edge_radius_logit_velocity,
        oracle_edge_radius_logits,
        atol=1e-5,
    )


def test_local_flow_step_targets_use_dt_correctly(tmp_path: Path) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    _, step_sizes = rollout_step_schedule(batch.initial_times, num_integration_steps=4)

    (
        target_position_velocity,
        target_adjacency_logit_velocity,
        target_edge_radius_logit_velocity,
    ) = local_flow_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        epsilon=1e-4,
    )
    (
        target_position_step,
        target_adjacency_step,
        target_edge_radius_step,
    ) = local_flow_step_targets(
        batch.initial_structures,
        batch.oracle_structures,
        batch.initial_times,
        step_sizes,
        epsilon=1e-4,
    )

    step_scale = step_sizes[:, None, None]
    assert torch.allclose(target_position_step, step_scale * target_position_velocity)
    assert torch.allclose(
        target_adjacency_step,
        step_scale * target_adjacency_logit_velocity,
    )
    assert torch.allclose(
        target_edge_radius_step,
        step_scale * target_edge_radius_logit_velocity,
    )


def test_flow_step_predictions_return_model_steps_directly() -> None:
    position_step = torch.tensor([[[0.05, 0.0, 0.0]]])
    adjacency_logit_step = torch.tensor([[[0.0]]])
    edge_radius_logit_step = torch.tensor([[[0.2]]])

    (
        predicted_position_step,
        predicted_adjacency_step,
        predicted_edge_radius_step,
    ) = flow_step_predictions(
        position_step=position_step,
        adjacency_logit_step=adjacency_logit_step,
        edge_radius_logit_step=edge_radius_logit_step,
    )

    assert torch.allclose(predicted_position_step, position_step)
    assert torch.allclose(predicted_adjacency_step, torch.tensor([[[0.0]]]))
    assert torch.allclose(predicted_edge_radius_step, edge_radius_logit_step)


def test_local_flow_step_targets_are_coherent_for_single_integration_step(
    tmp_path: Path,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    _, step_sizes = rollout_step_schedule(batch.initial_times, num_integration_steps=1)

    (
        target_position_step,
        target_adjacency_step,
        target_edge_radius_step,
    ) = local_flow_step_targets(
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
    current_logits = logits_from_adjacency(batch.initial_structures.adjacency)
    oracle_logits = logits_from_adjacency(batch.oracle_structures.adjacency)
    assert torch.allclose(
        current_logits + target_adjacency_step,
        oracle_logits,
        atol=1e-5,
    )
    current_edge_radius_logits = logits_from_unit_interval(
        batch.initial_structures.edge_radius
    )
    oracle_edge_radius_logits = logits_from_unit_interval(
        batch.oracle_structures.edge_radius
    )
    assert torch.allclose(
        current_edge_radius_logits + target_edge_radius_step,
        oracle_edge_radius_logits,
        atol=1e-5,
    )


def test_euler_step_updates_adjacency_in_logit_space() -> None:
    structures = Structures(
        positions=torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]]),
        roles=torch.tensor([[0, 2, 2]], dtype=torch.long),
        adjacency=torch.tensor([[[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]]]),
    )
    adjacency_logit_step = torch.tensor(
        [[[0.0, 1.0, -2.0], [1.0, 0.0, 0.5], [-2.0, 0.5, 0.0]]]
    )

    next_structures = _euler_step(
        current_structures=structures,
        position_step=torch.zeros_like(structures.positions),
        adjacency_logit_step=adjacency_logit_step,
        edge_radius_logit_step=torch.zeros_like(structures.adjacency),
    )

    expected = torch.sigmoid(
        logits_from_adjacency(structures.adjacency) + adjacency_logit_step
    )
    expected[:, 0, 0] = 0.0
    expected[:, 1, 1] = 0.0
    expected[:, 2, 2] = 0.0
    assert torch.allclose(next_structures.adjacency, expected)
    assert bool(
        ((0.0 <= next_structures.adjacency) & (next_structures.adjacency <= 1.0))
        .all()
        .item()
    )


def test_euler_step_does_not_use_direct_adjacency_clamp_dynamics() -> None:
    structures = Structures(
        positions=torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]]),
        roles=torch.tensor([[0, 2, 2]], dtype=torch.long),
        adjacency=torch.tensor([[[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]]]),
    )
    adjacency_logit_step = torch.tensor(
        [[[0.0, 4.0, -4.0], [4.0, 0.0, 0.5], [-4.0, 0.5, 0.0]]]
    )

    next_structures = _euler_step(
        current_structures=structures,
        position_step=torch.zeros_like(structures.positions),
        adjacency_logit_step=adjacency_logit_step,
        edge_radius_logit_step=torch.zeros_like(structures.adjacency),
    )
    direct_clamp = torch.clamp(
        structures.adjacency + adjacency_logit_step,
        0.0,
        1.0,
    )
    assert not torch.allclose(next_structures.adjacency, direct_clamp)


def test_supervised_refiner_has_head_norms_and_positive_branch_scales() -> None:
    model = SupervisedRefiner(SupervisedRefinerConfig())

    assert isinstance(model.position_head_norm, nn.LayerNorm)
    assert isinstance(model.connectivity_head_norm, nn.LayerNorm)
    assert float(model._positive_scale(model.nodal_raw_scale).detach().item()) > 0.0
    assert float(model._positive_scale(model.mechanics_raw_scale).detach().item()) > 0.0
    assert (
        float(model._positive_scale(model.local_geometry_raw_scale).detach().item())
        > 0.0
    )
    assert float(model._positive_scale(model.time_raw_scale).detach().item()) > 0.0
    assert (
        float(model._positive_scale(model.style_residual_raw_scale).detach().item())
        > 0.0
    )


def test_normalized_edge_von_mises_features_scale_by_allowable_stress() -> None:
    edge_von_mises = torch.tensor([[[[0.0, 250e6, 500e6, float("nan")]]]])

    normalized = _normalized_edge_von_mises_features(edge_von_mises)

    expected = torch.tensor(
        [[[[0.0, torch.log1p(torch.tensor(1.0)), torch.log1p(torch.tensor(2.0)), 0.0]]]]
    )
    assert torch.allclose(normalized, expected)


def test_supervised_step_losses_use_smooth_l1() -> None:
    roles = torch.tensor([[0, 2]])
    position_loss = _position_step_loss(
        predicted_step=torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        target_step=torch.zeros(1, 2, 3),
        roles=roles,
        beta=1.0,
    )
    adjacency_loss = _adjacency_step_loss(
        predicted_step=torch.tensor([[[0.0, 2.0], [2.0, 0.0]]]),
        target_step=torch.zeros(1, 2, 2),
        beta=1.0,
    )
    edge_radius_loss = _edge_radius_step_loss(
        predicted_step=torch.tensor([[[0.0, 2.0], [2.0, 0.0]]]),
        target_step=torch.zeros(1, 2, 2),
        beta=1.0,
    )

    assert torch.allclose(position_loss, torch.tensor([1.50]))
    assert torch.allclose(adjacency_loss, torch.tensor([0.75]))
    assert torch.allclose(edge_radius_loss, torch.tensor([0.75]))


def test_euler_step_keeps_position_unclamped_and_reports_real_motion() -> None:
    structures = Structures(
        positions=torch.tensor([[[0.99, 0.01, 0.5], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]]),
        roles=torch.tensor([[0, 2, 2]], dtype=torch.long),
        adjacency=torch.tensor([[[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]]]),
    )

    next_structures = _euler_step(
        current_structures=structures,
        position_step=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.1, -0.1, 0.0], [0.0, 0.0, 0.0]]]
        ),
        adjacency_logit_step=torch.zeros_like(structures.adjacency),
        edge_radius_logit_step=torch.zeros_like(structures.adjacency),
    )

    assert torch.allclose(
        next_structures.positions,
        torch.tensor([[[0.99, 0.01, 0.5], [0.6, 0.4, 0.0], [1.0, 0.0, 0.0]]]),
    )


def test_euler_step_updates_edge_radius_in_logit_space() -> None:
    structures = Structures(
        positions=torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]]),
        roles=torch.tensor([[0, 2, 2]], dtype=torch.long),
        adjacency=torch.tensor([[[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]]]),
        edge_radius=torch.tensor(
            [[[0.0, 0.25, 0.50], [0.25, 0.0, 0.75], [0.50, 0.75, 0.0]]]
        ),
    )
    edge_radius_logit_step = torch.tensor(
        [[[0.0, 0.5, -0.5], [0.5, 0.0, 0.25], [-0.5, 0.25, 0.0]]]
    )

    next_structures = _euler_step(
        current_structures=structures,
        position_step=torch.zeros_like(structures.positions),
        adjacency_logit_step=torch.zeros_like(structures.adjacency),
        edge_radius_logit_step=edge_radius_logit_step,
    )

    expected = torch.sigmoid(
        logits_from_unit_interval(structures.edge_radius) + edge_radius_logit_step
    )
    expected[:, 0, 0] = 0.0
    expected[:, 1, 1] = 0.0
    expected[:, 2, 2] = 0.0
    assert torch.allclose(next_structures.edge_radius, expected)


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


def test_objective_clipped_gradients_clip_each_loss_and_sum_correctly() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    main_loss = 6.0 * model.weight.squeeze()
    physical_loss = 3.0 * model.weight.squeeze()

    combined_gradients, metrics = _objective_clipped_gradients(
        main_loss=main_loss,
        physical_loss=physical_loss,
        parameters=[model.weight],
        config=FlowCurriculumTrainingConfig(
            dataset_path="dataset.pt",
            main_grad_clip_norm=1.0,
            physical_grad_clip_norm=0.5,
        ),
    )

    assert torch.allclose(metrics["main_grad_norm_pre_clip"], torch.tensor(6.0))
    assert torch.allclose(metrics["main_grad_norm_post_clip"], torch.tensor(1.0))
    assert torch.allclose(metrics["physical_grad_norm_pre_clip"], torch.tensor(3.0))
    assert torch.allclose(metrics["physical_grad_norm_post_clip"], torch.tensor(0.5))
    assert torch.allclose(
        metrics["combined_grad_norm_pre_global_clip"],
        torch.tensor(1.5),
    )
    assert torch.allclose(combined_gradients[0], torch.tensor([[1.5]]))
    assert torch.allclose(
        metrics["main_physical_grad_cosine"],
        torch.tensor(1.0),
    )


def test_objective_clipped_gradients_handle_zero_physical_loss() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    main_loss = 3.0 * model.weight.squeeze()
    physical_loss = model.weight.squeeze() * 0.0

    combined_gradients, metrics = _objective_clipped_gradients(
        main_loss=main_loss,
        physical_loss=physical_loss,
        parameters=[model.weight],
        config=FlowCurriculumTrainingConfig(
            dataset_path="dataset.pt",
            main_grad_clip_norm=1.0,
            physical_grad_clip_norm=0.5,
        ),
    )

    assert torch.allclose(metrics["physical_grad_norm_pre_clip"], torch.tensor(0.0))
    assert torch.allclose(metrics["physical_grad_norm_post_clip"], torch.tensor(0.0))
    assert torch.allclose(metrics["main_grad_norm_pre_clip"], torch.tensor(3.0))
    assert torch.allclose(metrics["main_grad_norm_post_clip"], torch.tensor(1.0))
    assert torch.allclose(combined_gradients[0], torch.tensor([[1.0]]))
    assert torch.allclose(
        metrics["main_physical_grad_cosine"],
        torch.tensor(0.0),
    )


def test_objective_clipped_gradients_main_loss_can_include_kl_without_extra_group() -> (
    None
):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    main_loss = 3.0 * model.weight.squeeze()
    physical_loss = 4.0 * model.weight.squeeze()

    combined_gradients, metrics = _objective_clipped_gradients(
        main_loss=main_loss,
        physical_loss=physical_loss,
        parameters=[model.weight],
        config=FlowCurriculumTrainingConfig(
            dataset_path="dataset.pt",
            main_grad_clip_norm=1.0,
            physical_grad_clip_norm=0.5,
        ),
    )

    assert torch.allclose(metrics["main_grad_norm_pre_clip"], torch.tensor(3.0))
    assert torch.allclose(metrics["main_grad_norm_post_clip"], torch.tensor(1.0))
    assert torch.allclose(metrics["physical_grad_norm_pre_clip"], torch.tensor(4.0))
    assert torch.allclose(metrics["physical_grad_norm_post_clip"], torch.tensor(0.5))
    assert torch.allclose(combined_gradients[0], torch.tensor([[1.5]]))


def test_clip_grad_norm_per_sample_scales_each_sample_independently() -> None:
    hook = _clip_grad_norm_per_sample(2.0)
    grad = torch.tensor([[[3.0, 4.0]], [[0.3, 0.4]]])

    clipped = hook(grad)

    assert torch.allclose(clipped[0], torch.tensor([[1.2, 1.6]]))
    assert torch.allclose(clipped[1], grad[1])


def test_optimizer_parameter_groups_disable_decay_for_norms_biases_embeddings_and_scales() -> (
    None
):
    refiner = SupervisedRefiner(SupervisedRefinerConfig())

    parameter_groups = _optimizer_parameter_groups(refiner, weight_decay=1e-4)
    decay_ids = {id(parameter) for parameter in parameter_groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in parameter_groups[1]["params"]}

    assert id(refiner.role_embedding.weight) in no_decay_ids
    assert id(refiner.input_norm.weight) in no_decay_ids
    assert id(refiner.position_head_norm.weight) in no_decay_ids
    assert id(refiner.position_head[0].bias) in no_decay_ids
    assert id(refiner.mechanics_raw_scale) in no_decay_ids
    assert id(refiner.position_head[0].weight) in decay_ids


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
                    "position_step": torch.zeros_like(structures.positions),
                    "adjacency_logit_step": torch.zeros_like(structures.adjacency),
                    "edge_radius_logit_step": torch.zeros_like(structures.adjacency),
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
            absolute_physical_loss_weight=1.0,
            relative_physical_loss_weight=0.0,
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


def test_trajectory_loss_terms_keeps_mechanics_conditioning_when_physics_is_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases = _build_cases(tmp_path)
    batch = make_training_batch(cases, seed=5)
    analysis_calls: list[bool] = []

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
                    "position_step": torch.zeros_like(structures.positions),
                    "adjacency_logit_step": torch.zeros_like(structures.adjacency),
                    "edge_radius_logit_step": torch.zeros_like(structures.adjacency),
                    "style_kl": None,
                },
            )()

    def fake_analyze_structures(current: Structures, profile=None) -> Analyses:
        del profile
        analysis_calls.append(torch.is_grad_enabled())
        return _fake_analyses_from_structures(current)

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
    assert analysis_calls == [False, False, False]
    assert torch.allclose(loss_terms["physical_loss"], torch.tensor(0.0))
    assert torch.allclose(loss_terms["physical_loss_contribution"], torch.tensor(0.0))
    assert torch.allclose(metrics["stiffness_error"], torch.tensor(0.0))


def test_trajectory_loss_terms_can_use_relative_endpoint_physical_loss(
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
                current_stiffness,
                nodal_displacements,
                edge_von_mises,
                flow_times,
                style_structures,
                style_analyses,
                style_token_mask,
            )
            return type(
                "Prediction",
                (),
                {
                    "position_step": torch.full_like(structures.positions, 0.01),
                    "adjacency_logit_step": torch.zeros_like(structures.adjacency),
                    "edge_radius_logit_step": torch.zeros_like(structures.adjacency),
                    "style_kl": None,
                },
            )()

    def fake_analyze_structures(current: Structures, profile=None) -> Analyses:
        del profile
        base = current.positions.sum(dim=(1, 2))
        batch_size, num_nodes, _ = current.positions.shape
        return Analyses(
            generalized_stiffness=base[:, None, None].expand(batch_size, 6, 6),
            material_usage=base,
            short_beam_penalty=torch.zeros_like(base),
            long_beam_penalty=torch.zeros_like(base),
            thin_beam_penalty=torch.zeros_like(base),
            thick_beam_penalty=torch.zeros_like(base),
            free_node_spacing_penalty=torch.zeros_like(base),
            nodal_displacements=torch.zeros(batch_size, num_nodes, 18),
            edge_von_mises=torch.zeros(batch_size, num_nodes, num_nodes, 6),
        )

    def fake_structural_objective_terms(**kwargs):
        analyses = kwargs["analyses"]
        value = analyses.material_usage
        zeros = torch.zeros_like(value)
        return (
            {
                "stiffness_loss_contribution": value,
                "stress_loss_contribution": zeros,
            },
            {
                "stiffness_error": value,
                "stress_violation": zeros,
                "mean_stress_ratio": zeros,
                "max_stress_ratio": zeros,
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

    _, _, loss_terms = _trajectory_loss_terms(
        model=FakeModel(),
        batch=batch,
        config=FlowCurriculumTrainingConfig(
            dataset_path="dataset.pt",
            num_integration_steps=2,
            physical_weight_start=1.0,
            physical_weight_end=1.0,
            physical_transition_start_step=0,
            physical_transition_end_step=0,
            absolute_physical_loss_weight=0.5,
            relative_physical_loss_weight=0.5,
        ),
        step=0,
    )

    _, _, free_mask = role_masks(batch.initial_structures.roles)
    position_delta = 0.02 * free_mask.unsqueeze(-1).to(
        dtype=batch.initial_structures.positions.dtype
    )
    initial_loss = batch.initial_structures.positions.sum(dim=(1, 2)).mean()
    endpoint_loss = (
        (batch.initial_structures.positions + position_delta).sum(dim=(1, 2)).mean()
    )
    expected = 0.5 * endpoint_loss + 0.5 * (
        torch.log1p(endpoint_loss) - torch.log1p(initial_loss)
    )
    assert torch.allclose(loss_terms["physical_loss"], expected)


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


def test_train_flow_refiner_uses_single_optimizer_step_per_iteration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path, cases = _build_cases(tmp_path)
    step_calls = 0
    original_step = torch.optim.AdamW.step

    def counted_step(self, closure=None):
        nonlocal step_calls
        step_calls += 1
        return original_step(self, closure=closure)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)

    train_flow_refiner(
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
            checkpoint_path=str(tmp_path / "refiner_step_count.pt"),
            logdir=str(tmp_path / "runs_step_count"),
            seed=13,
        ),
    )

    assert step_calls == 2


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
