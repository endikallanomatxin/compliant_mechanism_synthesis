from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from compliant_mechanism_synthesis.common import (
    ROLE_FIXED,
    ROLE_FREE,
    ROLE_MOBILE,
    symmetrize_adjacency,
)
from compliant_mechanism_synthesis.data import (
    generate_graph_sample,
    generate_noise_sample,
    proximity_bias_matrix,
    rigid_endpoint_scaffold,
)
from compliant_mechanism_synthesis.mechanics import (
    FrameFEMConfig,
    GeometryRegularizationConfig,
    assemble_global_stiffness,
    effective_response,
    fixed_mobile_connectivity_penalty,
    geometric_regularization_terms,
    mechanical_response_fields,
    load_case_deformations,
    mechanical_terms,
    refine_connectivity,
    rigid_attachment_penalty,
    threshold_connectivity,
)
from compliant_mechanism_synthesis.viz import plot_graph_design


def _test_run_dir(test_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-test-{test_name}"


def test_global_stiffness_is_symmetric() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    stiffness = assemble_global_stiffness(
        positions.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert stiffness.shape == (1, 30, 30)
    assert torch.allclose(stiffness, stiffness.transpose(1, 2), atol=1e-5)


def test_rigid_body_effective_response_is_finite_and_symmetric() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    response_matrix, stiffness_matrix = effective_response(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert response_matrix.shape == (1, 3, 3)
    assert stiffness_matrix.shape == (1, 3, 3)
    assert torch.isfinite(response_matrix).all()
    assert torch.isfinite(stiffness_matrix).all()
    assert torch.allclose(response_matrix, response_matrix.transpose(1, 2), atol=1e-5)
    assert torch.allclose(stiffness_matrix, stiffness_matrix.transpose(1, 2), atol=1e-5)


def test_load_case_deformations_match_response_shape() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    deformations, response_matrix = load_case_deformations(
        positions.unsqueeze(0),
        roles.unsqueeze(0),
        adjacency.unsqueeze(0),
    )
    assert deformations.shape == (1, 3, 10, 2)
    assert response_matrix.shape == (1, 3, 3)
    assert torch.isfinite(deformations).all()
    assert torch.isfinite(response_matrix).all()


def test_mechanical_terms_have_expected_keys() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    response_matrix, stiffness_matrix = effective_response(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    terms = mechanical_terms(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    assert terms["response_matrix"].shape == (1, 3, 3)
    assert terms["stiffness_matrix"].shape == (1, 3, 3)
    assert terms["translations"].shape == (1, 3, 10, 2)
    assert terms["nodal_stress"].shape == (1, 10)
    assert terms["normalized_stiffness_matrix"].shape == (1, 3, 3)
    assert terms["normalized_response_matrix"].shape == (1, 3, 3)
    assert terms["normalized_translations"].shape == (1, 3, 10, 2)
    assert terms["normalized_nodal_stress"].shape == (1, 10)
    assert terms["normalized_material"].shape == (1,)
    assert torch.allclose(terms["response_matrix"], response_matrix)
    assert torch.allclose(terms["stiffness_matrix"], stiffness_matrix)
    assert terms["connectivity_penalty"].shape == (1,)
    assert terms["fixed_mobile_connectivity_penalty"].shape == (1,)
    assert terms["sparsity"].shape == (1,)
    assert terms["material"].shape == (1,)
    assert terms["short_beam_penalty"].shape == (1,)
    assert terms["long_beam_penalty"].shape == (1,)
    assert terms["thin_diameter_penalty"].shape == (1,)
    assert terms["thick_diameter_penalty"].shape == (1,)
    assert terms["node_spacing_penalty"].shape == (1,)
    assert terms["free_repulsion_penalty"].shape == (1,)
    assert terms["rigid_attachment_penalty"].shape == (1,)
    assert terms["centroid_penalty"].shape == (1,)
    assert terms["spread_penalty"].shape == (1,)
    assert terms["soft_domain_penalty"].shape == (1,)
    assert terms["structural_integrity_penalty"].shape == (1,)


def test_structural_integrity_terms_are_zero_for_absent_bars() -> None:
    positions, roles, _ = generate_graph_sample(10)
    adjacency = torch.zeros((1, 10, 10), dtype=torch.float32)
    fields = mechanical_response_fields(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency
    )

    assert torch.allclose(
        fields["nodal_stress"], torch.zeros_like(fields["nodal_stress"])
    )
    assert torch.isclose(fields["structural_integrity_penalty"][0], torch.tensor(0.0))


def test_structural_integrity_penalty_accumulates_with_reference_loads() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    fields = mechanical_response_fields(
        positions.unsqueeze(0),
        roles.unsqueeze(0),
        adjacency.unsqueeze(0),
        config=FrameFEMConfig(yield_stress=1e12),
    )

    assert fields["nodal_stress"].max() > 0.0
    assert fields["structural_integrity_penalty"][0] > 0.0


def test_reference_loads_preserve_stiffness_matrix_units() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    default_fields = mechanical_response_fields(
        positions.unsqueeze(0), roles.unsqueeze(0), adjacency.unsqueeze(0)
    )
    stronger_load_fields = mechanical_response_fields(
        positions.unsqueeze(0),
        roles.unsqueeze(0),
        adjacency.unsqueeze(0),
        config=FrameFEMConfig(reference_force=20.0, reference_moment=4.0),
    )

    assert torch.allclose(
        default_fields["stiffness_matrix"],
        stronger_load_fields["stiffness_matrix"],
        atol=1e-4,
        rtol=1e-4,
    )
    assert stronger_load_fields["structural_integrity_penalty"][0] > default_fields[
        "structural_integrity_penalty"
    ][0]


def test_geometric_regularization_penalizes_extreme_lengths_and_diameters() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [0.0025, 0.0], [0.15, 0.0]]], dtype=torch.float32
    )
    roles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    adjacency = torch.tensor(
        [[[0.0, 0.1, 0.95], [0.1, 0.0, 0.0], [0.95, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(
            min_length=1e-3,
            max_length=2e-2,
            min_diameter=2.5e-4,
            max_diameter=1.5e-3,
        ),
        frame_config=FrameFEMConfig(workspace_size=0.2, r_max=1e-3),
    )
    assert penalties["short_beam_penalty"][0] > 0.0
    assert penalties["long_beam_penalty"][0] > 0.0
    assert penalties["thin_diameter_penalty"][0] > 0.0
    assert penalties["thick_diameter_penalty"][0] > 0.0


def test_thin_diameter_penalty_is_zero_for_absent_or_fabricable_bars() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]], dtype=torch.float32
    )
    roles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    config = GeometryRegularizationConfig(min_diameter=2e-4, max_diameter=2e-3)
    frame = FrameFEMConfig(workspace_size=0.2, r_max=1e-3)

    absent = geometric_regularization_terms(
        positions,
        roles,
        torch.zeros((1, 3, 3), dtype=torch.float32),
        config,
        frame_config=frame,
    )
    fabricable = geometric_regularization_terms(
        positions,
        roles,
        torch.tensor(
            [[[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        config,
        frame_config=frame,
    )

    assert torch.isclose(absent["thin_diameter_penalty"][0], torch.tensor(0.0))
    assert torch.isclose(fabricable["thin_diameter_penalty"][0], torch.tensor(0.0))


def test_thin_diameter_penalty_targets_intermediate_subfabricable_bars() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]], dtype=torch.float32
    )
    roles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    config = GeometryRegularizationConfig(min_diameter=8e-4, max_diameter=2e-3)
    frame = FrameFEMConfig(workspace_size=0.2, r_max=1e-3)

    nearly_absent = geometric_regularization_terms(
        positions,
        roles,
        torch.tensor(
            [[[0.0, 0.02, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        config,
        frame_config=frame,
    )
    intermediate = geometric_regularization_terms(
        positions,
        roles,
        torch.tensor(
            [[[0.0, 0.25, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        config,
        frame_config=frame,
    )
    fabricable = geometric_regularization_terms(
        positions,
        roles,
        torch.tensor(
            [[[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        config,
        frame_config=frame,
    )

    assert nearly_absent["thin_diameter_penalty"][0] > 0.0
    assert intermediate["thin_diameter_penalty"][0] > nearly_absent[
        "thin_diameter_penalty"
    ][0]
    assert torch.isclose(fabricable["thin_diameter_penalty"][0], torch.tensor(0.0))


def test_geometric_regularization_penalizes_node_clustering_and_soft_domain_escape() -> (
    None
):
    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-0.05, 0.02],
                [-0.049, 0.021],
            ]
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)
    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(min_free_node_spacing=5e-3),
        frame_config=FrameFEMConfig(workspace_size=0.2, r_max=1e-3),
    )
    assert penalties["node_spacing_penalty"][0] > 0.0
    assert penalties["free_repulsion_penalty"][0] > 0.0
    assert penalties["spread_penalty"][0] > 0.0
    assert penalties["soft_domain_penalty"][0] > 0.0


def test_geometric_regularization_penalizes_insufficient_free_node_spread() -> None:
    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.48, 0.49],
                [0.50, 0.50],
                [0.52, 0.51],
            ]
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 0, 1, 1, 2, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 7, 7), dtype=torch.float32)

    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(min_free_node_spacing=1e-3),
    )

    assert penalties["spread_penalty"][0] > 0.0


def test_geometric_regularization_penalizes_dense_free_clusters_before_overlap() -> None:
    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.45, 0.50],
                [0.48, 0.50],
                [0.51, 0.50],
            ]
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 0, 1, 1, 2, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 7, 7), dtype=torch.float32)

    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(min_free_node_spacing=5e-3),
    )

    assert penalties["node_spacing_penalty"][0] == 0.0
    assert penalties["free_repulsion_penalty"][0] > 0.0


def test_geometric_regularization_penalizes_free_node_centroid_drift() -> None:
    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.78, 0.80],
                [0.82, 0.79],
                [0.80, 0.83],
            ]
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 0, 1, 1, 2, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 7, 7), dtype=torch.float32)

    penalties = geometric_regularization_terms(
        positions,
        roles,
        adjacency,
        GeometryRegularizationConfig(min_free_node_spacing=1e-3),
    )

    assert penalties["centroid_penalty"][0] > 0.0


def test_soft_domain_penalty_detects_far_escaped_free_nodes() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 7, 7), dtype=torch.float32)

    inside_cluster = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.48, 0.49],
                [0.50, 0.50],
                [0.52, 0.51],
            ]
        ],
        dtype=torch.float32,
    )
    escaped_cluster = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-2.0, -2.0],
                [0.50, 0.50],
                [3.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )

    escaped_penalties = geometric_regularization_terms(
        escaped_cluster,
        roles,
        adjacency,
        GeometryRegularizationConfig(min_free_node_spacing=1e-3),
    )

    assert escaped_penalties["soft_domain_penalty"][0] > 0.0


def test_thresholded_connectivity_is_symmetric_and_zero_diagonal() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    thresholded = threshold_connectivity(
        adjacency.unsqueeze(0), roles.unsqueeze(0), threshold=0.4
    )
    assert thresholded.shape == (1, 10, 10)
    assert torch.allclose(thresholded, thresholded.transpose(1, 2))
    assert torch.allclose(torch.diagonal(thresholded[0]), torch.zeros(10))
    assert torch.allclose(thresholded, symmetrize_adjacency(thresholded))


def test_rigid_endpoint_nodes_never_connect_directly() -> None:
    positions, roles, adjacency = generate_graph_sample(10)
    fixed = torch.where(roles == 0)[0]
    mobile = torch.where(roles == 1)[0]
    assert torch.isclose(adjacency[fixed[0], fixed[1]], torch.tensor(0.0))
    assert torch.isclose(adjacency[mobile[0], mobile[1]], torch.tensor(0.0))
    for fixed_idx in fixed.tolist():
        for mobile_idx in mobile.tolist():
            assert torch.isclose(adjacency[fixed_idx, mobile_idx], torch.tensor(0.0))

    forced = torch.ones((1, 10, 10), dtype=torch.float32)
    thresholded = threshold_connectivity(forced, roles.unsqueeze(0), threshold=0.4)
    assert torch.isclose(thresholded[0, fixed[0], fixed[1]], torch.tensor(0.0))
    assert torch.isclose(thresholded[0, mobile[0], mobile[1]], torch.tensor(0.0))
    for fixed_idx in fixed.tolist():
        for mobile_idx in mobile.tolist():
            assert torch.isclose(
                thresholded[0, fixed_idx, mobile_idx], torch.tensor(0.0)
            )


def test_sparsity_penalty_increases_with_more_active_edges() -> None:
    positions, roles, _ = generate_graph_sample(10)
    sparse = torch.zeros((1, 10, 10), dtype=torch.float32)
    sparse[:, 0, 4] = 0.2
    sparse[:, 4, 0] = 0.2
    dense = torch.full((1, 10, 10), 0.2, dtype=torch.float32)
    dense = dense - torch.diag_embed(torch.diagonal(dense, dim1=1, dim2=2))
    sparse_terms = mechanical_terms(positions.unsqueeze(0), roles.unsqueeze(0), sparse)
    dense_terms = mechanical_terms(positions.unsqueeze(0), roles.unsqueeze(0), dense)
    assert dense_terms["sparsity"][0] > sparse_terms["sparsity"][0]


def test_fixed_mobile_connectivity_penalty_detects_flying_mobile_nodes() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)

    penalty = fixed_mobile_connectivity_penalty(roles, adjacency)

    assert torch.isclose(penalty[0], torch.tensor(1.0))


def test_fixed_mobile_connectivity_penalty_drops_when_mobile_is_anchored() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)
    adjacency[0, 0, 4] = 1.0
    adjacency[0, 4, 0] = 1.0
    adjacency[0, 4, 2] = 1.0
    adjacency[0, 2, 4] = 1.0
    adjacency[0, 1, 5] = 1.0
    adjacency[0, 5, 1] = 1.0
    adjacency[0, 5, 3] = 1.0
    adjacency[0, 3, 5] = 1.0

    penalty = fixed_mobile_connectivity_penalty(roles, adjacency)

    assert torch.isclose(penalty[0], torch.tensor(0.0))


def test_rigid_attachment_penalty_detects_weak_endpoint_attachment() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)
    adjacency[0, 0, 4] = 0.1
    adjacency[0, 4, 0] = 0.1
    adjacency[0, 2, 5] = 0.1
    adjacency[0, 5, 2] = 0.1

    penalty = rigid_attachment_penalty(
        roles,
        adjacency,
        min_attachment_activation=0.3,
    )

    assert penalty[0] > 0.0


def test_rigid_attachment_penalty_drops_with_strong_endpoint_attachment() -> None:
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    adjacency = torch.zeros((1, 6, 6), dtype=torch.float32)
    adjacency[0, 0, 4] = 0.6
    adjacency[0, 4, 0] = 0.6
    adjacency[0, 1, 5] = 0.6
    adjacency[0, 5, 1] = 0.6
    adjacency[0, 2, 4] = 0.6
    adjacency[0, 4, 2] = 0.6
    adjacency[0, 3, 5] = 0.6
    adjacency[0, 5, 3] = 0.6

    penalty = rigid_attachment_penalty(
        roles,
        adjacency,
        min_attachment_activation=0.3,
    )

    assert torch.isclose(penalty[0], torch.tensor(0.0))


def test_proximity_bias_prefers_closer_pairs() -> None:
    positions = torch.tensor(
        [[0.10, 0.10], [0.18, 0.10], [0.75, 0.10], [0.90, 0.10]],
        dtype=torch.float32,
    )
    bias = proximity_bias_matrix(positions, length_scale=0.2)
    assert bias[0, 1] > bias[0, 2]
    assert bias[2, 3] > bias[0, 3]


def test_noise_connectivity_gives_each_node_a_local_allowed_edge() -> None:
    _, _, adjacency = generate_noise_sample(10)
    incident = adjacency.sum(dim=1)
    assert torch.all(incident > 0.0)


def test_noise_starting_configurations_have_reasonable_batch_quality() -> None:
    random.seed(7)
    torch.manual_seed(7)

    batch_size = 24
    log_dir = _test_run_dir("noise-starting-configurations")
    writer = SummaryWriter(log_dir=str(log_dir))
    min_incident_strengths: list[float] = []
    sparsities: list[float] = []
    fixed_mobile_penalties: list[float] = []
    samples: list[dict[str, torch.Tensor | float]] = []

    for _ in range(batch_size):
        positions, roles, adjacency = generate_noise_sample(32)
        terms = mechanical_terms(
            positions.unsqueeze(0),
            roles.unsqueeze(0),
            adjacency.unsqueeze(0),
        )

        incident = adjacency.sum(dim=1)
        min_incident_strengths.append(float(incident.min()))
        sparsities.append(float(terms["sparsity"][0]))
        fixed_mobile_penalties.append(
            float(terms["fixed_mobile_connectivity_penalty"][0])
        )
        samples.append(
            {
                "positions": positions,
                "roles": roles,
                "adjacency": adjacency,
                "min_incident_strength": float(incident.min()),
                "sparsity": float(terms["sparsity"][0]),
                "fixed_mobile_connectivity_penalty": float(
                    terms["fixed_mobile_connectivity_penalty"][0]
                ),
            }
        )

        assert torch.all(incident > 0.0)
        assert torch.isclose(terms["rigid_attachment_penalty"][0], torch.tensor(0.0))

        for rigid_idx in range(4):
            free_neighbors = adjacency[rigid_idx, 4:]
            assert torch.count_nonzero(free_neighbors >= 0.75) >= 3

    mean_min_incident = sum(min_incident_strengths) / batch_size
    mean_sparsity = sum(sparsities) / batch_size
    mean_fixed_mobile_penalty = sum(fixed_mobile_penalties) / batch_size

    assert mean_min_incident >= 1.0
    assert 0.06 <= mean_sparsity <= 0.09
    assert mean_fixed_mobile_penalty <= 0.9

    writer.add_scalar("starting/mean_min_incident_strength", mean_min_incident, 0)
    writer.add_scalar("starting/mean_sparsity", mean_sparsity, 0)
    writer.add_scalar(
        "starting/mean_fixed_mobile_connectivity_penalty",
        mean_fixed_mobile_penalty,
        0,
    )
    writer.add_scalar(
        "starting/min_min_incident_strength",
        min(min_incident_strengths),
        0,
    )
    writer.add_scalar(
        "starting/max_fixed_mobile_connectivity_penalty",
        max(fixed_mobile_penalties),
        0,
    )

    ranked_samples = sorted(
        samples,
        key=lambda sample: (
            sample["fixed_mobile_connectivity_penalty"],
            -sample["min_incident_strength"],
        ),
    )
    for idx, sample in enumerate(ranked_samples[:6]):
        figure = plot_graph_design(
            sample["positions"],
            sample["roles"],
            sample["adjacency"],
            threshold=0.5,
            title=(
                f"start {idx} "
                f"conn={sample['fixed_mobile_connectivity_penalty']:.3f} "
                f"min_inc={sample['min_incident_strength']:.3f}"
            ),
        )
        writer.add_figure(f"starting/examples/{idx:02d}", figure, global_step=0)
        plt.close(figure)

    torch.save(
        {
            "mean_min_incident_strength": mean_min_incident,
            "mean_sparsity": mean_sparsity,
            "mean_fixed_mobile_connectivity_penalty": mean_fixed_mobile_penalty,
            "samples": ranked_samples,
        },
        log_dir / "starting_configurations.pt",
    )
    writer.close()


def test_rigid_endpoint_scaffold_connects_each_rigid_node_to_three_nearest_free_nodes() -> (
    None
):
    positions = torch.tensor(
        [
            [0.05, 0.05],
            [0.95, 0.05],
            [0.05, 0.95],
            [0.95, 0.95],
            [0.12, 0.10],
            [0.14, 0.16],
            [0.18, 0.08],
            [0.86, 0.12],
            [0.90, 0.16],
            [0.82, 0.08],
            [0.12, 0.84],
            [0.18, 0.90],
            [0.08, 0.80],
            [0.88, 0.84],
            [0.82, 0.90],
            [0.92, 0.80],
        ],
        dtype=torch.float32,
    )
    roles = torch.tensor(
        [
            ROLE_FIXED,
            ROLE_FIXED,
            ROLE_MOBILE,
            ROLE_MOBILE,
            *([ROLE_FREE] * 12),
        ],
        dtype=torch.long,
    )

    adjacency = rigid_endpoint_scaffold(positions, roles)

    for rigid_idx in range(4):
        free_neighbors = torch.where(adjacency[rigid_idx, 4:] > 0.0)[0]
        assert free_neighbors.numel() == 3
        assert torch.all(adjacency[rigid_idx, 4:][free_neighbors] >= 0.75)


def test_refine_connectivity_updates_near_edges_more_than_far_edges() -> None:
    positions = torch.tensor(
        [[[0.10, 0.10], [0.18, 0.10], [0.85, 0.10], [0.92, 0.10]]],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 2, 2, 1]], dtype=torch.long)
    adjacency = torch.zeros((1, 4, 4), dtype=torch.float32)
    delta = torch.ones((1, 4, 4), dtype=torch.float32)
    updated = refine_connectivity(adjacency, positions, roles, delta, step_size=0.1)
    assert updated[0, 0, 1] > updated[0, 0, 2]


def test_refine_connectivity_decays_far_active_edges() -> None:
    positions = torch.tensor(
        [[[0.10, 0.10], [0.18, 0.10], [0.85, 0.10], [0.92, 0.10]]],
        dtype=torch.float32,
    )
    roles = torch.tensor([[0, 2, 2, 1]], dtype=torch.long)
    adjacency = torch.zeros((1, 4, 4), dtype=torch.float32)
    adjacency[0, 0, 2] = 0.8
    adjacency[0, 2, 0] = 0.8
    delta = torch.ones((1, 4, 4), dtype=torch.float32)
    updated = refine_connectivity(adjacency, positions, roles, delta, step_size=0.1)
    assert updated[0, 0, 2] < adjacency[0, 0, 2]
