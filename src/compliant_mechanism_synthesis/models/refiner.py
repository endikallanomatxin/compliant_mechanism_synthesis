from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from compliant_mechanism_synthesis.dataset.types import Structures
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import (
    distance_affinity,
    enforce_role_adjacency_constraints,
    max_length_gate,
    symmetric_matrix_unique_values,
)


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(1000.0), steps=half, device=values.device) * -1.0
    )
    angles = values[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _masked_centroid(positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = positions * mask.unsqueeze(-1).to(dtype=positions.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=positions.dtype)
    return weighted.sum(dim=1) / count


def _node_context_features(
    positions: torch.Tensor,
    roles: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    degree = adjacency.sum(dim=-1, keepdim=True)
    pairwise = torch.linalg.vector_norm(
        positions[:, :, None, :] - positions[:, None, :, :],
        dim=-1,
    )
    mean_edge_length = (
        (adjacency * pairwise).sum(dim=-1, keepdim=True) / degree.clamp_min(1e-6)
    )
    structure_centroid = positions.mean(dim=1, keepdim=True)
    fixed_mask, mobile_mask, _ = role_masks(roles)
    fixed_centroid = _masked_centroid(positions, fixed_mask)[:, None, :]
    mobile_centroid = _masked_centroid(positions, mobile_mask)[:, None, :]
    distance_to_structure = torch.linalg.vector_norm(positions - structure_centroid, dim=-1, keepdim=True)
    distance_to_fixed = torch.linalg.vector_norm(positions - fixed_centroid, dim=-1, keepdim=True)
    distance_to_mobile = torch.linalg.vector_norm(positions - mobile_centroid, dim=-1, keepdim=True)
    anchor_mask = (fixed_mask | mobile_mask).to(dtype=adjacency.dtype)
    anchor_attachment = (adjacency * anchor_mask.unsqueeze(1)).sum(dim=-1, keepdim=True)
    return torch.cat(
        [
            degree,
            mean_edge_length,
            distance_to_structure,
            distance_to_fixed,
            distance_to_mobile,
            anchor_attachment,
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class FlowPrediction:
    position_velocity: torch.Tensor
    adjacency_velocity: torch.Tensor
    predicted_adjacency: torch.Tensor
    node_latents: torch.Tensor


@dataclass(frozen=True)
class SupervisedRefinerConfig:
    hidden_dim: int = 128
    latent_dim: int = 64
    num_attention_layers: int = 6
    num_heads: int = 4
    num_integration_steps: int = 8
    max_distance: float = 0.24
    transition_width: float = 0.08


class GraphAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mode: str) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if mode not in {"distance", "connectivity", "free"}:
            raise ValueError("mode must be distance, connectivity, or free")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _conditioning_matrix(
        self,
        adjacency: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.mode == "free":
            return None
        if self.mode == "connectivity":
            return adjacency
        return distance_affinity(positions, length_scale=0.22)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_nodes, _ = hidden.shape
        normalized = self.norm1(hidden)
        query = self.q_proj(normalized).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(normalized).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(normalized).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        conditioning = self._conditioning_matrix(adjacency, positions)
        if conditioning is not None:
            logits = logits + (2.0 * conditioning[:, None, :, :] - 1.0)

        weights = torch.softmax(logits, dim=-1)
        if conditioning is not None:
            weights = weights * (0.25 + 0.75 * conditioning[:, None, :, :])
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, num_nodes, self.hidden_dim)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ff(self.norm2(hidden))
        return hidden


class SupervisedRefiner(nn.Module):
    def __init__(self, config: SupervisedRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SupervisedRefinerConfig()
        self.position_mlp = nn.Sequential(
            nn.Linear(9, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.role_embedding = nn.Embedding(3, self.config.hidden_dim)
        self.mechanics_condition_mlp = nn.Sequential(
            nn.Linear(63, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.noise_condition_mlp = nn.Sequential(
            nn.Linear(3, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)
        layer_modes = ("distance", "connectivity", "free")
        self.layers = nn.ModuleList(
            [
                GraphAttentionBlock(
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    mode=layer_modes[index % len(layer_modes)],
                )
                for index in range(self.config.num_attention_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_dim)
        self.position_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 3),
        )
        self.node_latent_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )
        nn.init.normal_(self.position_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.position_head[-1].bias)

    def predict_flow(
        self,
        structures: Structures,
        target_stiffness: torch.Tensor,
        current_stiffness: torch.Tensor,
        flow_times: torch.Tensor,
        position_noise_levels: torch.Tensor,
        adjacency_noise_levels: torch.Tensor,
    ) -> FlowPrediction:
        structures.validate()
        if target_stiffness.shape != (structures.batch_size, 6, 6):
            raise ValueError("target_stiffness must have shape [batch, 6, 6]")
        if current_stiffness.shape != (structures.batch_size, 6, 6):
            raise ValueError("current_stiffness must have shape [batch, 6, 6]")
        for name, value in (
            ("flow_times", flow_times),
            ("position_noise_levels", position_noise_levels),
            ("adjacency_noise_levels", adjacency_noise_levels),
        ):
            if value.shape != (structures.batch_size,):
                raise ValueError(f"{name} must have shape [batch]")

        positions = structures.positions
        roles = structures.roles
        current_adjacency = enforce_role_adjacency_constraints(structures.adjacency, roles)
        residual_stiffness = target_stiffness - current_stiffness
        hidden = self.position_mlp(
            torch.cat([positions, _node_context_features(positions, roles, current_adjacency)], dim=-1)
        )
        hidden = hidden + self.role_embedding(roles)
        hidden = self.input_norm(hidden)

        mechanics_features = torch.cat(
            [
                symmetric_matrix_unique_values(target_stiffness),
                symmetric_matrix_unique_values(current_stiffness),
                symmetric_matrix_unique_values(residual_stiffness),
            ],
            dim=1,
        )
        hidden = hidden + self.mechanics_condition_mlp(mechanics_features)[:, None, :]
        hidden = hidden + self.time_mlp(
            sinusoidal_embedding(flow_times, self.config.hidden_dim)
        )[:, None, :]
        noise_features = torch.stack(
            [flow_times, position_noise_levels, adjacency_noise_levels],
            dim=1,
        )
        hidden = hidden + self.noise_condition_mlp(noise_features)[:, None, :]

        for layer in self.layers:
            hidden = layer(hidden, current_adjacency, positions)

        hidden = self.final_norm(hidden)
        position_velocity = self.position_head(hidden)
        node_latents = self.node_latent_head(hidden)
        # Connectivity updates stay factored through node latents so the model
        # cannot bypass the intended graph-inductive bias with edge-wise heads.
        scores = torch.matmul(node_latents, node_latents.transpose(1, 2)) / math.sqrt(self.config.latent_dim)
        delta_scores = torch.tanh(scores)
        update_gate = max_length_gate(
            positions,
            max_distance=self.config.max_distance,
            transition_width=self.config.transition_width,
        )
        predicted_adjacency = enforce_role_adjacency_constraints(
            (
                current_adjacency
                + delta_scores * update_gate
                - current_adjacency * (1.0 - update_gate)
            ).clamp(0.0, 1.0),
            roles,
        )
        adjacency_velocity = predicted_adjacency - current_adjacency
        return FlowPrediction(
            position_velocity=position_velocity,
            adjacency_velocity=adjacency_velocity,
            predicted_adjacency=predicted_adjacency,
            node_latents=node_latents,
        )

    def rollout(
        self,
        source_structures: Structures,
        target_stiffness: torch.Tensor,
        analysis_fn,
        num_steps: int | None = None,
    ) -> Structures:
        source_structures.validate()
        num_steps = num_steps or self.config.num_integration_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        current = source_structures
        initial_position_gap = current.positions.new_zeros(current.batch_size)
        initial_adjacency_gap = current.adjacency.new_zeros(current.batch_size)
        _, _, free_mask = role_masks(current.roles)
        free_mask = free_mask.unsqueeze(-1).to(dtype=current.positions.dtype)

        for step in range(num_steps):
            analyses = analysis_fn(current)
            flow_time = current.positions.new_full(
                (current.batch_size,),
                (step + 0.5) / num_steps,
            )
            remaining = 1.0 - flow_time
            prediction = self.predict_flow(
                structures=current,
                target_stiffness=target_stiffness,
                current_stiffness=analyses.generalized_stiffness,
                flow_times=flow_time,
                position_noise_levels=remaining * initial_position_gap,
                adjacency_noise_levels=remaining * initial_adjacency_gap,
            )
            step_size = 1.0 / num_steps
            positions = (current.positions + step_size * prediction.position_velocity * free_mask).clamp(0.0, 1.0)
            adjacency = enforce_role_adjacency_constraints(
                (current.adjacency + step_size * prediction.adjacency_velocity).clamp(0.0, 1.0),
                current.roles,
            )
            current = Structures(
                positions=positions,
                roles=current.roles,
                adjacency=adjacency,
            )
        current.validate()
        return current

    def forward(
        self,
        source_structures: Structures,
        target_stiffness: torch.Tensor,
        analysis_fn,
        num_steps: int | None = None,
    ) -> Structures:
        return self.rollout(
            source_structures=source_structures,
            target_stiffness=target_stiffness,
            analysis_fn=analysis_fn,
            num_steps=num_steps,
        )
