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
    use_style_token: bool = True
    style_token_dropout: float = 0.1
    style_token_noise_std: float = 0.01


class GraphAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mode: str) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if mode not in {"distance", "connectivity", "stress", "free"}:
            raise ValueError("mode must be distance, connectivity, stress, or free")
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
        if self.mode == "stress":
            return None
        return distance_affinity(positions, length_scale=0.22)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        positions: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
        edge_head_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, num_nodes, _ = hidden.shape
        normalized = self.norm1(hidden)
        query = (
            self.q_proj(normalized)
            .view(batch, num_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_source = normalized
        if context_tokens is not None:
            normalized_context = self.norm1(context_tokens)
            key_source = torch.cat([normalized, normalized_context], dim=1)
        key = (
            self.k_proj(key_source)
            .view(batch, key_source.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(key_source)
            .view(batch, key_source.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        conditioning = self._conditioning_matrix(adjacency, positions)
        if conditioning is not None:
            if context_tokens is not None:
                context_count = context_tokens.shape[1]
                conditioning_bias = torch.cat(
                    [
                        2.0 * conditioning - 1.0,
                        torch.zeros(
                            batch,
                            num_nodes,
                            context_count,
                            device=conditioning.device,
                            dtype=conditioning.dtype,
                        ),
                    ],
                    dim=-1,
                )
            else:
                conditioning_bias = 2.0 * conditioning - 1.0
            logits = logits + conditioning_bias[:, None, :, :]

        if edge_head_conditioning is not None:
            if context_tokens is not None:
                context_count = context_tokens.shape[1]
                edge_head_conditioning = torch.cat(
                    [
                        edge_head_conditioning,
                        torch.zeros(
                            batch,
                            self.num_heads,
                            num_nodes,
                            context_count,
                            device=edge_head_conditioning.device,
                            dtype=edge_head_conditioning.dtype,
                        ),
                    ],
                    dim=-1,
                )
            logits = logits + edge_head_conditioning

        weights = torch.softmax(logits, dim=-1)
        if conditioning is not None:
            if context_tokens is not None:
                context_count = context_tokens.shape[1]
                weight_scale = torch.cat(
                    [
                        0.25 + 0.75 * conditioning,
                        torch.ones(
                            batch,
                            num_nodes,
                            context_count,
                            device=conditioning.device,
                            dtype=conditioning.dtype,
                        ),
                    ],
                    dim=-1,
                )
            else:
                weight_scale = 0.25 + 0.75 * conditioning
            weights = weights * weight_scale[:, None, :, :]
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, num_nodes, self.hidden_dim)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ff(self.norm2(hidden))
        return hidden


class StyleTokenEncoder(nn.Module):
    def __init__(self, config: SupervisedRefinerConfig) -> None:
        super().__init__()
        style_hidden_dim = max(config.num_heads, config.hidden_dim // 2)
        if style_hidden_dim % config.num_heads != 0:
            style_hidden_dim = config.num_heads * max(
                1, style_hidden_dim // config.num_heads
            )
        self.hidden_dim = style_hidden_dim
        self.position_mlp = nn.Sequential(
            nn.Linear(3, style_hidden_dim),
            nn.GELU(),
            nn.Linear(style_hidden_dim, style_hidden_dim),
        )
        self.role_embedding = nn.Embedding(3, style_hidden_dim)
        self.input_norm = nn.LayerNorm(style_hidden_dim)
        layer_modes = ("distance", "connectivity", "free")
        self.layers = nn.ModuleList(
            [
                GraphAttentionBlock(
                    hidden_dim=style_hidden_dim,
                    num_heads=config.num_heads,
                    mode=layer_modes[index % len(layer_modes)],
                )
                for index in range(max(1, config.num_attention_layers // 2))
            ]
        )
        self.final_norm = nn.LayerNorm(style_hidden_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(style_hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.dropout = nn.Dropout(config.style_token_dropout)
        self.noise_std = config.style_token_noise_std

    def forward(self, structures: Structures) -> torch.Tensor:
        hidden = self.position_mlp(structures.positions)
        hidden = hidden + self.role_embedding(structures.roles)
        hidden = self.input_norm(hidden)
        for layer in self.layers:
            hidden = layer(
                hidden,
                structures.adjacency,
                structures.positions,
            )
        hidden = self.final_norm(hidden)
        token = self.token_proj(hidden.mean(dim=1, keepdim=True))
        token = self.dropout(token)
        if self.training and self.noise_std > 0.0:
            token = token + self.noise_std * torch.randn_like(token)
        return token


class SupervisedRefiner(nn.Module):
    def __init__(self, config: SupervisedRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SupervisedRefinerConfig()
        self.position_mlp = nn.Sequential(
            nn.Linear(3, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.nodal_displacement_mlp = nn.Sequential(
            nn.Linear(18, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.edge_von_mises_mlp = nn.Sequential(
            nn.Linear(6, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_heads),
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
        self.style_token_encoder = (
            StyleTokenEncoder(self.config) if self.config.use_style_token else None
        )
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)
        layer_modes = ("distance", "connectivity", "stress", "free")
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
        nodal_displacements: torch.Tensor,
        edge_von_mises: torch.Tensor,
        flow_times: torch.Tensor,
        position_noise_levels: torch.Tensor,
        adjacency_noise_levels: torch.Tensor,
        style_structures: Structures | None = None,
    ) -> FlowPrediction:
        structures.validate()
        if self.config.use_style_token and style_structures is not None:
            style_structures.validate()
        if target_stiffness.shape != (structures.batch_size, 6, 6):
            raise ValueError("target_stiffness must have shape [batch, 6, 6]")
        if current_stiffness.shape != (structures.batch_size, 6, 6):
            raise ValueError("current_stiffness must have shape [batch, 6, 6]")
        if nodal_displacements.shape != (
            structures.batch_size,
            structures.positions.shape[1],
            18,
        ):
            raise ValueError("nodal_displacements must have shape [batch, nodes, 18]")
        if edge_von_mises.shape != (
            structures.batch_size,
            structures.positions.shape[1],
            structures.positions.shape[1],
            6,
        ):
            raise ValueError("edge_von_mises must have shape [batch, nodes, nodes, 6]")
        if (
            self.config.use_style_token
            and style_structures is not None
            and (
                style_structures.positions.shape != structures.positions.shape
                or style_structures.adjacency.shape != structures.adjacency.shape
            )
        ):
            raise ValueError(
                "style_structures must match structures batch and node dimensions"
            )
        if (
            self.config.use_style_token
            and style_structures is not None
            and not torch.equal(style_structures.roles, structures.roles)
        ):
            raise ValueError("style_structures must use the same node roles")
        for name, value in (
            ("flow_times", flow_times),
            ("position_noise_levels", position_noise_levels),
            ("adjacency_noise_levels", adjacency_noise_levels),
        ):
            if value.shape != (structures.batch_size,):
                raise ValueError(f"{name} must have shape [batch]")

        positions = structures.positions
        roles = structures.roles
        current_adjacency = enforce_role_adjacency_constraints(
            structures.adjacency, roles
        )
        residual_stiffness = target_stiffness - current_stiffness
        hidden = self.position_mlp(positions) + self.nodal_displacement_mlp(
            nodal_displacements
        )
        hidden = hidden + self.role_embedding(roles)
        hidden = self.input_norm(hidden)
        edge_stress_conditioning = self.edge_von_mises_mlp(
            torch.log1p(edge_von_mises.clamp_min(0.0))
        ).permute(0, 3, 1, 2)

        mechanics_features = torch.cat(
            [
                symmetric_matrix_unique_values(target_stiffness),
                symmetric_matrix_unique_values(current_stiffness),
                symmetric_matrix_unique_values(residual_stiffness),
            ],
            dim=1,
        )
        hidden = hidden + self.mechanics_condition_mlp(mechanics_features)[:, None, :]
        hidden = (
            hidden
            + self.time_mlp(sinusoidal_embedding(flow_times, self.config.hidden_dim))[
                :, None, :
            ]
        )
        noise_features = torch.stack(
            [flow_times, position_noise_levels, adjacency_noise_levels],
            dim=1,
        )
        hidden = hidden + self.noise_condition_mlp(noise_features)[:, None, :]
        style_context = None
        if self.config.use_style_token and style_structures is not None:
            if self.style_token_encoder is None:
                raise RuntimeError("style token encoder is not initialized")
            style_context = self.style_token_encoder(style_structures)

        for layer in self.layers:
            hidden = layer(
                hidden,
                current_adjacency,
                positions,
                style_context,
                edge_stress_conditioning if layer.mode == "stress" else None,
            )

        hidden = self.final_norm(hidden)
        position_velocity = self.position_head(hidden)
        node_latents = self.node_latent_head(hidden)
        # Connectivity updates stay factored through node latents so the model
        # cannot bypass the intended graph-inductive bias with edge-wise heads.
        scores = torch.matmul(node_latents, node_latents.transpose(1, 2)) / math.sqrt(
            self.config.latent_dim
        )
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
        style_structures: Structures | None = None,
    ) -> Structures:
        source_structures.validate()
        if self.config.use_style_token and style_structures is not None:
            style_structures.validate()
        num_steps = num_steps or self.config.num_integration_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if (
            self.config.use_style_token
            and style_structures is not None
            and (
                style_structures.positions.shape != source_structures.positions.shape
                or style_structures.adjacency.shape != source_structures.adjacency.shape
            )
        ):
            raise ValueError(
                "style_structures must match source_structures batch and node dimensions"
            )
        if (
            self.config.use_style_token
            and style_structures is not None
            and not torch.equal(style_structures.roles, source_structures.roles)
        ):
            raise ValueError("style_structures must use the same node roles")

        current = source_structures
        initial_position_gap = current.positions.new_zeros(current.batch_size)
        initial_adjacency_gap = current.adjacency.new_zeros(current.batch_size)
        _, _, free_mask = role_masks(current.roles)
        free_mask = free_mask.unsqueeze(-1).to(dtype=current.positions.dtype)

        for step in range(num_steps):
            analyses = analysis_fn(current)
            if analyses.nodal_displacements is None:
                raise ValueError("analysis_fn must provide nodal_displacements")
            if analyses.edge_von_mises is None:
                raise ValueError("analysis_fn must provide edge_von_mises")
            flow_time = current.positions.new_full(
                (current.batch_size,),
                (step + 0.5) / num_steps,
            )
            remaining = 1.0 - flow_time
            prediction = self.predict_flow(
                structures=current,
                target_stiffness=target_stiffness,
                current_stiffness=analyses.generalized_stiffness,
                nodal_displacements=analyses.nodal_displacements,
                edge_von_mises=analyses.edge_von_mises,
                flow_times=flow_time,
                position_noise_levels=remaining * initial_position_gap,
                adjacency_noise_levels=remaining * initial_adjacency_gap,
                style_structures=style_structures,
            )
            step_size = 1.0 / num_steps
            positions = (
                current.positions + step_size * prediction.position_velocity * free_mask
            ).clamp(0.0, 1.0)
            adjacency = enforce_role_adjacency_constraints(
                (current.adjacency + step_size * prediction.adjacency_velocity).clamp(
                    0.0, 1.0
                ),
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
        style_structures: Structures | None = None,
    ) -> Structures:
        return self.rollout(
            source_structures=source_structures,
            target_stiffness=target_stiffness,
            analysis_fn=analysis_fn,
            num_steps=num_steps,
            style_structures=style_structures,
        )
