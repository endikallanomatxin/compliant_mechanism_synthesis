from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from compliant_mechanism_synthesis.dataset.types import Analyses, Structures
from compliant_mechanism_synthesis.mechanics import normalize_generalized_stiffness
from compliant_mechanism_synthesis.roles import role_masks
from compliant_mechanism_synthesis.tensor_ops import (
    distance_affinity,
    enforce_role_adjacency_constraints,
    max_length_gate,
    symmetrize_matrix,
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


def _signed_log1p_features(values: torch.Tensor) -> torch.Tensor:
    safe_values = torch.nan_to_num(values, nan=0.0, posinf=1e30, neginf=-1e30)
    return torch.sign(safe_values) * torch.log1p(safe_values.abs())


@dataclass(frozen=True)
class FlowPrediction:
    position_velocity: torch.Tensor
    adjacency_velocity: torch.Tensor
    predicted_adjacency: torch.Tensor
    connectivity_latents: torch.Tensor
    style_context: torch.Tensor | None = None
    style_residual: torch.Tensor | None = None
    style_available: torch.Tensor | None = None
    style_mean: torch.Tensor | None = None
    style_logvar: torch.Tensor | None = None
    style_kl: torch.Tensor | None = None


@dataclass(frozen=True)
class StyleTokenDistribution:
    token: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    kl: torch.Tensor


@dataclass(frozen=True)
class SupervisedRefinerConfig:
    # Global
    hidden_dim: int = 1024
    num_attention_layers: int = 6
    num_heads: int = 16
    # Output
    connectivity_latent_dim: int = 128
    pair_edge_hidden_dim: int = 256
    pair_edge_logit_eps: float = 1e-4
    # Local bar angles nodal encoding
    local_incident_bar_limit: int = 5
    local_relation_hidden_dim: int = 32
    local_bar_hidden_dim: int = 64
    local_num_heads: int = 4
    local_pair_transformer_layers: int = 1
    local_bar_transformer_layers: int = 1
    # Inference
    num_integration_steps: int = 8
    # Units
    max_distance: float = 0.24
    transition_width: float = 0.08
    # Style token
    use_style_token: bool = True
    style_token_count: int = 1
    style_token_dropout: float = 0.1
    style_token_logvar_min: float = -6.0
    style_token_logvar_max: float = 2.0


class GraphAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_heads != 16:
            raise ValueError("GraphAttentionBlock requires exactly 16 heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hybrid_head_slices = (
            ("distance", slice(0, 4)),
            ("connectivity", slice(4, 8)),
            ("stress", slice(8, 12)),
            ("free", slice(12, 16)),
        )
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
        mode: str,
        adjacency: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | None:
        if mode == "free":
            return None
        if mode == "connectivity":
            return adjacency
        if mode == "stress":
            return None
        return distance_affinity(positions, length_scale=0.22)

    def _extend_nodal_conditioning(
        self,
        conditioning: torch.Tensor,
        context_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if context_tokens is None:
            return conditioning
        batch, num_nodes, _ = conditioning.shape
        context_count = context_tokens.shape[1]
        return torch.cat(
            [
                conditioning,
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

    def _extend_edge_conditioning(
        self,
        edge_head_conditioning: torch.Tensor,
        num_nodes: int,
        context_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if context_tokens is None:
            return edge_head_conditioning
        batch, num_heads, _, _ = edge_head_conditioning.shape
        context_count = context_tokens.shape[1]
        return torch.cat(
            [
                edge_head_conditioning,
                torch.zeros(
                    batch,
                    num_heads,
                    num_nodes,
                    context_count,
                    device=edge_head_conditioning.device,
                    dtype=edge_head_conditioning.dtype,
                ),
            ],
            dim=-1,
        )

    def _hybrid_attention(
        self,
        logits: torch.Tensor,
        adjacency: torch.Tensor,
        positions: torch.Tensor,
        context_tokens: torch.Tensor | None,
        edge_head_conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        logits_bias = torch.zeros_like(logits)
        weight_scale = torch.ones_like(logits)
        for mode, head_slice in self.hybrid_head_slices:
            conditioning = self._conditioning_matrix(mode, adjacency, positions)
            if conditioning is not None:
                extended_conditioning = self._extend_nodal_conditioning(
                    conditioning,
                    context_tokens,
                )
                logits_bias[:, head_slice] = (
                    2.0 * extended_conditioning[:, None, :, :] - 1.0
                )
                weight_scale[:, head_slice] = (
                    0.25 + 0.75 * extended_conditioning[:, None]
                )
        if edge_head_conditioning is not None:
            if edge_head_conditioning.shape[1] != 4:
                raise ValueError("hybrid stress conditioning must have 4 head channels")
            extended_edge_conditioning = self._extend_edge_conditioning(
                edge_head_conditioning,
                logits.shape[2],
                context_tokens,
            )
            logits_bias[:, 8:12] = logits_bias[:, 8:12] + extended_edge_conditioning
        weights = torch.softmax(logits + logits_bias, dim=-1)
        weights = weights * weight_scale
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights

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
        weights = self._hybrid_attention(
            logits,
            adjacency,
            positions,
            context_tokens,
            edge_head_conditioning,
        )

        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, num_nodes, self.hidden_dim)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ff(self.norm2(hidden))
        return hidden


def _cross2d_abs(vectors_a: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
    return (
        vectors_a[..., 0] * vectors_b[..., 1] - vectors_a[..., 1] * vectors_b[..., 0]
    ).abs()


def _incident_bar_pair_features(
    directions_2d: torch.Tensor,
    lengths: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    direction_a = directions_2d.unsqueeze(-2)
    direction_b = directions_2d.unsqueeze(-3)
    dot_products = (direction_a * direction_b).sum(dim=-1).clamp(-1.0, 1.0)
    sin_magnitudes = _cross2d_abs(direction_a, direction_b).clamp(0.0, 1.0)
    length_a = lengths.unsqueeze(-1).expand_as(dot_products)
    length_b = lengths.unsqueeze(-2).expand_as(dot_products)
    weight_a = weights.unsqueeze(-1).expand_as(dot_products)
    weight_b = weights.unsqueeze(-2).expand_as(dot_products)
    return torch.stack(
        [dot_products, sin_magnitudes, length_a, length_b, weight_a, weight_b],
        dim=-1,
    )


class LocalNodeGeometryEncoder(nn.Module):
    def __init__(self, config: SupervisedRefinerConfig) -> None:
        super().__init__()
        if config.local_relation_hidden_dim % config.local_num_heads != 0:
            raise ValueError(
                "local_relation_hidden_dim must be divisible by local_num_heads"
            )
        if config.local_bar_hidden_dim % config.local_num_heads != 0:
            raise ValueError(
                "local_bar_hidden_dim must be divisible by local_num_heads"
            )
        if config.local_incident_bar_limit <= 0:
            raise ValueError("local_incident_bar_limit must be positive")
        self.max_incident_bars = config.local_incident_bar_limit
        self.relation_hidden_dim = config.local_relation_hidden_dim
        self.bar_hidden_dim = config.local_bar_hidden_dim
        self.bar_feature_mlp = nn.Sequential(
            nn.Linear(4, self.relation_hidden_dim),
            nn.GELU(),
            nn.Linear(self.relation_hidden_dim, self.relation_hidden_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(6, self.relation_hidden_dim),
            nn.GELU(),
            nn.Linear(self.relation_hidden_dim, self.relation_hidden_dim),
        )
        self.bar_up_proj = nn.Sequential(
            nn.Linear(self.relation_hidden_dim, self.bar_hidden_dim),
            nn.GELU(),
            nn.Linear(self.bar_hidden_dim, self.bar_hidden_dim),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=self.relation_hidden_dim,
            nhead=config.local_num_heads,
            dim_feedforward=self.relation_hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            relation_layer,
            num_layers=config.local_pair_transformer_layers,
            enable_nested_tensor=False,
        )
        bar_layer = nn.TransformerEncoderLayer(
            d_model=self.bar_hidden_dim,
            nhead=config.local_num_heads,
            dim_feedforward=self.bar_hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bar_encoder = nn.TransformerEncoder(
            bar_layer,
            num_layers=config.local_bar_transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.bar_hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def _pool_encoded_set(
        self,
        encoded_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid_mask.to(dtype=encoded_tokens.dtype).unsqueeze(-1)
        return (encoded_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _encode_relation_sets(
        self,
        relation_tokens: torch.Tensor,
        relation_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, num_bars, _, hidden_dim = relation_tokens.shape
        flat_tokens = relation_tokens.reshape(
            batch_size * num_nodes * num_bars, num_bars, hidden_dim
        )
        flat_valid = relation_valid.reshape(batch_size * num_nodes * num_bars, num_bars)
        if flat_valid.shape[1] == 0:
            return flat_tokens.new_zeros(batch_size, num_nodes, num_bars, hidden_dim)
        empty_rows = ~flat_valid.any(dim=1)
        flat_tokens = flat_tokens.clone()
        flat_valid = flat_valid.clone()
        flat_valid[empty_rows, 0] = True
        flat_tokens[empty_rows, 0] = 0.0
        encoded = self.relation_encoder(
            flat_tokens,
            src_key_padding_mask=~flat_valid,
        )
        pooled = self._pool_encoded_set(encoded, flat_valid)
        return pooled.view(batch_size, num_nodes, num_bars, hidden_dim)

    def _encode_bar_set(
        self,
        bar_tokens: torch.Tensor,
        bar_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, num_bars, hidden_dim = bar_tokens.shape
        flat_tokens = bar_tokens.reshape(batch_size * num_nodes, num_bars, hidden_dim)
        flat_valid = bar_valid.reshape(batch_size * num_nodes, num_bars)
        if flat_valid.shape[1] == 0:
            return flat_tokens.new_zeros(batch_size, num_nodes, hidden_dim)
        empty_rows = ~flat_valid.any(dim=1)
        flat_tokens = flat_tokens.clone()
        flat_valid = flat_valid.clone()
        flat_valid[empty_rows, 0] = True
        flat_tokens[empty_rows, 0] = 0.0
        encoded = self.bar_encoder(flat_tokens, src_key_padding_mask=~flat_valid)
        pooled = self._pool_encoded_set(encoded, flat_valid)
        return pooled.view(batch_size, num_nodes, hidden_dim)

    def forward(self, positions: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, spatial_dim = positions.shape
        if spatial_dim != 3:
            raise ValueError("LocalNodeGeometryEncoder expects 3D positions")
        if adjacency.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError("adjacency must match positions batch and node dimensions")
        if num_nodes <= 1:
            return positions.new_zeros(
                batch_size, num_nodes, self.output_proj[-1].out_features
            )

        num_bars = min(self.max_incident_bars, num_nodes - 1)
        edge_weights = adjacency.clamp_min(0.0)
        edge_weights = edge_weights.masked_fill(
            torch.eye(num_nodes, device=adjacency.device, dtype=torch.bool).unsqueeze(
                0
            ),
            0.0,
        )
        top_weights, top_indices = torch.topk(edge_weights, k=num_bars, dim=-1)
        bar_valid = top_weights > 0.0
        batch_index = torch.arange(batch_size, device=positions.device)[:, None, None]
        neighbor_positions = positions[batch_index, top_indices]
        node_positions = positions[:, :, None, :]
        bar_vectors = neighbor_positions - node_positions
        bar_lengths = torch.linalg.vector_norm(bar_vectors, dim=-1)
        planar_vectors = bar_vectors[..., :2]
        planar_lengths = torch.linalg.vector_norm(planar_vectors, dim=-1, keepdim=True)
        safe_planar_lengths = planar_lengths.clamp_min(1e-6)
        directions_2d = planar_vectors / safe_planar_lengths
        directions_2d = directions_2d * bar_valid.unsqueeze(-1)
        bar_lengths = bar_lengths * bar_valid
        top_weights = top_weights * bar_valid

        pair_features = _incident_bar_pair_features(
            directions_2d, bar_lengths, top_weights
        )
        relation_tokens = self.pair_mlp(pair_features)
        eye_mask = torch.eye(num_bars, device=adjacency.device, dtype=torch.bool).view(
            1, 1, num_bars, num_bars
        )
        relation_valid = bar_valid.unsqueeze(-1) & bar_valid.unsqueeze(-2) & ~eye_mask
        relation_summary = self._encode_relation_sets(relation_tokens, relation_valid)

        bar_features = torch.cat(
            [directions_2d, bar_lengths.unsqueeze(-1), top_weights.unsqueeze(-1)],
            dim=-1,
        )
        bar_tokens = self.bar_feature_mlp(bar_features) + relation_summary
        bar_tokens = self.bar_up_proj(bar_tokens) * bar_valid.unsqueeze(-1)
        node_summary = self._encode_bar_set(bar_tokens, bar_valid)
        return self.output_proj(node_summary)


def _symmetric_pair_edge_features(
    connectivity_latents: torch.Tensor,
    positions: torch.Tensor,
    current_adjacency: torch.Tensor,
) -> torch.Tensor:
    latent_i = connectivity_latents.unsqueeze(2)
    latent_j = connectivity_latents.unsqueeze(1)
    position_deltas = positions.unsqueeze(2) - positions.unsqueeze(1)
    pair_distance = torch.linalg.vector_norm(position_deltas, dim=-1, keepdim=True)
    latent_dot_product = (latent_i * latent_j).sum(dim=-1, keepdim=True)
    return torch.cat(
        [
            latent_i + latent_j,
            torch.abs(latent_i - latent_j),
            latent_i * latent_j,
            latent_dot_product,
            pair_distance,
            current_adjacency.unsqueeze(-1),
        ],
        dim=-1,
    )


class StyleTokenEncoder(nn.Module):
    def __init__(
        self,
        config: SupervisedRefinerConfig,
        position_mlp: nn.Module,
        nodal_displacement_mlp: nn.Module,
        edge_von_mises_mlp: nn.Module,
        role_embedding: nn.Embedding,
        mechanics_condition_mlp: nn.Module,
        local_geometry_encoder: nn.Module,
    ) -> None:
        super().__init__()
        style_hidden_dim = max(config.num_heads, config.hidden_dim // 2)
        if style_hidden_dim % config.num_heads != 0:
            style_hidden_dim = config.num_heads * max(
                1, style_hidden_dim // config.num_heads
            )
        self.hidden_dim = style_hidden_dim
        self.shared_hidden_dim = config.hidden_dim
        self.position_mlp = position_mlp
        self.nodal_displacement_mlp = nodal_displacement_mlp
        self.edge_von_mises_mlp = edge_von_mises_mlp
        self.role_embedding = role_embedding
        self.mechanics_condition_mlp = mechanics_condition_mlp
        self.local_geometry_encoder = local_geometry_encoder
        self.stem_down_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, style_hidden_dim),
            nn.GELU(),
            nn.Linear(style_hidden_dim, style_hidden_dim),
        )
        self.style_token_count = config.style_token_count
        self.input_norm = nn.LayerNorm(style_hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphAttentionBlock(
                    hidden_dim=style_hidden_dim,
                    num_heads=config.num_heads,
                )
                for index in range(max(1, config.num_attention_layers // 2))
            ]
        )
        self.final_norm = nn.LayerNorm(style_hidden_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(style_hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.token_logvar_proj = nn.Sequential(
            nn.Linear(style_hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.token_seed = nn.Parameter(
            torch.zeros(1, config.style_token_count, style_hidden_dim)
        )
        self.dropout = nn.Dropout(config.style_token_dropout)
        self.logvar_min = config.style_token_logvar_min
        self.logvar_max = config.style_token_logvar_max
        nn.init.normal_(self.token_seed, mean=0.0, std=0.02)

    def forward(
        self,
        structures: Structures,
        analyses: Analyses,
        target_stiffness: torch.Tensor,
    ) -> StyleTokenDistribution:
        if analyses.nodal_displacements is None:
            raise ValueError("style analyses must provide nodal_displacements")
        if analyses.edge_von_mises is None:
            raise ValueError("style analyses must provide edge_von_mises")
        shared_hidden = self.position_mlp(
            structures.positions
        ) + self.nodal_displacement_mlp(
            _signed_log1p_features(analyses.nodal_displacements)
        )
        shared_hidden = shared_hidden + self.local_geometry_encoder(
            structures.positions,
            structures.adjacency,
        )
        shared_hidden = shared_hidden + self.role_embedding(structures.roles)
        normalized_target_stiffness = normalize_generalized_stiffness(target_stiffness)
        normalized_current_stiffness = normalize_generalized_stiffness(
            analyses.generalized_stiffness
        )
        normalized_residual_stiffness = (
            normalized_target_stiffness - normalized_current_stiffness
        )
        mechanics_features = torch.cat(
            [
                symmetric_matrix_unique_values(normalized_target_stiffness),
                symmetric_matrix_unique_values(normalized_current_stiffness),
                symmetric_matrix_unique_values(normalized_residual_stiffness),
            ],
            dim=1,
        )
        shared_hidden = (
            shared_hidden + self.mechanics_condition_mlp(mechanics_features)[:, None, :]
        )
        hidden = self.input_norm(self.stem_down_proj(shared_hidden))
        edge_stress_conditioning = self.edge_von_mises_mlp(
            torch.log1p(
                torch.nan_to_num(
                    analyses.edge_von_mises,
                    nan=0.0,
                    posinf=1e30,
                    neginf=0.0,
                ).clamp_min(0.0)
            )
        ).permute(0, 3, 1, 2)
        for layer in self.layers:
            hidden = layer(
                hidden,
                structures.adjacency,
                structures.positions,
                edge_head_conditioning=edge_stress_conditioning,
            )
        hidden = self.final_norm(hidden)
        pooled_hidden = hidden.mean(dim=1, keepdim=True).expand(
            -1, self.style_token_count, -1
        )
        pooled_hidden = pooled_hidden + self.token_seed
        mean = self.dropout(self.token_proj(pooled_hidden))
        logvar = self.token_logvar_proj(pooled_hidden).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        if self.training:
            std = torch.exp(0.5 * logvar)
            token = mean + std * torch.randn_like(std)
        else:
            token = mean
        kl = -0.5 * (1.0 + logvar - mean.square() - torch.exp(logvar)).mean(dim=(1, 2))
        return StyleTokenDistribution(token=token, mean=mean, logvar=logvar, kl=kl)


def load_refiner_state_dict_compatible(
    model: "SupervisedRefiner",
    state_dict: dict[str, torch.Tensor],
) -> None:
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    allowed_missing: set[str] = set()
    allowed_unexpected: set[str] = set()
    if model.config.use_style_token:
        allowed_missing.update(
            {
                "style_token_encoder.token_seed",
                "style_base_token",
                "style_availability_embedding.weight",
            }
        )
    if missing_keys - allowed_missing:
        raise RuntimeError(
            f"checkpoint is missing unsupported parameters: {sorted(missing_keys)}"
        )
    if unexpected_keys - allowed_unexpected:
        raise RuntimeError(
            f"checkpoint contains unexpected parameters: {sorted(unexpected_keys)}"
        )


class SupervisedRefiner(nn.Module):
    def __init__(self, config: SupervisedRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SupervisedRefinerConfig()
        if not (0.0 < self.config.pair_edge_logit_eps < 0.5):
            raise ValueError("pair_edge_logit_eps must be in (0, 0.5)")
        if self.config.style_token_count <= 0:
            raise ValueError("style_token_count must be positive")
        self.position_mlp = nn.Sequential(
            nn.Linear(3, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.nodal_displacement_mlp = nn.Sequential(
            nn.Linear(18, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.edge_von_mises_mlp = nn.Sequential(
            nn.Linear(6, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 4),
        )
        self.local_geometry_encoder = LocalNodeGeometryEncoder(self.config)
        self.style_local_geometry_encoder = LocalNodeGeometryEncoder(self.config)
        self.role_embedding = nn.Embedding(3, self.config.hidden_dim)
        self.mechanics_condition_mlp = nn.Sequential(
            nn.Linear(63, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.style_token_encoder = (
            StyleTokenEncoder(
                self.config,
                position_mlp=self.position_mlp,
                nodal_displacement_mlp=self.nodal_displacement_mlp,
                edge_von_mises_mlp=self.edge_von_mises_mlp,
                role_embedding=self.role_embedding,
                mechanics_condition_mlp=self.mechanics_condition_mlp,
                local_geometry_encoder=self.style_local_geometry_encoder,
            )
            if self.config.use_style_token
            else None
        )
        self.style_base_token = (
            nn.Parameter(
                torch.zeros(1, self.config.style_token_count, self.config.hidden_dim)
            )
            if self.config.use_style_token
            else None
        )
        self.style_availability_embedding = (
            nn.Embedding(2, self.config.hidden_dim)
            if self.config.use_style_token
            else None
        )
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphAttentionBlock(
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                )
                for index in range(self.config.num_attention_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_dim)
        self.position_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 3),
        )
        self.connectivity_latent_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.connectivity_latent_dim),
        )
        pair_edge_input_dim = 3 * self.config.connectivity_latent_dim + 3
        self.pair_edge_mlp = nn.Sequential(
            nn.Linear(pair_edge_input_dim, self.config.pair_edge_hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.config.pair_edge_hidden_dim, self.config.pair_edge_hidden_dim
            ),
            nn.GELU(),
            nn.Linear(self.config.pair_edge_hidden_dim, 1),
        )
        if self.style_base_token is not None:
            nn.init.normal_(self.style_base_token, mean=0.0, std=0.02)
        if self.style_availability_embedding is not None:
            nn.init.normal_(
                self.style_availability_embedding.weight, mean=0.0, std=0.02
            )
        nn.init.normal_(self.position_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.position_head[-1].bias)

    def _style_conditioning(
        self,
        *,
        structures: Structures,
        target_stiffness: torch.Tensor,
        style_structures: Structures | None,
        style_analyses: Analyses | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.config.use_style_token:
            return None, None, None, None, None, None
        if self.style_base_token is None or self.style_availability_embedding is None:
            raise RuntimeError("style conditioning parameters are not initialized")

        batch_size = structures.batch_size
        style_context = self.style_base_token.expand(batch_size, -1, -1)
        availability_index = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=structures.positions.device,
        )
        if style_structures is not None:
            availability_index.fill_(1)
        availability_embedding = self.style_availability_embedding(availability_index)
        availability_context = availability_embedding[:, None, :].expand(
            -1,
            self.config.style_token_count,
            -1,
        )
        style_available = availability_index.to(dtype=structures.positions.dtype)[
            :, None, None
        ]
        style_available = style_available.expand(-1, self.config.style_token_count, 1)
        style_context = style_context + availability_context
        style_residual = style_context.new_zeros(style_context.shape)
        style_mean = style_context.new_zeros(style_context.shape)
        style_logvar = style_context.new_zeros(style_context.shape)
        style_kl = style_context.new_zeros((batch_size,))
        if style_structures is not None:
            if self.style_token_encoder is None:
                raise RuntimeError("style token encoder is not initialized")
            if style_analyses is None:
                raise RuntimeError("style analyses are required for style conditioning")
            style_distribution = self.style_token_encoder(
                structures=style_structures,
                analyses=style_analyses,
                target_stiffness=target_stiffness,
            )
            style_residual = style_distribution.token
            style_mean = style_distribution.mean
            style_logvar = style_distribution.logvar
            style_kl = style_distribution.kl
            style_context = style_context + style_available * style_residual
        return (
            style_context,
            style_residual,
            style_available,
            style_mean,
            style_logvar,
            style_kl,
        )

    def predict_flow(
        self,
        structures: Structures,
        target_stiffness: torch.Tensor,
        current_stiffness: torch.Tensor,
        nodal_displacements: torch.Tensor,
        edge_von_mises: torch.Tensor,
        flow_times: torch.Tensor,
        style_structures: Structures | None = None,
        style_analyses: Analyses | None = None,
    ) -> FlowPrediction:
        structures.validate()
        if self.config.use_style_token and style_structures is not None:
            style_structures.validate()
        if self.config.use_style_token and style_analyses is not None:
            style_analyses.validate(structures.batch_size)
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
            and style_analyses is None
        ):
            raise ValueError("style_analyses must be provided with style_structures")
        if (
            self.config.use_style_token
            and style_structures is None
            and style_analyses is not None
        ):
            raise ValueError("style_structures must be provided with style_analyses")
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
        if flow_times.shape != (structures.batch_size,):
            raise ValueError("flow_times must have shape [batch]")

        positions = structures.positions
        roles = structures.roles
        current_adjacency = enforce_role_adjacency_constraints(
            structures.adjacency, roles
        )
        normalized_target_stiffness = normalize_generalized_stiffness(target_stiffness)
        normalized_current_stiffness = normalize_generalized_stiffness(
            current_stiffness
        )
        normalized_residual_stiffness = (
            normalized_target_stiffness - normalized_current_stiffness
        )
        hidden = self.position_mlp(positions) + self.nodal_displacement_mlp(
            _signed_log1p_features(nodal_displacements)
        )
        hidden = hidden + self.local_geometry_encoder(positions, current_adjacency)
        hidden = hidden + self.role_embedding(roles)
        hidden = self.input_norm(hidden)
        edge_stress_conditioning = self.edge_von_mises_mlp(
            torch.log1p(
                torch.nan_to_num(
                    edge_von_mises, nan=0.0, posinf=1e30, neginf=0.0
                ).clamp_min(0.0)
            )
        ).permute(0, 3, 1, 2)

        mechanics_features = torch.cat(
            [
                symmetric_matrix_unique_values(normalized_target_stiffness),
                symmetric_matrix_unique_values(normalized_current_stiffness),
                symmetric_matrix_unique_values(normalized_residual_stiffness),
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
        (
            style_context,
            style_residual,
            style_available,
            style_mean,
            style_logvar,
            style_kl,
        ) = self._style_conditioning(
            structures=structures,
            target_stiffness=target_stiffness,
            style_structures=style_structures,
            style_analyses=style_analyses,
        )

        for layer in self.layers:
            hidden = layer(
                hidden,
                current_adjacency,
                positions,
                style_context,
                edge_stress_conditioning,
            )

        hidden = self.final_norm(hidden)
        position_velocity = self.position_head(hidden)
        connectivity_latents = self.connectivity_latent_head(hidden)
        # Connectivity updates stay grounded in node latents, but use a
        # symmetric pair scorer and update adjacency in logit space.
        pair_features = _symmetric_pair_edge_features(
            connectivity_latents,
            positions,
            current_adjacency,
        )
        delta_logit = self.pair_edge_mlp(pair_features).squeeze(-1)
        delta_logit = symmetrize_matrix(delta_logit)
        update_gate = max_length_gate(
            positions,
            max_distance=self.config.max_distance,
            transition_width=self.config.transition_width,
        )
        current_logits = torch.logit(
            current_adjacency.clamp(
                self.config.pair_edge_logit_eps,
                1.0 - self.config.pair_edge_logit_eps,
            )
        )
        updated_logits = current_logits + delta_logit * update_gate
        predicted_adjacency = enforce_role_adjacency_constraints(
            torch.sigmoid(updated_logits),
            roles,
        )
        adjacency_velocity = predicted_adjacency - current_adjacency
        return FlowPrediction(
            position_velocity=position_velocity,
            adjacency_velocity=adjacency_velocity,
            predicted_adjacency=predicted_adjacency,
            connectivity_latents=connectivity_latents,
            style_context=style_context,
            style_residual=style_residual,
            style_available=style_available,
            style_mean=style_mean,
            style_logvar=style_logvar,
            style_kl=style_kl,
        )

    def rollout_trajectory(
        self,
        source_structures: Structures,
        target_stiffness: torch.Tensor,
        analysis_fn,
        num_steps: int | None = None,
        style_structures: Structures | None = None,
        style_analyses: Analyses | None = None,
    ) -> list[Structures]:
        source_structures.validate()
        if self.config.use_style_token and style_structures is not None:
            style_structures.validate()
        if self.config.use_style_token and style_analyses is not None:
            style_analyses.validate(source_structures.batch_size)
        num_steps = num_steps or self.config.num_integration_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if (
            self.config.use_style_token
            and style_structures is not None
            and style_analyses is None
        ):
            style_analyses = analysis_fn(style_structures)
        if (
            self.config.use_style_token
            and style_structures is None
            and style_analyses is not None
        ):
            raise ValueError("style_structures must be provided with style_analyses")
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
        trajectory = [current]
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
            prediction = self.predict_flow(
                structures=current,
                target_stiffness=target_stiffness,
                current_stiffness=analyses.generalized_stiffness,
                nodal_displacements=analyses.nodal_displacements,
                edge_von_mises=analyses.edge_von_mises,
                flow_times=flow_time,
                style_structures=style_structures,
                style_analyses=style_analyses,
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
            trajectory.append(current)
        trajectory[-1].validate()
        return trajectory

    def rollout(
        self,
        source_structures: Structures,
        target_stiffness: torch.Tensor,
        analysis_fn,
        num_steps: int | None = None,
        style_structures: Structures | None = None,
        style_analyses: Analyses | None = None,
    ) -> Structures:
        return self.rollout_trajectory(
            source_structures=source_structures,
            target_stiffness=target_stiffness,
            analysis_fn=analysis_fn,
            num_steps=num_steps,
            style_structures=style_structures,
            style_analyses=style_analyses,
        )[-1]

    def forward(
        self,
        source_structures: Structures,
        target_stiffness: torch.Tensor,
        analysis_fn,
        num_steps: int | None = None,
        style_structures: Structures | None = None,
        style_analyses: Analyses | None = None,
    ) -> Structures:
        return self.rollout(
            source_structures=source_structures,
            target_stiffness=target_stiffness,
            analysis_fn=analysis_fn,
            num_steps=num_steps,
            style_structures=style_structures,
            style_analyses=style_analyses,
        )
