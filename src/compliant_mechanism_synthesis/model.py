from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from compliant_mechanism_synthesis.common import (
    NUM_ROLES,
    logits_to_adjacency,
    symmetrize_adjacency,
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


class GraphAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, conditioned: bool) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.conditioned = conditioned
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _ = hidden.shape
        normed = self.norm1(hidden)
        q = (
            self.q_proj(normed)
            .view(batch, num_nodes, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(normed)
            .view(batch, num_nodes, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(normed)
            .view(batch, num_nodes, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if self.conditioned:
            bias = 2.0 * adjacency[:, None, :, :] - 1.0
            logits = logits + bias

        weights = torch.softmax(logits, dim=-1)
        if self.conditioned:
            weights = weights * (0.25 + 0.75 * adjacency[:, None, :, :])
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch, num_nodes, self.d_model)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ff(self.norm2(hidden))
        return hidden


class GraphRefinementModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.position_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.role_embedding = nn.Embedding(NUM_ROLES, d_model)
        self.target_mlp = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [
                GraphAttentionBlock(d_model, nhead, conditioned=(idx % 2 == 0))
                for idx in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.displacement_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.node_latent_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )
        nn.init.zeros_(self.displacement_head[-1].weight)
        nn.init.zeros_(self.displacement_head[-1].bias)

    def forward(
        self,
        positions: torch.Tensor,
        roles: torch.Tensor,
        adjacency: torch.Tensor,
        targets: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.position_mlp(positions) + self.role_embedding(roles)
        hidden = self.input_norm(hidden)
        hidden = (
            hidden + self.target_mlp(targets.reshape(targets.shape[0], -1))[:, None, :]
        )
        hidden = (
            hidden
            + self.time_mlp(sinusoidal_embedding(timesteps, hidden.shape[-1]))[
                :, None, :
            ]
        )

        current_adjacency = symmetrize_adjacency(adjacency)
        for layer in self.layers:
            hidden = layer(hidden, current_adjacency)

        hidden = self.final_norm(hidden)
        displacements = self.displacement_head(hidden)
        node_latents = self.node_latent_head(hidden)
        scores = torch.matmul(node_latents, node_latents.transpose(1, 2)) / math.sqrt(
            node_latents.shape[-1]
        )
        delta_scores = symmetrize_adjacency(scores)
        predicted_adjacency = logits_to_adjacency(
            torch.logit(current_adjacency.clamp(1e-4, 1 - 1e-4)) + delta_scores
        )
        return {
            "displacements": displacements,
            "node_latents": node_latents,
            "delta_scores": delta_scores,
            "predicted_adjacency": predicted_adjacency,
        }
