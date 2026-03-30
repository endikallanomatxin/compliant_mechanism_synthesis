from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


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


class ConditionedDenoiser(nn.Module):
    def __init__(
        self, grid_size: int, patch_size: int, d_model: int, nhead: int, num_layers: int
    ) -> None:
        super().__init__()
        if grid_size % patch_size != 0:
            raise ValueError("grid_size must be divisible by patch_size")

        self.grid_size = grid_size
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size
        self.num_patches = (grid_size // patch_size) ** 2

        self.patch_in = nn.Linear(self.patch_dim, d_model)
        self.target_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.patch_out = nn.Linear(d_model, self.patch_dim)

    def _patchify(self, grids: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(grids, kernel_size=self.patch_size, stride=self.patch_size)
        return patches.transpose(1, 2)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        folded = F.fold(
            patches.transpose(1, 2),
            output_size=(self.grid_size, self.grid_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return folded

    def forward(
        self, noisy_grids: torch.Tensor, targets: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.patch_in(self._patchify(noisy_grids))
        target_cond = self.target_mlp(targets)[:, None, :]
        time_cond = self.time_mlp(sinusoidal_embedding(timesteps, tokens.shape[-1]))[
            :, None, :
        ]
        hidden = tokens + self.position + target_cond + time_cond
        hidden = self.encoder(hidden)
        hidden = self.norm(hidden)
        patches = self.patch_out(hidden)
        return self._unpatchify(patches)
