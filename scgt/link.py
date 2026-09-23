"""Concat MLP link predictor."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return self.decoder(torch.cat([src, dst], dim=-1))
