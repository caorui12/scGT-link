"""Node encoder: proj(scGPT) + LapPE."""

from __future__ import annotations

import torch
import torch.nn as nn


class ScGPTEncoder(nn.Module):
    """Frozen scGPT gene embeddings projected to d_model, plus Laplacian PE."""

    def __init__(self, scgpt_gene_emb: torch.Tensor, d_model: int):
        super().__init__()
        if scgpt_gene_emb.dim() != 2:
            raise ValueError(f"scgpt_gene_emb must be (G, D), got {tuple(scgpt_gene_emb.shape)}")
        self.d_model = d_model
        self.register_buffer("scgpt_gene_emb", scgpt_gene_emb.detach().float())
        self.proj = nn.Linear(scgpt_gene_emb.shape[1], d_model)

    def forward(self, structural_pe: torch.Tensor) -> torch.Tensor:
        return self.proj(self.scgpt_gene_emb.to(dtype=structural_pe.dtype)) + structural_pe
