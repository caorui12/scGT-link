"""Directed sparse Graph Transformer (scGT-Link v2 default)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _grouped_softmax(logits: torch.Tensor, group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    group_max = torch.full(
        (n_groups, logits.size(1)),
        neg_inf,
        device=logits.device,
        dtype=logits.dtype,
    )
    idx = group_index.unsqueeze(1).expand_as(logits)
    group_max.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - group_max[group_index])
    group_sum = torch.zeros_like(group_max)
    group_sum.scatter_add_(0, idx, exp)
    return exp / (group_sum[group_index].clamp_min(1e-9))


class SparseMHA(nn.Module):
    """Sparse multi-head attention with dst aggregation and edge conditioning."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_dir_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_tf2t_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_msg_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        nh = self.num_heads
        dh = self.hidden_dim // nh
        src, dst = g.edges()
        e = src.numel()

        q = self.linear_q(h).reshape(n, dh, nh)
        k = self.linear_k(h).reshape(n, dh, nh)
        v = self.linear_v(h).reshape(n, dh, nh)

        w = g.edata["w"].to(device=h.device, dtype=h.dtype).reshape(-1)
        ddir = g.edata["dir"].to(device=h.device, dtype=h.dtype).reshape(-1)
        tf2t = g.edata["tf2t"].to(device=h.device, dtype=h.dtype).reshape(-1)

        edge_q = q[dst]
        edge_k = k[src]
        neighbor_v = v[src]
        group = dst

        edge_attention = (edge_q * edge_k).sum(dim=1) / (dh ** 0.5)
        edge_attention = edge_attention + self.edge_logit_scale * w.unsqueeze(-1)
        edge_attention = edge_attention + self.edge_dir_scale * ddir.unsqueeze(-1)
        edge_attention = edge_attention + self.edge_tf2t_scale * tf2t.unsqueeze(-1)

        attn_weights = self.attn_drop(_grouped_softmax(edge_attention, group, n))
        e_msg = self.edge_msg_mlp(torch.stack([w, ddir], dim=-1)).reshape(e, dh, nh)
        neighbor_v = neighbor_v + e_msg

        weighted = attn_weights.unsqueeze(1) * neighbor_v
        out = torch.zeros(n, dh, nh, device=h.device, dtype=h.dtype)
        out.index_add_(0, group, weighted)
        return self.out_proj(out.reshape(n, -1))


class GTLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = SparseMHA(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.dropout(self.attention(g, h)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class GraphTransformer(nn.Module):
    """Directed Graph Transformer over the train-positive prior."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.pos_linear = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, g, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(g, h)
        return h
