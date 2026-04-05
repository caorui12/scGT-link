"""Build a sparse gene graph from expression correlation (Pearson)."""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import torch
import dgl


def build_correlation_graph(
    expr: torch.Tensor,
    top_k: int = 20,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    expr: (G, C) genes × cells (float).
    For each gene i, keep top_k neighbors j (j≠i) by |correlation(i,j)|; add undirected edges i↔j.
    """
    z = expr.detach().float().cpu().numpy()
    g_n = z.shape[0]
    if g_n < 2:
        raise ValueError("Need at least 2 genes for correlation graph.")

    r = np.corrcoef(z)
    np.nan_to_num(r, copy=False, nan=0.0)
    np.fill_diagonal(r, 0.0)

    k_eff = min(top_k, g_n - 1)
    pairs: Set[Tuple[int, int]] = set()
    for i in range(g_n):
        abs_row = np.abs(r[i]).copy()
        abs_row[i] = -np.inf
        if k_eff <= 0:
            break
        j_idx = np.argpartition(-abs_row, k_eff - 1)[:k_eff]
        for j in j_idx:
            if j == i:
                continue
            pairs.add((int(i), int(j)))
            pairs.add((int(j), int(i)))

    if add_self_loops:
        for i in range(g_n):
            pairs.add((i, i))

    if not pairs:
        # fallback: chain + self
        for i in range(g_n - 1):
            pairs.add((i, i + 1))
            pairs.add((i + 1, i))
        for i in range(g_n):
            pairs.add((i, i))

    src = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=g_n)
    g = dgl.to_simple(g)
    return g
