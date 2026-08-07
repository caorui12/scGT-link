"""Build a sparse gene graph from expression correlation (Pearson)."""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import torch
import dgl


def build_correlation_graph(
    expr: torch.Tensor,
    top_k: int = 20,
    *,
    min_abs_corr: float | None = None,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    expr: (G, C) genes × cells (float).

    * ``min_abs_corr`` set: undirected edges for all pairs with ``|r_ij| >= min_abs_corr``.
    * else: for each gene i, keep ``top_k`` neighbors j (j≠i) by |correlation(i,j)|.
    """
    z = expr.detach().float().cpu().numpy()
    g_n = z.shape[0]
    if g_n < 2:
        raise ValueError("Need at least 2 genes for correlation graph.")

    r_full = np.corrcoef(z)
    np.nan_to_num(r_full, copy=False, nan=0.0)
    # Selection uses off-diagonal only; keep r_full for edge weights (Pearson r_ij).
    r_sel = r_full.copy()
    np.fill_diagonal(r_sel, 0.0)

    pairs: Set[Tuple[int, int]] = set()
    if min_abs_corr is not None:
        thr = float(min_abs_corr)
        if thr < 0.0:
            raise ValueError(f"min_abs_corr must be >= 0, got {thr}")
        for i in range(g_n):
            for j in range(g_n):
                if i == j:
                    continue
                if abs(float(r_sel[i, j])) >= thr:
                    pairs.add((int(i), int(j)))
    else:
        k_eff = min(top_k, g_n - 1)
        for i in range(g_n):
            abs_row = np.abs(r_sel[i]).copy()
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
    # Pearson on each directed edge; self-loops use 1.0 (diagonal was cleared in r_sel).
    esrc, edst = g.edges()
    s = esrc.cpu().numpy()
    d = edst.cpu().numpy()
    w = np.where(s == d, 1.0, r_full[s, d]).astype(np.float32)
    g.edata["w"] = torch.from_numpy(w)
    return g
