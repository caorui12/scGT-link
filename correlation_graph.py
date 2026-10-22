"""Build a sparse gene graph from expression correlation (Pearson)."""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import torch
import dgl


def select_correlation_pairs(
    r_full: np.ndarray,
    *,
    top_k: int = 20,
    min_abs_corr: float | None = None,
    min_corr: float | None = None,
    abs_with_top_k: bool = False,
) -> Set[Tuple[int, int]]:
    """
    Choose undirected gene pairs from a full Pearson matrix ``r_full`` (G×G).

    Selection modes (priority):
      1. ``min_corr`` set: for each gene i, keep neighbors with ``r_ij > min_corr``,
         then take up to ``top_k`` by descending ``r_ij`` (signed). Adds both (i,j) and (j,i).
      2. ``min_abs_corr`` set with ``abs_with_top_k=True``: for each gene i, keep neighbors with
         ``|r_ij| >= min_abs_corr``, then take up to ``top_k`` by descending ``|r_ij|``.
      3. ``min_abs_corr`` set (legacy): keep all pairs with ``|r_ij| >= min_abs_corr``
         (ignores ``top_k``).
      4. else: for each gene i, keep ``top_k`` neighbors by ``|r_ij|``; add both directions.
    """
    g_n = int(r_full.shape[0])
    if g_n < 2:
        raise ValueError("Need at least 2 genes for correlation graph.")

    r_sel = np.array(r_full, dtype=np.float64, copy=True)
    np.nan_to_num(r_sel, copy=False, nan=0.0)
    np.fill_diagonal(r_sel, 0.0)

    pairs: Set[Tuple[int, int]] = set()
    if min_corr is not None:
        thr = float(min_corr)
        k_eff = max(0, min(int(top_k), g_n - 1))
        for i in range(g_n):
            row = r_sel[i]
            cand = np.where(row > thr)[0]
            if cand.size == 0 or k_eff <= 0:
                continue
            if cand.size > k_eff:
                part = np.argpartition(-row[cand], k_eff - 1)[:k_eff]
                cand = cand[part]
            for j in cand:
                ji = int(j)
                if ji == i:
                    continue
                pairs.add((i, ji))
                pairs.add((ji, i))
        return pairs

    if min_abs_corr is not None:
        thr = float(min_abs_corr)
        if thr < 0.0:
            raise ValueError(f"min_abs_corr must be >= 0, got {thr}")
        if abs_with_top_k:
            k_eff = max(0, min(int(top_k), g_n - 1))
            for i in range(g_n):
                abs_row = np.abs(r_sel[i])
                cand = np.where(abs_row >= thr)[0]
                if cand.size == 0 or k_eff <= 0:
                    continue
                if cand.size > k_eff:
                    part = np.argpartition(-abs_row[cand], k_eff - 1)[:k_eff]
                    cand = cand[part]
                for j in cand:
                    ji = int(j)
                    if ji == i:
                        continue
                    pairs.add((i, ji))
                    pairs.add((ji, i))
            return pairs
        for i in range(g_n):
            for j in range(g_n):
                if i == j:
                    continue
                if abs(float(r_sel[i, j])) >= thr:
                    pairs.add((int(i), int(j)))
        return pairs

    k_eff = min(int(top_k), g_n - 1)
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
    return pairs


def build_correlation_graph(
    expr: torch.Tensor,
    top_k: int = 20,
    *,
    min_abs_corr: float | None = None,
    min_corr: float | None = None,
    abs_with_top_k: bool = False,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    expr: (G, C) genes × cells (float).

    See :func:`select_correlation_pairs` for edge selection. Edge weights are Pearson ``r_ij``
    (self-loops use 1.0).
    """
    z = expr.detach().float().cpu().numpy()
    g_n = z.shape[0]
    r_full = np.corrcoef(z)
    np.nan_to_num(r_full, copy=False, nan=0.0)

    pairs = select_correlation_pairs(
        r_full,
        top_k=top_k,
        min_abs_corr=min_abs_corr,
        min_corr=min_corr,
        abs_with_top_k=abs_with_top_k,
    )

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
    esrc, edst = g.edges()
    s = esrc.cpu().numpy()
    d = edst.cpu().numpy()
    w = np.where(s == d, 1.0, r_full[s, d]).astype(np.float32)
    g.edata["w"] = torch.from_numpy(w)
    return g
