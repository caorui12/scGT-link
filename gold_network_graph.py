"""
Build a prior graph from a reference network CSV or from **training positives only**:

* CSV-based: bidirectional edges, ``edata['w']`` (min–max normalized), ``edata['dir']``
  (+1 TF→target, −1 mirror, 0 self).
* :func:`build_train_positive_pearson_graph`: edges only from ``Train_set`` Label=1 pairs;
  weights from Pearson correlation (same layout as ``gold_pearson`` via ``_triples_to_dgl``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import dgl


def _triples_to_dgl(
    triples: List[Tuple[int, int, float]],
    g_n: int,
    add_self_loops: bool,
) -> dgl.DGLGraph:
    """Shared layout for gold-style priors after triples (i, j, weight) are fixed."""
    if not triples:
        raise RuntimeError("Gold network: no edges mapped to gene indices.")

    imps = np.array([t[2] for t in triples], dtype=np.float64)
    lo, hi = float(imps.min()), float(imps.max())
    if hi - lo < 1e-12:
        norm = np.ones(len(triples), dtype=np.float32)
    else:
        norm = ((imps - lo) / (hi - lo)).astype(np.float32)

    pairs_w: dict[Tuple[int, int], float] = {}
    pairs_dir: dict[Tuple[int, int], float] = {}
    for (i, j, _), nw in zip(triples, norm):
        wij = float(max(float(nw), 1e-6))
        pairs_w[(i, j)] = max(pairs_w.get((i, j), 0.0), wij)
        pairs_dir[(i, j)] = 1.0
        pairs_w[(j, i)] = max(pairs_w.get((j, i), 0.0), wij)
        pairs_dir[(j, i)] = -1.0

    pair_set: Set[Tuple[int, int]] = set(pairs_w.keys())
    if add_self_loops:
        for i in range(g_n):
            pair_set.add((i, i))

    src = torch.tensor([p[0] for p in pair_set], dtype=torch.int64)
    dst = torch.tensor([p[1] for p in pair_set], dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=g_n)
    g = dgl.to_simple(g)

    esrc, edst = g.edges()
    s = esrc.cpu().numpy()
    d = edst.cpu().numpy()
    w_list: List[float] = []
    dir_list: List[float] = []
    for a, b in zip(s, d):
        ia, ib = int(a), int(b)
        if ia == ib:
            w_list.append(1.0)
            dir_list.append(0.0)
        else:
            w_list.append(pairs_w.get((ia, ib), 0.5))
            dir_list.append(pairs_dir.get((ia, ib), 0.0))
    g.edata["w"] = torch.tensor(w_list, dtype=torch.float32)
    g.edata["dir"] = torch.tensor(dir_list, dtype=torch.float32)
    return g


def _resolve_endpoints_by_symbol(
    row_tf,
    row_tg,
    gene_to_idx: dict[str, int],
) -> Tuple[int, int] | None:
    """Map CSV endpoints to gene indices by **symbol only** (case-insensitive); row indices are not accepted."""

    def one(val) -> int | None:
        s = str(val).strip()
        if s in gene_to_idx:
            return gene_to_idx[s]
        su = s.upper()
        if su in gene_to_idx:
            return gene_to_idx[su]
        return None

    ti = one(row_tf)
    tj = one(row_tg)
    if ti is None or tj is None or ti == tj:
        return None
    return ti, tj


def _resolve_edge_column_names(col_map: dict[str, str]) -> tuple[str, str]:
    """Map lowercase keys to actual CSV headers. Accepts TF/Target, Gene1/Gene2, Source/Target, etc."""
    pairs = (
        ("tf", "target"),
        ("gene1", "gene2"),
        ("source", "target"),
        ("regulator", "target"),
    )
    for a, b in pairs:
        if a in col_map and b in col_map:
            return col_map[a], col_map[b]
    raise ValueError(
        "Gold network CSV needs endpoint pairs such as (TF, Target) or (Gene1, Gene2); "
        f"found columns: {list(col_map.values())}"
    )


def build_gold_network_graph(
    path: Path | str,
    gene_names: list[str],
    *,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    Load directed TF→target edges from a CSV file and build a DGL graph with edge weights and ``dir``.

    **Endpoint columns** (case-insensitive), first column is the regulator/source, second is the target:

    - ``TF`` + ``Target``, or ``Gene1`` + ``Gene2``, or ``Source`` + ``Target``, or ``Regulator`` + ``Target``.

    Values must be **gene symbols** matching expression matrix row labels in ``gene_names``
    (case-insensitive); integer row indices are not used.

    **Optional weight column:** ``weight``, ``importance``, or ``w`` (case-insensitive). If absent,
    all edges use weight 1.0 before min–max normalization to ``edata['w']``.

    Rows that do not map to two distinct genes in ``gene_names`` are skipped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gold network file not found: {path}")

    df = pd.read_csv(path)
    col_map = {c.lower(): c for c in df.columns}
    c_tf, c_tg = _resolve_edge_column_names(col_map)
    wcol = np.ones(len(df), dtype=np.float64)
    for key in ("weight", "importance", "w"):
        if key in col_map:
            wcol = df[col_map[key]].astype(np.float64).values
            break

    gene_to_idx: dict[str, int] = {}
    for i, g in enumerate(gene_names):
        gs = str(g).strip()
        gene_to_idx[gs] = i
        gene_to_idx[gs.upper()] = i
    triples: List[Tuple[int, int, float]] = []
    n = len(df)
    for k in range(n):
        pair = _resolve_endpoints_by_symbol(
            df[c_tf].iloc[k],
            df[c_tg].iloc[k],
            gene_to_idx,
        )
        if pair is None:
            continue
        i, j = pair
        triples.append((i, j, float(max(wcol[k], 0.0))))

    g_n = len(gene_names)
    return _triples_to_dgl(triples, g_n, add_self_loops=add_self_loops)


def build_gold_pearson_graph(
    path: Path | str,
    gene_names: list[str],
    expr: torch.Tensor,
    *,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    Same **directed** gold edges as :func:`build_gold_network_graph` (**gene symbols** in CSV), but each TF→target
    edge is weighted by the **Pearson correlation** of expression between regulator and target across cells
    (``r_ij = r_ji`` numerically; :func:`_triples_to_dgl` assigns ``edata['dir']``: +1 on TF→target, −1 on mirror).

    ``expr``: (G, C) genes × cells, same layout as training.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gold network file not found: {path}")

    df = pd.read_csv(path)
    col_map = {c.lower(): c for c in df.columns}
    c_tf, c_tg = _resolve_edge_column_names(col_map)

    gene_to_idx: dict[str, int] = {}
    for i, g in enumerate(gene_names):
        gs = str(g).strip()
        gene_to_idx[gs] = i
        gene_to_idx[gs.upper()] = i

    gold_edges: List[Tuple[int, int]] = []
    n = len(df)
    for k in range(n):
        pair = _resolve_endpoints_by_symbol(
            df[c_tf].iloc[k],
            df[c_tg].iloc[k],
            gene_to_idx,
        )
        if pair is None:
            continue
        gold_edges.append(pair)

    if not gold_edges:
        raise RuntimeError("Gold+Pearson: no edges from CSV mapped to gene indices.")

    z = expr.detach().float().cpu().numpy()
    if z.shape[0] != len(gene_names):
        raise ValueError("expr rows must match len(gene_names).")
    r_full = np.corrcoef(z)
    np.nan_to_num(r_full, copy=False, nan=0.0)

    triples: List[Tuple[int, int, float]] = []
    for i, j in gold_edges:
        triples.append((i, j, float(r_full[i, j])))

    g_n = len(gene_names)
    return _triples_to_dgl(triples, g_n, add_self_loops=add_self_loops)


def build_train_positive_pearson_graph(
    train_edge_index: torch.Tensor,
    train_labels: torch.Tensor,
    expr: torch.Tensor,
    *,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    Prior graph edges = **only** ``Train_set`` rows with Label==1 (directed TF→target indices).

    Each edge weight is the Pearson correlation between regulator and target across cells
    (same numeric ``r`` field as :func:`build_gold_pearson_graph` before min–max inside
    :func:`_triples_to_dgl`). Duplicate directed pairs keep the larger ``r``.

    ``train_edge_index``: long ``(2, E)`` with gene row indices (same convention as split CSVs).
    ``expr``: ``(G, C)`` genes × cells.
    """
    ei = train_edge_index.detach().long().cpu()
    y = train_labels.detach().float().cpu().view(-1)
    pos = y > 0.5
    src = ei[0, pos]
    dst = ei[1, pos]
    if src.numel() == 0:
        raise RuntimeError("train_pos_pearson: no positive edges in Train_set.")

    z = expr.detach().float().cpu().numpy()
    g_n = z.shape[0]
    hi = int(ei.max().item())
    if hi >= g_n:
        raise ValueError(
            f"train_pos_pearson: edge index {hi} out of range for expr with {g_n} genes."
        )
    r_full = np.corrcoef(z)
    np.nan_to_num(r_full, copy=False, nan=0.0)

    best_r: dict[Tuple[int, int], float] = {}
    for a, b in zip(src.tolist(), dst.tolist()):
        i, j = int(a), int(b)
        if i == j:
            continue
        rij = float(r_full[i, j])
        prev = best_r.get((i, j))
        if prev is None or rij > prev:
            best_r[(i, j)] = rij

    if not best_r:
        raise RuntimeError(
            "train_pos_pearson: no usable directed edges (only self-loop positives?)."
        )

    triples = [(i, j, w) for (i, j), w in best_r.items()]
    return _triples_to_dgl(triples, g_n, add_self_loops=add_self_loops)
