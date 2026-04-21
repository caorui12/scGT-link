"""
Build a prior graph from a user-provided reference network (CSV), same message-passing layout as GRNBoost2:
bidirectional edges, ``edata['w']`` (min–max normalized weights), ``edata['dir']`` (+1 TF→target, −1 mirror, 0 self).
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
    """Shared layout with ``build_grnboost2_graph`` after triples (i, j, weight) are fixed."""
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


def _resolve_endpoints(
    row_tf,
    row_tg,
    gene_names: list[str],
    gene_to_idx: dict[str, int],
    force_symbols: bool,
) -> Tuple[int, int] | None:
    def one(val) -> int | None:
        if not force_symbols:
            if isinstance(val, (int, np.integer)):
                ii = int(val)
                if 0 <= ii < len(gene_names):
                    return ii
            s = str(val).strip()
            if s.isdigit():
                ii = int(s)
                if 0 <= ii < len(gene_names):
                    return ii
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


def build_gold_network_graph(
    path: Path | str,
    gene_names: list[str],
    *,
    add_self_loops: bool = True,
    force_gene_symbols: bool = False,
) -> dgl.DGLGraph:
    """
    Load directed TF→target edges from a CSV file and build the same DGL graph as GRNBoost2-style priors.

    **Columns** (TF / Target matched case-insensitively, e.g. ``TF``, ``tf``, ``Target``):

    - Required: TF and Target — either **0-based gene indices** (same as split CSVs) or **gene
      symbols**; use ``force_gene_symbols=True`` to disable index parsing (symbols only).
    - Optional: ``weight``, ``importance``, ``Weight``, ``Importance``, or ``w`` — non-negative edge
      strength; if missing, all edges use weight 1.0 before min–max normalization.

    Rows that do not map to two distinct genes in ``gene_names`` are skipped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gold network file not found: {path}")

    df = pd.read_csv(path)
    col_map = {c.lower(): c for c in df.columns}
    for need in ("tf", "target"):
        if need not in col_map:
            raise ValueError(
                f"Gold network CSV must contain TF and Target columns; got {list(df.columns)}"
            )
    c_tf = col_map["tf"]
    c_tg = col_map["target"]
    wcol = None
    for col in ("weight", "importance", "Weight", "Importance", "w"):
        if col in df.columns:
            wcol = df[col].astype(np.float64).values
            break
    if wcol is None:
        wcol = np.ones(len(df), dtype=np.float64)

    gene_to_idx: dict[str, int] = {}
    for i, g in enumerate(gene_names):
        gs = str(g).strip()
        gene_to_idx[gs] = i
        gene_to_idx[gs.upper()] = i
    triples: List[Tuple[int, int, float]] = []
    n = len(df)
    for k in range(n):
        pair = _resolve_endpoints(
            df[c_tf].iloc[k],
            df[c_tg].iloc[k],
            gene_names,
            gene_to_idx,
            force_gene_symbols,
        )
        if pair is None:
            continue
        i, j = pair
        triples.append((i, j, float(max(wcol[k], 0.0))))

    g_n = len(gene_names)
    return _triples_to_dgl(triples, g_n, add_self_loops=add_self_loops)
