"""Build a sparse prior graph from GRNBoost2-style boosting (arboreto core), with importance as edge weights.

Uses arboreto's ``infer_partial_network`` + ``to_tf_matrix`` in a serial loop (no dask graph), which avoids
known ``from_delayed([])`` failures with some dask/dask-expr versions when calling ``grnboost2()`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import dgl


def resolve_tf_gene_names(
    data_dir: Path,
    split_dir: Path,
    gene_names: list[str],
    tf_genes_file: Optional[str],
) -> list[str]:
    """
    TF gene symbols for arboreto (must match expression index names).
    Priority: explicit file → data_dir/BL--TFs.txt (or TFs.txt) → TF indices from edge splits.
    """
    if tf_genes_file:
        p = Path(tf_genes_file)
        if not p.is_file():
            raise FileNotFoundError(f"--tf_genes_file not found: {p}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        names = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return names

    for fname in ("BL--TFs.txt", "TFs.txt", "TFlist.txt"):
        p = data_dir / fname
        if p.is_file():
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            names = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
            if names:
                return names

    tfs: Set[str] = set()
    for split_name in ("Train_set.csv", "Validation_set.csv", "Test_set.csv"):
        p = split_dir / split_name
        if not p.is_file():
            continue
        df = pd.read_csv(p, index_col=0)
        for idx in df["TF"].astype(np.int64).values:
            ii = int(idx)
            if 0 <= ii < len(gene_names):
                tfs.add(gene_names[ii])
    if tfs:
        return sorted(tfs)

    raise ValueError(
        "GRNBoost2 needs TF gene names: set --tf_genes_file, or add data_dir/BL--TFs.txt "
        "(one symbol per line), or ensure split CSVs list TF indices."
    )


def build_grnboost2_graph(
    expr: torch.Tensor,
    gene_names: list[str],
    tf_names: list[str],
    top_k: int = 20,
    seed: int = 42,
    grnboost2_limit: Optional[int] = None,
    n_estimators: Optional[int] = None,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    """
    expr: (G, C) genes × cells (same preprocessing as training).
    Serial GRNBoost2-style inference via arboreto (GradientBoostingRegressor per target gene).
    Keeps top_k incoming links per target by importance, mirrors edges, sets edata['w'] to min–max
    normalized importance; self-loops 1.0.

    ``grnboost2_limit``: if set, keep only this many rows globally after concat (by importance desc),
    before per-target top_k pruning.
    """
    try:
        from arboreto.core import (
            EARLY_STOP_WINDOW_LENGTH,
            SGBM_KWARGS,
            infer_partial_network,
            to_tf_matrix,
        )
    except ImportError as e:
        raise ImportError(
            "GRNBoost2 prior requires arboreto. Install: pip install arboreto"
        ) from e

    g_n = expr.shape[0]
    if g_n < 2:
        raise ValueError("Need at least 2 genes for GRNBoost2 graph.")

    expression_matrix = expr.detach().float().cpu().numpy().T
    assert expression_matrix.shape[1] == len(gene_names)

    reg_kwargs = dict(SGBM_KWARGS)
    if n_estimators is not None:
        reg_kwargs["n_estimators"] = int(n_estimators)

    tf_matrix, tf_matrix_gene_names = to_tf_matrix(
        expression_matrix, list(gene_names), tf_names
    )

    print(
        f"Building GRNBoost2 prior (serial arboreto, {g_n} targets; may take a long time)…",
        flush=True,
    )

    all_parts: list[pd.DataFrame] = []
    for ti, target_gene_name in enumerate(gene_names):
        if ti % 100 == 0 or ti == g_n - 1:
            print(f"  GRNBoost2 target gene {ti + 1}/{g_n} …", flush=True)
        y = expression_matrix[:, ti]
        try:
            links_df = infer_partial_network(
                "GBM",
                reg_kwargs,
                tf_matrix,
                tf_matrix_gene_names,
                target_gene_name,
                y,
                include_meta=False,
                early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                seed=seed,
            )
        except Exception:
            continue
        if isinstance(links_df, pd.DataFrame) and len(links_df) > 0:
            all_parts.append(links_df)

    if not all_parts:
        raise RuntimeError("GRNBoost2 produced no edges (all targets failed?)")

    network = pd.concat(all_parts, ignore_index=True)
    network = network.sort_values("importance", ascending=False)
    if grnboost2_limit is not None and grnboost2_limit > 0:
        network = network.head(int(grnboost2_limit))

    parts = []
    for tgt, sub in network.groupby("target"):
        parts.append(sub.head(min(top_k, len(sub))))
    pruned = pd.concat(parts, ignore_index=True)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    triples: list[Tuple[int, int, float]] = []
    for _, row in pruned.iterrows():
        tf = str(row["TF"])
        tg = str(row["target"])
        imp = float(row["importance"])
        if tf not in gene_to_idx or tg not in gene_to_idx:
            continue
        if tf == tg:
            continue
        ti, gi = gene_to_idx[tf], gene_to_idx[tg]
        triples.append((ti, gi, imp))

    if not triples:
        raise RuntimeError("No GRNBoost2 edges mapped to gene indices.")

    imps = np.array([t[2] for t in triples], dtype=np.float64)
    lo, hi = float(imps.min()), float(imps.max())
    if hi - lo < 1e-12:
        norm = np.ones_like(imps, dtype=np.float32)
    else:
        norm = ((imps - lo) / (hi - lo)).astype(np.float32)

    pairs_w: dict[Tuple[int, int], float] = {}
    for (i, j, _), nw in zip(triples, norm):
        wij = float(max(nw, 1e-6))
        pairs_w[(i, j)] = max(pairs_w.get((i, j), 0.0), wij)
        pairs_w[(j, i)] = max(pairs_w.get((j, i), 0.0), wij)

    pair_set: Set[Tuple[int, int]] = set(pairs_w.keys())

    if add_self_loops:
        for i in range(g_n):
            pair_set.add((i, i))

    if not pair_set:
        raise RuntimeError("Empty edge set after GRNBoost2.")

    src = torch.tensor([p[0] for p in pair_set], dtype=torch.int64)
    dst = torch.tensor([p[1] for p in pair_set], dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=g_n)
    g = dgl.to_simple(g)

    esrc, edst = g.edges()
    s = esrc.cpu().numpy()
    d = edst.cpu().numpy()
    w_list = []
    for a, b in zip(s, d):
        if a == b:
            w_list.append(1.0)
        else:
            w_list.append(pairs_w.get((int(a), int(b)), 0.5))
    g.edata["w"] = torch.tensor(w_list, dtype=torch.float32)
    return g
