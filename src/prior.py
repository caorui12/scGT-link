"""Train-positive Pearson prior graph (scGT-Link v2 default)."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import dgl
import numpy as np
import torch


def _triples_to_dgl(
    triples: List[Tuple[int, int, float]],
    n_genes: int,
    add_self_loops: bool = True,
) -> dgl.DGLGraph:
    imps = np.array([t[2] for t in triples], dtype=np.float64)
    lo, hi = float(imps.min()), float(imps.max())
    if hi - lo < 1e-12:
        norm = np.ones(len(triples), dtype=np.float32)
    else:
        norm = ((imps - lo) / (hi - lo)).astype(np.float32)

    pairs_w: Dict[Tuple[int, int], float] = {}
    pairs_dir: Dict[Tuple[int, int], float] = {}
    for (i, j, _), nw in zip(triples, norm):
        wij = float(max(float(nw), 1e-6))
        pairs_w[(i, j)] = max(pairs_w.get((i, j), 0.0), wij)
        pairs_dir[(i, j)] = 1.0
        pairs_w[(j, i)] = max(pairs_w.get((j, i), 0.0), wij)
        pairs_dir[(j, i)] = -1.0

    pair_set: Set[Tuple[int, int]] = set(pairs_w.keys())
    if add_self_loops:
        for i in range(n_genes):
            pair_set.add((i, i))

    src = torch.tensor([p[0] for p in pair_set], dtype=torch.int64)
    dst = torch.tensor([p[1] for p in pair_set], dtype=torch.int64)
    g = dgl.to_simple(dgl.graph((src, dst), num_nodes=n_genes))

    esrc, edst = g.edges()
    w_list: List[float] = []
    dir_list: List[float] = []
    tf2t_list: List[float] = []
    for a, b in zip(esrc.cpu().tolist(), edst.cpu().tolist()):
        if a == b:
            w_list.append(1.0)
            dir_list.append(0.0)
            tf2t_list.append(0.0)
        else:
            dval = float(pairs_dir.get((a, b), 0.0))
            w_list.append(pairs_w.get((a, b), 0.5))
            dir_list.append(dval)
            tf2t_list.append(1.0 if dval > 0.0 else 0.0)
    g.edata["w"] = torch.tensor(w_list, dtype=torch.float32)
    g.edata["dir"] = torch.tensor(dir_list, dtype=torch.float32)
    g.edata["tf2t"] = torch.tensor(tf2t_list, dtype=torch.float32)
    return g


def build_train_positive_pearson_graph(
    train_edge_index: torch.Tensor,
    train_labels: torch.Tensor,
    expr: torch.Tensor,
) -> dgl.DGLGraph:
    """Prior edges = Train_set Label=1 only; weights = Pearson r across cells."""
    ei = train_edge_index.detach().long().cpu()
    y = train_labels.detach().float().cpu().view(-1)
    pos = y > 0.5
    src = ei[0, pos]
    dst = ei[1, pos]
    if src.numel() == 0:
        raise RuntimeError("No positive edges in Train_set.")

    z = expr.detach().float().cpu().numpy()
    n_genes = z.shape[0]
    if int(ei.max().item()) >= n_genes:
        raise ValueError("Edge index out of range for expression matrix.")

    r_full = np.corrcoef(z)
    np.nan_to_num(r_full, copy=False, nan=0.0)
    best_r: Dict[Tuple[int, int], float] = {}
    for a, b in zip(src.tolist(), dst.tolist()):
        i, j = int(a), int(b)
        if i == j:
            continue
        rij = float(r_full[i, j])
        prev = best_r.get((i, j))
        if prev is None or rij > prev:
            best_r[(i, j)] = rij
    if not best_r:
        raise RuntimeError("No usable directed train-positive edges.")
    triples = [(i, j, w) for (i, j), w in best_r.items()]
    return _triples_to_dgl(triples, n_genes, add_self_loops=True)
