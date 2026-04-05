#!/usr/bin/env python3
"""
Stage 1: Pearson correlation → sparse DGL graph.
Stage 2: GeneExpressionTransformer(expression + lap_pe); GraphTransformer(X, PE, semantic_feat) + LinkPredictor.
         Optional semantic_feat from precomputed gene-symbol embeddings (--gene_bert_emb).

Example (from this repo root; ``src/`` holds GraphTransformer + LinkPredictor, no PYTHONPATH needed):
  python train_grn_graph.py \\
    --data_dir data/hESC/TFs+500 \\
    --split_dir data/Train_validation_test/hESC_500 \\
    --epochs 50 --output_dir out_grn_gt
"""

from __future__ import annotations

import os
import sys


def _prefer_conda_over_user_site() -> None:
    """Drop ~/.local/.../site-packages so a user ``pip install --user dgl`` does not shadow conda."""
    if os.environ.get("GT_GRN_ALLOW_USER_SITE") == "1":
        return
    try:
        import site

        u = site.getusersitepackages()
        if not u:
            return
        sys.path[:] = [p for p in sys.path if p != u and not p.startswith(u + os.sep)]
    except Exception:
        pass


_prefer_conda_over_user_site()

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("DGLBACKEND", "pytorch")
try:
    import dgl  # noqa: E402
except ImportError as e:
    raise ImportError(
        "DGL is required (and must match your PyTorch CUDA build). "
        "If you use conda: install dgl in the active env; remove user shadowing with "
        "`python -m pip uninstall -y dgl` (or set GT_GRN_ALLOW_USER_SITE=1 to allow ~/.local). "
        "Example: pip install 'dgl==2.1.0+cu121' -f https://data.dgl.ai/wheels/cu121/repo.html"
    ) from e

# This file lives at repo root; ``src`` is alongside it (not parent of transformer_expression).
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import GraphTransformer  # noqa: E402
from utils import LinkPredictor, evaluate  # noqa: E402

from correlation_graph import build_correlation_graph  # noqa: E402
from grn_data import load_edge_split, load_expression_for_grn, load_gene_bert_embeddings  # noqa: E402
from grn_model import GeneExpressionTransformer  # noqa: E402


def semantic_features(
    n_genes: int,
    in_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    gene_bert_raw: Optional[torch.Tensor],
    bert_proj: Optional[nn.Module],
) -> torch.Tensor:
    if bert_proj is not None and gene_bert_raw is not None:
        return bert_proj(gene_bert_raw)
    return torch.zeros(n_genes, in_dim, device=device, dtype=dtype)


def parse_args():
    p = argparse.ArgumentParser(description="Correlation graph + Graph Transformer GRN")
    p.add_argument("--data_dir", type=str, default="data/hESC/TFs+500")
    p.add_argument("--split_dir", type=str, default="data/Train_validation_test/hESC_500")
    p.add_argument("--output_dir", type=str, default="out_grn_gt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--edge_batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=128, help="GeneExpressionTransformer d_model = GT in_dim = lap_pe k")
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4, help="GeneExpressionTransformer layers")
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gt_hidden_dim", type=int, default=80)
    p.add_argument("--gt_num_layers", type=int, default=6)
    p.add_argument("--gt_num_heads", type=int, default=4)
    p.add_argument("--top_k", type=int, default=20, help="Correlation graph: top-k neighbors per gene")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gene_bert_emb",
        type=str,
        default="",
        help="Path to .pt from scripts/encode_gene_symbols.py (empty = zero semantic, ablation baseline)",
    )
    p.add_argument(
        "--test_prob_threshold",
        type=float,
        default=0.5,
        help="Test precision/recall/F1/accuracy: predict positive if sigmoid(logit) >= this",
    )
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def logits_all_edges(
    seq_model: GeneExpressionTransformer,
    gt_model: GraphTransformer,
    predictor: LinkPredictor,
    gene_expr: torch.Tensor,
    g: dgl.DGLGraph,
    pe: torch.Tensor,
    edge_index: torch.Tensor,
    chunk: int,
    gene_bert_raw: Optional[torch.Tensor],
    bert_proj: Optional[nn.Module],
    in_dim: int,
) -> torch.Tensor:
    seq_model.eval()
    gt_model.eval()
    predictor.eval()
    n_genes = g.num_nodes()
    sem = semantic_features(
        n_genes, in_dim, gene_expr.device, gene_expr.dtype, gene_bert_raw, bert_proj
    )
    h_seq = seq_model(gene_expr, pe)
    g.ndata["feat"] = h_seq
    g.ndata["PE"] = pe
    g.ndata["semantic"] = sem
    h = gt_model(g, h_seq, pe, sem)
    outs = []
    e = edge_index.size(1)
    for s in range(0, e, chunk):
        outs.append(predictor(h, edge_index[:, s : s + chunk]).squeeze(-1))
    return torch.cat(outs, dim=0)


def compute_test_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    prob_threshold: float = 0.5,
) -> dict:
    """AUROC, AUPRC (ranking); precision, recall, F1, accuracy at sigmoid(logits) >= threshold."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = labels.astype(np.int64)
    logits_f = logits.astype(np.float64)
    # numerically stable sigmoid
    probs = np.where(
        logits_f >= 0,
        1.0 / (1.0 + np.exp(-logits_f)),
        np.exp(logits_f) / (1.0 + np.exp(logits_f)),
    )
    y_pred = (probs >= prob_threshold).astype(np.int64)

    if len(np.unique(y)) < 2:
        nan = float("nan")
        return {
            "test_auroc": nan,
            "test_auprc": nan,
            "test_precision": nan,
            "test_recall": nan,
            "test_f1": nan,
            "test_accuracy": nan,
            "test_prob_threshold": prob_threshold,
        }

    return {
        "test_auroc": float(roc_auc_score(y, logits_f)),
        "test_auprc": float(average_precision_score(y, logits_f)),
        "test_precision": float(precision_score(y, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y, y_pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y, y_pred)),
        "test_prob_threshold": prob_threshold,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    exp_path = data_dir / "BL--ExpressionData.csv"
    if not exp_path.exists():
        raise SystemExit(f"Missing {exp_path}")
    if not split_dir.is_dir():
        raise SystemExit(f"Missing split dir {split_dir}")

    gene_expr, gene_names, n_cells = load_expression_for_grn(exp_path)
    n_genes = len(gene_names)
    gene_expr = gene_expr.to(device)

    train_ei, train_y = load_edge_split(split_dir, "Train_set")
    val_ei, val_y = load_edge_split(split_dir, "Validation_set")
    test_ei, test_y = load_edge_split(split_dir, "Test_set")
    train_ei, train_y = train_ei.to(device), train_y.to(device)
    val_ei, val_y = val_ei.to(device), val_y.to(device)
    test_ei, test_y = test_ei.to(device), test_y.to(device)

    in_dim = args.d_model
    g = build_correlation_graph(gene_expr.detach().cpu(), top_k=args.top_k)
    g = g.to(device)
    with torch.no_grad():
        pe = dgl.lap_pe(g, k=in_dim, padding=True)
    pe = pe.to(device=device, dtype=gene_expr.dtype)

    gene_bert_raw: Optional[torch.Tensor] = None
    bert_proj: Optional[nn.Linear] = None
    bert_dim: Optional[int] = None
    emb_path = args.gene_bert_emb.strip()
    if emb_path:
        emb_p = Path(emb_path)
        if not emb_p.is_file():
            raise SystemExit(f"--gene_bert_emb not found: {emb_p}")
        raw_cpu = load_gene_bert_embeddings(emb_p, gene_names)
        bert_dim = raw_cpu.shape[1]
        gene_bert_raw = raw_cpu.to(device=device, dtype=gene_expr.dtype)
        bert_proj = nn.Linear(bert_dim, in_dim).to(device)

    seq_model = GeneExpressionTransformer(
        n_cells=n_cells,
        d_model=in_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    gt_model = GraphTransformer(
        in_dim, args.gt_hidden_dim, args.gt_num_heads, args.gt_num_layers
    ).to(device)
    predictor = LinkPredictor(args.gt_hidden_dim).to(device)

    n_pos = float((train_y == 1).sum().item())
    n_neg = float((train_y == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    params = (
        list(seq_model.parameters())
        + list(gt_model.parameters())
        + list(predictor.parameters())
    )
    if bert_proj is not None:
        params += list(bert_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    val_pos = val_ei[:, val_y == 1]
    val_neg = val_ei[:, val_y == 0]
    test_pos = test_ei[:, test_y == 1]
    test_neg = test_ei[:, test_y == 0]

    n_train = train_ei.size(1)
    idx_perm = torch.arange(n_train, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_edges_graph = g.num_edges()
    print(
        f"Genes={n_genes} cells={n_cells} | corr graph edges={n_edges_graph} | "
        f"train edges={n_train} (pos={int(n_pos)} neg={int(n_neg)}) | pos_weight={pos_weight.item():.3f}"
        + (f" | gene_bert_emb D={bert_dim}" if bert_dim else " | gene_bert_emb (none)")
    )
    print("-" * 72)

    best_val_auroc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        seq_model.train()
        gt_model.train()
        predictor.train()
        if bert_proj is not None:
            bert_proj.train()
        idx_perm = idx_perm[torch.randperm(n_train)]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.edge_batch_size):
            sel = idx_perm[start : start + args.edge_batch_size]
            ei = train_ei[:, sel]
            y = train_y[sel]
            opt.zero_grad()
            sem = semantic_features(
                n_genes, in_dim, device, gene_expr.dtype, gene_bert_raw, bert_proj
            )
            h_seq = seq_model(gene_expr, pe)
            h = gt_model(g, h_seq, pe, sem)
            logits = predictor(h, ei).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        seq_model.eval()
        gt_model.eval()
        predictor.eval()
        if bert_proj is not None:
            bert_proj.eval()
        with torch.no_grad():
            h_seq = seq_model(gene_expr, pe)
            sem = semantic_features(
                n_genes, in_dim, device, gene_expr.dtype, gene_bert_raw, bert_proj
            )
        g.ndata["feat"] = h_seq
        g.ndata["PE"] = pe
        g.ndata["semantic"] = sem
        v_auroc, v_ap, _, _ = evaluate(gt_model, predictor, g, val_pos, val_neg)
        seq_model.train()
        gt_model.train()
        predictor.train()
        if bert_proj is not None:
            bert_proj.train()

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_auroc": float(v_auroc),
                "val_auprc": float(v_ap),
            }
        )

        if v_auroc > best_val_auroc:
            best_val_auroc = float(v_auroc)
            ckpt: dict = {
                "seq_model": seq_model.state_dict(),
                "gt_model": gt_model.state_dict(),
                "predictor": predictor.state_dict(),
                "meta": {
                    "n_genes": n_genes,
                    "n_cells": n_cells,
                    "d_model": in_dim,
                    "gt_hidden_dim": args.gt_hidden_dim,
                    "gt_num_layers": args.gt_num_layers,
                    "top_k": args.top_k,
                    "n_edges_graph": int(n_edges_graph),
                    "use_gene_bert": bool(bert_proj is not None),
                    "bert_dim": bert_dim,
                },
            }
            if bert_proj is not None:
                ckpt["bert_proj"] = bert_proj.state_dict()
            torch.save(ckpt, out_dir / "best.pt")

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | train_loss={epoch_loss/max(n_batches,1):.4f} | "
                f"val_AUROC={v_auroc:.4f} val_AUPRC={v_ap:.4f}"
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    seq_model.load_state_dict(ckpt["seq_model"])
    gt_model.load_state_dict(ckpt["gt_model"])
    predictor.load_state_dict(ckpt["predictor"])
    meta = ckpt.get("meta", {})
    if "bert_proj" in ckpt:
        bd = meta.get("bert_dim")
        if bd is None:
            raise SystemExit("checkpoint has bert_proj but meta.bert_dim missing")
        bert_proj = nn.Linear(int(bd), in_dim).to(device)
        bert_proj.load_state_dict(ckpt["bert_proj"])
        bert_proj.eval()
        if gene_bert_raw is None:
            if not emb_path:
                raise SystemExit(
                    "Checkpoint uses gene BioBERT; re-run with --gene_bert_emb pointing to the same .pt file"
                )
            raw_cpu = load_gene_bert_embeddings(Path(emb_path), gene_names)
            gene_bert_raw = raw_cpu.to(device=device, dtype=gene_expr.dtype)
    else:
        bert_proj = None
        gene_bert_raw = None

    te_logits = logits_all_edges(
        seq_model,
        gt_model,
        predictor,
        gene_expr,
        g,
        pe,
        test_ei,
        args.edge_batch_size,
        gene_bert_raw,
        bert_proj,
        in_dim,
    )
    test_metrics = compute_test_metrics(
        test_y.cpu().numpy(),
        te_logits.cpu().numpy(),
        prob_threshold=args.test_prob_threshold,
    )

    print("-" * 72)
    print(
        f"Test AUROC: {test_metrics['test_auroc']:.4f}  Test AUPRC: {test_metrics['test_auprc']:.4f}  "
        f"(threshold={test_metrics['test_prob_threshold']} on sigmoid)"
    )
    print(
        f"Test Precision: {test_metrics['test_precision']:.4f}  Recall: {test_metrics['test_recall']:.4f}  "
        f"F1: {test_metrics['test_f1']:.4f}  Accuracy: {test_metrics['test_accuracy']:.4f}"
    )

    results = {
        **test_metrics,
        "best_val_auroc": best_val_auroc,
        "n_edges_correlation_graph": int(n_edges_graph),
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "history": history,
        "config": vars(args),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        out_dir / "test_predictions.npz",
        logits=te_logits.cpu().numpy(),
        labels=test_y.cpu().numpy(),
    )
    print(f"Saved: {out_dir / 'results.json'}, {out_dir / 'best.pt'}, {out_dir / 'test_predictions.npz'}")


if __name__ == "__main__":
    main()
