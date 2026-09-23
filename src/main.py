#!/usr/bin/env python3
"""Train scGT-Link v2 on the bundled STRING hESC TFs+500 demo dataset.

Paper defaults:
  expr_encoder = scGPT + LapPE
  prior_graph  = train_pos_pearson
  gt_variant   = directed, L=2, H=4, hidden=128
  d_model=768, lr=3e-4, epochs=200, seed=42

Run from repo root or from src/:
  cd src/
  python main.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from encoder import ScGPTEncoder
from input_data import load_edge_split, load_expression, load_scgpt_emb_pt
from link import LinkPredictor
from model import GraphTransformer
from prior import build_train_positive_pearson_graph

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_ROOT / "data" / "STRING_hESC_TFs500"
_DEFAULT_SCGPT = _REPO_ROOT / "scGPT" / "STRING_hESC_TFs500" / "scgpt_gene_emb.pt"
_DEFAULT_OUT = _REPO_ROOT / "out" / "STRING_hESC_TFs500"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_split(seq_model, gt_model, predictor, gene_expr, g, pe, ei, y):
    seq_model.eval()
    gt_model.eval()
    predictor.eval()
    h_seq = seq_model(pe)
    h = gt_model(g, h_seq, pe)
    logits = predictor(h, ei).squeeze(-1)
    labels = y.detach().cpu().numpy()
    scores = logits.detach().cpu().numpy()
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scGT-Link v2 demo (STRING hESC TFs+500)")
    p.add_argument("--data_dir", type=Path, default=_DEFAULT_DATA)
    p.add_argument(
        "--scgpt_emb",
        type=Path,
        default=_DEFAULT_SCGPT,
        help="Precomputed scGPT gene embedding .pt",
    )
    p.add_argument(
        "--split_dir",
        type=Path,
        default=None,
        help="Default: <data_dir>/Train_validation_test",
    )
    p.add_argument("--output_dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--gt_hidden_dim", type=int, default=128)
    p.add_argument("--gt_num_layers", type=int, default=2)
    p.add_argument("--gt_num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = args.data_dir.expanduser().resolve()
    emb_path = args.scgpt_emb.expanduser().resolve()
    split_dir = (args.split_dir or (data_dir / "Train_validation_test")).expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    expr_path = data_dir / "BL--ExpressionData.csv"
    for path in (expr_path, emb_path, split_dir / "Train_set.csv"):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    gene_expr, gene_names, n_cells = load_expression(expr_path)
    scgpt_mat, scgpt_dim = load_scgpt_emb_pt(emb_path, gene_names)
    train_ei, train_y = load_edge_split(split_dir, "Train_set")
    val_ei, val_y = load_edge_split(split_dir, "Validation_set")
    test_ei, test_y = load_edge_split(split_dir, "Test_set")

    g = build_train_positive_pearson_graph(train_ei, train_y, gene_expr)
    pe = dgl.lap_pe(g, k=args.d_model, padding=True)

    gene_expr = gene_expr.to(device)
    pe = pe.to(device=device, dtype=gene_expr.dtype)
    g = g.to(device)
    train_ei, train_y = train_ei.to(device), train_y.to(device)
    val_ei, val_y = val_ei.to(device), val_y.to(device)
    test_ei, test_y = test_ei.to(device), test_y.to(device)
    scgpt_mat = scgpt_mat.to(device)

    seq_model = ScGPTEncoder(scgpt_mat, d_model=args.d_model).to(device)
    gt_model = GraphTransformer(
        in_dim=args.d_model,
        hidden_dim=args.gt_hidden_dim,
        num_heads=args.gt_num_heads,
        num_layers=args.gt_num_layers,
        dropout=args.dropout,
    ).to(device)
    predictor = LinkPredictor(args.gt_hidden_dim).to(device)

    n_pos = float((train_y > 0.5).sum().item())
    n_neg = float((train_y <= 0.5).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(
        list(seq_model.parameters()) + list(gt_model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    print(
        f"Demo | genes={len(gene_names)} cells={n_cells} scgpt_dim={scgpt_dim} "
        f"| prior_edges={g.num_edges()} | train pos={int(n_pos)} neg={int(n_neg)} "
        f"| device={device}",
        flush=True,
    )

    best_val = -1.0
    best_epoch = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        seq_model.train()
        gt_model.train()
        predictor.train()
        opt.zero_grad(set_to_none=True)
        h_seq = seq_model(pe)
        h = gt_model(g, h_seq, pe)
        logits = predictor(h, train_ei).squeeze(-1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        opt.step()

        val_au, val_ap = eval_split(seq_model, gt_model, predictor, gene_expr, g, pe, val_ei, val_y)
        if val_au > best_val:
            best_val = val_au
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "seq_model": seq_model.state_dict(),
                    "gt_model": gt_model.state_dict(),
                    "predictor": predictor.state_dict(),
                    "val_auroc": val_au,
                    "val_auprc": val_ap,
                },
                out_dir / "best.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | loss={loss.item():.4f} "
                f"| val AUROC={val_au:.4f} AUPRC={val_ap:.4f}",
                flush=True,
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    seq_model.load_state_dict(ckpt["seq_model"])
    gt_model.load_state_dict(ckpt["gt_model"])
    predictor.load_state_dict(ckpt["predictor"])
    test_au, test_ap = eval_split(seq_model, gt_model, predictor, gene_expr, g, pe, test_ei, test_y)
    elapsed = time.perf_counter() - t0
    results = {
        "dataset": "STRING__hESC__500",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val,
        "test_auroc": test_au,
        "test_auprc": test_ap,
        "training_time_sec": elapsed,
        "device": str(device),
        "config": {
            "expr_encoder": "scgpt",
            "prior_graph": "train_pos_pearson",
            "gt_variant": "directed",
            "d_model": args.d_model,
            "gt_hidden_dim": args.gt_hidden_dim,
            "gt_num_layers": args.gt_num_layers,
            "gt_num_heads": args.gt_num_heads,
            "lr": args.lr,
            "dropout": args.dropout,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("-" * 64)
    print(
        f"Done | best_epoch={best_epoch} | test AUROC={test_au:.4f} AUPRC={test_ap:.4f} "
        f"| time={elapsed:.1f}s | wrote {out_dir / 'results.json'}"
    )


if __name__ == "__main__":
    main()
