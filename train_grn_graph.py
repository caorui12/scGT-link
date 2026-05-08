#!/usr/bin/env python3
"""
Stage 1: Prior graph — ``correlation``: Pearson |r| top-k; ``gold`` / ``gold_pearson``: edges from ``--gold_network_file``
         (**gene symbols** in CSV, matching expression row names); ``gold_pearson`` sets ``edata['w']`` from Pearson r.
         ``train_pos_pearson``: **only** ``Train_set`` rows with Label=1 as directed edges; weights from Pearson r (no CSV).
         ``edata['w']``: Pearson r (or min–max normalized); self-loops 1.0.
         For ``prior_graph=gold``, ``gold_pearson``, or ``train_pos_pearson``, ``edata['dir']`` is +1 on TF→target edges and −1 on mirrored edges
         (unless ``--no_edge_dir_attn``); SparseMHA adds ``edge_dir_scale * dir`` to logits.
Stage 2: Gene encoder + GraphTransformer + LinkPredictor. Defaults follow best hESC_500 tune (``scgpt_concat`` + ``gold_pearson``).
         Use ``--expr_encoder transformer`` for expression-only; ``scgpt`` for scGPT-only (see ``--scgpt_emb_pt`` / ``--scgpt_model_dir``).
         SparseMHA adds learnable-scaled ``w`` to attention logits unless ``--no_edge_weight_attn``.

Example (from this repo root; ``src/`` holds GraphTransformer + LinkPredictor, no PYTHONPATH needed):
  python train_grn_graph.py \\
    --data_dir data/hESC/TFs+500 \\
    --split_dir data/Train_validation_test/hESC_500 \\
    --epochs 50 --output_dir out_grn_gt

Registered datasets (``--dataset NAME`` sets ``data_dir`` + ``split_dir``): hESC_500, hESC_1000, mESC_500, mESC_1000.

Train all four and write ``all_results.json`` under the output root:
  python train_grn_graph.py --train_all --output_dir out_multi ...

Paper-style splits live under a sibling folder name (e.g. ``Train_validation_test_gnnlink_paper``).
Use ``--split_subdir Train_validation_test_gnnlink_paper`` so any ``--split_dir`` / registry path
that contains a ``Train_validation_test`` segment is rewritten (also works for BEELINE leaves like
``.../TFs+500/Train_validation_test``).

Programmatic single run: ``from train_grn_graph import train_grn_dataset`` then ``train_grn_dataset("hESC_500", epochs=50)``.
"""

from __future__ import annotations

import os
import re
import sys
from argparse import Namespace


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
from gold_network_graph import (  # noqa: E402
    build_gold_network_graph,
    build_gold_pearson_graph,
    build_train_positive_pearson_graph,
)
from grn_data import load_edge_split, load_expression_for_grn  # noqa: E402
from grn_model import (  # noqa: E402
    GeneExpressionTransformer,
    ScGPTExprConcatEncoder,
    ScGPTGeneExpressionEncoder,
)
from scgpt_loader import (  # noqa: E402
    load_scgpt_emb_from_pt,
    load_scgpt_gene_embedding_matrix,
)

# Registered datasets under ``data/``: expression folder + Train_validation_test split folder.
DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "hESC_500": ("data/hESC/TFs+500", "data/Train_validation_test/hESC_500"),
    "hESC_1000": ("data/hESC/TFs+1000", "data/Train_validation_test/hESC_1000"),
    "mESC_500": ("data/mESC/TFs+500", "data/Train_validation_test/mESC_500"),
    "mESC_1000": ("data/mESC/TFs+1000", "data/Train_validation_test/mESC_1000"),
}
DATASET_ORDER = ("hESC_500", "hESC_1000", "mESC_500", "mESC_1000")

_SPLIT_SEG_PATTERN = re.compile(r"(^|[\\/])Train_validation_test(?=[\\/]|$)")


def apply_split_subdir(split_dir_str: str, subdir: str) -> str:
    """Replace path segment ``Train_validation_test`` with *subdir* (first occurrence only)."""
    subdir = (subdir or "").strip() or "Train_validation_test"
    if subdir == "Train_validation_test":
        return split_dir_str

    def repl(m: re.Match[str]) -> str:
        return m.group(1) + subdir

    new_s, n = _SPLIT_SEG_PATTERN.subn(repl, split_dir_str, count=1)
    return new_s if n else split_dir_str


# argparse defaults point here so ``--dataset mESC_500`` (etc.) can remap to sibling files under that data_dir.
_DEFAULT_TUNE_GOLD_REL = "data/hESC/TFs+500/BL--network.csv"
_DEFAULT_TUNE_SCGPT_REL = "data/hESC/TFs+500/scgpt_gene_emb.pt"


def _remap_tune_default_paths(args: Namespace) -> None:
    """Keep best-tune defaults portable: same basenames under whatever ``--data_dir`` is active."""
    data_dir = Path(args.data_dir)
    if (
        args.prior_graph in ("gold", "gold_pearson")
        and args.gold_network_file.strip() == _DEFAULT_TUNE_GOLD_REL
    ):
        args.gold_network_file = str(data_dir / "BL--network.csv")
    if (
        args.expr_encoder in ("scgpt", "scgpt_concat")
        and args.scgpt_emb_pt.strip() == _DEFAULT_TUNE_SCGPT_REL
    ):
        args.scgpt_emb_pt = str(data_dir / "scgpt_gene_emb.pt")


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Prior graph (Pearson / gold) + Graph Transformer GRN")
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        metavar="NAME",
        help="Registered key: sets --data_dir and --split_dir. "
        f"One of: {', '.join(DATASET_ORDER)}. Overrides --data_dir/--split_dir when non-empty.",
    )
    p.add_argument(
        "--train_all",
        action="store_true",
        help="Train all four registry datasets sequentially; --output_dir is the root "
        "(one subfolder per dataset). Writes all_results.json at the root.",
    )
    p.add_argument("--data_dir", type=str, default="data/hESC/TFs+500")
    p.add_argument("--split_dir", type=str, default="data/Train_validation_test/hESC_500")
    p.add_argument(
        "--split_subdir",
        type=str,
        default="Train_validation_test",
        metavar="NAME",
        help="Rewrite split path: replace segment Train_validation_test with this folder name "
        "(e.g. Train_validation_test_gnnlink_paper). Ignored when unchanged from default.",
    )
    p.add_argument("--output_dir", type=str, default="out_grn_gt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--edge_batch_size", type=int, default=4096)
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Default 3e-4 matches best hESC_500 tune (summary.csv test_auroc ~0.945).",
    )
    p.add_argument(
        "--expr_encoder",
        type=str,
        choices=("transformer", "scgpt", "scgpt_concat"),
        default="scgpt_concat",
        help="transformer: Linear(n_cells)+TransformerEncoder. "
        "scgpt: scGPT token emb + proj (+ optional refine). "
        "scgpt_concat: same expression Transformer as transformer, concat with proj(scGPT), Linear fuse.",
    )
    p.add_argument(
        "--d_model",
        type=int,
        default=768,
        help="GT in_dim = lap_pe k; expression encoder output dim (default from best tune).",
    )
    p.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="GeneExpressionTransformer heads (expr_encoder=transformer or scgpt_concat; "
        "for scgpt only when --scgpt_refine_layers>0)",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="GeneExpressionTransformer layers (expr_encoder=transformer or scgpt_concat; "
        "ignored for plain scgpt unless using refine)",
    )
    p.add_argument(
        "--scgpt_model_dir",
        type=str,
        default="",
        help="Directory with args.json, vocab.json, best_model.pt (expr_encoder=scgpt or scgpt_concat). "
        "Ignored if --scgpt_emb_pt is set.",
    )
    p.add_argument(
        "--scgpt_emb_pt",
        type=str,
        default=_DEFAULT_TUNE_SCGPT_REL,
        help="Precomputed .pt from scripts/precompute_scgpt_gene_emb.py (overrides --scgpt_model_dir). Empty to disable.",
    )
    p.add_argument(
        "--scgpt_refine_layers",
        type=int,
        default=0,
        help="Optional TransformerEncoder layers on top of proj(scGPT)+Lap PE (expr_encoder=scgpt).",
    )
    p.add_argument("--scgpt_refine_nhead", type=int, default=4)
    p.add_argument("--scgpt_refine_dim_feedforward", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--gt_hidden_dim",
        type=int,
        default=128,
        help="GraphTransformer hidden size; must be divisible by --gt_num_heads",
    )
    p.add_argument(
        "--gt_num_layers",
        type=int,
        default=6,
        help="GraphTransformer layers (default 6 matches best hESC_500 tune).",
    )
    p.add_argument(
        "--gt_num_heads",
        type=int,
        default=4,
        help="Sparse MHA heads; --gt_hidden_dim %% this must be 0 (e.g. 80/6 is invalid).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="prior_graph=correlation: top-k neighbors per gene by |Pearson r|.",
    )
    p.add_argument(
        "--prior_graph",
        type=str,
        choices=("correlation", "gold", "gold_pearson", "train_pos_pearson"),
        default="gold_pearson",
        help="correlation: Pearson |r| top-k graph. gold: reference network from --gold_network_file (optional CSV weights). "
        "gold_pearson: gold edges + Pearson r as weights (directed structure via edata['dir']). "
        "train_pos_pearson: only Train_set Label=1 edges + Pearson r (ignores --gold_network_file; aligns with train-only adjacency).",
    )
    p.add_argument(
        "--gold_network_file",
        type=str,
        default=_DEFAULT_TUNE_GOLD_REL,
        help="prior_graph=gold or gold_pearson: CSV with TF/Target or Gene1/Gene2 (etc.); "
        "endpoint values must be gene symbols matching expression row names (not integer indices). "
        "See gold_network_graph.build_gold_network_graph / build_gold_pearson_graph.",
    )
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_edge_weight_attn",
        action="store_true",
        help="Disable edge importance in SparseMHA (ablation). Default: add learnable-scaled edata['w'] to logits.",
    )
    p.add_argument(
        "--no_edge_dir_attn",
        action="store_true",
        help="For prior_graph=gold, gold_pearson, or train_pos_pearson: do not add learnable edata['dir'] (±1) bias in SparseMHA. "
        "Ignored for correlation prior (no dir field).",
    )
    p.add_argument(
        "--test_prob_threshold",
        type=float,
        default=0.5,
        help="Test precision/recall/F1/accuracy: predict positive if sigmoid(logit) >= this",
    )
    if argv is None:
        return p.parse_args()
    return p.parse_args(argv)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def logits_all_edges(
    seq_model: nn.Module,
    gt_model: GraphTransformer,
    predictor: LinkPredictor,
    gene_expr: torch.Tensor,
    g: dgl.DGLGraph,
    pe: torch.Tensor,
    edge_index: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    seq_model.eval()
    gt_model.eval()
    predictor.eval()
    h_seq = seq_model(gene_expr, pe)
    g.ndata["feat"] = h_seq
    g.ndata["PE"] = pe
    h = gt_model(g, h_seq, pe)
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


def run_training(args: Namespace) -> dict:
    """Run one training job. ``args`` is an argparse namespace (see :func:`parse_args`). Returns results dict."""
    args.split_dir = apply_split_subdir(args.split_dir, getattr(args, "split_subdir", "Train_validation_test"))
    _remap_tune_default_paths(args)
    if args.expr_encoder in ("transformer", "scgpt_concat") and args.d_model % args.nhead != 0:
        raise SystemExit(
            f"--d_model ({args.d_model}) must be divisible by --nhead ({args.nhead})"
        )
    if (
        args.expr_encoder == "scgpt"
        and args.scgpt_refine_layers > 0
        and args.d_model % args.scgpt_refine_nhead != 0
    ):
        raise SystemExit(
            f"--d_model ({args.d_model}) must be divisible by --scgpt_refine_nhead ({args.scgpt_refine_nhead})"
        )
    if args.gt_hidden_dim % args.gt_num_heads != 0:
        raise SystemExit(
            f"--gt_hidden_dim ({args.gt_hidden_dim}) must be divisible by --gt_num_heads "
            f"({args.gt_num_heads}); e.g. use --gt_hidden_dim 84 or --gt_num_heads 5 with 80."
        )
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
    expr_cpu = gene_expr.detach().cpu()
    if args.prior_graph == "correlation":
        g = build_correlation_graph(expr_cpu, top_k=args.top_k)
    elif args.prior_graph == "gold":
        if not args.gold_network_file.strip():
            raise SystemExit(
                "prior_graph=gold requires --gold_network_file path/to.csv (TF, Target; optional weight)."
            )
        try:
            g = build_gold_network_graph(
                Path(args.gold_network_file.strip()),
                gene_names,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise SystemExit(str(e)) from e
    elif args.prior_graph == "gold_pearson":
        if not args.gold_network_file.strip():
            raise SystemExit(
                "prior_graph=gold_pearson requires --gold_network_file (gold structure; weights from Pearson r)."
            )
        try:
            g = build_gold_pearson_graph(
                Path(args.gold_network_file.strip()),
                gene_names,
                expr_cpu,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise SystemExit(str(e)) from e
    elif args.prior_graph == "train_pos_pearson":
        try:
            g = build_train_positive_pearson_graph(
                train_ei.detach().cpu(),
                train_y.detach().cpu(),
                expr_cpu,
            )
        except (ValueError, RuntimeError) as e:
            raise SystemExit(str(e)) from e
    g = g.to(device)
    with torch.no_grad():
        pe = dgl.lap_pe(g, k=in_dim, padding=True)
    pe = pe.to(device=device, dtype=gene_expr.dtype)

    scgpt_dim: Optional[int] = None
    if args.expr_encoder in ("scgpt", "scgpt_concat"):
        if not args.scgpt_emb_pt.strip() and not args.scgpt_model_dir.strip():
            raise SystemExit(
                "expr_encoder=scgpt or scgpt_concat requires --scgpt_emb_pt and/or --scgpt_model_dir "
                "(precomputed .pt or scGPT checkpoint directory)."
            )
        if args.scgpt_emb_pt.strip():
            scgpt_mat_cpu, scgpt_dim = load_scgpt_emb_from_pt(
                Path(args.scgpt_emb_pt.strip()), gene_names
            )
        else:
            scgpt_mat_cpu, scgpt_dim = load_scgpt_gene_embedding_matrix(
                Path(args.scgpt_model_dir.strip()),
                gene_names,
                device=torch.device("cpu"),
            )
        scgpt_mat = scgpt_mat_cpu.to(device=device)
        if args.expr_encoder == "scgpt_concat":
            seq_model = ScGPTExprConcatEncoder(
                scgpt_mat,
                n_cells=n_cells,
                d_model=in_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
            ).to(device)
        else:
            seq_model = ScGPTGeneExpressionEncoder(
                scgpt_mat,
                d_model=in_dim,
                refine_layers=args.scgpt_refine_layers,
                refine_nhead=args.scgpt_refine_nhead,
                dim_feedforward=args.scgpt_refine_dim_feedforward,
                dropout=args.dropout,
            ).to(device)
    else:
        seq_model = GeneExpressionTransformer(
            n_cells=n_cells,
            d_model=in_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)
    use_edge_w = not args.no_edge_weight_attn
    use_dir_attn = (
        args.prior_graph in ("gold", "gold_pearson", "train_pos_pearson")
        and not args.no_edge_dir_attn
    )
    gt_model = GraphTransformer(
        in_dim,
        args.gt_hidden_dim,
        args.gt_num_heads,
        args.gt_num_layers,
        use_edge_weight_attn=use_edge_w,
        use_directed_edge_bias=use_dir_attn,
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
        f"Genes={n_genes} cells={n_cells} | expr_encoder={args.expr_encoder}"
        + (
            f" scgpt_dim={scgpt_dim}"
            if args.expr_encoder in ("scgpt", "scgpt_concat") and scgpt_dim is not None
            else ""
        )
        + f" | prior={args.prior_graph} edges={n_edges_graph} | "
        f"train edges={n_train} (pos={int(n_pos)} neg={int(n_neg)}) | pos_weight={pos_weight.item():.3f}"
        + f" | edge_weight_attn={'on' if use_edge_w else 'off'}"
        + (
            f" | edge_dir_attn={'on' if use_dir_attn else 'off'}"
            if args.prior_graph in ("gold", "gold_pearson", "train_pos_pearson")
            else ""
        )
    )
    print("-" * 72)

    best_val_auroc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        seq_model.train()
        gt_model.train()
        predictor.train()
        idx_perm = idx_perm[torch.randperm(n_train)]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.edge_batch_size):
            sel = idx_perm[start : start + args.edge_batch_size]
            ei = train_ei[:, sel]
            y = train_y[sel]
            opt.zero_grad()
            h_seq = seq_model(gene_expr, pe)
            h = gt_model(g, h_seq, pe)
            logits = predictor(h, ei).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        seq_model.eval()
        gt_model.eval()
        predictor.eval()
        with torch.no_grad():
            h_seq = seq_model(gene_expr, pe)
        g.ndata["feat"] = h_seq
        g.ndata["PE"] = pe
        v_auroc, v_ap, _, _ = evaluate(gt_model, predictor, g, val_pos, val_neg)
        seq_model.train()
        gt_model.train()
        predictor.train()

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
                    "expr_encoder": args.expr_encoder,
                    "scgpt_dim": scgpt_dim,
                    "scgpt_refine_layers": args.scgpt_refine_layers,
                    "scgpt_refine_nhead": args.scgpt_refine_nhead,
                    "scgpt_refine_dim_feedforward": args.scgpt_refine_dim_feedforward,
                    "gt_hidden_dim": args.gt_hidden_dim,
                    "gt_num_layers": args.gt_num_layers,
                    "top_k": args.top_k,
                    "n_edges_graph": int(n_edges_graph),
                    "use_edge_weight_attn": use_edge_w,
                    "prior_graph": args.prior_graph,
                    "gold_network_file": str(Path(args.gold_network_file).resolve())
                    if args.prior_graph in ("gold", "gold_pearson") and args.gold_network_file.strip()
                    else "",
                    "gold_network_endpoints": (
                        "gene_symbol"
                        if args.prior_graph in ("gold", "gold_pearson")
                        else ("train_positive_indices" if args.prior_graph == "train_pos_pearson" else "")
                    ),
                    "use_directed_edge_attn": use_dir_attn,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "dim_feedforward": args.dim_feedforward,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | train_loss={epoch_loss/max(n_batches,1):.4f} | "
                f"val_AUROC={v_auroc:.4f} val_AUPRC={v_ap:.4f}"
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    meta = ckpt.get("meta", {})
    use_dir_eval = meta.get("use_directed_edge_attn", False)
    use_ew_eval = meta.get("use_edge_weight_attn", True)
    predictor = LinkPredictor(int(meta.get("gt_hidden_dim", args.gt_hidden_dim))).to(device)
    gt_model = GraphTransformer(
        in_dim,
        int(meta.get("gt_hidden_dim", args.gt_hidden_dim)),
        args.gt_num_heads,
        int(meta.get("gt_num_layers", args.gt_num_layers)),
        use_edge_weight_attn=use_ew_eval,
        use_directed_edge_bias=use_dir_eval,
    ).to(device)
    expr_enc_saved = meta.get("expr_encoder", "transformer")
    sd_seq = ckpt["seq_model"]
    if expr_enc_saved == "scgpt":
        if "scgpt_gene_emb" not in sd_seq:
            raise SystemExit("checkpoint meta says expr_encoder=scgpt but seq_model has no scgpt_gene_emb buffer")
        seq_model = ScGPTGeneExpressionEncoder(
            sd_seq["scgpt_gene_emb"].cpu(),
            d_model=int(meta.get("d_model", in_dim)),
            refine_layers=int(meta.get("scgpt_refine_layers", 0)),
            refine_nhead=int(meta.get("scgpt_refine_nhead", 4)),
            dim_feedforward=int(meta.get("scgpt_refine_dim_feedforward", 256)),
            dropout=float(args.dropout),
        ).to(device)
        seq_model.load_state_dict(sd_seq)
    elif expr_enc_saved == "scgpt_concat":
        if "fusion.weight" not in sd_seq or "scgpt_gene_emb" not in sd_seq:
            raise SystemExit(
                "checkpoint meta says expr_encoder=scgpt_concat but seq_model missing fusion.* or scgpt_gene_emb"
            )
        seq_model = ScGPTExprConcatEncoder(
            sd_seq["scgpt_gene_emb"].cpu(),
            n_cells=int(meta.get("n_cells", n_cells)),
            d_model=int(meta.get("d_model", in_dim)),
            nhead=int(meta.get("nhead", args.nhead)),
            num_layers=int(meta.get("num_layers", args.num_layers)),
            dim_feedforward=int(meta.get("dim_feedforward", args.dim_feedforward)),
            dropout=float(args.dropout),
        ).to(device)
        seq_model.load_state_dict(sd_seq)
    else:
        seq_model = GeneExpressionTransformer(
            n_cells=n_cells,
            d_model=in_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)
        seq_model.load_state_dict(sd_seq)
    gt_model.load_state_dict(ckpt["gt_model"])
    predictor.load_state_dict(ckpt["predictor"])

    te_logits = logits_all_edges(
        seq_model,
        gt_model,
        predictor,
        gene_expr,
        g,
        pe,
        test_ei,
        args.edge_batch_size,
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
        "dataset_key": (args.dataset or None),
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
    return results


def train_grn_dataset(dataset_key: str, **kwargs) -> dict:
    """Programmatic single-dataset training. ``dataset_key`` must be in :data:`DATASET_REGISTRY`.
    ``kwargs`` override argparse defaults (e.g. ``epochs``, ``lr``, ``output_dir``, ``prior_graph``).
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_key={dataset_key!r}; expected one of {list(DATASET_REGISTRY)}"
        )
    args = parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)
    dd, sd = DATASET_REGISTRY[dataset_key]
    args.data_dir = dd
    args.split_dir = sd if "split_dir" not in kwargs else args.split_dir
    args.dataset = dataset_key
    if "output_dir" not in kwargs:
        args.output_dir = str(_ROOT / "out_grn" / dataset_key)
    return run_training(args)


def main():
    args = parse_args()
    if args.train_all and args.dataset:
        raise SystemExit("Use either --train_all or --dataset, not both.")
    if args.train_all:
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        per_ds: dict[str, dict] = {}
        for name in DATASET_ORDER:
            sub = Namespace(**vars(args))
            sub.train_all = False
            sub.dataset = name
            sub.data_dir, sub.split_dir = DATASET_REGISTRY[name]
            sub.output_dir = str(root / name)
            print(f"\n{'=' * 72}\nDataset {name}  ->  {sub.output_dir}\n{'=' * 72}\n", flush=True)
            per_ds[name] = run_training(sub)
        # Aggregate summary (mean of test metrics where finite)
        metric_keys = [
            "test_auroc",
            "test_auprc",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_accuracy",
            "best_val_auroc",
        ]
        aggregate: dict[str, float] = {}
        for mk in metric_keys:
            vals = []
            for name in DATASET_ORDER:
                v = per_ds[name].get(mk)
                if v is not None and isinstance(v, (int, float)) and v == v:  # not NaN
                    vals.append(float(v))
            if vals:
                aggregate[f"mean_{mk}"] = float(np.mean(vals))
        all_results = {
            "datasets": {
                k: {
                    **{m: per_ds[k].get(m) for m in metric_keys},
                    "dataset_key": per_ds[k].get("dataset_key"),
                    "n_edges_correlation_graph": per_ds[k].get("n_edges_correlation_graph"),
                    "output_dir": str(root / k),
                }
                for k in DATASET_ORDER
            },
            "aggregate": aggregate,
            "config": vars(args),
        }
        with open(root / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote aggregate summary: {root / 'all_results.json'}", flush=True)
        return

    if args.dataset:
        if args.dataset not in DATASET_REGISTRY:
            raise SystemExit(
                f"Unknown --dataset {args.dataset!r}; choose one of {list(DATASET_REGISTRY)}"
            )
        args.data_dir, args.split_dir = DATASET_REGISTRY[args.dataset]

    run_training(args)


if __name__ == "__main__":
    main()
