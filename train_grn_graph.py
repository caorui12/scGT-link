#!/usr/bin/env python3
"""
scGT-Link training entrypoint (``train_grn_graph.py``).

Stage 1: Prior graph — ``correlation``: Pearson |r| top-k; ``gold`` / ``gold_pearson``: edges from ``--gold_network_file``
         (**gene symbols** in CSV, matching expression row names); ``gold_pearson`` sets ``edata['w']`` from Pearson r.
         ``train_pos_pearson``: **only** ``Train_set`` rows with Label=1 as directed edges; weights from Pearson r (no CSV).
         ``edata['w']``: Pearson r (min–max normalized); self-loops 1.0.
         For ``prior_graph=gold``, ``gold_pearson``, or ``train_pos_pearson``, ``edata['dir']`` is +1 on TF→target edges and −1 on mirrored edges
         (unless ``--no_edge_dir_attn``); SparseMHA adds ``edge_dir_scale * dir`` to logits.
Stage 2: Gene encoder + GraphTransformer + LinkPredictor.
         **scGT-Link v2** defaults: ``expr_encoder=scgpt`` (scGPT token emb + LapPE; ``gene_expr`` is ignored by the
         node encoder) + ``prior_graph=train_pos_pearson`` + ``gt_variant=directed``.
         Expression matrix is still loaded for Pearson prior weights / LapPE graph construction when required.
         Keep ``--expr_encoder scgpt_concat`` for the v1 dual-encoder ablation. SparseMHA adds learnable-scaled
         ``w`` to attention logits unless ``--no_edge_weight_attn``.

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
import time
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
from utils import LinkPredictor  # noqa: E402

from correlation_graph import build_correlation_graph  # noqa: E402
from gold_network_graph import (  # noqa: E402
    build_gold_network_graph,
    build_gold_pearson_graph,
    build_train_positive_pearson_graph,
)
from grnboost2_graph import build_grnboost2_graph, resolve_tf_gene_names  # noqa: E402
from grn_data import load_edge_split, load_expression_for_grn  # noqa: E402
from grn_model import (  # noqa: E402
    GeneExpressionTransformer,
    ScGPTExprConcatEncoder,
    ScGPTExprParallelEncoder,
    ScGPTExprTripleBranchEncoder,
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
        args.expr_encoder in ("scgpt", "scgpt_concat", "scgpt_parallel", "scgpt_triple")
        and args.scgpt_emb_pt.strip() == _DEFAULT_TUNE_SCGPT_REL
    ):
        args.scgpt_emb_pt = str(data_dir / "scgpt_gene_emb.pt")


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="scGT-Link: prior graph (Pearson / gold) + graph transformer GRN")
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
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training in --output_dir from last.pt (or best.pt + progress if last.pt missing).",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Explicit checkpoint path to resume (overrides --resume search).",
    )
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
        choices=("transformer", "scgpt", "scgpt_concat", "scgpt_parallel", "scgpt_triple"),
        default="scgpt",
        help="scGT-Link v2 default=scgpt: scGPT token emb + proj (+ optional refine) + LapPE; "
        "node features ignore gene_expr under scgpt (matrix still used for priors/LapPE). "
        "transformer: Linear(n_cells)+LapPE+TransformerEncoder. "
        "scgpt_concat: expression Transformer + LapPE, concat with proj(scGPT), Linear fuse (v1). "
        "scgpt_parallel: GeneTransformerOnly (no Linear/LapPE on expr), parallel fuse "
        "expr + scGPT + LapPE at GraphTransformer input (GT-GRN style). "
        "scgpt_triple: expr Transformer (no LapPE inside), scGPT, LapPE as three branches; "
        "concat-fuse before GT (no second LapPE at GT entry).",
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
        "0 = linear proj (+ LapPE) only, no gene-level Transformer; "
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
        help="GraphTransformer message-passing layers (default 6). "
        "0 = skip GT layers; encoder(X)+LapPE fusion still runs.",
    )
    p.add_argument(
        "--gt_num_heads",
        type=int,
        default=4,
        help="Sparse MHA heads; --gt_hidden_dim %% this must be 0 (e.g. 80/6 is invalid).",
    )
    p.add_argument(
        "--gt_variant",
        type=str,
        choices=("legacy", "directed"),
        default="directed",
        help="GraphTransformer: directed=dst-aggregation + edge-conditioned messages + "
        "LayerNorm/dropout (default; vectorized SparseMHA). "
        "legacy=src-aggregation+BN for older checkpoints / ablation.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="prior_graph=correlation (no --corr_min_abs): top-k neighbors per gene by |Pearson r|. "
        "prior_graph=grnboost2: top-k incoming links per target after GRNBoost2 inference.",
    )
    p.add_argument(
        "--corr_min_abs",
        type=float,
        default=None,
        metavar="T",
        help="prior_graph=correlation: keep edges with |Pearson r| >= T (e.g. 0.1). "
        "When set, --top_k is ignored.",
    )
    p.add_argument(
        "--prior_graph",
        type=str,
        choices=(
            "correlation",
            "gold",
            "gold_pearson",
            "train_pos_pearson",
            "grnboost2",
        ),
        default="train_pos_pearson",
        help="scGT-Link v2 default=train_pos_pearson (train-only positives + Pearson r; avoids gold leakage). "
        "correlation: Pearson |r| top-k graph. gold: reference network from --gold_network_file (optional CSV weights). "
        "gold_pearson: gold edges + Pearson r as weights (directed structure via edata['dir']). "
        "train_pos_pearson: only Train_set Label=1 edges + Pearson r (ignores --gold_network_file; aligns with train-only adjacency). "
        "grnboost2: arboreto GRNBoost2 prior + top_k pruning (requires TF names).",
    )
    p.add_argument(
        "--gold_network_file",
        type=str,
        default=_DEFAULT_TUNE_GOLD_REL,
        help="prior_graph=gold or gold_pearson: CSV with TF/Target or Gene1/Gene2 (etc.); "
        "endpoint values must be gene symbols matching expression row names (not integer indices). "
        "See gold_network_graph.build_gold_network_graph / build_gold_pearson_graph.",
    )
    p.add_argument(
        "--ckpt_metric",
        type=str,
        choices=("auroc", "auprc"),
        default="auroc",
        help="Validation metric used to save best.pt (default: auroc). "
        "Use auprc on highly imbalanced nets (e.g. STRING) to prioritize top-precision.",
    )
    p.add_argument(
        "--hard_neg_mix",
        type=float,
        default=0.0,
        metavar="R",
        help="Hard-negative mining: fraction of training negatives drawn from the hard pool "
        "(highest predicted scores). 0=disabled (default). E.g. 0.7 with --hard_neg_pool_frac 0.25.",
    )
    p.add_argument(
        "--hard_neg_pool_frac",
        type=float,
        default=0.25,
        metavar="F",
        help="Hard pool = top fraction of train negatives by logit each epoch (default 0.25).",
    )
    p.add_argument(
        "--hard_neg_start_epoch",
        type=int,
        default=3,
        help="Start hard-negative mining at this epoch (earlier epochs use all train edges).",
    )
    p.add_argument(
        "--same_tf_hard_neg",
        action="store_true",
        help="Rebuild train (and val) as 1:1 same-TF hard negatives per positive "
        "(GNNLink/GENELink-style). Test set unchanged. Recommended for STRING AUPRC.",
    )
    p.add_argument(
        "--same_tf_hard_neg_refresh",
        action="store_true",
        help="With --same_tf_hard_neg: resample train hard negatives every epoch "
        "(default: sample once at start).",
    )
    p.add_argument(
        "--link_decoder",
        type=str,
        choices=("concat", "rich"),
        default="concat",
        help="LinkPredictor: concat=[hi||hj] MLP (default, matches older ckpts); "
        "rich=fuse concat + Hadamard + |hi-hj| + bilinear channels then MLP.",
    )
    p.add_argument(
        "--link_bilinear_dim",
        type=int,
        default=0,
        help="rich decoder only: bilinear output dim (0 = max(8, gt_hidden_dim//8)).",
    )
    p.add_argument(
        "--tf_genes_file",
        type=str,
        default="",
        help="Optional: one TF gene symbol per line (for prior_graph=grnboost2). "
        "Default: data_dir/BL--TFs.txt or TF indices from splits.",
    )
    p.add_argument(
        "--grnboost2_limit",
        type=int,
        default=None,
        help="prior_graph=grnboost2: keep at most this many links globally (by importance) before per-target top_k.",
    )
    p.add_argument(
        "--grnboost2_n_estimators",
        type=int,
        default=500,
        help="prior_graph=grnboost2: LightGBM n_estimators per target (default 500).",
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
        help="For prior_graph=gold, gold_pearson, train_pos_pearson, or grnboost2: do not add learnable edata['dir'] (±1) bias in SparseMHA. "
        "Ignored for correlation prior (no dir field).",
    )
    p.add_argument(
        "--no_lappe",
        action="store_true",
        help="Ablation: zero Laplacian positional encoding (expr encoder + GraphTransformer entry). "
        "Prior graph structure for message passing is unchanged.",
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


def _uses_triple_gt_fuse(expr_encoder: str) -> bool:
    return expr_encoder == "scgpt_parallel"


def _uses_skip_pos_at_gt(expr_encoder: str) -> bool:
    return expr_encoder == "scgpt_triple"


def _fmt_duration(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _load_progress_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _make_ckpt_meta(
    *,
    n_genes: int,
    n_cells: int,
    in_dim: int,
    args: Namespace,
    triple_fuse: bool,
    skip_pos_at_gt: bool,
    seq_model: nn.Module,
    scgpt_dim: Optional[int],
    n_edges_graph: int,
    use_edge_w: bool,
    use_dir_attn: bool,
) -> dict:
    return {
        "n_genes": n_genes,
        "n_cells": n_cells,
        "d_model": in_dim,
        "expr_encoder": args.expr_encoder,
        "triple_gt_fuse": triple_fuse,
        "skip_pos_at_gt": skip_pos_at_gt,
        "expr_out_dim": int(getattr(seq_model, "expr_out_dim", n_cells)),
        "scgpt_dim": scgpt_dim,
        "scgpt_refine_layers": args.scgpt_refine_layers,
        "scgpt_refine_nhead": args.scgpt_refine_nhead,
        "scgpt_refine_dim_feedforward": args.scgpt_refine_dim_feedforward,
        "gt_hidden_dim": args.gt_hidden_dim,
        "gt_num_layers": args.gt_num_layers,
        "gt_num_heads": args.gt_num_heads,
        "gt_variant": getattr(args, "gt_variant", "directed"),
        "hard_neg_mix": float(getattr(args, "hard_neg_mix", 0.0) or 0.0),
        "hard_neg_pool_frac": float(getattr(args, "hard_neg_pool_frac", 0.25) or 0.25),
        "hard_neg_start_epoch": int(getattr(args, "hard_neg_start_epoch", 3)),
        "same_tf_hard_neg": bool(getattr(args, "same_tf_hard_neg", False)),
        "same_tf_hard_neg_refresh": bool(getattr(args, "same_tf_hard_neg_refresh", False)),
        "link_decoder": getattr(args, "link_decoder", "concat"),
        "link_bilinear_dim": int(getattr(args, "link_bilinear_dim", 0) or 0),
        "top_k": args.top_k,
        "corr_min_abs": args.corr_min_abs,
        "n_edges_graph": int(n_edges_graph),
        "use_edge_weight_attn": use_edge_w,
        "prior_graph": args.prior_graph,
        "gold_network_file": str(Path(args.gold_network_file).resolve())
        if args.prior_graph in ("gold", "gold_pearson") and args.gold_network_file.strip()
        else "",
        "gold_network_endpoints": (
            "gene_symbol"
            if args.prior_graph in ("gold", "gold_pearson")
            else (
                "train_positive_indices"
                if args.prior_graph == "train_pos_pearson"
                else ""
            )
        ),
        "use_directed_edge_attn": use_dir_attn,
        "no_lappe": bool(args.no_lappe),
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
    }


def _link_predictor_from_meta_or_args(meta: dict, args: Namespace) -> LinkPredictor:
    hidden = int(meta.get("gt_hidden_dim", getattr(args, "gt_hidden_dim", 128)))
    mode = str(meta.get("link_decoder", getattr(args, "link_decoder", "concat")))
    bdim_raw = meta.get("link_bilinear_dim", getattr(args, "link_bilinear_dim", 0))
    bdim = int(bdim_raw) if bdim_raw else None
    drop = float(getattr(args, "dropout", 0.1))
    return LinkPredictor(hidden, mode=mode, dropout=drop, bilinear_dim=bdim)


def _gt_variant_from_meta_or_args(meta: dict, args: Namespace) -> str:
    if "gt_variant" in meta:
        return str(meta["gt_variant"])
    # Checkpoints saved before this flag used the legacy GT.
    if meta:
        return "legacy"
    return str(getattr(args, "gt_variant", "directed"))


def _resume_training_state(
    out_dir: Path,
    args: Namespace,
    device: torch.device,
    seq_model: nn.Module,
    gt_model: GraphTransformer,
    predictor: LinkPredictor,
    opt: torch.optim.Optimizer,
) -> tuple[int, float, int, list[dict], float]:
    """
    Restore weights (and optimizer when possible). Returns
    (start_epoch, best_val_score, best_epoch, history, elapsed_sec_prior).
    ``best_val_score`` follows ``args.ckpt_metric`` (val AUROC or val AUPRC).
    """
    resume_path: Path | None = None
    partial = False
    resume_from = getattr(args, "resume_from", "").strip()
    ckpt_metric = getattr(args, "ckpt_metric", "auroc")
    score_key = "val_auprc" if ckpt_metric == "auprc" else "val_auroc"
    if resume_from:
        resume_path = Path(resume_from).expanduser().resolve()
    elif args.resume:
        last_pt = out_dir / "last.pt"
        if last_pt.is_file():
            resume_path = last_pt
        elif (out_dir / "best.pt").is_file() and (out_dir / "training_progress.jsonl").is_file():
            resume_path = out_dir / "best.pt"
            partial = True

    if resume_path is None or not resume_path.is_file():
        return 1, -1.0, 0, [], 0.0

    ckpt = torch.load(resume_path, map_location=device)
    seq_model.load_state_dict(ckpt["seq_model"])
    gt_model.load_state_dict(ckpt["gt_model"])
    predictor.load_state_dict(ckpt["predictor"])

    history: list[dict] = list(ckpt.get("history") or [])
    if not history:
        history = _load_progress_jsonl(out_dir / "training_progress.jsonl")

    timing = ckpt.get("timing") or {}
    elapsed_prior = float(timing.get("elapsed_sec_total", 0.0))

    if partial or "epoch" not in ckpt:
        if not history:
            raise SystemExit(f"Cannot resume from {resume_path}: missing training history.")
        last_ep = int(history[-1]["epoch"])
        best_row = max(history, key=lambda r: float(r.get(score_key, r["val_auroc"])))
        best_val = float(best_row.get(score_key, best_row["val_auroc"]))
        best_ep = int(best_row["epoch"])
        print(
            f"[resume] partial from {resume_path.name}: model weights restored; optimizer reset; "
            f"continue at epoch {last_ep + 1}/{args.epochs} (best {ckpt_metric} @ ep {best_ep})",
            flush=True,
        )
        return last_ep + 1, best_val, best_ep, history, elapsed_prior

    if ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])

    start_epoch = int(ckpt["epoch"]) + 1
    if "best_val_score" in ckpt:
        best_val = float(ckpt["best_val_score"])
    elif ckpt_metric == "auprc" and "best_val_auprc" in ckpt:
        best_val = float(ckpt["best_val_auprc"])
    else:
        best_val = float(ckpt.get("best_val_auroc", -1.0))
    best_ep = int(ckpt.get("best_epoch", 0))
    print(
        f"[resume] from {resume_path.name}: continue epoch {start_epoch}/{args.epochs} "
        f"(best {ckpt_metric} {best_val:.4f} @ ep {best_ep}, elapsed {_fmt_duration(elapsed_prior)})",
        flush=True,
    )
    return start_epoch, best_val, best_ep, history, elapsed_prior


def _state_dict_has_module(state_dict: dict, prefix: str) -> bool:
    return any(k == prefix or k.startswith(f"{prefix}.") for k in state_dict)


def _graph_transformer_hidden(
    seq_model: nn.Module,
    gt_model: GraphTransformer,
    gene_expr: torch.Tensor,
    g: dgl.DGLGraph,
    pe: torch.Tensor,
    *,
    triple_fuse: bool,
) -> torch.Tensor:
    if triple_fuse:
        h_expr, h_scgpt = seq_model(gene_expr, pe)
        return gt_model(g, None, pe, h_expr=h_expr, h_scgpt=h_scgpt)
    h_seq = seq_model(gene_expr, pe)
    return gt_model(g, h_seq, pe)


@torch.no_grad()
def _hard_neg_train_indices(
    seq_model: nn.Module,
    gt_model: GraphTransformer,
    predictor: LinkPredictor,
    gene_expr: torch.Tensor,
    g: dgl.DGLGraph,
    pe: torch.Tensor,
    train_ei: torch.Tensor,
    train_y: torch.Tensor,
    *,
    hard_mix: float,
    hard_pool_frac: float,
    triple_fuse: bool,
    score_chunk: int = 8192,
) -> torch.Tensor:
    """Epoch index pool: all positives + hard/random negatives (same #negs as full train)."""
    device = train_y.device
    pos_idx = (train_y > 0.5).nonzero(as_tuple=False).view(-1)
    neg_idx = (train_y <= 0.5).nonzero(as_tuple=False).view(-1)
    n_neg = int(neg_idx.numel())
    if n_neg == 0 or hard_mix <= 0:
        return torch.arange(train_y.numel(), device=device)

    hard_mix = float(min(max(hard_mix, 0.0), 1.0))
    hard_pool_frac = float(min(max(hard_pool_frac, 1e-6), 1.0))

    was_training = (seq_model.training, gt_model.training, predictor.training)
    seq_model.eval()
    gt_model.eval()
    predictor.eval()
    h = _graph_transformer_hidden(
        seq_model, gt_model, gene_expr, g, pe, triple_fuse=triple_fuse
    )
    chunks = []
    for s in range(0, n_neg, score_chunk):
        sl = neg_idx[s : s + score_chunk]
        chunks.append(predictor(h, train_ei[:, sl]).squeeze(-1))
    neg_logits = torch.cat(chunks, dim=0)

    n_hard = int(round(hard_mix * n_neg))
    n_hard = min(max(n_hard, 0), n_neg)
    n_rand = n_neg - n_hard
    pool_k = max(n_hard, int(round(hard_pool_frac * n_neg)))
    pool_k = min(max(pool_k, 1), n_neg)
    hard_local = torch.topk(neg_logits, k=pool_k, largest=True).indices
    hard_pool = neg_idx[hard_local]

    if n_hard > 0:
        if n_hard <= hard_pool.numel():
            pick = torch.randperm(hard_pool.numel(), device=device)[:n_hard]
            hard_sel = hard_pool[pick]
        else:
            pick = torch.randint(0, hard_pool.numel(), (n_hard,), device=device)
            hard_sel = hard_pool[pick]
    else:
        hard_sel = neg_idx.new_empty(0)

    if n_rand > 0:
        easy_mask = torch.ones(n_neg, dtype=torch.bool, device=device)
        easy_mask[hard_local] = False
        easy_pool = neg_idx[easy_mask]
        if easy_pool.numel() == 0:
            easy_pool = neg_idx
        if n_rand <= easy_pool.numel():
            pick = torch.randperm(easy_pool.numel(), device=device)[:n_rand]
            rand_sel = easy_pool[pick]
        else:
            pick = torch.randint(0, easy_pool.numel(), (n_rand,), device=device)
            rand_sel = easy_pool[pick]
    else:
        rand_sel = neg_idx.new_empty(0)

    pool = torch.cat([pos_idx, hard_sel, rand_sel])
    pool = pool[torch.randperm(pool.numel(), device=device)]

    if was_training[0]:
        seq_model.train()
    if was_training[1]:
        gt_model.train()
    if was_training[2]:
        predictor.train()
    return pool


def _resample_same_tf_hard_neg(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    n_genes: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GNNLink/GENELink-style: one same-TF hard negative per positive edge.

    For each positive (tf, target), sample (tf, g) with g not equal to tf and not any
    positive target of that tf in *this* edge set. Returns balanced 1:1 tensors on the
    same device as ``edge_index``.
    """
    device = edge_index.device
    y = labels.view(-1)
    pos_mask = y > 0.5
    if not bool(pos_mask.any()):
        raise RuntimeError("same_tf_hard_neg: no positive edges in the provided split.")
    pos_ei = edge_index[:, pos_mask]
    tfs = pos_ei[0].tolist()
    tgs = pos_ei[1].tolist()

    forbidden: dict[int, set[int]] = {}
    for tf, tg in zip(tfs, tgs):
        forbidden.setdefault(int(tf), set()).add(int(tg))
        forbidden[int(tf)].add(int(tf))

    neg_cols: list[list[int]] = []
    gene_ids = list(range(int(n_genes)))
    for tf, _tg in zip(tfs, tgs):
        tf_i = int(tf)
        ban = forbidden.get(tf_i, {tf_i})
        cand = [g for g in gene_ids if g not in ban]
        if not cand:
            # Extremely dense TF: fall back to any non-self gene.
            cand = [g for g in gene_ids if g != tf_i]
        if not cand:
            raise RuntimeError(f"same_tf_hard_neg: TF={tf_i} has no valid negative target.")
        if generator is None:
            g = int(cand[int(torch.randint(0, len(cand), (1,)).item())])
        else:
            g = int(cand[int(torch.randint(0, len(cand), (1,), generator=generator).item())])
        neg_cols.append([tf_i, g])

    neg_ei = torch.tensor(neg_cols, dtype=edge_index.dtype, device=device).t().contiguous()
    out_ei = torch.cat([pos_ei, neg_ei], dim=1)
    out_y = torch.cat(
        [
            torch.ones(pos_ei.size(1), dtype=labels.dtype, device=device),
            torch.zeros(neg_ei.size(1), dtype=labels.dtype, device=device),
        ]
    )
    perm = torch.randperm(out_ei.size(1), device=device, generator=generator)
    return out_ei[:, perm], out_y[perm]


@torch.no_grad()
def evaluate_link_prediction(
    seq_model: nn.Module,
    gt_model: GraphTransformer,
    predictor: LinkPredictor,
    gene_expr: torch.Tensor,
    g: dgl.DGLGraph,
    pe: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    *,
    triple_fuse: bool,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    seq_model.eval()
    gt_model.eval()
    predictor.eval()
    h = _graph_transformer_hidden(
        seq_model, gt_model, gene_expr, g, pe, triple_fuse=triple_fuse
    )
    pos_logits = predictor(h, pos_edges)
    neg_logits = predictor(h, neg_edges)
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return (
        roc_auc_score(labels.cpu(), logits.cpu()),
        average_precision_score(labels.cpu(), logits.cpu()),
        labels.cpu(),
        logits.cpu(),
    )


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
    *,
    triple_fuse: bool,
) -> torch.Tensor:
    seq_model.eval()
    gt_model.eval()
    predictor.eval()
    h = _graph_transformer_hidden(
        seq_model, gt_model, gene_expr, g, pe, triple_fuse=triple_fuse
    )
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
    if args.expr_encoder in ("transformer", "scgpt_concat", "scgpt_triple") and args.d_model % args.nhead != 0:
        raise SystemExit(
            f"--d_model ({args.d_model}) must be divisible by --nhead ({args.nhead})"
        )
    triple_fuse = _uses_triple_gt_fuse(args.expr_encoder)
    skip_pos_at_gt = _uses_skip_pos_at_gt(args.expr_encoder)
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
        g = build_correlation_graph(
            expr_cpu,
            top_k=args.top_k,
            min_abs_corr=args.corr_min_abs,
        )
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
    elif args.prior_graph == "grnboost2":
        try:
            tf_names = resolve_tf_gene_names(
                data_dir,
                split_dir,
                gene_names,
                args.tf_genes_file.strip() or None,
            )
            g = build_grnboost2_graph(
                expr_cpu,
                gene_names,
                tf_names,
                top_k=args.top_k,
                seed=args.seed,
                grnboost2_limit=args.grnboost2_limit,
                n_estimators=args.grnboost2_n_estimators,
            )
        except (ImportError, FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e)) from e
    else:
        raise SystemExit(f"Unknown prior_graph={args.prior_graph!r}")
    g = g.to(device)
    with torch.no_grad():
        pe = dgl.lap_pe(g, k=in_dim, padding=True)
    pe = pe.to(device=device, dtype=gene_expr.dtype)
    if args.no_lappe:
        pe = torch.zeros_like(pe)

    scgpt_dim: Optional[int] = None
    if args.expr_encoder in ("scgpt", "scgpt_concat", "scgpt_parallel", "scgpt_triple"):
        if not args.scgpt_emb_pt.strip() and not args.scgpt_model_dir.strip():
            raise SystemExit(
                "expr_encoder=scgpt, scgpt_concat, scgpt_parallel, or scgpt_triple requires --scgpt_emb_pt "
                "and/or --scgpt_model_dir (precomputed .pt or scGPT checkpoint directory)."
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
        if args.expr_encoder == "scgpt_parallel":
            seq_model = ScGPTExprParallelEncoder(
                scgpt_mat,
                n_cells=n_cells,
                d_model=in_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
            ).to(device)
        elif args.expr_encoder == "scgpt_concat":
            seq_model = ScGPTExprConcatEncoder(
                scgpt_mat,
                n_cells=n_cells,
                d_model=in_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
            ).to(device)
        elif args.expr_encoder == "scgpt_triple":
            seq_model = ScGPTExprTripleBranchEncoder(
                scgpt_mat,
                n_cells=n_cells,
                d_model=in_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
            ).to(device)
        else:
            # expr_encoder=scgpt (v2): node features = proj(scGPT) + LapPE; gene_expr ignored in encoder.
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
        args.prior_graph
        in (
            "gold",
            "gold_pearson",
            "train_pos_pearson",
            "grnboost2",
        )
        and not args.no_edge_dir_attn
    )
    gt_model = GraphTransformer(
        in_dim,
        args.gt_hidden_dim,
        args.gt_num_heads,
        args.gt_num_layers,
        use_edge_weight_attn=use_edge_w,
        use_directed_edge_bias=use_dir_attn,
        triple_fuse=triple_fuse,
        skip_pos_at_gt=skip_pos_at_gt,
        expr_in_dim=(
            seq_model.expr_out_dim
            if triple_fuse and hasattr(seq_model, "expr_out_dim")
            else n_cells if triple_fuse else None
        ),
        scgpt_in_dim=in_dim if triple_fuse else None,
        gt_variant=getattr(args, "gt_variant", "directed"),
        dropout=float(args.dropout),
    ).to(device)
    predictor = _link_predictor_from_meta_or_args({}, args).to(device)

    if bool(getattr(args, "same_tf_hard_neg", False)):
        # Keep prior graph built from original train positives; only rebuild supervised train edges.
        # Val stays density-style (from split files) so ckpt_metric=auprc matches the imbalanced test.
        n_tr_pos = int((train_y > 0.5).sum().item())
        train_ei, train_y = _resample_same_tf_hard_neg(train_ei, train_y, n_genes)
        print(
            f"[same_tf_hard_neg] train {n_tr_pos} pos + {n_tr_pos} hard-neg | "
            f"val/test unchanged (val pos={int((val_y > 0.5).sum())} "
            f"neg={int((val_y <= 0.5).sum())}; test pos={int((test_y > 0.5).sum())} "
            f"neg={int((test_y <= 0.5).sum())})",
            flush=True,
        )

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
            if args.expr_encoder in ("scgpt", "scgpt_concat", "scgpt_parallel", "scgpt_triple") and scgpt_dim is not None
            else ""
        )
        + f" | prior={args.prior_graph} edges={n_edges_graph} | "
        f"train edges={n_train} (pos={int(n_pos)} neg={int(n_neg)}) | pos_weight={pos_weight.item():.3f}"
        + f" | edge_weight_attn={'on' if use_edge_w else 'off'}"
        + (
            f" | edge_dir_attn={'on' if use_dir_attn else 'off'}"
            if args.prior_graph
            in (
                "gold",
                "gold_pearson",
                "train_pos_pearson",
                "grnboost2",
            )
            else ""
        )
        + f" | gt_variant={getattr(args, 'gt_variant', 'directed')}"
        + f" | ckpt_metric={getattr(args, 'ckpt_metric', 'auroc')}"
        + (
            f" | hard_neg_mix={args.hard_neg_mix:g}/pool={args.hard_neg_pool_frac:g}"
            f"@ep>={args.hard_neg_start_epoch}"
            if float(getattr(args, "hard_neg_mix", 0) or 0) > 0
            else ""
        )
        + (" | same_tf_hard_neg=on" if bool(getattr(args, "same_tf_hard_neg", False)) else "")
        + f" | lappe={'off' if args.no_lappe else 'on'}",
        flush=True,
    )
    print("-" * 72, flush=True)

    progress_path = out_dir / "training_progress.jsonl"
    if not args.resume and not getattr(args, "resume_from", "").strip():
        progress_path.write_text("", encoding="utf-8")

    start_epoch, best_val_score, best_epoch, history, elapsed_prior = _resume_training_state(
        out_dir, args, device, seq_model, gt_model, predictor, opt
    )
    train_t0 = time.perf_counter() - elapsed_prior
    ckpt_metric = getattr(args, "ckpt_metric", "auroc")
    best_val_auroc = -1.0
    best_val_auprc = -1.0
    if history:
        best_val_auroc = max(float(r["val_auroc"]) for r in history)
        best_val_auprc = max(float(r["val_auprc"]) for r in history)

    if start_epoch > args.epochs:
        print(
            f"[resume] already completed {history[-1]['epoch'] if history else 0}/{args.epochs} epochs; "
            "running test eval from best.pt",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        seq_model.train()
        gt_model.train()
        predictor.train()
        hard_mix = float(getattr(args, "hard_neg_mix", 0.0) or 0.0)
        if bool(getattr(args, "same_tf_hard_neg", False)) and bool(
            getattr(args, "same_tf_hard_neg_refresh", False)
        ):
            # Resample train hard negatives; keep current positives, new same-TF negs.
            train_ei, train_y = _resample_same_tf_hard_neg(train_ei, train_y, n_genes)
            n_train = train_ei.size(1)
            n_pos = float((train_y == 1).sum().item())
            n_neg = float((train_y == 0).sum().item())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            val_pos = val_ei[:, val_y == 1]
            val_neg = val_ei[:, val_y == 0]
        if hard_mix > 0 and epoch >= int(getattr(args, "hard_neg_start_epoch", 3)):
            idx_perm = _hard_neg_train_indices(
                seq_model,
                gt_model,
                predictor,
                gene_expr,
                g,
                pe,
                train_ei,
                train_y,
                hard_mix=hard_mix,
                hard_pool_frac=float(getattr(args, "hard_neg_pool_frac", 0.25)),
                triple_fuse=triple_fuse,
            )
        else:
            idx_perm = torch.arange(n_train, device=device)
            idx_perm = idx_perm[torch.randperm(n_train, device=device)]
        epoch_loss = 0.0
        n_batches = 0
        n_epoch_edges = int(idx_perm.numel())
        for start in range(0, n_epoch_edges, args.edge_batch_size):
            sel = idx_perm[start : start + args.edge_batch_size]
            ei = train_ei[:, sel]
            y = train_y[sel]
            opt.zero_grad()
            h = _graph_transformer_hidden(
                seq_model, gt_model, gene_expr, g, pe, triple_fuse=triple_fuse
            )
            logits = predictor(h, ei).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        seq_model.eval()
        gt_model.eval()
        predictor.eval()
        v_auroc, v_ap, _, _ = evaluate_link_prediction(
            seq_model,
            gt_model,
            predictor,
            gene_expr,
            g,
            pe,
            val_pos,
            val_neg,
            triple_fuse=triple_fuse,
        )
        seq_model.train()
        gt_model.train()
        predictor.train()

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_auroc": float(v_auroc),
                "val_auprc": float(v_ap),
                "epoch_sec": round(time.perf_counter() - epoch_t0, 2),
            }
        )

        ckpt_meta = _make_ckpt_meta(
            n_genes=n_genes,
            n_cells=n_cells,
            in_dim=in_dim,
            args=args,
            triple_fuse=triple_fuse,
            skip_pos_at_gt=skip_pos_at_gt,
            seq_model=seq_model,
            scgpt_dim=scgpt_dim,
            n_edges_graph=n_edges_graph,
            use_edge_w=use_edge_w,
            use_dir_attn=use_dir_attn,
        )
        ckpt_meta["ckpt_metric"] = ckpt_metric
        elapsed_total = time.perf_counter() - train_t0

        best_val_auroc = max(best_val_auroc, float(v_auroc))
        best_val_auprc = max(best_val_auprc, float(v_ap))
        cur_score = float(v_ap if ckpt_metric == "auprc" else v_auroc)
        if cur_score > best_val_score:
            best_val_score = cur_score
            best_epoch = epoch
            best_ckpt: dict = {
                "seq_model": seq_model.state_dict(),
                "gt_model": gt_model.state_dict(),
                "predictor": predictor.state_dict(),
                "epoch": epoch,
                "best_val_score": best_val_score,
                "best_val_auroc": float(v_auroc),
                "best_val_auprc": float(v_ap),
                "best_epoch": best_epoch,
                "ckpt_metric": ckpt_metric,
                "meta": ckpt_meta,
            }
            torch.save(best_ckpt, out_dir / "best.pt")

        last_ckpt: dict = {
            "seq_model": seq_model.state_dict(),
            "gt_model": gt_model.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "best_val_score": best_val_score,
            "best_val_auroc": best_val_auroc,
            "best_val_auprc": best_val_auprc,
            "best_epoch": best_epoch,
            "ckpt_metric": ckpt_metric,
            "history": history,
            "timing": {
                "elapsed_sec_total": elapsed_total,
                "epoch_sec_last": history[-1]["epoch_sec"],
            },
            "meta": ckpt_meta,
        }
        torch.save(last_ckpt, out_dir / "last.pt")

        row = history[-1]
        done_epochs = epoch - start_epoch + 1
        avg_ep = (elapsed_total - elapsed_prior) / max(done_epochs, 1)
        remaining = max(args.epochs - epoch, 0) * avg_ep
        print(
            f"Epoch {epoch:3d}/{args.epochs} | train_loss={row['train_loss']:.4f} | "
            f"val_AUROC={v_auroc:.4f} val_AUPRC={v_ap:.4f} | "
            f"{row['epoch_sec']:.1f}s/ep | elapsed {_fmt_duration(elapsed_total)}"
            + (f" | ETA {_fmt_duration(remaining)}" if remaining > 0 else ""),
            flush=True,
        )
        with open(progress_path, "a", encoding="utf-8") as pf:
            pf.write(json.dumps(row) + "\n")
            pf.flush()

    elapsed_total = time.perf_counter() - train_t0
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    meta = ckpt.get("meta", {})
    use_dir_eval = meta.get("use_directed_edge_attn", False)
    use_ew_eval = meta.get("use_edge_weight_attn", True)
    expr_enc_saved = meta.get("expr_encoder", "transformer")
    triple_fuse_eval = bool(
        meta.get("triple_gt_fuse", _uses_triple_gt_fuse(expr_enc_saved))
    )
    skip_pos_at_gt_eval = bool(
        meta.get("skip_pos_at_gt", _uses_skip_pos_at_gt(expr_enc_saved))
    )
    predictor = _link_predictor_from_meta_or_args(meta, args).to(device)
    gt_model = GraphTransformer(
        in_dim,
        int(meta.get("gt_hidden_dim", args.gt_hidden_dim)),
        int(meta.get("gt_num_heads", args.gt_num_heads)),
        int(meta.get("gt_num_layers", args.gt_num_layers)),
        use_edge_weight_attn=use_ew_eval,
        use_directed_edge_bias=use_dir_eval,
        triple_fuse=triple_fuse_eval,
        skip_pos_at_gt=skip_pos_at_gt_eval,
        expr_in_dim=int(meta.get("expr_out_dim", meta.get("n_cells", n_cells)))
        if triple_fuse_eval
        else None,
        scgpt_in_dim=int(meta.get("d_model", in_dim)) if triple_fuse_eval else None,
        gt_variant=_gt_variant_from_meta_or_args(meta, args),
        dropout=float(args.dropout),
    ).to(device)
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
    elif expr_enc_saved == "scgpt_triple":
        if "fusion.weight" not in sd_seq or "scgpt_gene_emb" not in sd_seq:
            raise SystemExit(
                "checkpoint meta says expr_encoder=scgpt_triple but seq_model missing fusion.* or scgpt_gene_emb"
            )
        seq_model = ScGPTExprTripleBranchEncoder(
            sd_seq["scgpt_gene_emb"].cpu(),
            n_cells=int(meta.get("n_cells", n_cells)),
            d_model=int(meta.get("d_model", in_dim)),
            nhead=int(meta.get("nhead", args.nhead)),
            num_layers=int(meta.get("num_layers", args.num_layers)),
            dim_feedforward=int(meta.get("dim_feedforward", args.dim_feedforward)),
            dropout=float(args.dropout),
        ).to(device)
        seq_model.load_state_dict(sd_seq)
    elif expr_enc_saved == "scgpt_parallel":
        if not _state_dict_has_module(sd_seq, "expr_transformer") or "scgpt_gene_emb" not in sd_seq:
            raise SystemExit(
                "checkpoint meta says expr_encoder=scgpt_parallel but seq_model missing "
                "expr_transformer.* or scgpt_gene_emb"
            )
        seq_model = ScGPTExprParallelEncoder(
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
        triple_fuse=triple_fuse_eval,
    )
    test_metrics = compute_test_metrics(
        test_y.cpu().numpy(),
        te_logits.cpu().numpy(),
        prob_threshold=args.test_prob_threshold,
    )

    print("-" * 72, flush=True)
    print(
        f"Test AUROC: {test_metrics['test_auroc']:.4f}  Test AUPRC: {test_metrics['test_auprc']:.4f}  "
        f"(threshold={test_metrics['test_prob_threshold']} on sigmoid)",
        flush=True,
    )
    print(
        f"Test Precision: {test_metrics['test_precision']:.4f}  Recall: {test_metrics['test_recall']:.4f}  "
        f"F1: {test_metrics['test_f1']:.4f}  Accuracy: {test_metrics['test_accuracy']:.4f}",
        flush=True,
    )

    results = {
        **test_metrics,
        "best_val_auroc": best_val_auroc,
        "best_val_auprc": best_val_auprc,
        "best_val_score": best_val_score,
        "ckpt_metric": ckpt_metric,
        "best_epoch": best_epoch,
        "training_time_sec": round(elapsed_total, 2),
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
    print(
        f"Training time: {_fmt_duration(elapsed_total)} ({elapsed_total:.1f}s) | "
        f"best epoch {best_epoch} (ckpt_metric={ckpt_metric}, score={best_val_score:.4f})",
        flush=True,
    )
    print(
        f"Saved: {out_dir / 'results.json'}, {out_dir / 'best.pt'}, {out_dir / 'last.pt'}, "
        f"{out_dir / 'test_predictions.npz'}",
        flush=True,
    )
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
