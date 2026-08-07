#!/usr/bin/env python3
"""
scGT-Link v2 inference helper: load a saved ``best.pt``, rebuild the graph exactly as training,
score **test** (and optionally **train** / **val**) edges and write logits + CSV.

Typical (**hESC TF+500**, registry-aligned paths)::

  cd /path/to/Graph_Transformer
  python scripts/infer_grn_checkpoint.py \\
    --checkpoint out_gt_data_four/hESC_500/best.pt \\
    --data_dir data/hESC/TFs+500 \\
    --split_dir data/Train_validation_test/hESC_500 \\
    --output_dir out_infer/hESC_500_best

Hyperparameters match:

  * ``<checkpoint_dir>/results.json`` → ``config`` (if present), else
  * ``best.pt`` → ``meta`` (``gt_num_heads`` added in newer runs; default 4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_grn_graph import (  # noqa: E402
    _remap_tune_default_paths,
    _uses_skip_pos_at_gt,
    _uses_triple_gt_fuse,
    apply_split_subdir,
    compute_test_metrics,
    logits_all_edges,
    load_expression_for_grn,
    parse_args,
    set_seed,
)
from grn_model import ScGPTExprParallelEncoder  # noqa: E402
from grn_data import load_edge_split  # noqa: E402


def _apply_config_dict(args: Namespace, cfg: dict) -> None:
    """Overlay keys from saved ``results.json`` config onto ``args``."""
    for k, v in cfg.items():
        setattr(args, k, v)


def _apply_meta_fallback(args: Namespace, meta: dict) -> None:
    """When ``results.json`` is missing; use ckpt meta (older fields)."""
    args.expr_encoder = meta.get("expr_encoder", args.expr_encoder)
    args.d_model = int(meta.get("d_model", args.d_model))
    args.nhead = int(meta.get("nhead", args.nhead))
    args.num_layers = int(meta.get("num_layers", args.num_layers))
    args.dim_feedforward = int(meta.get("dim_feedforward", args.dim_feedforward))
    args.gt_hidden_dim = int(meta.get("gt_hidden_dim", args.gt_hidden_dim))
    args.gt_num_layers = int(meta.get("gt_num_layers", args.gt_num_layers))
    args.gt_num_heads = int(meta.get("gt_num_heads", getattr(args, "gt_num_heads", 4)))
    args.scgpt_refine_layers = int(meta.get("scgpt_refine_layers", args.scgpt_refine_layers))
    args.scgpt_refine_nhead = int(meta.get("scgpt_refine_nhead", args.scgpt_refine_nhead))
    args.scgpt_refine_dim_feedforward = int(
        meta.get("scgpt_refine_dim_feedforward", args.scgpt_refine_dim_feedforward)
    )
    args.top_k = int(meta.get("top_k", args.top_k))
    cma = meta.get("corr_min_abs", getattr(args, "corr_min_abs", None))
    args.corr_min_abs = float(cma) if cma is not None else None
    args.prior_graph = meta.get("prior_graph", args.prior_graph)
    gn = meta.get("gold_network_file") or ""
    args.gold_network_file = gn
    uw = meta.get("use_edge_weight_attn", True)
    args.no_edge_weight_attn = not bool(uw)
    ud = meta.get("use_directed_edge_attn", True)
    args.no_edge_dir_attn = not bool(ud)


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="scGT-Link v2: inference from best.pt (encoder/prior from checkpoint meta)")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "out_gt_data_four" / "hESC_500" / "best.pt"),
        help="Path to best.pt (validation-AUROC checkpoint).",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=str(_ROOT / "data" / "hESC" / "TFs+500"),
        help="Leaf with BL--ExpressionData.csv (+ scgpt_emb if needed).",
    )
    ap.add_argument(
        "--split_dir",
        type=str,
        default=str(_ROOT / "data" / "Train_validation_test" / "hESC_500"),
        help="Folder with Train_set.csv Validation_set.csv Test_set.csv",
    )
    ap.add_argument(
        "--split_subdir",
        type=str,
        default="Train_validation_test",
        help="Same as training (usually unused when split_dir is explicit).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(_ROOT / "out_infer" / "hESC_500_best"),
        help="Writes inference_metrics.json test_predictions.npz application_edges_<split>.csv",
    )
    ap.add_argument(
        "--edge_batch_size",
        type=int,
        default=None,
        help="Override chunk size for logits_all_edges (default: training default).",
    )
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument(
        "--export_splits",
        type=str,
        default="test",
        help="Comma list: train,val,test (which splits get application_edges_<split>.csv).",
    )
    ap.add_argument(
        "--test_prob_threshold",
        type=float,
        default=-1,
        help="If >=0 overrides threshold for precision/recall in metrics.",
    )
    return ap.parse_args()


def main() -> None:
    cli = parse_cli()
    ckpt_path = Path(cli.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    args = parse_args([])
    res_path = ckpt_path.parent / "results.json"
    if res_path.is_file():
        cfg = json.loads(res_path.read_text(encoding="utf-8")).get("config")
        if isinstance(cfg, dict):
            _apply_config_dict(args, cfg)
    else:
        ck = torch.load(ckpt_path, map_location="cpu")
        _apply_meta_fallback(args, ck.get("meta", {}))

    args.data_dir = str(Path(cli.data_dir).expanduser().resolve())
    args.split_dir = str(Path(cli.split_dir).expanduser().resolve())
    args.split_subdir = cli.split_subdir
    args.output_dir = str(Path(cli.output_dir).expanduser().resolve())
    args.no_cuda = cli.no_cuda
    if cli.edge_batch_size is not None:
        args.edge_batch_size = cli.edge_batch_size
    if cli.test_prob_threshold >= 0:
        args.test_prob_threshold = cli.test_prob_threshold

    args.split_dir = apply_split_subdir(args.split_dir, getattr(args, "split_subdir", "Train_validation_test"))
    _remap_tune_default_paths(args)

    set_seed(args.seed)
    import dgl  # noqa: E402

    from correlation_graph import build_correlation_graph  # noqa: E402
    from gold_network_graph import (  # noqa: E402
        build_gold_network_graph,
        build_gold_pearson_graph,
        build_train_positive_pearson_graph,
    )
    from grn_model import (  # noqa: E402
        GeneExpressionTransformer,
        ScGPTExprConcatEncoder,
        ScGPTExprTripleBranchEncoder,
        ScGPTGeneExpressionEncoder,
    )
    from model import GraphTransformer  # noqa: E402
    from utils import LinkPredictor  # noqa: E402

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    exp_path = data_dir / "BL--ExpressionData.csv"
    if not exp_path.exists():
        raise SystemExit(f"Missing {exp_path}")

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
            expr_cpu, top_k=args.top_k, min_abs_corr=getattr(args, "corr_min_abs", None)
        )
    elif args.prior_graph == "gold":
        g = build_gold_network_graph(Path(args.gold_network_file.strip()), gene_names)
    elif args.prior_graph == "gold_pearson":
        g = build_gold_pearson_graph(Path(args.gold_network_file.strip()), gene_names, expr_cpu)
    elif args.prior_graph == "train_pos_pearson":
        g = build_train_positive_pearson_graph(
            train_ei.detach().cpu(), train_y.detach().cpu(), expr_cpu
        )
    else:
        raise SystemExit(f"Unknown prior_graph={args.prior_graph!r}")
    g = g.to(device)
    with torch.no_grad():
        pe = dgl.lap_pe(g, k=in_dim, padding=True)
    pe = pe.to(device=device, dtype=gene_expr.dtype)

    ckpt = torch.load(ckpt_path, map_location=device)
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
    predictor = LinkPredictor(
        int(meta.get("gt_hidden_dim", args.gt_hidden_dim)),
        mode=str(meta.get("link_decoder", "concat")),
        dropout=float(args.dropout),
        bilinear_dim=(int(meta["link_bilinear_dim"]) if meta.get("link_bilinear_dim") else None),
    ).to(device)
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
        gt_variant=str(meta.get("gt_variant", "legacy")),
        dropout=float(args.dropout),
    ).to(device)
    sd_seq = ckpt["seq_model"]
    if expr_enc_saved == "scgpt":
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

    seq_model.eval()
    gt_model.eval()
    predictor.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Test metrics (same as end of run_training) ---
    te_logits = logits_all_edges(
        seq_model, gt_model, predictor, gene_expr, g, pe, test_ei, args.edge_batch_size,
        triple_fuse=triple_fuse_eval,
    )
    test_metrics = compute_test_metrics(
        test_y.cpu().numpy(),
        te_logits.cpu().numpy(),
        prob_threshold=args.test_prob_threshold,
    )
    np.savez(
        out_dir / "test_predictions.npz",
        logits=te_logits.cpu().numpy(),
        labels=test_y.cpu().numpy(),
    )

    idx = np.array(gene_names)

    def _export_split(name: str, ei: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> None:
        ei_np = ei.cpu().numpy()
        y_np = y.cpu().numpy()
        log_np = logits.cpu().numpy()
        sig_np = torch.sigmoid(logits).cpu().numpy()
        # columns: tf_idx, tgt_idx, tf_gene, target_gene, label, logit, prob
        import csv

        p = out_dir / f"application_edges_{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["TF_index", "Target_index", "TF", "Target", "Label", "logit", "prob"]
            )
            for k in range(ei_np.shape[1]):
                ti, tj = int(ei_np[0, k]), int(ei_np[1, k])
                w.writerow(
                    [
                        ti,
                        tj,
                        idx[ti],
                        idx[tj],
                        int(y_np[k]),
                        float(log_np[k]),
                        float(sig_np[k]),
                    ]
                )

    for part in [s.strip().lower() for s in cli.export_splits.split(",") if s.strip()]:
        if part == "test":
            _export_split("test", test_ei, test_y, te_logits)
        elif part == "val":
            vl = logits_all_edges(
                seq_model, gt_model, predictor, gene_expr, g, pe, val_ei, args.edge_batch_size,
                triple_fuse=triple_fuse_eval,
            )
            _export_split("val", val_ei, val_y, vl)
        elif part == "train":
            tr = logits_all_edges(
                seq_model, gt_model, predictor, gene_expr, g, pe, train_ei, args.edge_batch_size,
                triple_fuse=triple_fuse_eval,
            )
            _export_split("train", train_ei, train_y, tr)
        else:
            raise SystemExit(f"Unknown split in --export_splits: {part!r}")

    payload = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "test_metrics": test_metrics,
        "config_source": str(res_path) if res_path.is_file() else "checkpoint_meta",
    }
    with open(out_dir / "inference_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("-" * 72)
    print(f"Checkpoint: {ckpt_path}")
    print(
        f"Test AUROC={test_metrics['test_auroc']:.4f}  AUPRC={test_metrics['test_auprc']:.4f}  "
        f"(threshold={test_metrics['test_prob_threshold']})"
    )
    print(f"Wrote {out_dir / 'inference_metrics.json'}")
    print(f"Wrote {out_dir / 'test_predictions.npz'}")
    print(f"Wrote {out_dir / 'application_edges_*.csv'} (splits={cli.export_splits})")


if __name__ == "__main__":
    main()
