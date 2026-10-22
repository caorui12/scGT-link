#!/usr/bin/env python3
"""
scGT-Link v2 full-panel **candidate** link prediction: load ``best.pt`` + expression + prior graph,
then score many TF→target directed edges **not restricted to Train/Val/Test CSVs**.

Modes (``--candidate_mode``)::

    tf_cross_panel — For every gene index listed in ``TF.csv`` (column ``index``),
    pair with **all** panel genes except self-loops (and optional exclusions).

    prior_csv_rows — Directed edges **(Gene1 → Gene2)** read from ``BL--network.csv``
    (or ``--prior_csv``); resolves symbols like training. Optional: restrict Gene1 to TF.csv set.

Needs the same artefacts as ``train_grn_graph.py`` for ``gold_pearson`` /
``gold`` / ``correlation``. If checkpoint uses ``prior_graph=train_pos_pearson``, you must
pass ``--split_dir`` pointing at splits so the script can rebuild the prior graph.

Example (**hESC 500**, TF × full panel; writes one CSV + small JSON summary)::

  cd /path/to/Graph_Transformer
  python scripts/infer_grn_predict_network.py \\
    --checkpoint out_gt_data_four/hESC_500/best.pt \\
    --data_dir data/hESC/TFs+500 \\
    --candidate_mode tf_cross_panel \\
    --output_dir out_infer/hESC_500_predicted_network

Optional: ``--exclude_gold_rows`` removes directed rows present in BL--network before scoring
(tf_cross_panel mode). ``--top_k_per_tf K`` emits only Top-K edges per TF by probability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_grn_graph import (  # noqa: E402
    _remap_tune_default_paths,
    _uses_skip_pos_at_gt,
    _uses_triple_gt_fuse,
    apply_split_subdir,
    compute_structural_pe,
    load_expression_for_grn,
    logits_all_edges,
    parse_args,
    set_seed,
    set_structural_pe_backend,
)

from argparse import Namespace  # noqa: E402 — type hints


def _apply_config_dict(args: Namespace, cfg: dict) -> None:
    for k, v in cfg.items():
        setattr(args, k, v)


def _apply_meta_fallback(args: Namespace, meta: dict) -> None:
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


def _gene_symbol_index_map(gene_names: list[str]) -> dict[str, int]:
    """Case-sensitive first; uppercase alias for lookups."""
    m: dict[str, int] = {}
    for i, g in enumerate(gene_names):
        s = str(g).strip()
        m[s] = i
        m.setdefault(s.upper(), i)
    return m


def _resolve_gene_idx(gmap: dict[str, int], symbol: str) -> int | None:
    s = str(symbol).strip()
    if not s:
        return None
    if s in gmap:
        i = gmap[s]
        return int(i) if i is not None else None
    su = s.upper()
    if su in gmap:
        i = gmap[su]
        return int(i) if i is not None else None
    return None


def _directed_pairs_from_prior_csv(net_path: Path, gmap: dict[str, int]) -> list[tuple[int, int]]:
    df = pd.read_csv(net_path)
    if df.shape[1] < 2:
        raise SystemExit(f"{net_path}: need at least two columns (Gene1, Gene2 or TF/Target).")
    c0, c1 = df.columns[0], df.columns[1]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, row in df.iterrows():
        a, b = str(row[c0]).strip(), str(row[c1]).strip()
        ia = _resolve_gene_idx(gmap, a)
        ib = _resolve_gene_idx(gmap, b)
        if ia is None or ib is None or ia == ib:
            continue
        key = (ia, ib)
        if key not in seen:
            seen.add(key)
            out.append(key)
    if not out:
        raise SystemExit(f"No usable directed edges after mapping symbols in {net_path}.")
    return out


def _load_tf_indices(tf_csv: Path) -> np.ndarray:
    df = pd.read_csv(tf_csv, index_col=0)
    col = None
    for name in ("index", "Index", "idx"):
        if name in df.columns:
            col = name
            break
    if col is None:
        raise SystemExit(f"{tf_csv}: need an 'index' column with integer gene indices (TF.csv).")
    idx = df[col].astype(np.int64).to_numpy()
    idx = idx[(idx >= 0)]
    idx = np.unique(idx)
    if idx.size == 0:
        raise SystemExit(f"{tf_csv}: no TF indices.")
    return idx.astype(np.int64)


def build_edge_index_tf_cross_panel(
    n_genes: int,
    tf_indices: np.ndarray,
    *,
    exclude_pairs: set[tuple[int, int]] | None,
) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    for ti in tf_indices.astype(np.int64):
        ti_i = int(ti)
        for tj in range(n_genes):
            if tj == ti_i:
                continue
            pair = (ti_i, tj)
            if exclude_pairs and pair in exclude_pairs:
                continue
            rows.append(pair)
    if not rows:
        raise SystemExit("Empty candidate edge set after filters.")
    arr = np.asarray(rows, dtype=np.int64).T
    return torch.from_numpy(arr)


def pairs_to_edge_index(pairs: list[tuple[int, int]]) -> torch.Tensor:
    arr = np.asarray(pairs, dtype=np.int64).T
    return torch.from_numpy(arr)


def _apply_top_k_per_tf(
    ei: torch.Tensor, probs: np.ndarray, k: int, gene_names: list[str]
) -> tuple[torch.Tensor, np.ndarray]:
    """Keep at most k outgoing edges per source TF ordered by descending prob."""
    ei_np = ei.cpu().numpy()
    buckets: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for idx in range(ei_np.shape[1]):
        tfi = int(ei_np[0, idx])
        tgt = int(ei_np[1, idx])
        p = float(probs[idx])
        buckets[tfi].append((p, tfi, tgt))
    sel_cols: list[tuple[int, int]] = []
    sel_prob: list[float] = []
    for tfi in sorted(buckets.keys()):
        arr = sorted(buckets[tfi], key=lambda t: (-t[0], t[2]))
        for p, ti, tg in arr[:k]:
            sel_cols.append((ti, tg))
            sel_prob.append(p)
    if not sel_cols:
        raise SystemExit("top_k filtering removed all edges.")
    a = np.asarray(sel_cols, dtype=np.int64).T
    new_ei = torch.from_numpy(a)
    new_probs = np.asarray(sel_prob, dtype=np.float64)
    return new_ei, new_probs


def _resolve_prior_csv_path(raw: str, data_dir: Path) -> Path:
    if not raw.strip():
        raise SystemExit("prior_csv_rows needs a network CSV path (--prior_csv or checkpoint gold_network_file).")
    path = Path(raw.strip()).expanduser()
    if path.is_file():
        return path.resolve()
    r = _ROOT / raw.strip()
    if r.is_file():
        return r.resolve()
    d = data_dir / Path(raw.strip()).name
    if d.is_file():
        return d.resolve()
    raise SystemExit(f"Could not find prior network CSV: {raw!r}")


def write_prediction_csv(path: Path, ei: torch.Tensor, logits: torch.Tensor, gene_names: list[str]) -> None:
    ei_np = ei.cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    probs = torch.sigmoid(torch.as_tensor(logits_np, dtype=torch.float32)).numpy()
    gn = np.array(gene_names, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TF_index", "Target_index", "TF", "Target", "logit", "prob"])
        for i in range(ei_np.shape[1]):
            ti, tg = int(ei_np[0, i]), int(ei_np[1, i])
            w.writerow(
                [
                    ti,
                    tg,
                    str(gn[ti]),
                    str(gn[tg]),
                    float(logits_np[i]),
                    float(probs[i]),
                ]
            )


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="scGT-Link v2: score full TF→gene candidate panels from best.pt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=str(_ROOT / "out_gt_data_four/hESC_500/best.pt"))
    p.add_argument("--data_dir", type=str, default=str(_ROOT / "data/hESC/TFs+500"))
    p.add_argument(
        "--split_dir",
        type=str,
        default="",
        help="Train_validation_test/... Required if prior_graph=train_pos_pearson.",
    )
    p.add_argument("--split_subdir", type=str, default="Train_validation_test")
    p.add_argument("--output_dir", type=str, default=str(_ROOT / "out_infer/hESC_500_predicted_network"))
    p.add_argument(
        "--output_csv",
        type=str,
        default="predicted_tf_target_edges.csv",
        help="CSV filename relative to output_dir unless absolute.",
    )
    p.add_argument(
        "--candidate_mode",
        type=str,
        choices=("tf_cross_panel", "prior_csv_rows"),
        default="tf_cross_panel",
    )
    p.add_argument("--tf_csv", type=str, default="", help="Default: <data_dir>/TF.csv.")
    p.add_argument(
        "--prior_csv",
        type=str,
        default="",
        help="prior_csv_rows mode only; defaults to checkpoint gold_network_file / args.",
    )
    p.add_argument(
        "--prior_only_gene1_in_tf_csv",
        action="store_true",
        help="prior_csv_rows: keep edge only when Gene1 is in TF.csv index column.",
    )
    p.add_argument(
        "--exclude_gold_rows",
        action="store_true",
        help="tf_cross_panel: exclude directed Gene1→Gene2 pairs listed in BL--network (same file as prior).",
    )
    p.add_argument("--exclude_self_loops", action="store_true", default=True)
    p.add_argument("--no_exclude_self_loops", action="store_false", dest="exclude_self_loops")
    p.add_argument("--top_k_per_tf", type=int, default=0, help="If >0, keep only Top-K probs per TF (after logits).")
    p.add_argument("--edge_batch_size", type=int, default=None)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


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
        ck0 = torch.load(ckpt_path, map_location="cpu")
        _apply_meta_fallback(args, ck0.get("meta", {}))

    args.data_dir = str(Path(cli.data_dir).expanduser().resolve())
    sd = cli.split_dir.strip()
    args.split_dir = str(Path(sd).expanduser().resolve()) if sd else ""
    args.split_subdir = cli.split_subdir
    args.no_cuda = cli.no_cuda
    if cli.edge_batch_size is not None:
        args.edge_batch_size = cli.edge_batch_size

    args.split_dir = (
        apply_split_subdir(args.split_dir, getattr(args, "split_subdir", "Train_validation_test"))
        if args.split_dir
        else ""
    )
    _remap_tune_default_paths(args)

    if args.prior_graph == "train_pos_pearson" and not (args.split_dir and Path(args.split_dir).is_dir()):
        raise SystemExit("prior_graph=train_pos_pearson requires existing --split_dir with Train_set.csv.")

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
        ScGPTGeneExpressionEncoder,
    )
    from grn_data import load_edge_split  # noqa: E402
    from model import GraphTransformer  # noqa: E402
    from utils import LinkPredictor  # noqa: E402

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    data_dir = Path(args.data_dir)
    exp_path = data_dir / "BL--ExpressionData.csv"
    if not exp_path.is_file():
        raise SystemExit(f"Missing {exp_path}")

    tf_csv = Path(cli.tf_csv).expanduser().resolve() if cli.tf_csv.strip() else (data_dir / "TF.csv")

    gene_expr, gene_names, n_cells = load_expression_for_grn(exp_path)
    n_genes = len(gene_names)
    gmap = _gene_symbol_index_map(gene_names)

    ck0 = torch.load(ckpt_path, map_location="cpu")
    meta_ck = ck0.get("meta", {})
    if int(meta_ck.get("n_genes", n_genes)) != n_genes:
        print(
            f"WARN: ckpt meta n_genes={meta_ck.get('n_genes')} vs current expression n_genes={n_genes} — risky.",
            flush=True,
        )

    gene_expr = gene_expr.to(device)
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
        train_ei, train_y = load_edge_split(Path(args.split_dir), "Train_set")
        g = build_train_positive_pearson_graph(
            train_ei.detach().cpu(), train_y.detach().cpu(), expr_cpu
        )
    else:
        raise SystemExit(f"Unsupported prior_graph={args.prior_graph!r} for this script.")

    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    set_structural_pe_backend(str(meta.get("struct_pe", "lap_pe")))

    g = g.to(device)
    with torch.no_grad():
        pe = compute_structural_pe(g, in_dim)
    pe = pe.to(device=device, dtype=gene_expr.dtype)
    use_dir_eval = meta.get("use_directed_edge_attn", False)
    use_ew_eval = meta.get("use_edge_weight_attn", True)
    expr_enc_saved = meta.get("expr_encoder", "transformer")
    triple_fuse_eval = bool(meta.get("triple_gt_fuse", _uses_triple_gt_fuse(expr_enc_saved)))
    skip_pos_at_gt_eval = bool(meta.get("skip_pos_at_gt", _uses_skip_pos_at_gt(expr_enc_saved)))
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
        expr_in_dim=(
            int(meta.get("n_cells", n_cells))
            if triple_fuse_eval
            else None
        ),
        scgpt_in_dim=int(meta.get("d_model", in_dim)) if triple_fuse_eval else None,
        skip_pos_at_gt=skip_pos_at_gt_eval,
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

    cand_raw_prior = cli.prior_csv.strip() if cli.prior_csv.strip() else args.gold_network_file.strip()

    exclude: set[tuple[int, int]] | None = None
    gold_pairs_directed: list[tuple[int, int]] = []
    if cli.candidate_mode == "tf_cross_panel":
        if cli.exclude_gold_rows:
            gps = _resolve_prior_csv_path(args.gold_network_file.strip(), data_dir)
            gold_pairs_directed = _directed_pairs_from_prior_csv(gps, gmap)
            exclude = set(gold_pairs_directed)
            print(f"Excluding |gold directed rows|={len(exclude)} from tf_cross_panel.", flush=True)
        tf_idx = _load_tf_indices(tf_csv)
        ei = build_edge_index_tf_cross_panel(
            n_genes, tf_idx, exclude_pairs=exclude
        )
    elif cli.candidate_mode == "prior_csv_rows":
        net_path = _resolve_prior_csv_path(cand_raw_prior, data_dir)
        pairs = _directed_pairs_from_prior_csv(net_path, gmap)
        if cli.prior_only_gene1_in_tf_csv:
            tf_set = set(_load_tf_indices(tf_csv).tolist())
            pairs = [(a, b) for (a, b) in pairs if a in tf_set]
            if not pairs:
                raise SystemExit("prior_csv_rows + TF filter removed all pairs.")
        ei = pairs_to_edge_index(pairs)
    else:
        raise SystemExit("internal: unknown mode")

    print(
        f"Scoring edges: {ei.shape[1]}  ({cli.candidate_mode})  genes={n_genes} cells={n_cells}",
        flush=True,
    )

    logits = logits_all_edges(
        seq_model,
        gt_model,
        predictor,
        gene_expr,
        g,
        pe,
        ei,
        args.edge_batch_size,
        triple_fuse=triple_fuse_eval,
    )
    if cli.top_k_per_tf > 0:
        probs_all = torch.sigmoid(logits.detach().cpu()).numpy()
        ei, _ = _apply_top_k_per_tf(ei, probs_all, int(cli.top_k_per_tf), gene_names)
        logits = logits_all_edges(
            seq_model,
            gt_model,
            predictor,
            gene_expr,
            g,
            pe,
            ei,
            args.edge_batch_size,
            triple_fuse=triple_fuse_eval,
        )

    out_dir = Path(cli.output_dir).expanduser().resolve()
    # Do not `.resolve()` a bare filename first: `Path("x.csv").resolve()` becomes cwd-absolute and
    # would bypass `--output_dir`. Only honor true user absolute paths under `csv_raw`.
    csv_raw = cli.output_csv
    p_csv = Path(csv_raw).expanduser()
    out_csv = p_csv.resolve() if p_csv.is_absolute() else (out_dir / p_csv).resolve()

    write_prediction_csv(out_csv, ei, logits, gene_names)

    summary: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "prior_graph": args.prior_graph,
        "candidate_mode": cli.candidate_mode,
        "n_candidates": int(ei.shape[1]),
        "n_genes": n_genes,
        "output_csv": str(out_csv),
        "top_k_per_tf": int(cli.top_k_per_tf),
    }
    if args.split_dir:
        summary["split_dir"] = str(args.split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predict_network_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_dir / 'predict_network_summary.json'}")


if __name__ == "__main__":
    main()
