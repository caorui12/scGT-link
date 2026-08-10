#!/usr/bin/env python3
"""
Ablate Graph Transformer depth (gt_num_layers) on the locked 8-dataset set.

scGT-Link v2 defaults: expr_encoder=scgpt, prior_graph=train_pos_pearson,
gt_variant=directed. Records test metrics + training / per-epoch time.

Default layers: 0, 2, 4, 6 (baseline), 8. Epochs: 50.

Example::

  python scripts/run_ablation_gt_layers_8ds.py --dry_run
  python scripts/run_ablation_gt_layers_8ds.py --skip_existing --continue_on_error
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _REPO_ROOT / "train_grn_graph.py"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "out_ablation_gt_layers_50ep"
_DEFAULT_BENCH = Path(
    "/Users/drruicao/Desktop/deeplearning/benchmark/Dataset/Benchmark Dataset"
)
_DEFAULT_LAYERS: tuple[int, ...] = (0, 2, 4, 6, 8)

_STORE_TRUE_KEYS = frozenset(
    {"no_cuda", "no_edge_weight_attn", "no_edge_dir_attn", "no_lappe", "no_structural_pe"}
)

DATASET_SLUGS: tuple[str, ...] = (
    "STRING__hESC__500",
    "STRING__mESC__500",
    "Specific__hESC__500",
    "Specific__mESC__500",
    "Non_Specific__hESC__500",
    "Non_Specific__mESC__500",
    "Lofgof__mESC__500",
    "Lofgof__mESC__1000",
)

_BASE_CFG: dict[str, Any] = {
    "expr_encoder": "scgpt",
    "prior_graph": "train_pos_pearson",
    "gt_variant": "directed",
}


def _import_discover():
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from benchmark_train_grn_suite import discover_training_tasks  # noqa: WPS433

    return discover_training_tasks


def _arm_name(n_layers: int) -> str:
    return f"L{n_layers}"


def _build_cmd(
    data_dir: Path,
    split_dir: Path,
    out_dir: Path,
    *,
    gt_num_layers: int,
    epochs: int,
    seed: int,
    no_cuda: bool,
) -> list[str]:
    cfg = {**_BASE_CFG, "gt_num_layers": int(gt_num_layers)}
    if no_cuda:
        cfg["no_cuda"] = True
    cmd: list[str] = [
        sys.executable,
        str(_TRAIN),
        "--data_dir",
        str(data_dir),
        "--split_dir",
        str(split_dir),
        "--output_dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    ]
    for k, v in sorted(cfg.items()):
        if v is None or v == "":
            continue
        if k in _STORE_TRUE_KEYS:
            if v is True:
                cmd.append(f"--{k}")
            continue
        cmd.extend([f"--{k}", str(v)])
    return cmd


def _read_results(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _mean_epoch_sec(res: dict[str, Any] | None) -> float | None:
    if not res:
        return None
    hist = res.get("history") or []
    xs = [
        float(h["epoch_sec"])
        for h in hist
        if isinstance(h, dict) and h.get("epoch_sec") is not None
    ]
    return statistics.mean(xs) if xs else None


def _row_from_result(
    slug: str,
    n_layers: int,
    out_dir: Path,
    res: dict[str, Any] | None,
    *,
    status: str,
    exit_code: int,
    skipped: bool = False,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "gt_num_layers": n_layers,
        "arm": _arm_name(n_layers),
        "status": status,
        "exit_code": exit_code,
        "skipped": skipped,
        "test_auroc": None if res is None else res.get("test_auroc"),
        "test_auprc": None if res is None else res.get("test_auprc"),
        "best_val_auroc": None if res is None else res.get("best_val_auroc"),
        "best_val_auprc": None if res is None else res.get("best_val_auprc"),
        "training_time_sec": None if res is None else res.get("training_time_sec"),
        "mean_epoch_sec": _mean_epoch_sec(res),
        "run_dir": str(out_dir),
    }


def _write_summary(output_root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    summary = {"meta": meta, "datasets": list(DATASET_SLUGS), "runs": rows}
    out_json = output_root / "ablation_gt_layers_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "slug",
        "gt_num_layers",
        "arm",
        "status",
        "exit_code",
        "test_auroc",
        "test_auprc",
        "best_val_auroc",
        "best_val_auprc",
        "training_time_sec",
        "mean_epoch_sec",
        "run_dir",
    ]
    csv_path = output_root / "ablation_gt_layers_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate by layer (ok runs only).
    by_L: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_L.setdefault(int(r["gt_num_layers"]), []).append(r)

    agg_rows: list[dict[str, Any]] = []
    for L in sorted(by_L):
        xs = by_L[L]
        aurocs = [float(x["test_auroc"]) for x in xs if x.get("test_auroc") is not None]
        auprcs = [float(x["test_auprc"]) for x in xs if x.get("test_auprc") is not None]
        times = [float(x["training_time_sec"]) for x in xs if x.get("training_time_sec") is not None]
        eps = [float(x["mean_epoch_sec"]) for x in xs if x.get("mean_epoch_sec") is not None]
        agg_rows.append(
            {
                "gt_num_layers": L,
                "n": len(xs),
                "mean_test_auroc": statistics.mean(aurocs) if aurocs else None,
                "mean_test_auprc": statistics.mean(auprcs) if auprcs else None,
                "mean_training_time_sec": statistics.mean(times) if times else None,
                "sum_training_time_sec": sum(times) if times else None,
                "mean_epoch_sec": statistics.mean(eps) if eps else None,
            }
        )
    agg_path = output_root / "ablation_gt_layers_by_depth.csv"
    if agg_rows:
        with open(agg_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)

    # Δ vs L6 baseline when present.
    base_by_slug = {
        r["slug"]: r
        for r in rows
        if r.get("status") == "ok" and int(r["gt_num_layers"]) == 6
    }
    delta_rows: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "ok" or int(r["gt_num_layers"]) == 6:
            continue
        b = base_by_slug.get(r["slug"])
        if not b or b.get("test_auprc") is None or r.get("test_auprc") is None:
            continue
        delta_rows.append(
            {
                "slug": r["slug"],
                "gt_num_layers": r["gt_num_layers"],
                "test_auprc": r["test_auprc"],
                "delta_auprc_vs_L6": float(r["test_auprc"]) - float(b["test_auprc"]),
                "test_auroc": r["test_auroc"],
                "delta_auroc_vs_L6": (
                    None
                    if b.get("test_auroc") is None or r.get("test_auroc") is None
                    else float(r["test_auroc"]) - float(b["test_auroc"])
                ),
                "mean_epoch_sec": r.get("mean_epoch_sec"),
                "epoch_time_ratio_vs_L6": (
                    None
                    if not r.get("mean_epoch_sec") or not b.get("mean_epoch_sec")
                    else float(r["mean_epoch_sec"]) / float(b["mean_epoch_sec"])
                ),
                "training_time_sec": r.get("training_time_sec"),
            }
        )
    delta_path = output_root / "ablation_gt_layers_delta_vs_L6.csv"
    if delta_rows:
        with open(delta_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            w.writeheader()
            w.writerows(delta_rows)

    print(f"Wrote {out_json}")
    print(f"Wrote {csv_path}")
    if agg_rows:
        print(f"Wrote {agg_path}")
    if delta_rows:
        print(f"Wrote {delta_path}")


def parse_args() -> argparse.Namespace:
    env_root = os.environ.get("GRAPH_TRANSFORMER_BENCHMARK_ROOT", "").strip()
    p = argparse.ArgumentParser(description="8-dataset GT depth ablation (v2 defaults).")
    p.add_argument(
        "--benchmark_root",
        type=str,
        default=env_root
        or (
            "/home/rui.cao/deeplearning/benchmark/Dataset/Benchmark Dataset"
            if Path("/home/rui.cao/deeplearning/benchmark/Dataset/Benchmark Dataset").is_dir()
            else str(_DEFAULT_BENCH)
        ),
    )
    p.add_argument("--output_root", type=str, default=str(_DEFAULT_OUTPUT_ROOT))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_LAYERS),
        help="Comma-separated gt_num_layers values (default: 0,2,4,6,8).",
    )
    p.add_argument("--slug", action="append", default=[], metavar="SLUG")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layers = tuple(int(x.strip()) for x in args.layers.split(",") if x.strip() != "")
    if not layers:
        raise SystemExit("Empty --layers")

    bench_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    discover = _import_discover()
    tasks = {t.slug: t for t in discover(bench_root)}
    missing = [s for s in DATASET_SLUGS if s not in tasks]
    if missing:
        raise SystemExit(f"Missing slugs under {bench_root}: {missing}")

    want_slugs = list(args.slug) if args.slug else list(DATASET_SLUGS)
    for s in want_slugs:
        if s not in DATASET_SLUGS:
            raise SystemExit(f"Slug {s!r} not in locked 8-dataset list")

    jobs = [(slug, L) for slug in want_slugs for L in layers]
    print(f"Benchmark root: {bench_root}")
    print(f"Output root:    {output_root}")
    print(f"Epochs:         {args.epochs}  seed={args.seed}")
    print(f"Layers:         {list(layers)}  (v2: scgpt + train_pos_pearson + directed)")
    print(f"Jobs:           {len(jobs)}")
    print("-" * 72)

    rows: list[dict[str, Any]] = []
    failures = 0

    for i, (slug, L) in enumerate(jobs, start=1):
        task = tasks[slug]
        arm = _arm_name(L)
        out_dir = output_root / slug / arm
        res_path = out_dir / "results.json"
        print(f"[{i}/{len(jobs)}] {slug} / {arm} (gt_num_layers={L})")

        if args.skip_existing and res_path.is_file():
            res = _read_results(res_path)
            print(f"  skip existing {res_path}")
            rows.append(_row_from_result(slug, L, out_dir, res, status="ok", exit_code=0, skipped=True))
            continue

        emb = task.data_dir / "scgpt_gene_emb.pt"
        if not emb.is_file():
            failures += 1
            print(f"  ERROR: missing {emb}")
            rows.append(_row_from_result(slug, L, out_dir, None, status="missing_scgpt", exit_code=1))
            if not args.continue_on_error:
                break
            continue

        cmd = _build_cmd(
            task.data_dir,
            task.split_dir,
            out_dir,
            gt_num_layers=L,
            epochs=args.epochs,
            seed=args.seed,
            no_cuda=args.no_cuda,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ablation_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        print(f"  {' '.join(cmd)}")

        if args.dry_run:
            rows.append(_row_from_result(slug, L, out_dir, None, status="dry_run", exit_code=0))
            continue

        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        code = int(proc.returncode)
        if code != 0:
            failures += 1
            print(f"  FAILED exit_code={code}")
            rows.append(_row_from_result(slug, L, out_dir, None, status="failed", exit_code=code))
            if not args.continue_on_error:
                break
            continue

        res = _read_results(res_path)
        rows.append(_row_from_result(slug, L, out_dir, res, status="ok", exit_code=0))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(bench_root),
        "output_root": str(output_root),
        "epochs": args.epochs,
        "seed": args.seed,
        "layers": list(layers),
        "base_cfg": _BASE_CFG,
        "dry_run": bool(args.dry_run),
        "failures": failures,
        "n_jobs": len(jobs),
        "n_recorded": len(rows),
    }
    _write_summary(output_root, rows, meta)
    print("-" * 72)
    print(f"Done. failures={failures} recorded={len(rows)}")
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
