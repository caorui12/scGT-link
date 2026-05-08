#!/usr/bin/env python3
"""
PCC, mutual information, and GRNBoost2 on the four **Graph_Transformer** ``data/`` registry splits
(same layout as ``train_grn_graph.py --train_all``): ``hESC_500``, ``hESC_1000``, ``mESC_500``, ``mESC_1000``.

Uses ``evaluate_task_methods`` from ``baseline_methods.py`` (identical to
``run_benchmark_baselines_suite.py`` for a subset of tasks).

Run **from the Graph_Transformer repo root**::

  python benchmarks/statistical_baselines/run_gt_data_four_baselines.py

Defaults::

  * ``--gt_data_root``: ``<repo>/data``
  * ``--output_root``: ``benchmarks/statistical_baselines/outputs_gt_data_four``

Environment::

  GT_BASELINE_DATA_ROOT   (overrides ``--gt_data_root``)
  GT_BASELINE_OUTPUT_ROOT (overrides ``--output_root``)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BASELINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BASELINE_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.statistical_baselines.baseline_methods import evaluate_task_methods
from benchmarks.statistical_baselines.run_benchmark_baselines_suite import BenchmarkTask

_DEFAULT_GT_DATA = _REPO_ROOT / "data"
_DEFAULT_OUTPUT_ROOT = _BASELINE_DIR / "outputs_gt_data_four"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_gt_data_tasks(gt_data: Path) -> list[BenchmarkTask]:
    specs: list[tuple[str, Path, str, int]] = [
        ("hESC_500", Path("hESC") / "TFs+500", "hESC_500", 500),
        ("hESC_1000", Path("hESC") / "TFs+1000", "hESC_1000", 1000),
        ("mESC_500", Path("mESC") / "TFs+500", "mESC_500", 500),
        ("mESC_1000", Path("mESC") / "TFs+1000", "mESC_1000", 1000),
    ]
    out: list[BenchmarkTask] = []
    for slug, expr_sub, split_leaf, n_genes in specs:
        data_dir = (gt_data / expr_sub).resolve()
        split_dir = (gt_data / "Train_validation_test" / split_leaf).resolve()
        if not (data_dir / "BL--ExpressionData.csv").is_file():
            raise FileNotFoundError(f"Missing {data_dir / 'BL--ExpressionData.csv'}")
        if not (data_dir / "TF.csv").is_file() or not (data_dir / "Target.csv").is_file():
            raise FileNotFoundError(f"Missing TF.csv or Target.csv under {data_dir}")
        if not (split_dir / "Train_set.csv").is_file():
            raise FileNotFoundError(f"Missing splits under {split_dir}")
        species = slug.split("_")[0]
        out.append(
            BenchmarkTask(
                data_dir=data_dir,
                split_dir=split_dir,
                net_folder="Graph_Transformer_data",
                species=species,
                n_genes=n_genes,
                slug=slug,
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    env_gt = os.environ.get("GT_BASELINE_DATA_ROOT", "").strip()
    env_out = os.environ.get("GT_BASELINE_OUTPUT_ROOT", "").strip()
    p = argparse.ArgumentParser(
        description="PCC / MI / GRNBoost2 on Graph_Transformer data/ (four registry datasets).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gt_data_root",
        type=str,
        default=env_gt or str(_DEFAULT_GT_DATA),
        help=f"Graph_Transformer ``data`` folder. Default: {_DEFAULT_GT_DATA}",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=env_out or str(_DEFAULT_OUTPUT_ROOT),
        help=f"Per-slug JSON. Default: {_DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pcc", "mi", "grnboost2"],
        choices=("pcc", "mi", "grnboost2"),
        help="Which baselines to run per task.",
    )
    p.add_argument("--seed", type=int, default=_env_int("GRAPH_TRANSFORMER_BASELINE_SEED") or 42)
    p.add_argument(
        "--no_standardize",
        action="store_true",
        help="Skip z-scoring genes across cells (default: standardize before scores).",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt_data = Path(args.gt_data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    methods_set = frozenset(args.methods)

    try:
        tasks = resolve_gt_data_tasks(gt_data)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    rows: list[dict[str, Any]] = []

    print(f"gt_data_root:  {gt_data}", flush=True)
    print(f"output_root:   {output_root}", flush=True)
    print(f"Tasks:         {len(tasks)}", flush=True)
    print(f"Methods:       {sorted(methods_set)}", flush=True)
    print(f"Seed:          {args.seed}", flush=True)
    print(f"Standardize:   {not args.no_standardize}", flush=True)
    print("-" * 72, flush=True)

    for i, task in enumerate(tasks):
        slug_dir = output_root / task.slug
        out_json = slug_dir / "results.json"
        print(f"[{i + 1}/{len(tasks)}] {task.slug}", flush=True)
        if args.dry_run:
            print(f"  data_dir={task.data_dir}", flush=True)
            print(f"  split_dir={task.split_dir}", flush=True)
            print(f"  -> {slug_dir}", flush=True)
            rows.append({"slug": task.slug, "dry_run": True})
            continue
        slug_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = evaluate_task_methods(
                task.data_dir,
                task.split_dir,
                standardize=not args.no_standardize,
                seed=args.seed,
                methods=methods_set,
            )
            record = {
                "slug": task.slug,
                "data_dir": str(task.data_dir),
                "split_dir": str(task.split_dir),
                "methods": payload,
                "seed": args.seed,
                "standardize_expression": not args.no_standardize,
            }
            out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

            flat: dict[str, Any] = {"slug": task.slug, "run_dir": str(slug_dir)}
            msg_parts = []
            for m in sorted(payload.keys()):
                pm = payload[m]
                if "error" in pm:
                    msg_parts.append(f"{m}: ERROR {pm['error']}")
                    flat[f"{m}_test_auroc"] = None
                    flat[f"{m}_test_auprc"] = None
                else:
                    flat[f"{m}_test_auroc"] = pm.get("test_auroc")
                    flat[f"{m}_test_auprc"] = pm.get("test_auprc")
                    msg_parts.append(
                        f"{m}: test AUROC={pm.get('test_auroc'):.4f} AUPRC={pm.get('test_auprc'):.4f}"
                    )
            rows.append(flat)
            print("  " + " | ".join(msg_parts), flush=True)
        except Exception as e:
            failures += 1
            rows.append({"slug": task.slug, "error": str(e), "run_dir": str(slug_dir)})
            print(f"  !! {e}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise

    meta = {
        "suite": "statistical_baselines_gt_data_four",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gt_data_root": str(gt_data),
        "output_root": str(output_root),
        "methods": sorted(methods_set),
        "seed": args.seed,
        "standardize_expression": not args.no_standardize,
        "n_tasks": len(tasks),
        "dry_run": args.dry_run,
        "failures": failures,
    }
    if not args.dry_run:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path = output_root / "benchmark_summary.json"
        summary_path.write_text(json.dumps({"meta": meta, "runs": rows}, indent=2), encoding="utf-8")
        print("-" * 72, flush=True)
        print(f"Wrote {summary_path}", flush=True)

    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
