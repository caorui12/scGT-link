#!/usr/bin/env python3
"""
PCC |r|, sklearn mutual information, and GRNBoost2 (arboreto) on **44 GENELink-style benchmark leaves**.

Uses **Train_validation_test** CSV edge splits (same supervised evaluation as GENELink / CNNC).

Run from repo root::

  python benchmarks/statistical_baselines/run_benchmark_baselines_suite.py --continue_on_error

Defaults::

  * ``--benchmark_root``: ``<Graph_Transformer>/../benchmark/Dataset/Benchmark Dataset``
  * ``--output_root``: ``benchmarks/statistical_baselines/outputs``

Environment::

  GRAPH_TRANSFORMER_BENCHMARK_ROOT
  GRAPH_TRANSFORMER_BASELINE_OUTPUT_ROOT

Requires ``arboreto`` for GRNBoost2 (``pip install -r benchmarks/statistical_baselines/requirements-benchmark.txt``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.statistical_baselines.baseline_methods import evaluate_task_methods

_DEFAULT_BENCHMARK_ROOT = _REPO_ROOT.parent / "benchmark" / "Dataset" / "Benchmark Dataset"
_DEFAULT_OUTPUT_ROOT = _BASELINE_DIR / "outputs"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class BenchmarkTask:
    data_dir: Path
    split_dir: Path
    net_folder: str
    species: str
    n_genes: int
    slug: str


def _slug(net_folder: str, species: str, n_genes: int) -> str:
    nf = net_folder.replace(" Dataset", "").replace(" ", "_").replace("-", "_")
    return f"{nf}__{species}__{n_genes}"


def discover_training_tasks(benchmark_root: Path) -> list[BenchmarkTask]:
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {benchmark_root}")
    tasks: list[BenchmarkTask] = []
    for label_p in sorted(benchmark_root.rglob("Label.csv")):
        data_dir = label_p.parent
        split_dir = data_dir / "Train_validation_test"
        if not (split_dir / "Train_set.csv").is_file():
            continue
        if not (data_dir / "BL--ExpressionData.csv").is_file():
            continue
        if not (data_dir / "TF.csv").is_file() or not (data_dir / "Target.csv").is_file():
            continue
        try:
            rel = data_dir.relative_to(benchmark_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        net_folder, species, tfs_folder = parts[0], parts[1], parts[2]
        if not tfs_folder.startswith("TFs+"):
            continue
        try:
            n_genes = int(tfs_folder.split("+", 1)[1])
        except (IndexError, ValueError):
            continue
        slug = _slug(net_folder, species, n_genes)
        tasks.append(
            BenchmarkTask(
                data_dir=data_dir.resolve(),
                split_dir=split_dir.resolve(),
                net_folder=net_folder,
                species=species,
                n_genes=n_genes,
                slug=slug,
            )
        )
    return tasks


def parse_args() -> argparse.Namespace:
    env_root = os.environ.get("GRAPH_TRANSFORMER_BENCHMARK_ROOT", "").strip()
    env_out = os.environ.get("GRAPH_TRANSFORMER_BASELINE_OUTPUT_ROOT", "").strip()
    p = argparse.ArgumentParser(description="PCC / MI / GRNBoost2 benchmark suite (44 leaves).")
    p.add_argument(
        "--benchmark_root",
        type=str,
        default=env_root or str(_DEFAULT_BENCHMARK_ROOT),
        help=f"Benchmark Dataset root. Default: {_DEFAULT_BENCHMARK_ROOT}",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=env_out or str(_DEFAULT_OUTPUT_ROOT),
        help=f"Per-task JSON outputs. Default: {_DEFAULT_OUTPUT_ROOT}",
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
    p.add_argument("--filter_slug_regex", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    methods_set = frozenset(args.methods)

    tasks = discover_training_tasks(benchmark_root)
    if not tasks:
        raise SystemExit(f"No tasks under {benchmark_root}")

    if args.filter_slug_regex.strip():
        rx = re.compile(args.filter_slug_regex)
        tasks = [t for t in tasks if rx.search(t.slug)]
        if not tasks:
            raise SystemExit("No tasks after --filter_slug_regex.")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    rows: list[dict[str, Any]] = []

    print(f"Benchmark root: {benchmark_root}", flush=True)
    print(f"Output root:    {output_root}", flush=True)
    print(f"Tasks:          {len(tasks)}", flush=True)
    print(f"Methods:        {sorted(methods_set)}", flush=True)
    print(f"Seed:           {args.seed}", flush=True)
    print(f"Standardize:    {not args.no_standardize}", flush=True)
    print("-" * 72, flush=True)

    for i, task in enumerate(tasks):
        slug_dir = output_root / task.slug
        out_json = slug_dir / "results.json"
        print(f"[{i + 1}/{len(tasks)}] {task.slug}", flush=True)
        if args.dry_run:
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

            flat = {"slug": task.slug, "run_dir": str(slug_dir)}
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
        "suite": "statistical_baselines_pcc_mi_grnboost2",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(benchmark_root),
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
