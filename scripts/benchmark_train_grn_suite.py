#!/usr/bin/env python3
"""
**Graph Transformer GRN** — batch driver for ``train_grn_graph.py`` only.

Scans a GENELink-style **Benchmark Dataset** tree, runs one training job per leaf
(``.../<Net>/<species>/TFs+{500|1000}/``), and writes an aggregate summary next to per-task
``results.json`` (each task’s ``--output_dir`` is ``<output_root>/<slug>/``).

Hyperparameters: by default this script **only** passes ``--data_dir``, ``--split_dir``,
``--output_dir``. Everything else uses ``train_grn_graph.py`` argparse defaults (including
``expr_encoder=scgpt_concat``, ``prior_graph=gold_pearson``, architecture sizes, lr, etc.).
Use ``--epochs`` / ``--seed`` only when you want to override those defaults.

Each leaf must contain:

* ``BL--ExpressionData.csv``, ``BL--network.csv``
* ``<split_subdir>/{Train,Validation,Test}_set.csv`` (default: ``Train_validation_test``;
  use ``Train_validation_test_gnnlink_paper`` for paper_bbad414 splits from
  ``split_grn_train_val_test.py``).
* ``scgpt_gene_emb.pt`` (see ``scripts/precompute_scgpt_all_benchmark.py``)

Server-friendly env (optional)::

  export GRAPH_TRANSFORMER_BENCHMARK_ROOT="/path/to/Benchmark Dataset"
  export GRAPH_TRANSFORMER_BENCHMARK_OUTPUT_ROOT="/path/to/out_benchmark_gt"
  # Optional overrides forwarded to train_grn_graph when set:
  export GRAPH_TRANSFORMER_BENCHMARK_EPOCHS=80
  export GRAPH_TRANSFORMER_BENCHMARK_SEED=42

Minimal run (after setting env or passing paths)::

  python scripts/benchmark_train_grn_suite.py

CPU-only (forward ``--no_cuda`` to training)::

  python scripts/benchmark_train_grn_suite.py --no_cuda

Paper-style splits (``Train_validation_test_gnnlink_paper``)::

  python scripts/benchmark_train_grn_suite.py \\
    --benchmark_root "/path/to/Benchmark Dataset" \\
    --split_subdir Train_validation_test_gnnlink_paper \\
    --output_root out_benchmark_graph_transformer_gnnlink_paper
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _REPO_ROOT / "train_grn_graph.py"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "out_benchmark_graph_transformer"

_STORE_TRUE_KEYS = frozenset({"no_cuda", "no_edge_weight_attn", "no_edge_dir_attn"})


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


def discover_training_tasks(
    benchmark_root: Path, *, split_subdir: str = "Train_validation_test"
) -> list[BenchmarkTask]:
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {benchmark_root}")
    tasks: list[BenchmarkTask] = []
    for label_p in sorted(benchmark_root.rglob("Label.csv")):
        data_dir = label_p.parent
        split_dir = data_dir / split_subdir
        if not (split_dir / "Train_set.csv").is_file():
            continue
        if not (data_dir / "BL--ExpressionData.csv").is_file():
            continue
        if not (data_dir / "BL--network.csv").is_file():
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


def _parse_train_arg(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Expected KEY=VALUE, got {s!r}")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        raise ValueError(s)
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    try:
        if "." in v:
            return k, float(v)
        return k, int(v)
    except ValueError:
        return k, v


def _extra_cfg_from_train_args(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k, v = _parse_train_arg(item)
        out[k] = v
    return out


def _build_cmd(
    train_py: Path,
    data_dir: Path,
    split_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    epochs: int | None,
    seed: int | None,
) -> list[str]:
    cmd: list[str] = [sys.executable, str(train_py)]
    cmd.extend(["--data_dir", str(data_dir)])
    cmd.extend(["--split_dir", str(split_dir)])
    cmd.extend(["--output_dir", str(out_dir)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph Transformer GRN: batch train_grn_graph on all Benchmark Dataset leaves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    env_root = os.environ.get("GRAPH_TRANSFORMER_BENCHMARK_ROOT", "").strip()
    env_out = os.environ.get("GRAPH_TRANSFORMER_BENCHMARK_OUTPUT_ROOT", "").strip()
    p.add_argument(
        "--benchmark_root",
        type=str,
        default=env_root or str(_REPO_ROOT / "data/Dataset/Benchmark Dataset"),
        help="Benchmark Dataset root (or GRAPH_TRANSFORMER_BENCHMARK_ROOT).",
    )
    p.add_argument(
        "--split_subdir",
        type=str,
        default="Train_validation_test",
        help="Folder under each .../TFs+N/ holding Train_set.csv etc. "
        "(use Train_validation_test_gnnlink_paper for paper_bbad414 splits).",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=env_out or str(_DEFAULT_OUTPUT_ROOT),
        help=f"Per-task runs under <slug>/ (or GRAPH_TRANSFORMER_BENCHMARK_OUTPUT_ROOT). Default: {_DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=_env_int("GRAPH_TRANSFORMER_BENCHMARK_EPOCHS"),
        help="If set, passed to train_grn_graph; else train script default (50). Env: GRAPH_TRANSFORMER_BENCHMARK_EPOCHS.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=_env_int("GRAPH_TRANSFORMER_BENCHMARK_SEED"),
        help="If set, passed to train_grn_graph; else train script default (42). Env: GRAPH_TRANSFORMER_BENCHMARK_SEED.",
    )
    p.add_argument(
        "--train_arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra train_grn_graph kwargs (repeatable). Prefer --no_cuda over train_arg for CPU.",
    )
    p.add_argument(
        "--no_cuda",
        action="store_true",
        help="Forward --no_cuda to train_grn_graph (force CPU). Default: off (use GPU if available).",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep going after a failed subprocess; summary lists exit codes.",
    )
    p.add_argument(
        "--skip_missing_scgpt",
        action="store_true",
        help="Skip tasks where scgpt_gene_emb.pt is missing under data_dir.",
    )
    p.add_argument(
        "--filter_slug_regex",
        type=str,
        default="",
        help="Only tasks whose slug matches this regex (e.g. 'Specific__hESC').",
    )
    return p.parse_args()


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    args = parse_args()
    split_subdir = args.split_subdir.strip() or "Train_validation_test"
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    tasks = discover_training_tasks(benchmark_root, split_subdir=split_subdir)
    if not tasks:
        raise SystemExit(f"No runnable tasks under {benchmark_root} (need splits + BL-- files).")

    if args.filter_slug_regex.strip():
        rx = re.compile(args.filter_slug_regex)
        tasks = [t for t in tasks if rx.search(t.slug)]
        if not tasks:
            raise SystemExit("No tasks left after --filter_slug_regex.")

    extra_cfg = _extra_cfg_from_train_args(args.train_arg)
    if args.no_cuda:
        extra_cfg["no_cuda"] = True
    epochs: int | None = args.epochs
    seed: int | None = args.seed

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0

    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    _log(f"Split subdir:   {split_subdir}")
    _log(f"Output root:    {output_root}")
    _log(f"Train script:   {_TRAIN.name} (Graph Transformer GRN, default hyperparameters except paths)")
    _log(f"Forward epochs: {epochs if epochs is not None else '(train default)'}")
    _log(f"Forward seed:   {seed if seed is not None else '(train default)'}")
    _log(f"no_cuda:        {bool(args.no_cuda)}")
    _log(f"Tasks:          {len(tasks)}")
    _log("-" * 72)

    for i, task in enumerate(tasks):
        if args.skip_missing_scgpt:
            scgpt_pt = task.data_dir / "scgpt_gene_emb.pt"
            if not scgpt_pt.is_file():
                _log(f"[skip {i + 1}/{len(tasks)}] {task.slug} (missing {scgpt_pt})")
                rows.append(
                    {
                        "slug": task.slug,
                        "data_dir": str(task.data_dir),
                        "split_dir": str(task.split_dir),
                        "skipped": True,
                        "reason": "missing_scgpt_emb",
                    }
                )
                continue

        run_out = output_root / task.slug
        cmd = _build_cmd(_TRAIN, task.data_dir, task.split_dir, run_out, extra_cfg, epochs, seed)

        _log(f"[{i + 1}/{len(tasks)}] {task.slug}")
        if args.dry_run:
            _log("  " + " ".join(cmd))
            rows.append({"slug": task.slug, "dry_run_cmd": " ".join(cmd)})
            continue

        run_out.mkdir(parents=True, exist_ok=True)
        with open(run_out / "benchmark_cmd.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")

        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=child_env)
        res_path = run_out / "results.json"
        res = _read_results(res_path) if proc.returncode == 0 else None
        row: dict[str, Any] = {
            "slug": task.slug,
            "data_dir": str(task.data_dir),
            "split_dir": str(task.split_dir),
            "run_dir": str(run_out),
            "exit_code": proc.returncode,
            "test_auroc": res.get("test_auroc") if res else None,
            "test_auprc": res.get("test_auprc") if res else None,
            "best_val_auroc": res.get("best_val_auroc") if res else None,
        }
        rows.append(row)
        if proc.returncode != 0:
            failures += 1
            print(f"  !! exit {proc.returncode}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    meta = {
        "suite": "graph_transformer_grn",
        "train_script": str(_TRAIN),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "split_subdir": split_subdir,
        "output_root": str(output_root),
        "epochs_forwarded": epochs,
        "seed_forwarded": seed,
        "n_tasks": len(tasks),
        "dry_run": args.dry_run,
        "train_grn_graph_defaults_for_rest": True,
        "no_cuda": bool(args.no_cuda),
        "extra_train_args": args.train_arg,
    }

    if not args.dry_run:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta["failures"] = failures
        summary_path = output_root / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "runs": rows}, f, indent=2)

        csv_path = output_root / "benchmark_summary.csv"
        if rows and isinstance(rows[0], dict):
            keys = sorted({k for r in rows for k in r})
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for r in rows:
                    f.write(
                        ",".join(
                            str(r.get(k, "")).replace(",", ";") for k in keys
                        )
                        + "\n"
                    )
        _log("-" * 72)
        _log(f"Wrote {summary_path}")
        if failures and not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
