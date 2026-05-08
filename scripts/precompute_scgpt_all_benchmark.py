#!/usr/bin/env python3
"""Write ``scgpt_gene_emb.pt`` in every Benchmark Dataset leaf (same discovery as benchmark_train_grn_suite).

Requires ``BL--ExpressionData.csv`` in each ``data_dir``. Output path is always
``<data_dir>/scgpt_gene_emb.pt``.

Example::

  python scripts/precompute_scgpt_all_benchmark.py \\
    --benchmark_root \"/path/to/benchmark/Dataset/Benchmark Dataset\" \\
    --scgpt_model_dir /path/to/scGPT_human

Environment::

  GRAPH_TRANSFORMER_BENCHMARK_ROOT  optional default for --benchmark_root
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
_PRECOMPUTE = _SCRIPTS / "precompute_scgpt_gene_emb.py"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import benchmark_train_grn_suite as _bench  # noqa: E402


def parse_args() -> argparse.Namespace:
    env_default = os.environ.get("GRAPH_TRANSFORMER_BENCHMARK_ROOT", "").strip()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--benchmark_root",
        type=str,
        default=env_default or str(_REPO_ROOT / "data/Dataset/Benchmark Dataset"),
        help="Benchmark Dataset root.",
    )
    p.add_argument("--scgpt_model_dir", type=str, required=True, help="scGPT checkpoint directory")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue after a failed precompute subprocess.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if data_dir/scgpt_gene_emb.pt already exists.",
    )
    p.add_argument(
        "--filter_slug_regex",
        type=str,
        default="",
        help="Only tasks whose slug matches this regex.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not _PRECOMPUTE.is_file():
        raise SystemExit(f"Missing {_PRECOMPUTE}")

    root = Path(args.benchmark_root).expanduser().resolve()
    tasks = _bench.discover_training_tasks(root)
    if not tasks:
        raise SystemExit(f"No tasks under {root}")

    if args.filter_slug_regex.strip():
        rx = re.compile(args.filter_slug_regex)
        tasks = [t for t in tasks if rx.search(t.slug)]
        if not tasks:
            raise SystemExit("No tasks left after --filter_slug_regex.")

    print(f"Benchmark root: {root}")
    print(f"Tasks: {len(tasks)}")
    print("-" * 72)

    failures = 0
    for i, task in enumerate(tasks):
        out_pt = task.data_dir / "scgpt_gene_emb.pt"
        if args.skip_existing and out_pt.is_file():
            print(f"[skip {i + 1}/{len(tasks)}] {task.slug} (exists {out_pt})")
            continue

        cmd = [
            sys.executable,
            str(_PRECOMPUTE),
            "--data_dir",
            str(task.data_dir),
            "--scgpt_model_dir",
            str(Path(args.scgpt_model_dir).expanduser().resolve()),
            "--output",
            str(out_pt),
        ]
        print(f"[{i + 1}/{len(tasks)}] {task.slug}")
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue

        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if proc.returncode != 0:
            failures += 1
            print(f"  !! exit {proc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                raise SystemExit(proc.returncode)

    print("-" * 72)
    if failures:
        print(f"Finished with {failures} failure(s).", file=sys.stderr)
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
