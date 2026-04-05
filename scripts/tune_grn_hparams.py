#!/usr/bin/env python3
"""
Grid search over train_grn_graph.py hyperparameters (no gene symbol embeddings).

Runs train_grn_graph.py as a subprocess per configuration, reads each run's results.json,
and writes a summary table under --output_root.

Example:
  python scripts/tune_grn_hparams.py \\
    --data_dir data/hESC/TFs+500 \\
    --split_dir data/Train_validation_test/hESC_500 \\
    --epochs 50 --seed 42 \\
    --output_root out_hparam_nobert

  python scripts/tune_grn_hparams.py --grid_json my_grid.json ...

Default grid is small; override with --grid_json (see DEFAULT_GRID shape below).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN = _REPO_ROOT / "train_grn_graph.py"

# Keys must match train_grn_graph.py argparse names. Lists are Cartesian-producted.
# d_model must be divisible by nhead for the gene Transformer.
DEFAULT_GRID: dict[str, list[Any]] = {
    "lr": [1e-4, 3e-4],
    "d_model": [128],
    "nhead": [4],
    "top_k": [15, 20, 30],
    "gt_num_layers": [4, 6],
}


def parse_args():
    p = argparse.ArgumentParser(description="GRN hyperparameter grid (no --gene_bert_emb)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default="out_hparam_nobert")
    p.add_argument(
        "--grid_json",
        type=str,
        default="",
        help="JSON object: param_name -> list of values (same shape as DEFAULT_GRID)",
    )
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    p.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first nonzero exit from train_grn_graph.py",
    )
    return p.parse_args()


def load_grid(path: str) -> dict[str, list[Any]]:
    with open(path) as f:
        g = json.load(f)
    if not isinstance(g, dict) or not g:
        raise SystemExit("--grid_json must be a non-empty JSON object")
    for k, v in g.items():
        if not isinstance(v, list) or not v:
            raise SystemExit(f"grid[{k!r}] must be a non-empty list")
    return g


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        rows.append(dict(zip(keys, combo)))
    return rows


def validate_transformer_divisible(cfg: dict[str, Any]) -> bool:
    dm = cfg.get("d_model")
    nh = cfg.get("nhead")
    if dm is not None and nh is not None:
        if int(dm) % int(nh) != 0:
            return False
    return True


def slug_cfg(cfg: dict[str, Any], idx: int) -> str:
    parts = [f"{idx:04d}"]
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, float):
            s = f"{v:.0e}" if v != int(v) else str(int(v))
        else:
            s = str(v).replace(".", "p")
        parts.append(f"{k}={s}")
    return "_".join(parts)[:200]


def run_one(
    cfg: dict[str, Any],
    run_dir: Path,
    base_cmd: list[str],
    dry_run: bool,
) -> tuple[int, dict[str, Any] | None]:
    cmd = list(base_cmd)
    cmd.extend(["--output_dir", str(run_dir)])
    for k, v in cfg.items():
        cmd.extend([f"--{k}", str(v)])
    if dry_run:
        print(" ".join(cmd))
        return 0, None
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "cmd.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if proc.returncode != 0:
        return proc.returncode, None
    res_path = run_dir / "results.json"
    if not res_path.is_file():
        return 1, None
    with open(res_path) as f:
        data = json.load(f)
    return 0, data


def main():
    args = parse_args()
    grid = load_grid(args.grid_json) if args.grid_json.strip() else DEFAULT_GRID

    rows = expand_grid(grid)
    valid = [c for c in rows if validate_transformer_divisible(c)]
    skipped = len(rows) - len(valid)
    if skipped:
        print(f"Skipped {skipped} configs where d_model % nhead != 0", file=sys.stderr)

    base_cmd = [
        sys.executable,
        str(_TRAIN),
        "--data_dir",
        args.data_dir,
        "--split_dir",
        args.split_dir,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
    ]
    if args.no_cuda:
        base_cmd.append("--no_cuda")
    # Do not pass --gene_bert_emb (train_grn_graph default = no symbol embeddings)

    root = Path(args.output_root)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "split_dir": args.split_dir,
        "epochs": args.epochs,
        "seed": args.seed,
        "gene_bert_emb": "",
        "grid": grid,
        "n_configs": len(valid),
        "skipped_invalid_nhead": skipped,
    }

    for i, cfg in enumerate(valid):
        name = slug_cfg(cfg, i)
        run_dir = root / name
        code, res = run_one(cfg, run_dir, base_cmd, args.dry_run)
        if args.dry_run:
            continue
        row = {**cfg, "run_dir": str(run_dir), "exit_code": code}
        if res is not None:
            row["best_val_auroc"] = res.get("best_val_auroc")
            row["test_auroc"] = res.get("test_auroc")
            row["test_auprc"] = res.get("test_auprc")
        summary_rows.append(row)
        print(
            f"[{i+1}/{len(valid)}] {name} | exit={code} | "
            f"test_auroc={row.get('test_auroc')} best_val={row.get('best_val_auroc')}",
            flush=True,
        )
        if code != 0 and args.fail_fast:
            print("fail_fast: stopping.", file=sys.stderr)
            break

    if args.dry_run:
        return

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(root / "summary.json", "w") as f:
        json.dump({"meta": meta, "runs": summary_rows}, f, indent=2)

    # Flat CSV-friendly copy
    with open(root / "summary.csv", "w") as f:
        if summary_rows:
            keys = list(summary_rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in summary_rows:
                f.write(
                    ",".join(
                        str(r.get(k, "")).replace(",", ";") for k in keys
                    )
                    + "\n"
                )

    best = max(
        (r for r in summary_rows if r.get("test_auroc") is not None),
        key=lambda r: float(r["test_auroc"]),
        default=None,
    )
    if best:
        print("-" * 72)
        print(f"Best test_auroc={best['test_auroc']:.6f} -> {best.get('run_dir')}")
        print(f"Wrote {root / 'summary.json'} and {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
