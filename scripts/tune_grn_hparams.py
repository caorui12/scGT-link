#!/usr/bin/env python3
"""
Hyperparameter grid search for ``train_grn_graph.py`` (ground-truth / gold networks + scGPT-friendly).

对每组超参启动一次独立子进程训练，读取各 run 目录下的 ``results.json``，汇总为
``summary.json`` / ``summary.csv``，并打印 test AUROC 最优的一次。

与 ``train_grn_graph.py`` 并列的参数名可直接放进 ``--grid_json``；网格做笛卡尔积。
布尔型开关（如 ``no_cuda``）在 JSON 里用 true/false 表示，脚本会按 argparse 规则加上 ``--flag``。

示例
====

  # 默认即 hESC_500（data/hESC/TFs+500 + Train_validation_test/hESC_500），无需传 dataset
  python scripts/tune_grn_hparams.py \\
    --epochs 80 --seed 42 \\
    --fixed_json scripts/tune_fixed_scgpt_gold_pearson.example.json \\
    --output_root out_tune_hESC500

  # 同时覆盖 data_dir + split_dir（其它任务时再使用）
  python scripts/tune_grn_hparams.py \\
    --data_dir path/to/expression_dir \\
    --split_dir path/to/splits \\
    --epochs 50 \\
    --output_root out_tune_manual

  # 自定义网格
  python scripts/tune_grn_hparams.py --epochs 100 \\
    --grid_json my_grid.json --output_root out_tune_custom

  # 只打印命令不执行
  python scripts/tune_grn_hparams.py --epochs 1 --dry_run
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

# 微调只针对 hESC_500；默认路径如下。若需其它数据请同时传入 --data_dir 与 --split_dir。
DEFAULT_TUNE_DATA_DIR = "data/hESC/TFs+500"
DEFAULT_TUNE_SPLIT_DIR = "data/Train_validation_test/hESC_500"
DEFAULT_TUNE_DATASET_KEY = "hESC_500"

# train_grn_graph 中 action="store_true" 的参数（网格里 true=加 flag，false=省略）
STORE_TRUE_FLAGS = frozenset(
    {
        "no_cuda",
        "no_edge_weight_attn",
        "no_edge_dir_attn",
    }
)

# 默认网格：与仓库根 ``summary.csv`` 中最佳 test_auroc 配置一致（单组，便于复现）。
# 需要再扫参时请使用 ``--grid_json``。
DEFAULT_GRID_REST: dict[str, list[Any]] = {
    "lr": [3e-4],
    "num_layers": [4],
    "gt_num_layers": [6],
    "gt_hidden_dim": [128],
    "gt_num_heads": [4],
}
DEFAULT_D_MODEL_NHEAD_PAIRS: list[tuple[int, int]] = [
    (768, 8),
]


def expand_default_grid() -> list[dict[str, Any]]:
    """Cartesian product of DEFAULT_GRID_REST × DEFAULT_D_MODEL_NHEAD_PAIRS (no invalid d_model/nhead)."""
    base_rows = expand_grid(DEFAULT_GRID_REST)
    rows: list[dict[str, Any]] = []
    for dm, nh in DEFAULT_D_MODEL_NHEAD_PAIRS:
        for r in base_rows:
            rows.append({**r, "d_model": int(dm), "nhead": int(nh)})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid search on hESC_500 (default paths): subprocess train_grn_graph.py per configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Default data: {DEFAULT_TUNE_DATA_DIR} + {DEFAULT_TUNE_SPLIT_DIR}. "
            "Override with both --data_dir and --split_dir."
        ),
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=f"Override expression folder (must use with --split_dir). Default: {DEFAULT_TUNE_DATA_DIR}",
    )
    p.add_argument(
        "--split_dir",
        type=str,
        default="",
        help=f"Override train/val/test split dir. Default: {DEFAULT_TUNE_SPLIT_DIR}",
    )
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_root",
        type=str,
        default="out_tune_grn",
        help="Each run writes to output_root/<slug>/",
    )
    p.add_argument(
        "--grid_json",
        type=str,
        default="",
        help="JSON object: param_name -> list of values (Cartesian product). "
        "Default: built-in DEFAULT_GRID_REST × DEFAULT_D_MODEL_NHEAD_PAIRS.",
    )
    p.add_argument(
        "--fixed_json",
        type=str,
        default="",
        help="JSON object: kwargs appended to every train command (e.g. expr_encoder, scgpt_emb_pt, prior_graph).",
    )
    p.add_argument("--no_cuda", action="store_true", help="Pass --no_cuda to every training run.")
    p.add_argument("--dry_run", action="store_true", help="Print commands only.")
    p.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first nonzero exit from train_grn_graph.py",
    )
    return p.parse_args()


def _load_json_dict(path: str, label: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"{label} not found: {p}")
    with open(p) as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return obj


def load_grid(path: str) -> dict[str, list[Any]]:
    return _load_json_dict(path, "--grid_json")


def merge_cmd_kv(cmd: list[str], key: str, value: Any) -> None:
    """Append ``--key value`` or ``--key`` for store_true flags."""
    if value is None:
        return
    if key in STORE_TRUE_FLAGS:
        if value is True:
            cmd.append(f"--{key}")
        elif value is False:
            return
        else:
            raise SystemExit(f"Grid/fixed key {key!r} must be true or false")
        return
    # empty string: skip optional string args
    if value == "":
        return
    cmd.extend([f"--{key}", str(value)])


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        rows.append(dict(zip(keys, combo)))
    return rows


def validate_config(cfg: dict[str, Any]) -> bool:
    """Drop invalid transformer / GraphTransformer head splits."""
    dm, nh = cfg.get("d_model"), cfg.get("nhead")
    if dm is not None and nh is not None:
        if int(dm) % int(nh) != 0:
            return False
    gh, gnh = cfg.get("gt_hidden_dim"), cfg.get("gt_num_heads")
    if gh is not None and gnh is not None:
        if int(gh) % int(gnh) != 0:
            return False
    # expr_encoder=scgpt refine head split (only if both appear in same cfg)
    if cfg.get("expr_encoder") == "scgpt":
        dm = cfg.get("d_model")
        rh = cfg.get("scgpt_refine_nhead")
        if dm is not None and rh is not None and int(cfg.get("scgpt_refine_layers", 0) or 0) > 0:
            if int(dm) % int(rh) != 0:
                return False
    return True


def slug_cfg(cfg: dict[str, Any], idx: int) -> str:
    parts = [f"{idx:04d}"]
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, float):
            s = f"{v:.0e}" if v != int(v) else str(int(v))
        elif isinstance(v, bool):
            s = "1" if v else "0"
        else:
            s = str(v).replace(".", "p").replace("/", "-")
        parts.append(f"{k}={s}")
    return "_".join(parts)[:220]


def run_one(
    cfg: dict[str, Any],
    run_dir: Path,
    base_cmd: list[str],
    dry_run: bool,
) -> tuple[int, dict[str, Any] | None]:
    cmd = list(base_cmd)
    cmd.extend(["--output_dir", str(run_dir)])
    for k, v in cfg.items():
        merge_cmd_kv(cmd, k, v)
    if dry_run:
        print(" ".join(cmd))
        return 0, None
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "cmd.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if proc.returncode != 0:
        return proc.returncode, None
    res_path = run_dir / "results.json"
    if not res_path.is_file():
        return 1, None
    with open(res_path, encoding="utf-8") as f:
        data = json.load(f)
    return 0, data


def main() -> None:
    args = parse_args()
    has_dd = bool(args.data_dir.strip())
    has_sd = bool(args.split_dir.strip())
    if has_dd != has_sd:
        raise SystemExit("Use both --data_dir and --split_dir together, or neither (default hESC_500).")
    if has_dd and has_sd:
        data_dir, split_dir = args.data_dir.strip(), args.split_dir.strip()
        dataset_key: str | None = None
    else:
        data_dir, split_dir = DEFAULT_TUNE_DATA_DIR, DEFAULT_TUNE_SPLIT_DIR
        dataset_key = DEFAULT_TUNE_DATASET_KEY

    if args.grid_json.strip():
        grid_meta: Any = load_grid(args.grid_json)
        rows = expand_grid(grid_meta)
    else:
        grid_meta = {
            "rest_axes": DEFAULT_GRID_REST,
            "d_model_nhead_pairs": [list(p) for p in DEFAULT_D_MODEL_NHEAD_PAIRS],
        }
        rows = expand_default_grid()

    fixed: dict[str, Any] = {}
    if args.fixed_json.strip():
        fixed = _load_json_dict(args.fixed_json, "--fixed_json")

    # grid 项覆盖同名的 fixed 项（便于在 grid_json 里单独 override 某一固定键）
    merged = [{**fixed, **c} for c in rows]
    valid = [c for c in merged if validate_config(c)]
    skipped = len(merged) - len(valid)
    if skipped:
        print(
            f"Skipped {skipped} configs (d_model%nhead, gt_hidden_dim%gt_num_heads, or scgpt refine nhead)",
            file=sys.stderr,
        )

    base_cmd: list[str] = [
        sys.executable,
        str(_TRAIN),
        "--data_dir",
        data_dir,
        "--split_dir",
        split_dir,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
    ]
    if args.no_cuda:
        base_cmd.append("--no_cuda")

    root = Path(args.output_root)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": dataset_key,
        "data_dir": data_dir,
        "split_dir": split_dir,
        "epochs": args.epochs,
        "seed": args.seed,
        "fixed": fixed,
        "grid": grid_meta,
        "n_configs": len(valid),
        "skipped_invalid": skipped,
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
            f"[{i + 1}/{len(valid)}] {name} | exit={code} | "
            f"test_auroc={row.get('test_auroc')} best_val={row.get('best_val_auroc')}",
            flush=True,
        )
        if code != 0 and args.fail_fast:
            print("fail_fast: stopping.", file=sys.stderr)
            break

    if args.dry_run:
        return

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "runs": summary_rows}, f, indent=2)

    with open(root / "summary.csv", "w", encoding="utf-8") as f:
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
    else:
        print("No successful runs with test_auroc.", file=sys.stderr)


if __name__ == "__main__":
    main()
