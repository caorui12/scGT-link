#!/usr/bin/env python3
"""
Aggregate GENELink-style 44-task benchmark results from multiple suites and write tables + figures.

Default layout assumes this file lives under ``…/deeplearning/Graph_Transformer/benchmarks/``.

Auto-loaded summaries under ``deeplearning/`` (each file must exist):

* ``Graph_Transformer/benchmark_summary.json`` → Graph Transformer
* ``GENELink/benchmark_summary.json`` → GENELink (uses ``test_aupr`` if ``test_auprc`` absent)
* ``GNNLink/out_benchmark_gnnlink/benchmark_summary.json`` (else ``GNNLink/benchmark_summary.json``) → GNNLink
* ``CNNC/out_benchmark_cnn/benchmark_summary.json`` → CNNC
* ``scGREAT/out_benchmark_scgreat/benchmark_summary.json`` → scGREAT
* ``Graph_Transformer/benchmarks/statistical_baselines/outputs/benchmark_summary.json`` → PCC / MI / GRNBoost2

Also scans ``deeplearning/**/benchmark_summary.json`` and saves the list beside the figures.

Output directory (default): ``<deeplearning>/benchmark/benchmark_comparison/``

Figures:

1. ``fig_mean_auroc_auprc.{pdf,png}``
2. ``fig_per_task_auroc.{pdf,png}``
3. ``fig_per_task_auprc.{pdf,png}``

Run::

  python benchmarks/summarize_benchmark_comparison.py

  python benchmarks/summarize_benchmark_comparison.py --deeplearning_root /path/to/deeplearning
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DEEPLEARNING_ROOT = _REPO_ROOT.parent
_DEFAULT_OUT = _DEFAULT_DEEPLEARNING_ROOT / "benchmark" / "benchmark_comparison"


def _load_standard_runs(path: Path, method_name: str) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for r in raw.get("runs", []):
        if r.get("skipped"):
            continue
        if r.get("exit_code") not in (0, None):
            continue
        slug = r.get("slug")
        if not slug:
            continue
        au = r.get("test_auroc")
        ap = r.get("test_auprc")
        if ap is None:
            ap = r.get("test_aupr")
        if au is None or ap is None:
            continue
        rows.append(
            {"slug": slug, "method": method_name, "auroc": float(au), "auprc": float(ap)}
        )
    return pd.DataFrame(rows)


def _load_statistical_baselines(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    labels = {"pcc": "PCC", "mi": "MI", "grnboost2": "GRNBoost2"}
    for r in raw.get("runs", []):
        slug = r.get("slug")
        if not slug:
            continue
        for key, label in labels.items():
            au = r.get(f"{key}_test_auroc")
            ap = r.get(f"{key}_test_auprc")
            if au is None or ap is None:
                continue
            rows.append(
                {"slug": slug, "method": label, "auroc": float(au), "auprc": float(ap)}
            )
    return pd.DataFrame(rows)


def _method_order(df: pd.DataFrame) -> list[str]:
    """Order methods by descending mean AUROC for legends / x-axis."""
    m = df.groupby("method")["auroc"].mean().sort_values(ascending=False)
    return list(m.index)


def _fig_mean_jitter(df: pd.DataFrame, out_base: Path) -> None:
    methods = _method_order(df)
    rng = np.random.default_rng(42)

    nt = int(df["slug"].nunique())
    fig_w = max(10.0, len(methods) * 1.35)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.8))
    titles = (f"Test AUROC ({nt} tasks)", f"Test AUPRC ({nt} tasks)")
    cols = ("auroc", "auprc")

    for ax, title, col in zip(axes, titles, cols):
        for i, method in enumerate(methods):
            sub = df.loc[df["method"] == method, col].values
            if len(sub) == 0:
                continue
            jitter = rng.normal(0, 0.07, size=len(sub))
            ax.scatter(
                np.full(len(sub), i) + jitter,
                sub,
                alpha=0.35,
                s=22,
                edgecolors="none",
            )
            ax.scatter(
                [i],
                [np.nanmean(sub)],
                marker="D",
                s=70,
                color="black",
                zorder=6,
                label=None,
            )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=28, ha="right")
        ax.set_ylabel(col.upper())
        ax.set_title(title)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    for suf in ("pdf", "png"):
        fig.savefig(f"{out_base}_mean_auroc_auprc.{suf}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_heatmap(df: pd.DataFrame, value_col: str, out_base: Path, title_suffix: str) -> None:
    methods = _method_order(df)
    slugs = sorted(df["slug"].unique())
    mat = np.full((len(methods), len(slugs)), np.nan)
    for i, m in enumerate(methods):
        for j, s in enumerate(slugs):
            sel = df[(df["method"] == m) & (df["slug"] == s)][value_col]
            if len(sel) == 1:
                mat[i, j] = sel.iloc[0]
            elif len(sel) > 1:
                mat[i, j] = sel.iloc[0]

    fig_h = max(4.0, len(methods) * 0.55 + 2)
    fig_w = max(14.0, len(slugs) * 0.22 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if value_col == "auroc":
        vmin, vmax = 0.5, 1.0
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(slugs, rotation=75, ha="right", fontsize=6)
    ax.set_title(f"Per-task {value_col.upper()} — {title_suffix}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(value_col.upper())
    fig.tight_layout()
    stem = f"{out_base}_per_task_{value_col}"
    for suf in ("pdf", "png"):
        fig.savefig(f"{stem}.{suf}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scan_benchmark_summaries(deep_root: Path) -> list[str]:
    found: list[Path] = []
    if not deep_root.is_dir():
        return []
    skip_parts = frozenset({".git", "__pycache__", "node_modules", ".venv"})
    for p in deep_root.rglob("benchmark_summary.json"):
        if any(part in skip_parts for part in p.parts):
            continue
        found.append(p.resolve())
    return [str(x) for x in sorted(set(found))]


def _default_suites(deep_root: Path) -> list[tuple[str, Path, str]]:
    """(display_name, path, kind). kind standard | baselines_flat."""
    cand: list[tuple[str, Path, str]] = []
    gt = deep_root / "Graph_Transformer" / "benchmark_summary.json"
    if gt.is_file():
        cand.append(("Graph Transformer", gt, "standard"))
    gen = deep_root / "GENELink" / "benchmark_summary.json"
    if gen.is_file():
        cand.append(("GENELink", gen, "standard"))
    gnn_o = deep_root / "GNNLink" / "out_benchmark_gnnlink" / "benchmark_summary.json"
    gnn_r = deep_root / "GNNLink" / "benchmark_summary.json"
    if gnn_o.is_file():
        cand.append(("GNNLink", gnn_o, "standard"))
    elif gnn_r.is_file():
        cand.append(("GNNLink", gnn_r, "standard"))
    cnn = deep_root / "CNNC" / "out_benchmark_cnn" / "benchmark_summary.json"
    if cnn.is_file():
        cand.append(("CNNC", cnn, "standard"))
    sc = deep_root / "scGREAT" / "out_benchmark_scgreat" / "benchmark_summary.json"
    if sc.is_file():
        cand.append(("scGREAT", sc, "standard"))
    bl = (
        deep_root
        / "Graph_Transformer"
        / "benchmarks"
        / "statistical_baselines"
        / "outputs"
        / "benchmark_summary.json"
    )
    if bl.is_file():
        cand.append(("__baselines__", bl, "baselines_flat"))
    return cand


def _write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    long_csv = out_dir / "benchmark_long.csv"
    df.sort_values(["method", "slug"]).to_csv(long_csv, index=False)

    pivot_auroc = df.pivot_table(index="slug", columns="method", values="auroc", aggfunc="first")
    pivot_auprc = df.pivot_table(index="slug", columns="method", values="auprc", aggfunc="first")
    pivot_auroc.round(2).to_csv(out_dir / "benchmark_wide_auroc.csv", float_format="%.2f")
    pivot_auprc.round(2).to_csv(out_dir / "benchmark_wide_auprc.csv", float_format="%.2f")

    agg = df.groupby("method").agg(
        n_tasks=("slug", "count"),
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_auprc=("auprc", "mean"),
        std_auprc=("auprc", "std"),
    ).reset_index()
    agg = agg.sort_values("mean_auroc", ascending=False)
    agg.to_csv(out_dir / "benchmark_method_summary.csv", index=False)

    summary_md = out_dir / "SUMMARY.md"
    lines = [
        "# Benchmark comparison summary",
        "",
        "Means ± std over tasks present for each method (task counts may differ if a suite is incomplete).",
        "",
        "| Method | n tasks | AUROC mean ± std | AUPRC mean ± std |",
        "|--------|---------|------------------|------------------|",
    ]
    for _, row in agg.iterrows():
        sau = row["std_auroc"]
        sap = row["std_auprc"]
        sau_s = float(sau) if sau == sau and not math.isnan(float(sau)) else 0.0
        sap_s = float(sap) if sap == sap and not math.isnan(float(sap)) else 0.0
        lines.append(
            f"| {row['method']} | {int(row['n_tasks'])} | "
            f"{row['mean_auroc']:.4f} ± {sau_s:.4f} | "
            f"{row['mean_auprc']:.4f} ± {sap_s:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Methods included",
            "",
            "See ``comparison_meta.json`` → ``sources`` for exact JSON paths. Typical suites: Graph Transformer, GENELink, GNNLink, CNNC, scGREAT; PCC/MI/GRNBoost2 from statistical_baselines outputs.",
            "",
            "## Figures",
            "",
            "- ``scan_benchmark_summary_paths.txt`` — all ``benchmark_summary.json`` under ``deeplearning_root``.",
            "- ``fig_mean_auroc_auprc.png`` — per-task jitter + black diamond = mean.",
            "- ``fig_per_task_auroc.png`` — heatmap methods × slug.",
            "- ``fig_per_task_auprc.png`` — heatmap methods × slug.",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize multi-method GENELink benchmark JSON suites.")
    p.add_argument(
        "--deeplearning_root",
        type=str,
        default=str(_DEFAULT_DEEPLEARNING_ROOT),
        help=f"Folder containing Graph_Transformer/, benchmark/, GENELink/, … Default: {_DEFAULT_DEEPLEARNING_ROOT}",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_DEFAULT_OUT),
        help=f"Write CSV + figures here (default: <deeplearning>/benchmark/benchmark_comparison)",
    )
    p.add_argument(
        "--no-defaults",
        action="store_true",
        help="Do not auto-add discovered suites under deeplearning_root.",
    )
    p.add_argument(
        "--suite",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        default=[],
        help="Additional NAME + PATH to benchmark_summary.json (standard runs format). "
        "Repeatable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deep_root = Path(args.deeplearning_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_txt = out_dir / "scan_benchmark_summary_paths.txt"
    scanned = _scan_benchmark_summaries(deep_root)
    scan_txt.write_text("\n".join(scanned) + ("\n" if scanned else ""), encoding="utf-8")

    frames: list[pd.DataFrame] = []

    suites: list[tuple[str, Path | None, str]] = []

    if not args.no_defaults:
        suites.extend(_default_suites(deep_root))

    for name, path_str in args.suite:
        suites.append((name, Path(path_str).expanduser().resolve(), "standard"))

    used_sources: list[dict[str, str]] = []
    for label, path, kind in suites:
        if path is None or not path.is_file():
            continue
        used_sources.append({"label": label, "path": str(path.resolve()), "kind": kind})
        if kind == "baselines_flat":
            frames.append(_load_statistical_baselines(path))
        else:
            frames.append(_load_standard_runs(path, label))

    if not frames:
        raise SystemExit(
            "No benchmark_summary.json sources found. Pass --suite NAME PATH "
            "or run without --no-defaults from Graph_Transformer."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["slug", "method"], keep="last")

    stem = str(out_dir / "fig")
    _write_tables(df, out_dir)
    _fig_mean_jitter(df, stem)
    _fig_heatmap(df, "auroc", stem, "all methods")
    _fig_heatmap(df, "auprc", stem, "all methods")

    meta = {
        "deeplearning_root": str(deep_root),
        "n_rows": len(df),
        "methods": sorted(df["method"].unique().tolist()),
        "n_slugs": int(df["slug"].nunique()),
        "output_dir": str(out_dir),
        "sources": used_sources,
        "scan_benchmark_summary_paths": scanned,
    }
    (out_dir / "comparison_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"deeplearning_root: {deep_root}")
    print(f"Scan wrote {len(scanned)} paths → {scan_txt}")
    print(f"Wrote tables + figures under {out_dir}")
    print(pd.read_csv(out_dir / "benchmark_method_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
