#!/usr/bin/env python3
"""
Bar plot (mean over 4 datasets) and heatmap (methods × datasets) for the
Graph_Transformer ``gt_data_four`` suite plus peer baselines.

Default inputs (sibling repos under the same ``deeplearning`` parent as this repo)::

  Graph_Transformer/out_gt_data_four/all_results.json          → Graph Transformer
  scGREAT/out_scgreat_gt_data_four/benchmark_summary.json
  CNNC/out_cnn_gt_data_four/benchmark_summary.json
  GENELink/out_genelink_gt_data_four/benchmark_summary.json   (test_aupr → AUPRC)
  GNNLink/out_gnnlink_gt_data_four/benchmark_summary.json
  Graph_Transformer/benchmarks/statistical_baselines/outputs_gt_data_four/benchmark_summary.json

Outputs (default ``out_gt_data_four/figures/`` under repo root)::

  fig_gt_four_mean_bar_auroc_auprc.{pdf,png}   — 1×2 subplots (AUROC | AUPRC)
  fig_gt_four_heatmap_auroc_auprc.{pdf,png}   — 1×2 subplots (AUROC | AUPRC)

Run from repo root::

  python scripts/plot_gt_data_four_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DL = _REPO_ROOT.parent

_DATASETS = ["hESC_500", "hESC_1000", "mESC_500", "mESC_1000"]


def _load_gt_transformer(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for slug, block in raw["datasets"].items():
        out[slug] = (float(block["test_auroc"]), float(block["test_auprc"]))
    return out


def _auprc_key(run: dict[str, Any]) -> float | None:
    if run.get("test_auprc") is not None:
        return float(run["test_auprc"])
    if run.get("test_aupr") is not None:
        return float(run["test_aupr"])
    return None


def _load_runs_benchmark(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for r in raw.get("runs", []):
        slug = r.get("slug")
        if not slug:
            continue
        au = r.get("test_auroc")
        ap = _auprc_key(r)
        if au is None or ap is None:
            continue
        out[slug] = (float(au), float(ap))
    return out


def _load_statistical(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_method: dict[str, dict[str, tuple[float, float]]] = {
        "PCC": {},
        "MI": {},
        "GRNBoost2": {},
    }
    key_map = {
        "PCC": ("pcc_test_auroc", "pcc_test_auprc"),
        "MI": ("mi_test_auroc", "mi_test_auprc"),
        "GRNBoost2": ("grnboost2_test_auroc", "grnboost2_test_auprc"),
    }
    for r in raw.get("runs", []):
        slug = r.get("slug")
        if not slug:
            continue
        for label, (k_au, k_ap) in key_map.items():
            if k_au in r and k_ap in r:
                by_method[label][slug] = (float(r[k_au]), float(r[k_ap]))
    return by_method


def build_tables(
    *,
    gt_all_results: Path,
    scgreat_summary: Path,
    cnn_summary: Path,
    genelink_summary: Path,
    gnnlink_summary: Path,
    statistical_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    gt = _load_gt_transformer(gt_all_results)

    blocks: list[tuple[str, dict[str, tuple[float, float]]]] = [
        ("Graph Transformer", gt),
        ("scGREAT", _load_runs_benchmark(scgreat_summary)),
        ("CNNC", _load_runs_benchmark(cnn_summary)),
        ("GENELink", _load_runs_benchmark(genelink_summary)),
        ("GNNLink", _load_runs_benchmark(gnnlink_summary)),
    ]
    stat = _load_statistical(statistical_summary)
    for m in ["PCC", "MI", "GRNBoost2"]:
        blocks.append((m, stat[m]))

    rows_au: dict[str, list[float]] = {}
    rows_ap: dict[str, list[float]] = {}
    method_order: list[str] = []
    for label, d in blocks:
        method_order.append(label)
        rows_au[label] = [d[s][0] for s in _DATASETS]
        rows_ap[label] = [d[s][1] for s in _DATASETS]

    df_au = pd.DataFrame(rows_au, index=_DATASETS).T
    df_ap = pd.DataFrame(rows_ap, index=_DATASETS).T
    df_au = df_au.loc[method_order]
    df_ap = df_ap.loc[method_order]
    return df_au, df_ap, method_order


def _save_fig(path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")


def plot_mean_bars(df_au: pd.DataFrame, df_ap: pd.DataFrame, out_base: Path) -> None:
    means_au = df_au.mean(axis=1)
    means_ap = df_ap.mean(axis=1)
    x = np.arange(len(means_au))
    w = 0.7

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.suptitle("GT data four — mean test metrics (n=4 datasets)", fontsize=13, y=1.02)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(means_au)))

    ax = axes[0]
    ax.bar(x, means_au.values, width=w, color=colors, edgecolor="0.2", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(means_au.index, rotation=35, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.02)
    ax.set_title("AUROC (mean)")
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    ax.bar(x, means_ap.values, width=w, color=colors, edgecolor="0.2", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(means_ap.index, rotation=35, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1.02)
    ax.set_title("AUPRC (mean)")
    ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    _save_fig(out_base)
    plt.close()


def plot_heatmaps(df_au: pd.DataFrame, df_ap: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("GT data four — per-dataset test metrics", fontsize=13, y=1.02)

    for ax, df, title, cbar_label in (
        (axes[0], df_au, "AUROC", "AUROC"),
        (axes[1], df_ap, "AUPRC", "AUPRC"),
    ):
        arr = df.values.astype(float)
        im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_xticklabels(df.columns, rotation=30, ha="right")
        ax.set_yticklabels(df.index)
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                txt_color = "white" if v > 0.55 else "0.15"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=txt_color, fontsize=8)

    plt.tight_layout()
    _save_fig(out_base)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    dl = _DEFAULT_DL
    p.add_argument(
        "--deeplearning_root",
        type=Path,
        default=dl,
        help=f"Parent of Graph_Transformer / scGREAT / … (default: {dl})",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=_REPO_ROOT / "out_gt_data_four" / "figures",
        help="Directory for PDF/PNG outputs",
    )
    p.add_argument("--gt_all_results", type=Path, default=None)
    p.add_argument("--scgreat_summary", type=Path, default=None)
    p.add_argument("--cnn_summary", type=Path, default=None)
    p.add_argument("--genelink_summary", type=Path, default=None)
    p.add_argument("--gnnlink_summary", type=Path, default=None)
    p.add_argument("--statistical_summary", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dl: Path = args.deeplearning_root

    def R(p: Path | None, default: Path) -> Path:
        return p if p is not None else default

    paths = {
        "gt_all_results": R(
            args.gt_all_results, _REPO_ROOT / "out_gt_data_four" / "all_results.json"
        ),
        "scgreat_summary": R(
            args.scgreat_summary, dl / "scGREAT" / "out_scgreat_gt_data_four" / "benchmark_summary.json"
        ),
        "cnn_summary": R(
            args.cnn_summary, dl / "CNNC" / "out_cnn_gt_data_four" / "benchmark_summary.json"
        ),
        "genelink_summary": R(
            args.genelink_summary, dl / "GENELink" / "out_genelink_gt_data_four" / "benchmark_summary.json"
        ),
        "gnnlink_summary": R(
            args.gnnlink_summary, dl / "GNNLink" / "out_gnnlink_gt_data_four" / "benchmark_summary.json"
        ),
        "statistical_summary": R(
            args.statistical_summary,
            _REPO_ROOT
            / "benchmarks"
            / "statistical_baselines"
            / "outputs_gt_data_four"
            / "benchmark_summary.json",
        ),
    }

    missing = [k for k, v in paths.items() if not v.is_file()]
    if missing:
        sys.stderr.write("Missing input files:\n")
        for k in missing:
            sys.stderr.write(f"  {k}: {paths[k]}\n")
        return 1

    df_au, df_ap, _ = build_tables(**paths)  # type: ignore[arg-type]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_base = out_dir / "fig_gt_four_mean_bar_auroc_auprc"
    heat_base = out_dir / "fig_gt_four_heatmap_auroc_auprc"
    plot_mean_bars(df_au, df_ap, bar_base)
    plot_heatmaps(df_au, df_ap, heat_base)

    print(f"Wrote {bar_base.with_suffix('.pdf')} (+ .png)")
    print(f"Wrote {heat_base.with_suffix('.pdf')} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
