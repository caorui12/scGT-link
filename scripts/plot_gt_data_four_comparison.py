#!/usr/bin/env python3
"""
Bar plot (mean over 4 datasets) and heatmap (methods × datasets) for the
Graph_Transformer ``gt_data_four`` suite plus peer baselines.

Default inputs (sibling repos under the same ``deeplearning`` parent as this repo)::

  Graph_Transformer/out_gt_data_four/all_results.json          → scGT-Link
  benchmark/gt_grn_local_outputs.json (or GT-GRN/outputs/)   → GT-GRN (local runs)
  scGREAT/out_scgreat_gt_data_four/benchmark_summary.json
  CNNC/out_cnn_gt_data_four/benchmark_summary.json
  GENELink/out_genelink_gt_data_four/benchmark_summary.json   (test_aupr → AUPRC)
  GNNLink/out_gnnlink_gt_data_four/benchmark_summary.json
  HGATLink/out_hgatlink_gt_data_four/benchmark_summary.json
  Graph_Transformer/benchmarks/statistical_baselines/outputs_gt_data_four/benchmark_summary.json

(scGREAT: local GT four only; omitted from 44-task suite until new run.)

Outputs (default ``out_gt_data_four/figures/`` under repo root)::

  fig_gt_four_mean_bar_auroc_auprc.{pdf,png}   — 1×2 subplots (AUROC | AUPRC)
  fig_gt_four_heatmap_auroc_auprc.{pdf,png}   — 1×2 subplots (AUROC | AUPRC)
  fig_gt_four_per_dataset.{pdf,png}           — 2×2 subplots (one panel per dataset)

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
_METHOD_SCGT_LINK = "scGT-Link"


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


def _load_gt_grn_local(outputs_root: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for slug in _DATASETS:
        p = outputs_root / slug / "metrics.json"
        if not p.is_file():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        au = raw.get("test_auc", raw.get("test_auroc"))
        ap = raw.get("test_ap", raw.get("test_auprc"))
        if au is not None and ap is not None:
            out[slug] = (float(au), float(ap))
    return out


_BENCHMARK44_TO_GT4 = {
    "Specific__hESC__500": "hESC_500",
    "Specific__hESC__1000": "hESC_1000",
    "Specific__mESC__500": "mESC_500",
    "Specific__mESC__1000": "mESC_1000",
}


def _load_hgatlink_gt_four(hgat_summary: Path, hgat_benchmark44: Path | None) -> dict[str, tuple[float, float]]:
    """Prefer ``out_hgatlink_gt_data_four``; else map ``Specific__*`` from full 44-task JSON."""
    out: dict[str, tuple[float, float]] = {}
    if hgat_summary.is_file():
        out.update(_load_runs_benchmark(hgat_summary))
    if len(out) >= len(_DATASETS):
        return {k: out[k] for k in _DATASETS if k in out}
    if hgat_benchmark44 and hgat_benchmark44.is_file():
        for slug, (au, ap) in _load_runs_benchmark(hgat_benchmark44).items():
            key = _BENCHMARK44_TO_GT4.get(slug)
            if key and key not in out:
                out[key] = (au, ap)
    return {k: out[k] for k in _DATASETS if k in out}


def build_tables(
    *,
    gt_all_results: Path,
    gt_grn_outputs: Path,
    scgreat_summary: Path,
    cnn_summary: Path,
    genelink_summary: Path,
    gnnlink_summary: Path | None,
    hgatlink_summary: Path | None,
    hgatlink_benchmark44: Path | None,
    statistical_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    gt = _load_gt_transformer(gt_all_results)
    gt_grn = _load_gt_grn_local(gt_grn_outputs)

    blocks: list[tuple[str, dict[str, tuple[float, float]]]] = [
        (_METHOD_SCGT_LINK, gt),
    ]
    if gt_grn:
        blocks.append(("GT-GRN", gt_grn))
    blocks.extend(
        [
            ("scGREAT", _load_runs_benchmark(scgreat_summary)),
            ("CNNC", _load_runs_benchmark(cnn_summary)),
            ("GENELink", _load_runs_benchmark(genelink_summary)),
        ]
    )
    if gnnlink_summary is not None and gnnlink_summary.is_file():
        blocks.append(("GNNLink", _load_runs_benchmark(gnnlink_summary)))
    if hgatlink_summary is not None and (
        hgatlink_summary.is_file()
        or (hgatlink_benchmark44 is not None and hgatlink_benchmark44.is_file())
    ):
        hgat = _load_hgatlink_gt_four(hgatlink_summary, hgatlink_benchmark44)
        if hgat:
            blocks.append(("HGATLink", hgat))
    stat = _load_statistical(statistical_summary)
    for m in ["PCC", "MI", "GRNBoost2"]:
        blocks.append((m, stat[m]))

    rows_au: dict[str, list[float]] = {}
    rows_ap: dict[str, list[float]] = {}
    method_order: list[str] = []
    for label, d in blocks:
        method_order.append(label)
        rows_au[label] = [d.get(s, (np.nan, np.nan))[0] for s in _DATASETS]
        rows_ap[label] = [d.get(s, (np.nan, np.nan))[1] for s in _DATASETS]

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


def plot_per_dataset(df_au: pd.DataFrame, df_ap: pd.DataFrame, out_base: Path) -> None:
    """2×2 panels: grouped AUROC / AUPRC bars per method, one subplot per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Local GT data four — test metrics by dataset",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    methods = df_au.index.tolist()
    x = np.arange(len(methods))
    w = 0.36
    colors = ("#4C72B0", "#DD8452")

    for ax, dataset in zip(axes.flat, _DATASETS):
        au = df_au[dataset].values.astype(float)
        ap = df_ap[dataset].values.astype(float)
        ax.bar(x - w / 2, au, w, label="AUROC", color=colors[0], edgecolor="0.25", linewidth=0.4)
        ax.bar(x + w / 2, ap, w, label="AUPRC", color=colors[1], edgecolor="0.25", linewidth=0.4)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=42, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)
        for i, (a, p) in enumerate(zip(au, ap)):
            if not np.isnan(a):
                ax.text(i - w / 2, min(a + 0.02, 1.01), f"{a:.2f}", ha="center", va="bottom", fontsize=6)
            if not np.isnan(p):
                ax.text(i + w / 2, min(p + 0.02, 1.01), f"{p:.2f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
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
    p.add_argument("--gt_grn_outputs", type=Path, default=None)
    p.add_argument("--scgreat_summary", type=Path, default=None)
    p.add_argument("--cnn_summary", type=Path, default=None)
    p.add_argument("--genelink_summary", type=Path, default=None)
    p.add_argument("--gnnlink_summary", type=Path, default=None)
    p.add_argument("--hgatlink_summary", type=Path, default=None)
    p.add_argument("--hgatlink_benchmark44", type=Path, default=None)
    p.add_argument("--statistical_summary", type=Path, default=None)
    p.add_argument(
        "--also_benchmark_comparison",
        action="store_true",
        help="Also write fig_gt_four_* and CSVs under deeplearning/benchmark/benchmark_comparison/",
    )
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
        "gt_grn_outputs": R(args.gt_grn_outputs, dl / "GT-GRN" / "outputs"),
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
        "hgatlink_summary": R(
            args.hgatlink_summary,
            dl / "HGATLink" / "out_hgatlink_gt_data_four" / "benchmark_summary.json",
        ),
        "hgatlink_benchmark44": R(
            args.hgatlink_benchmark44, dl / "HGATLink" / "benchmark_summary.json"
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

    required = [
        "gt_all_results",
        "scgreat_summary",
        "cnn_summary",
        "genelink_summary",
        "statistical_summary",
    ]
    missing = [k for k in required if not paths[k].is_file()]
    if not paths["gt_grn_outputs"].is_dir():
        missing.append("gt_grn_outputs")
    elif not _load_gt_grn_local(paths["gt_grn_outputs"]):
        missing.append("gt_grn_outputs (no metrics.json under any dataset)")
    if missing:
        sys.stderr.write("Missing input files:\n")
        for k in missing:
            sys.stderr.write(f"  {k}: {paths[k]}\n")
        return 1

    gnnlink_path = paths.pop("gnnlink_summary")
    hgat_path = paths.pop("hgatlink_summary")
    hgat_b44 = paths.pop("hgatlink_benchmark44")
    if not gnnlink_path.is_file():
        sys.stderr.write(f"Optional (skipped): gnnlink_summary: {gnnlink_path}\n")
    if not hgat_path.is_file() and not hgat_b44.is_file():
        sys.stderr.write(
            f"Optional (skipped): hgatlink_summary / hgatlink_benchmark44: {hgat_path}\n"
        )

    df_au, df_ap, _ = build_tables(
        **paths,
        gnnlink_summary=gnnlink_path,
        hgatlink_summary=hgat_path,
        hgatlink_benchmark44=hgat_b44,
    )  # type: ignore[arg-type]

    out_dirs = [args.out_dir]
    if args.also_benchmark_comparison:
        out_dirs.append(dl / "benchmark" / "benchmark_comparison")

    for out_dir in out_dirs:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df_au.round(3).to_csv(out_dir / "gt_four_wide_auroc.csv")
        df_ap.round(3).to_csv(out_dir / "gt_four_wide_auprc.csv")

        bar_base = out_dir / "fig_gt_four_mean_bar_auroc_auprc"
        heat_base = out_dir / "fig_gt_four_heatmap_auroc_auprc"
        per_ds_base = out_dir / "fig_gt_four_per_dataset"
        plot_mean_bars(df_au, df_ap, bar_base)
        plot_heatmaps(df_au, df_ap, heat_base)
        plot_per_dataset(df_au, df_ap, per_ds_base)

        print(f"\n[{out_dir}]")
        print(f"  {bar_base.with_suffix('.pdf')} (+ .png)")
        print(f"  {heat_base.with_suffix('.pdf')} (+ .png)")
        print(f"  {per_ds_base.with_suffix('.pdf')} (+ .png)")
        print("Mean AUROC:\n", df_au.mean(axis=1).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
