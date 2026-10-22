#!/usr/bin/env python3
"""
Aggregate GENELink-style 44-task benchmark results from multiple suites and write tables + figures.

Default layout assumes this file lives under ``…/deeplearning/Graph_Transformer/benchmarks/``.

Auto-loaded summaries under ``deeplearning/`` (each file must exist):

* ``Graph_Transformer/benchmark_summary.json`` → scGT-Link
* ``GENELink/benchmark_summary.json`` → GENELink (uses ``test_aupr`` if ``test_auprc`` absent)
* ``GNNLink/out_benchmark_gnnlink/benchmark_summary.json`` (else ``GNNLink/benchmark_summary.json``) → GNNLink
* ``CNNC/out_benchmark_cnn/benchmark_summary.json`` → CNNC
* ``Graph_Transformer/benchmarks/statistical_baselines/outputs/benchmark_summary.json`` → PCC / MI / GRNBoost2
* ``HGATLink/benchmark_summary.json`` (preferred) or ``out_benchmark44`` + optional ``out_hgatlink_gt_data_four`` → HGATLink
* ``scGREAT/out_benchmark_scgreat/benchmark_summary.json`` → scGREAT
* ``GNE/out_benchmark_gne/benchmark_summary.json`` (else ``GNE/benchmark_summary.json``) → GNE

Also scans ``deeplearning/**/benchmark_summary.json`` and saves the list beside the figures.

Output directory (default): ``<deeplearning>/benchmark/benchmark_comparison/``

Figures (PPT-friendly defaults — large fonts, 300 dpi PNG):

1. ``fig_mean_auroc_auprc.{pdf,png}`` — mean ± std bar charts
2. ``fig_faceted_44_auroc.{pdf,png}`` / ``fig_faceted_44_auprc.{pdf,png}`` — GENELink-style panels
3. ``fig_by_network_auroc.{pdf,png}`` / ``fig_by_network_auprc.{pdf,png}`` — methods × network type

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
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DEEPLEARNING_ROOT = _REPO_ROOT.parent
_DEFAULT_OUT = _DEFAULT_DEEPLEARNING_ROOT / "benchmark" / "benchmark_comparison"
METHOD_SCGT_LINK = "scGT-Link"
METHOD_SCGT_TEST = "scGT-Link (test)"
_HIGHLIGHT_METHODS = frozenset({METHOD_SCGT_LINK, METHOD_SCGT_TEST})
_SAVE_FORMATS: tuple[str, ...] = ("pdf", "png")
_NET_ORDER = ("Lofgof", "Non_Specific", "STRING", "Specific")
_NET_LABELS = {
    "Lofgof": "Lofgof",
    "Non_Specific": "Non-specific",
    "STRING": "STRING",
    "Specific": "Specific",
}
_PPT_DPI = 300
_SPECIES_ORDER = ("hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L")
_FACETED_NET_BLOCKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("STRING", "STRING", _SPECIES_ORDER),
    ("Non_Specific", "Non-specific ChIP-seq", _SPECIES_ORDER),
    ("Lofgof", "LOF / GOF", ("mESC",)),
    ("Specific", "Cell-type specific ChIP-seq", _SPECIES_ORDER),
)
_GENE_PANELS: tuple[tuple[str, str], ...] = (
    ("500", "TFs + 500 genes"),
    ("1000", "TFs + 1000 genes"),
)
_BENCHMARK_CMAP = LinearSegmentedColormap.from_list(
    "benchmark_mag",
    ["#0a0a0a", "#4a148c", "#ad1457", "#f06292", "#fce4ec"],
    N=256,
)


def _apply_ppt_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.2,
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _method_colors(methods: list[str]) -> list[str]:
    palette = plt.cm.tab10(np.linspace(0, 0.85, max(len(methods), 1)))
    out: list[str] = []
    for i, m in enumerate(methods):
        if m == METHOD_SCGT_LINK:
            out.append("#C0392B")
        elif m == METHOD_SCGT_TEST:
            out.append("#3C5488")
        else:
            out.append(plt.matplotlib.colors.to_hex(palette[i]))
    return out


def _savefig(fig: Any, path_stem: str) -> None:
    for suf in _SAVE_FORMATS:
        kw: dict[str, Any] = {"bbox_inches": "tight"}
        if suf == "png":
            kw["dpi"] = _PPT_DPI
        fig.savefig(f"{path_stem}.{suf}", **kw)


def _slug_net_type(slug: str) -> str:
    return slug.split("__", 1)[0]


def _load_standard_runs(path: Path, method_name: str) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for r in raw.get("runs", []):
        if r.get("exit_code") not in (0, None):
            continue
        slug = r.get("slug")
        if not slug:
            continue
        au = r.get("test_auroc")
        if au is None:
            au = r.get("test_auc")
        ap = r.get("test_auprc")
        if ap is None:
            ap = r.get("test_aupr")
        if ap is None:
            ap = r.get("test_ap")
        if au is None or ap is None:
            continue
        # ``skipped`` in suite JSON means "did not re-train" (e.g. --skip_existing), not "drop metrics".
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


def _method_agg(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby("method")
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
        )
        .reindex(methods)
    )
    return agg


def _fig_mean_bars(df: pd.DataFrame, out_base: Path) -> None:
    """Mean ± std bar charts (readable in PowerPoint)."""
    _apply_ppt_style()
    methods = _method_order(df)
    agg = _method_agg(df, methods)
    colors = _method_colors(methods)
    nt = int(df["slug"].nunique())

    fig_w = max(16.0, len(methods) * 1.15)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 7.5))
    fig.suptitle(
        f"GENELink benchmark — {nt} datasets",
        fontsize=24,
        fontweight="bold",
        y=1.02,
    )

    specs = (
        ("mean_auroc", "std_auroc", "AUROC", 0.45, 1.02),
        ("mean_auprc", "std_auprc", "AUPRC", 0.0, 1.02),
    )
    x = np.arange(len(methods))
    for ax, (mcol, scol, ylab, ymin, ymax) in zip(axes, specs):
        means = agg[mcol].values.astype(float)
        stds = agg[scol].fillna(0).values.astype(float)
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            width=0.72,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
            capsize=5,
            error_kw={"elinewidth": 1.8, "capthick": 1.8},
            zorder=3,
        )
        for bar, h in zip(bars, ["///" if m == METHOD_SCGT_TEST else None for m in methods]):
            if h:
                bar.set_hatch(h)
        for bar, v, s, mname in zip(bars, means, stds, methods):
            weight = "bold" if mname in _HIGHLIGHT_METHODS else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(v + s + 0.015, ymax - 0.02),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight=weight,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=32, ha="right", fontsize=15)
        ax.set_ylabel(ylab, fontsize=18, fontweight="bold")
        ax.set_title(f"Mean test {ylab}", fontsize=20, fontweight="bold", pad=12)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=14)

    fig.tight_layout()
    _savefig(fig, f"{out_base}_mean_auroc_auprc")
    plt.close(fig)


def _fig_nettype_heatmap(df: pd.DataFrame, value_col: str, out_base: Path) -> None:
    """Methods × network-type mean (4 columns — readable on slides)."""
    _apply_ppt_style()
    methods = _method_order(df)
    work = df.copy()
    work["net"] = work["slug"].map(_slug_net_type)
    pivot = work.groupby(["method", "net"], as_index=False)[value_col].mean()
    mat = pivot.pivot(index="method", columns="net", values=value_col)
    mat = mat.reindex(index=methods, columns=[c for c in _NET_ORDER if c in mat.columns])

    if value_col == "auroc":
        vmin, vmax = 0.55, 1.0
        cmap = "YlOrRd"
    else:
        vmin, vmax = 0.0, 1.0
        cmap = "YlOrRd"

    n_m, n_n = mat.shape
    fig, ax = plt.subplots(figsize=(max(9.0, n_n * 2.2), max(7.0, n_m * 0.72 + 2.5)))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    col_labels = [_NET_LABELS.get(c, c) for c in mat.columns]
    ax.set_xticks(np.arange(n_n))
    ax.set_xticklabels(col_labels, fontsize=18, fontweight="bold")
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(mat.index, fontsize=16)
    ax.set_title(
        f"Mean test {value_col.upper()} by network type",
        fontsize=22,
        fontweight="bold",
        pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(value_col.upper(), fontsize=16, fontweight="bold")

    arr = mat.values.astype(float)
    for i in range(n_m):
        for j in range(n_n):
            v = arr[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > (vmin + vmax) * 0.58 else "#1a1a1a"
            fw = "bold" if mat.index[i] in _HIGHLIGHT_METHODS else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=15, fontweight=fw)

    fig.tight_layout()
    _savefig(fig, f"{out_base}_by_network_{value_col}")
    plt.close(fig)


def _panel_matrix(
    df: pd.DataFrame,
    methods: list[str],
    net: str,
    species_list: tuple[str, ...],
    n_genes: str,
    value_col: str,
) -> np.ndarray:
    mat = np.full((len(species_list), len(methods)), np.nan, dtype=float)
    for i, sp in enumerate(species_list):
        slug = f"{net}__{sp}__{n_genes}"
        sub = df.loc[df["slug"] == slug]
        for j, m in enumerate(methods):
            row = sub.loc[sub["method"] == m, value_col]
            if len(row) == 1:
                mat[i, j] = float(row.iloc[0])
    return mat


def _heatmap_text_color(v: float, vmin: float, vmax: float) -> str:
    if np.isnan(v):
        return "#888888"
    t = (v - vmin) / max(vmax - vmin, 1e-9)
    return "white" if t < 0.55 else "#111111"


def _draw_faceted_panel(
    ax: plt.Axes,
    mat: np.ndarray,
    methods: list[str],
    species_list: tuple[str, ...],
    *,
    vmin: float,
    vmax: float,
    show_method_labels: bool,
    show_species_labels: bool,
    show_cell_values: bool = False,
) -> None:
    n_r, n_c = mat.shape
    im = ax.imshow(mat, aspect="auto", cmap=_BENCHMARK_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(n_c))
    ax.set_yticks(np.arange(n_r))
    if show_method_labels:
        ax.set_xticklabels(methods, rotation=42, ha="right", fontsize=12)
        ax.tick_params(axis="x", length=0, pad=6)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if show_species_labels:
        ax.set_yticklabels(species_list, fontsize=13)
        ax.tick_params(axis="y", length=0, pad=4)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    if show_cell_values:
        for i in range(n_r):
            for j in range(n_c):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                fw = "bold" if methods[j] in _HIGHLIGHT_METHODS else "normal"
                fs = 11 if n_r == 1 else 10
                disp_v = min(float(v), 0.99)
                ax.text(
                    j,
                    i,
                    f"{disp_v:.2f}",
                    ha="center",
                    va="center",
                    color=_heatmap_text_color(v, vmin, vmax),
                    fontsize=fs,
                    fontweight=fw,
                )

    for edge in ("top", "right", "bottom", "left"):
        ax.spines[edge].set_linewidth(0.8)
        ax.spines[edge].set_color("#cccccc")
    return im


def _fig_faceted_44(df: pd.DataFrame, value_col: str, out_base: Path) -> None:
    """GENELink-style faceted heatmap: network blocks × {500, 1000} gene panels."""
    _apply_ppt_style()
    methods = _method_order(df)
    if value_col == "auroc":
        vmin, vmax = 0.5, 1.0
        metric_label = "AUROC"
    else:
        vmin, vmax = 0.0, 1.0
        metric_label = "AUPRC"

    height_ratios = [len(b[2]) for b in _FACETED_NET_BLOCKS]
    fig_h = sum(height_ratios) * 0.42 + 3.5
    fig_w = max(20.0, len(methods) * 1.05 + 6.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"GENELink benchmark — test {metric_label} (44 datasets)",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )

    gs = gridspec.GridSpec(
        len(_FACETED_NET_BLOCKS),
        3,
        figure=fig,
        width_ratios=[0.11, 1.0, 1.0],
        height_ratios=height_ratios,
        wspace=0.12,
        hspace=0.22,
        left=0.06,
        right=0.92,
        top=0.94,
        bottom=0.12,
    )

    # Column headers (500 / 1000)
    for j, (_, title) in enumerate(_GENE_PANELS):
        fig.text(
            0.28 + j * 0.34,
            0.955,
            title,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    last_im = None
    for row_i, (net_key, net_label, species_list) in enumerate(_FACETED_NET_BLOCKS):
        # Section label on the left
        label_ax = fig.add_subplot(gs[row_i, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.5,
            net_label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            rotation=90,
            transform=label_ax.transAxes,
        )

        for col_i, (n_genes, _) in enumerate(_GENE_PANELS):
            ax = fig.add_subplot(gs[row_i, col_i + 1])
            mat = _panel_matrix(df, methods, net_key, species_list, n_genes, value_col)
            last_im = _draw_faceted_panel(
                ax,
                mat,
                methods,
                species_list,
                vmin=vmin,
                vmax=vmax,
                show_method_labels=(row_i == len(_FACETED_NET_BLOCKS) - 1),
                show_species_labels=(col_i == 0),
                show_cell_values=True,
            )

    if last_im is not None:
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(f"Test {metric_label}", fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

    _savefig(fig, f"{out_base}_faceted_44_{value_col}")
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


# Graph_Transformer gt_data_four registry → GENELink 44-task slug (same expression CSV).
_GT4_SLUG_TO_BENCHMARK44 = {
    "hESC_500": "Specific__hESC__500",
    "hESC_1000": "Specific__hESC__1000",
    "mESC_500": "Specific__mESC__500",
    "mESC_1000": "Specific__mESC__1000",
}


def _hgatlink_run_metrics(run: dict[str, Any]) -> tuple[str, float, float] | None:
    if run.get("skipped"):
        return None
    if run.get("status") not in (None, "ok") and run.get("exit_code") not in (0, None):
        return None
    slug = run.get("slug")
    if not slug:
        return None
    slug = _GT4_SLUG_TO_BENCHMARK44.get(slug, slug)
    au = run.get("test_auroc", run.get("test_auc"))
    ap = run.get("test_auprc", run.get("test_aupr", run.get("test_ap")))
    if au is None or ap is None:
        return None
    return slug, float(au), float(ap)


def _load_hgatlink_merged(deep_root: Path) -> pd.DataFrame:
    """Merge HGATLink summaries; later paths override (full 44-task JSON wins over partial)."""
    by_slug: dict[str, tuple[float, float]] = {}
    paths = [
        deep_root / "HGATLink" / "out_hgatlink_gt_data_four" / "benchmark_summary.json",
        deep_root / "HGATLink" / "out_benchmark44" / "benchmark_summary.json",
        deep_root / "HGATLink" / "benchmark_summary.json",
    ]
    for path in paths:
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for r in raw.get("runs", []):
            parsed = _hgatlink_run_metrics(r)
            if parsed is None:
                continue
            slug, au, ap = parsed
            by_slug[slug] = (au, ap)
    rows = [
        {"slug": s, "method": "HGATLink", "auroc": au, "auprc": ap}
        for s, (au, ap) in sorted(by_slug.items())
    ]
    return pd.DataFrame(rows)


def _suite_n_ok_runs(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for r in raw.get("runs", []):
        if r.get("exit_code") not in (0, None):
            continue
        slug = r.get("slug")
        if not slug:
            continue
        au = r.get("test_auroc", r.get("test_auc"))
        ap = r.get("test_auprc", r.get("test_aupr", r.get("test_ap")))
        if au is not None and ap is not None:
            n += 1
    return n


def _default_suites(
    deep_root: Path,
    scgt_link_summary: Path | None = None,
    scgt_link_label: str = METHOD_SCGT_LINK,
) -> list[tuple[str, Path, str]]:
    """(display_name, path, kind). kind standard | baselines_flat."""
    cand: list[tuple[str, Path, str]] = []
    gt = scgt_link_summary
    if gt is None or not gt.is_file():
        for rel in (
            "Graph_Transformer/out_benchmark_train_pos_only_200ep/benchmark_summary.json",
            "Graph_Transformer/benchmark_summary.json",
        ):
            p = deep_root / rel
            if p.is_file():
                gt = p
                break
    if gt is not None and gt.is_file():
        cand.append((scgt_link_label, gt, "standard"))
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
    hgat44 = deep_root / "HGATLink" / "out_benchmark44" / "benchmark_summary.json"
    hgat4 = deep_root / "HGATLink" / "out_hgatlink_gt_data_four" / "benchmark_summary.json"
    if hgat44.is_file() or hgat4.is_file():
        cand.append(("HGATLink", None, "hgatlink_merged"))
    scgreat = deep_root / "scGREAT" / "out_benchmark_scgreat" / "benchmark_summary.json"
    if scgreat.is_file():
        cand.append(("scGREAT", scgreat, "standard"))
    gne_o = deep_root / "GNE" / "out_benchmark_gne" / "benchmark_summary.json"
    gne_r = deep_root / "GNE" / "benchmark_summary.json"
    if gne_o.is_file():
        cand.append(("GNE", gne_o, "standard"))
    elif gne_r.is_file():
        cand.append(("GNE", gne_r, "standard"))
    deeprig = deep_root / "benchmark" / "out_deeprig" / "benchmark_summary.json"
    if deeprig.is_file():
        cand.append(("DeepRIG", deeprig, "standard"))
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
            "See ``comparison_meta.json`` → ``sources`` for exact JSON paths. Typical suites: scGT-Link, scGREAT, GENELink, GNNLink, GNE, CNNC, HGATLink; PCC/MI/GRNBoost2 from statistical_baselines outputs.",
            "",
            "## Figures",
            "",
            "- ``scan_benchmark_summary_paths.txt`` — all ``benchmark_summary.json`` under ``deeplearning_root``.",
            "- ``fig_mean_auroc_auprc.png`` — mean ± std bar charts (PPT-friendly).",
            "- ``fig_faceted_44_auroc.png`` / ``fig_faceted_44_auprc.png`` — GENELink-style split panels.",
            "- ``fig_by_network_auroc.png`` / ``fig_by_network_auprc.png`` — methods × network type.",
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
        "--scgt_link_summary",
        type=str,
        default="",
        help="scGT-Link benchmark_summary.json (default: out_benchmark_train_pos_only_200ep, "
        "else Graph_Transformer/benchmark_summary.json).",
    )
    p.add_argument(
        "--scgt_link_label",
        type=str,
        default=METHOD_SCGT_LINK,
        help="Display name for scGT-Link in tables/figures (default: scGT-Link).",
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
    p.add_argument(
        "--pdf_only",
        action="store_true",
        help="Write PDF figures only (no PNG).",
    )
    return p.parse_args()


def main() -> None:
    global _SAVE_FORMATS
    args = parse_args()
    if args.pdf_only:
        _SAVE_FORMATS = ("pdf",)
    deep_root = Path(args.deeplearning_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_txt = out_dir / "scan_benchmark_summary_paths.txt"
    scanned = _scan_benchmark_summaries(deep_root)
    scan_txt.write_text("\n".join(scanned) + ("\n" if scanned else ""), encoding="utf-8")

    frames: list[pd.DataFrame] = []

    suites: list[tuple[str, Path | None, str]] = []

    scgt_override = (
        Path(args.scgt_link_summary).expanduser().resolve()
        if args.scgt_link_summary.strip()
        else None
    )
    if not args.no_defaults:
        suites.extend(
            _default_suites(
                deep_root,
                scgt_link_summary=scgt_override,
                scgt_link_label=args.scgt_link_label.strip() or METHOD_SCGT_LINK,
            )
        )

    for name, path_str in args.suite:
        suites.append((name, Path(path_str).expanduser().resolve(), "standard"))

    used_sources: list[dict[str, str]] = []
    for label, path, kind in suites:
        if kind == "hgatlink_merged":
            df_h = _load_hgatlink_merged(deep_root)
            if df_h.empty:
                continue
            srcs = []
            for rel in (
                "HGATLink/out_hgatlink_gt_data_four/benchmark_summary.json",
                "HGATLink/out_benchmark44/benchmark_summary.json",
                "HGATLink/benchmark_summary.json",
            ):
                p = deep_root / rel
                if p.is_file():
                    srcs.append(str(p.resolve()))
            used_sources.append(
                {
                    "label": label,
                    "path": "; ".join(srcs) if srcs else "",
                    "kind": kind,
                    "n_tasks": int(df_h["slug"].nunique()),
                }
            )
            frames.append(df_h)
            continue
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
    _fig_mean_bars(df, stem)
    _fig_faceted_44(df, "auroc", stem)
    _fig_faceted_44(df, "auprc", stem)
    _fig_nettype_heatmap(df, "auroc", stem)
    _fig_nettype_heatmap(df, "auprc", stem)

    scgt_hybrid_meta: dict[str, Any] | None = None
    scgt_meta_path: Path | None = None
    if scgt_override is not None and scgt_override.is_file():
        scgt_meta_path = scgt_override
    else:
        for src in used_sources:
            if src.get("label") == METHOD_SCGT_LINK or src.get("label") == (
                args.scgt_link_label.strip() or METHOD_SCGT_LINK
            ):
                p = Path(str(src.get("path") or ""))
                if p.is_file():
                    scgt_meta_path = p
                break
    if scgt_meta_path is not None and scgt_meta_path.is_file():
        try:
            scgt_raw = json.loads(scgt_meta_path.read_text(encoding="utf-8"))
            sm = scgt_raw.get("meta")
            if isinstance(sm, dict) and (
                sm.get("selection_counts") is not None or sm.get("candidates") is not None
            ):
                scgt_hybrid_meta = {
                    "path": str(scgt_meta_path.resolve()),
                    "method": sm.get("method"),
                    "rule": sm.get("rule"),
                    "candidates": sm.get("candidates"),
                    "selection_counts": sm.get("selection_counts"),
                    "n_tasks": sm.get("n_tasks"),
                }
        except (OSError, json.JSONDecodeError, TypeError):
            scgt_hybrid_meta = None

    meta: dict[str, Any] = {
        "deeplearning_root": str(deep_root),
        "n_rows": len(df),
        "methods": sorted(df["method"].unique().tolist()),
        "n_slugs": int(df["slug"].nunique()),
        "output_dir": str(out_dir),
        "sources": used_sources,
        "scan_benchmark_summary_paths": scanned,
    }
    if scgt_hybrid_meta is not None:
        meta["scgt_link_hybrid"] = scgt_hybrid_meta
    (out_dir / "comparison_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if scgt_hybrid_meta is not None:
        summary_md = out_dir / "SUMMARY.md"
        extra = [
            "",
            "## scGT-Link hybrid selection",
            "",
            f"- Source: `{scgt_hybrid_meta.get('path', '')}`",
            f"- Method: {scgt_hybrid_meta.get('method', '')}",
            f"- Rule: {scgt_hybrid_meta.get('rule', '')}",
            "",
        ]
        cands = scgt_hybrid_meta.get("candidates") or []
        if cands:
            extra.append("Candidates:")
            extra.append("")
            for c in cands:
                if isinstance(c, dict):
                    extra.append(
                        f"- `{c.get('label')}` ← `{c.get('path')}` "
                        f"(n_complete={c.get('n_complete')})"
                    )
            extra.append("")
        sel = scgt_hybrid_meta.get("selection_counts") or {}
        au_wins = sel.get("auroc_wins") if isinstance(sel, dict) else None
        ap_wins = sel.get("auprc_wins") if isinstance(sel, dict) else None
        if isinstance(au_wins, dict) or isinstance(ap_wins, dict):
            extra.append("| Source | AUROC wins | AUPRC wins |")
            extra.append("|--------|------------|------------|")
            keys = sorted(
                set(au_wins or {}) | set(ap_wins or {}),
                key=lambda k: (-int((au_wins or {}).get(k, 0)), str(k)),
            )
            for k in keys:
                extra.append(
                    f"| {k} | {int((au_wins or {}).get(k, 0))} | "
                    f"{int((ap_wins or {}).get(k, 0))} |"
                )
            extra.append("")
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8") + "\n".join(extra),
            encoding="utf-8",
        )

    print(f"deeplearning_root: {deep_root}")
    print(f"Scan wrote {len(scanned)} paths → {scan_txt}")
    print(f"Wrote tables + figures under {out_dir}")
    print(pd.read_csv(out_dir / "benchmark_method_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
