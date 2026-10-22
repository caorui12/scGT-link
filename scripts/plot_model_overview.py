#!/usr/bin/env python3
"""Vector reconstruction of the scGT-Link paper-mechanism overview (PDF only).

Layout reference: assets/fig_model_overview_ai_v46.png (scGREAT-style 3-band AI layout) / fig_model_overview.pdf (vector).
Fig. 1 shows model mechanism; inputs A explains parallel G0 (topology + Pearson w) and scGPT (gene names).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
    RegularPolygon,
)
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

INK = "#222222"
MUTED = "#666666"
LINE = "#C8C8C8"
NAVY = "#3C5488"
TEAL = "#00A087"
SLATE = "#4A6FA5"
VERM = "#E64B35"
TF = "#7E6148"
TG = "#00A087"
GT_BG = "#EAF1F8"
CARD = "#FFFFFF"
SPLIT_BG = "#F7F9FC"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
        }
    )


def _box(ax, xy, w, h, *, fc=CARD, ec=LINE, lw=0.9, r=0.012, z=1):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def _txt(ax, x, y, s, *, size=7.0, weight="normal", color=INK, ha="center", va="center", z=5, linespacing=1.25):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, linespacing=linespacing)


def _arrow(ax, p0, p1, *, color=MUTED, lw=1.15, ms=10, z=4, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=1,
            shrinkB=1,
            zorder=z,
        )
    )


def _heatmap(ax, x, y, w, h):
    rng = np.array(
        [
            [0.2, 0.55, 0.9, 0.4],
            [0.7, 0.25, 0.35, 0.8],
            [0.3, 0.85, 0.5, 0.15],
            [0.95, 0.45, 0.2, 0.65],
        ]
    )
    nr, nc = rng.shape
    cw, ch = w / nc, h / nr
    lo = np.array([0.25, 0.45, 0.75])
    mid = np.array([0.97, 0.97, 0.96])
    hi = np.array([0.90, 0.40, 0.20])
    for i in range(nr):
        for j in range(nc):
            t = float(rng[i, j])
            c = (1 - 2 * t) * lo + 2 * t * mid if t < 0.5 else (2 - 2 * t) * mid + (2 * t - 1) * hi
            ax.add_patch(
                Rectangle(
                    (x + j * cw, y + (nr - 1 - i) * ch),
                    cw * 0.9,
                    ch * 0.85,
                    facecolor=c,
                    edgecolor="white",
                    lw=0.25,
                    zorder=2,
                )
            )


def _mini_graph(ax, x, y, w, h):
    tfs = [(x + 0.22 * w, y + 0.72 * h), (x + 0.22 * w, y + 0.28 * h)]
    tgs = [(x + 0.78 * w, y + 0.80 * h), (x + 0.82 * w, y + 0.50 * h), (x + 0.78 * w, y + 0.18 * h)]
    for i, j in ((0, 0), (0, 1), (1, 2)):
        _arrow(ax, tfs[i], tgs[j], color=NAVY, lw=0.95, ms=7)
    for cx, cy in tfs:
        ax.add_patch(
            RegularPolygon((cx, cy), 5, radius=0.018, orientation=np.pi / 2, facecolor=TF, edgecolor=INK, lw=0.4, zorder=5)
        )
    for cx, cy in tgs:
        ax.add_patch(Circle((cx, cy), 0.014, facecolor=TG, edgecolor=INK, lw=0.4, zorder=5))


def _hex_grid(ax, x, y, w, h):
    shades = ["#3C5488", "#4C6494", "#6B84AE", "#8491B4", "#9AA6C2", "#B8C1D6"]
    nr, nc = 2, 6
    cw, ch = w / nc, h / nr
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(
                RegularPolygon(
                    (x + (j + 0.5) * cw, y + (nr - i - 0.5) * ch),
                    6,
                    radius=min(cw, ch) * 0.36,
                    orientation=0,
                    facecolor=shades[(i + j) % len(shades)],
                    edgecolor="white",
                    lw=0.3,
                    zorder=2,
                )
            )


def _capsules(ax, x, y, n=6, color=TEAL, gap=0.028):
    for i in range(n):
        ax.add_patch(
            FancyBboxPatch(
                (x + i * gap, y),
                0.022,
                0.028,
                boxstyle="round,pad=0.001,rounding_size=0.008",
                facecolor=color,
                edgecolor=INK,
                linewidth=0.35,
                zorder=4,
            )
        )


def _split_panel(ax) -> tuple[float, float, float, float]:
    """Top panel: BEELINE supervised edge splits. Returns Train box anchor (x, y)."""
    y0, h = 0.845, 0.115
    _box(ax, (0.04, y0), 0.92, h, ec=NAVY, lw=1.05, fc=SPLIT_BG)
    _txt(
        ax,
        0.50,
        y0 + h - 0.018,
        "Supervised edge splits (BEELINE / GENELink protocol)",
        size=8.2,
        weight="bold",
    )

    _box(ax, (0.06, y0 + 0.028), 0.15, 0.058, ec=MUTED, lw=0.75)
    _txt(ax, 0.135, y0 + 0.068, r"Gold TF$\rightarrow$target", size=6.4, weight="bold")
    _txt(ax, 0.135, y0 + 0.044, "pairs (Label.csv)", size=5.4, color=MUTED)
    _arrow(ax, (0.21, y0 + 0.057), (0.245, y0 + 0.057), color=MUTED, ms=7)

    splits = (
        ("Train", NAVY, "pos + matched negatives\nTrain⁺ only builds $G_0$"),
        ("Validation", SLATE, "held-out pos/neg pairs\ncheckpoint selection"),
        ("Test", "#8491B4", "held-out pos/neg pairs\nAUROC / AUPRC"),
    )
    train_x = 0.375
    for i, (name, col, sub) in enumerate(splits):
        x = 0.27 + i * 0.22
        _box(ax, (x, y0 + 0.022), 0.19, 0.072, ec=col, lw=0.95)
        _txt(ax, x + 0.095, y0 + 0.072, name, size=7.2, weight="bold", color=col)
        _txt(ax, x + 0.095, y0 + 0.042, sub, size=5.2, color=MUTED, linespacing=1.12)
        if i == 0:
            train_x = x + 0.095

    _txt(
        ax,
        0.50,
        y0 + 0.006,
        "Specific: per-TF hard negatives (~70/10/20).  "
        "STRING / Non-Specific / Lofgof: density-matched negatives.  "
        "Val/Test positives never enter $G_0$.",
        size=5.5,
        color=MUTED,
    )
    return train_x, y0 + 0.022, 0.50, y0 + 0.34


def plot(out_pdf: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    train_x, train_y, g0_x, _ = _split_panel(ax)

    # ── Bottom inputs ──────────────────────────────────────────────────────
    _box(ax, (0.04, 0.06), 0.28, 0.28, ec=NAVY, lw=1.1)
    _txt(ax, 0.18, 0.315, r"$X$  (genes $\times$ cells)", size=8.0, weight="bold")
    _heatmap(ax, 0.07, 0.11, 0.22, 0.16)
    _txt(ax, 0.18, 0.085, "log-normalized scRNA-seq", size=6.0, color=MUTED)

    _box(ax, (0.36, 0.06), 0.28, 0.28, ec=NAVY, lw=1.1)
    _txt(ax, 0.50, 0.315, r"$G_0$  train-positive prior", size=8.0, weight="bold")
    _mini_graph(ax, 0.39, 0.10, 0.22, 0.18)
    _txt(ax, 0.50, 0.085, r"Train$^+$ edges only;  $w{=}r_{ij}$,  $d{=}\pm 1$", size=6.0, color=MUTED)

    _box(ax, (0.68, 0.06), 0.28, 0.28, ec=NAVY, lw=1.1)
    _txt(ax, 0.82, 0.315, r"scGPT  $E$", size=8.0, weight="bold")
    _hex_grid(ax, 0.71, 0.12, 0.22, 0.14)
    _txt(ax, 0.82, 0.085, "pretrained gene tokens (lookup)", size=6.0, color=MUTED)

    # split Train⁺ → G0; scGPT / G0 into encoder; G0 topology into GT
    _arrow(ax, (train_x, train_y), (g0_x, 0.34), color=NAVY, lw=1.05, ls="--", ms=8)
    _txt(ax, g0_x + 0.06, 0.58, "Train$^+$", size=5.8, color=NAVY, ha="left")
    _arrow(ax, (0.50, 0.34), (0.50, 0.42), color=NAVY, lw=1.2)
    _arrow(ax, (0.50, 0.42), (0.28, 0.48), color=NAVY, lw=1.0)
    _arrow(ax, (0.82, 0.34), (0.82, 0.42), color=TEAL, lw=1.2)
    _arrow(ax, (0.82, 0.42), (0.28, 0.55), color=TEAL, lw=1.0)

    # ── Left: node encoder (v2: proj(scGPT)+LapPE, then GT-entry projections) ─
    _box(ax, (0.04, 0.42), 0.30, 0.40, ec=TEAL, lw=1.2, fc="#F3FAF7")
    _txt(ax, 0.19, 0.78, "Node encoder (scGPT + LapPE)", size=8.0, weight="bold", color=TEAL)
    steps = [
        (0.705, r"Linear projection on $E$"),
        (0.625, r"$+$ LapPE$(G_0)$"),
    ]
    for y, lab in steps:
        _box(ax, (0.07, y - 0.028), 0.24, 0.058, ec=TEAL, lw=0.7)
        _txt(ax, 0.19, y, lab, size=6.2)
    _box(ax, (0.07, 0.478), 0.24, 0.058, ec=TEAL, lw=0.7)
    _txt(ax, 0.19, 0.507, r"Node features $h$", size=6.0)
    _box(ax, (0.34, 0.455), 0.22, 0.055, ec=SLATE, lw=0.8, fc="#EEF3FA")
    _txt(ax, 0.45, 0.482, r"GT entry: Linear$(h)$ $+$ Linear(LapPE)", size=5.8, color=SLATE)
    _capsules(ax, 0.10, 0.445, n=6, color=TEAL)
    _txt(ax, 0.19, 0.435, r"Initial node features $h$", size=7.0, weight="bold")
    for y0, y1 in ((0.705, 0.653), (0.625, 0.536)):
        _arrow(ax, (0.19, y0 - 0.028), (0.19, y1 + 0.028), color=TEAL, lw=0.9, ms=8)
    _arrow(ax, (0.19, 0.478), (0.34, 0.482), color=TEAL, lw=1.0, ms=8)
    _arrow(ax, (0.50, 0.42), (0.45, 0.455), color=NAVY, lw=0.9, ls="--", ms=7)
    _txt(ax, 0.52, 0.44, "LapPE", size=5.5, color=NAVY, ha="left")

    # ── Center: Directed GT (novelty) ──────────────────────────────────────
    _box(ax, (0.38, 0.42), 0.36, 0.40, ec=SLATE, lw=1.5, fc=GT_BG)
    _txt(ax, 0.56, 0.78, "Directed Graph Transformer", size=8.5, weight="bold", color=SLATE)
    _txt(ax, 0.56, 0.745, r"dst-agg, edge msg $[w,d]$", size=6.5, color=MUTED)

    _box(ax, (0.41, 0.52), 0.30, 0.20, ec=SLATE, lw=0.9)
    _txt(ax, 0.56, 0.695, "Each layer", size=7.0, weight="bold")
    _mini_graph(ax, 0.43, 0.55, 0.12, 0.12)
    _txt(ax, 0.49, 0.535, "Sparse MHA on $G_0$", size=5.8, color=MUTED)
    items = ["dst-agg", r"$[w,d]$ msg", "LN + drop", "FFN residual block"]
    for i, lab in enumerate(items):
        yy = 0.68 - i * 0.035
        ax.add_patch(Circle((0.58, yy), 0.005, facecolor=SLATE, edgecolor="none", zorder=5))
        _txt(ax, 0.595, yy, lab, size=6.0, ha="left")

    _capsules(ax, 0.46, 0.455, n=6, color=SLATE)
    _txt(ax, 0.56, 0.445, r"Output node embeddings $z$", size=7.0, weight="bold")
    _arrow(ax, (0.45, 0.482), (0.41, 0.52), color=SLATE, lw=1.0, ms=8)
    _arrow(ax, (0.56, 0.52), (0.56, 0.485), color=SLATE, lw=0.9, ms=8)

    _arrow(ax, (0.50, 0.34), (0.56, 0.42), color=NAVY, lw=1.1, ls="--")
    _txt(ax, 0.60, 0.38, r"graph $G_0$", size=6.0, color=NAVY, ha="left")

    # ── Top-right: link predictor ──────────────────────────────────────────
    _box(ax, (0.78, 0.42), 0.18, 0.40, ec=VERM, lw=1.2, fc="#FDF4F0")
    _txt(ax, 0.87, 0.78, "Link head", size=8.0, weight="bold", color=VERM)
    ax.add_patch(
        RegularPolygon((0.835, 0.72), 5, radius=0.016, orientation=np.pi / 2, facecolor=TF, edgecolor=INK, lw=0.4, zorder=5)
    )
    ax.add_patch(Circle((0.905, 0.72), 0.013, facecolor=TG, edgecolor=INK, lw=0.4, zorder=5))
    _txt(ax, 0.835, 0.695, r"TF $i$", size=5.8, color=TF)
    _txt(ax, 0.905, 0.695, r"gene $j$", size=5.8, color=TEAL)

    _box(ax, (0.80, 0.60), 0.14, 0.06, ec=VERM, lw=0.7)
    _txt(ax, 0.87, 0.63, r"Concat $[z_i; z_j]$", size=6.0)

    _box(ax, (0.80, 0.50), 0.14, 0.07, ec=VERM, lw=0.7)
    _txt(ax, 0.87, 0.535, r"MLP (ReLU)", size=7.0, weight="bold")

    _box(ax, (0.80, 0.44), 0.14, 0.045, ec=VERM, lw=0.9, fc="#F8D3CB")
    _txt(ax, 0.87, 0.462, r"logit  $\hat{y}_{ij}$", size=7.0, weight="bold")

    _arrow(ax, (0.74, 0.46), (0.80, 0.46), color=SLATE, lw=1.3)
    for y0, y1 in ((0.70, 0.66), (0.60, 0.57), (0.50, 0.485)):
        _arrow(ax, (0.87, y0), (0.87, y1), color=VERM, lw=0.85, ms=7)

    # Train split supervises link head only (no generic eval box)
    _arrow(ax, (train_x + 0.08, train_y), (0.87, 0.42), color=VERM, lw=0.95, ls="--", ms=7)
    _txt(ax, 0.72, 0.395, "Train pairs", size=5.8, color=VERM, ha="center")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "manuscript" / "figures" / "Fig_1_model_structure.pdf"),
    )
    args = p.parse_args()
    out = Path(args.output).expanduser().resolve()
    if out.suffix.lower() != ".pdf":
        out = out.with_suffix(".pdf")
    plot(out)


if __name__ == "__main__":
    main()
