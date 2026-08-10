#!/usr/bin/env python3
"""Publication-quality architecture overview for scGT-Link v2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

# Nature / Cell inspired palette (colorblind-friendly)
C = {
    "ink": "#1a1a1a",
    "muted": "#5c5c5c",
    "line": "#cccccc",
    "input": "#2166ac",
    "input_bg": "#eef4fb",
    "encoder": "#1b7837",
    "encoder_bg": "#edf7f0",
    "gt": "#404040",
    "gt_bg": "#f3f3f3",
    "head": "#d95f02",
    "head_bg": "#fef3e8",
    "train": "#762a83",
    "train_bg": "#f6eef8",
    "arrow": "#666666",
    "accent": "#4575b4",
}


def _style_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def _box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    fc="#ffffff",
    ec=C["ink"],
    lw=1.0,
    fontsize=8.5,
    fontweight="normal",
    ha="center",
    va="center",
    radius=0.02,
    zorder=2,
    text_color=None,
    pad_x=0.012,
    pad_y=0.012,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if ha == "left":
        tx = x + pad_x
    elif ha == "right":
        tx = x + w - pad_x
    else:
        tx = x + w / 2
    if va == "top":
        ty = y + h - pad_y
    elif va == "bottom":
        ty = y + pad_y
    else:
        ty = y + h / 2
    ax.text(
        tx,
        ty,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color or C["ink"],
        zorder=zorder + 1,
        linespacing=1.35,
    )
    return patch


def _arrow(ax, p0, p1, *, color=C["arrow"], lw=1.2, style="-|>", mutation=12, zorder=1):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def _panel_label(ax, letter: str, x: float, y: float) -> None:
    circ = plt.Circle((x, y), 0.018, facecolor=C["ink"], edgecolor="none", zorder=5)
    ax.add_patch(circ)
    ax.text(x, y, letter, ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)


def _section_frame(ax, xy, w, h, title: str, accent: str, bg: str, letter: str) -> None:
    x, y = xy
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor=bg,
        edgecolor=accent,
        linewidth=1.6,
        zorder=0,
    )
    ax.add_patch(frame)
    ax.text(x + 0.012, y + h - 0.018, title, fontsize=10.5, fontweight="bold", color=accent, va="top", zorder=3)
    _panel_label(ax, letter, x + 0.018, y + h - 0.018)


def plot_overview(out_base: Path, *, journal: str = "nature") -> None:
    _style_rc()
    # scipilot-figure-skill journal_specs: Nature full-width ~190 mm (7.48 in); min label 6 pt
    fig_w, fig_h = (7.2, 4.8) if journal == "nature" else (7.48, 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "scGT-Link v2: scGPT-informed graph transformer for GRN inference",
        fontsize=12.5,
        fontweight="bold",
        color=C["ink"],
        y=0.985,
    )
    ax.text(
        0.5,
        0.952,
        "Train-positive prior graph + expression + scGPT embeddings (no gold-network leakage)",
        ha="center",
        va="top",
        fontsize=9,
        color=C["muted"],
    )

    y0, ph = 0.14, 0.76  # panel bottom, panel height

    # ── A. Inputs ──────────────────────────────────────────────────────────
    _section_frame(ax, (0.02, y0), 0.168, ph, "Inputs", C["input"], C["input_bg"], "A")
    _box(
        ax,
        (0.032, y0 + 0.58),
        0.146,
        0.11,
        "Expression matrix\n$X \\in \\mathbb{R}^{G \\times C}$\nBL--ExpressionData.csv",
        fc="white",
        ec=C["input"],
        fontsize=7.6,
    )
    _box(
        ax,
        (0.032, y0 + 0.42),
        0.146,
        0.12,
        "Train positive edges\n$E_+ \\subset$ Train\\_set (Label=1)\nTF $\\rightarrow$ target only",
        fc="white",
        ec=C["input"],
        fontsize=7.6,
    )
    _box(
        ax,
        (0.032, y0 + 0.26),
        0.146,
        0.12,
        "Prior graph $\\mathcal{G}_0$\nPearson $r_{ij}$ on $X$\nedge weight $w$, direction $d$\n"
        r"train\_pos\_pearson",
        fc="white",
        ec=C["input"],
        fontsize=7.3,
    )
    _box(
        ax,
        (0.032, y0 + 0.08),
        0.146,
        0.12,
        "scGPT embeddings\n$E \\in \\mathbb{R}^{G \\times 768}$\nscgpt\\_gene\\_emb.pt",
        fc="white",
        ec=C["input"],
        fontsize=7.6,
    )

    # ── B. Gene encoder ────────────────────────────────────────────────────
    _section_frame(ax, (0.205, y0), 0.195, ph, "Dual gene encoder", C["encoder"], C["encoder_bg"], "B")
    _box(
        ax,
        (0.218, y0 + 0.56),
        0.082,
        0.15,
        "Expression branch\nLinear($C{\\rightarrow}768$)\n+ Laplacian PE\nTransformerEnc\n$L{=}4$, $H{=}8$",
        fc="white",
        ec=C["encoder"],
        fontsize=7.0,
    )
    _box(
        ax,
        (0.312, y0 + 0.56),
        0.075,
        0.15,
        "scGPT branch\nLinear($768{\\rightarrow}768$)\nfrozen $E_g$",
        fc="white",
        ec=C["encoder"],
        fontsize=7.0,
    )
    _box(
        ax,
        (0.248, y0 + 0.38),
        0.108,
        0.09,
        "Fusion\nConcat $\\rightarrow$ Linear\n$h_g \\in \\mathbb{R}^{768}$",
        fc="white",
        ec=C["encoder"],
        fontsize=7.6,
        fontweight="bold",
    )
    _box(
        ax,
        (0.218, y0 + 0.08),
        0.17,
        0.24,
        "Graph positional encoding\nLapPE on $\\mathcal{G}_0$ ($k{=}768$)\n\n"
        "Graph Transformer input\nLinear($768{\\rightarrow}128$)\n$z_g^{(0)} \\in \\mathbb{R}^{128}$",
        fc="white",
        ec=C["encoder"],
        fontsize=7.3,
    )
    _arrow(ax, (0.259, y0 + 0.56), (0.285, y0 + 0.47), color=C["encoder"])
    _arrow(ax, (0.349, y0 + 0.56), (0.325, y0 + 0.47), color=C["encoder"])
    _arrow(ax, (0.302, y0 + 0.38), (0.302, y0 + 0.32), color=C["encoder"])

    # ── C. Graph Transformer ───────────────────────────────────────────────
    _section_frame(ax, (0.415, y0), 0.205, ph, "scGT-Link core", C["gt"], C["gt_bg"], "C")
    _box(
        ax,
        (0.428, y0 + 0.66),
        0.18,
        0.08,
        "Node features $z^{(0)} \\in \\mathbb{R}^{128}$, edges $(w,d)$",
        fc="white",
        ec=C["gt"],
        fontsize=7.6,
    )
    # Single detailed layer + repeat indicator (cleaner for print)
    _box(
        ax,
        (0.428, y0 + 0.28),
        0.18,
        0.34,
        "GT layer  ($L = 2$ stacked)\n\n"
        "1.  Sparse multi-head attention ($H{=}4$)\n"
        "     messages only on graph edges\n"
        "     attn $\\leftarrow$ learnable $\\alpha w + \\beta d$\n\n"
        "2.  Residual + BatchNorm\n\n"
        "3.  FFN: 128 $\\rightarrow$ 256 $\\rightarrow$ 128\n\n"
        "4.  Residual + BatchNorm",
        fc="white",
        ec=C["gt"],
        fontsize=7.0,
        ha="left",
        va="top",
        pad_x=0.014,
        pad_y=0.014,
    )
    _box(
        ax,
        (0.428, y0 + 0.08),
        0.18,
        0.08,
        "Output $z \\in \\mathbb{R}^{128}$ per gene",
        fc="white",
        ec=C["gt"],
        fontsize=7.8,
        fontweight="bold",
    )
    _arrow(ax, (0.518, y0 + 0.66), (0.518, y0 + 0.62), color=C["gt"], lw=0.9, mutation=8)
    _arrow(ax, (0.518, y0 + 0.28), (0.518, y0 + 0.16), color=C["gt"], lw=0.9, mutation=8)

    # ── D. Link prediction ─────────────────────────────────────────────────
    _section_frame(ax, (0.635, y0), 0.145, ph, "Link predictor", C["head"], C["head_bg"], "D")
    _box(
        ax,
        (0.648, y0 + 0.54),
        0.118,
        0.1,
        "Candidate edge\n(TF $i$, target $j$)",
        fc="white",
        ec=C["head"],
        fontsize=7.8,
    )
    _box(
        ax,
        (0.648, y0 + 0.46),
        0.118,
        0.1,
        "Concat $[z_i; z_j]$\n$\\in \\mathbb{R}^{256}$",
        fc="white",
        ec=C["head"],
        fontsize=7.8,
    )
    _box(
        ax,
        (0.648, y0 + 0.28),
        0.118,
        0.12,
        "MLP decoder\n256 $\\rightarrow$ 128 $\\rightarrow$ 1\nReLU + Dropout",
        fc="white",
        ec=C["head"],
        fontsize=7.6,
    )
    _box(
        ax,
        (0.648, y0 + 0.08),
        0.118,
        0.1,
        "Logit $\\hat{y}_{ij}$\nedge probability",
        fc="white",
        ec=C["head"],
        fontsize=7.8,
        fontweight="bold",
    )
    _arrow(ax, (0.707, y0 + 0.54), (0.707, y0 + 0.48), color=C["head"])
    _arrow(ax, (0.707, y0 + 0.46), (0.707, y0 + 0.40), color=C["head"])
    _arrow(ax, (0.707, y0 + 0.28), (0.707, y0 + 0.18), color=C["head"])

    # ── E. Training ────────────────────────────────────────────────────────
    _section_frame(ax, (0.795, y0), 0.185, ph, "Training", C["train"], C["train_bg"], "E")
    _box(
        ax,
        (0.808, y0 + 0.58),
        0.158,
        0.14,
        "Supervised pairs\nTrain\\_set.csv\npositive / negative\nTF–target edges",
        fc="white",
        ec=C["train"],
        fontsize=7.6,
    )
    _box(
        ax,
        (0.808, y0 + 0.36),
        0.158,
        0.14,
        "Loss\nBCEWithLogitsLoss\n+ pos\\_weight",
        fc="white",
        ec=C["train"],
        fontsize=7.8,
    )
    _box(
        ax,
        (0.808, y0 + 0.08),
        0.158,
        0.2,
        "Evaluation\nAUROC / AUPRC\nvalidation & test",
        fc="white",
        ec=C["train"],
        fontsize=8,
        fontweight="bold",
    )

    # Inter-panel flow: one arrow per adjacent module (no input-specific cross-links)
    mid_y = y0 + ph / 2
    panel_edges = (
        (0.188, 0.205),  # A → B
        (0.40, 0.415),   # B → C
        (0.62, 0.635),   # C → D
    )
    for x0, x1 in panel_edges:
        _arrow(ax, (x0, mid_y), (x1, mid_y), lw=1.5)

    # Bottom summary
    summary = FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.075,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor="#f8f8f8",
        edgecolor=C["line"],
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(summary)
    ax.text(
        0.04,
        0.067,
        "Pipeline:  train-positive prior + Pearson weights  →  scGPT encoder + LapPE (v2; no expr branch)  →  "
        "2-layer edge-aware Graph Transformer  →  MLP link head  →  weighted BCE.  "
        "Default (scGT-Link v2): scgpt + train_pos_pearson.",
        ha="left",
        va="center",
        fontsize=7.6,
        color=C["muted"],
        zorder=2,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "figures/fig_model_overview"),
        help="Output path stem (writes .pdf, .png, .svg).",
    )
    p.add_argument(
        "--journal",
        type=str,
        default="nature",
        choices=("nature", "wide"),
        help="Figure width preset (scipilot-style final size, no post-scale).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output).expanduser().resolve()
    plot_overview(out, journal=args.journal)
    print(f"Wrote {out.with_suffix('.pdf')}")
    print(f"Wrote {out.with_suffix('.png')}")
    print(f"Wrote {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
