#!/usr/bin/env python3
"""Precompute scGPT gene embeddings for genes in BL--ExpressionData.csv (row order) and save a .pt for training.

Example:
  python scripts/precompute_scgpt_gene_emb.py \\
    --data_dir data/hESC/TFs+500 \\
    --scgpt_model_dir /path/to/scGPT_human \\
    --output data/hESC/TFs+500/scgpt_gene_emb.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from grn_data import load_expression_for_grn
from scgpt_loader import load_scgpt_gene_embedding_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder with BL--ExpressionData.csv")
    p.add_argument("--scgpt_model_dir", type=str, required=True, help="scGPT checkpoint directory")
    p.add_argument("--output", type=str, required=True, help="Output .pt path")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    exp_path = data_dir / "BL--ExpressionData.csv"
    if not exp_path.is_file():
        raise SystemExit(f"Missing {exp_path}")

    _, gene_names, _ = load_expression_for_grn(exp_path)
    mat, dim = load_scgpt_gene_embedding_matrix(
        Path(args.scgpt_model_dir),
        gene_names,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    out = {
        "embeddings": mat,
        "gene_names": gene_names,
        "scgpt_dim": dim,
        "scgpt_model_dir": str(Path(args.scgpt_model_dir).resolve()),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved {mat.shape[0]} x {mat.shape[1]} to {out_path}")


if __name__ == "__main__":
    main()
