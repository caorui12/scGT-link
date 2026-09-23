"""Load expression matrix and BEELINE-style edge splits."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def load_expression(csv_path: Path) -> Tuple[torch.Tensor, list[str], int]:
    """genes × cells CSV → float tensor (G, C), log1p + per-gene z-score."""
    df = pd.read_csv(csv_path, index_col=0)
    gene_names = df.index.astype(str).tolist()
    x = df.values.astype(np.float64)
    logn = np.log1p(np.maximum(x, 0.0))
    mean = logn.mean(axis=1, keepdims=True)
    std = logn.std(axis=1, keepdims=True) + 1e-8
    z = np.nan_to_num((logn - mean) / std, nan=0.0).astype(np.float32)
    return torch.from_numpy(z), gene_names, int(z.shape[1])


def load_edge_split(split_dir: Path, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return edge_index (2, E) long and labels (E,) float {0,1}."""
    path = split_dir / f"{name}.csv"
    df = pd.read_csv(path, index_col=0)
    tf = df["TF"].values.astype(np.int64)
    tgt = df["Target"].values.astype(np.int64)
    lab = df["Label"].values.astype(np.float32)
    ei = np.stack([tf, tgt], axis=0)
    return torch.from_numpy(ei), torch.from_numpy(lab)


def load_scgpt_emb_pt(path: Path, gene_names: list[str]) -> Tuple[torch.Tensor, int]:
    """Load precomputed scGPT gene embeddings. Rows must match expression gene order."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        emb = obj["embeddings"]
        stored = obj.get("gene_names")
        dim = int(obj.get("scgpt_dim", emb.shape[1]))
    else:
        emb = obj
        stored = None
        dim = int(emb.shape[1])
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    if emb.dim() != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(emb.shape)}")
    if emb.shape[0] != len(gene_names):
        raise ValueError(
            f"Embedding rows {emb.shape[0]} != len(gene_names) {len(gene_names)}"
        )
    if stored is not None and list(stored) != list(gene_names):
        raise ValueError("gene_names in embedding file do not match expression matrix order")
    return emb.float(), dim
