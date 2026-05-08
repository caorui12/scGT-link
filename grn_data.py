"""Load hESC (or any TFs+N) expression matrix + hESC_N edge splits for GRN training."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def load_expression_for_grn(csv_path: Path) -> Tuple[torch.Tensor, list[str], int]:
    """
    genes × cells CSV → float tensor (n_genes, n_cells), log1p + z-score per gene (across cells).
    """
    df = pd.read_csv(csv_path, index_col=0)
    gene_names = df.index.tolist()
    x = df.values.astype(np.float64)
    logn = np.log1p(np.maximum(x, 0))
    mean = logn.mean(axis=1, keepdims=True)
    std = logn.std(axis=1, keepdims=True) + 1e-8
    z = (logn - mean) / std
    z = np.nan_to_num(z, nan=0.0).astype(np.float32)
    return torch.from_numpy(z), gene_names, z.shape[1]


def load_edge_split(split_dir: Path, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns edge_index (2, E) long, labels (E,) float 0/1."""
    path = split_dir / f"{name}.csv"
    df = pd.read_csv(path, index_col=0)
    tf = df["TF"].values.astype(np.int64)
    tgt = df["Target"].values.astype(np.int64)
    lab = df["Label"].values.astype(np.float32)
    ei = np.stack([tf, tgt], axis=0)
    return torch.from_numpy(ei), torch.from_numpy(lab)
