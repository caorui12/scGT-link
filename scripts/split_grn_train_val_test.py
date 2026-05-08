#!/usr/bin/env python3
"""
Build ``Train_set.csv`` / ``Validation_set.csv`` / ``Test_set.csv`` in the same
format as ``data/Train_validation_test/<name>/`` (columns TF, Target, Label;
default pandas index 0,1,… — compatible with :func:`grn_data.load_edge_split`).

Two splitting protocols (aligned with GENELink / scGREAT ``pre-processing``):

* ``hard_negative`` — ``Hard_Negative_Specific_train_test_val``: per-TF positives split
  by ``ratio`` / ``p_val``; negatives are **hard** (genes not in the gold positive
  set for that TF). Use for **Specific** / cell-type-specific ChIP-style ``Label.csv``.

* ``density_negative`` — ``train_val_test_set``: random negatives for train/val;
  test negatives sampled globally with count ``int(count // density - count)``
  (same expression as reference code). Pass ``--net_type`` + species/genes to pick
  density from the GENELink table, or ``--density`` to override.

* ``paper_bbad414`` — Mao et al. GNNLink (Brief Bioinform 2023) prose: **all** gold edges are
  shuffled and split **3/5 train · 1/5 validation · 1/5 test** positives; train and validation
  use **one hard negative per positive** (same TF, uniform random non-edge target); test
  negatives are sampled globally so positive fraction matches **Table 1 density** (same
  ``int(count // density - count)`` rule). Use ``--benchmark_all --mode paper_bbad414``
  with ``--benchmark_out_subdir`` (default below) so original GENELink splits stay intact.

Inputs (BEELINE-style under ``.../TFs+500/``):

* ``Label.csv`` — columns ``TF``, ``Target`` (integer gene indices; positive edges only).
* ``TF.csv`` — column ``index`` (TF row indices).
* ``Target.csv`` — column ``index`` (all gene indices in the panel).

Example::

  python scripts/split_grn_train_val_test.py \\
    --label data/hESC/TFs+500/Label.csv \\
    --tf data/hESC/TFs+500/TF.csv \\
    --target data/hESC/TFs+500/Target.csv \\
    --out_dir data/Train_validation_test/hESC_500 \\
    --seed 42 \\
    --mode hard_negative

  python scripts/split_grn_train_val_test.py \\
    --preset hESC_500 \\
    --mode hard_negative \\
    --seed 42

  # All GENELink-style folders under ``data/Dataset/Benchmark Dataset``:
  python scripts/split_grn_train_val_test.py \\
    --benchmark_all \\
    --repo_root . \\
    --seed 42

  # GNNLink paper (bbad414) split tree-wide into ``Train_validation_test_gnnlink_paper/``:
  python scripts/split_grn_train_val_test.py \\
    --benchmark_all \\
    --mode paper_bbad414 \\
    --benchmark_root \"/path/to/Benchmark Dataset\" \\
    --repo_root . \\
    --seed 42

  # ``--benchmark_all`` uses ``--mode`` for **every** leaf (hard_negative / density_negative /
  # paper_bbad414), not only per-folder auto mode.

  Writes ``Train_validation_test/{Train,Validation,Test}_set.csv`` next to each
  ``Label.csv`` (e.g. ``.../Specific Dataset/hESC/TFs+500/Train_validation_test/``).
  With ``--mode paper_bbad414``, the default output folder name is
  ``Train_validation_test_gnnlink_paper`` unless ``--benchmark_out_subdir`` is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Matches GENELink ``utils.Network_Statistic`` (Brief Bioinform benchmark table).
_DENSITY_STRING = {
    "hESC500": 0.024,
    "hESC1000": 0.021,
    "hHEP500": 0.028,
    "hHEP1000": 0.024,
    "mDC500": 0.038,
    "mDC1000": 0.032,
    "mESC500": 0.024,
    "mESC1000": 0.021,
    "mHSC-E500": 0.029,
    "mHSC-E1000": 0.027,
    "mHSC-GM500": 0.040,
    "mHSC-GM1000": 0.037,
    "mHSC-L500": 0.048,
    "mHSC-L1000": 0.045,
}
_DENSITY_NONSPECIFIC = {
    "hESC500": 0.016,
    "hESC1000": 0.014,
    "hHEP500": 0.015,
    "hHEP1000": 0.013,
    "mDC500": 0.019,
    "mDC1000": 0.016,
    "mESC500": 0.015,
    "mESC1000": 0.013,
    "mHSC-E500": 0.022,
    "mHSC-E1000": 0.020,
    "mHSC-GM500": 0.030,
    "mHSC-GM1000": 0.029,
    "mHSC-L500": 0.048,
    "mHSC-L1000": 0.043,
}
_DENSITY_SPECIFIC = {
    "hESC500": 0.164,
    "hESC1000": 0.165,
    "hHEP500": 0.379,
    "hHEP1000": 0.377,
    "mDC500": 0.085,
    "mDC1000": 0.082,
    "mESC500": 0.345,
    "mESC1000": 0.347,
    "mHSC-E500": 0.578,
    "mHSC-E1000": 0.566,
    "mHSC-GM500": 0.543,
    "mHSC-GM1000": 0.565,
    "mHSC-L500": 0.525,
    "mHSC-L1000": 0.507,
}
_DENSITY_LOFGOF = {"mESC500": 0.158, "mESC1000": 0.154}

_PRESET_TABLE: dict[str, tuple[str, str]] = {
    "hESC_500": ("data/hESC/TFs+500", "data/Train_validation_test/hESC_500"),
    "hESC_1000": ("data/hESC/TFs+1000", "data/Train_validation_test/hESC_1000"),
    "mESC_500": ("data/mESC/TFs+500", "data/Train_validation_test/mESC_500"),
    "mESC_1000": ("data/mESC/TFs+1000", "data/Train_validation_test/mESC_1000"),
}


def _resolve_density(net_type: str, species: str, n_genes: int) -> float:
    key = f"{species}{n_genes}"
    nt = net_type.strip().lower().replace("_", "-")
    if nt in ("string",):
        d = _DENSITY_STRING
    elif nt in ("non-specific", "nonspecific", "non_specific"):
        d = _DENSITY_NONSPECIFIC
    elif nt in ("specific",):
        d = _DENSITY_SPECIFIC
    elif nt in ("lofgof", "lof-gof", "lof_gof"):
        d = _DENSITY_LOFGOF
    else:
        raise SystemExit(
            f"Unknown --net_type {net_type!r}; use STRING, Non-Specific, Specific, Lofgof"
        )
    if key not in d:
        raise SystemExit(f"No density entry for {key!r} under net_type={net_type!r}")
    return float(d[key])


def _write_split_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["TF"] = out["TF"].astype(np.int64)
    out["Target"] = out["Target"].astype(np.int64)
    out["Label"] = out["Label"].astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)


def hard_negative_split(
    label: pd.DataFrame,
    gene_set: np.ndarray,
    tf_set: np.ndarray,
    ratio: float,
    p_val: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """GENELink / scGREAT ``Hard_Negative_Specific_train_test_val``."""
    tf = label["TF"].values
    tf_list = np.unique(tf)
    pos_dict: dict = {i: [] for i in tf_list}
    for i, j in label[["TF", "Target"]].values:
        pos_dict[i].append(j)

    neg_dict: dict = {i: [] for i in tf_set}
    for i in tf_set:
        if i in pos_dict:
            pos_item = list(pos_dict[i])
            pos_item.append(i)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i, assume_unique=False)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item.tolist())
        else:
            neg_item = np.setdiff1d(gene_set, [i])
            neg_dict[i].extend(neg_item.tolist())

    train_pos: dict = {}
    val_pos: dict = {}
    test_pos: dict = {}
    for k in pos_dict:
        if len(pos_dict[k]) == 1:
            if rng.uniform() <= p_val:
                train_pos[k] = list(pos_dict[k])
            else:
                test_pos[k] = list(pos_dict[k])
        elif len(pos_dict[k]) == 2:
            pair = list(pos_dict[k])
            rng.shuffle(pair)
            train_pos[k] = [pair[0]]
            test_pos[k] = [pair[1]]
        else:
            arr = np.array(pos_dict[k], dtype=np.int64)
            rng.shuffle(arr)
            n = len(arr)
            train_pos[k] = arr[: int(n * ratio)].tolist()
            val_pos[k] = arr[int(n * ratio) : int(n * (ratio + 0.1))].tolist()
            test_pos[k] = arr[int(n * (ratio + 0.1)) :].tolist()

    train_neg: dict = {}
    val_neg: dict = {}
    test_neg: dict = {}
    for k in pos_dict:
        neg_list = list(neg_dict[k])
        rng.shuffle(neg_list)
        neg_num = len(neg_list)
        train_neg[k] = neg_list[: int(neg_num * ratio)]
        val_neg[k] = neg_list[int(neg_num * ratio) : int(neg_num * (0.1 + ratio))]
        test_neg[k] = neg_list[int(neg_num * (0.1 + ratio)) :]

    def _pairs_from_dict(d: dict) -> list[list[int]]:
        rows: list[list[int]] = []
        for k, js in d.items():
            for j in js:
                rows.append([int(k), int(j)])
        return rows

    train_pos_set = _pairs_from_dict(train_pos)
    train_neg_set = _pairs_from_dict(train_neg)
    train_rows = train_pos_set + train_neg_set
    train_lab = [1] * len(train_pos_set) + [0] * len(train_neg_set)
    train_df = pd.DataFrame(train_rows, columns=["TF", "Target"])
    train_df["Label"] = train_lab

    val_pos_set = _pairs_from_dict(val_pos)
    val_neg_set = _pairs_from_dict(val_neg)
    val_rows = val_pos_set + val_neg_set
    val_lab = [1] * len(val_pos_set) + [0] * len(val_neg_set)
    val_df = pd.DataFrame(val_rows, columns=["TF", "Target"])
    val_df["Label"] = val_lab

    test_pos_set = _pairs_from_dict(test_pos)
    test_neg_set = _pairs_from_dict(test_neg)
    test_rows = test_pos_set + test_neg_set
    test_lab = [1] * len(test_pos_set) + [0] * len(test_neg_set)
    test_df = pd.DataFrame(test_rows, columns=["TF", "Target"])
    test_df["Label"] = test_lab

    return train_df, val_df, test_df


def density_negative_split(
    label: pd.DataFrame,
    gene_set: np.ndarray,
    tf_set: np.ndarray,
    density: float,
    p_val: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """GENELink / scGREAT ``train_val_test_set`` (random train/val negs)."""
    tf_list = np.unique(label["TF"].values)
    pos_dict: dict = {i: [] for i in tf_list}
    for i, j in label[["TF", "Target"]].values:
        pos_dict[i].append(j)

    train_pos: dict = {}
    val_pos: dict = {}
    test_pos: dict = {}
    for k in pos_dict:
        if len(pos_dict[k]) <= 1:
            if rng.uniform() <= p_val:
                train_pos[k] = list(pos_dict[k])
            else:
                test_pos[k] = list(pos_dict[k])
        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            arr = list(pos_dict[k])
            rng.shuffle(arr)
            n = len(arr)
            train_pos[k] = arr[: n * 2 // 3]
            test_pos[k] = arr[n * 2 // 3 :]
            val_pos[k] = train_pos[k][: len(train_pos[k]) // 5]
            train_pos[k] = train_pos[k][len(train_pos[k]) // 5 :]

    train_neg: dict = {}
    for k in train_pos:
        train_neg[k] = []
        for _ in range(len(train_pos[k])):
            neg = int(rng.choice(gene_set))
            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = int(rng.choice(gene_set))
            train_neg[k].append(neg)

    train_pos_set: list[list[int]] = []
    train_neg_set: list[list[int]] = []
    for k, js in train_pos.items():
        for j in js:
            train_pos_set.append([k, j])
    for k, js in train_neg.items():
        for j in js:
            train_neg_set.append([k, j])

    train_set_pairs = train_pos_set + train_neg_set
    train_label = [1] * len(train_pos_set) + [0] * len(train_neg_set)
    train_df = pd.DataFrame(
        [row + [train_label[i]] for i, row in enumerate(train_set_pairs)],
        columns=["TF", "Target", "Label"],
    )

    val_pos_set: list[list[int]] = []
    for k, js in val_pos.items():
        for j in js:
            val_pos_set.append([k, j])

    val_neg: dict = {}
    for k in val_pos:
        val_neg[k] = []
        for _ in range(len(val_pos[k])):
            neg = int(rng.choice(gene_set))
            while (
                neg == k
                or neg in pos_dict[k]
                or neg in train_neg[k]
                or neg in val_neg[k]
            ):
                neg = int(rng.choice(gene_set))
            val_neg[k].append(neg)

    val_neg_set: list[list[int]] = []
    for k, js in val_neg.items():
        for j in js:
            val_neg_set.append([k, j])

    val_rows = val_pos_set + val_neg_set
    val_lab = [1] * len(val_pos_set) + [0] * len(val_neg_set)
    val_df = pd.DataFrame(
        {"TF": [r[0] for r in val_rows], "Target": [r[1] for r in val_rows], "Label": val_lab}
    )

    test_pos_set: list[list[int]] = []
    for k, js in test_pos.items():
        for j in js:
            test_pos_set.append([k, j])

    count = sum(len(js) for js in test_pos.values())
    # Same formula as reference: ``int(count // density - count)``
    test_neg_num = int(count // float(density) - count)
    test_neg_num = max(0, test_neg_num)

    used: set[tuple[int, int]] = set()
    for a, b in train_set_pairs:
        used.add((int(a), int(b)))
    for a, b in val_rows:
        used.add((int(a), int(b)))
    for a, b in test_pos_set:
        used.add((int(a), int(b)))

    test_neg_set: list[list[int]] = []
    seen_neg: set[tuple[int, int]] = set()
    batch = max(4096, min(65536, test_neg_num // 4 + 1))
    guard = 0
    while len(test_neg_set) < test_neg_num and guard < max(200, test_neg_num // 1000 + 50):
        guard += 1
        need = test_neg_num - len(test_neg_set)
        t1s = rng.choice(tf_set, size=min(batch, need * 2))
        t2s = rng.choice(gene_set, size=min(batch, need * 2))
        for t1, t2 in zip(t1s.astype(np.int64), t2s.astype(np.int64)):
            if t1 == t2:
                continue
            pair = (int(t1), int(t2))
            if pair in used or pair in seen_neg:
                continue
            seen_neg.add(pair)
            used.add(pair)
            test_neg_set.append([pair[0], pair[1]])
            if len(test_neg_set) >= test_neg_num:
                break
    if len(test_neg_set) < test_neg_num:
        raise RuntimeError(
            f"Could not sample {test_neg_num} test negatives (got {len(test_neg_set)}); "
            "graph may be too dense."
        )

    test_rows = test_pos_set + test_neg_set
    test_lab = [1] * len(test_pos_set) + [0] * len(test_neg_set)
    test_df = pd.DataFrame(
        [row + [test_lab[i]] for i, row in enumerate(test_rows)],
        columns=["TF", "Target", "Label"],
    )

    return train_df, val_df, test_df


def paper_bbad414_split(
    label: pd.DataFrame,
    gene_set: np.ndarray,
    tf_set: np.ndarray,
    density: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    GNNLink paper (bbad414): 3/5 · 1/5 · 1/5 positive split over **all** directed gold edges;
    train/val: 1 hard negative per positive (same TF); test: density-aligned global negatives.

    Hard negatives prefer unused directed pairs when many positives share one TF; if the
    non-gold panel is exhausted, sampling falls back to uniform over **any** non-gold target
    (may repeat (TF, Target) rows — needed for dense Specific ChIP benchmarks).
    """
    all_pos: list[list[int]] = []
    gold_pairs: set[tuple[int, int]] = set()
    for i, j in label[["TF", "Target"]].values.astype(np.int64):
        k, t = int(i), int(j)
        all_pos.append([k, t])
        gold_pairs.add((k, t))

    n = len(all_pos)
    if n < 3:
        raise RuntimeError(
            f"paper_bbad414 needs at least 3 gold edges for a 3:1:1 split, got {n}"
        )

    rng.shuffle(all_pos)
    n_train = (n * 3) // 5
    n_val = n // 5
    n_test = n - n_train - n_val
    train_pos_pairs = all_pos[:n_train]
    val_pos_pairs = all_pos[n_train : n_train + n_val]
    test_pos_pairs = all_pos[n_train + n_val :]

    genes = np.asarray(gene_set, dtype=np.int64)
    tfs_arr = np.asarray(tf_set, dtype=np.int64)

    gold_by_tf: dict[int, set[int]] = {}
    for k, t in gold_pairs:
        kk = int(k)
        gold_by_tf.setdefault(kk, set()).add(int(t))

    tabu: dict[int, set[int]] = {}
    for kk in np.unique(tfs_arr.astype(np.int64)):
        kki = int(kk)
        tabu[kki] = {kki} | set(gold_by_tf.get(kki, ()))

    def sample_hard_neg(tf_key: int) -> int:
        kk = int(tf_key)
        avoid = np.array(list(tabu[kk]), dtype=np.int64)
        cand = np.setdiff1d(genes, avoid, assume_unique=False)
        if len(cand) > 0:
            g = int(rng.choice(cand))
            tabu[kk].add(g)
            return g
        gset = gold_by_tf.get(kk, set()) | {kk}
        loose = np.setdiff1d(
            genes, np.array(list(gset), dtype=np.int64), assume_unique=False
        )
        if len(loose) == 0:
            raise RuntimeError(
                f"paper_bbad414: TF={kk} regulates every gene in the panel "
                "(no non-edge target exists)."
            )
        return int(rng.choice(loose))

    train_rows: list[list[int]] = []
    train_lab: list[int] = []
    for k, j in train_pos_pairs:
        train_rows.append([k, j])
        train_lab.append(1)
        g = sample_hard_neg(k)
        train_rows.append([k, g])
        train_lab.append(0)

    val_rows: list[list[int]] = []
    val_lab: list[int] = []
    for k, j in val_pos_pairs:
        val_rows.append([k, j])
        val_lab.append(1)
        g = sample_hard_neg(k)
        val_rows.append([k, g])
        val_lab.append(0)

    test_pos_set = [[int(a), int(b)] for a, b in test_pos_pairs]
    count = len(test_pos_set)
    test_neg_num = int(count // float(density) - count)
    test_neg_num = max(0, test_neg_num)

    used: set[tuple[int, int]] = set(gold_pairs)
    used.update((int(r[0]), int(r[1])) for r in train_rows)
    used.update((int(r[0]), int(r[1])) for r in val_rows)
    for a, b in test_pos_set:
        used.add((int(a), int(b)))

    test_neg_set: list[list[int]] = []
    if test_neg_num > 0:
        seen_neg: set[tuple[int, int]] = set()
        batch = max(4096, min(65536, test_neg_num // 4 + 1))
        guard = 0
        while len(test_neg_set) < test_neg_num and guard < max(200, test_neg_num // 1000 + 50):
            guard += 1
            need = test_neg_num - len(test_neg_set)
            t1s = rng.choice(tfs_arr, size=min(batch, max(need * 2, 1024)))
            t2s = rng.choice(genes, size=min(batch, max(need * 2, 1024)))
            for t1, t2 in zip(t1s.astype(np.int64), t2s.astype(np.int64)):
                if int(t1) == int(t2):
                    continue
                pair = (int(t1), int(t2))
                if pair in used or pair in seen_neg:
                    continue
                seen_neg.add(pair)
                used.add(pair)
                test_neg_set.append([pair[0], pair[1]])
                if len(test_neg_set) >= test_neg_num:
                    break
        if len(test_neg_set) < test_neg_num:
            raise RuntimeError(
                f"paper_bbad414: need {test_neg_num} test negatives, got {len(test_neg_set)} "
                "(density too high or graph too dense)."
            )

    test_rows = test_pos_set + test_neg_set
    test_lab = [1] * len(test_pos_set) + [0] * len(test_neg_set)

    train_df = pd.DataFrame(
        {"TF": [r[0] for r in train_rows], "Target": [r[1] for r in train_rows], "Label": train_lab}
    )
    val_df = pd.DataFrame(
        {"TF": [r[0] for r in val_rows], "Target": [r[1] for r in val_rows], "Label": val_lab}
    )
    test_df = pd.DataFrame(
        {"TF": [r[0] for r in test_rows], "Target": [r[1] for r in test_rows], "Label": test_lab}
    )
    return train_df, val_df, test_df


def _benchmark_folder_to_mode_net(net_folder: str) -> tuple[str, str]:
    """Return (mode, net_type_arg for _resolve_density)."""
    key = net_folder.strip()
    mapping: dict[str, tuple[str, str]] = {
        "Specific Dataset": ("hard_negative", "Specific"),
        "STRING Dataset": ("density_negative", "STRING"),
        "Non-Specific Dataset": ("density_negative", "Non-Specific"),
        "Lofgof Dataset": ("density_negative", "Lofgof"),
    }
    if key in mapping:
        return mapping[key]
    raise ValueError(
        f"Unknown benchmark top folder: {net_folder!r}; expected one of {list(mapping)}"
    )


def run_split(
    label_p: Path,
    tf_p: Path,
    target_p: Path,
    out_dir: Path,
    mode: str,
    seed: int,
    ratio: float,
    p_val: float,
    net_type: str = "Specific",
    species: str = "hESC",
    n_genes: int = 500,
    density_override: float = -1.0,
    quiet: bool = False,
) -> tuple[int, int, int]:
    """
    Write Train_set.csv, Validation_set.csv, Test_set.csv under ``out_dir``.
    Returns (n_train, n_val, n_test).
    """
    for path, name in ((label_p, "label"), (tf_p, "tf"), (target_p, "target")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} file: {path}")

    label = pd.read_csv(label_p, index_col=0)
    if "TF" not in label.columns or "Target" not in label.columns:
        raise ValueError(f"{label_p} must have columns TF, Target")

    gene_set = pd.read_csv(target_p, index_col=0)["index"].values
    tf_set = pd.read_csv(tf_p, index_col=0)["index"].values
    rng = np.random.default_rng(seed)

    if mode == "hard_negative":
        train_df, val_df, test_df = hard_negative_split(
            label, gene_set, tf_set, ratio, p_val, rng
        )
    elif mode == "paper_bbad414":
        density = (
            float(density_override)
            if density_override > 0
            else _resolve_density(net_type, species, n_genes)
        )
        train_df, val_df, test_df = paper_bbad414_split(
            label, gene_set, tf_set, density, rng
        )
    else:
        density = (
            float(density_override)
            if density_override > 0
            else _resolve_density(net_type, species, n_genes)
        )
        train_df, val_df, test_df = density_negative_split(
            label, gene_set, tf_set, density, p_val, rng
        )

    _write_split_csv(train_df, out_dir / "Train_set.csv")
    _write_split_csv(val_df, out_dir / "Validation_set.csv")
    _write_split_csv(test_df, out_dir / "Test_set.csv")

    if not quiet:
        print(f"Wrote Train / Validation / Test under {out_dir}")
        print(
            f"  rows: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"| mode={mode} seed={seed}"
        )
    return len(train_df), len(val_df), len(test_df)


def discover_benchmark_tasks(
    benchmark_root: Path, out_subdir: str = "Train_validation_test"
) -> list[tuple[Path, Path, Path, Path, str, str, str, int]]:
    """
    Each item: (label, tf, target, out_dir, mode, net_type, species, n_genes).
    ``out_dir`` = data_dir / ``out_subdir``.
    """
    tasks: list[tuple[Path, Path, Path, Path, str, str, str, int]] = []
    if not benchmark_root.is_dir():
        raise FileNotFoundError(benchmark_root)

    for label_p in sorted(benchmark_root.rglob("Label.csv")):
        data_dir = label_p.parent
        tf_p = data_dir / "TF.csv"
        target_p = data_dir / "Target.csv"
        if not tf_p.is_file() or not target_p.is_file():
            continue
        try:
            rel = data_dir.relative_to(benchmark_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        net_folder, species, tfs_folder = parts[0], parts[1], parts[2]
        if not tfs_folder.startswith("TFs+"):
            continue
        try:
            n_genes = int(tfs_folder.split("+", 1)[1])
        except (IndexError, ValueError):
            continue
        mode, net_type = _benchmark_folder_to_mode_net(net_folder)
        out_dir = data_dir / out_subdir
        tasks.append((label_p, tf_p, target_p, out_dir, mode, net_type, species, n_genes))
    return tasks


def run_benchmark_all(
    benchmark_root: Path,
    seed: int,
    ratio: float,
    p_val: float,
    *,
    out_subdir: str = "Train_validation_test",
    split_mode_override: str = "",
) -> None:
    tasks = discover_benchmark_tasks(benchmark_root, out_subdir=out_subdir)
    if not tasks:
        raise SystemExit(f"No Label.csv tasks under {benchmark_root}")
    ok = 0
    errors: list[str] = []
    for label_p, tf_p, target_p, out_dir, mode, net_type, species, n_genes in tasks:
        rel = label_p.parent.relative_to(benchmark_root)
        mode_use = split_mode_override if split_mode_override else mode
        try:
            run_split(
                label_p,
                tf_p,
                target_p,
                out_dir,
                mode=mode_use,
                seed=seed,
                ratio=ratio,
                p_val=p_val,
                net_type=net_type,
                species=species,
                n_genes=n_genes,
                quiet=True,
            )
            ok += 1
            print(f"OK {rel} -> {out_dir.name}/ ({mode_use})")
        except Exception as e:
            errors.append(f"{rel}: {e}")
            print(f"FAIL {rel}: {e}", file=sys.stderr)
    print(f"\nDone: {ok}/{len(tasks)} succeeded.")
    if errors:
        print(f"Failures ({len(errors)}):", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write Train/Validation/Test edge CSVs for train_grn_graph.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--benchmark_all",
        action="store_true",
        help="Process every Label.csv under --benchmark_root (GENELink-style tree).",
    )
    p.add_argument(
        "--benchmark_root",
        type=str,
        default="data/Dataset/Benchmark Dataset",
        help="Root folder containing 'Specific Dataset', 'STRING Dataset', etc.",
    )
    p.add_argument(
        "--benchmark_out_subdir",
        type=str,
        default="",
        help="Per-task output folder name under each TFs+*/ (default: "
        "Train_validation_test_gnnlink_paper if --mode paper_bbad414 else Train_validation_test)",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="",
        choices=list(_PRESET_TABLE.keys()) + [""],
        help=f"Use bundled paths: {list(_PRESET_TABLE.keys())}",
    )
    p.add_argument("--label", type=str, default="", help="Label.csv (TF, Target positives)")
    p.add_argument("--tf", type=str, default="", help="TF.csv with index column")
    p.add_argument("--target", type=str, default="", help="Target.csv with index column")
    p.add_argument("--out_dir", type=str, default="", help="Output folder for three CSVs")
    p.add_argument("--repo_root", type=str, default=str(_REPO_ROOT), help="Root for preset paths")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        type=str,
        choices=("hard_negative", "density_negative", "paper_bbad414"),
        default="hard_negative",
    )
    p.add_argument("--ratio", type=float, default=0.67, help="hard_negative: train fraction of positives")
    p.add_argument(
        "--p_val",
        type=float,
        default=0.5,
        help="For TF with <=1 positive: prob train vs test (both modes)",
    )
    p.add_argument(
        "--net_type",
        type=str,
        default="Specific",
        help="density_negative: STRING | Non-Specific | Specific | Lofgof",
    )
    p.add_argument("--species", type=str, default="hESC", help="density_negative: hESC, mESC, …")
    p.add_argument("--n_genes", type=int, default=500, help="500 or 1000")
    p.add_argument(
        "--density",
        type=float,
        default=-1.0,
        help="Override network density for density_negative (default: from table)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()

    if args.benchmark_all:
        benchmark_root = Path(args.benchmark_root)
        if not benchmark_root.is_absolute():
            benchmark_root = (root / benchmark_root).resolve()
        else:
            benchmark_root = benchmark_root.resolve()
        out_subdir = args.benchmark_out_subdir.strip() or (
            "Train_validation_test_gnnlink_paper"
            if args.mode == "paper_bbad414"
            else "Train_validation_test"
        )
        # CLI --mode applies to every leaf (otherwise folder-based defaults are never used when
        # split_override was only paper_bbad414, and default --mode hard_negative was ignored).
        split_override = args.mode
        run_benchmark_all(
            benchmark_root,
            seed=args.seed,
            ratio=args.ratio,
            p_val=args.p_val,
            out_subdir=out_subdir,
            split_mode_override=split_override,
        )
        return

    if args.preset:
        rel_data, rel_out = _PRESET_TABLE[args.preset]
        data_dir = root / rel_data
        label_p = data_dir / "Label.csv"
        tf_p = data_dir / "TF.csv"
        target_p = data_dir / "Target.csv"
        out_dir = root / rel_out
    else:
        if not args.label or not args.tf or not args.target or not args.out_dir:
            raise SystemExit(
                "Provide --preset NAME or all of --label --tf --target --out_dir"
            )
        label_p = Path(args.label).resolve()
        tf_p = Path(args.tf).resolve()
        target_p = Path(args.target).resolve()
        out_dir = Path(args.out_dir).resolve()

    try:
        run_split(
            label_p,
            tf_p,
            target_p,
            out_dir,
            mode=args.mode,
            seed=args.seed,
            ratio=args.ratio,
            p_val=args.p_val,
            net_type=args.net_type,
            species=args.species,
            n_genes=args.n_genes,
            density_override=args.density,
            quiet=False,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
