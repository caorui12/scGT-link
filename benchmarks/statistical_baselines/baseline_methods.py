"""PCC, mutual information, and GRNBoost2 edge scoring for GENELink-style benchmark splits."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


def read_tf_names(tf_csv: Path) -> list[str]:
    df = pd.read_csv(tf_csv, index_col=0)
    if "TF" in df.columns:
        return df["TF"].astype(str).tolist()
    if "Gene" in df.columns:
        return df["Gene"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def load_expression(expr_path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(expr_path, index_col=0)
    genes = df.index.astype(str).tolist()
    X = np.asarray(df.values, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    return X, genes


def preprocess_expression(X: np.ndarray, standardize: bool) -> np.ndarray:
    """Genes × cells. Optionally z-score each gene across cells."""
    if not standardize:
        return X
    scaler = StandardScaler()
    # cells × genes
    return scaler.fit_transform(X.T).T


def load_edges(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, index_col=0)
    if not {"TF", "Target", "Label"}.issubset(df.columns):
        raise ValueError(f"Expected TF, Target, Label columns in {csv_path}")
    tf = df["TF"].values.astype(np.int64)
    tgt = df["Target"].values.astype(np.int64)
    lab = df["Label"].values.astype(np.float64)
    return tf, tgt, lab


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def compute_abs_pearson_matrix(X: np.ndarray) -> np.ndarray:
    """X: genes × cells → |Pearson correlation| between rows."""
    X = np.asarray(X, dtype=np.float64)
    # numpy corrcoef treats rows as variables
    r = np.corrcoef(X)
    r = np.nan_to_num(r, nan=0.0)
    return np.abs(r)


def scores_from_corr(corr_abs: np.ndarray, tf: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    out = np.empty(tf.shape[0], dtype=np.float64)
    for k in range(tf.shape[0]):
        i, j = int(tf[k]), int(tgt[k])
        out[k] = float(corr_abs[i, j])
    return out


def mutual_information_scores(
    X: np.ndarray,
    tf: np.ndarray,
    tgt: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Edge-wise sklearn MI between TF row and Target row (genes × cells)."""
    pairs = [(int(tf[k]), int(tgt[k])) for k in range(tf.shape[0])]
    uniq: dict[tuple[int, int], float] = {}
    rng = random_state

    def mi_pair(i: int, j: int) -> float:
        xi = X[i]
        xj = X[j]
        if float(np.std(xi)) < 1e-12 or float(np.std(xj)) < 1e-12:
            return 0.0
        # sklearn MI regression: MI(features=tf_expr, target=tgt_expr)
        return float(
            mutual_info_regression(xi.reshape(-1, 1), xj, random_state=rng)[0]
        )

    for i, j in dict.fromkeys(pairs):
        uniq[(i, j)] = mi_pair(i, j)

    return np.array([uniq[p] for p in pairs], dtype=np.float64)


def run_grnboost2_network(
    gene_names: list[str],
    X_genes_cells: np.ndarray,
    tf_names: list[str],
    seed: int,
) -> pd.DataFrame:
    # Mitigate arboreto + dask-expr / newer Dask breaking delayed graphs (arboreto#42, pySCENIC#561).
    os.environ.setdefault("DASK_EXPR_ENABLED", "false")
    os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "false")

    from arboreto.algo import grnboost2

    # cells × genes; columns must match gene symbols
    cols = [str(g) for g in gene_names]
    col_set = set(cols)
    ex_T = pd.DataFrame(X_genes_cells.T, columns=cols)
    tf_ok = [str(t) for t in tf_names if str(t) in col_set]
    if not tf_ok:
        raise RuntimeError("No TF names intersect expression columns.")
    net = grnboost2(expression_data=ex_T, tf_names=tf_ok, seed=seed)
    if not isinstance(net, pd.DataFrame):
        net = pd.DataFrame(net)
    return net


def grnboost2_importance_map(network_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    df = network_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in df.columns]
    tf_col = tg_col = imp_col = None
    for i, c in enumerate(lower):
        if c == "tf":
            tf_col = df.columns[i]
        elif c in ("target", "gene"):
            tg_col = df.columns[i]
        elif "importance" in c:
            imp_col = df.columns[i]
    if tf_col is None:
        tf_col = df.columns[0]
    if tg_col is None:
        tg_col = df.columns[1]
    if imp_col is None:
        imp_col = df.columns[min(2, len(df.columns) - 1)]

    out: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        tf_sym = str(row[tf_col])
        tg_sym = str(row[tg_col])
        imp = float(row[imp_col])
        out[(tf_sym, tg_sym)] = imp
    return out


def scores_from_grnboost_map(
    gene_names: list[str],
    tf_idx: np.ndarray,
    tgt_idx: np.ndarray,
    importance: dict[tuple[str, str], float],
) -> np.ndarray:
    """Importance keys use original symbols; indices map into expression row order."""
    g = [str(x) for x in gene_names]
    out = np.zeros(tf_idx.shape[0], dtype=np.float64)
    for k in range(tf_idx.shape[0]):
        gi = g[int(tf_idx[k])]
        gj = g[int(tgt_idx[k])]
        out[k] = importance.get((gi, gj), 0.0)
    return out


def evaluate_task_methods(
    data_dir: Path,
    split_dir: Path,
    *,
    standardize: bool,
    seed: int,
    methods: frozenset[str],
) -> dict[str, Any]:
    """Returns metrics dict per method plus timings."""
    expr_path = data_dir / "BL--ExpressionData.csv"
    tf_path = data_dir / "TF.csv"

    X_raw, genes = load_expression(expr_path)
    n_genes = len(genes)
    tf_names = read_tf_names(tf_path)

    tf_val, tg_val, y_val = load_edges(split_dir / "Validation_set.csv")
    tf_te, tg_te, y_te = load_edges(split_dir / "Test_set.csv")

    for arr in (tf_val, tg_val, tf_te, tg_te):
        if np.any(arr >= n_genes) or np.any(arr < 0):
            raise ValueError("Edge indices out of range for expression rows.")

    X = preprocess_expression(X_raw, standardize)
    results: dict[str, Any] = {}

    corr_abs = None
    if "pcc" in methods:
        t0 = time.perf_counter()
        try:
            corr_abs = compute_abs_pearson_matrix(X)
            s_val = scores_from_corr(corr_abs, tf_val, tg_val)
            s_te = scores_from_corr(corr_abs, tf_te, tg_te)
            v_auroc, v_auprc = auc_safe(y_val, s_val)
            t_auroc, t_auprc = auc_safe(y_te, s_te)
            results["pcc"] = {
                "val_auroc": v_auroc,
                "val_auprc": v_auprc,
                "test_auroc": t_auroc,
                "test_auprc": t_auprc,
                "seconds": time.perf_counter() - t0,
            }
        except Exception as e:
            results["pcc"] = {"error": repr(e), "seconds": time.perf_counter() - t0}

    if "mi" in methods:
        t0 = time.perf_counter()
        try:
            s_val = mutual_information_scores(X, tf_val, tg_val, seed)
            s_te = mutual_information_scores(X, tf_te, tg_te, seed + 1)
            v_auroc, v_auprc = auc_safe(y_val, s_val)
            t_auroc, t_auprc = auc_safe(y_te, s_te)
            results["mi"] = {
                "val_auroc": v_auroc,
                "val_auprc": v_auprc,
                "test_auroc": t_auroc,
                "test_auprc": t_auprc,
                "seconds": time.perf_counter() - t0,
            }
        except Exception as e:
            results["mi"] = {"error": repr(e), "seconds": time.perf_counter() - t0}

    if "grnboost2" in methods:
        t0 = time.perf_counter()
        try:
            net = run_grnboost2_network(genes, X, tf_names, seed=seed)
            imp_map = grnboost2_importance_map(net)
            s_val = scores_from_grnboost_map(genes, tf_val, tg_val, imp_map)
            s_te = scores_from_grnboost_map(genes, tf_te, tg_te, imp_map)
            v_auroc, v_auprc = auc_safe(y_val, s_val)
            t_auroc, t_auprc = auc_safe(y_te, s_te)
            results["grnboost2"] = {
                "val_auroc": v_auroc,
                "val_auprc": v_auprc,
                "test_auroc": t_auroc,
                "test_auprc": t_auprc,
                "seconds": time.perf_counter() - t0,
                "n_edges_inferred": int(len(net)),
            }
        except ImportError as e:
            results["grnboost2"] = {
                "error": f"arboreto not installed: {e}",
                "seconds": time.perf_counter() - t0,
            }
        except Exception as e:
            msg = (
                f"{type(e).__name__}: {e}. "
                "Known trigger: arboreto GRNBoost2 vs newer Dask/dask-expr "
                '(try env DASK_EXPR_ENABLED=false or pip pin "dask<2024.8").'
            )
            results["grnboost2"] = {"error": msg, "seconds": time.perf_counter() - t0}

    return results
