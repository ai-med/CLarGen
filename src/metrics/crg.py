"""CRG score (Hamamci et al., MIDL 2025): a distribution-aware clinical metric."""
import numpy as np


def crg_score(true_labels, pred_labels):
    """Normalised CRG score in [1/3, 1] over all cells of the label matrices (NaN if A = 0 or A = T).

    ``true_labels`` and ``pred_labels`` are binary arrays of identical shape.
    """
    y_true = np.asarray(true_labels).astype(int).ravel()
    y_pred = np.asarray(pred_labels).astype(int).ravel()

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    T = TP + FP + FN + TN
    A = TP + FN
    if A == 0 or A == T:
        return float("nan")

    wTP = wFN = (T - A) / (2 * A)  # distribution-aware weights
    wFP = 1.0
    s = TP * wTP - FN * wFN - FP * wFP
    Smax = A * wTP
    denom = 2 * Smax - s
    return Smax / denom if denom != 0 else 1.0


def crg_against_database(query_labels, database_labels):
    """``crg_score(query_labels, database_labels[i])`` for every row i, vectorised.

    The operations mirror :func:`crg_score` exactly so that rankings built on these
    scores (Stage-2 fallback) are identical to the per-row computation.
    """
    q = np.asarray(query_labels).astype(int)
    L = np.asarray(database_labels).astype(int)
    T = q.size
    A = int(q.sum())
    if A == 0 or A == T:
        return np.full(len(L), np.nan)

    TP = L[:, q == 1].sum(axis=1)
    FN = A - TP
    FP = L[:, q == 0].sum(axis=1)
    wTP = wFN = (T - A) / (2 * A)
    wFP = 1.0
    s = TP * wTP - FN * wFN - FP * wFP
    Smax = A * wTP
    denom = 2 * Smax - s
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom != 0, Smax / denom, 1.0)
