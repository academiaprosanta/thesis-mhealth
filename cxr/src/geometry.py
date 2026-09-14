"""
SUPERVISED features -- these need target labels.

Kept for comparison against the unsupervised set. Computed on a shared
frozen ImageNet backbone so the values are comparable across domains.
"""
import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

from .config import GEOM_SAMPLE, SEED


def fisher_ratio(X, y):
    """between-class scatter / within-class scatter. Higher = separated."""
    mu = X.mean(axis=0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        sb += len(Xc) * np.sum((mc - mu) ** 2)
        sw += np.sum((Xc - mc) ** 2)
    return float(sb / sw) if sw > 0 else np.nan


def effective_rank(X):
    """exp(entropy of the normalised eigenvalue spectrum). LABEL-FREE."""
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    if tot <= 0:
        return np.nan
    p = ev / tot
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def spectral_decay(X):
    """
    Slope of log(eigenvalue) vs log(rank). LABEL-FREE.
    Steeper (more negative) = information concentrated in few directions.
    """
    Xc = X - X.mean(axis=0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev[ev > 1e-12][:200]
    if len(ev) < 10:
        return np.nan
    r = np.arange(1, len(ev) + 1)
    return float(np.polyfit(np.log(r), np.log(ev), 1)[0])


def balance_entropy(y):
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


def _sub(X, y=None, n=GEOM_SAMPLE, seed=SEED):
    if len(X) <= n:
        return (X, y) if y is not None else X
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return (X[idx], y[idx]) if y is not None else X[idx]


def supervised_geometry(X, y, seed=SEED):
    """Needs labels. Silhouette / Fisher / DB / CH plus label-free extras."""
    row = dict(
        n_samples=int(len(X)),
        abnormal_rate=float(np.mean(y)),
        balance_entropy=balance_entropy(y),
        effective_rank=effective_rank(_sub(X)),
        spectral_decay=spectral_decay(_sub(X)),
    )
    if len(np.unique(y)) < 2:
        row.update(silhouette=np.nan, davies_bouldin=np.nan,
                   calinski_harabasz=np.nan, fisher_ratio=np.nan)
        return row
    Xs, ys = _sub(X, y)
    row["silhouette"] = float(silhouette_score(Xs, ys, random_state=seed))
    row["davies_bouldin"] = float(davies_bouldin_score(Xs, ys))
    row["calinski_harabasz"] = float(calinski_harabasz_score(Xs, ys))
    row["fisher_ratio"] = fisher_ratio(Xs, ys)
    return row
