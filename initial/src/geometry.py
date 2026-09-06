"""
The dataset-geometry metrics -- your input features.

Mental picture: every image is now a point in 512-dimensional space.
Points of the same class ought to sit near each other. These functions
all measure, in different ways, "how cleanly do the classes separate?"
"""
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .config import SILHOUETTE_SAMPLE, SEED


def fisher_ratio(X, y):
    """
    trace(between-class scatter) / trace(within-class scatter).

    Big  = class means are far apart relative to the spread inside a class.
    This is the "Fisher ratio" in the statistical report.
    """
    overall_mean = X.mean(axis=0)
    sb = 0.0
    sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        sb += len(Xc) * np.sum((mu_c - overall_mean) ** 2)
        sw += np.sum((Xc - mu_c) ** 2)
    return float(sb / sw) if sw > 0 else np.nan


def effective_rank(X):
    """
    How many dimensions the features actually use.

    Take the eigenvalues of the covariance, normalise them into a probability
    distribution, take the exponential of its entropy. If the features spread
    across 200 directions you get ~200; if they collapse onto a line, ~1.

    Not in the original report -- one of the cheap extras worth trying.
    """
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    total = ev.sum()
    if total <= 0:
        return np.nan
    p = ev / total
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def class_balance_entropy(y):
    """0 = one class dominates everything. 1 = perfectly balanced."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


def geometry_metrics(X, y, seed=SEED):
    """All of the above, as one dictionary = one row of your geometry table."""
    n_cls = len(np.unique(y))
    row = {
        "n_samples": int(len(X)),
        "n_classes": int(n_cls),
        "balance_entropy": class_balance_entropy(y),
        "effective_rank": effective_rank(X),
    }

    if n_cls < 2:
        row.update(silhouette=np.nan, davies_bouldin=np.nan,
                   calinski_harabasz=np.nan, fisher_ratio=np.nan)
        return row

    # Silhouette compares every point to every other point -- O(n^2).
    # At 50,000 points that is 2.5 billion distances. Subsample.
    row["silhouette"] = float(silhouette_score(
        X, y, sample_size=min(SILHOUETTE_SAMPLE, len(X)), random_state=seed))
    row["davies_bouldin"] = float(davies_bouldin_score(X, y))
    row["calinski_harabasz"] = float(calinski_harabasz_score(X, y))
    row["fisher_ratio"] = fisher_ratio(X, y)
    return row
