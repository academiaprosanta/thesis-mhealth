"""
UNSUPERVISED (label-free) pair features -- the core of the new experiment.

NONE of these functions receives the target's true labels. Everything is
computed from:
   * the target's images, run once through the source model
   * the source's own images, run through the same model

This is what makes the method deployable: a hospital hands over unlabelled
X-rays, you run one forward pass, and you get a prediction.

Six families:
  A  source-model behaviour on the target   (entropy, confidence, skew)
  B  pseudo-label geometry                  (separation using PREDICTED labels)
  C  unsupervised cluster structure         (k-means, no labels at all)
  D  spectral shape of the target features  (label-free by construction)
  E  source-to-target distance              (MMD, CORAL, Frechet, A-distance)
  F  simple summaries                       (norms, mean-vector alignment)
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score

from .config import GEOM_SAMPLE, MMD_SAMPLE, SEED
from .geometry import fisher_ratio, effective_rank, spectral_decay


def _sub(X, n, seed=SEED):
    if len(X) <= n:
        return X
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[idx]


# --------------------------------------------------------------- family A
def prediction_behaviour(P):
    """
    How does the source model BEHAVE on the target's unlabelled images?

    P : (N, 2) softmax probabilities from the source's own classifier.

    Intuition: a model that feels at home on a dataset is confident and
    splits it sensibly. A model out of its depth is uncertain, or collapses
    to predicting one class for everything.
    """
    eps = 1e-12
    ent = -(P * np.log(P + eps)).sum(axis=1)          # per-image uncertainty
    conf = P.max(axis=1)                              # per-image confidence
    p_abn = P[:, 1]
    return dict(
        pred_entropy_mean=float(ent.mean()),
        pred_entropy_std=float(ent.std()),
        pred_conf_mean=float(conf.mean()),
        pred_conf_std=float(conf.std()),
        pred_frac_confident=float((conf > 0.9).mean()),
        pred_positive_rate=float((p_abn >= 0.5).mean()),
        pred_prob_mean=float(p_abn.mean()),
        pred_prob_std=float(p_abn.std()),
    )


# --------------------------------------------------------------- family B
def pseudo_label_geometry(X, P, seed=SEED):
    """
    Take the source model's PREDICTIONS as if they were labels, then measure
    class separation exactly as the supervised metrics do.

    Asks: does the source model carve this target into two clean groups?
    Uses no ground truth.
    """
    yhat = P.argmax(axis=1)
    if len(np.unique(yhat)) < 2:
        # model predicted one class for everything -- itself informative
        return dict(pseudo_silhouette=np.nan, pseudo_davies_bouldin=np.nan,
                    pseudo_fisher=np.nan, pseudo_collapsed=1.0)
    n = min(GEOM_SAMPLE, len(X))
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    Xs, ys = X[idx], yhat[idx]
    return dict(
        pseudo_silhouette=float(silhouette_score(Xs, ys, random_state=seed)),
        pseudo_davies_bouldin=float(davies_bouldin_score(Xs, ys)),
        pseudo_fisher=fisher_ratio(Xs, ys),
        pseudo_collapsed=0.0,
    )


# --------------------------------------------------------------- family C
def cluster_structure(X, seed=SEED):
    """
    Ignore the model entirely: does the target split into two clean groups
    on its own? k-means finds two clusters without being told anything.
    """
    n = min(GEOM_SAMPLE, len(X))
    Xs = _sub(X, n, seed)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xs)
    lab = km.labels_
    if len(np.unique(lab)) < 2:
        return dict(kmeans_silhouette=np.nan, kmeans_balance=np.nan,
                    kmeans_inertia_ratio=np.nan)
    frac = lab.mean()
    total_var = float(((Xs - Xs.mean(0)) ** 2).sum())
    return dict(
        kmeans_silhouette=float(silhouette_score(Xs, lab, random_state=seed)),
        kmeans_balance=float(min(frac, 1 - frac) * 2),   # 1.0 = even split
        kmeans_inertia_ratio=float(km.inertia_ / total_var) if total_var > 0 else np.nan,
    )


# --------------------------------------------------------------- family D
def spectral_shape(X, seed=SEED):
    Xs = _sub(X, GEOM_SAMPLE, seed)
    return dict(
        eff_rank=effective_rank(Xs),
        spec_decay=spectral_decay(Xs),
        feat_norm_mean=float(np.linalg.norm(Xs, axis=1).mean()),
        feat_norm_std=float(np.linalg.norm(Xs, axis=1).std()),
    )


# --------------------------------------------------------------- family E
def _mmd_rbf(A, B, seed=SEED):
    """
    Maximum Mean Discrepancy: how different are two clouds of points?
    0 = indistinguishable. Uses the median pairwise distance as bandwidth.
    """
    A, B = _sub(A, MMD_SAMPLE, seed), _sub(B, MMD_SAMPLE, seed + 1)
    Z = np.vstack([A, B]).astype(np.float64)
    # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b  -- never materialise an (n,n,d) array
    sq = (Z ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0, out=d2)
    med = np.median(d2[d2 > 0])
    if med <= 0:
        return np.nan
    g = 1.0 / med
    na = len(A)
    K = np.exp(-g * d2)
    return float(K[:na, :na].mean() + K[na:, na:].mean() - 2 * K[:na, na:].mean())


def _coral(A, B):
    """Distance between the two covariance matrices."""
    Ca = np.cov(A, rowvar=False)
    Cb = np.cov(B, rowvar=False)
    d = A.shape[1]
    return float(((Ca - Cb) ** 2).sum() / (4 * d * d))


def _frechet(A, B, dim=64, seed=SEED):
    """FID-style distance, computed after a shared PCA for stability."""
    from scipy.linalg import sqrtm
    Z = np.vstack([A, B])
    k = min(dim, Z.shape[1], len(Z) - 1)
    p = PCA(n_components=k, random_state=seed).fit(Z)
    a, b = p.transform(A), p.transform(B)
    mu = ((a.mean(0) - b.mean(0)) ** 2).sum()
    Ca, Cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cov = sqrtm(Ca @ Cb)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(mu + np.trace(Ca + Cb - 2 * cov))


def _a_distance(A, B, seed=SEED):
    """
    Train a classifier to tell source from target. If it cannot, the domains
    look alike. Needs only which dataset each image came from -- free.
    """
    A, B = _sub(A, MMD_SAMPLE, seed), _sub(B, MMD_SAMPLE, seed + 1)
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
    err = 1 - acc
    return float(max(0.0, 2 * (1 - 2 * err)))


def domain_distance(Xs, Xt, seed=SEED):
    """All source-to-target distances, computed in the SOURCE feature space."""
    ms, mt = Xs.mean(0), Xt.mean(0)
    cos = float(ms @ mt / (np.linalg.norm(ms) * np.linalg.norm(mt) + 1e-12))
    return dict(
        mmd=_mmd_rbf(Xs, Xt, seed),
        coral=_coral(Xs, Xt),
        frechet=_frechet(Xs, Xt, seed=seed),
        a_distance=_a_distance(Xs, Xt, seed),
        mean_cosine=cos,
        mean_shift=float(np.linalg.norm(ms - mt)),
        norm_ratio=float(np.linalg.norm(Xt, axis=1).mean()
                         / (np.linalg.norm(Xs, axis=1).mean() + 1e-12)),
    )


# --------------------------------------------------------------- assemble
def unsupervised_pair_features(Xs, Xt, Pt, seed=SEED):
    """
    Every label-free feature for one (source, target) pair.

    Xs : source images through the source backbone   (N_s, 512)
    Xt : target images through the source backbone   (N_t, 512)
    Pt : source classifier applied to Xt             (N_t, 2)

    No target labels anywhere.
    """
    out = {}
    out.update(prediction_behaviour(Pt))
    out.update(pseudo_label_geometry(Xt, Pt, seed))
    out.update({f"tgt_{k}": v for k, v in cluster_structure(Xt, seed).items()})
    out.update({f"tgt_{k}": v for k, v in spectral_shape(Xt, seed).items()})
    out.update({f"src_{k}": v for k, v in spectral_shape(Xs, seed).items()})
    out.update(domain_distance(Xs, Xt, seed))
    # signed spectral differences -- direction matters (report Q2 vs Q3)
    for k in ("eff_rank", "spec_decay", "feat_norm_mean"):
        out[f"delta_{k}"] = out[f"src_{k}"] - out[f"tgt_{k}"]
    return out


UNSUP_FEATURES = [
    # A: source-model behaviour on unlabelled target
    "pred_entropy_mean", "pred_entropy_std", "pred_conf_mean", "pred_conf_std",
    "pred_frac_confident", "pred_positive_rate", "pred_prob_mean", "pred_prob_std",
    # B: pseudo-label geometry
    "pseudo_silhouette", "pseudo_davies_bouldin", "pseudo_fisher", "pseudo_collapsed",
    # C: unsupervised clustering
    "tgt_kmeans_silhouette", "tgt_kmeans_balance", "tgt_kmeans_inertia_ratio",
    # D: spectral shape
    "tgt_eff_rank", "tgt_spec_decay", "tgt_feat_norm_mean", "tgt_feat_norm_std",
    "src_eff_rank", "src_spec_decay", "src_feat_norm_mean",
    "delta_eff_rank", "delta_spec_decay", "delta_feat_norm_mean",
    # E: domain distance
    "mmd", "coral", "frechet", "a_distance", "mean_cosine", "mean_shift", "norm_ratio",
    # source quality (measured on the source's own test set -- no target labels)
    "auc_source_on_source",
]
