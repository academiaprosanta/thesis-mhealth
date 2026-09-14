"""
Building the pair-level design matrix, and the three feature sets we compare.

  SUPERVISED    : needs target labels (Silhouette, Fisher, DB and friends)
  UNSUPERVISED  : needs no target labels at all -- the deployable set
  COMBINED      : both, as an upper bound
"""
import numpy as np

from .unsupervised import UNSUP_FEATURES

# metrics computed per domain by scripts/01_geometry.py on the shared
# neutral ImageNet backbone
GEOM_COLS = ["silhouette", "davies_bouldin", "fisher_ratio", "calinski_harabasz",
             "effective_rank", "spectral_decay", "balance_entropy",
             "abnormal_rate", "n_samples"]

SUP_FEATURES = (
    [f"{c}_source" for c in GEOM_COLS if c != "n_samples"]
    + [f"{c}_target" for c in GEOM_COLS if c != "n_samples"]
    + [f"delta_{c}" for c in ["silhouette", "davies_bouldin", "fisher_ratio",
                              "effective_rank", "spectral_decay",
                              "balance_entropy", "abnormal_rate"]]
    + ["log_n_source", "log_n_target", "delta_log_n", "auc_source_on_source"]
)

FEATURE_SETS = {
    "unsupervised": UNSUP_FEATURES,
    "supervised": SUP_FEATURES,
    "combined": sorted(set(UNSUP_FEATURES) | set(SUP_FEATURES)),
}


def add_supervised_columns(df, geo):
    """
    df  : transfer matrix, one row per pair, with 'source' and 'target'
    geo : per-domain geometry table indexed by domain name
    """
    df = df.merge(geo.add_suffix("_source"),
                  left_on="source", right_index=True, how="left")
    df = df.merge(geo.add_suffix("_target"),
                  left_on="target", right_index=True, how="left")
    for c in GEOM_COLS:
        if f"{c}_source" in df.columns:
            df[f"delta_{c}"] = df[f"{c}_source"] - df[f"{c}_target"]
    for role in ("source", "target"):
        df[f"log_n_{role}"] = np.log10(df[f"n_samples_{role}"].clip(lower=1))
    df["delta_log_n"] = df["log_n_source"] - df["log_n_target"]
    return df


def add_targets(df, protocol="zeroshot"):
    """
    y_raw   measured AUC of source -> target
    y_norm  divided by the ceiling (a model trained on the target itself)
    y_gap   ceiling minus measured
    """
    auc = f"{protocol}_auc"
    ceiling = (df[df.source == df.target][["target", auc]]
               .rename(columns={auc: "ceiling"}))
    df = df.merge(ceiling, on="target", how="left")
    df["y_raw"] = df[auc]
    df["y_norm"] = df[auc] / df["ceiling"]
    df["y_gap"] = df["ceiling"] - df[auc]
    df["is_diagonal"] = df.source == df.target
    return df
