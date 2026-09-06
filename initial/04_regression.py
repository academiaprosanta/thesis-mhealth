"""
STEP 4  (laptop, ~30 seconds)

Fit the predictor and evaluate it honestly.

Two things here are non-negotiable:

LEAVE-ONE-DOMAIN-OUT. If you split the rows randomly, pair (A->B) lands in
train and (A->C) in test -- and both contain A's silhouette value. The model
memorises A and your score is inflated. Instead, hold out every row that
touches one domain.

PERMUTATION TEST. Shuffle the outcomes, refit, record the score. Do it 200
times. Your real score must sit outside that distribution. With ~100 rows,
respectable-looking R2 values arise from pure noise all the time.

Output: results/regression_report.txt, results/lodo_predictions.csv

Run:  python scripts/04_regression.py [--smoke] [--target y_norm]
"""

# --- make "import src..." work no matter where you run this from ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULT_DIR, SEED

FEATURES = [
    "silhouette_source", "silhouette_target", "delta_silhouette",
    "fisher_ratio_source", "fisher_ratio_target", "delta_fisher_ratio",
    "davies_bouldin_source", "davies_bouldin_target", "delta_davies_bouldin",
    "effective_rank_source", "effective_rank_target", "delta_effective_rank",
    "balance_entropy_source", "balance_entropy_target",
    "log_n_source", "log_n_target", "delta_log_n",
    "n_classes_source", "n_classes_target",
    "auc_source_on_source",
]


def make_models():
    return {
        "mean_baseline": None,   # handled specially
        "ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "gbm": GradientBoostingRegressor(random_state=SEED, n_estimators=200,
                                         max_depth=2, learning_rate=0.05),
    }


def lodo_predict(df, feats, target, model):
    """Leave-one-domain-out. Returns predictions aligned to df's index."""
    domains = sorted(set(df.source) | set(df.target))
    preds = pd.Series(np.nan, index=df.index)
    for d in domains:
        te = df[(df.source == d) | (df.target == d)]
        tr = df[(df.source != d) & (df.target != d)]
        if len(tr) < 10 or len(te) == 0:
            continue
        if model is None:
            preds.loc[te.index] = tr[target].mean()
        else:
            from sklearn.base import clone
            m = clone(model).fit(tr[feats], tr[target])
            preds.loc[te.index] = m.predict(te[feats])
    return preds


def score(y, yhat):
    ok = ~(y.isna() | yhat.isna())
    y, yhat = y[ok], yhat[ok]
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(ss_res / len(y))),
        "spearman": float(spearmanr(y, yhat).statistic),
        "kendall": float(kendalltau(y, yhat).statistic),
        "n": int(len(y)),
    }


def main(smoke=False, target="y_raw", n_perm=200):
    suffix = "_smoke" if smoke else ""
    df = pd.read_csv(RESULT_DIR / f"design_matrix{suffix}.csv")
    df = df[~df.is_diagonal].dropna(subset=[target]).reset_index(drop=True)

    feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feats).reset_index(drop=True)

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say(f"target = {target}")
    say(f"{len(df)} rows from {df.source.nunique()} independent domains")
    say(f"{len(feats)} features")
    say(f"rows per feature = {len(df)/len(feats):.1f}   (want >= 10)")
    say()

    results = {}
    for name, model in make_models().items():
        preds = lodo_predict(df, feats, target, model)
        results[name] = score(df[target], preds)
        s = results[name]
        say(f"{name:15s} R2={s['r2']:+.3f}  RMSE={s['rmse']:.4f}  "
            f"rho={s['spearman']:+.3f}  tau={s['kendall']:+.3f}")
        if name == "ridge":
            df["pred_ridge"] = preds

    # ---- permutation test on the best real model -------------------------
    say()
    say(f"permutation test on ridge, {n_perm} shuffles")
    rng = np.random.default_rng(SEED)
    null = []
    ridge = make_models()["ridge"]
    for _ in range(n_perm):
        shuffled = df.copy()
        shuffled[target] = rng.permutation(shuffled[target].values)
        null.append(score(shuffled[target], lodo_predict(shuffled, feats, target, ridge))["r2"])
    null = np.array(null)
    real = results["ridge"]["r2"]
    pval = float((null >= real).mean())
    say(f"  null R2: mean {null.mean():+.3f}, 95th pct {np.percentile(null,95):+.3f}")
    say(f"  real R2: {real:+.3f}")
    say(f"  p = {pval:.3f}   {'SIGNAL' if pval < 0.05 else 'NOT DISTINGUISHABLE FROM NOISE'}")

    # ---- which features did ridge lean on? -------------------------------
    say()
    fitted = make_models()["ridge"].fit(df[feats], df[target])
    coefs = fitted[-1].coef_
    order = np.argsort(-np.abs(coefs))
    say("ridge coefficients (standardised, full-data fit, largest first):")
    for i in order[:10]:
        say(f"  {feats[i]:28s} {coefs[i]:+.4f}")

    (RESULT_DIR / f"regression_report{suffix}.txt").write_text("\n".join(lines))
    df.to_csv(RESULT_DIR / f"lodo_predictions{suffix}.csv", index=False)
    say()
    say(f"wrote results/regression_report{suffix}.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--target", default="y_raw", choices=["y_raw", "y_norm", "y_gap"])
    p.add_argument("--n-perm", dest="n_perm", type=int, default=200)
    main(**vars(p.parse_args()))
