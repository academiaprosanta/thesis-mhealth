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
from sklearn.base import clone
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

# A fold is only usable if it leaves enough rows to fit on. Below this the
# fold is skipped -- and if ALL folds are skipped you get no predictions,
# which is what happens with only 3 domains.
MIN_TRAIN_ROWS = 10
MIN_DOMAINS_FOR_LODO = 5


def make_models():
    return {
        "mean_baseline": None,   # handled specially
        "ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "gbm": GradientBoostingRegressor(random_state=SEED, n_estimators=200,
                                         max_depth=2, learning_rate=0.05),
    }


def lodo_predict(df, feats, target, model):
    """
    Leave-one-domain-out.

    Returns (predictions, n_folds_run, n_folds_skipped).
    A fold is skipped when holding out that domain leaves too few rows to
    train on -- unavoidable when you have very few domains.
    """
    domains = sorted(set(df.source) | set(df.target))
    preds = pd.Series(np.nan, index=df.index)
    ran = skipped = 0
    for d in domains:
        te = df[(df.source == d) | (df.target == d)]
        tr = df[(df.source != d) & (df.target != d)]
        if len(tr) < MIN_TRAIN_ROWS or len(te) == 0:
            skipped += 1
            continue
        if model is None:
            preds.loc[te.index] = tr[target].mean()
        else:
            m = clone(model).fit(tr[feats], tr[target])
            preds.loc[te.index] = m.predict(te[feats])
        ran += 1
    return preds, ran, skipped


def score(y, yhat):
    """Returns NaNs rather than nonsense when there is nothing to score."""
    ok = ~(y.isna() | yhat.isna())
    y, yhat = y[ok], yhat[ok]
    blank = {"r2": np.nan, "rmse": np.nan, "spearman": np.nan,
             "kendall": np.nan, "n": int(len(y))}
    if len(y) < 3:
        return blank
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return blank
    ss_res = float(((y - yhat) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(ss_res / len(y))),
        "spearman": float(spearmanr(y, yhat).statistic),
        "kendall": float(kendalltau(y, yhat).statistic),
        "n": int(len(y)),
    }


def fmt(v, spec="+.3f"):
    return "  n/a " if (v is None or (isinstance(v, float) and np.isnan(v))) else format(v, spec)


def main(smoke=False, target="y_raw", n_perm=200):
    suffix = "_smoke" if smoke else ""
    df = pd.read_csv(RESULT_DIR / f"design_matrix{suffix}.csv")
    df = df[~df.is_diagonal].dropna(subset=[target]).reset_index(drop=True)

    feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feats).reset_index(drop=True)
    n_dom = df.source.nunique()

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say(f"target = {target}")
    say(f"{len(df)} rows from {n_dom} independent domains")
    say(f"{len(feats)} features")
    say(f"rows per feature = {len(df)/len(feats):.1f}   (want >= 10)")
    say()

    # ---- can we even do leave-one-domain-out? ----------------------------
    if n_dom < MIN_DOMAINS_FOR_LODO:
        say("=" * 66)
        say(f"NOT ENOUGH DOMAINS. Leave-one-domain-out needs at least "
            f"{MIN_DOMAINS_FOR_LODO}.")
        say(f"With {n_dom} domains, holding one out leaves only "
            f"{(n_dom-1)*(n_dom-2)} training rows -- too few to fit anything.")
        say("Every fold will be skipped and every score will be blank.")
        say()
        say("This is EXPECTED for the smoke test. The smoke test only checks")
        say("that the code runs. Run the full 11-dataset pipeline to get")
        say("scores that mean something.")
        say("=" * 66)
        say()

    results = {}
    for name, model in make_models().items():
        preds, ran, skipped = lodo_predict(df, feats, target, model)
        results[name] = score(df[target], preds)
        s = results[name]
        say(f"{name:15s} R2={fmt(s['r2'])}  RMSE={fmt(s['rmse'], '.4f')}  "
            f"rho={fmt(s['spearman'])}  tau={fmt(s['kendall'])}  "
            f"[{ran} folds ran, {skipped} skipped, n={s['n']}]")
        if name == "ridge":
            df["pred_ridge"] = preds

    real = results["ridge"]["r2"]

    # ---- permutation test -------------------------------------------------
    say()
    if np.isnan(real):
        say("permutation test SKIPPED -- no valid predictions to test.")
    else:
        say(f"permutation test on ridge, {n_perm} shuffles")
        rng = np.random.default_rng(SEED)
        ridge = make_models()["ridge"]
        null = []
        for _ in range(n_perm):
            sh = df.copy()
            sh[target] = rng.permutation(sh[target].values)
            p, _, _ = lodo_predict(sh, feats, target, ridge)
            null.append(score(sh[target], p)["r2"])
        null = np.array(null, dtype=float)
        valid = null[~np.isnan(null)]
        if len(valid) < n_perm // 2:
            say("  too many failed shuffles -- result unreliable")
        else:
            pval = float((valid >= real).mean())
            say(f"  null R2: mean {valid.mean():+.3f}, "
                f"95th pct {np.percentile(valid, 95):+.3f}  "
                f"({len(valid)}/{n_perm} valid)")
            say(f"  real R2: {real:+.3f}")
            verdict = "SIGNAL" if pval < 0.05 else "NOT DISTINGUISHABLE FROM NOISE"
            say(f"  p = {pval:.3f}   {verdict}")

    # ---- which features did ridge lean on? --------------------------------
    say()
    if len(df) < 2 * len(feats):
        say(f"NOTE: {len(df)} rows against {len(feats)} features. The fit below")
        say("is hopelessly overfit -- read nothing into these coefficients.")
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
