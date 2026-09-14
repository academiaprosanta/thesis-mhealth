"""
STEP 4  (laptop, ~1 min)  -- THE HEADLINE COMPARISON

Fits the predictor three times and compares:

  unsupervised : NO target labels used.  <-- the deployable method
  supervised   : needs target labels.    <-- the baseline
  combined     : both, as an upper bound

Whatever the outcome, it is a result:
  unsup ~= sup   -> the method works without annotation. Best case.
  unsup <  sup   -> you have quantified what labels are worth.
  unsup >  sup   -> the model's own uncertainty beats ground-truth separation.

Validation is leave-one-domain-out, with a 200-shuffle permutation null.

Outputs: results/regression_<protocol>_<target>.txt
         results/predictions_<protocol>_<target>_<featureset>.csv
         results/summary_<protocol>_<target>.csv

Run:  python scripts/04_regression.py --protocol zeroshot --target y_norm
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULT_DIR, SEED
from src.design import FEATURE_SETS

HIGHER_IS_BETTER = {"y_raw": True, "y_norm": True, "y_gap": False}
MIN_TRAIN_ROWS = 8
MIN_DOMAINS = 5


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler(),
                         RidgeCV(alphas=np.logspace(-3, 4, 30)))


def gbm():
    return make_pipeline(SimpleImputer(strategy="median"),
                         GradientBoostingRegressor(random_state=SEED,
                                                   n_estimators=200,
                                                   max_depth=2,
                                                   learning_rate=0.05))


def lodo_predict(df, feats, target, model):
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
            preds.loc[te.index] = clone(model).fit(tr[feats], tr[target]).predict(te[feats])
        ran += 1
    return preds, ran, skipped


def score(y, yhat):
    ok = ~(y.isna() | yhat.isna())
    y, yhat = y[ok], yhat[ok]
    blank = dict(r2=np.nan, rmse=np.nan, spearman=np.nan, kendall=np.nan, n=len(y))
    if len(y) < 3:
        return blank
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return blank
    ssr = float(((y - yhat) ** 2).sum())
    return dict(r2=1 - ssr / sst, rmse=float(np.sqrt(ssr / len(y))),
                spearman=float(spearmanr(y, yhat).statistic),
                kendall=float(kendalltau(y, yhat).statistic), n=int(len(y)))


def source_selection(df, target, pred_col):
    higher = HIGHER_IS_BETTER[target]
    rows = []
    for tgt, g in df.groupby("target"):
        g = g.dropna(subset=[target, pred_col])
        if len(g) < 3:
            continue
        best = g[target].max() if higher else g[target].min()
        idx = g[pred_col].idxmax() if higher else g[pred_col].idxmin()
        rows.append(dict(target=tgt,
                         within_tau=float(kendalltau(g[target], g[pred_col]).statistic),
                         regret=float(abs(best - g.loc[idx, target])),
                         random_regret=float((best - g[target]).abs().mean()),
                         best_source=g.loc[g[target].idxmax() if higher
                                           else g[target].idxmin(), "source"],
                         picked_source=g.loc[idx, "source"]))
    return pd.DataFrame(rows)


def fmt(v, spec="+.3f"):
    return " n/a " if (v is None or (isinstance(v, float) and np.isnan(v))) else format(v, spec)


def main(protocol="zeroshot", target="y_norm", n_perm=200):
    path = RESULT_DIR / f"design_matrix_{protocol}.csv"
    if not path.exists():
        sys.exit(f"missing {path} -- run 03_build_design_matrix.py first")
    full = pd.read_csv(path)
    full = full[~full.is_diagonal].dropna(subset=[target]).reset_index(drop=True)
    n_dom = full.source.nunique()

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say(f"protocol = {protocol}    target = {target}")
    say(f"{len(full)} pairs from {n_dom} independent domains")
    if n_dom < MIN_DOMAINS:
        say(f"WARNING: leave-one-domain-out needs >= {MIN_DOMAINS} domains.")
    say()

    summary = []
    for set_name, feat_list in FEATURE_SETS.items():
        feats = [f for f in feat_list if f in full.columns
                 and full[f].notna().sum() > 0.5 * len(full)
                 and full[f].nunique(dropna=True) > 1]
        if not feats:
            say(f"{set_name}: no usable features, skipping"); continue

        df = full.copy()
        say(f"--- {set_name.upper()}  ({len(feats)} features, "
            f"{len(df)/len(feats):.1f} rows per feature) ---")

        best_r2, best_preds = -np.inf, None
        for mname, model in (("mean", None), ("ridge", ridge()), ("gbm", gbm())):
            preds, ran, skipped = lodo_predict(df, feats, target, model)
            s = score(df[target], preds)
            say(f"  {mname:6s} R2={fmt(s['r2'])}  RMSE={fmt(s['rmse'],'.4f')}  "
                f"rho={fmt(s['spearman'])}  tau={fmt(s['kendall'])}  "
                f"[{ran} folds, {skipped} skipped]")
            summary.append(dict(feature_set=set_name, model=mname, **s))
            if mname == "ridge":
                df["pred"] = preds
                ridge_r2 = s["r2"]

        # permutation null on ridge
        if not np.isnan(ridge_r2):
            rng = np.random.default_rng(SEED)
            null = []
            for _ in range(n_perm):
                sh = df.copy()
                sh[target] = rng.permutation(sh[target].values)
                p, _, _ = lodo_predict(sh, feats, target, ridge())
                null.append(score(sh[target], p)["r2"])
            null = np.array(null, float)
            valid = null[~np.isnan(null)]
            pval = float((valid >= ridge_r2).mean()) if len(valid) else np.nan
            say(f"  permutation: null 95th pct {np.percentile(valid,95):+.3f}, "
                f"real {ridge_r2:+.3f}, p = {pval:.3f}  "
                f"{'SIGNAL' if pval < 0.05 else 'NO SIGNAL'}")
            summary[-2]["p_value"] = pval

            sel = source_selection(df, target, "pred")
            if len(sel):
                say(f"  source selection: within-target tau {sel.within_tau.mean():+.3f}, "
                    f"regret {sel.regret.mean():.4f} vs random {sel.random_regret.mean():.4f}, "
                    f"{int((sel.regret<1e-9).sum())}/{len(sel)} perfect")

            fitted = ridge().fit(df[feats], df[target])
            coefs = fitted[-1].coef_
            order = np.argsort(-np.abs(coefs))[:8]
            say("  top standardised coefficients (collinear -- do not over-read):")
            for i in order:
                say(f"    {feats[i]:26s} {coefs[i]:+.4f}")

            df.to_csv(RESULT_DIR /
                      f"predictions_{protocol}_{target}_{set_name}.csv", index=False)
        say()

    sm = pd.DataFrame(summary)
    sm.to_csv(RESULT_DIR / f"summary_{protocol}_{target}.csv", index=False)
    say("=== HEADLINE: ridge, by feature set ===")
    r = sm[sm.model == "ridge"].set_index("feature_set")
    for k in ("unsupervised", "supervised", "combined"):
        if k in r.index:
            say(f"  {k:14s} R2={fmt(r.loc[k,'r2'])}  tau={fmt(r.loc[k,'kendall'])}  "
                f"p={fmt(r.loc[k].get('p_value', np.nan), '.3f')}")

    (RESULT_DIR / f"regression_{protocol}_{target}.txt").write_text("\n".join(lines))
    print(f"\nwrote regression_{protocol}_{target}.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="zeroshot", choices=["zeroshot", "probe"])
    p.add_argument("--target", default="y_norm", choices=["y_raw", "y_norm", "y_gap"])
    p.add_argument("--n-perm", dest="n_perm", type=int, default=200)
    main(**vars(p.parse_args()))
