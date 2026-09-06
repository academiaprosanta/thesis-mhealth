"""
STEP 3  (laptop, instant -- no GPU needed)

Join the two CSVs into the table you actually model.

One row = one source->target pair, with:
  - the outcome you want to predict (transfer AUC, and a normalised version)
  - geometry of the source
  - geometry of the target
  - the SIGNED difference between them

The signed differences matter: the statistical report found that the
absolute gap between two datasets carried almost no signal (|r| ~ 0.05)
while the signed difference did (|r| ~ 0.6). Direction is the whole story.

Output: results/design_matrix.csv

Run:  python scripts/03_build_design_matrix.py [--smoke]
"""

# --- make "import src..." work no matter where you run this from ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd

from src.config import RESULT_DIR

GEOM_COLS = ["silhouette", "davies_bouldin", "fisher_ratio",
             "calinski_harabasz", "effective_rank",
             "balance_entropy", "n_samples", "n_classes"]


def main(smoke=False):
    suffix = "_smoke" if smoke else ""
    geo = pd.read_csv(RESULT_DIR / f"geometry{suffix}.csv")
    tm = pd.read_csv(RESULT_DIR / f"transfer_matrix{suffix}.csv")

    # ---- the ceiling: how well does a model trained on T do on T? --------
    ceiling = (tm[tm.source == tm.target][["target", "transfer_auc"]]
               .rename(columns={"transfer_auc": "auc_target_ceiling"}))
    df = tm.merge(ceiling, on="target", how="left")

    # ---- attach geometry for both roles ----------------------------------
    df = df.merge(geo.add_suffix("_source"),
                  left_on="source", right_on="dataset_source", how="left")
    df = df.merge(geo.add_suffix("_target"),
                  left_on="target", right_on="dataset_target", how="left")
    df = df.drop(columns=["dataset_source", "dataset_target"])

    # ---- signed differences ---------------------------------------------
    for c in GEOM_COLS:
        df[f"delta_{c}"] = df[f"{c}_source"] - df[f"{c}_target"]

    # log-scale the count columns: 500 vs 5000 samples is a bigger jump
    # than 50000 vs 54500, and logs express that.
    for role in ("source", "target"):
        df[f"log_n_{role}"] = np.log10(df[f"n_samples_{role}"])
    df["delta_log_n"] = df["log_n_source"] - df["log_n_target"]

    # ---- the two outcome variables ---------------------------------------
    df["y_raw"] = df["transfer_auc"]
    df["y_norm"] = df["transfer_auc"] / df["auc_target_ceiling"]
    df["y_gap"] = df["auc_target_ceiling"] - df["transfer_auc"]

    df["is_diagonal"] = df.source == df.target

    out = RESULT_DIR / f"design_matrix{suffix}.csv"
    df.to_csv(out, index=False)
    n_off = int((~df.is_diagonal).sum())
    print(f"wrote {out}")
    print(f"  {len(df)} rows total, {n_off} off-diagonal (the modelling set)")
    print(f"  {df.source.nunique()} independent domains")
    print(f"\ny_raw : mean {df.loc[~df.is_diagonal,'y_raw'].mean():.3f} "
          f"sd {df.loc[~df.is_diagonal,'y_raw'].std():.3f}")
    print(f"y_norm: mean {df.loc[~df.is_diagonal,'y_norm'].mean():.3f} "
          f"sd {df.loc[~df.is_diagonal,'y_norm'].std():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    main(**vars(p.parse_args()))
