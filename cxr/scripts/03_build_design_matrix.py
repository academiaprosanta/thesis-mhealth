"""
STEP 3  (laptop, instant)

Join the transfer matrix with the supervised geometry and compute the three
outcome variables. Produces one design matrix per protocol.

Output: results/design_matrix_zeroshot.csv
        results/design_matrix_probe.csv

Run:  python scripts/03_build_design_matrix.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import RESULT_DIR
from src.design import add_supervised_columns, add_targets, FEATURE_SETS


def main():
    tm = pd.read_csv(RESULT_DIR / "transfer_matrix.csv")
    geo = pd.read_csv(RESULT_DIR / "geometry.csv").set_index("dataset")

    for protocol in ("zeroshot", "probe"):
        if f"{protocol}_auc" not in tm.columns:
            continue
        df = add_supervised_columns(tm.copy(), geo)
        df = add_targets(df, protocol)
        out = RESULT_DIR / f"design_matrix_{protocol}.csv"
        df.to_csv(out, index=False)

        off = df[~df.is_diagonal]
        print(f"\n=== {protocol} ===")
        print(f"  {len(df)} rows, {len(off)} off-diagonal, "
              f"{df.source.nunique()} domains")
        print(f"  y_raw  mean {off.y_raw.mean():.3f}  sd {off.y_raw.std():.3f}")
        print(f"  y_norm mean {off.y_norm.mean():.3f}  sd {off.y_norm.std():.3f}")
        for k, feats in FEATURE_SETS.items():
            have = [f for f in feats if f in df.columns]
            print(f"  {k:14s} {len(have)}/{len(feats)} features present, "
                  f"{len(off)/max(len(have),1):.1f} rows per feature")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
