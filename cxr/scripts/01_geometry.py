"""
STEP 1  (GPU, ~10-15 min)

Supervised geometry for each domain, on a shared frozen ImageNet backbone.
These are the LABEL-DEPENDENT features -- the comparison baseline for the
unsupervised set.

NOTE on n_samples: the metrics are computed on a capped subsample (MAX_EVAL)
because Silhouette is O(n^2). But the SIZE feature must record the TRUE
dataset size, not the sample size -- otherwise log_n is near-constant and
three features are dead. We read the true size from domain_summary.csv,
written by 00_check_domains.py.

Output: results/geometry.csv
Run:  python scripts/01_geometry.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.config import RESULT_DIR, MAX_EVAL, domain_names
from src.domains import get_loader
from src.models import build_model, split_model
from src.features import extract
from src.geometry import supervised_geometry


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neutral, head = split_model(build_model(pretrained=True))
    print(f"device={device}")

    summ_path = RESULT_DIR / "domain_summary.csv"
    if not summ_path.exists():
        sys.exit("missing results/domain_summary.csv -- run 00_check_domains.py first")
    summ = pd.read_csv(summ_path).set_index("domain")

    rows = []
    for name in domain_names():
        loader = get_loader(name, "train", cap=MAX_EVAL)
        X, _, y = extract(neutral, head, loader, device, desc=name)
        row = {"dataset": name, **supervised_geometry(X, y)}

        # what we measured on vs how big the domain actually is
        row["n_sampled"] = row.pop("n_samples")
        row["n_samples"] = int(summ.loc[name, "n_train"])
        row["n_total"] = int(summ.loc[name, "n_total"])

        rows.append(row)
        print(f"{name:16s} sil={row['silhouette']:+.4f} "
              f"fisher={row['fisher_ratio']:.4f} "
              f"abn={row['abnormal_rate']:.3f} "
              f"n_true={row['n_samples']:6d} (measured on {row['n_sampled']})")

    out = RESULT_DIR / "geometry.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    import numpy as np
    ln = np.log10(pd.DataFrame(rows)["n_samples"])
    print(f"log_n spread: {ln.min():.2f} to {ln.max():.2f} "
          f"({ln.nunique()} distinct values across {len(rows)} domains)")


if __name__ == "__main__":
    main()
