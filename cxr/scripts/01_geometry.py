"""
STEP 1  (GPU, ~10 min)

Supervised geometry for each domain, on a shared frozen ImageNet backbone.
These are the LABEL-DEPENDENT features -- the comparison baseline for the
unsupervised set.

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

    rows = []
    for name in domain_names():
        loader = get_loader(name, "train", cap=MAX_EVAL)
        X, _, y = extract(neutral, head, loader, device, desc=name)
        row = {"dataset": name, **supervised_geometry(X, y)}
        rows.append(row)
        print(f"{name:22s} sil={row['silhouette']:+.4f} "
              f"fisher={row['fisher_ratio']:.4f} n={row['n_samples']} "
              f"abn={row['abnormal_rate']:.3f}")

    out = RESULT_DIR / "geometry.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
