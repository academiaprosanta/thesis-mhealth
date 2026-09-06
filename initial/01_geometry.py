"""
STEP 1  (GPU, ~5 min)

Describe each dataset with a handful of numbers, using a single NEUTRAL
encoder -- plain ImageNet ResNet-18, not fine-tuned on anything.

Why neutral: if you used each source's own model, the numbers would not be
comparable across datasets. This matches what the statistical report did
(it used one fixed CheXNet for all five domains).

Output: results/geometry.csv  -- one row per dataset.

Run:  python scripts/01_geometry.py            # all 11 datasets
      python scripts/01_geometry.py --smoke    # 3 datasets, ~1 min
"""

# --- make "import src..." work no matter where you run this from ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import torch

from src.config import DATASETS, SMOKE_DATASETS, RESULT_DIR, MAX_EVAL
from src.data import get_loader
from src.models import build_model, as_backbone
from src.features import extract_features
from src.geometry import geometry_metrics


def main(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = SMOKE_DATASETS if smoke else DATASETS
    print(f"device={device}  datasets={len(datasets)}")

    # One frozen ImageNet backbone, reused for every dataset.
    neutral = as_backbone(build_model(n_out=2, pretrained=True))

    rows = []
    for name in datasets:
        loader = get_loader(name, "train", cap=MAX_EVAL)
        X, y = extract_features(neutral, loader, device, desc=name)
        row = {"dataset": name, **geometry_metrics(X, y)}
        rows.append(row)
        print(f"{name:16s} silhouette={row['silhouette']:+.4f} "
              f"fisher={row['fisher_ratio']:.4f} n={row['n_samples']}")

    out = RESULT_DIR / ("geometry_smoke.csv" if smoke else "geometry.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    main(**vars(p.parse_args()))
