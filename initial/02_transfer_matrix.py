"""
STEP 2  (GPU, ~45-70 min for the full 11 datasets)

Train one model per dataset, then evaluate every model on every dataset.
This produces the ground truth you are trying to predict.

  diagonal (source == target)  -> the CEILING: best achievable on that target
  off-diagonal                 -> the transfer result you want to predict

Output: results/transfer_matrix.csv -- one row per (source, target) pair.

Run:  python scripts/02_transfer_matrix.py --smoke   # DO THIS FIRST
      python scripts/02_transfer_matrix.py

Safe to interrupt and rerun: finished models are cached as checkpoints and
completed pairs are appended to the CSV as they finish.
"""

# --- make "import src..." work no matter where you run this from ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import torch

from src.config import DATASETS, SMOKE_DATASETS, RESULT_DIR
from src.train import train_source_model
from src.models import as_backbone
from src.transfer import linear_probe


def main(smoke=False, epochs=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = SMOKE_DATASETS if smoke else DATASETS
    out = RESULT_DIR / ("transfer_matrix_smoke.csv" if smoke else "transfer_matrix.csv")

    # resume support: skip pairs already recorded
    done = set()
    rows = []
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(zip(prev["source"], prev["target"]))
        print(f"resuming, {len(done)} pairs already done")

    for source in datasets:
        kw = {"epochs": epochs} if epochs else {}
        model, in_domain = train_source_model(source, device=device, **kw)
        backbone = as_backbone(model)

        for target in datasets:
            if (source, target) in done:
                continue
            m = linear_probe(backbone, target, device=device)
            rows.append({
                "source": source,
                "target": target,
                "transfer_auc": m["auc"],
                "transfer_auprc": m["auprc"],
                "transfer_balanced_acc": m["balanced_acc"],
                "transfer_macro_f1": m["macro_f1"],
                "auc_source_on_source": in_domain["auc"],
            })
            print(f"  {source:15s} -> {target:15s}  AUC {m['auc']:.4f}")
            # write after every pair, so a crash costs you one pair
            pd.DataFrame(rows).to_csv(out, index=False)

        del model, backbone
        torch.cuda.empty_cache()

    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    main(**vars(p.parse_args()))
