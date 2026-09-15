"""
STEP 2  (GPU, ~3-4 h for 8 domains)  -- the expensive one

For every ordered pair:
  * ZERO-SHOT  AUC   (source model applied unchanged -- the primary outcome)
  * LINEAR PROBE AUC (fresh head on frozen features -- secondary)
  * all UNSUPERVISED pair features, computed from the target's UNLABELLED
    training images plus the source's own images

Nothing here uses target labels except to SCORE. The unsupervised features
never see them.

Output: results/transfer_matrix.csv  (one row per pair, features included)

Resumable: writes after every pair, skips pairs already recorded.
Run:  python scripts/02_transfer_matrix.py [--epochs N] [--limit N]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import torch

from src.config import RESULT_DIR, MAX_TRAIN, MAX_EVAL, domain_names
from src.domains import get_loader
from src.models import split_model
from src.features import extract
from src.train import train_source_model
from src.transfer import zero_shot_score, linear_probe_score
from src.unsupervised import unsupervised_pair_features


def main(epochs=None, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = domain_names()[:limit] if limit else domain_names()
    out = RESULT_DIR / "transfer_matrix.csv"

    rows, done = [], set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(zip(prev["source"], prev["target"]))
        print(f"resuming: {len(done)} pairs already done")

    for source in names:
        kw = {"epochs": epochs} if epochs else {}
        model, in_domain = train_source_model(source, device=device, **kw)
        bb, head = split_model(model)

        # the source's own images, in its own feature space -- needed for
        # every source-to-target distance
        Xs, _, _ = extract(bb, head, get_loader(source, "train", cap=MAX_EVAL),
                           device, desc=f"{source}/self")

        for target in names:
            if (source, target) in done:
                continue
            try:
                # target TRAIN split = the unlabelled pool a hospital would have
                Xt_tr, Pt_tr, yt_tr = extract(
                    bb, head, get_loader(target, "train", cap=MAX_TRAIN),
                    device, desc=f"{source}->{target}/train")
                Xt_te, Pt_te, yt_te = extract(
                    bb, head, get_loader(target, "test", cap=MAX_EVAL),
                    device, desc=f"{source}->{target}/test")

                zs = zero_shot_score(Pt_te, yt_te)
                lp = linear_probe_score(Xt_tr, yt_tr, Xt_te, yt_te)
                unsup = unsupervised_pair_features(Xs, Xt_tr, Pt_tr)

                rows.append({
                    "source": source, "target": target,
                    "zeroshot_auc": zs["auc"], "zeroshot_auprc": zs["auprc"],
                    "zeroshot_balanced_acc": zs["balanced_acc"],
                    "zeroshot_brier": zs["brier"],
                    "probe_auc": lp["auc"], "probe_auprc": lp["auprc"],
                    "probe_balanced_acc": lp["balanced_acc"],
                    "auc_source_on_source": in_domain["auc"],
                    "pair_failed": 0,
                    **unsup,
                })
                print(f"  {source:20s} -> {target:20s} "
                      f"zeroshot {zs['auc']:.4f}  probe {lp['auc']:.4f}")

            except Exception as e:
                # NEVER let one bad pair abort a multi-hour run. Record the
                # failure and move on; 04_regression.py imputes the NaNs.
                import traceback
                print(f"  !! {source} -> {target} FAILED: "
                      f"{type(e).__name__}: {str(e)[:160]}")
                traceback.print_exc()
                rows.append({"source": source, "target": target,
                             "pair_failed": 1,
                             "auc_source_on_source": in_domain["auc"]})

            # write after every pair, so a crash costs you one pair
            pd.DataFrame(rows).to_csv(out, index=False)

        del model, bb, head, Xs
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    n_fail = int(df.get("pair_failed", pd.Series(dtype=int)).fillna(0).sum())
    print(f"\nwrote {out}  ({len(df)} rows, {n_fail} failed)")
    if n_fail:
        print("failed pairs:")
        print(df[df.pair_failed == 1][["source", "target"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N domains (for a quick test)")
    main(**vars(p.parse_args()))
