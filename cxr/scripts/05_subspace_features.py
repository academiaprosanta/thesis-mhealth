"""
STEP 5  (GPU, ~1.5-2 h)  -- adds subspace-geometry features to an existing run

Does NOT retrain anything. It reuses the checkpoints from step 2 and adds four
families of columns to results/transfer_matrix.csv:

  fsa_*        feature-subspace principal angles  (Baseline 3, label-free)
  gsa_ent_*    GSA with an entropy loss           (label-free)
  gsa_pse_*    GSA with pseudo-label loss         (label-free)
  gsa_ora_*    GSA with true target labels        (DIAGNOSTIC ONLY - uses labels)

Put this in scripts/, and put subspace.py and gradients.py in src/.

Run:  python scripts/05_subspace_features.py
      python scripts/05_subspace_features.py --rank 16 --layer layer3.1.conv2
      python scripts/05_subspace_features.py --limit 3     # quick test
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
import torch

from src.config import RESULT_DIR, MAX_EVAL, domain_names
from src.domains import get_loader
from src.models import build_model, split_model
from src.features import extract
from src.subspace import feature_subspace_metrics, gsa
from src.gradients import layer_gradient, source_basis
from src.config import CKPT_DIR


def main(rank=32, layer="layer4.1.conv2", max_batches=20, limit=None,
         include_oracle=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = domain_names()[:limit] if limit else domain_names()

    tm_path = RESULT_DIR / "transfer_matrix.csv"
    tm = pd.read_csv(tm_path)
    tm.to_csv(RESULT_DIR / "transfer_matrix_backup.csv", index=False)
    print(f"loaded {len(tm)} rows; backup written")

    out_path = RESULT_DIR / "subspace_features.csv"
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.source, prev.target))
        print(f"resuming: {len(done)} pairs already done")

    for source in names:
        ckpt = CKPT_DIR / f"{source}.pt"
        if not ckpt.exists():
            print(f"!! no checkpoint for {source}, skipping")
            continue
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        bb, head = split_model(model)

        # ---- the source-learned basis U, and the source's own features ----
        src_loader = get_loader(source, "train", cap=MAX_EVAL)
        U, Gs = source_basis(model, src_loader, layer, rank, device,
                             desc=f"{source}/basis")
        Xs, _, _ = extract(bb, head, get_loader(source, "train", cap=MAX_EVAL),
                           device, desc=f"{source}/feat")
        print(f"[{source}] basis U: {U.shape},  gradient G_s: {Gs.shape}")

        for target in names:
            if (source, target) in done:
                continue
            rec = {"source": source, "target": target}
            try:
                # ---- feature subspaces (Baseline 3) -----------------------
                Xt, _, _ = extract(bb, head,
                                   get_loader(target, "train", cap=MAX_EVAL),
                                   device, desc=f"{source}->{target}/feat")
                rec.update(feature_subspace_metrics(Xs, Xt, r=rank, prefix="fsa_"))

                # ---- GSA on gradient subspaces ----------------------------
                variants = [("gsa_ent_", "entropy"), ("gsa_pse_", "pseudo")]
                if include_oracle:
                    variants.append(("gsa_ora_", "oracle"))
                for pre, lossname in variants:
                    Gt = layer_gradient(model,
                                        get_loader(target, "train", cap=MAX_EVAL),
                                        layer, lossname, device, max_batches,
                                        desc=f"{source}->{target}/{lossname}")
                    rec.update(gsa(U, Gt, r=rank, prefix=pre))

                print(f"  {source:16s} -> {target:16s}  "
                      f"fsa={rec['fsa_pa_similarity']:.4f}  "
                      f"gsa_ent={rec['gsa_ent_pa_similarity']:.4f}  "
                      f"gsa_pse={rec['gsa_pse_pa_similarity']:.4f}"
                      + (f"  gsa_ora={rec['gsa_ora_pa_similarity']:.4f}"
                         if include_oracle else ""))
                rec["subspace_failed"] = 0
            except Exception as e:
                import traceback
                print(f"  !! {source} -> {target} FAILED: {type(e).__name__}: {str(e)[:140]}")
                traceback.print_exc()
                rec["subspace_failed"] = 1

            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_path, index=False)

        del model, bb, head, Xs, U
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- merge into the transfer matrix ----------------------------------
    sub = pd.DataFrame(rows)
    drop = [c for c in sub.columns if c in tm.columns and c not in ("source", "target")]
    tm = tm.drop(columns=drop, errors="ignore").merge(sub, on=["source", "target"], how="left")
    tm.to_csv(tm_path, index=False)
    print(f"\nwrote {out_path}  ({len(sub)} rows)")
    print(f"merged into {tm_path}  ({len(tm.columns)} columns)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--layer", default="layer4.1.conv2")
    p.add_argument("--max-batches", dest="max_batches", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-oracle", dest="include_oracle", action="store_false")
    main(**vars(p.parse_args()))
