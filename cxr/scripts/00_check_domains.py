"""
STEP 0  -- ALWAYS RUN THIS FIRST

Scans /kaggle/input, works out which chest X-ray datasets you attached,
builds the domain list, and verifies each one actually loads.

Writes results/domains_resolved.json, which every later script reads. So
discovery happens ONCE and the rest of the pipeline is deterministic.

Run:  python scripts/00_check_domains.py
      python scripts/00_check_domains.py --input /some/other/path
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd

from src import config
from src.config import RESULT_DIR, EXCLUDE, MIN_TRAIN_IMAGES, save_domains
from src.registry import discover


def main(input_root="/kaggle/input"):
    print(f"scanning {input_root}\n")
    specs = discover(input_root, verbose=True)
    if not specs:
        sys.exit("\nNo recognised chest X-ray datasets found.\n"
                 "Attach one of the supported datasets (see src/registry.py) "
                 "or add a recipe for yours.")

    dropped = [s["name"] for s in specs if s["name"] in EXCLUDE]
    specs = [s for s in specs if s["name"] not in EXCLUDE]
    if dropped:
        print(f"\nexcluded by config.EXCLUDE: {dropped}")

    save_domains(specs)
    config._CACHE = None                     # force reload from the json
    from src.domains import domain_summary

    rows, bad, small = [], [], []
    print()
    for s in specs:
        try:
            r = domain_summary(s["name"])
        except Exception as e:
            bad.append((s["name"], str(e)[:150]))
            print(f"FAIL {s['name']}: {str(e)[:150]}")
            continue
        if r["n_train"] < MIN_TRAIN_IMAGES:
            small.append(s["name"])
            print(f"SMALL {s['name']}: only {r['n_train']} train images -- dropping")
            continue
        if r["n_normal"] == 0 or r["n_abnormal"] == 0:
            bad.append((s["name"], "only one class present"))
            print(f"FAIL {s['name']}: only one class present")
            continue
        rows.append(r)
        print(f"OK   {s['name']:16s} n={r['n_total']:6d}  "
              f"train={r['n_train']:5d}  abnormal={r['abnormal_rate']:.3f}")

    keep = {r["domain"] for r in rows}
    save_domains([s for s in specs if s["name"] in keep])
    config._CACHE = None

    if not rows:
        sys.exit("\nNo usable domains.")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))
    df.to_csv(RESULT_DIR / "domain_summary.csv", index=False)

    k = len(df)
    print(f"\n{k} usable domains -> {k*(k-1)} ordered pairs")
    if k < 5:
        print("STOP: leave-one-domain-out needs at least 5 domains. Attach more.")
    elif k < 8:
        print("Workable. 8+ would be noticeably better -- attach another dataset.")
    else:
        print("Good sample size.")

    skew = df[(df.abnormal_rate < 0.12) | (df.abnormal_rate > 0.88)]
    if len(skew):
        print(f"NOTE severe imbalance (class weighting will handle it): {list(skew.domain)}")

    print(f"\nwrote {RESULT_DIR/'domain_summary.csv'}")
    print(f"wrote {RESULT_DIR/'domains_resolved.json'}  <- commit this")
    if bad:
        print("\nfailed:")
        for n, e in bad:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", dest="input_root", default="/kaggle/input")
    main(**vars(p.parse_args()))
