"""
Settings. Domains are DISCOVERED, not hardcoded -- see src/registry.py.

Order of resolution for the domain list:
  1. results/domains_resolved.json  if it exists  (written by 00_check_domains)
  2. live discovery of /kaggle/input
  3. empty
"""
import json
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/kaggle/working") if ON_KAGGLE else REPO_ROOT

CKPT_DIR = SCRATCH / "checkpoints"
RESULT_DIR = REPO_ROOT / "results"
for _d in (CKPT_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

KAGGLE_INPUT = "/kaggle/input"
RESOLVED = RESULT_DIR / "domains_resolved.json"

# Domains you want to exclude after seeing the checker output.
# Montgomery is only ~138 images -- usually too small to fine-tune on.
EXCLUDE = {"montgomery"}

# Hard floor: a domain with fewer than this many training images is dropped.
MIN_TRAIN_IMAGES = 250

# ---------------------------------------------------------------- knobs
IMG_SIZE = 224          # drop to 160 if the run is too slow
BATCH_SIZE = 64
EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 0
NUM_WORKERS = 2
VAL_FRACTION = 0.15
TEST_FRACTION = 0.20

MAX_TRAIN = 5_000       # cap so source quality is not confounded by size
MAX_EVAL = 2_000
GEOM_SAMPLE = 2_000     # subsample for O(n^2) metrics
MMD_SAMPLE = 800

_CACHE = None


def _to_paths(spec):
    for k in ("normal_dirs", "abnormal_dirs", "dirs", "image_dirs"):
        if k in spec:
            spec[k] = [Path(p) for p in spec[k]]
    for k in ("csv", "projections", "reports"):
        if k in spec:
            spec[k] = Path(spec[k])
    if "partition" in spec and spec["partition"] is not None:
        spec["partition"] = tuple(spec["partition"])
    return spec


def load_domains(force_discover=False):
    global _CACHE
    if _CACHE is not None and not force_discover:
        return _CACHE
    specs = []
    if RESOLVED.exists() and not force_discover:
        specs = [_to_paths(s) for s in json.loads(RESOLVED.read_text())]
    else:
        from .registry import discover
        specs = discover(KAGGLE_INPUT, verbose=False)
    specs = [s for s in specs if s["name"] not in EXCLUDE]
    _CACHE = specs
    return specs


def save_domains(specs):
    def enc(o):
        return str(o) if isinstance(o, Path) else o
    RESOLVED.write_text(json.dumps(specs, indent=1, default=enc))


def domain_names():
    return [d["name"] for d in load_domains()]


def get_domain(name):
    for d in load_domains():
        if d["name"] == name:
            return d
    raise KeyError(f"unknown domain '{name}'. Known: {domain_names()}")
