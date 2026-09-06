"""
Every knob in one place. Change things HERE, never inside a script.

Why: when a result looks wrong in month three, you want one file to check,
not fifteen scripts each with their own hardcoded learning rate.
"""
from pathlib import Path

# --- where files go ---------------------------------------------------
# Kaggle wipes everything except /kaggle/working when a session ends.
ON_KAGGLE = Path("/kaggle/working").exists()

REPO_ROOT = Path(__file__).resolve().parents[1]

# Big scratch files (downloaded images, model weights). NOT in git.
SCRATCH = Path("/kaggle/working") if ON_KAGGLE else REPO_ROOT
CACHE_DIR = SCRATCH / "cache"
CKPT_DIR = SCRATCH / "checkpoints"

# Small CSVs. These DO go in git -- they are your actual results.
RESULT_DIR = REPO_ROOT / "results"

for _d in (CACHE_DIR, CKPT_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- which datasets -----------------------------------------------------
# 11 MedMNIST datasets => 11 * 10 = 110 ordered source->target pairs.
#
# chestmnist is deliberately EXCLUDED: it is multi-label (14 independent
# binary labels per image), which does not fit a single-label pipeline.
# Add it later as a special case if you want it.
DATASETS = [
    "pathmnist",      # colon pathology,        9 classes, RGB
    "dermamnist",     # skin lesions,           7 classes, RGB
    "octmnist",       # retinal OCT,            4 classes, grayscale
    "pneumoniamnist", # chest x-ray,            2 classes, grayscale
    "retinamnist",    # fundus,                 5 classes, RGB
    "breastmnist",    # breast ultrasound,      2 classes, grayscale
    "bloodmnist",     # blood cells,            8 classes, RGB
    "tissuemnist",    # kidney cortex,          8 classes, grayscale
    "organamnist",    # abdominal CT, axial,   11 classes, grayscale
    "organcmnist",    # abdominal CT, coronal, 11 classes, grayscale
    "organsmnist",    # abdominal CT, sagittal,11 classes, grayscale
]

# A 3-dataset subset for testing that your code runs at all.
# Use this first. It finishes in ~3 minutes instead of ~1 hour.
SMOKE_DATASETS = ["pneumoniamnist", "breastmnist", "bloodmnist"]

# --- hyperparameters ----------------------------------------------------
IMG_SIZE = 64          # MedMNIST ships at 28x28; we upsample for ResNet
BATCH_SIZE = 256
EPOCHS = 5             # enough for a first pass; raise to 15 later
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0
NUM_WORKERS = 2

# --- caps (keep the first run fast) -------------------------------------
MAX_TRAIN = 10_000     # subsample big training sets
MAX_EVAL = 5_000       # subsample big test sets
SILHOUETTE_SAMPLE = 3_000   # silhouette is O(n^2) -- MUST subsample
