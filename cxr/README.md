# Chest X-ray transferability — unsupervised prediction

Predict how well a chest X-ray model will transfer to a new hospital's data
**without training on that data and without labelling it.**

## What changed from the MedMNIST run

| | MedMNIST | This |
|---|---|---|
| Modality | 7 different ones | chest X-ray only |
| Domain shift | cross-modality | cross-site / country / view |
| Label space | 2–11 classes, all different | **shared binary: Normal vs Abnormal** |
| Primary protocol | linear probe | **zero-shot** (classifier reused intact) |
| Primary features | supervised geometry | **unsupervised, label-free** |

The shared binary label is the key change. It makes zero-shot possible, which
means the source classifier transfers intact — so source-side effects can
finally show up.

## Pipeline

| Script | Where | Time | Output |
|---|---|---|---|
| `00_check_domains.py` | anywhere | seconds | `domain_summary.csv` |
| `01_geometry.py` | Kaggle GPU | ~10 min | `geometry.csv` |
| `02_transfer_matrix.py` | Kaggle GPU | ~3–4 h | `transfer_matrix.csv` |
| `03_build_design_matrix.py` | laptop | instant | `design_matrix_*.csv` |
| `04_regression.py` | laptop | ~1 min | `regression_*.txt` |

## Setup

Attach datasets on Kaggle, then let the code discover them:

```bash
python scripts/00_check_domains.py     # must pass before anything else
```

## The three feature sets

- **unsupervised** — no target labels. Source-model entropy and confidence,
  pseudo-label geometry, k-means structure, spectral shape, MMD / CORAL /
  Frechet / A-distance. This is the deployable set.
- **supervised** — Silhouette, Fisher ratio, Davies-Bouldin and friends.
  Needs target labels. The comparison baseline.
- **combined** — both, as an upper bound.

`04_regression.py` fits all three and prints them side by side.
