# Transferability prediction — initial pipeline

Predict how well a model trained on dataset A will perform on dataset B,
using only cheap summary statistics of the two datasets.

## What runs where

| Script | Where | Time | Produces |
|---|---|---|---|
| `01_geometry.py` | Kaggle GPU | ~5 min | `results/geometry.csv` |
| `02_transfer_matrix.py` | Kaggle GPU | ~60 min | `results/transfer_matrix.csv` |
| `03_build_design_matrix.py` | laptop | instant | `results/design_matrix.csv` |
| `04_regression.py` | laptop | ~30 s | `results/regression_report.txt` |

## First time

```bash
git clone https://github.com/YOURNAME/transferability.git
cd transferability
pip install -r requirements.txt
```

## The loop

1. Edit code on your laptop in `src/`
2. Commit and push
3. On Kaggle: open `notebooks/kaggle_runner.ipynb`, rerun the clone cell, run the scripts
4. Push results from Kaggle
5. `git pull` on your laptop, run `04_regression.py`

## Always smoke-test first

```bash
python scripts/01_geometry.py --smoke
python scripts/02_transfer_matrix.py --smoke --epochs 2
python scripts/03_build_design_matrix.py --smoke
python scripts/04_regression.py --smoke
```

Three datasets instead of eleven. Finishes in minutes. Catches bugs before
they cost you an hour of GPU time.

## Notes

- `chestmnist` is excluded (multi-label, breaks the single-label pipeline)
- Transfer = **linear probe**: freeze the source backbone, fit a logistic
  regression on target features. Zero-shot is undefined here because
  MedMNIST datasets have different numbers of classes.
- Evaluation is **leave-one-domain-out**, never a random split.
