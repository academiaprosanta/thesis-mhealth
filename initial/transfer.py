"""
The linear probe: how well does a source model's feature space serve a
different dataset?

Recipe for one (source, target) pair:
  1. take the source model, throw away its classifier, freeze the rest
  2. push the TARGET's training images through it -> feature vectors
  3. fit a plain logistic regression on those features
  4. score it on the TARGET's test images

Nothing about the backbone changes. We only ask: are the source model's
features arranged such that a straight line can separate the target's classes?
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import MAX_TRAIN, MAX_EVAL, SEED
from .data import get_loader, n_classes
from .features import extract_features
from .metrics import compute_metrics, expand_proba


def linear_probe(backbone, target, device="cuda"):
    tr_loader = get_loader(target, "train", cap=MAX_TRAIN)
    te_loader = get_loader(target, "test", cap=MAX_EVAL)

    Xtr, ytr = extract_features(backbone, tr_loader, device, desc=f"{target}/train")
    Xte, yte = extract_features(backbone, te_loader, device, desc=f"{target}/test")

    # Logistic regression converges much faster on standardised inputs.
    # Fit the scaler on TRAIN only -- fitting it on test would be leakage.
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)
    clf.fit(Xtr, ytr)

    proba = expand_proba(clf.predict_proba(Xte), clf.classes_, n_classes(target))
    return compute_metrics(yte, proba)
