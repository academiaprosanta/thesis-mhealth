"""
Two transfer protocols on a shared binary task.

ZERO-SHOT (primary): apply the source model unchanged -- backbone AND its
  trained classifier -- to the target. Nothing is fitted. Possible only
  because every domain now shares Normal-vs-Abnormal.

LINEAR PROBE (secondary): discard the source classifier, fit a fresh logistic
  regression on target features. Needs target labels. Kept so we can compare
  the two and test why source-side effects vanished in the MedMNIST run.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .metrics import compute_metrics


def zero_shot_score(P_test, y_test):
    """No fitting at all. P_test already came from the source's own head."""
    return compute_metrics(y_test, P_test[:, 1])


def linear_probe_score(X_train, y_train, X_test, y_test):
    """Fresh classifier on frozen features. Scaler fitted on train only."""
    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(X_train), y_train)
    p = clf.predict_proba(sc.transform(X_test))
    col = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    return compute_metrics(y_test, p[:, col])
