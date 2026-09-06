"""
Turning model predictions into scores.

AUC is the primary outcome, but computing the others costs nothing extra
since they all come from the same probability matrix.
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, f1_score)


def expand_proba(proba, present_classes, n_cls):
    """
    sklearn's predict_proba returns one column per class it SAW during fit.
    If a class was missing from the training subsample, the column is absent
    and the column order no longer matches label 0,1,2,...

    This pads it back to full width so probs[:, c] always means "class c".
    """
    full = np.zeros((proba.shape[0], n_cls), dtype=float)
    full[:, np.asarray(present_classes, dtype=int)] = proba
    return full


def compute_metrics(y_true, probs):
    """
    y_true : (N,)   integer labels
    probs  : (N, C) predicted probabilities, column c = P(class c)
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    n_cls = probs.shape[1]
    out = {}

    if n_cls == 2:
        # Binary: AUC is the probability that a random positive scores
        # higher than a random negative. 0.5 = coin flip.
        if len(np.unique(y_true)) < 2:
            out["auc"] = np.nan
            out["auprc"] = np.nan
        else:
            out["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            out["auprc"] = float(average_precision_score(y_true, probs[:, 1]))
    else:
        # Multi-class: one-vs-rest, averaged. Skip classes absent from the
        # test set, otherwise sklearn raises.
        aucs, aps = [], []
        for c in np.unique(y_true):
            binary = (y_true == c).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                continue
            aucs.append(roc_auc_score(binary, probs[:, c]))
            aps.append(average_precision_score(binary, probs[:, c]))
        out["auc"] = float(np.mean(aucs)) if aucs else np.nan
        out["auprc"] = float(np.mean(aps)) if aps else np.nan

    pred = probs.argmax(axis=1)
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, pred))
    out["macro_f1"] = float(f1_score(y_true, pred, average="macro"))
    return out
