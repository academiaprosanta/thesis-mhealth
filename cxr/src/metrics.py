"""Binary classification scores. Every domain shares the same task now."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, f1_score, brier_score_loss)


def compute_metrics(y_true, p_abnormal):
    """
    y_true      : (N,) 0 = Normal, 1 = Abnormal
    p_abnormal  : (N,) predicted probability of Abnormal
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(p_abnormal).ravel()
    if len(np.unique(y_true)) < 2:
        return dict(auc=np.nan, auprc=np.nan, balanced_acc=np.nan,
                    macro_f1=np.nan, brier=np.nan)
    pred = (p >= 0.5).astype(int)
    return dict(
        auc=float(roc_auc_score(y_true, p)),
        auprc=float(average_precision_score(y_true, p)),
        balanced_acc=float(balanced_accuracy_score(y_true, pred)),
        macro_f1=float(f1_score(y_true, pred, average="macro")),
        brier=float(brier_score_loss(y_true, p)),
    )
