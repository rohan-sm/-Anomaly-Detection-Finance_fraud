"""Evaluation utilities shared by the training pipeline and the comparison report.

Two distinct notions of ROC-AUC appear in this project and they are not
interchangeable:

* **Ranking ROC-AUC** - computed from the continuous anomaly score. This is what
  the notebooks report (0.5991 for Isolation Forest) and it is INVARIANT to the
  operating threshold, because re-thresholding a score does not change its order.
* **Threshold ROC-AUC** - computed from the binary flags at a given operating
  point. Much lower, and it changes with every threshold.

Only the ranking AUC is a meaningful measure of model quality here, so it is
reported in the sweep and the threshold AUC is not.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ranking_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    higher_is_anomalous: bool = False,
) -> float:
    """ROC-AUC from a continuous ranking score.

    Isolation Forest and One-Class SVM emit `decision_function`, where a LOWER
    value means MORE anomalous, so the score must be negated before scoring the
    ROC curve. Autoencoder emits reconstruction error, where HIGHER is more
    anomalous, so it is passed through unchanged.
    """
    oriented = np.asarray(scores, dtype=float)
    if not higher_is_anomalous:
        oriented = -oriented
    return float(roc_auc_score(y_true, oriented))


def evaluate_flags(y_true: np.ndarray, flags: np.ndarray) -> dict:
    """Precision / recall / F1 at a fixed operating point.

    Deliberately excludes ROC-AUC: see the module docstring.
    """
    (tn, fp), (fn, tp) = confusion_matrix(y_true, flags, labels=[0, 1])
    return {
        "anomaly_rate": float(np.mean(flags)),
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
        "f1": float(f1_score(y_true, flags, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def contamination_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    rates: list[float],
    model_name: str,
    higher_is_anomalous: bool = False,
) -> pd.DataFrame:
    """Evaluate one model across several target flag rates.

    Thresholds are taken as empirical percentiles of the score distribution so
    every model is compared at the same operational alert volume. The ranking
    ROC-AUC is computed once and repeated, since re-thresholding cannot change
    the score ordering.
    """
    auc = ranking_auc(y_true, scores, higher_is_anomalous=higher_is_anomalous)

    rows = []
    for rate in rates:
        if higher_is_anomalous:
            threshold = float(np.percentile(scores, 100 * (1 - rate)))
            flags = scores > threshold
        else:
            threshold = float(np.percentile(scores, 100 * rate))
            flags = scores < threshold

        metrics = evaluate_flags(y_true, flags)
        rows.append(
            {
                "model": model_name,
                "anomaly_rate": round(rate, 4),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)
