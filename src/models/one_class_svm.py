import joblib
import numpy as np


def load_one_class_svm(model_path: str):
    return joblib.load(model_path)


def score_transactions(model, X_scaled: np.ndarray) -> np.ndarray:
    # Compute anomaly scores.
    # Lower (more negative) = more anomalous.
    return model.decision_function(X_scaled)


def flag_anomalies(scores: np.ndarray, threshold: float) -> np.ndarray:
    # Convert scores to anomaly flags.
    return scores < threshold


def run_one_class_svm(
    model,
    X_scaled: np.ndarray,
    threshold: float
):
    
    # Full One-Class SVM inference pipeline.
    scores = score_transactions(model, X_scaled)
    flags = flag_anomalies(scores, threshold)
    return scores, flags
