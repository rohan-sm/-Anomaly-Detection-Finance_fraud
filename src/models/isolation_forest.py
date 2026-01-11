import joblib
import numpy as np


def load_isolation_forest(model_path: str):
    return joblib.load(model_path)

def score_transactions(model, X_scaled: np.ndarray) -> np.ndarray:
    return model.decision_function(X_scaled)  
# Compute anomaly scores for transactions. Lower score = more anomalous.

def flag_anomalies(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores < threshold

def run_isolation_forest(
    model,
    X_scaled: np.ndarray,
    threshold: float
):
    
    """
    Full IF inference pipeline.
    Returns anomaly scores and flags.
    """
    scores = score_transactions(model, X_scaled)
    flags = flag_anomalies(scores, threshold)
    return scores, flags
