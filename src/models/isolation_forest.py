import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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


# ---------------- TRAINING PIPELINE ----------------

N_ESTIMATORS = 300
CONTAMINATION = 0.01
RANDOM_STATE = 42


def train_isolation_forest(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    model_path: str,
    scaler_path: str,
    features_path: str,
    thresholds_path: str,
    features: list[str],
    contamination: float = CONTAMINATION,
):
    """Fit scaler + Isolation Forest and persist the versioned artifact bundle.

    The threshold is stored on the raw `decision_function` scale (lower = more
    anomalous) to stay compatible with `app/model_loader.py`, which negates the
    score at serving time.
    """
    missing = [f for f in features if f not in features_df.columns]
    if missing:
        raise ValueError(f"Missing features in training data: {missing}")

    X = features_df[features]
    if len(X) != len(labels):
        raise ValueError(
            f"Row misalignment: {len(X)} feature rows vs {len(labels)} labels"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    scores = score_transactions(model, X_scaled)
    # `decision_function` is lower-is-anomalous, so the operating threshold is the
    # `contamination`-th percentile: exactly `contamination` of rows fall below it.
    threshold = float(np.percentile(scores, 100 * contamination))

    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)

    with open(features_path, "w") as f:
        json.dump({"features": features}, f, indent=2)

    with open(thresholds_path) as f:
        thresholds = json.load(f)
    thresholds["isolation_forest"] = {
        "threshold_type": "percentile",
        "percentile": 100 * contamination,
        "threshold_value": threshold,
    }
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    return model, scaler, scores, threshold

