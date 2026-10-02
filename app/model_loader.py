import json
import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest_v1.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "standard_scaler_v1.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features_v1.json")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "models", "thresholds_v1.json")

# For these features only an INCREASE away from the population mean is anomalous:
# few transactions per hour, a long gap since the last transaction, or a long
# jump from home are all unremarkable. The rest are two-sided, where a value far
# below the mean is just as unusual.
ONE_SIDED_HIGH_ANOMALOUS = frozenset(
    {
        "txn_count_1h",
        "txn_count_24h",
        "time_since_last_txn_sec",
        "distance_from_home",
        "travel_speed_kmh",
    }
)


class ModelService:
    def __init__(self):
        # Load model artifacts
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        with open(FEATURES_PATH) as f:
            self.features = json.load(f)["features"]

        with open(THRESHOLDS_PATH) as f:
            thresholds = json.load(f)

        # Isolation Forest threshold config
        model_key = "isolation_forest"
        if model_key not in thresholds:
            raise ValueError(f"Threshold config for {model_key} not found")

        self.threshold = thresholds[model_key]["threshold_value"]
        self.threshold_percentile = thresholds[model_key]["percentile"]

        if len(self.features) != len(self.scaler.mean_):
            raise ValueError(
                f"Feature contract drift: {len(self.features)} features declared "
                f"but scaler was fitted on {len(self.scaler.mean_)}"
            )

    def _validate(self, feature_dict: dict) -> None:
        missing = set(self.features) - set(feature_dict.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

    def _ordered_frame(self, rows) -> pd.DataFrame:
        """Order and coerce features to match the training matrix exactly."""
        return pd.DataFrame(rows)[self.features].astype(float)

    def predict(self, feature_dict: dict) -> float:
        # Validate feature contract
        self._validate(feature_dict)
        return float(self.predict_batch(feature_dict)[0])

    def predict_batch(self, rows) -> np.ndarray:
        """Score one or many feature rows in a single vectorised call.

        `rows` may be a dict or a DataFrame. Isolation Forest's
        `decision_function` is lower-is-anomalous, so the score is negated here
        to give every model in this project a single convention:
        higher score == more anomalous.
        """
        frame = self._ordered_frame([rows] if isinstance(rows, dict) else rows)
        X_scaled = self.scaler.transform(frame)
        return -self.model.decision_function(X_scaled)

    def feature_contributions(self, feature_dict: dict, top_n: int = 3) -> list[dict]:
        """Rank features by how far they sit from the population the model saw.

        Isolation Forest is not natively additive, so this is NOT a Shapley
        attribution. It is a z-score distance against the training scaler's
        mean/scale, which is a faithful proxy for "which inputs dragged this
        transaction out of the normal region".
        """
        self._validate(feature_dict)

        mean = np.asarray(self.scaler.mean_, dtype=float)
        scale = np.asarray(self.scaler.scale_, dtype=float)

        contributions = []
        for i, name in enumerate(self.features):
            value = float(feature_dict[name])
            z = (value - mean[i]) / scale[i] if scale[i] else 0.0
            one_sided = name in ONE_SIDED_HIGH_ANOMALOUS
            relevance = max(z, 0.0) if one_sided else abs(z)
            contributions.append(
                {
                    "feature": name,
                    "value": round(value, 4),
                    "z_score": round(float(z), 4),
                    "direction": "above_normal" if z >= 0 else "below_normal",
                    "_relevance": relevance,
                }
            )

        contributions.sort(key=lambda c: c["_relevance"], reverse=True)
        top = contributions[:top_n]
        for item in top:
            item.pop("_relevance")
        return top
