import json
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest_v1.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "standard_scaler_v1.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features_v1.json")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "models", "thresholds_v1.json")


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

    def predict(self, feature_dict: dict) -> float:
        # Validate feature contract
        missing = set(self.features) - set(feature_dict.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Order features correctly
        X = pd.DataFrame([feature_dict])[self.features]
        X_scaled = self.scaler.transform(X)

        # Isolation Forest anomaly score
        score = -self.model.decision_function(X_scaled)[0]
        return float(score)
