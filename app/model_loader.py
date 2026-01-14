import json
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest_v1.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "standard_scaler_v1.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features_v1.json")
class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH) as f:
            self.features = json.load(f)["features"]

    def predict(self, input_features: dict):
        # Validate feature presence
        missing = set(self.features) - set(input_features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = pd.DataFrame([input_features])[self.features]
        X_scaled = self.scaler.transform(X)

        score = -self.model.decision_function(X_scaled)[0]
        return score
