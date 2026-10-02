import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def model_service():
    from app.model_loader import ModelService

    return ModelService()


@pytest.fixture(scope="session")
def contract():
    """The frozen model contract: feature order and per-model thresholds.

    Reading these from the versioned JSON artifacts (rather than hardcoding)
    means a contract change surfaces here as a test failure.
    """
    from src.utils.config import MODELS_DIR

    with open(MODELS_DIR / "model_features_v1.json") as f:
        features = json.load(f)["features"]

    with open(MODELS_DIR / "thresholds_v1.json") as f:
        thresholds = json.load(f)

    return {"features": features, "thresholds": thresholds}


@pytest.fixture(scope="session")
def scaler():
    import joblib

    from src.utils.config import MODELS_DIR

    return joblib.load(MODELS_DIR / "standard_scaler_v1.pkl")


@pytest.fixture(scope="session")
def features_frame():
    """The engineered feature matrix, read once for the whole session.

    100k rows x 19 cols, shared by the three model tests.
    """
    from src.utils.config import PROCESSED_DATA_DIR

    return pd.read_csv(PROCESSED_DATA_DIR / "transactions_features.csv")


@pytest.fixture(scope="session")
def labels(features_frame):
    return features_frame["is_fraud"].to_numpy(dtype=int)
