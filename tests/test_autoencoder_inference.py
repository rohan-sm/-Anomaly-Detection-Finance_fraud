import numpy as np
import pytest
from tensorflow.keras.models import load_model

from src.models.autoencoder import run_autoencoder
from src.utils.config import MODELS_DIR

MIN_ANOMALY_RATE = 0.005
MAX_ANOMALY_RATE = 0.02


def test_autoencoder_inference(contract, scaler, features_frame):
    model = load_model(MODELS_DIR / "autoencoder_v1.keras")
    features = contract["features"]
    threshold = contract["thresholds"]["autoencoder"]["threshold_value"]

    X_scaled = scaler.transform(features_frame[features])

    errors, flags = run_autoencoder(model, X_scaled, threshold)

    assert len(errors) == len(features_frame)
    assert len(flags) == len(features_frame)

    anomaly_rate = flags.mean()
    assert MIN_ANOMALY_RATE <= anomaly_rate <= MAX_ANOMALY_RATE, (
        f"anomaly rate {anomaly_rate:.4%} outside "
        f"[{MIN_ANOMALY_RATE:.1%}, {MAX_ANOMALY_RATE:.1%}]"
    )


def test_autoencoder_threshold_matches_declared_percentile(
    contract, scaler, features_frame
):
    """Regression: the stored threshold must be the percentile it declares.

    `thresholds_v1.json` claimed `"percentile": 99` while storing the 99.8th
    percentile of reconstruction error, which flagged only 0.2% of rows instead
    of the intended 1%.
    """
    model = load_model(MODELS_DIR / "autoencoder_v1.keras")
    features = contract["features"]
    ae_config = contract["thresholds"]["autoencoder"]

    X_scaled = scaler.transform(features_frame[features])
    errors, _ = run_autoencoder(model, X_scaled, threshold=0.0)

    expected = float(np.percentile(errors, ae_config["percentile"]))
    assert ae_config["threshold_value"] == pytest.approx(expected, rel=1e-6)
