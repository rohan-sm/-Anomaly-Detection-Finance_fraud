import json
import pandas as pd
import joblib

from src.models.autoencoder import run_autoencoder
from tensorflow.keras.models import load_model


def test_autoencoder_inference():
    
    # Load artifacts
    scaler = joblib.load("models/standard_scaler_v1.pkl")
    ae_model = load_model("models/autoencoder_v1.keras")

    with open("models/model_features_v1.json") as f:
        FEATURES = json.load(f)["features"]

    with open("models/thresholds_v1.json") as f:
        AE_THRESHOLD = json.load(f)["autoencoder"]["threshold_value"]

    # Load data
    df = pd.read_csv("data/processed/transactions_features.csv")

    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    # Run Autoencoder inference
    errors, flags = run_autoencoder(
        ae_model,
        X_scaled,
        AE_THRESHOLD
    )

    # Sanity checks
    assert len(errors) == len(df)
    assert len(flags) == len(df)

    anomaly_rate = flags.mean()
    print("Autoencoder anomaly rate:", anomaly_rate)

    # Expect around 1%
    assert 0.005 <= anomaly_rate <= 0.02

    # Inspect top anomalies
    df["ae_score"] = errors
    df["ae_flag"] = flags

    print("\nTop 10 Autoencoder anomalies:")
    print(
        df.sort_values("ae_score", ascending=False)
        .head(10)[FEATURES + ["ae_score"]]
    )


if __name__ == "__main__":
    test_autoencoder_inference()
