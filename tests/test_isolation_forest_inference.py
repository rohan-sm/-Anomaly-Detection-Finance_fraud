import json
import pandas as pd
import joblib

from src.models.isolation_forest import run_isolation_forest


def test_isolation_forest_inference():

    # Load artifacts
    scaler = joblib.load("models/standard_scaler_v1.pkl")
    iso_model = joblib.load("models/isolation_forest_v1.pkl")

    with open("models/model_features_v1.json") as f:
        features = json.load(f)["features"]

    with open("models/thresholds_v1.json") as f:
        threshold = json.load(f)["isolation_forest"]["threshold_value"]

    # Load data
    df = pd.read_csv("data/processed/transactions_features.csv")

    X = df[features]
    X_scaled = scaler.transform(X)

    # Run inference
    scores, flags = run_isolation_forest(
        iso_model,
        X_scaled,
        threshold
    )

    # Assertions (THIS makes it a test)
    assert len(scores) == len(df)
    assert len(flags) == len(df)

    anomaly_rate = flags.mean()
    print("Anomaly rate:", anomaly_rate)

    # Expect ~1% anomalies
    assert 0.005 <= anomaly_rate <= 0.02

    # Attach & inspect
    df["iso_score"] = scores
    df["iso_flag"] = flags

    print("\nTop 10 most anomalous transactions:")
    print(
        df.sort_values("iso_score")
        .head(10)[features + ["iso_score"]]
    )

    # Save output (optional but useful)
    df.to_csv(
        "data/processed/transactions_with_iso_inference.csv",
        index=False
    )


if __name__ == "__main__":
    test_isolation_forest_inference()
