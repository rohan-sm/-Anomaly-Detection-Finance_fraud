import json
import pandas as pd
import joblib

from src.models.one_class_svm import run_one_class_svm


def test_one_class_svm_inference():
    
    # Load artifacts
    scaler = joblib.load("models/standard_scaler_v1.pkl")
    svm_model = joblib.load("models/one_class_svm_v1.pkl")

    with open("models/model_features_v1.json") as f:
        FEATURES = json.load(f)["features"]

    with open("models/thresholds_v1.json") as f:
        THRESHOLD = json.load(f)["one_class_svm"]["threshold_value"]

    
    # Load data
    df = pd.read_csv("data/processed/transactions_features.csv")

    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    # Run SVM inference
    scores, flags = run_one_class_svm(
        svm_model,
        X_scaled,
        THRESHOLD
    )

    # Assertions (sanity checks)
    assert len(scores) == len(df)
    assert len(flags) == len(df)

    anomaly_rate = flags.mean()
    print("One-Class SVM anomaly rate:", anomaly_rate)

    # Should be close to nu (~1%)
    assert 0.005 <= anomaly_rate <= 0.02

    # Inspect top anomalies
    df["svm_score"] = scores
    df["svm_flag"] = flags

    print("\nTop 10 SVM anomalies:")
    print(
        df.sort_values("svm_score")
        .head(10)[FEATURES + ["svm_score"]]
    )


if __name__ == "__main__":
    test_one_class_svm_inference()
