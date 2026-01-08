import pandas as pd
from pathlib import Path

from src.feature_engineering.behavioral_features import add_behavioral_features

BASE_DIR = Path(__file__).resolve()
while BASE_DIR.name != "fraud-anamoly-detection":
    BASE_DIR = BASE_DIR.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "transactions_raw.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "transactions_features.csv"

def run_feature_engineering():
    print(f"[INFO] Loading: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Behavioral / velocity features
    df = add_behavioral_features(df)

    # Drop leakage / non-model columns
    df = df.drop(
        columns=[
            "transaction_id",
            "card_number",
            "timestamp",
            "fraud_type"
        ],
        errors="ignore"
    )

    df.to_csv(OUT_PATH, index=False)
    print(f"[SUCCESS] Saved → {OUT_PATH}")

if __name__ == "__main__":
    run_feature_engineering()
