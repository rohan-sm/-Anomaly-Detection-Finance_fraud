import os

import pandas as pd
import numpy as np

from src.feature_engineering.behavioral_features import add_behavioral_features
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])


    # Columns used by behavioral_features.py
    defaults = {
        "transaction_id": "API_TXN",
        "merchant_lat": 0.0,
        "merchant_long": 0.0,
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Feature engineering (shared)
    df = add_behavioral_features(df)

    # Log-transform amount deviation
    df["amount_dev_log"] = np.sign(df["amount_deviation"]) * np.log1p(
        np.abs(df["amount_deviation"])
    )
    df = df.drop(columns=["amount_deviation"], errors="ignore")

    # Drop leakage / non-model columns
    df = df.drop(
        columns=[
            "transaction_id",
            "card_number",
            "timestamp",
            "fraud_type",
        ],
        errors="ignore",
    )

    return df



# ---------------- BATCH PIPELINE ----------------

RAW_PATH = RAW_DATA_DIR / "transactions_raw.csv"
OUT_PATH = PROCESSED_DATA_DIR / "transactions_features.csv"


def run_feature_engineering():
    print(f"[INFO] Loading: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df = build_features(df)

    os.makedirs(OUT_PATH.parent, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[SUCCESS] Saved -> {OUT_PATH}")


if __name__ == "__main__":
    run_feature_engineering()
