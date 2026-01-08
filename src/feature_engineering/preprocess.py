import json
import pandas as pd
from src.feature_engineering.temporal_features import add_temporal_features
from src.feature_engineering.spatial_features import add_spatial_features
from src.feature_engineering.behavioral_features import add_behavioral_features

RAW_PATH = "data/raw/transactions_raw.csv"
OUT_PATH = "data/processed/transactions_features.csv"
CITY_COORDS_PATH = "data/metadata/city_coordinates.json"

def run_feature_engineering():
    df = pd.read_csv(RAW_PATH)

    with open(CITY_COORDS_PATH) as f:
        city_coords = json.load(f)

    df = add_temporal_features(df)
    df = add_spatial_features(df, city_coords)
    df = add_behavioral_features(df)

    # Drop leakage columns
    df = df.drop(columns=["timestamp", "is_fraud"], errors="ignore")

    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] Feature dataset saved to {OUT_PATH}")

if __name__ == "__main__":
    run_feature_engineering()
