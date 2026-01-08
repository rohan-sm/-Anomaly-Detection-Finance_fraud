# Fraud spikes at night + weekends

import pandas as pd

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Night transactions (fraud-prone)
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    return df
