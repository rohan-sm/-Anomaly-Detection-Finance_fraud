import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    # ---- Time since last transaction ----
    prev_time = df.groupby("customer_id")["timestamp"].shift(1)
    df["time_since_last_txn_sec"] = (
        df["timestamp"] - prev_time
    ).dt.total_seconds().fillna(0)

    # ---- Transaction velocity ----
    df["txn_count_1h"] = (
        df.groupby("customer_id")
          .rolling("1h", on="timestamp")["transaction_id"]
          .count()
          .reset_index(drop=True)
          .values
    )

    df["txn_count_24h"] = (
        df.groupby("customer_id")
          .rolling("24h", on="timestamp")["transaction_id"]
          .count()
          .reset_index(drop=True)
          .values
    )

    # ---- Amount baseline ----
    df["avg_amount_24h"] = (
        df.groupby("customer_id")
          .rolling("24h", on="timestamp")["amount"]
          .mean()
          .reset_index(drop=True)
          .values
    )

    df["amount_deviation"] = df["amount"] - df["avg_amount_24h"]

    # ---- Impossible travel feature ----
    prev_lat = df.groupby("customer_id")["merchant_lat"].shift(1)
    prev_long = df.groupby("customer_id")["merchant_long"].shift(1)

    travel_distance = [
        haversine_km(pl, plo, cl, clo)
        if not pd.isna(pl) else 0
        for pl, plo, cl, clo in zip(
            prev_lat, prev_long,
            df["merchant_lat"], df["merchant_long"]
        )
    ]

    time_diff_hours = df["time_since_last_txn_sec"] / 3600

    df["travel_speed_kmh"] = np.divide(
        travel_distance,
        time_diff_hours,
        out=np.zeros_like(time_diff_hours),
        where=time_diff_hours > 0
    )

    df["travel_speed_kmh"] = df["travel_speed_kmh"].clip(upper=20000)

    # ---- Cyclical time ----
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df
