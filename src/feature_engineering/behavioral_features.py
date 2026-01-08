#Fraud = velocity + deviation --> (used by Stripe, Visa, PayPal)

import pandas as pd

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "timestamp"])

    # Time since last transaction
    df["prev_txn_time"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["time_since_last_txn_sec"] = (
        df["timestamp"] - df["prev_txn_time"]
    ).dt.total_seconds().fillna(0)

    # Velocity features
    df["txn_count_1h"] = (
        df.groupby("customer_id")["timestamp"]
        .rolling("1H")
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["txn_count_24h"] = (
        df.groupby("customer_id")["timestamp"]
        .rolling("24H")
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Amount behavior
    df["avg_amount_24h"] = (
        df.groupby("customer_id")["amount"]
        .rolling("24H")
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(df["amount"])
    )

    df["amount_deviation"] = df["amount"] - df["avg_amount_24h"]

    return df.drop(columns=["prev_txn_time"])
