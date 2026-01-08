import pandas as pd

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    # Always work on sorted data
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    # Time since last transaction
    prev_time = df.groupby("customer_id")["timestamp"].shift(1)
    df["time_since_last_txn_sec"] = (
        df["timestamp"] - prev_time
    ).dt.total_seconds().fillna(0)

    # Rolling transaction counts (POSITIONAL assignment – critical)
    txn_count_1h = (
        df.groupby("customer_id")
          .rolling("1h", on="timestamp")["transaction_id"]
          .count()
          .reset_index(drop=True)
          .values
    )

    txn_count_24h = (
        df.groupby("customer_id")
          .rolling("24h", on="timestamp")["transaction_id"]
          .count()
          .reset_index(drop=True)
          .values
    )

    avg_amount_24h = (
        df.groupby("customer_id")
          .rolling("24h", on="timestamp")["amount"]
          .mean()
          .reset_index(drop=True)
          .values
    )

    df["txn_count_1h"] = txn_count_1h
    df["txn_count_24h"] = txn_count_24h
    df["avg_amount_24h"] = avg_amount_24h

    # Amount deviation from recent behavior
    df["amount_deviation"] = df["amount"] - df["avg_amount_24h"]

    return df
