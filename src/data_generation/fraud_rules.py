# src/data_generation/fraud_rules.py

import numpy as np
import pandas as pd
from collections import defaultdict

from src.utils.geo_utils import haversine_distance
from src.utils.config import (
    CARD_CLONING_DISTANCE_KM,
    CARD_CLONING_TIME_MINUTES,
)


def inject_card_cloning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject card cloning fraud based on geo + time + amount violations.
    """
    df = df.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)

    last_txn = {}
    fraud_indices = []

    for idx, row in df.iterrows():
        cust_id = row["customer_id"]

        if cust_id not in last_txn:
            last_txn[cust_id] = row
            continue

        prev = last_txn[cust_id]

        # Time difference (minutes)
        time_diff = (row["timestamp"] - prev["timestamp"]).total_seconds() / 60

        # Distance between consecutive txns
        distance = haversine_distance(
            prev["merchant_lat"],
            prev["merchant_long"],
            row["merchant_lat"],
            row["merchant_long"],
        )

        # Amount deviation (z-score proxy)
        if row["amount"] > prev["amount"] * 2.5:
            amount_flag = True
        else:
            amount_flag = False

        if (
            distance > CARD_CLONING_DISTANCE_KM
            and time_diff < CARD_CLONING_TIME_MINUTES
            and amount_flag
        ):
            fraud_indices.append(idx)

        last_txn[cust_id] = row

    df.loc[fraud_indices, "is_fraud"] = 1
    df.loc[fraud_indices, "fraud_type"] = "card_cloning"

    return df


def inject_account_takeover(df: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
    """
    Inject account takeover using behavioral drift.
    """
    df = df.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)

    for cust_id, cust_df in df.groupby("customer_id"):
        if len(cust_df) < window_size * 2:
            continue

        for i in range(window_size, len(cust_df) - window_size):
            before = cust_df.iloc[i - window_size : i]
            after = cust_df.iloc[i : i + window_size]

            # Merchant category entropy
            entropy_before = before["merchant_category"].value_counts(normalize=True)
            entropy_after = after["merchant_category"].value_counts(normalize=True)

            common = entropy_before.index.intersection(entropy_after.index)
            kl_div = np.sum(
                entropy_before[common]
                * np.log(
                    (entropy_before[common] + 1e-6)
                    / (entropy_after[common] + 1e-6)
                )
            )

            # Night transaction spike
            night_ratio_before = np.mean(before["hour"] < 6)
            night_ratio_after = np.mean(after["hour"] < 6)

            if kl_div > 0.8 and night_ratio_after > night_ratio_before + 0.3:
                df.loc[after.index, "is_fraud"] = 1
                df.loc[after.index, "fraud_type"] = "account_takeover"

    return df


def inject_merchant_collusion(df: pd.DataFrame, target_cases: int = 120) -> pd.DataFrame:
    """
    Inject merchant collusion fraud by selecting a few high-volume merchants
    and flagging consistent high-value transactions.
    """
    df = df.copy()

    merchant_volume = df["merchant_id"].value_counts()
    candidate_merchants = merchant_volume[merchant_volume > 80].index.tolist()

    colluding_merchants = np.random.choice(
        candidate_merchants,
        size=min(7, len(candidate_merchants)),
        replace=False,
    )

    fraud_indices = []

    for merchant_id in colluding_merchants:
        mdf = df[df["merchant_id"] == merchant_id]

        high_value = mdf[mdf["amount"] > mdf["amount"].quantile(0.85)]
        sampled = high_value.sample(
            n=min(len(high_value), target_cases // len(colluding_merchants)),
            random_state=42,
        )

        fraud_indices.extend(sampled.index.tolist())

    df.loc[fraud_indices, "is_fraud"] = 1
    df.loc[fraud_indices, "fraud_type"] = "merchant_collusion"

    return df
