# src/data_generation/transaction_simulator.py

import random
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.config import (
    NUM_TRANSACTIONS,
    SIMULATION_DAYS,
    START_DATE,
    RANDOM_SEED,
)
from src.utils.geo_utils import haversine_distance

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def simulate_transactions(customers, merchants):
    """
    Simulate NORMAL (non-fraudulent) transactions with realistic
    distance-aware merchant selection.
    """
    transactions = []

    start_dt = datetime.fromisoformat(START_DATE)
    end_dt = start_dt + timedelta(days=SIMULATION_DAYS)

    # -------------------------------
    # Index merchants by city (KEY FIX)
    # -------------------------------
    merchant_by_city = defaultdict(list)
    for m in merchants:
        merchant_by_city[m["city"]].append(m)

    # Track last transaction time per customer
    last_txn_time = {c["customer_id"]: None for c in customers}

    for txn_id in range(NUM_TRANSACTIONS):
        customer = random.choice(customers)
        cust_id = customer["customer_id"]

        # -------------------------------
        # Timestamp generation (Poisson)
        # -------------------------------
        if last_txn_time[cust_id] is None:
            timestamp = start_dt + timedelta(
                minutes=random.randint(0, SIMULATION_DAYS * 24 * 60)
            )
        else:
            gap_minutes = np.random.exponential(
                scale=1440 / customer["daily_txn_rate"]
            )
            timestamp = last_txn_time[cust_id] + timedelta(minutes=gap_minutes)

            if timestamp > end_dt:
                timestamp = start_dt + timedelta(
                    minutes=random.randint(0, SIMULATION_DAYS * 24 * 60)
                )

        last_txn_time[cust_id] = timestamp

        # ------------------------------------------------
        # DISTANCE-AWARE merchant selection (BUG FIX)
        # ------------------------------------------------
        # 90% local merchants, 10% non-local (travel / online)
        if random.random() < 0.92 and merchant_by_city[customer["home_city"]]:
            merchant = random.choice(
                merchant_by_city[customer["home_city"]]
            )
        else:
            merchant = random.choice(merchants)

        # -------------------------------
        # Amount generation
        # -------------------------------
        amount = max(
            10.0,
            np.random.normal(
                customer["avg_amount"],
                customer["amount_std"],
            ),
        )

        # -------------------------------
        # Distance calculation
        # -------------------------------
        distance_from_home = haversine_distance(
            customer["home_lat"],
            customer["home_lon"],
            merchant["merchant_lat"],
            merchant["merchant_long"],
        )

        transactions.append(
            {
                "transaction_id": f"TXN_{txn_id:08d}",
                "customer_id": cust_id,
                "card_number": customer["card_number"],
                "timestamp": timestamp,
                "amount": round(amount, 2),
                "merchant_id": merchant["merchant_id"],
                "merchant_category": merchant["merchant_category"],
                "merchant_lat": merchant["merchant_lat"],
                "merchant_long": merchant["merchant_long"],
                "distance_from_home": round(distance_from_home, 2),
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "month": timestamp.month,
                "is_fraud": 0,
                "fraud_type": "none",
            }
        )

    return pd.DataFrame(transactions)
