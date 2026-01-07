# src/data_generation/customer_generator.py

import random
import numpy as np
import hashlib


from src.data_generation.city_loader import sample_home_location
from src.utils.config import NUM_CUSTOMERS, RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "gas",
    "restaurant",
    "retail",
    "jewelry",
    "luxury_goods",
]


def generate_card_number(customer_id: str) -> str:
    """
    Generate a stable, hashed card number for a customer.
    """
    return hashlib.sha256(customer_id.encode()).hexdigest()[:16]


def generate_customer_behavior():
    """
    Generate latent behavioral traits for a customer.
    """
    avg_amount = np.random.lognormal(mean=6.5, sigma=0.6)  # ~₹500–₹5000
    amount_std = avg_amount * np.random.uniform(0.2, 0.5)

    daily_txn_rate = np.random.poisson(lam=2) + 1  # 1–5 txns/day

    active_hours = random.choices(
        population=list(range(24)),
        weights=[
            2 if 8 <= h <= 11 or 18 <= h <= 22 else 0.5 for h in range(24)
        ],
        k=10,
    )

    merchant_pref = np.random.dirichlet(
        alpha=np.random.uniform(0.5, 2.0, size=len(MERCHANT_CATEGORIES))
    )

    return {
        "avg_amount": avg_amount,
        "amount_std": amount_std,
        "daily_txn_rate": daily_txn_rate,
        "active_hours": list(set(active_hours)),
        "merchant_pref": merchant_pref,
    }


def generate_customers():
    """
    Generate full customer profiles.
    """
    customers = []

    for i in range(NUM_CUSTOMERS):
        customer_id = f"CUST_{i:05d}"
        card_number = generate_card_number(customer_id)

        home = sample_home_location()
        behavior = generate_customer_behavior()

        customers.append(
            {
                "customer_id": customer_id,
                "card_number": card_number,
                "home_city": home["city"],
                "home_lat": home["home_lat"],
                "home_lon": home["home_lon"],
                **behavior,
            }
        )

    return customers
