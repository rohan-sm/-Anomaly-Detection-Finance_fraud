# src/data_generation/generate_dataset.py

import os

from src.data_generation.customer_generator import generate_customers
from src.data_generation.merchant_generator import generate_merchants
from src.data_generation.transaction_simulator import simulate_transactions
from src.data_generation.fraud_rules import (
    inject_card_cloning,
    inject_account_takeover,
    inject_merchant_collusion,
)
from src.utils.config import RAW_DATA_DIR


def main():
    
    print(" FRAUD DATASET GENERATION ")

    # Ensure output directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("[1/6] Generating customers...")
    customers = generate_customers()
    print(f"    → {len(customers)} customers created")

    print("[2/6] Generating merchants...")
    merchants = generate_merchants()
    print(f"    → {len(merchants)} merchants created")

    print("[3/6] Simulating normal transactions...")
    df = simulate_transactions(customers, merchants)
    print(f"    → {len(df)} transactions simulated")

    print("[4/6] Injecting fraud patterns...")
    df = inject_card_cloning(df)
    df = inject_account_takeover(df)
    df = inject_merchant_collusion(df)

    print("[5/6] Final validation checks...")
    fraud_rate = df["is_fraud"].mean()
    fraud_counts = df["fraud_type"].value_counts()
    distance_stats = df.groupby("fraud_type")["distance_from_home"].mean()

    print("\n--- Fraud Rate ---")
    print(f"Fraud Rate: {fraud_rate:.4%}")

    print("\n--- Fraud Counts ---")
    print(fraud_counts)

    print("\n--- Mean Distance from Home (km) ---")
    print(distance_stats)

    print("[6/6] Saving dataset...")

    output_path = RAW_DATA_DIR / "transactions.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset successfully saved to:")
    print(f"   {output_path}")
    
    print(" DATASET GENERATION COMPLETE ")
    


if __name__ == "__main__":
    main()
