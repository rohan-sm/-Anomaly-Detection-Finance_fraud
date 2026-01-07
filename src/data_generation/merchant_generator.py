# src/data_generation/merchant_generator.py

import random
import numpy as np

from src.data_generation.city_loader import INDIAN_CITIES
from src.utils.geo_utils import sample_suburb
from src.utils.config import NUM_MERCHANTS, RANDOM_SEED

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

# Category → preferred cities (realism)
CATEGORY_CITY_CONSTRAINTS = {
    "luxury_goods": ["Mumbai", "Delhi", "Bengaluru"],
    "jewelry": ["Mumbai", "Delhi", "Bengaluru", "Chennai"],
    "electronics": list(INDIAN_CITIES.keys()),
    "restaurant": list(INDIAN_CITIES.keys()),
    "retail": list(INDIAN_CITIES.keys()),
    "grocery": list(INDIAN_CITIES.keys()),
    "gas": list(INDIAN_CITIES.keys()),
}


def generate_merchants():
    """
    Generate merchant profiles with realistic geo clustering.
    """
    merchants = []

    for i in range(NUM_MERCHANTS):
        merchant_id = f"MERCHANT_{i:05d}"
        category = random.choice(MERCHANT_CATEGORIES)

        allowed_cities = CATEGORY_CITY_CONSTRAINTS[category]
        city = random.choice(allowed_cities)
        city_lat, city_lon = INDIAN_CITIES[city]

        merchant_lat, merchant_lon = sample_suburb(
            city_lat,
            city_lon,
            radius_km_min=0.5,
            radius_km_max=3,
        )

        merchants.append(
            {
                "merchant_id": merchant_id,
                "merchant_category": category,
                "city": city,
                "merchant_lat": merchant_lat,
                "merchant_long": merchant_lon,
            }
        )

    return merchants
