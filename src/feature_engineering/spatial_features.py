#Uses customer city → merchant city distance to add spatial features

import pandas as pd
from src.utils.geo_utils import haversine_distance

def add_spatial_features(df: pd.DataFrame, city_coords: dict) -> pd.DataFrame:
    cust_lat = df["customer_city"].map(lambda x: city_coords[x]["lat"])
    cust_lon = df["customer_city"].map(lambda x: city_coords[x]["lon"])

    merch_lat = df["merchant_city"].map(lambda x: city_coords[x]["lat"])
    merch_lon = df["merchant_city"].map(lambda x: city_coords[x]["lon"])

    df["txn_distance_km"] = haversine_distance(
        cust_lat, cust_lon, merch_lat, merch_lon
    )

    # Suspicious if unusually far
    df["is_long_distance"] = (df["txn_distance_km"] > 300).astype(int)

    return df
