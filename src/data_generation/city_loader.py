# src/data_generation/city_loader.py

import random
from src.utils.geo_utils import sample_suburb
from src.utils.config import (
    SUBURB_RADIUS_KM_MIN,
    SUBURB_RADIUS_KM_MAX,
)

# Real Indian city centers (lat, lon)
INDIAN_CITIES = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567),
    "Mysuru": (12.2958, 76.6394),
    "Indore": (22.7196, 75.8577),
    "Coimbatore": (11.0168, 76.9558),
}

def sample_home_location():

    """Assign a customer to a city + suburb."""
    
    city = random.choice(list(INDIAN_CITIES.keys()))
    city_lat, city_lon = INDIAN_CITIES[city]

    home_lat, home_lon = sample_suburb(
        city_lat,
        city_lon,
        SUBURB_RADIUS_KM_MIN,
        SUBURB_RADIUS_KM_MAX,
    )

    return {
        "city": city,
        "home_lat": home_lat,
        "home_lon": home_lon,
    }
