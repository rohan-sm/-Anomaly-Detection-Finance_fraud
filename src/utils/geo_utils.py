
# distance and sampling utilities for geographic calculations
import math
import random

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two lat/long points in KM.
    """
    R = 6371  # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sample_suburb(lat, lon, radius_km_min, radius_km_max):
    """
    Sample a random point within a ring (suburban area).
    """
    radius = random.uniform(radius_km_min, radius_km_max)
    angle = random.uniform(0, 2 * math.pi)

    delta_lat = radius / 111
    delta_lon = radius / (111 * math.cos(math.radians(lat)))

    return (
        lat + delta_lat * math.cos(angle),
        lon + delta_lon * math.sin(angle),
    )
