# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:47:21 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
stations.py
===========
Gauge station coordinates for the GBM delta network, plus helpers to compute
along-channel / coastal distances used in the asymmetry and phase-lag figures.

``STATIONS`` is the full 12-station table; ``station_frame()`` filters it to
the stations present in a water-level DataFrame (e.g. the 6 projection
stations, matching on the station *name* prefix of the CSV column labels
such as ``"Chandpur 1640"``).

Coastal reference point is the approximate Meghna estuary mouth
(Hatiya / Shahbazpur channel outlet).  ``distance_to_coast`` is great-circle
distance to that point — a proxy for position along the estuary axis that is
adequate for qualitative distance-decay plots; replace with along-channel
distances from a channel polyline if your study needs exact values.
"""



import numpy as np
import pandas as pd

# Coastal reference (Meghna estuary mouth, approximate)
COAST_LAT, COAST_LON = 22.20, 90.60

STATIONS = pd.DataFrame(
    {
        "station": [
            "Chandpur", "Char Ramdaspur", "Chitalkhali", "Dasmina",
            "Dhulia", "Charchenga", "Char Montaj", "Dhulasur",
            "Monpura", "Tajumuddin", "Rupsha", "Meghna",
        ],
        "latitude": [
            23.22098, 22.818138, 22.659409, 22.235616,
            22.559863, 22.227753, 21.904494, 21.870721,
            22.25855, 22.336107, 22.696012, 22.778728,
        ],
        "longitude": [
            90.651716, 90.595889, 90.903213, 90.551491,
            90.533287, 91.066911, 90.509741, 90.256854,
            90.961526, 90.832967, 90.379032, 90.650268,
        ],
    }
).set_index("station")


def _resolve_name(col: str) -> str:
    """'Char Ramdaspur 1620' / 'Char Ramdaspur' -> 'Char Ramdaspur'.

    Tries the longest station name that prefixes the column label
    (case-insensitive, ignoring extra tokens such as gauge IDs).
    """
    c = " ".join(str(col).split()).lower()
    for name in sorted(STATIONS.index, key=len, reverse=True):
        n = name.lower()
        if c == n or c.startswith(n + " "):
            return name
    # fall back to first token (single-word station names)
    return str(col).split()[0].strip()


def station_frame(wl_columns) -> pd.DataFrame:
    """Coordinates for the stations named in a list of CSV columns."""
    names = [_resolve_name(c) for c in wl_columns]
    missing = [n for n in names if n not in STATIONS.index]
    if missing:
        raise KeyError(
            f"Station(s) {missing} not found in the coordinates table. "
            f"Known stations: {list(STATIONS.index)}")
    return STATIONS.loc[names].copy()


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (vectorised)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def distance_to_coast(coords: pd.DataFrame,
                       coast_lat=COAST_LAT, coast_lon=COAST_LON) -> pd.Series:
    """Distance (km) from each station to the coastal reference point."""
    d = haversine(coords["latitude"].to_numpy(),
                  coords["longitude"].to_numpy(),
                  coast_lat, coast_lon)
    return pd.Series(d, index=coords.index, name="dist_coast_km")


def enrich(wl_columns) -> pd.DataFrame:
    """Coordinates + coastal distance for the stations in `wl_columns`,
    sorted by increasing distance from the coast."""
    df = station_frame(wl_columns)
    df["dist_coast_km"] = distance_to_coast(df)
    return df.sort_values("dist_coast_km")