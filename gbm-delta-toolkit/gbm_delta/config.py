"""
gbm_delta.config
================
Shared constants for the GBM delta toolkit: seasons, raster class
encodings, CRS, tidal settings and gauge-station coordinates.

Nothing here is executed at import time and no heavy dependency is
required, so every subpackage can safely import it.
"""

from __future__ import annotations

# ── Seasons ──────────────────────────────────────────────────────────────────
WET_MONTHS = [6, 7, 8, 9, 10]         # June–October    (monsoon / fluvial)
DRY_MONTHS = [11, 12, 1, 2, 3, 4, 5]  # November–May    (tidal dominant)

def season_of_month(month: int) -> str:
    return "wet" if month in WET_MONTHS else "dry"

# ── Raster class encodings ───────────────────────────────────────────────────
# Intertidal classification rasters (intertidal_*.tif, occurrence→state)
LAND       = 0
RIVER      = 1
INTERTIDAL = 2
NODATA     = 255

# WSE (water-surface elevation) rasters from the surface profiler
LAND_VALUE   = 999.0    # pixel marked as land
RIVER_MARKER = -999.0   # pixel marked as permanent river

# ── Spatial reference ────────────────────────────────────────────────────────
CRS_UTM46N = "EPSG:32646"      # UTM zone 46N — GBM delta Landsat working CRS
CRS_WGS84  = "EPSG:4326"

# ── Landsat ──────────────────────────────────────────────────────────────────
PIXEL_SIZE_M = 30.0            # Landsat TM/OLI multispectral
PX_AREA_KM2  = PIXEL_SIZE_M ** 2 / 1e6

# ── Main tidal constituents used by the harmonic reconstruction ─────────────
TIDAL_CONSTITUENTS = {
    "M2":  12.4206012,   # principal lunar semidiurnal
    "S2":  12.0,         # principal solar semidiurnal
    "N2":  13.930,       # larger lunar elliptic semidiurnal
    "K1":  23.93447213,  # lunisolar diurnal
    "O1":  25.81933871,  # principal lunar diurnal
    "P1":  24.06588766,  # solar diurnal
}

# ── GBM gauge stations (coordinates, WGS84) ──────────────────────────────────
# Mirrors gbm_delta.water_levels.stations.STATIONS
STATION_COORDS = {
    "Chandpur":        (23.22098, 90.651716),
    "Char Ramdaspur":  (22.818138, 90.595889),
    "Chitalkhali":     (22.659409, 90.903213),
    "Dasmina":         (22.235616, 90.551491),
    "Dhulia":          (22.559863, 90.533287),
    "Charchenga":      (22.227753, 91.066911),
    "Char Montaj":     (21.904494, 90.509741),
    "Dhulasur":        (21.870721, 90.256854),
    "Monpura":         (22.25855,  90.961526),
    "Tajumuddin":      (22.336107, 90.832967),
    "Rupsha":          (22.696012, 90.379032),
    "Meghna":          (22.778728, 90.650268),
}

# Coastal reference point (approx. Meghna estuary mouth, Hatiya)
COAST_LAT, COAST_LON = 22.20, 90.60
