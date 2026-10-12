# -*- coding: utf-8 -*-
r"""
Created on Fri Oct  2 15:14:51 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
r"""
intertidal_raster.py
====================
Derive per-year class-area statistics from the intertidal GeoTIFF maps.

Input
-----
GeoTIFFs named ``intertidal_filtered_{year}.tif`` (e.g. 1996, 1998, ... 2018),
one per observation year, in a single folder. Pixel classes:

    0 = land
    1 = river
    2 = intertidal

Pixel area is derived from the raster's own geotransform (``res_x * res_y``),
so any CRS / resolution works.  No-data or NaN pixels are ignored.

Output
------
``intertidal_raster_summary.csv`` with one row per year:

    year, n_scenes(=1), px_land, km2_land, px_river, km2_river,
    px_intertidal, km2_intertidal, km2_total, frac_intertidal_water

Usage
-----
    from gbm_delta.morphodynamics.stats import summarize_rasters, export_raster_summary

    df = summarize_rasters(r"C:\data\intertidal_maps")
    export_raster_summary(df, "intertidal_raster_summary.csv")
"""


import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "land", 1: "river", 2: "intertidal"}
FILENAME_RE = re.compile(r"intertidal_fil(?:t|)ered?_(\d{4})\.tif$",
                         re.IGNORECASE)


def _year_from_name(path: Path) -> int | None:
    m = FILENAME_RE.search(path.name)
    return int(m.group(1)) if m else None


def list_rasters(folder: str) -> dict[int, str]:
    """{year: path} for every intertidal_filtered_{year}.tif in `folder`."""
    out = {}
    for p in sorted(Path(folder).glob("*.tif")):
        yr = _year_from_name(p)
        if yr is not None:
            out[yr] = str(p)
    if not out:
        raise FileNotFoundError(
            f"No files matching 'intertidal_filtered_{{year}}.tif' "
            f"found in {folder}")
    return out


def raster_year_stats(path: str) -> dict:
    """Class pixel counts and areas (km²) for one GeoTIFF."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1)
        res_x, res_y = src.res          # degrees or metres per pixel
        px_area_km2 = abs(res_x * res_y)
        # convert if the CRS is geographic (degrees → km², ~lat-dependent)
        if src.crs and src.crs.is_geographic:
            lat = (src.bounds.top + src.bounds.bottom) / 2.0
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * np.cos(np.radians(lat))
            px_area_km2 *= km_per_deg_lat * km_per_deg_lon

    data = np.asarray(data)
    valid = np.isin(data, [0, 1, 2])
    row = {"file": Path(path).name}
    for code, name in CLASS_NAMES.items():
        n = int(np.count_nonzero(data[valid] == code))
        row[f"px_{name}"] = n
        row[f"km2_{name}"] = n * px_area_km2
    row["px_total_valid"] = int(valid.sum())
    row["km2_total"] = row["px_total_valid"] * px_area_km2
    iw = row["px_intertidal"] + row["px_river"]
    row["frac_intertidal_water"] = (row["px_intertidal"] / iw) if iw else np.nan
    return row


def summarize_rasters(folder: str) -> pd.DataFrame:
    """Per-year class-area table for all rasters in `folder`."""
    rows = []
    for yr, path in list_rasters(folder).items():
        r = raster_year_stats(path)
        r["year"] = yr
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("year").set_index("year")
    # drop the helper column
    df = df.drop(columns=["km2_total_valid"], errors="ignore")
    return df


def export_raster_summary(df: pd.DataFrame,
                          out_csv: str = "intertidal_raster_summary.csv"):
    df.round(4).to_csv(out_csv)
    print(f"saved {out_csv}")
    return df