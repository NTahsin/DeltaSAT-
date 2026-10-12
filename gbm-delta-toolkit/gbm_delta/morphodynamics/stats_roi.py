# -*- coding: utf-8 -*-
r"""
Created on Fri Oct  2 17:13:03 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
r"""
intertidal_roi.py
=================
Per-year intertidal class-area statistics inside regions of interest (ROIs).

Inputs
------
* GeoTIFF maps ``intertidal_filtered_{year}.tif`` (0 = land, 1 = river,
  2 = intertidal), one folder.
* A shapefile (or any vector readable by geopandas) with polygon ROIs;
  the ROI id column defaults to the first column or ``roi``/``id``/``name``
  if present.

For each ROI polygon and each year the module clips the raster to the
polygon (rasterized mask, all-touched=False), counts pixels per class and
converts to km² (degrees → km², latitude-corrected, when the CRS is
geographic; per-ROI latitude).

Output tables
-------------
* ``roi_year_stats``  : long format — roi, year, px/km² per class, fractions.
* ``intertidal_wide`` : years × ROIs km² intertidal (pivot, easy plotting).

Usage
-----
    from gbm_delta.morphodynamics.stats_roi import summarize_rois, export_roi_summary
    df = summarize_rois(r"C:\data\intertidal_maps", r"C:\data\rois.shp")
"""

#from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "land", 1: "river", 2: "intertidal"}
FILENAME_RE = re.compile(r"intertidal_fil(?:t|)ered?_(\d{4})\.tif$",
                         re.IGNORECASE)


def list_rasters(folder: str) -> dict[int, str]:
    out = {}
    for p in sorted(Path(folder).glob("*.tif")):
        m = FILENAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = str(p)
    if not out:
        raise FileNotFoundError(
            f"No 'intertidal_filtered_{{year}}.tif' files in {folder}")
    return out


def load_rois(roi_path: str) -> "gpd.GeoDataFrame":
    import geopandas as gpd
    rois = gpd.read_file(roi_path)
    for col in ("roi", "id", "name", "Name", "ROI"):
        if col in rois.columns:
            rois = rois.rename(columns={col: "roi"})
            break
    else:
        rois["roi"] = [f"roi_{i}" for i in range(len(rois))]
    rois["roi"] = rois["roi"].astype(str)
    return rois[["roi", "geometry"]]


def roi_year_stats(raster_path: str, roi_geom, roi_crs,
                   raster_crs) -> dict:
    """Class stats for one raster clipped to one ROI polygon."""
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import mapping
    import geopandas as gpd

    if roi_crs is not None and raster_crs is not None and roi_crs != raster_crs:
        g = gpd.GeoSeries([roi_geom], crs=roi_crs).to_crs(raster_crs)
        geom = g.iloc[0]
    else:
        geom = roi_geom

    with rasterio.open(raster_path) as src:
        try:
            clipped, _ = mask(src, [mapping(geom)], crop=True,
                              filled=True, nodata=255)
            data = clipped[0]
            res_x, res_y = src.res
            bounds = src.bounds
            geographic = src.crs and src.crs.is_geographic
        except ValueError:      # ROI does not overlap this raster
            return None

    if geographic:
        lat = (bounds.top + bounds.bottom) / 2.0
        px_km2 = (abs(res_x) * 111.32 * np.cos(np.radians(lat))
                  * abs(res_y) * 111.32)
    else:
        px_km2 = abs(res_x * res_y) / 1e6      # m² → km²

    valid = np.isin(data, [0, 1, 2])
    row = {}
    for code, name in CLASS_NAMES.items():
        n = int(np.count_nonzero(data[valid] == code))
        row[f"px_{name}"] = n
        row[f"km2_{name}"] = n * px_km2
    row["px_valid"] = int(valid.sum())
    iw = row["px_intertidal"] + row["px_river"]
    row["frac_intertidal_wet"] = (row["px_intertidal"] / iw
                                  if iw else np.nan)
    return row


def summarize_rois(raster_folder: str, roi_path: str) -> pd.DataFrame:
    """Long table: roi × year with class pixel counts, km² and fractions."""
    import rasterio

    rois = load_rois(roi_path)
    rasters = list_rasters(raster_folder)
    rows = []
    with rasterio.open(next(iter(rasters.values()))) as src0:
        raster_crs = src0.crs

    for _, roi in rois.iterrows():
        for yr, path in rasters.items():
            stats = roi_year_stats(path, roi.geometry, rois.crs, raster_crs)
            if stats is None:
                continue
            rows.append({"roi": roi["roi"], "year": yr, **stats})
    df = pd.DataFrame(rows).sort_values(["roi", "year"]).reset_index(drop=True)
    return df


def intertidal_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: years (rows) × ROIs (columns), intertidal km²."""
    p = df.pivot(index="year", columns="roi", values="km2_intertidal")
    p.index.name, p.columns.name = "year", "roi"
    return p


def export_roi_summary(df: pd.DataFrame,
                       out_csv: str = "intertidal_roi_summary.csv"):
    df.round(4).to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    return df