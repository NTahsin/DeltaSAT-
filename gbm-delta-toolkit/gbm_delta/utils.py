"""
gbm_delta.utils
===============
Small shared helpers used across the toolkit: filename/date parsing,
Otsu thresholding, GeoTIFF I/O, IDW interpolation and haversine distance.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ── Dates from filenames ─────────────────────────────────────────────────────
_DATE_PATS = [
    (r"(\d{4}-\d{2}-\d{2})_(\d{6})", "%Y-%m-%d %H%M%S"),
    (r"(\d{4}-\d{2}-\d{2})",          "%Y-%m-%d"),
    (r"(\d{8})_(\d{6})",              "%Y%m%d %H%M%S"),
    (r"(\d{8})",                       "%Y%m%d"),
]


def parse_datetime_from_filename(fname: str) -> Optional[datetime]:
    """First YYYY-MM-DD / YYYYMMDD datetime embedded in a filename."""
    for pat, fmt in _DATE_PATS:
        m = re.search(pat, str(fname))
        if m:
            try:
                return datetime.strptime(" ".join(m.groups()).strip(), fmt)
            except ValueError:
                continue
    return None


def parse_year(text: str) -> Optional[int]:
    """First 4-digit year (1980–2099) in a string, else None."""
    m = re.search(r"(19[89]\d|20\d{2})", str(text))
    return int(m.group(1)) if m else None


# ── Otsu thresholding ────────────────────────────────────────────────────────
def otsu_threshold(values: np.ndarray, nbins: int = 256,
                   fallback: float = 0.0) -> float:
    """
    Classic Otsu (1979) threshold from the finite values in `values`.
    Returns `fallback` when undefined (empty or constant input).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(fallback)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return float(fallback)

    hist, edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    csum = np.cumsum(hist * centers)
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return float(fallback)

    total = csum[-1]
    mu0 = csum / np.maximum(w0, 1e-12)
    mu1 = (total - csum) / np.maximum(w1, 1e-12)
    var_between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, 0.0)
    if var_between.max() <= 0:
        return float(fallback)
    return float(centers[int(np.argmax(var_between))])


# ── GeoTIFF helpers ──────────────────────────────────────────────────────────
def read_band(path, band: int = 1) -> Tuple[np.ndarray, dict]:
    """Read one band as float32 (nodata → NaN) with the rasterio profile."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        nd = src.nodata
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            arr[arr == nd] = np.nan
        return arr, src.profile.copy()


def write_geotiff(data: np.ndarray, path, profile: dict,
                  dtype: str = "float32", nodata=np.nan) -> None:
    """Write a single-band GeoTIFF with LZW compression."""
    import rasterio
    prof = profile.copy()
    prof.update(dtype=dtype, count=1, nodata=nodata, compress="lzw",
                tiled=True, blockxsize=256, blockysize=256)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.asarray(data).astype(dtype), 1)


# ── Spatial helpers ──────────────────────────────────────────────────────────
def idw(values, src_xy, tgt_xy, power: float = 2.0, min_dist: float = 1.0):
    """Inverse-distance-weighted interpolation (exact at source points)."""
    values = np.asarray(values, dtype=np.float64)
    src_xy = np.asarray(src_xy, dtype=np.float64)
    tgt_xy = np.asarray(tgt_xy, dtype=np.float64)
    diff = src_xy[:, np.newaxis, :] - tgt_xy[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    dists = np.where(dists < min_dist, min_dist, dists)
    w = 1.0 / dists ** power
    res = (w * values[:, np.newaxis]).sum(axis=0) / w.sum(axis=0)
    exact = dists.min(axis=0) < min_dist
    if exact.any():
        res[exact] = values[dists.argmin(axis=0)[exact]]
    return res


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
