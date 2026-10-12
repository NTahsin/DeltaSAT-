# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:34:47 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
water_index_ashman.py
=====================
Per-raster water-index quality metrics for a folder of water-index GeoTIFFs
(e.g. NDWI / MNDWI / AWEI scenes):

    1. Ashman's D separability score between the index-value distributions
       of WATER pixels and NON-WATER pixels:
           D = sqrt(2) * |mu_w - mu_n| / sqrt(s_w^2 + s_n^2)
       D > 2  → excellent separability (classes well separated)
       D ~ 1  → moderate;  D < 0.8 → poor separability.

    2. % pixels with a valid value (excluding raster nodata and, if a
       cloud mask raster exists, cloud pixels).

Water / non-water split (choose one per call):
    * class raster  — intertidal_filtered_{year}.tif convention
                      (0 = land, 1 = river, 2 = intertidal;
                       water = 1 ∪ 2, non-water = 0)
    * threshold     — Otsu on the index histogram if no class raster
                      is supplied (or a fixed index threshold, e.g. 0).

Usage
-----
    from water_index_ashman import summarize_index_folder
    df = summarize_index_folder(index_dir, class_dir=None,
                                out_csv="water_index_ashman.csv")
"""



import logging
import re
from pathlib import Path
from typing import Dict, Optional
import os
import numpy as np
import pandas as pd
import rasterio

try:
    from rasterio.mask import mask as rio_mask
except ImportError:
    rio_mask = None

log = logging.getLogger(__name__)

RE_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
RE_DATE = re.compile(r"(\d{4})[-_]?(0[1-9]|1[0-2])[-_]?([0-3]\d)")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_date_or_year(name: str) -> str:
    m = RE_DATE.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = RE_YEAR.search(name)
    return m.group(0) if m else Path(name).stem


def ashman_d(x: np.ndarray, y: np.ndarray) -> float:
    """Ashman's D between two samples. NaN if either is empty/degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    s = np.sqrt(np.var(x) + np.var(y))
    if s == 0:
        return np.nan
    return float(np.sqrt(2.0) * abs(np.mean(x) - np.mean(y)) / s)


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu threshold on a 1-D sample (NaN-safe)."""
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return np.nan
    hist, edges = np.histogram(v, bins=n_bins)
    hist = hist.astype(float)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
    mu = np.cumsum(hist * centres)
    tot = mu[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.where(w0 > 0, mu / w0, 0)
        m1 = np.where(w1 > 0, (tot - mu) / w1, 0)
        var_between = w0 * w1 * (m0 - m1) ** 2
    var_between[~np.isfinite(var_between)] = -1
    return float(centres[int(np.argmax(var_between))])


def _read_band(path: Path) -> tuple[np.ndarray, Optional[float]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        return arr, src.nodata


def _read_class(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _match_class_for(index_path: Path, class_dir: Optional[Path]) -> Optional[Path]:
    """Class raster with the same date/year as the index raster, if any."""
    if class_dir is None:
        return None
    tag = _parse_date_or_year(index_path.name)
    for f in sorted(class_dir.glob("*.tif")):
        if tag == _parse_date_or_year(f.name):
            return f
    return None


# ─────────────────────────────────────────────────────────────────────────────
# core per-raster stats
# ─────────────────────────────────────────────────────────────────────────────
def index_stats(index_path: Path,
                class_dir: Optional[Path] = None,
                cloud_mask_dir: Optional[Path] = None,
                index_threshold: Optional[float] = None,
                max_samples: int = 2_000_000) -> dict:
    """
    Ashman D (water vs non-water index distributions) and valid-pixel %
    for one water-index raster.
    """
    arr, nodata = _read_band(index_path)

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    # cloud mask (if provided): files matching the same date, 1 = cloud
    if cloud_mask_dir is not None:
        cm_path = _match_class_for(index_path, cloud_mask_dir)
        if cm_path is not None:
            try:
                cm, _ = _read_band(cm_path)
                h = min(cm.shape[0], arr.shape[0])
                w = min(cm.shape[1], arr.shape[1])
                valid[:h, :w] &= cm[:h, :w] != 1
            except Exception as exc:
                log.warning("cloud mask failed for %s: %s",
                            index_path.name, exc)

    n_total = arr.size
    n_valid = int(valid.sum())
    pct_valid = 100.0 * n_valid / n_total if n_total else np.nan

    row = dict(file=index_path.name, date=_parse_date_or_year(index_path.name),
               n_total=int(n_total), n_valid=n_valid,
               pct_valid=round(pct_valid, 2))

    v = arr[valid]
    if len(v) < 100:
        row.update(ashman_D=np.nan, threshold=np.nan, method="too-few-valid",
                   n_water=0, n_nonwater=0)
        return row

    # ── water / non-water split ─────────────────────────────────────────
    cls_path = _match_class_for(index_path, class_dir)
    if cls_path is not None:
        cls = _read_class(cls_path)
        h = min(cls.shape[0], arr.shape[0])
        w = min(cls.shape[1], arr.shape[1])
        vidx = valid[:h, :w]
        water = ((cls[:h, :w] == 1) | (cls[:h, :w] == 2)) & vidx
        land = (cls[:h, :w] == 0) & vidx
        method = "class-raster"
    else:
        thr = (index_threshold if index_threshold is not None
               else otsu_threshold(v))
        water = valid & (arr > thr)
        land = valid & (arr <= thr)
        method = ("fixed" if index_threshold is not None else "otsu")

    # subsample for speed on huge scenes
    def sample(m: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(m.ravel())
        if len(idx) > max_samples:
            idx = np.random.default_rng(42).choice(
                idx, max_samples, replace=False)
        return arr.ravel()[idx]

    w_vals, n_vals = sample(water), sample(land)
    thr_report = (index_threshold if index_threshold is not None
                  else round(float((np.nanmean(w_vals) +
                                    np.nanmean(n_vals)) / 2), 4))
    row.update(ashman_D=round(ashman_d(w_vals, n_vals), 3),
               threshold=thr_report,
               method=method,
               n_water=int(water.sum()), n_nonwater=int(land.sum()),
               mean_index_water=round(float(np.nanmean(w_vals)), 4),
               mean_index_nonwater=round(float(np.nanmean(n_vals)), 4))
    return row


# ─────────────────────────────────────────────────────────────────────────────
# folder summary
# ─────────────────────────────────────────────────────────────────────────────
def summarize_index_folder(index_dir: str,
                           class_dir: Optional[str] = None,
                           cloud_mask_dir: Optional[str] = None,
                           index_glob: str = "*.tif",
                           index_threshold: Optional[float] = None,
                           out_csv: str = "water_index_ashman.csv"
                           ) -> pd.DataFrame:
    """
    Metrics for every water-index GeoTIFF in `index_dir`.

    index_threshold : fixed index threshold (e.g. 0 for NDWI/MNDWI);
                      if None and no matching class raster, Otsu is used.
    """
    files = sorted(Path(index_dir).rglob(index_glob))
    if not files:
        raise FileNotFoundError(f"No rasters matching {index_glob} in {index_dir}")

    rows = []
    for f in files:
        try:
            rows.append(index_stats(
                f,
                class_dir=Path(class_dir) if class_dir else None,
                cloud_mask_dir=Path(cloud_mask_dir) if cloud_mask_dir else None,
                index_threshold=index_threshold))
        except Exception as exc:
            log.warning("Failed on %s: %s", f.name, exc)
            rows.append(dict(file=f.name, error=str(exc)))

    df = pd.DataFrame(rows)
    
    

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)


    df.to_csv(out_csv, index=False)
    #log.info("Wrote %s (%d rasters)", out_csv, len(df))
    return df