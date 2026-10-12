# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:12:29 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
planet_mosaic.py
================
Build a mosaic reference water-fraction layer on the classification grid
from many Planet AnalyticMS SR tiles + cloud masks (UDM2), for validating
water/land classification.

Pipeline per tile
-----------------
  NDWI = (Green − NIR)/(Green + NIR)  (SR scale cancels)
  nodata (0 / negative) and UDM2 cloud/shadow → NaN
  threshold NDWI (default 0.0) → water 0/1
  reproject water & validity to the class grid (Resampling.average)
Aggregate across tiles:  frac = Σ(w·v) / Σ(v)  (overlap-safe)
Rules:  Σv ≥ 0.9 to be covered; ≥0.5 water; ≤0.3 land; else mixed (excluded)

Usage
-----
    from gbm_delta.extraction.planet_mosaic import build_reference
    frac, valid = build_reference(sr_tiles, class_grid_path,
                                  cloud_dir=udm2_dir)
"""


import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

log = logging.getLogger(__name__)

RE_ACQ = re.compile(r"_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_")


# ─────────────────────────────────────────────────────────────────────────────
# inventory
# ─────────────────────────────────────────────────────────────────────────────
def inventory(sr_dir: str, pattern: str = "*AnalyticMS_SR*.tif",
              cloud_dir: Optional[str] = None,
              cloud_pattern: str = "*udm2*.tif") -> pd.DataFrame:
    import pandas as pd
    rows = []
    for f in sorted(Path(sr_dir).glob(pattern)):
        m = RE_ACQ.search(f.name)
        date = (f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None)
        with rasterio.open(f) as src:
            crs, b = src.crs, src.bounds
            cloud = 0
            # % cloud from the UDM2 clear-map band if present
        cfile = None
        if cloud_dir:
            tag = f.name.split("AnalyticMS_SR")[0]     # scene id prefix
            cands = list(Path(cloud_dir).glob(f"{tag}*{cloud_pattern[1:]}"))
            cfile = str(cands[0]) if cands else None
        rows.append(dict(file=str(f), date=date, cloud_file=cfile))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# per-tile NDWI water mask (with cloud + nodata masking)
# ─────────────────────────────────────────────────────────────────────────────
def tile_water_fraction(sr_path: str, cloud_path: Optional[str],
                        green: int = 2, nir: int = 4,
                        threshold: float = 0.0) -> Tuple[np.ndarray, object, object, str]:
    """
    Returns (water01 float32 with NaN outside valid, transform, crs, date).
    green/nir are 1-based band numbers (PlanetScope 4-band: 2=Green, 4=NIR).
    """
    with rasterio.open(sr_path) as src:
        g = src.read(green).astype(np.float32)
        n = src.read(nir).astype(np.float32)
        nd = src.nodata
        tr, crs = src.transform, src.crs
    valid = np.isfinite(g) & np.isfinite(n) & (g > -1000) & (n > -1000)
    if nd is not None:
        valid &= (g != nd) & (n != nd)
    if cloud_path is not None:
        with rasterio.open(cloud_path) as csrc:
            udm2 = csrc.read(1)     # bit 1 = cloud, bit 2 = shadow (UDM2)
            bad = (udm2 & 0b0000_0011) > 0        # cloud or shadow
            h = min(bad.shape[0], valid.shape[0])
            w = min(bad.shape[1], valid.shape[1])
            valid[:h, :w] &= ~bad[:h, :w]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(valid, (g - n) / (g + n), np.nan)
    water = np.where(np.isfinite(ndwi), (ndwi > threshold).astype(np.float32), np.nan)
    return water, tr, crs


# ─────────────────────────────────────────────────────────────────────────────
# mosaic onto the classification grid
# ─────────────────────────────────────────────────────────────────────────────
def build_reference(sr_tiles: Sequence[str],
                     class_grid_path: str,
                     cloud_files: Optional[Sequence[Optional[str]]] = None,
                     green: int = 2, nir: int = 4,
                     ndwi_threshold: float = 0.0,
                     min_coverage: float = 0.9,
                     water_frac: float = 0.5,
                     land_frac: float = 0.3
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (water_fraction, coverage) arrays on the classification grid.
    water_fraction: NaN outside coverage; coverage: Σ validity fraction.
    """
    with rasterio.open(class_grid_path) as csrc:
        shape = csrc.read(1).shape
        ctransform, ccrs = csrc.transform, csrc.crs

    num = np.zeros(shape, dtype=np.float64)     # Σ water·valid
    den = np.zeros(shape, dtype=np.float64)     # Σ valid

    for i, sr in enumerate(sr_tiles):
        cf = cloud_files[i] if cloud_files else None
        try:
            water, tr, crs = tile_water_fraction(
                sr, cf, green=green, nir=nir, threshold=ndwi_threshold)
        except Exception as exc:
            log.warning("tile %s failed: %s", Path(sr).name, exc)
            continue
        w = np.nan_to_num(water, nan=0.0).astype(np.float32)
        v = np.isfinite(water).astype(np.float32)
        dst_w = np.zeros(shape, dtype=np.float32)
        dst_v = np.zeros(shape, dtype=np.float32)
        reproject(w, dst_w, src_transform=tr, src_crs=crs,
                  dst_transform=ctransform, dst_crs=ccrs,
                  resampling=Resampling.average, src_nodata=0,
                  dst_nodata=0)
        reproject(v, dst_v, src_transform=tr, src_crs=crs,
                  dst_transform=ctransform, dst_crs=ccrs,
                  resampling=Resampling.average, src_nodata=0,
                  dst_nodata=0)
        num += dst_w * dst_v
        den += dst_v
        log.info("tile %d/%d %s done", i + 1, len(sr_tiles),
                 Path(sr).name)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, np.nan)
    coverage = den
    frac[coverage < min_coverage] = np.nan
    return frac, coverage


def reference_labels(frac: np.ndarray,
                     water_frac: float = 0.5,
                     land_frac: float = 0.3) -> np.ndarray:
    """0 = land, 1 = water, 255 = mixed/uncovered (excluded from validation)."""
    lab = np.full(frac.shape, 255, dtype=np.uint8)
    lab[(frac <= land_frac)] = 0
    lab[(frac >= water_frac)] = 1
    lab[~np.isfinite(frac)] = 255
    return lab


def save_reference(frac: np.ndarray, coverage: np.ndarray,
                   class_grid_path: str, out_path: str):
    with rasterio.open(class_grid_path) as src:
        p = src.profile.copy()
    p.update(dtype="float32", count=2, nodata=np.nan, compress="lzw")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(frac.astype(np.float32), 1)
        dst.write(coverage.astype(np.float32), 2)
    log.info("wrote %s", out_path)

