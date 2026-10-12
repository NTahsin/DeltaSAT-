# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:22:15 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
planet_water_anyindex.py
========================
Find water in a Planet raster that is NOT a well-behaved NDWI (values
squeezed below 0), then validate the fused Landsat index against it.

Strategy
--------
1. Use the fused Landsat index (Otsu) as an initial, approximate water map.
2. Sample the Planet raster where Landsat is confidently water vs
   confidently land (erosion applied, so only the cores of each class are
   used → registration noise and mixed edges mostly excluded).
3. From those two populations decide:
      - the direction: is Planet-water HIGHER or LOWER in the index?
      - the separability (Ashman's D)
      - a threshold: midpoint between the two medians, and Otsu on the
        Planet histogram for comparison.
4. Build the Planet water mask on the Landsat grid (fraction → majority,
   mixed cells excluded) and compute the full validation: confusion
   matrix, PA/UA/F1/OA/κ, block-bootstrap CIs.

⚠ The threshold choice uses the Landsat map → the comparison is partially
  circular. Always inspect the exported masks in QGIS against the Planet
  RGB; report in the paper that the reference threshold was guided by
  the map under validation, or (better) rebuild a true NDWI from the raw
  SR tiles with the correct band pair.
"""

#from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage

from gbm_delta.extraction.validation import (otsu, confusion_matrix, binary_metrics,
                                             block_bootstrap_ci)

log = logging.getLogger(__name__)


def _read1(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), dict(
            transform=src.transform, crs=src.crs, nodata=src.nodata)


def resample_to_grid(arr: np.ndarray, meta: dict, grid_path: str,
                     resampling=Resampling.nearest) -> np.ndarray:
    with rasterio.open(grid_path) as src:
        shape = src.read(1).shape
        dst = np.zeros(shape, dtype=np.float32)
        reproject(arr, dst, src_transform=meta["transform"],
                  src_crs=meta["crs"], dst_transform=src.transform,
                  dst_crs=src.crs, resampling=resampling,
                  src_nodata=np.nan, dst_nodata=0)
    return dst


def core_mask(binary: np.ndarray, erode_px: int = 2) -> np.ndarray:
    """Erode a boolean mask → only class cores (away from edges)."""
    m = binary.astype(np.uint8)
    for _ in range(erode_px):
        m = ndimage.binary_erosion(m)
    return m.astype(bool)


def ashman_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    s = np.sqrt(np.var(x) + np.var(y))
    return float(np.sqrt(2) * abs(x.mean() - y.mean()) / s) if s else np.nan


def planet_water_thresholds(
        planet_raster: str, fused_index: str,
        erode_px: int = 2,
        max_samples: int = 1_000_000, seed: int = 42) -> dict:
    """
    Determine the water direction + thresholds in the Planet raster using
    the fused Landsat Otsu water map as a guide.
    """
    p_arr, p_meta = _read1(planet_raster)
    f_arr, _ = _read1(fused_index)

    # Landsat's own Otsu water map
    t_fused = otsu(f_arr)
    ls_water = np.isfinite(f_arr) & (f_arr > t_fused)
    ls_land = np.isfinite(f_arr) & (f_arr <= t_fused)

    # Planet raster onto the Landsat grid (nearest is fine for sampling)
    p_on_grid = resample_to_grid(p_arr, p_meta, fused_index)

    rng = np.random.default_rng(seed)

    def sample(mask):
        idx = np.flatnonzero(mask.ravel())
        if len(idx) > max_samples:
            idx = rng.choice(idx, max_samples, replace=False)
        return p_on_grid.ravel()[idx]

    w_vals = sample(core_mask(ls_water, erode_px))
    l_vals = sample(core_mask(ls_land, erode_px))
    if len(w_vals) < 100 or len(l_vals) < 100:
        raise RuntimeError("Too few core pixels to guide the threshold — "
                            "check the fused index and erosion depth.")

    med_w, med_l = float(np.nanmedian(w_vals)), float(np.nanmedian(l_vals))
    direction = "higher" if med_w > med_l else "lower"
    midpoint = (med_w + med_l) / 2.0
    D = ashman_d(w_vals, l_vals)

    # Otsu on the planet histogram, interpreted in the same direction
    t_otsu = otsu(p_arr)
    # fraction above each candidate threshold in the Landsat-water cores
    frac_mid_water = float(np.mean(
        (w_vals > midpoint) if direction == "higher" else
        (w_vals < midpoint)))
    log.info("Planet index: water is %s (median water=%.4f, land=%.4f) "
             "Ashman D=%.2f | midpoint=%.4f otsu=%.4f | "
             "cores agreeing with midpoint: %.1f%%",
             direction, med_w, med_l, D, midpoint, t_otsu,
             100 * frac_mid_water)
    return dict(direction=direction, median_water=med_w, median_land=med_l,
                ashman_D=D, threshold_midpoint=midpoint,
                threshold_otsu=t_otsu, t_fused=t_fused)


def validate_with_any_index(
        planet_raster: str, fused_index: str,
        erode_px: int = 2,
        threshold: Optional[float] = None,     # None → midpoint rule
        mixed_band: Tuple[float, float] = (0.3, 0.7),
        min_coverage: float = 0.5,
        block_px: int = 64, n_boot: int = 200,
        out_prefix: Optional[str] = None
        ) -> Tuple[pd.DataFrame, dict]:
    """
    Full validation: Planet water (derived from the *whatever-it-is* index)
    vs fused Landsat index Otsu water, both on the Landsat grid.
    """
    diag = planet_water_thresholds(planet_raster, fused_index, erode_px)
    direction = diag["direction"]
    thr = threshold if threshold is not None \
        else diag["threshold_midpoint"]

    # ── Planet water mask at Planet resolution ──────────────────────────
    p_arr, p_meta = _read1(planet_raster)
    valid_p = np.isfinite(p_arr)
    p_water = np.where(valid_p,
                       ((p_arr > thr) if direction == "higher"
                        else (p_arr < thr)).astype(np.float32), np.nan)

    # ── Landsat prediction ────────────────────────────────────────────────
    f_arr, _ = _read1(fused_index)
    t_fused = otsu(f_arr)
    pred = np.where(np.isfinite(f_arr),
                    (f_arr > t_fused).astype(np.uint8), 255)

    # ── Planet fraction on the Landsat grid ──────────────────────────────
    with rasterio.open(fused_index) as src:
        shape = src.read(1).shape
        gtransform, gcrs = src.transform, src.crs
    num = np.zeros(shape, dtype=np.float32)
    den = np.zeros(shape, dtype=np.float32)
    layers = (p_water,
              np.isfinite(p_water).astype(np.float32))
    dsts = []
    for arr in layers:
        dst = np.zeros(shape, dtype=np.float32)
        reproject(arr, dst, src_transform=p_meta["transform"],
                  src_crs=p_meta["crs"], src_nodata=0, dst_nodata=0,
                  dst_transform=gtransform, dst_crs=gcrs,
                  resampling=Resampling.average)
        dsts.append(dst)
    num, den = dsts
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, np.nan)

    ref = np.full(shape, 255, dtype=np.uint8)
    ref[(den >= min_coverage) & (frac <= mixed_band[0])] = 0
    ref[(den >= min_coverage) & (frac >= mixed_band[1])] = 1

    ok = (ref != 255) & (pred != 255)
    cm = confusion_matrix(pred[ok], ref[ok])
    met = binary_metrics(cm)
    met.update(dict(planet_threshold=round(thr, 4),
                    planet_direction=direction,
                    planet_ashman_D=round(diag["ashman_D"], 2),
                    fused_otsu_threshold=round(t_fused, 4),
                    pct_cells_excluded=round(100 * (1 - ok.mean()), 2)))
    met.update(block_bootstrap_ci(pred, ref, block_px=block_px,
                                  n_boot=n_boot, valid=ok))

    if out_prefix:
        cm.to_csv(f"{out_prefix}_confusion_matrix.csv")
        pd.DataFrame([met]).to_csv(f"{out_prefix}_metrics.csv",
                                   index=False)
        # masks for visual inspection
        with rasterio.open(fused_index) as src:
            prof = src.profile.copy()
        for name, arr in (("planet_water_ref", ref),
                          ("fused_water_pred", pred),
                          ("disagree", ((ok) & (pred != ref)
                                        ).astype(np.uint8))):
            pr = prof.copy()
            pr.update(count=1, dtype="uint8", nodata=255,
                      compress="lzw", BIGTIFF="YES")
            with rasterio.open(f"{out_prefix}_{name}.tif", "w",
                              **pr) as dst:
                dst.write(arr.astype(np.uint8), 1)
    log.info("OA=%.3f PA=%.3f UA=%.3f F1=%.3f",
             met["OA"], met["PA_water"], met["UA_water"],
             met["F1_water"])
    return cm, met