"""
gbm_delta.extraction.validation
===============================
Validation of Otsu water delineation against a Planet reference.

This module provides the API of the original `otsu_validation.py`
research script, which the rest of the toolkit (notably
`gbm_delta.extraction.planet_anyindex`) imports:

    otsu, read_ndwi, planet_fraction_on_grid,
    confusion_matrix, binary_metrics, area_adjusted_metrics,
    block_bootstrap_ci, validate_fused_index

Class encoding throughout: 0 = land, 1 = water, 255 = nodata/excluded.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

from gbm_delta.utils import otsu_threshold

log = logging.getLogger(__name__)

LAND, WATER, NODATA = 0, 1, 255


# ── Basic rasters ────────────────────────────────────────────────────────────
def otsu(arr: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold of the finite values in `arr` (NaNs ignored)."""
    return otsu_threshold(arr, nbins=nbins, fallback=np.nan)


def read_ndwi(path: str) -> Tuple[np.ndarray, dict]:
    """Read a single-band index raster as float32 (nodata → NaN)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            arr[arr == nd] = np.nan
        return arr, dict(transform=src.transform, crs=src.crs,
                         nodata=src.nodata, profile=src.profile.copy())


def planet_fraction_on_grid(planet_path: str, planet_meta: dict,
                            grid_path: str,
                            min_coverage: float = 0.5,
                            mixed_band: Tuple[float, float] = (0.3, 0.7)
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a Planet water raster (any CRS) onto the Landsat grid and
    return (water_fraction, coverage) as float32 arrays.
    """
    with rasterio.open(grid_path) as src:
        shape = src.read(1).shape
        gtransform, gcrs = src.transform, src.crs

    p_arr, _ = read_ndwi(planet_path)
    layers = (np.nan_to_num(p_arr, nan=0.0).astype(np.float32),
              np.isfinite(p_arr).astype(np.float32))
    dsts = []
    for arr in layers:
        dst = np.zeros(shape, dtype=np.float32)
        reproject(arr, dst, src_transform=planet_meta["transform"],
                  src_crs=planet_meta["crs"], src_nodata=0, dst_nodata=0,
                  dst_transform=gtransform, dst_crs=gcrs,
                  resampling=Resampling.average)
        dsts.append(dst)
    num, den = dsts
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, np.nan)
    frac[den < min_coverage] = np.nan
    return frac, den


# ── Confusion matrix & metrics ───────────────────────────────────────────────
def confusion_matrix(pred: np.ndarray, ref: np.ndarray) -> pd.DataFrame:
    """2×2 confusion matrix (rows = prediction, cols = reference)."""
    pred = np.asarray(pred).ravel()
    ref = np.asarray(ref).ravel()
    ok = (pred != NODATA) & (ref != NODATA)
    p, r = pred[ok].astype(int), ref[ok].astype(int)
    n = int(((p == LAND) & (r == LAND)).sum())    # true negative
    w = int(((p == WATER) & (r == WATER)).sum())  # true positive
    fp = int(((p == WATER) & (r == LAND)).sum())
    fn = int(((p == LAND) & (r == WATER)).sum())
    return pd.DataFrame(
        [[n, fp], [fn, w]],
        index=pd.Index(["Pred: Land", "Pred: Water"]),
        columns=pd.Index(["Ref: Land", "Ref: Water"]))


def binary_metrics(cm: pd.DataFrame) -> Dict[str, float]:
    """
    Standard binary (water-positive) metrics from a confusion matrix.

    Returns dict with N, OA, PA_land, UA_land, PA_water, UA_water,
    F1_water, F1_land, prevalence_water, kappa, and balanced accuracy.
    """
    tn = float(cm.iloc[0, 0]); fp = float(cm.iloc[0, 1])
    fn = float(cm.iloc[1, 0]); tp = float(cm.iloc[1, 1])
    N = tn + fp + fn + tp
    if N == 0:
        return {k: np.nan for k in
                ("N", "OA", "PA_land", "UA_land", "PA_water", "UA_water",
                 "F1_water", "F1_land", "prevalence_water", "kappa",
                 "balanced_acc")}

    pa_w = tp / (tp + fn) if (tp + fn) else np.nan   # producer's (recall)
    ua_w = tp / (tp + fp) if (tp + fp) else np.nan   # user's (precision)
    pa_l = tn / (tn + fp) if (tn + fp) else np.nan
    ua_l = tn / (tn + fn) if (tn + fn) else np.nan
    f1_w = 2 * ua_w * pa_w / (ua_w + pa_w) if ua_w and pa_w else np.nan
    f1_l = 2 * ua_l * pa_l / (ua_l + pa_l) if ua_l and pa_l else np.nan

    pe = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (N * N)
    oa = (tn + tp) / N
    kappa = (oa - pe) / (1 - pe) if pe != 1 else np.nan

    return dict(
        N=int(N), OA=oa,
        PA_water=pa_w, UA_water=ua_w, PA_land=pa_l, UA_land=ua_l,
        F1_water=f1_w, F1_land=f1_l,
        prevalence_water=(fn + tp) / N,
        kappa=kappa,
        balanced_acc=(pa_w + pa_l) / 2 if pa_w == pa_w else np.nan,
    )


def area_adjusted_metrics(cm: pd.DataFrame,
                          class_props: Tuple[float, float]) -> Dict[str, float]:
    """
    Area-adjusted (stratified-equivalent) accuracy using known class
    proportions `class_props` = (p_land, p_water) in the population.
    """
    met = binary_metrics(cm)
    p_land, p_water = float(class_props[0]), float(class_props[1])
    if not np.isfinite(met.get("OA", np.nan)):
        return {}
    return dict(
        OA_areaadj=p_land * met["PA_land"] + p_water * met["PA_water"],
        PA_water_areaadj=met["PA_water"], UA_water_areaadj=met["UA_water"],
    )


# ── Spatial block bootstrap ───────────────────────────────────────────────────
def block_bootstrap_ci(pred: np.ndarray, ref: np.ndarray,
                       block_px: int = 64, n_boot: int = 200,
                       valid: Optional[np.ndarray] = None,
                       seed: int = 42) -> Dict[str, float]:
    """
    Spatial block-bootstrap 95% CIs for OA / PA / UA / F1 / kappa.

    Resamples non-overlapping `block_px` × `block_px` blocks of pixels
    (with replacement) `n_boot` times, recomputes the metrics per resample
    and returns the 2.5/97.5 percentiles as `<metric>_ci_lo` / `<metric>_ci_hi`.

    When `valid` is given (boolean mask), pixels outside it are excluded —
    pass full 2-D `pred`/`ref` with `valid=ok`, or pre-indexed 1-D arrays
    without `valid` (both call styles are supported).
    """
    pred2, ref2 = np.asarray(pred), np.asarray(ref)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != pred2.shape:          # 1-D pred/ref + 2-D mask
            flat_valid = valid.ravel()
            pred2 = pred2[flat_valid] if pred2.ndim == 1 else pred2[valid]
            ref2 = ref2[flat_valid] if ref2.ndim == 1 else ref2[valid]
        else:
            pred2, ref2 = pred2[valid], ref2[valid]
        pred2, ref2 = np.asarray(pred2), np.asarray(ref2)
    else:
        ok = (pred2 != NODATA) & (ref2 != NODATA)
        pred2, ref2 = pred2[ok], ref2[ok]

    n = pred2.size
    if n == 0:
        return {}

    # reshape into whole blocks
    b = int(block_px)
    n_blocks = n // (b * b)
    if n_blocks < 2:
        b = max(1, int(np.sqrt(n // 2)))
        n_blocks = n // (b * b)
    if n_blocks < 2:                             # degenerate — Wilson CI
        from math import sqrt
        tp = float(np.sum((pred2 == WATER) & (ref2 == WATER)))
        k = tp; m = float(np.sum(ref2 == WATER))
        p = k / m if m else 0.5
        z = 1.96; d = 1 + z * z / m
        c = (p + z * z / (2 * m)) / d
        marg = z * sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
        return {"PA_water_ci_lo": max(0, c - marg),
                "PA_water_ci_hi": min(1, c + marg)}

    use = n_blocks * b * b
    pb = pred2[:use].reshape(n_blocks, b, b)
    rb = ref2[:use].reshape(n_blocks, b, b)

    rng = np.random.default_rng(seed)
    stats = {k: [] for k in ("OA", "PA_water", "UA_water", "F1_water", "kappa")}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        p = pb[idx].ravel()
        r = rb[idx].ravel()
        cm = confusion_matrix(p, r)
        m = binary_metrics(cm)
        for k in stats:
            stats[k].append(m[k])

    out = {}
    for k, v in stats.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[f"{k}_ci_lo"] = float(lo)
            out[f"{k}_ci_hi"] = float(hi)
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────
def validate_fused_index(planet_ndwi: str, fused_index: str,
                        planet_bands: Tuple[Optional[int], Optional[int]] = (None, None),
                        mixed_band: Tuple[float, float] = (0.3, 0.7),
                        min_coverage: float = 0.5,
                        block_px: int = 64, n_boot: int = 200,
                        out_prefix: Optional[str] = None
                        ) -> Tuple[pd.DataFrame, dict]:
    """
    Validate an Otsu-thresholded fused Landsat index against a Planet NDWI
    reference, both on the Landsat grid.

    Returns (confusion_matrix, metrics_dict).
    """
    f_arr, f_meta = read_ndwi(fused_index)
    t_fused = otsu(f_arr)
    pred = np.where(np.isfinite(f_arr),
                    (f_arr > t_fused).astype(np.uint8), NODATA)

    p_arr, p_meta = read_ndwi(planet_ndwi)
    frac, cover = planet_fraction_on_grid(planet_ndwi, p_meta, fused_index,
                                          min_coverage=min_coverage)

    ref = np.full(frac.shape, NODATA, dtype=np.uint8)
    ref[(cover >= min_coverage) & (frac <= mixed_band[0])] = LAND
    ref[(cover >= min_coverage) & (frac >= mixed_band[1])] = WATER

    ok = (ref != NODATA) & (pred != NODATA)
    cm = confusion_matrix(pred[ok], ref[ok])
    met = binary_metrics(cm)
    met["fused_otsu_threshold"] = round(float(t_fused), 4)
    met["pct_cells_excluded"] = round(100 * float((~ok).mean()), 2)
    met.update(block_bootstrap_ci(pred, ref, block_px=block_px,
                                  n_boot=n_boot, valid=ok))

    if out_prefix:
        cm.to_csv(f"{out_prefix}_confusion_matrix.csv")
        pd.DataFrame([met]).to_csv(f"{out_prefix}_metrics.csv", index=False)
        prof = f_meta.get("profile") or {}
        if prof:
            prof = prof.copy()
            prof.update(count=1, dtype="uint8", nodata=255,
                        compress="lzw")
            disagree = ((ok) & (pred != ref)).astype(np.uint8)
            for name, arr in (("planet_water_ref", ref),
                              ("fused_water_pred", pred),
                              ("disagree", disagree)):
                with rasterio.open(f"{out_prefix}_{name}.tif", "w",
                                   **prof) as dst:
                    dst.write(arr.astype(np.uint8), 1)

    log.info("OA=%.3f PA=%.3f UA=%.3f F1=%.3f",
             met.get("OA", np.nan), met.get("PA_water", np.nan),
             met.get("UA_water", np.nan), met.get("F1_water", np.nan))
    return cm, met
