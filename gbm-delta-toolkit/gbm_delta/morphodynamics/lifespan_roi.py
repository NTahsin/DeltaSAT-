# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:28:44 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
intertidal_lifespan_roi.py
==========================
Per-ROI version of the intertidal lifespan (residence-time) analysis.

Reuses the method of ``intertidal_lifespan_v2`` (annualised persistence
P_II, lifespan L = 1/(1-P), Wilson + spatial block-bootstrap CIs,
Theil-Sen / Mann-Kendall trend on logit(P)), but evaluates every statistic
separately inside each ROI polygon of a shapefile.

Differences vs the delta-wide module
-------------------------------------
* Rasters are clipped to each ROI with ``rasterio.mask`` (nodata 255
  outside, matching the module's NODATA convention) before the interval
  analysis; transition counts, P_II, bootstrap, trend and windowed maps
  are then computed per ROI exactly as in v2.
* Windowed P_II / lifespan maps are written per ROI
  (``lifespan_win_{roi}_{y0}-{y1}.tif``) in the ROI's clipped grid.
* A combined summary CSV holds one row per ROI.

Usage
-----
    from gbm_delta.morphodynamics.lifespan_roi import run_roi_analysis

    df_all, summaries, trends = run_roi_analysis(
        input_dir, roi_path, output_dir, years)
"""

#from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

from gbm_delta.morphodynamics.lifespan_v2 import (Config, NODATA, annualise, lifespan,
                                   wilson_ci, _block_sum, _window_sum,
                                   _write_tif, trend_analysis,
                                   overall_summary, make_figures)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# ROI handling
# ----------------------------------------------------------------------------
def load_rois(roi_path: str):
    """(list of (name, geometry), crs) from a shapefile/vector."""
    import geopandas as gpd
    rois = gpd.read_file(roi_path)
    for col in ("roi", "id", "name", "Name", "ROI"):
        if col in rois.columns:
            rois["roi"] = rois[col].astype(str)
            break
    else:
        rois["roi"] = [f"roi_{i}" for i in range(len(rois))]
    return [(r.roi, r.geometry) for r in rois.itertuples()], rois.crs


def read_state_roi(path: Path, roi_geom, roi_crs, cfg: Config):
    """State raster clipped to one ROI polygon (NODATA outside)."""
    import geopandas as gpd

    with rasterio.open(path) as src:
        if roi_crs is not None and src.crs is not None and roi_crs != src.crs:
            g = gpd.GeoSeries([roi_geom], crs=roi_crs).to_crs(src.crs)
            geom = g.iloc[0]
        else:
            geom = roi_geom
        try:
            clipped, clipped_transform = mask(src, [mapping(geom)],
                                              crop=True, filled=True,
                                              nodata=NODATA)
        except ValueError:
            return None, None
        arr = clipped[0]
        profile = src.profile.copy()
        profile.update(height=arr.shape[0], width=arr.shape[1],
                       transform=clipped_transform)
        bad = ~np.isin(arr, list(cfg.codes.values()))
        if src.nodata is not None:
            bad |= np.isclose(arr, src.nodata)
        state = np.where(bad, NODATA, arr).astype(np.uint8)
        meta = (state.shape,)
        return state, profile


# ----------------------------------------------------------------------------
# Per-ROI interval analysis (mirrors analyse_intervals, ROI-scoped)
# ----------------------------------------------------------------------------
def analyse_intervals_roi(input_dir: Path, output_dir: Path,
                          years: Sequence[int], cfg: Config,
                          roi_name: str, roi_geom, roi_crs,
                          write_maps: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    files = {}
    for y in years:
        p = Path(input_dir) / cfg.glob_pattern.format(year=y)
        if p.exists():
            files[y] = p
        else:
            log.warning("Missing %s -> year %d skipped", p, y)
    yrs = sorted(files)
    if len(yrs) < 2:
        raise ValueError(f"ROI {roi_name}: need >= 2 available years.")

    rows = []
    acc_num = acc_den = None
    profile0 = None

    prev, profile0 = read_state_roi(files[yrs[0]], roi_geom, roi_crs, cfg)
    if prev is None:
        raise ValueError(f"ROI {roi_name} does not overlap raster {yrs[0]}")
    for y0, y1 in zip(yrs[:-1], yrs[1:]):
        cur, _ = read_state_roi(files[y1], roi_geom, roi_crs, cfg)
        if cur is None or cur.shape != prev.shape:
            log.warning("ROI %s: %d skipped (no overlap/grid change)",
                        roi_name, y1)
            continue
        dt = y1 - y0
        valid = (prev != NODATA) & (cur != NODATA)
        from_i = (prev == cfg.I) & valid
        stay = from_i & (cur == cfg.I)
        n_i, n_ii = int(from_i.sum()), int(stay.sum())
        if n_i == 0:
            log.warning("ROI %s %d-%d: no Intertidal pixels; skipped.",
                        roi_name, y0, y1)
            prev = cur
            continue

        p_gap = n_ii / n_i
        p_ann = float(annualise(p_gap, dt))
        lo, hi = wilson_ci(n_ii, n_i)

        bi, bii = _block_sum(from_i, cfg.block_px), _block_sum(stay, cfg.block_px)
        keep = bi > 0
        bi, bii = bi[keep], bii[keep]
        idx = rng.integers(0, len(bi), size=(cfg.n_boot, len(bi)))
        pb = bii[idx].sum(1) / bi[idx].sum(1)
        blo, bhi = np.percentile(pb, [2.5, 97.5])

        rows.append(dict(
            roi=roi_name,
            year_from=y0, year_to=y1, gap_yr=dt, n_I=n_i, n_II=n_ii,
            P_II_gap=p_gap, P_II_annual=p_ann,
            P_annual_wilson_lo=float(annualise(lo, dt)),
            P_annual_wilson_hi=float(annualise(hi, dt)),
            P_annual_boot_lo=float(annualise(blo, dt)),
            P_annual_boot_hi=float(annualise(bhi, dt)),
            lifespan_yr=float(lifespan(p_ann)),
            lifespan_boot_lo=float(lifespan(annualise(blo, dt))),
            lifespan_boot_hi=float(lifespan(annualise(bhi, dt))),
            mid_year=(y0 + y1) / 2.0,
        ))

        if write_maps and profile0 is not None:
            w = cfg.window_px
            Wi, Wii = _window_sum(from_i, w), _window_sum(stay, w)
            ok = Wi >= cfg.min_window_n
            with np.errstate(invalid="ignore", divide="ignore"):
                pw_ann = np.where(ok, annualise(np.clip(Wii / Wi, 0, 1), dt), np.nan)
            L_map = np.where(np.isfinite(pw_ann),
                             np.minimum(lifespan(np.nan_to_num(pw_ann)),
                                        cfg.lifespan_cap_yr), np.nan)
            safe = roi_name.replace(" ", "_")
            _write_tif(pw_ann, Path(output_dir) /
                       f"pii_annual_win_{safe}_{y0}-{y1}.tif", profile0)
            _write_tif(L_map, Path(output_dir) /
                       f"lifespan_win_{safe}_{y0}-{y1}.tif", profile0)
            if acc_num is None:
                acc_num = np.zeros_like(Wi, dtype=np.float64)
                acc_den = np.zeros_like(Wi, dtype=np.float64)
            acc_num += np.where(ok, np.nan_to_num(pw_ann) * Wi, 0)
            acc_den += np.where(ok, Wi, 0)
        prev = cur

    if write_maps and acc_den is not None and profile0 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = np.where(acc_den > 0, acc_num / acc_den, np.nan)
        Lm = np.where(np.isfinite(pm),
                      np.minimum(lifespan(np.nan_to_num(pm)),
                                 cfg.lifespan_cap_yr), np.nan)
        safe = roi_name.replace(" ", "_")
        _write_tif(pm, Path(output_dir) /
                   f"pii_annual_win_{safe}_period_mean.tif", profile0)
        _write_tif(Lm, Path(output_dir) /
                   f"lifespan_win_{safe}_period_mean.tif", profile0)

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) /
              f"interval_lifespan_stats_{roi_name.replace(' ', '_')}.csv",
              index=False)
    return df


# ----------------------------------------------------------------------------
# Orchestrator: all ROIs
# ----------------------------------------------------------------------------
def run_roi_analysis(input_dir, roi_path, output_dir,
                      years: Sequence[int],
                      cfg: Optional[Config] = None,
                      do_trend: bool = True):
    """Full lifespan analysis per ROI. Returns (df_all, summaries, trends)."""
    cfg = cfg or Config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rois, roi_crs = load_rois(roi_path)
    log.info("ROIs: %s", [n for n, _ in rois])

    df_all, summaries, trends = [], {}, {}
    for name, geom in rois:
        log.info("=== ROI %s ===", name)
        roi_out = output_dir / name.replace(" ", "_")
        roi_out.mkdir(parents=True, exist_ok=True)
        try:
            df = analyse_intervals_roi(Path(input_dir), roi_out, years,
                                       cfg, name, geom, roi_crs,
                                       write_maps=cfg.write_maps)
        except ValueError as e:
            log.warning("ROI %s skipped: %s", name, e)
            continue
        df_all.append(df)
        summaries[name] = overall_summary(df)
        pd.DataFrame([summaries[name]]).to_csv(
            roi_out / "overall_summary.csv", index=False)
        if do_trend:
            try:
                t = trend_analysis(df)
                t.insert(0, "roi", name)
                t.to_csv(roi_out / "trend_results.csv", index=False)
                trends[name] = t
            except ValueError as e:
                log.warning("ROI %s trend skipped: %s", name, e)
        make_figures(df, trends.get(name), roi_out)

    df_all = pd.concat(df_all, ignore_index=True) if df_all else pd.DataFrame()
    df_all.to_csv(output_dir / "interval_lifespan_stats_all_rois.csv",
                  index=False)
    pd.DataFrame(summaries).T.to_csv(output_dir / "overall_summary_rois.csv")
    if trends:
        pd.concat(trends.values(), ignore_index=True).to_csv(
            output_dir / "trend_results_rois.csv", index=False)
    return df_all, summaries, trends