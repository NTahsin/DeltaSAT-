# -*- coding: utf-8 -*-
"""
intertidal_tidal_prism.py
=========================
Computes the tidal prism captured by intertidal areas from per-year
GeoTIFF stacks produced by intertidal_elevation.py.

Theory
------
The intertidal tidal prism P is the volume of water that floods and drains
the intertidal surface between MLSW (low-water datum) and MHSW (high-water
datum) over one tidal cycle (O'Brien 1969; Friedrichs & Aubrey 1988):

    P(i,j)   = max(MHSW(i,j) − z(i,j), 0)   [m depth at pixel]
    P_vol(t) = Σ P(i,j) · ΔA                 [m³, summed over intertidal Ω]

where z(i,j) is the intertidal bed elevation and ΔA is the pixel area [m²].

This is numerically identical to the accommodation volume V_acc defined in
intertidal_accommodation_space.py.  The distinction is conceptual:
  • V_acc  → stratigraphic headroom (Jervey 1988)
  • P_vol  → hydrodynamic water exchange (O'Brien 1969)
Both equal Σ (MHSW − z) · ΔA.

Additional metrics derived per year
------------------------------------
  P_vol_m3          total intertidal tidal prism           [m³]
  P_vol_km3         same in km³
  P_normalised      P(t) / P(t_ref)   dimensionless, relative to reference yr
  P_per_area        P_vol / intertidal_area_m2             [m]   mean depth
  intertidal_area_m2   pixel count × ΔA                   [m²]
  fill_ratio_mean   mean φ = (z−MLSW)/(MHSW−MLSW)        [0–1]
  fill_ratio_p25/p75  quartiles of filling state

Inputs (per year, from gbm_delta.morphodynamics.elevation_v2/v3)
------------------------------------------------
  intertidal_h_{year}.tif    bed elevation z(i,j)          [m]
  mhsw_{year}.tif            pixel-wise MHSW               [m]
  mlsw_{year}.tif            pixel-wise MLSW               [m]
  occurrence_{year}.tif      water-occurrence f ∈ [0,1]

  Files must be in the same directory (elevation_dir).

Outputs
-------
  tidal_prism_summary.csv       one row per year, all metrics above
  tidal_prism_{year}.tif        per-pixel prism depth a(i,j)  [m]

References
----------
  O'Brien M.P. (1969). Equilibrium flow areas of inlets on sandy coasts.
      J. Waterways Harbors Div. ASCE, 95, 43–52.
  Friedrichs C.T. & Aubrey D.G. (1988). Non-linear tidal distortion in
      shallow well-mixed estuaries. Estuar. Coast. Shelf Sci., 27, 521–545.
  Townend I. (2010). An examination of empirical stability relationships for
      UK estuaries. J. Coast. Res., 26, 1103–1112.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio

log = logging.getLogger("tidal_prism")


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers  (mirror intertidal_accommodation_space.py)
# ─────────────────────────────────────────────────────────────────────────────
def _read(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None and not np.isnan(float(nd)):
            arr[arr == nd] = np.nan
        return arr, src.profile.copy()


def _write(arr: np.ndarray, path: Path, profile: dict) -> None:
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.float32(np.nan),
             compress="deflate", predictor=2, tiled=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)


def _pixel_area_m2(profile: dict) -> float:
    t = profile["transform"]
    return abs(t.a * t.e)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TidalPrismConfig:
    elevation_dir:   Path          # intertidal_h_{year}.tif lives here
    output_dir:      Path
    # Directory for mhsw, mlsw, and occurrence rasters (if different from elevation_dir).
    # Set to None to use elevation_dir for all inputs.
    datums_dir:      Optional[Path] = None   # mhsw, mlsw, occurrence share this dir
    years:           Optional[List[int]] = None   # None → autodetect
    # File name templates (use {year} placeholder)
    elev_tmpl:  str = "intertidal_h_{year}.tif"
    mhsw_tmpl:  str = "mhsw_{year}.tif"
    mlsw_tmpl:  str = "mlsw_{year}.tif"
    occ_tmpl:   str = "occurrence_{year}.tif"
    # Intertidal mask: 0 < occurrence < 1
    occ_low:    float = 0.0
    occ_high:   float = 1.0
    # Reference year for normalised prism (None → first available year)
    reference_year: Optional[int] = None
    overwrite:  bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Calculator
# ─────────────────────────────────────────────────────────────────────────────
class TidalPrismCalculator:
    """
    Per-year intertidal tidal prism from elevation + datum GeoTIFFs.

    Parameters
    ----------
    cfg : TidalPrismConfig
    """

    def __init__(self, cfg: TidalPrismConfig) -> None:
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ── discovery ────────────────────────────────────────────────────────────
    def _discover_years(self) -> List[int]:
        years = []
        for p in sorted(self.cfg.elevation_dir.glob("intertidal_h_*.tif")):
            try:
                years.append(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
        return years

    def _path(self, tmpl: str, year: int) -> Path:
        """Elevation from elevation_dir; mhsw/mlsw/occurrence from datums_dir (or elevation_dir)."""
        fname = tmpl.format(year=year)
        if tmpl == self.cfg.elev_tmpl:
            return self.cfg.elevation_dir / fname
        base = self.cfg.datums_dir or self.cfg.elevation_dir
        return base / fname

    # ── single year ──────────────────────────────────────────────────────────
    def process_year(self, year: int) -> Optional[dict]:
        out_tif = self.cfg.output_dir / f"tidal_prism_{year}.tif"
        if out_tif.exists() and not self.cfg.overwrite:
            log.info("%d  already exists, skipping (overwrite=False)", year)
            # Re-read summary stats from existing tif
            arr, prof = _read(out_tif)
            px = _pixel_area_m2(prof)
            valid = arr[~np.isnan(arr)]
            return self._row(year, valid, px)

        # --- load rasters ---
        paths = {k: self._path(t, year) for k, t in [
            ("elev", self.cfg.elev_tmpl),
            ("mhsw", self.cfg.mhsw_tmpl),
            ("mlsw", self.cfg.mlsw_tmpl),
            ("occ",  self.cfg.occ_tmpl),
        ]}
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            log.warning("%d  missing rasters: %s — skipping", year, missing)
            return None

        z,    prof = _read(paths["elev"])
        mhsw, _    = _read(paths["mhsw"])
        mlsw, _    = _read(paths["mlsw"])
        occ,  _    = _read(paths["occ"])

        px = _pixel_area_m2(prof)

        # --- intertidal mask ---
        itm = (occ > self.cfg.occ_low) & (occ < self.cfg.occ_high)

        # --- prism depth a(i,j) = max(MHSW − z, 0) ---
        tr     = mhsw - mlsw                       # spring tidal range
        prism  = np.where(itm, np.fmax(mhsw - z, 0.0), np.nan).astype(np.float32)

        # --- fill ratio φ = (z − MLSW) / TR ---
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.where(
                itm & (tr > 0),
                np.clip((z - mlsw) / tr, 0.0, 1.0),
                np.nan,
            ).astype(np.float32)

        _write(prism, out_tif, prof)

        valid_prism = prism[~np.isnan(prism)]
        return self._row(year, valid_prism, px, phi)

    # ── summary row ──────────────────────────────────────────────────────────
    @staticmethod
    def _row(year: int, valid_prism: np.ndarray,
             px_m2: float, phi: Optional[np.ndarray] = None) -> dict:
        n_px        = len(valid_prism)
        area_m2     = n_px * px_m2
        prism_m3    = float(np.nansum(valid_prism)) * px_m2

        row = dict(
            year              = year,
            intertidal_area_m2= area_m2,
            intertidal_area_km2 = area_m2 * 1e-6,
            P_vol_m3          = prism_m3,
            P_vol_km3         = prism_m3 * 1e-9,
            P_per_area_m      = prism_m3 / area_m2 if area_m2 > 0 else np.nan,
            P_mean_depth_m    = float(np.nanmean(valid_prism)) if n_px else np.nan,
        )
        if phi is not None:
            valid_phi = phi[~np.isnan(phi)]
            row["fill_ratio_mean"] = float(np.nanmean(valid_phi)) if len(valid_phi) else np.nan
            row["fill_ratio_p25"]  = float(np.nanpercentile(valid_phi, 25)) if len(valid_phi) else np.nan
            row["fill_ratio_p75"]  = float(np.nanpercentile(valid_phi, 75)) if len(valid_phi) else np.nan
        return row

    # ── full run ─────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        years = self.cfg.years or self._discover_years()
        if not years:
            log.error("No intertidal_h_*.tif found in %s", self.cfg.elevation_dir)
            return pd.DataFrame()

        rows = [r for y in years if (r := self.process_year(y)) is not None]
        if not rows:
            log.error("No years processed — check elevation_dir, datums_dir, and file templates.")
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

        # Normalised prism  P*(t) = P(t) / P(t_ref)
        ref_year = self.cfg.reference_year or int(df["year"].iloc[0])
        ref_row  = df[df["year"] == ref_year]
        if not ref_row.empty:
            p_ref = ref_row["P_vol_m3"].values[0]
            df["P_normalised"] = df["P_vol_m3"] / p_ref if p_ref > 0 else np.nan
        else:
            log.warning("Reference year %d not in results; P_normalised not computed.", ref_year)
            df["P_normalised"] = np.nan

        # Rate of change  ΔP/Δt  (Sen's slope, m³ yr⁻¹)
        df["dP_dt_m3_yr"] = _sens_slope_series(df["year"], df["P_vol_m3"])

        out_csv = self.cfg.output_dir / "tidal_prism_summary.csv"
        df.to_csv(out_csv, index=False, float_format="%.4f")
        log.info("Summary → %s", out_csv)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Sen's slope helper
# ─────────────────────────────────────────────────────────────────────────────
def _sens_slope_series(years: pd.Series, values: pd.Series) -> pd.Series:
    """Return a constant Series with the Sen's slope value (m³ yr⁻¹)."""
    x = years.values.astype(float)
    y = values.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return pd.Series(np.nan, index=values.index)
    from scipy.stats import theilslopes
    slope = theilslopes(y[mask], x[mask]).slope
    return pd.Series(slope, index=values.index)
