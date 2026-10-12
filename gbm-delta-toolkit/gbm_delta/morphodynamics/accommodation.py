# -*- coding: utf-8 -*-
"""
intertidal_accommodation_space.py
=================================
Pixel-wise intertidal accommodation space for the GBM delta study
(JGR Earth Surface, Section 3.2).

Definition
----------
Accommodation space = vertical space available between bed elevation and
the upper tidal datum (MHSW) into which sediment can accumulate before
exceeding base level (Jervey 1988; Muto & Steel 2000; Friedrichs 2011).

  Pointwise depth :  a(i,j) = max(MHSW(i,j) − z(i,j), 0)        [m]
  Volumetric      :  V_acc  = Σ a(i,j) · ΔA   over intertidal Ω [m³]
  Filling ratio   :  φ(i,j) = (z − MLSW)/(MHSW − MLSW) ∈ [0,1]

Inputs (per year, produced by intertidal_elevation.py)
------------------------------------------------------
  elevation_{year}.tif    intertidal elevation z      [m]
  mhsw_{year}.tif         pixel-wise MHSW             [m]
  mlsw_{year}.tif         pixel-wise MLSW             [m]
  occurrence_{year}.tif   water occurrence f ∈ [0,1]  (intertidal: 0<f<1)

Outputs
-------
  accommodation_depth_{year}.tif   a(i,j)         [m]
  fill_ratio_{year}.tif            φ(i,j)         [0,1]
  accommodation_summary.csv        per-year scalars
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio

log = logging.getLogger("accommodation")


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None and not np.isnan(nd):
            a[a == nd] = np.nan
        return a, src.profile.copy()


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
class AccommodationConfig:
    elevation_dir:   Path                              # reconciled intertidal_h_{year}.tif
    output_dir:      Path                              # results dir
    datums_dir:      Optional[Path] = None             # MHSW/MLSW/occurrence (default = elevation_dir)
    years:           Optional[List[int]] = None        # None = autodetect
    elevation_glob:  str = "intertidal_h_{year}.tif"
    mhsw_glob:       str = "mhsw_{year}.tif"
    mlsw_glob:       str = "mlsw_{year}.tif"
    occurrence_glob: str = "occurrence_dry_{year}.tif" # v3 reconciler default
    occ_low:         float = 0.0          # intertidal mask: occ_low < f < occ_high
    occ_high:        float = 1.0
    overwrite:       bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Calculator
# ─────────────────────────────────────────────────────────────────────────────
class AccommodationSpaceCalculator:
    """Per-year intertidal accommodation depth, volume, and filling ratio."""

    def __init__(self, cfg: AccommodationConfig):
        self.cfg = cfg
        self.cfg.elevation_dir = Path(self.cfg.elevation_dir)
        self.cfg.datums_dir    = (Path(self.cfg.datums_dir)
                                  if self.cfg.datums_dir
                                  else self.cfg.elevation_dir)
        self.cfg.output_dir    = Path(self.cfg.output_dir)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- #
    def _discover_years(self) -> List[int]:
        ys = []
        for p in self.cfg.elevation_dir.glob("intertidal_h_*.tif"):
            try:
                ys.append(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
        return sorted(set(ys))

    def _path(self, glob: str, year: int) -> Path:
        return self.cfg.elevation_dir / glob.format(year=year)

    # ----------------------------------------------------------------- #
    def process_year(self, year: int) -> Optional[Dict[str, float]]:
        z_p   = self.cfg.elevation_dir / self.cfg.elevation_glob.format(year=year)
        H_p   = self.cfg.datums_dir    / self.cfg.mhsw_glob.format(year=year)
        L_p   = self.cfg.datums_dir    / self.cfg.mlsw_glob.format(year=year)
        occ_p = self.cfg.datums_dir    / self.cfg.occurrence_glob.format(year=year)

        for p in (z_p, H_p, L_p):
            if not p.exists():
                log.warning("Year %d: missing %s — skipped.", year, p.name)
                return None

        a_path  = self.cfg.output_dir / f"accommodation_depth_{year}.tif"
        ph_path = self.cfg.output_dir / f"fill_ratio_{year}.tif"
        if a_path.exists() and ph_path.exists() and not self.cfg.overwrite:
            log.info("Year %d: outputs exist — re-summarising only.", year)
            return self._summarise_existing(year, a_path, ph_path)

        # ---- Load ---------------------------------------------------- #
        z, prof = _read(z_p)
        H, _    = _read(H_p)
        L, _    = _read(L_p)

        # ---- Intertidal mask ---------------------------------------- #
        base = np.isfinite(z) & np.isfinite(H) & np.isfinite(L)
        if occ_p.exists():
            occ, _ = _read(occ_p)
            mask = base & (occ > self.cfg.occ_low) & (occ < self.cfg.occ_high)
        else:
            log.info("Year %d: occurrence raster absent — using all "
                     "finite-elevation pixels as intertidal mask.", year)
            mask = base

        # ---- a(i,j) = max(MHSW − z, 0) ------------------------------ #
        a = np.full(z.shape, np.nan, dtype=np.float32)
        a[mask] = np.maximum(H[mask] - z[mask], 0.0)

        # ---- φ(i,j) ∈ [0,1] ---------------------------------------- #
        phi = np.full(z.shape, np.nan, dtype=np.float32)
        rng = H - L
        ok  = mask & (rng > 1e-3)
        phi[ok] = np.clip((z[ok] - L[ok]) / rng[ok], 0.0, 1.0)

        _write(a,   a_path,  prof)
        _write(phi, ph_path, prof)
        log.info("Year %d: wrote %s, %s", year, a_path.name, ph_path.name)

        return self._scalars(year, prof, mask, a, phi, ok)

    # ----------------------------------------------------------------- #
    @staticmethod
    def _scalars(year, prof, mask, a, phi, ok) -> Dict[str, float]:
        n  = int(mask.sum())
        if n == 0:
            return dict(year=year, n_pixels=0, area_km2=0.0,
                        V_acc_m3=np.nan, V_acc_Mm3=np.nan,
                        mean_a_m=np.nan, median_a_m=np.nan, std_a_m=np.nan,
                        p10_a_m=np.nan, p90_a_m=np.nan, mean_phi=np.nan)
        px    = _pixel_area_m2(prof)
        a_v   = a[mask]
        phi_v = phi[ok]
        V     = float(np.nansum(a_v) * px)
        return dict(
            year       = int(year),
            n_pixels   = n,
            area_km2   = n * px / 1e6,
            V_acc_m3   = V,
            V_acc_Mm3  = V / 1e6,
            mean_a_m   = float(np.nanmean(a_v)),
            median_a_m = float(np.nanmedian(a_v)),
            std_a_m    = float(np.nanstd(a_v)),
            p10_a_m    = float(np.nanpercentile(a_v, 10)),
            p90_a_m    = float(np.nanpercentile(a_v, 90)),
            mean_phi   = float(np.nanmean(phi_v)) if phi_v.size else np.nan,
        )

    # ----------------------------------------------------------------- #
    def _summarise_existing(self, year, a_path, ph_path):
        a,   prof = _read(a_path)
        phi, _    = _read(ph_path)
        mask = np.isfinite(a)
        ok   = np.isfinite(phi)
        return self._scalars(year, prof, mask, a, phi, ok)

    # ----------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        years = self.cfg.years or self._discover_years()
        if not years:
            log.error("No intertidal_h_*.tif rasters in %s", self.cfg.elevation_dir)
            return pd.DataFrame()

        rows = []
        for y in years:
            r = self.process_year(y)
            if r is not None:
                rows.append(r)

        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        out_csv = self.cfg.output_dir / "accommodation_summary.csv"
        df.to_csv(out_csv, index=False, float_format="%.4f")
        log.info("Summary CSV → %s", out_csv)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _cli():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(
        description="Intertidal accommodation space — GBM delta study.")
    p.add_argument("--elevation-dir", required=True,
                   help="Folder with reconciled intertidal_h_{year}.tif rasters")
    p.add_argument("--datums-dir", default=None,
                   help="Folder with mhsw/mlsw/occurrence_dry_{year}.tif "
                        "(defaults to --elevation-dir)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = AccommodationConfig(
        elevation_dir = Path(args.elevation_dir),
        datums_dir    = Path(args.datums_dir) if args.datums_dir else None,
        output_dir    = Path(args.output_dir),
        years         = args.years,
        overwrite     = args.overwrite,
    )
    df = AccommodationSpaceCalculator(cfg).run()
    if not df.empty:
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    _cli()
