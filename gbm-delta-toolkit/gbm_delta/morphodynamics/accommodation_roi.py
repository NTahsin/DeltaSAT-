# -*- coding: utf-8 -*-
"""
intertidal_accommodation_space_roi.py
=====================================
Per-ROI intertidal accommodation space for the GBM delta study
(JGR Earth Surface, Section 3.2).

Computes, within each polygon of a shapefile (e.g. roi_clean.shp):
    a(i,j) = max(MHSW − z, 0)            pointwise depth   [m]
    φ(i,j) = (z − MLSW)/(MHSW − MLSW)    filling ratio     [0,1]
    V_acc  = Σ a(i,j) · ΔA               integrated volume [m³]

Inputs (per year)
-----------------
  intertidal_h_{year}.tif      reconciled intertidal elevation z
  mhsw_{year}.tif              pixel-wise MHSW
  mlsw_{year}.tif              pixel-wise MLSW
  occurrence_dry_{year}.tif    water occurrence f ∈ [0,1]

Outputs (per ROI)
-----------------
  accommodation_depth_{roi}_{year}.tif   clipped a(i,j) [m]
  fill_ratio_{roi}_{year}.tif            clipped φ      [0,1]
  accommodation_summary_{roi}.csv        per-year scalars
  accommodation_summary_all_rois.csv     long-form combined table

NOTE
----
All input rasters (elevation + datums + occurrence) must share the same
grid (CRS, transform, resolution). The reconciler typically preserves
the gridded run's StudyGrid — this is assumed. A shape-mismatch is
logged and the (ROI, year) is skipped.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd

log = logging.getLogger("accommodation_roi")

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


# ─────────────────────────────────────────────────────────────────────────────
# ROI loading
# ─────────────────────────────────────────────────────────────────────────────
def load_rois(path, name_field: str = "name") -> gpd.GeoDataFrame:
    """Load polygon ROIs from .shp / .geojson / .gpkg."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    gdf = gpd.read_file(p)

    if name_field not in gdf.columns:
        for alt in ("Name", "NAME", "roi", "ROI",
                    "label", "Label", "id", "Id", "ID"):
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: name_field})
                break
        else:
            gdf[name_field] = [f"{p.stem}_{i}" for i in range(len(gdf))]

    gdf = gdf[gdf.geometry.type.isin(("Polygon", "MultiPolygon"))]\
            .reset_index(drop=True)
    if gdf.empty:
        raise ValueError(f"No polygon features in {p}")
    return gdf[[name_field, "geometry"]].copy()


def _slug(name: str) -> str:
    s = _SAFE.sub("_", str(name).strip()).strip("_")
    return s or "roi"


# ─────────────────────────────────────────────────────────────────────────────
# Raster helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clip_raster(path: Path, geoms) -> Tuple[np.ndarray, dict]:
    """Clip raster to *geoms* (must already be in raster CRS)."""
    with rasterio.open(path) as src:
        arr, transform = rio_mask(src, geoms, crop=True, filled=True,
                                  nodata=src.nodata)
        prof = src.profile.copy()
        nd   = src.nodata
    a = arr[0].astype(np.float32)
    if nd is not None and not np.isnan(nd):
        a[a == nd] = np.nan
    prof.update(height=a.shape[0], width=a.shape[1], transform=transform)
    return a, prof


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
class ROIAccommodationConfig:
    elevation_dir:   Path                   # reconciled intertidal_h_{year}.tif
    output_dir:      Path
    roi_shapefile:   Path                   # roi_clean.shp
    datums_dir:      Optional[Path] = None  # mhsw/mlsw/occurrence; default = elevation_dir
    years:           Optional[List[int]] = None
    elevation_glob:  str = "intertidal_h_{year}.tif"
    mhsw_glob:       str = "mhsw_{year}.tif"
    mlsw_glob:       str = "mlsw_{year}.tif"
    occurrence_glob: str = "occurrence_dry_{year}.tif"
    occ_low:         float = 0.0
    occ_high:        float = 1.0
    name_field:      str  = "name"
    overwrite:       bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Calculator
# ─────────────────────────────────────────────────────────────────────────────
class ROIAccommodationCalculator:
    """Per-ROI accommodation depth, volume, and filling ratio."""

    def __init__(self, cfg: ROIAccommodationConfig):
        self.cfg = cfg
        self.cfg.elevation_dir = Path(self.cfg.elevation_dir)
        self.cfg.datums_dir    = (Path(self.cfg.datums_dir)
                                  if self.cfg.datums_dir
                                  else self.cfg.elevation_dir)
        self.cfg.output_dir    = Path(self.cfg.output_dir)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.rois = load_rois(self.cfg.roi_shapefile, self.cfg.name_field)
        log.info("Loaded %d ROI(s) from %s: %s",
                 len(self.rois), Path(self.cfg.roi_shapefile).name,
                 ", ".join(self.rois[self.cfg.name_field].astype(str)))

    # -------------------------------------------------------------------- #
    def _discover_years(self) -> List[int]:
        ys = []
        for p in self.cfg.elevation_dir.glob("intertidal_h_*.tif"):
            try:
                ys.append(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
        return sorted(set(ys))

    # -------------------------------------------------------------------- #
    def _process_one(self, roi_name: str, geoms, year: int
                     ) -> Optional[Dict]:
        z_p   = self.cfg.elevation_dir / self.cfg.elevation_glob.format(year=year)
        H_p   = self.cfg.datums_dir    / self.cfg.mhsw_glob.format(year=year)
        L_p   = self.cfg.datums_dir    / self.cfg.mlsw_glob.format(year=year)
        occ_p = self.cfg.datums_dir    / self.cfg.occurrence_glob.format(year=year)

        for p in (z_p, H_p, L_p):
            if not p.exists():
                log.warning("[%s] Year %d: missing %s — skipped.",
                            roi_name, year, p.name)
                return None

        slug    = _slug(roi_name)
        a_path  = self.cfg.output_dir / f"accommodation_depth_{slug}_{year}.tif"
        ph_path = self.cfg.output_dir / f"fill_ratio_{slug}_{year}.tif"

        # Clip all inputs to the ROI
        try:
            z, prof = _clip_raster(z_p, geoms)
            H, _    = _clip_raster(H_p, geoms)
            L, _    = _clip_raster(L_p, geoms)
        except ValueError as exc:
            log.warning("[%s] Year %d: ROI does not overlap raster (%s).",
                        roi_name, year, exc)
            return None

        occ = None
        if occ_p.exists():
            try:
                occ, _ = _clip_raster(occ_p, geoms)
            except Exception as exc:
                log.warning("[%s] Year %d: occurrence clip failed (%s) — "
                            "using elevation-only mask.", roi_name, year, exc)

        if not (z.shape == H.shape == L.shape):
            log.error("[%s] Year %d: grid mismatch z=%s H=%s L=%s — "
                      "ensure elevation/datums share the same grid.",
                      roi_name, year, z.shape, H.shape, L.shape)
            return None

        base = np.isfinite(z) & np.isfinite(H) & np.isfinite(L)
        if occ is not None and occ.shape == z.shape:
            mask = base & (occ > self.cfg.occ_low) & (occ < self.cfg.occ_high)
        else:
            mask = base

        # a(i,j) and φ(i,j)
        a = np.full(z.shape, np.nan, dtype=np.float32)
        a[mask] = np.maximum(H[mask] - z[mask], 0.0)

        phi = np.full(z.shape, np.nan, dtype=np.float32)
        rng = H - L
        ok  = mask & (rng > 1e-3)
        phi[ok] = np.clip((z[ok] - L[ok]) / rng[ok], 0.0, 1.0)

        if (not a_path.exists()) or (not ph_path.exists()) or self.cfg.overwrite:
            _write(a,   a_path,  prof)
            _write(phi, ph_path, prof)
            log.info("[%s] Year %d: wrote %s, %s",
                     roi_name, year, a_path.name, ph_path.name)
        else:
            log.info("[%s] Year %d: rasters exist — re-summarising only.",
                     roi_name, year)

        # Scalars
        n = int(mask.sum())
        if n == 0:
            return dict(roi=str(roi_name), year=int(year), n_pixels=0,
                        area_km2=0.0, V_acc_m3=np.nan, V_acc_Mm3=np.nan,
                        mean_a_m=np.nan, median_a_m=np.nan, std_a_m=np.nan,
                        p10_a_m=np.nan, p90_a_m=np.nan, mean_phi=np.nan)
        px    = _pixel_area_m2(prof)
        a_v   = a[mask]
        phi_v = phi[ok]
        V     = float(np.nansum(a_v) * px)
        return dict(
            roi        = str(roi_name),
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

    # -------------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        years = self.cfg.years or self._discover_years()
        if not years:
            log.error("No intertidal_h_*.tif rasters in %s",
                      self.cfg.elevation_dir)
            return pd.DataFrame()

        # Reproject ROIs to raster CRS once
        ref = self.cfg.elevation_dir / self.cfg.elevation_glob.format(year=years[0])
        with rasterio.open(ref) as src:
            raster_crs = src.crs
        rois = (self.rois.to_crs(raster_crs)
                if self.rois.crs is not None
                and str(self.rois.crs) != str(raster_crs)
                else self.rois)

        all_rows = []
        for _, row in rois.iterrows():
            roi_name = row[self.cfg.name_field]
            geoms    = [row.geometry.__geo_interface__]
            log.info("=== ROI: %s ===", roi_name)

            roi_rows = []
            for y in years:
                r = self._process_one(roi_name, geoms, y)
                if r is not None:
                    roi_rows.append(r)

            if roi_rows:
                df_roi  = (pd.DataFrame(roi_rows)
                             .sort_values("year").reset_index(drop=True))
                csv_p   = self.cfg.output_dir / \
                          f"accommodation_summary_{_slug(roi_name)}.csv"
                df_roi.to_csv(csv_p, index=False, float_format="%.4f")
                log.info("Per-ROI summary → %s", csv_p)
                all_rows.extend(roi_rows)

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df = df.sort_values(["roi", "year"]).reset_index(drop=True)
            out = self.cfg.output_dir / "accommodation_summary_all_rois.csv"
            df.to_csv(out, index=False, float_format="%.4f")
            log.info("Combined summary → %s", out)
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
        description="Per-ROI intertidal accommodation space — GBM delta.")
    p.add_argument("--elevation-dir", required=True,
                   help="Folder with reconciled intertidal_h_{year}.tif")
    p.add_argument("--datums-dir", default=None,
                   help="Folder with mhsw/mlsw/occurrence_dry_{year}.tif "
                        "(defaults to --elevation-dir)")
    p.add_argument("--roi-shapefile", required=True,
                   help="Polygon shapefile (e.g. roi_clean.shp)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name-field", default="name",
                   help="ROI name attribute (default: name)")
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    cfg = ROIAccommodationConfig(
        elevation_dir = Path(a.elevation_dir),
        datums_dir    = Path(a.datums_dir) if a.datums_dir else None,
        roi_shapefile = Path(a.roi_shapefile),
        output_dir    = Path(a.output_dir),
        name_field    = a.name_field,
        years         = a.years,
        overwrite     = a.overwrite,
    )
    df = ROIAccommodationCalculator(cfg).run()
    if not df.empty:
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    _cli()
