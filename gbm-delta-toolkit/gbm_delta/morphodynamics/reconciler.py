r"""
intertidal_reconciler.py
========================
Per-year intertidal extent reconciliation.

Combines:
  - frequency-of-inundation mask  (Sagar et al. 2017)
  - JRC permanent-water mask      (Pekel et al. 2016) — optional
  - Bishop-Taylor h_median raster (your elevation_bt_YYYY.tif)
  - per-pixel MHSW / MLSW rasters (your mhsw_YYYY.tif, mlsw_YYYY.tif)

Outputs per year:
  intertidal_YYYY.tif       — uint8: 0=outside, 1=intertidal
  intertidal_h_YYYY.tif     — float32: elevation (m) where intertidal=1
  intertidal_summary.csv    — areas + diagnostics per year

Usage
-----
    from gbm_delta.morphodynamics.reconciler import IntertidalReconciler
    rec = IntertidalReconciler(
        channel_masks_dir = r"...\02_River Extracted",
        elevation_dir     = r"...\bishop taylor tiffs",
        mhsw_dir          = r"...\mhsw_grids",
        mlsw_dir          = r"...\mlsw_grids",
        output_dir        = r"...\07_intertidal_reconciled",
        jrc_occurrence    = r"...\JRC_occurrence.tif",  # optional, None to skip
    )
    rec.run(years=range(1996, 2021))
"""

from __future__ import annotations
import os, re, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


class IntertidalReconciler:
    def __init__(
        self,
        channel_masks_dir: str,
        elevation_dir: str,
        mhsw_dir: str,
        mlsw_dir: str,
        output_dir: str,
        jrc_occurrence: str | None = None,
        min_scenes: int = 5,
        f_low: float = 0.05,
        f_high: float = 0.95,
        jrc_water_threshold: int = 90,
        pixel_size_m: float = 30.0,
        water_value: float = -999.0,
        land_value:  float = 999.0,
    ):
        self.channel_masks_dir = channel_masks_dir
        self.elevation_dir = elevation_dir
        self.mhsw_dir = mhsw_dir
        self.mlsw_dir = mlsw_dir
        self.output_dir = output_dir
        self.jrc_occurrence = jrc_occurrence
        self.min_scenes = min_scenes
        self.f_low = f_low
        self.f_high = f_high
        self.jrc_thr = jrc_water_threshold
        self.px2km2 = pixel_size_m ** 2 / 1e6
        self.water_value = water_value
        self.land_value  = land_value
        os.makedirs(output_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    def _scenes_by_year(self):
        files = sorted(glob.glob(os.path.join(self.channel_masks_dir,
                                              "**", "*channel_mask_*.tif"),
                                 recursive=True))
        rx = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
        out = defaultdict(list)
        for f in files:
            m = rx.search(os.path.basename(f))
            if m: out[int(m.group(1))].append(f)
        return out

    def _read(self, path, ref=None):
        """Read raster; reproject to ref grid if needed."""
        with rasterio.open(path) as src:
            if ref is None:
                return src.read(1).astype(np.float32), src.profile
            arr = np.empty((ref["height"], ref["width"]), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref["transform"], dst_crs=ref["crs"],
                resampling=Resampling.nearest,
            )
            return arr, None

    # ────────────────────────────────────────────────────────────────────
    def _frequency_mask(self, scene_paths, ref):
        """Return (n_valid, n_water) on the ref grid."""
        n_valid = np.zeros((ref["height"], ref["width"]), dtype=np.int16)
        n_water = np.zeros_like(n_valid)
        for p in scene_paths:
            arr, _ = self._read(p, ref)
            valid = ~np.isnan(arr)
            water = valid & (arr == self.water_value)
            n_valid += valid
            n_water += water
        return n_valid, n_water

    # ────────────────────────────────────────────────────────────────────
    def run_year(self, year: int) -> dict:
        scenes_by_year = self._scenes_by_year()
        scene_paths = scenes_by_year.get(year, [])
        if not scene_paths:
            print(f"[{year}] no scenes — skipped"); return {}

        elev_path = os.path.join(self.elevation_dir, f"intertidal_h_{year}.tif")
        mhsw_path = os.path.join(self.mhsw_dir,      f"mhsw_{year}.tif")
        mlsw_path = os.path.join(self.mlsw_dir,      f"mlsw_{year}.tif")
        for p in (elev_path, mhsw_path, mlsw_path):
            if not os.path.exists(p):
                print(f"[{year}] missing {p} — skipped"); return {}

        # Reference grid = elevation raster
        with rasterio.open(elev_path) as src:
            ref = {"height": src.height, "width": src.width,
                   "transform": src.transform, "crs": src.crs,
                   "profile": src.profile}
        elev, _ = self._read(elev_path)
        mhsw, _ = self._read(mhsw_path, ref)
        mlsw, _ = self._read(mlsw_path, ref)

        # Step 1-2: frequency
        n_valid, n_water = self._frequency_mask(scene_paths, ref)
        sufficient = n_valid >= self.min_scenes
        with np.errstate(divide="ignore", invalid="ignore"):
            freq = np.where(sufficient, n_water / np.maximum(n_valid, 1), np.nan)

        # Step 3: candidate intertidal by frequency
        cand = sufficient & (freq >= self.f_low) & (freq <= self.f_high)

        # Step 4: JRC constraint (optional)
        if self.jrc_occurrence:
            jrc, _ = self._read(self.jrc_occurrence, ref)
            cand &= (jrc < self.jrc_thr) | np.isnan(jrc)

        # Step 5: elevation between MLSW and MHSW
        elev_ok = ~np.isnan(elev) & ~np.isnan(mhsw) & ~np.isnan(mlsw)
        n_elev_valid = int(elev_ok.sum())
        if n_elev_valid >= 1000:
            cand &= elev_ok & (elev >= mlsw) & (elev <= mhsw)
        else:
            print(f"  [{year}] elev raster too sparse ({n_elev_valid} px) — frequency-only")
        

        # Outputs
        out_mask = cand.astype(np.uint8)
        out_elev = np.where(cand, elev, np.nan).astype(np.float32)

        prof = ref["profile"].copy()
        prof.update(dtype="uint8", count=1, nodata=0, compress="lzw")
        with rasterio.open(os.path.join(self.output_dir,
                                        f"intertidal_{year}.tif"), "w", **prof) as dst:
            dst.write(out_mask, 1)
        prof.update(dtype="float32", nodata=np.nan)
        with rasterio.open(os.path.join(self.output_dir,
                                        f"intertidal_h_{year}.tif"), "w", **prof) as dst:
            dst.write(out_elev, 1)

        stats = {
            "year": year,
            "n_scenes": len(scene_paths),
            "px_intertidal": int(cand.sum()),
            "km2_intertidal": round(int(cand.sum()) * self.px2km2, 3),
            "elev_mean_m": float(np.nanmean(out_elev)) if cand.any() else np.nan,
            "elev_p10_m":  float(np.nanpercentile(out_elev, 10)) if cand.any() else np.nan,
            "elev_p90_m":  float(np.nanpercentile(out_elev, 90)) if cand.any() else np.nan,
            "px_insufficient_obs": int((~sufficient).sum()),
        }
        print(f"[{year}] scenes={stats['n_scenes']:>3}  "
              f"intertidal={stats['km2_intertidal']:>8.2f} km²  "
              f"elev μ={stats['elev_mean_m']:.2f} m")
        return stats

    # ────────────────────────────────────────────────────────────────────
    def run(self, years=None) -> list:
        scenes = self._scenes_by_year()
        ys = sorted(years) if years else sorted(scenes.keys())
        rows = [self.run_year(y) for y in ys]
        rows = [r for r in rows if r]

        import csv
        path = os.path.join(self.output_dir, "intertidal_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n→ summary  {path}")
        return rows
