# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:02:04 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
channel_metrics_lite.py
=======================
Reduced version of channel_metrics.py: channel migration rates for the
full estuary and for ROI polygons, using ONLY

    1. bankline rasters     bankline_YYYY-MM-DD.tif   (uint8, >0 = bank)
                            — multiple per year are averaged into one
                              annual composite
    2. intertidal rasters   intertidal_filtered_{year}.tif
                            (0 = land, 1 = river, 2 = intertidal)

The intertidal rasters are used for:
    * intertidal area per year (class 2 → km²)
    * channel-mask stability check (class 2 ∪ 1) — the bankline composite
      is intersected with the wet mask so that spurious edges far from
      the channel are discarded before migration is computed.
    * an optional bankline fallback: if a year has no bankline_*.tif but
      does have an intertidal raster, the land/wet boundary
      (land=0 vs river∪intertidal) is used as the bankline for that year.

Metrics per (roi, year-pair)
----------------------------
    migration_mean_m_yr / median / p95 / mm  — EDT-based migration rate
    n_bank_pixels_a / n_bank_pixels_b       — bank pixel counts

Run
---
    from channel_metrics_lite import MetricsConfig, ChannelMigrationAnalyzer

    cfg = MetricsConfig(
        bankline_dir=..., intertidal_dir=..., output_dir=...,
        intertidal_glob="intertidal_filtered_{year}.tif")
    ana = ChannelMigrationAnalyzer(cfg, rois="main_side_channels.shp")
    df = ana.run(years=range(1996, 2019))
"""

#from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.ndimage import distance_transform_edt

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

log = logging.getLogger(__name__)
PathLike = Union[str, Path]

# ── reused helpers from channel_metrics (no duplication) ────────────────────
from gbm_delta.channel.metrics import (_parse_year, _group_by_year, _read_masked,
                             _reproject_rois, compute_migration, _safe)

# intertidal raster classes
CLASS_LAND, CLASS_RIVER, CLASS_INTERTIDAL = 0, 1, 2


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MetricsConfig:
    bankline_dir:    Path
    intertidal_dir:  Path
    output_dir:      Path

    bankline_glob:   str = "bankline_*.tif"
    intertidal_glob: str = "intertidal_filtered_{year}.tif"

    pixel_size_m: float = 30.0
    bankline_occ_threshold: int = 1    # min #images per year to flag bank
    use_wet_mask: bool = True         # clip banklines to (river|intertidal)
    use_intertidal_fallback: bool = True  # land/wet boundary if no banklines
    max_edge_buffer_px: int = 2       # drop bank pixels within N px of ROI edge


# ─────────────────────────────────────────────────────────────────────────────
# Raster helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_class_raster(path, geom=None):
    with rasterio.open(path) as src:
        px_area = _raster_pixel_area(src)
        if geom is not None:
            try:
                arr, _ = rio_mask(src, geom, crop=True, filled=True,
                                  nodata=255)
                arr = arr[0]
            except ValueError:
                return None, px_area      # ROI outside this raster
        else:
            arr = src.read(1)
    return arr.astype(np.uint8), px_area


def _raster_pixel_area(src) -> float:
    t = src.transform
    if src.crs is not None and src.crs.is_geographic:
        lat = (src.bounds.top + src.bounds.bottom) / 2.0
        return (abs(t.a) * 111.32 * np.cos(np.radians(lat))) * \
               (abs(t.e) * 111.32)
    return abs(t.a * t.e) / 1e6


def intertidal_area_km2(classes: np.ndarray, px_area_km2: float) -> float:
    return float((classes == CLASS_INTERTIDAL).sum() * px_area_km2)


def wet_mask(classes: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """River ∪ Intertidal = wet."""
    if classes is None:
        return None
    return (classes == CLASS_RIVER) | (classes == CLASS_INTERTIDAL)


def bankline_from_classes(classes: np.ndarray) -> np.ndarray:
    """Land/wet boundary (dilated 1 px) as a bankline fallback."""
    wet = wet_mask(classes)
    # pixels that are wet with at least one non-wet 4-neighbour
    pad = np.pad(wet, 1, constant_values=False)
    nb = (pad[:-2, 1:-1] & ~wet) | (pad[2:, 1:-1] & ~wet) | \
         (pad[1:-1, :-2] & ~wet) | (pad[1:-1, 2:] & ~wet)
    return (wet & nb).astype(np.uint8)


def _trim_edges(mask: np.ndarray, buf: int) -> np.ndarray:
    """Zero out pixels within `buf` of the array edge (ROI-clip artefacts)."""
    out = mask.copy()
    if buf > 0:
        out[:buf, :] = False; out[-buf:, :] = False
        out[:, :buf] = False; out[:, -buf:] = False
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────
class ChannelMigrationAnalyzer:
    def __init__(self, cfg: MetricsConfig,
                 rois: Optional[PathLike] = None,
                 roi_name_field: str = "name"):
        self.cfg = cfg
        self._geom_cache: Dict[tuple, list] = {}
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        if rois is None:
            self.rois = None
        else:
            from gbm_delta.channel.metrics import load_rois
            self.rois = load_rois(rois, name_field=roi_name_field)

    # ── file discovery ────────────────────────────────────────────────────
    def _banklines(self) -> Dict[int, List[Path]]:
        files = sorted(self.cfg.bankline_dir.glob(self.cfg.bankline_glob))
        return _group_by_year(files)

    def _intertidal(self, year: int) -> Optional[Path]:
        g = self.cfg.intertidal_glob
        p = self.cfg.intertidal_dir / g.format(year=year)
        alt = self.cfg.intertidal_dir / str(year) / g.format(year=year)
        return p if p.exists() else (alt if alt.exists() else None)

    def _geoms(self, raster_path: Path, roi_name: str):
        if self.rois is None:
            return None
        with rasterio.open(raster_path) as src:
            tgt = src.crs
        key = (roi_name, str(tgt))
        if key not in self._geom_cache:
            sub = _reproject_rois(self.rois[self.rois["name"] == roi_name], tgt)
            self._geom_cache[key] = [g.__geo_interface__ for g in sub.geometry]
        return self._geom_cache[key]

    # ── annual bankline composite (wet-masked) ─────────────────────────────
    def _bankline_year(self, year, roi_name, bl_by_year):
        cls_p = self._intertidal(year)
        classes = None
        if cls_p is not None:
            classes, _ = _read_class_raster(cls_p,
                                            geom=self._geoms(cls_p, roi_name))

        if year in bl_by_year:
            stacks = []
            ref_shape = None
            for f in bl_by_year[year]:
                arr, _, _, _ = _read_masked(f, geom=self._geoms(f, roi_name),
                                            nodata_fill=0)
                arr = (arr > 0).astype(np.uint8)
                if ref_shape is None:
                    ref_shape = arr.shape
                elif arr.shape != ref_shape:
                    h = min(arr.shape[0], ref_shape[0])
                    w = min(arr.shape[1], ref_shape[1])
                    arr = arr[:h, :w]
                    stacks = [s[:h, :w] for s in stacks]
                    ref_shape = (h, w)
                stacks.append(arr)
            bank = (np.stack(stacks).sum(0) >=
                    self.cfg.bankline_occ_threshold).astype(np.uint8)
        elif classes is not None and self.cfg.use_intertidal_fallback:
            bank = bankline_from_classes(classes)
        else:
            bank = None

         # align bankline and class grids to their common overlap
        if bank is not None and classes is not None:
            h = min(bank.shape[0], classes.shape[0])
            w = min(bank.shape[1], classes.shape[1])
            bank    = bank[:h, :w]
            classes = classes[:h, :w]

        if bank is not None:
            if self.cfg.use_wet_mask and classes is not None:
                wm = wet_mask(classes)
                bank = (bank.astype(bool) & wm).astype(np.uint8)
            bank = _trim_edges(bank.astype(bool),
                               self.cfg.max_edge_buffer_px).astype(np.uint8)
        return bank, classes

    # ── main loop ──────────────────────────────────────────────────────────
    def run(self, years: Optional[Iterable[int]] = None) -> pd.DataFrame:
        bl_by_year = self._banklines()
        if years is None:
            years = sorted(set(list(bl_by_year) +
                               [y for y in
                                (self._parse_y_from_intertidal()
                                 for _ in [0]) if y is not None]))
        years = sorted(set(years))
        log.info("ChannelMigrationAnalyzer: years %s", years)

        roi_names = (list(self.rois["name"])
                     if self.rois is not None else ["ALL"])

        rows: List[dict] = []
        for roi_name in roi_names:
            log.info("── ROI: %s ──", roi_name)
            prev_bank = prev_year = None

            # resolve geometry once per ROI from any available raster


            for y in years:
                bank, classes = self._bankline_year(y, roi_name, bl_by_year)
                row = dict(roi=roi_name, year=y)

                if classes is not None:
                    cls_p = self._intertidal(y)
                    with rasterio.open(cls_p) as src:
                        px_area = _raster_pixel_area(src)
                    row["intertidal_area_km2"] = \
                        intertidal_area_km2(classes, px_area)
                else:
                    row["intertidal_area_km2"] = np.nan

                if bank is not None and prev_bank is not None:
                    a, b = prev_bank, bank
                    h = min(a.shape[0], b.shape[0])
                    w = min(a.shape[1], b.shape[1])
                    mig = compute_migration(
                        a[:h, :w], b[:h, :w],
                        pixel_size_m=self.cfg.pixel_size_m,
                        dt_years=float(y - prev_year))
                    row.update({f"migration_{k}": v
                                for k, v in mig.items()})
                else:
                    row.update(migration_mean_m_yr=np.nan,
                               migration_mean_mm_yr=np.nan,
                               migration_median_m_yr=np.nan,
                               migration_p95_m_yr=np.nan)

                rows.append(row)
                if bank is not None:
                    prev_bank, prev_year = bank, y

        df = pd.DataFrame(rows)
        if df.empty:
            log.warning("No metrics computed — check input paths.")
            return df

        df.to_csv(self.cfg.output_dir / "migration_all_rois.csv", index=False)
        for roi, sub in df.groupby("roi"):
            p = self.cfg.output_dir / f"migration_{_safe(roi)}.csv"
            sub.to_csv(p, index=False)
            log.info("Wrote %s", p)
        return df

    def _parse_y_from_intertidal(self):
        for f in sorted(self.cfg.intertidal_dir.glob("*.tif")):
            y = _parse_year(f.name)
            if y is not None:
                return y
        return None