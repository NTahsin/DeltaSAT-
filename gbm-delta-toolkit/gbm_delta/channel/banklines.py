"""
bankline_extractor.py
=====================
Batch-extract channel banklines from binary land-water GeoTIFFs.

Implements §Channel Network Extraction of methods_updated_v2.tex:

  1. Read binary land-water mask (water = 1, land = 0, NaN/nodata ignored)
  2. Optional morphological closing to bridge 1-px speckle gaps
  3. Connected-component labelling; keep the N largest water regions
     (N=1 → single main channel, matching the current script;
      N>1 → main + side channels in the same output file)
  4. Canny edge detection on the retained channel mask
  5. Exclude images with fewer than *min_edge_pixels* edge pixels
     (default 500, following Murray 2019 / Bishop-Taylor 2019)
  6. Save the edge raster as a single-band uint8 GeoTIFF with the input's
     original projection and transform

Outputs (consistent with channel_metrics.ChannelMetricsAnalyzer)
---------------------------------------------------------------
    <output_dir>/bankline_YYYY-MM-DD.tif       uint8  (0 or 255)
    <output_dir>/bankline_extraction_summary.csv
        one row per input file with the number of water pixels in the
        retained components, edge-pixel count, pass/fail flag, reason.
    <output_dir>/qc/<date>.png                 optional QC figures (opt-in)

ROI support
-----------
If a shapefile / KML / KMZ with one or more polygons is provided, the
connected-component step is performed **inside each polygon** and one
bankline raster per (date, ROI) is written to
    <output_dir>/<roi_name>/bankline_YYYY-MM-DD.tif
Useful when a noisy scene contains multiple disconnected water bodies
and the "largest component" rule would otherwise latch onto the wrong one.

Dependencies
------------
    pip install numpy pandas rasterio opencv-python-headless scipy tqdm geopandas
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.ndimage import label as cc_label, binary_closing

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Filename / date parsing
# ─────────────────────────────────────────────────────────────────────────────

_RE_DATE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")
_RE_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def _parse_date(name: str) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Return (datetime, canonical YYYY-MM-DD string) or (None, None)
    for a filename such as ``LW_ndwi_1996-12-07.tif`` or
    ``channel_mask_2003_03_15.tif``.
    """
    m = _RE_DATE.search(name)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt, dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    m = _RE_YEAR.search(name)
    if m:
        try:
            dt = datetime(int(m.group(0)), 1, 1)
            return dt, m.group(0)
        except ValueError:
            pass
    return None, None


def _pbar(iterable, desc: str = ""):
    return tqdm(iterable, desc=desc) if _TQDM else iterable


# ─────────────────────────────────────────────────────────────────────────────
# ROI loading (reuses load_rois from channel_metrics if available)
# ─────────────────────────────────────────────────────────────────────────────

def _load_rois(path: PathLike) -> "gpd.GeoDataFrame":
    """Thin wrapper that reuses channel_metrics.load_rois if importable."""
    try:
        from gbm_delta.channel.metrics import load_rois as _lr      # noqa: F401
        return _lr(path)
    except ImportError:
        if not _HAS_GPD:
            raise ImportError("geopandas is required for ROI support.")
        gdf = gpd.read_file(path)
        if "name" not in gdf.columns:
            gdf["name"] = [f"{Path(path).stem}_{i}" for i in range(len(gdf))]
        return gdf[["name", "geometry"]]


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction primitives
# ─────────────────────────────────────────────────────────────────────────────

def _read_binary_water(
    raster_path: Path,
    geom: Optional[Sequence] = None,
) -> Tuple[np.ndarray, rasterio.Affine, "rasterio.crs.CRS", dict]:
    """
    Read a binary LW GeoTIFF and return a uint8 water mask (1 = water).

    Recognised encodings:
      water = 1, land = 0  (primary — produced by the NDWI pipeline)
      water = 1, land = NaN (fallback)
      nodata pixels → 0 in the returned mask
    """
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        nd = src.nodata
        if geom is not None:
            arr, transform = rio_mask(src, geom, crop=True, filled=True,
                                      nodata=nd if nd is not None else 0)
            arr = arr[0]
        else:
            arr = src.read(1)
            transform = src.transform

    if arr.dtype.kind == "f":
        water = np.where(np.isnan(arr), 0, (arr == 1).astype(np.uint8))
    else:
        water = (arr == 1).astype(np.uint8)
        if nd is not None:
            water[arr == nd] = 0

    # update profile so it reflects the clipped window
    profile.update(height=water.shape[0], width=water.shape[1],
                   transform=transform, dtype="uint8", count=1,
                   nodata=None, compress="lzw")
    return water, transform, profile.get("crs"), profile


def _retain_largest_components(
    water: np.ndarray,
    n_keep: int = 1,
    min_component_pixels: int = 50,
) -> Tuple[np.ndarray, List[int]]:
    """
    Keep the *n_keep* largest 4-connected water components ≥
    *min_component_pixels* in size. Returns (mask_uint8_0_or_255, sizes).
    """
    if not water.any():
        return np.zeros_like(water, dtype=np.uint8), []

    labels, n = cc_label(water)
    if n == 0:
        return np.zeros_like(water, dtype=np.uint8), []

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0                                  # background

    # Indices of components that pass the min-size filter
    valid_idx = [i for i in range(1, len(sizes))
                 if sizes[i] >= min_component_pixels]
    # Sort by size (descending) and keep the top-N
    valid_idx.sort(key=lambda i: sizes[i], reverse=True)
    keep_idx = valid_idx[:n_keep]

    mask = np.isin(labels, keep_idx).astype(np.uint8) * 255
    return mask, [int(sizes[i]) for i in keep_idx]


def _canny_edges(
    mask_255: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
    close_radius: int = 0,
) -> np.ndarray:
    """Canny edge detection on a 0/255 uint8 mask."""
    if not _HAS_CV2:
        raise ImportError("opencv-python(-headless) is required for Canny.")
    src = mask_255
    if close_radius > 0:
        k = 2 * close_radius + 1
        kern = np.ones((k, k), np.uint8)
        src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kern)
    return cv2.Canny(src, low_threshold, high_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Config & orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BanklineConfig:
    """Configuration for :class:`BanklineExtractor`."""
    input_dir:   Path
    output_dir:  Path

    # Naming
    input_glob:  str = "*.tif"                 # matches LW_ndwi_*.tif / channel_mask_*.tif
    output_stem: str = "bankline"              # → bankline_YYYY-MM-DD.tif

    # Connected-component selection
    n_components:        int = 1               # N largest water blobs to keep
    min_component_pixels: int = 50             # discard components smaller than this

    # Canny + morphology
    canny_low:  int = 100
    canny_high: int = 200
    pre_close_radius: int = 0                  # morphological closing before Canny

    # QC / filtering
    min_edge_pixels: int = 500                 # methods paper threshold
    write_qc_png:    bool = False              # save side-by-side QC png per file

    # Dry-season-only filter (ignored if both are None)
    dry_season_months: Optional[Sequence[int]] = None       # e.g. (11, 12, 1, 2, 3, 4)

    # Housekeeping
    overwrite: bool = False


class BanklineExtractor:
    """
    Orchestrates bankline extraction for every LW raster in
    ``cfg.input_dir`` (optionally clipped to ROI polygons).

    Example
    -------
    >>> cfg = BanklineConfig(
    ...     input_dir  = Path(".../Landsat/LW"),
    ...     output_dir = Path(".../Landsat/Banklines"),
    ...     input_glob = "LW_ndwi_*.tif",
    ...     n_components = 1,
    ...     min_edge_pixels = 500,
    ...     dry_season_months = (11, 12, 1, 2, 3, 4),
    ... )
    >>> ext = BanklineExtractor(cfg)
    >>> summary = ext.run()
    """

    def __init__(
        self,
        cfg: BanklineConfig,
        rois: Optional[Union[PathLike, "gpd.GeoDataFrame"]] = None,
    ) -> None:
        if not _HAS_CV2:
            raise ImportError(
                "opencv-python(-headless) is required for bankline extraction.")

        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        if rois is None:
            self.rois = None
        elif _HAS_GPD and isinstance(rois, gpd.GeoDataFrame):
            self.rois = rois.copy()
        else:
            self.rois = _load_rois(rois)

    # ── helpers ──────────────────────────────────────────────────────────
    def _list_inputs(self) -> List[Path]:
        files = sorted(self.cfg.input_dir.glob(self.cfg.input_glob))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.cfg.input_glob}' in "
                f"{self.cfg.input_dir}")
        return files

    def _in_dry_season(self, dt: Optional[datetime]) -> bool:
        if not self.cfg.dry_season_months or dt is None:
            return True
        return dt.month in self.cfg.dry_season_months

    def _output_path(self, date_tag: str, roi_name: Optional[str]) -> Path:
        if roi_name:
            sub = self.cfg.output_dir / _safe(roi_name)
            sub.mkdir(parents=True, exist_ok=True)
            return sub / f"{self.cfg.output_stem}_{date_tag}.tif"
        return self.cfg.output_dir / f"{self.cfg.output_stem}_{date_tag}.tif"

    def _roi_geom(self, raster_path: Path, roi_name: str):
        """Return the ROI geometry re-projected into the raster's CRS."""
        with rasterio.open(raster_path) as src:
            tgt = src.crs
        sub = self.rois[self.rois["name"] == roi_name]
        if sub.empty:
            raise KeyError(f"ROI '{roi_name}' not found.")
        if sub.crs is None:
            sub = sub.set_crs("EPSG:4326")
        if str(sub.crs) != str(tgt):
            sub = sub.to_crs(tgt)
        return [g.__geo_interface__ for g in sub.geometry]

    # ── per-file extraction ──────────────────────────────────────────────
    def _extract_one(
        self,
        src_path: Path,
        date_tag: str,
        roi_name: Optional[str] = None,
    ) -> dict:
        """Run the full extraction for one input file (optionally one ROI)."""
        out_path = self._output_path(date_tag, roi_name)
        if out_path.exists() and not self.cfg.overwrite:
            return dict(source=src_path.name, roi=roi_name or "",
                        date=date_tag, output=out_path.name,
                        status="skipped_exists",
                        water_pixels=0, edge_pixels=0,
                        reason="overwrite disabled")

        geom = self._roi_geom(src_path, roi_name) if roi_name else None
        try:
            water, transform, crs, profile = _read_binary_water(
                src_path, geom=geom)
        except Exception as exc:
            log.warning("Failed to read %s (%s)", src_path.name, exc)
            return dict(source=src_path.name, roi=roi_name or "",
                        date=date_tag, output="",
                        status="read_error",
                        water_pixels=0, edge_pixels=0, reason=str(exc))

        # Optional pre-Canny morphological closing on the water mask itself
        if self.cfg.pre_close_radius > 0:
            r = self.cfg.pre_close_radius
            water = binary_closing(water, iterations=r).astype(np.uint8)

        # Keep N largest components
        mask_255, comp_sizes = _retain_largest_components(
            water,
            n_keep=self.cfg.n_components,
            min_component_pixels=self.cfg.min_component_pixels,
        )
        water_px = int(mask_255.sum() / 255)

        if water_px == 0:
            return dict(source=src_path.name, roi=roi_name or "",
                        date=date_tag, output="",
                        status="rejected_no_water",
                        water_pixels=0, edge_pixels=0,
                        reason="no water components after filtering")

        # Canny
        edges = _canny_edges(
            mask_255,
            low_threshold=self.cfg.canny_low,
            high_threshold=self.cfg.canny_high,
            close_radius=0,        # already applied above if requested
        )
        edge_px = int((edges > 0).sum())

        if edge_px < self.cfg.min_edge_pixels:
            return dict(source=src_path.name, roi=roi_name or "",
                        date=date_tag, output="",
                        status="rejected_low_edge_count",
                        water_pixels=water_px, edge_pixels=edge_px,
                        reason=f"{edge_px} < min_edge_pixels="
                               f"{self.cfg.min_edge_pixels}")

        # Write GeoTIFF with original CRS/transform (of the clipped window)
        profile.update(dtype="uint8", count=1, compress="lzw", nodata=None,
                       height=edges.shape[0], width=edges.shape[1],
                       transform=transform, crs=crs)
        # Drop any block-layout keys inherited from the source that would
        # trigger CPLE_IllegalArg without TILED=YES.
        for k in ("blockxsize", "blockysize", "tiled", "interleave",
                  "BLOCKXSIZE", "BLOCKYSIZE", "TILED"):
            profile.pop(k, None)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(edges.astype(np.uint8), 1)

        # QC plot
        if self.cfg.write_qc_png and _HAS_MPL:
            self._write_qc(mask_255, edges, date_tag, roi_name)

        return dict(source=src_path.name, roi=roi_name or "",
                    date=date_tag, output=out_path.name,
                    status="ok",
                    water_pixels=water_px, edge_pixels=edge_px,
                    component_sizes="|".join(str(s) for s in comp_sizes),
                    reason="")

    def _write_qc(
        self,
        mask_255: np.ndarray,
        edges: np.ndarray,
        date_tag: str,
        roi_name: Optional[str],
    ) -> None:
        qc_dir = self.cfg.output_dir / "qc"
        if roi_name:
            qc_dir = qc_dir / _safe(roi_name)
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Plot retained water as blue on white, bankline as red over it
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        water_viz = np.where(mask_255 > 0, 1, 0)

        axes[0].imshow(np.ones_like(water_viz), cmap="gray", vmin=0, vmax=1)
        axes[0].imshow(np.where(water_viz == 1, 1, np.nan),
                       cmap="Blues", vmin=0, vmax=1.5)
        axes[0].set_title(f"Retained channel mask "
                          f"({int(water_viz.sum())} px)")
        axes[0].axis("off")

        axes[1].imshow(np.ones_like(water_viz), cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(np.where(water_viz == 1, 1, np.nan),
                       cmap="Blues", vmin=0, vmax=1.5, alpha=0.8)
        axes[1].imshow(np.where(edges > 0, 1, np.nan),
                       cmap="autumn", vmin=0, vmax=1)
        axes[1].set_title(f"Bankline ({int((edges>0).sum())} px)")
        axes[1].axis("off")

        fig.suptitle(f"{date_tag}" + (f"  [{roi_name}]" if roi_name else ""))
        fig.tight_layout()
        fig.savefig(qc_dir / f"{date_tag}.png", dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ── main loop ────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """Process every input file; return a tidy summary DataFrame."""
        files = self._list_inputs()
        log.info("BanklineExtractor: %d input files", len(files))

        roi_names = (list(self.rois["name"])
                     if self.rois is not None else [None])

        rows: List[dict] = []
        for src in _pbar(files, desc="Banklines"):
            dt, tag = _parse_date(src.name)
            if tag is None:
                log.warning("Cannot parse date from '%s' — skipping.",
                            src.name)
                rows.append(dict(source=src.name, roi="", date="",
                                 output="", status="rejected_no_date",
                                 water_pixels=0, edge_pixels=0,
                                 reason="no parseable date in filename"))
                continue
            if not self._in_dry_season(dt):
                rows.append(dict(source=src.name, roi="", date=tag,
                                 output="", status="rejected_wet_season",
                                 water_pixels=0, edge_pixels=0,
                                 reason=f"month {dt.month} not in "
                                        f"{self.cfg.dry_season_months}"))
                continue

            for roi_name in roi_names:
                rows.append(self._extract_one(src, tag, roi_name=roi_name))

        df = pd.DataFrame(rows)
        out_csv = self.cfg.output_dir / "bankline_extraction_summary.csv"
        df.to_csv(out_csv, index=False)
        log.info("Wrote %s", out_csv)

        # Short console summary
        if not df.empty:
            counts = df["status"].value_counts().to_dict()
            log.info("Status counts: %s", counts)
        return df
