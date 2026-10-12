"""
channel_metrics.py
==================
Computes annual channel-morphology metrics over the period 1996-2018 for the
Lower Meghna / Tentulia (or any GBM delta sub-reach), optionally restricted
to user-supplied ROI polygons (e.g. Main Channel vs Side Channel).

Metrics produced (per year, per ROI)
------------------------------------
    migration_m_per_yr         Mean lateral displacement of bankline pixels
                               between year t and year t+1
    migration_mm_per_yr        same, scaled by 1000
    intertidal_area_km2        Σ pixels with 0 < occurrence < 1
    intertidal_elevation_m     Mean elevation of intertidal pixels (m MSL / PWD)
    lifespan_years             Mean pixel lifespan from the DTMC lifespan map
                               (Markov-chain self-transition 1/(1-P_II))

Input rasters (consistent with the rest of the WP1 pipeline)
------------------------------------------------------------
    bankline_dir   /  bankline_YYYY-MM-DD.tif          (uint8, 0/255)
                       — produced by Bankline.py (Canny edges)
                       — multiple per year are averaged to one annual line
    occurrence_dir /  occurrence_dry_YYYY.tif          (float, 0..1)
    elevation_dir  /  elevation_bt_YYYY.tif            (float, m)
                       falls back to elevation_median_YYYY.tif
    lifespan_dir   /  lifespan_map_YYYY-YYYY.tif       (float, years)
                       a single 3- to 5-year rolling window per year is used

ROI input
---------
    Path(s) to .shp / .kml / .kmz carrying one or more polygons.
    Each polygon must have a ``name`` attribute (or the filename stem is used).
    All rasters are auto-reprojected to the ROI CRS where needed.

Outputs
-------
    channel_metrics_<ROI>.csv          one row per year
    channel_metrics_all_rois.csv       tidy long-form table (roi, year, metric)

Dependencies
------------
    pip install numpy pandas rasterio geopandas shapely fiona scipy pyproj tqdm
"""

from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

try:
    import geopandas as gpd
    from shapely.geometry import shape, mapping, Polygon, MultiPolygon
    _HAS_GPD = True
except ImportError:  # pragma: no cover
    _HAS_GPD = False

try:
    import fiona
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
    _HAS_FIONA = True
except ImportError:  # pragma: no cover
    _HAS_FIONA = False

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

_RE_DATE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")
_RE_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def _parse_year(name: str) -> Optional[int]:
    """Return the first 4-digit year found in *name*, or None."""
    m = _RE_DATE.search(name)
    if m:
        return int(m.group(1))
    m = _RE_YEAR.search(name)
    if m:
        return int(m.group(0))
    return None


def _pbar(iterable, desc: str = ""):
    return tqdm(iterable, desc=desc) if _TQDM else iterable


def _raster_pixel_area_km2(src: rasterio.io.DatasetReader) -> float:
    """Pixel area in km², robust to non-square pixels & projected CRS."""
    trans = src.transform
    dx = abs(trans.a)
    dy = abs(trans.e)
    # Raster must be in a projected metric CRS for this to be meaningful
    if src.crs is None or src.crs.is_geographic:
        log.warning(
            "Raster %s is in a geographic CRS — pixel area assumes metres; "
            "reproject to a UTM CRS before area computation for accuracy.",
            Path(src.name).name,
        )
    return (dx * dy) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# ROI loading (shp / kml / kmz)
# ─────────────────────────────────────────────────────────────────────────────

def load_rois(path: PathLike,
              name_field: str = "name") -> "gpd.GeoDataFrame":
    """
    Load polygon ROIs from .shp, .kml, .kmz, .geojson or .gpkg.

    Multi-layer sources (common in Google Earth–exported KMZ files, which
    nest polygons under their own sub-layer inside a wrapping "Sightseeing
    Tour" or "My Places" folder) are enumerated and concatenated.
    Non-polygon features (Point, LineString, …) are discarded.

    The returned GeoDataFrame has at least a ``name`` column and a
    geometry column, always in the source file's native CRS (or
    EPSG:4326 for KML/KMZ).
    """
    if not _HAS_GPD:
        raise ImportError("geopandas is required for ROI support.")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix == ".kmz":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(p) as zf:
                zf.extractall(tmp)
            kml_files = list(Path(tmp).rglob("*.kml"))
            if not kml_files:
                raise ValueError(f"No .kml inside {p}")
            # Pick the largest KML (Google Earth's `doc.kml`) — sub-KMLs
            # reference files/ resources
            kml_files.sort(key=lambda f: f.stat().st_size, reverse=True)
            gdf = _read_any_layers(kml_files[0])
    elif suffix == ".kml":
        gdf = _read_any_layers(p)
    else:
        # Shapefile / GeoJSON / GeoPackage: usually single-layer, but try
        # multi-layer first in case it's a GPKG with several layers.
        gdf = _read_any_layers(p)

    # Normalise the name column
    if name_field not in gdf.columns:
        for alt in ("Name", "NAME", "roi", "ROI", "label", "Label"):
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: name_field})
                break
        else:
            gdf[name_field] = [f"{p.stem}_{i}" for i in range(len(gdf))]

    # Fill blank names with layer-derived or index-based fallbacks
    missing = gdf[name_field].isna() | (gdf[name_field].astype(str).str.strip() == "")
    if missing.any() and "_layer" in gdf.columns:
        gdf.loc[missing, name_field] = gdf.loc[missing, "_layer"]
    missing = gdf[name_field].isna() | (gdf[name_field].astype(str).str.strip() == "")
    if missing.any():
        gdf.loc[missing, name_field] = [f"{p.stem}_{i}"
                                        for i in range(missing.sum())]

    # Keep only polygons
    gdf = gdf[gdf.geometry.type.isin(("Polygon", "MultiPolygon"))].reset_index(drop=True)
    if gdf.empty:
        raise ValueError(
            f"No polygon features found in {p}. "
            f"(Tip: KMZ files exported from Google Earth often nest "
            f"polygons inside 'Sightseeing Tour' / 'My Places' folders — "
            f"check the source.)")

    keep_cols = [name_field, "geometry"]
    gdf = gdf[keep_cols].copy()

    log.info("Loaded %d polygon ROI(s) from %s [CRS=%s]: %s",
             len(gdf), p.name, gdf.crs,
             ", ".join(gdf[name_field].astype(str)))
    return gdf


def _read_any_layers(path: Path) -> "gpd.GeoDataFrame":
    """
    Read every layer in *path* and concatenate.  Empty / non-geometry
    layers are skipped.  Adds a ``_layer`` column recording each feature's
    source-layer name (used as a name fallback later).
    """
    import fiona                                              # local import
    # Enable KML drivers if still not enabled (idempotent)
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"

    # Enumerate layers.  Some KML drivers refuse listlayers on .kmz, so
    # fall back to a single read if listing fails.
    try:
        layers = fiona.listlayers(str(path))
    except Exception:
        layers = [None]
    if not layers:
        layers = [None]

    frames: List["gpd.GeoDataFrame"] = []
    for lyr in layers:
        try:
            g = gpd.read_file(path, layer=lyr) if lyr is not None \
                else gpd.read_file(path)
        except Exception as exc:
            log.debug("Skip layer %r: %s", lyr, exc)
            continue
        if len(g) == 0 or "geometry" not in g.columns:
            continue
        g = g[g.geometry.notna()].copy()
        if g.empty:
            continue
        g["_layer"] = lyr if lyr is not None else Path(path).stem
        frames.append(g)

    if not frames:
        # Very last resort: at least return the default read so the
        # "No polygon features" error below is meaningful.
        try:
            return gpd.read_file(path)
        except Exception:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry",
                            crs=frames[0].crs)


def _reproject_rois(rois: "gpd.GeoDataFrame",
                    target_crs) -> "gpd.GeoDataFrame":
    """Re-project ROIs to match raster CRS."""
    if rois.crs is None:
        log.warning("ROI CRS undefined; assuming EPSG:4326")
        rois = rois.set_crs("EPSG:4326")
    if str(rois.crs) != str(target_crs):
        return rois.to_crs(target_crs)
    return rois


# ─────────────────────────────────────────────────────────────────────────────
# Raster helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_masked(
    raster_path: Path,
    geom: Optional[Sequence] = None,
    nodata_fill: float = np.nan,
) -> Tuple[np.ndarray, rasterio.Affine, "rasterio.crs.CRS", float]:
    """
    Read a raster, optionally clipped to *geom* (iterable of shapely polygons
    already re-projected to the raster CRS). Returns (array, transform, crs,
    pixel_area_km2).
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        px_area = _raster_pixel_area_km2(src)
        if geom is not None:
            arr, transform = rio_mask(src, geom, crop=True, filled=True,
                                      nodata=src.nodata)
            arr = arr[0].astype(np.float32)
        else:
            arr = src.read(1).astype(np.float32)
            transform = src.transform
        # Convert raster nodata to NaN
        nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, nodata_fill, arr)
    return arr, transform, crs, px_area


def _group_by_year(files: Iterable[Path]) -> Dict[int, List[Path]]:
    by_year: Dict[int, List[Path]] = {}
    for f in files:
        y = _parse_year(f.name)
        if y is None:
            continue
        by_year.setdefault(y, []).append(f)
    return by_year


# ─────────────────────────────────────────────────────────────────────────────
# Migration-rate computation
# ─────────────────────────────────────────────────────────────────────────────

def _annual_bankline_composite(
    bankline_files: Sequence[Path],
    geom: Optional[Sequence],
    threshold: int = 1,
) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Combine all intra-year bankline rasters into one annual bankline.

    A pixel is considered "bank" if it is flagged as bank in ≥ *threshold*
    of the member images. All files must share the same grid (they do, given
    they come from the same Landsat pipeline).
    """
    stacks: List[np.ndarray] = []
    transform = None
    for f in bankline_files:
        arr, tr, _, _ = _read_masked(f, geom=geom, nodata_fill=0)
        arr = (arr > 0).astype(np.uint8)
        if transform is None:
            transform = tr
            ref_shape = arr.shape
        if arr.shape != ref_shape:
            # Different clipping of same window can produce 1-pixel diffs
            h = min(arr.shape[0], ref_shape[0])
            w = min(arr.shape[1], ref_shape[1])
            arr = arr[:h, :w]
            if stacks:
                stacks = [s[:h, :w] for s in stacks]
                ref_shape = (h, w)
        stacks.append(arr)

    if not stacks:
        raise ValueError("No valid bankline files to composite.")

    stack = np.stack(stacks, axis=0)
    annual = (stack.sum(axis=0) >= threshold).astype(np.uint8)
    return annual, transform


def compute_migration(
    bankline_year_a: np.ndarray,
    bankline_year_b: np.ndarray,
    pixel_size_m: float,
    dt_years: float = 1.0,
    trim_percentile: float = 99.0,
) -> Dict[str, float]:
    """
    Euclidean-distance-transform–based migration rate.

    For every bank pixel in year A we measure the distance to the nearest
    bank pixel in year B. The mean of this distribution (trimmed at the
    *trim_percentile* to remove long-range mismatches at the ROI edges)
    divided by *dt_years* is the migration rate.

    Parameters
    ----------
    bankline_year_a, bankline_year_b : 2-D uint8 arrays, same shape,
        non-zero where pixel belongs to the bankline.
    pixel_size_m : grid resolution in metres (e.g. 30 for Landsat).
    dt_years     : temporal gap between the two banklines.
    trim_percentile : discard distances above this percentile before averaging.

    Returns
    -------
    dict with keys:  mean_m_yr, mean_mm_yr, median_m_yr, p95_m_yr,
                     n_bank_pixels_a, n_bank_pixels_b.
    """
    if bankline_year_a.shape != bankline_year_b.shape:
        raise ValueError("Bankline arrays must share the same shape.")

    bank_b = bankline_year_b.astype(bool)
    if not bank_b.any() or not bankline_year_a.any():
        return dict(mean_m_yr=np.nan, mean_mm_yr=np.nan,
                    median_m_yr=np.nan, p95_m_yr=np.nan,
                    n_bank_pixels_a=int(bankline_year_a.sum()),
                    n_bank_pixels_b=int(bank_b.sum()))

    # EDT on the *inverse* of year-B banklines gives distance to nearest bank-B
    dist_px = distance_transform_edt(~bank_b)
    dists_at_a = dist_px[bankline_year_a.astype(bool)] * pixel_size_m

    cutoff = np.percentile(dists_at_a, trim_percentile)
    trimmed = dists_at_a[dists_at_a <= cutoff]

    mean_m = float(trimmed.mean()) / dt_years
    return dict(
        mean_m_yr     = mean_m,
        mean_mm_yr    = mean_m * 1000.0,
        median_m_yr   = float(np.median(trimmed)) / dt_years,
        p95_m_yr      = float(np.percentile(dists_at_a, 95)) / dt_years,
        n_bank_pixels_a = int(bankline_year_a.sum()),
        n_bank_pixels_b = int(bank_b.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intertidal metrics
# ─────────────────────────────────────────────────────────────────────────────

def intertidal_area_km2(
    occurrence: np.ndarray,
    pixel_area_km2: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """Σ pixels with low < f < high × pixel area (km²). Defaults to 0<f<1."""
    valid = np.isfinite(occurrence)
    mask = valid & (occurrence > low) & (occurrence < high)
    return float(mask.sum() * pixel_area_km2)


def intertidal_elevation_stats(
    elevation: np.ndarray,
    occurrence: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Summary stats of the intertidal elevation raster.

    If *occurrence* is supplied, the mask is 0 < f < 1; otherwise every
    finite elevation pixel is used (assumes the raster has already been
    masked to intertidal only, which is how intertidal_elevation.py writes it).
    """
    if occurrence is not None:
        mask = np.isfinite(elevation) & (occurrence > 0) & (occurrence < 1)
    else:
        mask = np.isfinite(elevation)
    if not mask.any():
        return dict(mean=np.nan, median=np.nan, std=np.nan,
                    p10=np.nan, p90=np.nan, n=0)
    z = elevation[mask]
    return dict(
        mean   = float(np.nanmean(z)),
        median = float(np.nanmedian(z)),
        std    = float(np.nanstd(z)),
        p10    = float(np.nanpercentile(z, 10)),
        p90    = float(np.nanpercentile(z, 90)),
        n      = int(mask.sum()),
    )


def lifespan_stats(
    lifespan: np.ndarray,
) -> Dict[str, float]:
    """Mean, median, std of the pixel-wise lifespan raster."""
    mask = np.isfinite(lifespan) & (lifespan > 0)
    if not mask.any():
        return dict(mean=np.nan, median=np.nan, std=np.nan,
                    p10=np.nan, p90=np.nan, n=0)
    L = lifespan[mask]
    return dict(
        mean   = float(np.nanmean(L)),
        median = float(np.nanmedian(L)),
        std    = float(np.nanstd(L)),
        p10    = float(np.nanpercentile(L, 10)),
        p90    = float(np.nanpercentile(L, 90)),
        n      = int(mask.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-window Markov-chain lifespan (year-varying)
# ─────────────────────────────────────────────────────────────────────────────
#
# The global DTMC lifespan from intertidal_lifespan_markov.py is a single,
# time-invariant raster ( L = 1 / (1 - P_II) estimated from transitions
# pooled across *all* years ).  For Fig. 7 we need a lifespan that
# evolves with the morphology.  This is achieved by fitting the DTMC on
# transitions drawn only from a moving window [Y-w, Y+w] around each
# target year Y, yielding one aggregate L per (ROI, year).  The classifier
# is identical to the one in intertidal_lifespan_markov.py:
#
#     LAND        occurrence == 0     (never wet in the year's dry-season stack)
#     INTERTIDAL  0 < occurrence < 1  (sometimes wet, sometimes dry)
#     RIVER       occurrence == 1     (always wet — permanent channel)
#     NODATA      NaN                 (no valid observation)
#
# The function is pure NumPy and operates on an (T, H, W) occurrence stack,
# so it can reuse the clipped rasters that the main loop already reads.

LAND_STATE       = 0
INTERTIDAL_STATE = 1
RIVER_STATE      = 2
NODATA_STATE     = 255


def classify_occurrence_to_state(occ: np.ndarray) -> np.ndarray:
    """
    Classify a float occurrence raster into the 3-state code used by the
    Markov chain.  Returns a uint8 array of the same shape.
    """
    state = np.full(occ.shape, NODATA_STATE, dtype=np.uint8)
    valid = np.isfinite(occ)
    state[valid & (occ <= 0.0)]                       = LAND_STATE
    state[valid & (occ >= 1.0)]                       = RIVER_STATE
    state[valid & (occ > 0.0) & (occ < 1.0)]          = INTERTIDAL_STATE
    return state


def compute_rolling_lifespan(
    state_stack: np.ndarray,
    years: Sequence[int],
    target_years: Sequence[int],
    window: int = 3,
) -> Dict[int, Dict[str, float]]:
    """
    Year-varying Markov-chain lifespan from a state-sequence stack.

    Parameters
    ----------
    state_stack : (T, H, W) uint8 array with values in {0,1,2,255}
    years       : length-T list of calendar years for each band
    target_years : years to produce L for (typically the whole run)
    window      : ± years around each target (so a 3-year half-window
                  gives a 7-year full window)

    For each target year Y, every consecutive year pair (t, t+1) with
    both years in [Y-w, Y+w] contributes transitions.  Pixels are masked
    where either year is NODATA.  The intertidal self-transition count
    gives:
        P_II = N_{I→I} / N_{I→*}
        L    = 1 / (1 − P_II)            (∞ when P_II = 1)

    Returns
    -------
    dict[year] = { 'lifespan': float, 'p_II': float,
                   'n_transitions': int,   # I → * transitions in window
                   'n_self':        int,   # I → I transitions in window
                   'window_years':  (lo, hi) }
    """
    if state_stack.ndim != 3:
        raise ValueError("state_stack must be (T, H, W)")
    if len(years) != state_stack.shape[0]:
        raise ValueError("len(years) must equal state_stack.shape[0]")

    year_to_idx = {int(y): i for i, y in enumerate(years)}
    year_sorted = sorted(year_to_idx.keys())
    results: Dict[int, Dict[str, float]] = {}

    for y in target_years:
        lo, hi = y - window, y + window
        window_years = [yy for yy in year_sorted if lo <= yy <= hi]
        if len(window_years) < 2:
            results[int(y)] = dict(lifespan=np.nan, p_II=np.nan,
                                   n_transitions=0, n_self=0,
                                   window_years=(lo, hi))
            continue

        n_self     = 0      # INTERTIDAL → INTERTIDAL
        n_from_int = 0      # INTERTIDAL → anything (valid next-year state)

        # Sum transitions over every consecutive year pair inside the window
        for ya, yb in zip(window_years[:-1], window_years[1:]):
            sa = state_stack[year_to_idx[ya]]
            sb = state_stack[year_to_idx[yb]]
            is_int_a = (sa == INTERTIDAL_STATE)
            valid_b  = (sb != NODATA_STATE)
            from_int = is_int_a & valid_b
            n_from_int += int(from_int.sum())
            n_self     += int((from_int & (sb == INTERTIDAL_STATE)).sum())

        if n_from_int == 0:
            results[int(y)] = dict(lifespan=np.nan, p_II=np.nan,
                                   n_transitions=0, n_self=0,
                                   window_years=(lo, hi))
            continue

        p_II = n_self / n_from_int
        L    = 1.0 / (1.0 - p_II) if p_II < 1.0 else np.inf
        results[int(y)] = dict(
            lifespan      = float(L),
            p_II          = float(p_II),
            n_transitions = int(n_from_int),
            n_self        = int(n_self),
            window_years  = (int(window_years[0]), int(window_years[-1])),
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsConfig:
    """File naming and paths used by :class:`ChannelMetricsAnalyzer`."""
    bankline_dir:   Path
    occurrence_dir: Path
    elevation_dir:  Path
    lifespan_dir:   Path
    output_dir:     Path

    bankline_glob:    str = "bankline_*.tif"
    occurrence_glob:  str = "occurrence_dry_{year}.tif"
    elevation_glob:   str = "elevation_bt_{year}.tif"
    elevation_fallback_glob: str = "elevation_median_{year}.tif"
    lifespan_glob:    str = "lifespan_map_{y0}-{y1}.tif"

    pixel_size_m:  float = 30.0
    lifespan_window: int = 3            # ± years for rolling window
    bankline_occ_threshold: int = 1     # min #images per year to flag bank
    migration_dt: int = 1               # years between bankline comparisons

    # Lifespan computation mode:
    #   "rolling"  — year-varying DTMC over a moving [Y-w, Y+w] window on
    #                the occurrence rasters (default; matches Fig. 7).
    #   "static"   — read lifespan_map_*.tif (the output of
    #                intertidal_lifespan_markov.py).  Returns the *same*
    #                value for every year if only a full-period file exists.
    #   "both"     — compute rolling AND read the static raster if found.
    lifespan_mode: str = "rolling"


class ChannelMetricsAnalyzer:
    """
    Computes annual migration, area, elevation and lifespan per ROI.

    Example
    -------
    >>> cfg = MetricsConfig(
    ...     bankline_dir   = Path(".../Banklines"),
    ...     occurrence_dir = Path(".../intertidal_elevation"),
    ...     elevation_dir  = Path(".../intertidal_elevation"),
    ...     lifespan_dir   = Path(".../lifespan"),
    ...     output_dir     = Path(".../channel_metrics"),
    ... )
    >>> ana = ChannelMetricsAnalyzer(cfg, rois="main_side_channels.shp")
    >>> df = ana.run(years=range(1996, 2019))
    """

    def __init__(
        self,
        cfg: MetricsConfig,
        rois: Optional[Union[PathLike, "gpd.GeoDataFrame"]] = None,
        roi_name_field: str = "name",
    ) -> None:
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        if rois is None:
            self.rois = None
        elif _HAS_GPD and isinstance(rois, gpd.GeoDataFrame):
            self.rois = rois.copy()
            if roi_name_field != "name":
                self.rois = self.rois.rename(columns={roi_name_field: "name"})
        else:
            self.rois = load_rois(rois, name_field=roi_name_field)

        self._bankline_cache: Dict[int, Tuple[np.ndarray, rasterio.Affine]] = {}

    # ── Catalogue files per year ────────────────────────────────────────
    def _catalogue_banklines(self) -> Dict[int, List[Path]]:
        files = sorted(self.cfg.bankline_dir.glob(self.cfg.bankline_glob))
        return _group_by_year(files)

    def _find_occurrence(self, year: int) -> Optional[Path]:
        p = self.cfg.occurrence_dir / self.cfg.occurrence_glob.format(year=year)
        # Some runs nest by year
        alt = self.cfg.occurrence_dir / str(year) / self.cfg.occurrence_glob.format(year=year)
        if p.exists():
            return p
        if alt.exists():
            return alt
        return None

    def _find_elevation(self, year: int) -> Optional[Path]:
        p = self.cfg.elevation_dir / self.cfg.elevation_glob.format(year=year)
        alt = self.cfg.elevation_dir / str(year) / self.cfg.elevation_glob.format(year=year)
        fb  = self.cfg.elevation_dir / self.cfg.elevation_fallback_glob.format(year=year)
        fb2 = self.cfg.elevation_dir / str(year) / self.cfg.elevation_fallback_glob.format(year=year)
        for cand in (p, alt, fb, fb2):
            if cand.exists():
                return cand
        return None

    def _find_lifespan(self, year: int) -> Optional[Path]:
        """
        Lifespan is a Markov-chain output over a window of years.
        Try, in order:
          lifespan_map_<year-w>-<year+w>.tif
          lifespan_map_<year>-<year+1>.tif
          any lifespan_map_*.tif spanning <year>
        """
        g = self.cfg.lifespan_glob
        w = self.cfg.lifespan_window
        cand = self.cfg.lifespan_dir / g.format(y0=year - w, y1=year + w)
        if cand.exists():
            return cand
        cand = self.cfg.lifespan_dir / g.format(y0=year, y1=year + 1)
        if cand.exists():
            return cand
        # scan
        for f in self.cfg.lifespan_dir.glob("lifespan_map_*.tif"):
            m = re.search(r"(\d{4})[-_](\d{4})", f.stem)
            if m and int(m.group(1)) <= year <= int(m.group(2)):
                return f
        return None

    # ── Per-ROI geometry in raster CRS ──────────────────────────────────
    def _geoms_for_raster(self, raster_path: Path, roi_name: str):
        """Return iterable of shapely geoms in raster CRS, or None."""
        if self.rois is None:
            return None
        with rasterio.open(raster_path) as src:
            tgt_crs = src.crs
        sub = self.rois[self.rois["name"] == roi_name]
        if sub.empty:
            raise KeyError(f"ROI '{roi_name}' not found.")
        sub = _reproject_rois(sub, tgt_crs)
        return [g.__geo_interface__ for g in sub.geometry]

    # ── Per-metric accessors ────────────────────────────────────────────
    def _bankline_for_year(
        self, year: int, geom, bankline_by_year: Dict[int, List[Path]]
    ) -> Optional[Tuple[np.ndarray, rasterio.Affine]]:
        if year not in bankline_by_year:
            return None
        arr, tr = _annual_bankline_composite(
            bankline_by_year[year], geom=geom,
            threshold=self.cfg.bankline_occ_threshold,
        )
        return arr, tr

    # ── Rolling-lifespan pre-computation (once per ROI) ──────────────────
    def _rolling_lifespan_for_roi(
        self,
        years: Sequence[int],
        geom,
    ) -> Dict[int, Dict[str, float]]:
        """
        Build the state-sequence stack for this ROI from the annual
        occurrence rasters, then fit a rolling-window DTMC and return
        per-year lifespan stats.

        One occurrence raster is read per target year (clipped to the ROI);
        the stack is held in memory as uint8 so it's ~1 byte × pixels × years.
        """
        state_layers: List[np.ndarray] = []
        year_list:    List[int]         = []
        ref_shape:    Optional[Tuple[int, int]] = None

        for y in years:
            p = self._find_occurrence(y)
            if p is None:
                continue
            try:
                occ, _, _, _ = _read_masked(p, geom=geom)
            except Exception as exc:
                log.warning("lifespan: cannot read occurrence for %d (%s)",
                            y, exc)
                continue
            state = classify_occurrence_to_state(occ)
            if ref_shape is None:
                ref_shape = state.shape
            elif state.shape != ref_shape:
                # Harmonise tiny 1-pixel clip differences
                h = min(state.shape[0], ref_shape[0])
                w = min(state.shape[1], ref_shape[1])
                state = state[:h, :w]
                state_layers = [s[:h, :w] for s in state_layers]
                ref_shape = (h, w)
            state_layers.append(state)
            year_list.append(int(y))

        if len(state_layers) < 2:
            log.warning("Rolling lifespan: only %d year(s) with occurrence "
                        "data for this ROI; skipping.", len(state_layers))
            return {}

        state_stack = np.stack(state_layers, axis=0)
        return compute_rolling_lifespan(
            state_stack=state_stack,
            years=year_list,
            target_years=list(years),
            window=self.cfg.lifespan_window,
        )

    # ── Main loop ───────────────────────────────────────────────────────
    def run(
        self,
        years: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """
        Process every year / every ROI and return a tidy DataFrame.
        Also writes per-ROI CSVs to ``cfg.output_dir``.
        """
        bankline_by_year = self._catalogue_banklines()
        if years is None:
            years = sorted(bankline_by_year.keys())
        years = list(years)
        log.info("ChannelMetricsAnalyzer: %d years × %d ROI(s)  "
                 "[lifespan_mode=%s, window=±%d yr]",
                 len(years),
                 len(self.rois) if self.rois is not None else 1,
                 self.cfg.lifespan_mode, self.cfg.lifespan_window)

        roi_names = (list(self.rois["name"])
                     if self.rois is not None else ["ALL"])

        rows: List[dict] = []
        for roi_name in roi_names:
            log.info("── ROI: %s ──", roi_name)

            # Pre-resolve the ROI geometry from any raster reference for the
            # rolling-lifespan pass (it's the same geometry as the per-year
            # loop below, just constructed once).
            roi_geom = None
            if self.rois is not None:
                ref = None
                for y in years:
                    ref = self._find_occurrence(y) or self._find_elevation(y)
                    if ref is None and bankline_by_year.get(y):
                        ref = bankline_by_year[y][0]
                    if ref is not None:
                        break
                if ref is not None:
                    roi_geom = self._geoms_for_raster(ref, roi_name)

            # Rolling-window Markov lifespan, one pass per ROI
            if self.cfg.lifespan_mode in ("rolling", "both"):
                rolling_L = self._rolling_lifespan_for_roi(years, roi_geom)
            else:
                rolling_L = {}

            prev_bank = prev_year = None
            for y in _pbar(years, desc=f"{roi_name}"):
                occ_p  = self._find_occurrence(y)
                elev_p = self._find_elevation(y)
                life_p = self._find_lifespan(y) \
                    if self.cfg.lifespan_mode in ("static", "both") else None

                # Geometry is ROI projected into the raster's CRS,
                # evaluated once per ROI and reused across the year's files.
                ref_raster = occ_p or elev_p
                if ref_raster is None and bankline_by_year.get(y):
                    ref_raster = bankline_by_year[y][0]
                if ref_raster is None:
                    log.warning("Year %d has no rasters at all — skipping.", y)
                    continue
                geom = self._geoms_for_raster(ref_raster, roi_name) \
                    if self.rois is not None else None

                row = dict(roi=roi_name, year=y)

                # ── 1. Bankline & migration ──────────────────────────────
                cur = self._bankline_for_year(y, geom, bankline_by_year)
                if cur is not None and prev_bank is not None:
                    a, b = prev_bank[0], cur[0]
                    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
                    mig = compute_migration(
                        a[:h, :w], b[:h, :w],
                        pixel_size_m=self.cfg.pixel_size_m,
                        dt_years=float(y - prev_year),
                    )
                    row.update({f"migration_{k}": v for k, v in mig.items()})
                else:
                    row.update(migration_mean_m_yr=np.nan,
                               migration_mean_mm_yr=np.nan)
                if cur is not None:
                    prev_bank, prev_year = cur, y

                # ── 2. Intertidal area & elevation ───────────────────────
                occ_arr = None
                if occ_p is not None:
                    occ_arr, _, _, px_area = _read_masked(occ_p, geom=geom)
                    row["intertidal_area_km2"] = intertidal_area_km2(
                        occ_arr, pixel_area_km2=px_area)
                else:
                    row["intertidal_area_km2"] = np.nan

                if elev_p is not None:
                    elev_arr, _, _, _ = _read_masked(elev_p, geom=geom)
                    if occ_arr is not None and occ_arr.shape == elev_arr.shape:
                        stats = intertidal_elevation_stats(elev_arr, occ_arr)
                    else:
                        stats = intertidal_elevation_stats(elev_arr)
                    row["intertidal_elevation_m"] = stats["mean"]
                    row["intertidal_elevation_median_m"] = stats["median"]
                    row["intertidal_elevation_std_m"] = stats["std"]
                else:
                    row["intertidal_elevation_m"] = np.nan

                # ── 3. Lifespan ──────────────────────────────────────────
                # Rolling-window DTMC (default for Fig. 7)
                if y in rolling_L:
                    r = rolling_L[y]
                    row["lifespan_years"]         = r["lifespan"]
                    row["lifespan_p_II"]          = r["p_II"]
                    row["lifespan_n_transitions"] = r["n_transitions"]
                    row["lifespan_window_y0"]     = r["window_years"][0]
                    row["lifespan_window_y1"]     = r["window_years"][1]
                else:
                    row["lifespan_years"] = np.nan

                # Static (global) lifespan from intertidal_lifespan_markov
                # — only populated when mode is "static" or "both"
                if life_p is not None:
                    life_arr, _, _, _ = _read_masked(life_p, geom=geom)
                    ls = lifespan_stats(life_arr)
                    row["lifespan_static_years"]        = ls["mean"]
                    row["lifespan_static_median_years"] = ls["median"]
                    if self.cfg.lifespan_mode == "static":
                        # Back-compat: old default column comes from static
                        row["lifespan_years"]        = ls["mean"]
                        row["lifespan_median_years"] = ls["median"]

                rows.append(row)

            prev_bank = prev_year = None      # reset for next ROI

        df = pd.DataFrame(rows)
        if df.empty:
            log.warning("No metrics computed — check input paths.")
            return df

        # Write CSVs
        tidy_out = self.cfg.output_dir / "channel_metrics_all_rois.csv"
        df.to_csv(tidy_out, index=False)
        log.info("Wrote %s", tidy_out)
        for roi, sub in df.groupby("roi"):
            p = self.cfg.output_dir / f"channel_metrics_{_safe(roi)}.csv"
            sub.to_csv(p, index=False)
            log.info("Wrote %s", p)

        return df


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))
