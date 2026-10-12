# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:06:13 2026

@author: tahsi001

intertidal_grided_elevation.py  (v3)
==============================
Pixel-wise intertidal area and elevation directly from WSE GeoTIFFs.

New in v3
----------
RiverMaskBuilder
    Derives a persistent river/intertidal domain directly from the WSE
    GeoTIFF archive without requiring a separately classified channel-mask.
    Three masks are built from the full image time series:

      ever_river   — pixel was river in ≥1 image (intertidal candidate zone)
      always_river — pixel was river in every valid image (permanent channel)
      candidate    — ever_river AND NOT always_river → intertidal zone

StudyGrid
    Optimal compact analysis domain for the Meghna–Tentulia estuary:
    bounding box of ever_river pixels + configurable land-side buffer
    (default 3 km = 100 px at 30 m) to capture the adjacent intertidal
    fringe, snapped to the native 30 m raster grid.  All output GeoTIFFs
    are clipped to this domain, reducing file size from ~6 000 km² (full
    raster) to ~600–900 km² (active channel zone).

River pixel detection
    A pixel carries a valid WSE observation when:
      - raster value is finite
      - value ≠ land sentinel (999 ± 0.5)
      - value is within WSE_RANGE (−30 to +30 m)
    This works directly on river_surface_profiler.py output without a
    separate channel-mask step.

Elevation methods (from v2)
    Method A  Bishop-Taylor (PRIMARY):  z = MHSW - f_dry · (MHSW - MLSW)
              Reference: Bishop-Taylor et al. (2019) RSE Eq.1; Tseng et al. (2017).
              f=0 → never wet → z ~ MHSW  |  f=1 → always wet → z ~ MLSW
    Method B  Waterline median (SUPPLEMENTARY): z = median(WSE | dry-season, pixel wet)

Output GeoTIFFs per year (in output_dir/YYYY/, clipped to study grid)
  occurrence_dry_YYYY.tif      dry-season inundation frequency
  occurrence_all_YYYY.tif      all-image occurrence (diagnostic)
  h_min_YYYY.tif               minimum dry-season WSE when wet
  h_median_YYYY.tif            median  dry-season WSE when wet
  mhsw_YYYY.tif                IDW-interpolated MHSW
  mlsw_YYYY.tif                IDW-interpolated MLSW
  elevation_bt_YYYY.tif        Method A  Bishop-Taylor  z = MHSW − f·ΔTide  (PRIMARY)
  elevation_median_YYYY.tif    Method B  waterline median  (SUPPLEMENTARY)

Study-domain outputs (in output_dir/)
  river_mask_union.tif         ever_river (full raster)
  river_mask_union_clipped.tif ever_river (study grid extent)
  river_mask_persistent.tif    always_river (full raster)
  study_grid_info.json         window offsets, transform, area

Dependencies
    pip install numpy pandas rasterio scipy pyproj tqdm
"""


from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.transform import xy as rasterio_xy
from rasterio.windows import Window

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LAND_VALUE          =  999.0
WSE_RANGE           = (-10.0, 10.0)
DRY_MONTHS          = [11, 12, 1, 2, 3, 4, 5]
MIN_DRY_OBS         = 2
STUDY_GRID_BUFFER_M = 3_000    # 3 km land buffer around union river mask

_DATE_PATS = [
    (r"(\d{4}-\d{2}-\d{2})_(\d{6})", "%Y-%m-%d %H%M%S"),
    (r"(\d{4}-\d{2}-\d{2})",          "%Y-%m-%d"),
    (r"(\d{8})_(\d{6})",              "%Y%m%d %H%M%S"),
    (r"(\d{8})",                       "%Y%m%d"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(fname: str) -> Optional[datetime]:
    for pat, fmt in _DATE_PATS:
        m = re.search(pat, fname)
        if m:
            try:
                return datetime.strptime(" ".join(m.groups()).strip(), fmt)
            except ValueError:
                continue
    return None


def _is_dry(dt: datetime) -> bool:
    return dt.month in DRY_MONTHS


def _is_river(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: True where arr is a valid WSE observation."""
    return (
        np.isfinite(arr)
        & ~np.isclose(arr, LAND_VALUE, atol=0.5)
        & (arr > WSE_RANGE[0])
        & (arr < WSE_RANGE[1])
    )


def _idw(values, src_xy, tgt_xy, power=2.0, min_dist=1.0):
    diff  = src_xy[:, np.newaxis, :] - tgt_xy[np.newaxis, :, :]
    dists = np.sqrt((diff**2).sum(-1))
    dists = np.where(dists < min_dist, min_dist, dists)
    w     = 1.0 / dists**power
    res   = (w * values[:, np.newaxis]).sum(0) / w.sum(0)
    exact = dists.min(0) < min_dist
    if exact.any():
        res[exact] = values[dists.argmin(0)[exact]]
    return res


def _write_tif(data, path, profile, nodata=np.nan):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(np.float32), 1)


def _write_mask_tif(data, path, profile):
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(np.uint8), 1)


def _pixel_centres(rows, cols, transform):
    xs, ys = rasterio_xy(transform, rows, cols)
    return np.column_stack([xs, ys]).astype(np.float64)


def _resolve_station_xy(coords_df, raster_crs_wkt, crs_wl):
    """
    Build {station_name: [x, y]} from a station coordinates DataFrame.

    Column detection is intentionally broad — tries many common header names
    and falls back to any non-numeric text column so that the function works
    with a wide range of CSV formats without requiring the user to rename columns.

    Station keys are stored both as-is AND as lowercase-stripped to enable
    case-insensitive matching against the datums CSV.
    """
    df = coords_df.copy()

    # ── Detect ID / name column ───────────────────────────────────────────────
    # Try explicit column names (case-insensitive search)
    _id_candidates = [
        "station_id", "stationid", "station_no", "station_number",
        "station_code", "stationcode", "stn_id", "stn_code",
        "station", "Station", "STATION",
        "name", "Name", "NAME", "station_name", "StationName",
        "STATION_NAME", "stn_name", "StnName",
        "id", "ID", "Id",
    ]
    col_lower = {c.lower().replace(" ", "_"): c for c in df.columns}
    id_col = None
    for cand in _id_candidates:
        if cand in df.columns:
            id_col = cand
            break
        if cand.lower().replace(" ", "_") in col_lower:
            id_col = col_lower[cand.lower().replace(" ", "_")]
            break

    # ── Fallback: first column whose values don't look like coordinates ───────
    def _looks_like_coords(col):
        try:
            v = df[col].dropna().astype(float)
            return len(v) > 0 and ((v.abs() < 200).mean() > 0.8)
        except (ValueError, TypeError):
            return False

    if id_col is None:
        for col in df.columns:
            if not _looks_like_coords(col):
                id_col = col
                log.warning(
                    "_resolve_station_xy: no standard name column found.\n"
                    "  CSV columns: %s\n"
                    "  Using '%s' as station ID. Rename a column to 'station' "
                    "or 'station_name' to suppress this warning.",
                    list(df.columns), col
                )
                break

    if id_col is None:
        # All columns look like coordinates — last resort: use first column
        id_col = df.columns[0]
        log.error(
            "_resolve_station_xy: cannot determine station ID column.\n"
            "  All columns appear numeric: %s\n"
            "  Add a 'station' column with station names to your coords CSV.",
            list(df.columns)
        )

    # ── Detect X / Y columns ─────────────────────────────────────────────────
    _x_cands = ["x", "X", "easting", "Easting", "EASTING",
                 "lon", "Lon", "LON", "longitude", "Longitude", "LONGITUDE"]
    _y_cands = ["y", "Y", "northing", "Northing", "NORTHING",
                 "lat", "Lat", "LAT", "latitude", "Latitude", "LATITUDE"]
    x_col = next((c for c in _x_cands if c in df.columns), None)
    y_col = next((c for c in _y_cands if c in df.columns), None)

    # Fallback: pick first/second numeric column that isn't id_col
    numeric_cols = [c for c in df.columns if c != id_col and _looks_like_coords(c)]
    if x_col is None and len(numeric_cols) >= 1:
        x_col = numeric_cols[0]
    if y_col is None and len(numeric_cols) >= 2:
        y_col = numeric_cols[1]

    if x_col is None or y_col is None:
        raise ValueError(
            f"Cannot determine X/Y columns from: {list(df.columns)}\n"
            "Add columns named 'Easting'/'Northing' or 'lon'/'lat'."
        )

    log.info(
        "Station coords: id_col='%s'  x_col='%s'  y_col='%s'  n=%d",
        id_col, x_col, y_col, len(df)
    )

    # ── CRS reprojection ──────────────────────────────────────────────────────
    needs_reproj = (crs_wl is not None
                    and crs_wl.upper() not in ("NONE", "")
                    and crs_wl != raster_crs_wkt)
    if needs_reproj and not _HAS_PYPROJ:
        raise ImportError("pyproj required for coordinate reprojection.")
    tr = (Transformer.from_crs(crs_wl, raster_crs_wkt, always_xy=True)
          if needs_reproj else None)

    # ── Build output dict ─────────────────────────────────────────────────────
    out = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        try:
            x, y = float(row[x_col]), float(row[y_col])
        except (ValueError, TypeError):
            continue
        if tr:
            x, y = tr.transform(x, y)
        norm = sid.lower()
        out[norm] = np.array([x, y], dtype=np.float64)
        if norm != sid:
            out[sid] = np.array([x, y], dtype=np.float64)
    return out



class StudyGrid:
    """
    Compact spatial domain for the Meghna–Tentulia analysis.

    Stores the rasterio Window (row_off, col_off, height, width) relative to
    the source WSE rasters, plus the affine transform and profile for the
    clipped domain.  All YearProcessor outputs are written in this CRS/extent.

    Attributes
    ----------
    row_off, col_off  top-left of the window in the source raster
    height, width     dimensions in pixels
    transform         affine transform of the clipped domain
    profile           rasterio profile for writing clipped GeoTIFFs
    """

    def __init__(self, row_off, col_off, height, width, transform, profile):
        self.row_off   = int(row_off)
        self.col_off   = int(col_off)
        self.height    = int(height)
        self.width     = int(width)
        self.transform = transform
        self.profile   = profile

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    @property
    def area_km2(self) -> float:
        return self.height * self.width * self.transform.a**2 / 1e6

    def read_band(self, path: Path) -> np.ndarray:
        """Read band 1 clipped to this grid, nodata → NaN."""
        with rasterio.open(path) as src:
            data = src.read(1, window=self.window).astype(np.float32)
            nd   = src.nodata
            if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                data[data == nd] = np.nan
        return data

    def to_dict(self) -> dict:
        t = self.transform
        return {
            "row_off":  self.row_off,  "col_off":  self.col_off,
            "height":   self.height,   "width":    self.width,
            "origin_x": t.c,           "origin_y": t.f,
            "pixel_x":  t.a,           "pixel_y":  t.e,
            "crs_wkt":  self.profile.get("crs", ""),
            "area_km2": round(self.area_km2, 1),
        }

    @classmethod
    def from_json(cls, json_path: Path, reference_tif: Path) -> "StudyGrid":
        with open(json_path) as fh:
            info = json.load(fh)
        t = from_origin(info["origin_x"], info["origin_y"],
                        abs(info["pixel_x"]), abs(info["pixel_y"]))
        with rasterio.open(reference_tif) as src:
            prof = src.profile.copy()
        prof.update(height=info["height"], width=info["width"], transform=t)
        return cls(info["row_off"], info["col_off"],
                   info["height"],  info["width"], t, prof)


# ─────────────────────────────────────────────────────────────────────────────
# RiverMaskBuilder
# ─────────────────────────────────────────────────────────────────────────────

class RiverMaskBuilder:
    """
    Derives persistent river and intertidal masks directly from WSE GeoTIFFs.

    Iterates through the WSE archive, applies the river pixel rule
    (_is_river), and accumulates pixel-wise counts:
      n_river[i,j]  — number of images where pixel was classified as river
      n_valid[i,j]  — number of images with any valid (non-NaN) data

    From these, three Boolean masks are derived:
      ever_river     = n_river >= 1
      always_river   = (n_valid >= 3) AND (n_river == n_valid)
      candidate      = ever_river AND NOT always_river

    The study grid is the bounding box of ever_river + buffer_m on each side,
    snapped to the native pixel grid.

    Parameters
    ----------
    wse_dir         directory of WSE GeoTIFFs
    glob_pattern    filename glob (default "*_wse.tif")
    sample_every    use every N-th file to build the mask (default 1 = all)
    buffer_m        land-side buffer in metres (default 3000)
    """

    def __init__(self, wse_dir: Union[str, Path], glob_pattern="*_wse.tif",
                 sample_every: int = 1, buffer_m: float = STUDY_GRID_BUFFER_M):
        self.wse_dir      = Path(wse_dir)
        self.glob_pattern = glob_pattern
        self.sample_every = max(1, int(sample_every))
        self.buffer_m     = float(buffer_m)

    def build(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[StudyGrid, Dict[str, np.ndarray]]:
        """
        Build river masks and StudyGrid.

        Returns
        -------
        study_grid  : StudyGrid  compact analysis domain
        masks       : dict with keys ever_river, always_river, n_river, n_valid
        """
        files = sorted(self.wse_dir.glob(self.glob_pattern))[::self.sample_every]
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.glob_pattern}' in {self.wse_dir}"
            )
        log.info("RiverMaskBuilder: processing %d files", len(files))

        with rasterio.open(files[0]) as src:
            H, W      = src.height, src.width
            transform = src.transform
            profile   = src.profile.copy()
            px_m      = abs(src.transform.a)

        n_river = np.zeros((H, W), dtype=np.int32)
        n_valid = np.zeros((H, W), dtype=np.int32)

        it = tqdm(files, desc="River mask") if _HAS_TQDM else files
        for f in it:
            with rasterio.open(f) as src:
                arr = src.read(1).astype(np.float32)
                nd  = src.nodata
                if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                    arr[arr == nd] = np.nan
            n_river += _is_river(arr).astype(np.int32)
            n_valid += np.isfinite(arr).astype(np.int32)

        ever_river   = n_river >= 1
        always_river = (n_valid >= 3) & (n_river == n_valid)

        area_ever = float(ever_river.sum()) * px_m**2 / 1e6
        log.info(
            "Masks: ever_river=%d px (%.0f km²), always_river=%d px",
            int(ever_river.sum()), area_ever, int(always_river.sum()),
        )

        # Build study grid — bbox of ever_river + buffer
        buf_px = max(1, int(round(self.buffer_m / px_m)))
        rows_with = np.where(ever_river.any(axis=1))[0]
        cols_with = np.where(ever_river.any(axis=0))[0]
        if len(rows_with) == 0:
            raise ValueError("No river pixels found in WSE archive.")

        r0 = max(0,   int(rows_with.min()) - buf_px)
        r1 = min(H-1, int(rows_with.max()) + buf_px)
        c0 = max(0,   int(cols_with.min()) - buf_px)
        c1 = min(W-1, int(cols_with.max()) + buf_px)

        clip_H, clip_W = r1 - r0 + 1, c1 - c0 + 1
        ox = transform.c + c0 * transform.a
        oy = transform.f + r0 * transform.e
        clip_tf = from_origin(ox, oy, abs(transform.a), abs(transform.e))
        clip_profile = profile.copy()
        clip_profile.update(height=clip_H, width=clip_W, transform=clip_tf)

        study_grid = StudyGrid(r0, c0, clip_H, clip_W, clip_tf, clip_profile)
        log.info(
            "Study grid: %d x %d px  |  %.1f x %.1f km  |  %.0f km²  "
            "|  buffer=%d m (%d px)",
            clip_W, clip_H,
            clip_W*px_m/1e3, clip_H*px_m/1e3,
            study_grid.area_km2,
            int(self.buffer_m), buf_px,
        )

        masks = dict(ever_river=ever_river, always_river=always_river,
                     n_river=n_river, n_valid=n_valid)

        if output_dir is not None:
            od = Path(output_dir)
            _write_mask_tif(ever_river.astype(np.uint8),
                            od/"river_mask_union.tif", profile)
            _write_mask_tif(always_river.astype(np.uint8),
                            od/"river_mask_persistent.tif", profile)
            _write_mask_tif(
                ever_river[r0:r0+clip_H, c0:c0+clip_W].astype(np.uint8),
                od/"river_mask_union_clipped.tif", clip_profile,
            )
            with open(od/"study_grid_info.json", "w") as fh:
                json.dump(study_grid.to_dict(), fh, indent=2)
            log.info("Masks and study_grid_info.json saved to %s", od)

        return study_grid, masks


# ─────────────────────────────────────────────────────────────────────────────
# Per-year processor
# ─────────────────────────────────────────────────────────────────────────────



def diagnose_station_matching(
    datums_csv:    Union[str, Path],
    station_csv:   Union[str, Path],
    year:          int = None,
    season:        str = "dry",
) -> None:
    """
    Print a diagnostic table showing which datum stations match the
    coordinate CSV — run this if Method A gives "no datum stations matched".

    Also prints the actual CSV column names so you know what to rename.

    Usage
    -----
    from gbm_delta.morphodynamics.elevation_v3 import diagnose_station_matching
    diagnose_station_matching(DATUMS_CSV, STATION_CSV, year=1996)
    """
    datums = pd.read_csv(datums_csv)
    coords = pd.read_csv(station_csv)

    print()
    print("=" * 65)
    print("  Station Matching Diagnostic")
    print("=" * 65)
    print(f"  Datums CSV : {datums_csv}")
    print(f"  Coords CSV : {station_csv}")
    print()
    print(f"  Datums CSV columns  : {list(datums.columns)}")
    print(f"  Coords CSV columns  : {list(coords.columns)}")
    print()

    # Show first few rows of coords CSV so user can see the structure
    print("  Coords CSV (first 3 rows):")
    for _, row in coords.head(3).iterrows():
        print(f"    {dict(row)}")
    print()

    # Find station column in datums
    datum_names_all = sorted(datums["station"].astype(str).str.strip().unique())

    # Detect which column the code will use as station ID
    _id_candidates = [
        "station_id","stationid","station_no","station_number",
        "station_code","stationcode","stn_id","stn_code",
        "station","Station","STATION",
        "name","Name","NAME","station_name","StationName",
        "STATION_NAME","stn_name","StnName","id","ID","Id",
    ]
    col_lower = {c.lower().replace(" ","_"): c for c in coords.columns}
    id_col = None
    for cand in _id_candidates:
        if cand in coords.columns:
            id_col = cand; break
        if cand.lower().replace(" ","_") in col_lower:
            id_col = col_lower[cand.lower().replace(" ","_")]; break

    def _looks_like_coords(col):
        try:
            v = coords[col].dropna().astype(float)
            return len(v) > 0 and ((v.abs() < 200).mean() > 0.8)
        except (ValueError, TypeError):
            return False

    if id_col is None:
        for col in coords.columns:
            if not _looks_like_coords(col):
                id_col = col; break

    if id_col is None:
        id_col = coords.columns[0]

    print(f"  Station ID column selected by code: '{id_col}'")
    coord_names = sorted(coords[id_col].astype(str).str.strip().unique())
    coord_norms = {s.strip().lower(): s for s in coord_names}
    print(f"  Coord station IDs (first 8): {coord_names[:8]}")
    print()

    # Filter datums for the year if requested
    if year is not None:
        d_yr = datums[
            (datums["season"] == season) &
            (datums["hyd_year"].isin([year, year - 1]))
        ]
        check_names = sorted(d_yr["station"].astype(str).str.strip().unique())
    else:
        check_names = datum_names_all

    print(f"  Matching datum stations → coord IDs  (year={year}, season={season})")
    print("  " + "-" * 60)
    n_match = 0
    for s in check_names:
        s_norm = s.strip().lower()
        if s in coord_names:
            match = "✓ exact"; key = s; n_match += 1
        elif s_norm in coord_norms:
            match = "✓ normalised"; key = coord_norms[s_norm]; n_match += 1
        else:
            partials = [k for k in coord_names
                        if s_norm[:6] in k.lower() or k.lower()[:6] in s_norm]
            match = f"? partial→{partials[0]}" if partials else "✗ NO MATCH"
            key   = partials[0] if partials else "—"
        print(f"    {s:<30}  {match:<28}  {key}")

    print()
    print(f"  Result: {n_match}/{len(check_names)} datum stations matched")

    if n_match == 0:
        print()
        print("  ══ DIAGNOSIS ════════════════════════════════════════")
        if all(_looks_like_coords(id_col) for _ in [1]):
            print("  ✗ The code picked a COORDINATE column as the station ID.")
            print(f"    Column '{id_col}' contains numbers like lat/lon, not names.")
            print()
            print("  ► FIX: Add a station name column to your coords CSV.")
            print("    The column must be named one of:")
            print("      'station', 'station_id', 'station_name', 'Name', 'ID'")
            print()
            print("  ► QUICK FIX in Python:")
            print("    import pandas as pd")
            print("    df = pd.read_csv(STATION_CSV)")
            print("    # Add station names matching your datums CSV exactly:")
            print("    df['station'] = ['Chandpur 1640', 'Char Ramdaspur 1620', ...]")
            print("    df.to_csv(STATION_CSV, index=False)")
        else:
            print("  ✗ Station names in datums CSV do not match coords CSV.")
            print("    Rename stations in one CSV to match the other.")
        print("  ════════════════════════════════════════════════════")
    print("=" * 65 + "\n")



class YearProcessor:
    """
    Processes one calendar year's WSE stack, clipped to the study grid.

    Design decisions
    ----------------
    • All rasters are read with a rasterio Window — only the study grid
      pixels are loaded (~15–40 MB/image vs ~260 MB for the full raster).
    • River pixels are identified directly from WSE values; no separate
      channel-mask is required.
    • Dry-season occurrence used for intertidal classification.
    • Both elevation methods (A: Tseng, B: waterline median) produced.
    • Datum year searched across hyd_year ∈ {year, year−1}.
    """

    def __init__(self, year, wse_files, datums_df, station_xy,
                 output_dir, study_grid: StudyGrid, idw_power=2.0):
        self.year        = year
        self.output_dir  = Path(output_dir) / str(year)
        self.idw_power   = idw_power
        self.study_grid  = study_grid
        self.station_xy  = station_xy
        self.all_files   = sorted(wse_files, key=lambda p: _parse_dt(p.name))
        self.dry_files   = [f for f in self.all_files if _is_dry(_parse_dt(f.name))]

        cands = datums_df[
            (datums_df["season"]   == "dry")
            & (datums_df["hyd_year"].isin([year, year - 1]))
            & (datums_df["valid"]  == True)
        ].copy()
        if len(cands) > 0 and "completeness_pct" in cands.columns:
            self.datums = (cands.sort_values("completeness_pct", ascending=False)
                           .drop_duplicates(subset=["station"])
                           .reset_index(drop=True))
        else:
            self.datums = cands.reset_index(drop=True)

        log.info("Year %d: %d total / %d dry imgs | %d datum rows",
                 year, len(self.all_files), len(self.dry_files), len(self.datums))
        if self.datums.empty:
            log.warning("Year %d: no dry datums → Method A NaN; Method B runs.", year)

    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Optional[Dict[str, Path]]:
        if not self.all_files:
            log.warning("Year %d: no files.", self.year)
            return None

        use_dry   = bool(self.dry_files)
        stack_all = self._load(self.all_files)
        H, W      = self.study_grid.height, self.study_grid.width
        stack_dry = self._load(self.dry_files) if use_dry else stack_all

        # River and valid masks
        riv_dry = _is_river(stack_dry); val_dry = np.isfinite(stack_dry)
        riv_all = _is_river(stack_all); val_all = np.isfinite(stack_all)

        with np.errstate(invalid="ignore", divide="ignore"):
            occ_dry = np.where(val_dry.sum(0) > 0,
                               riv_dry.sum(0) / val_dry.sum(0).astype(float),
                               np.nan).astype(np.float32)
            occ_all = np.where(val_all.sum(0) > 0,
                               riv_all.sum(0) / val_all.sum(0).astype(float),
                               np.nan).astype(np.float32)

        intertidal = (occ_dry > 0) & (occ_dry < 1) & (val_dry.sum(0) >= MIN_DRY_OBS)
        log.info("Year %d: %d intertidal px", self.year, int(intertidal.sum()))

        # h_min and h_median from dry-season wet observations
        wse_dry = np.where(riv_dry, stack_dry, np.nan)
        any_wet = riv_dry.any(0)
        h_min = h_med = None
        h_min = np.full((H, W), np.nan, dtype=np.float32)
        h_med = np.full((H, W), np.nan, dtype=np.float32)
        if any_wet.any():
            cols = wse_dry[:, any_wet]
            h_min[any_wet] = np.nanmin   (cols, 0).astype(np.float32)
            h_med[any_wet] = np.nanmedian(cols, 0).astype(np.float32)

        # Spatially interpolated datums
        mhsw_g, mlsw_g = self._interp_datums(H, W)

        # Method A — Bishop-Taylor / Tseng  (CORRECTED formula)
        # CORRECT:  z = MHSW - f * (MHSW - MLSW)
        # Physical logic:  f=0 (never wet)  -> z ~ MHSW (supratidal crest)
        #                  f=1 (always wet) -> z ~ MLSW (subtidal base)
        # Previous code used h_min + f*range, which OVERESTIMATES by ~f*range
        # because h_min ~ z_true, giving z_est ~ z_true + f*range >> z_true.
        # Reference: Bishop-Taylor et al. (2019) Eq.1; Tseng et al. (2017) Eq.3.
        elev_A  = np.full((H, W), np.nan, dtype=np.float32)
        tr      = mhsw_g - mlsw_g
        mA = intertidal & np.isfinite(mhsw_g) & np.isfinite(tr) & (tr > 0)
        if mA.any():
            elev_A[mA] = mhsw_g[mA] - occ_dry[mA] * tr[mA]
            log.info("Year %d  A (BishopTaylor): %d px [%.3f-%.3f m]",
                     self.year, mA.sum(),
                     float(np.nanmin(elev_A)), float(np.nanmax(elev_A)))
        else:
            log.warning("Year %d  A: no valid pixels.", self.year)
        # Method B — waterline median (preferred)
        elev_B = np.full((H, W), np.nan, dtype=np.float32)
        mB = intertidal & np.isfinite(h_med)
        if mB.any():
            elev_B[mB] = h_med[mB]
            log.info("Year %d  B: %d px [%.3f–%.3f m]", self.year, mB.sum(),
                     float(np.nanmin(elev_B)), float(np.nanmax(elev_B)))

        # Write outputs (clipped to study grid)
        prof = self.study_grid.profile
        out  = {}
        for key, data, stem in [
            ("occurrence_dry",   occ_dry,  "occurrence_dry"),
            ("occurrence_all",   occ_all,  "occurrence_all"),
            ("h_min",            h_min,    "h_min_wse"),
            ("h_median",         h_med,    "h_median_wse"),
            ("mhsw",             mhsw_g,   "mhsw"),
            ("mlsw",             mlsw_g,   "mlsw"),
            ("elevation_bt",     elev_A,   "elevation_bt"),
            ("elevation_median", elev_B,   "elevation_median"),
        ]:
            p = self.output_dir / f"{stem}_{self.year}.tif"
            _write_tif(data, p, prof)
            out[key] = p
        return out

    # ─────────────────────────────────────────────────────────────────────

    def _load(self, files: List[Path]) -> np.ndarray:
        H, W, win = self.study_grid.height, self.study_grid.width, self.study_grid.window
        arrays = []
        for f in files:
            with rasterio.open(f) as src:
                data = src.read(1, window=win).astype(np.float32)
                nd   = src.nodata
                if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                    data[data == nd] = np.nan
                if data.shape != (H, W):
                    padded = np.full((H, W), np.nan, dtype=np.float32)
                    padded[:data.shape[0], :data.shape[1]] = data
                    data = padded
            arrays.append(data)
        return np.stack(arrays, 0)

    def _interp_datums(self, H, W) -> Tuple[np.ndarray, np.ndarray]:
        empty = np.full((H, W), np.nan, dtype=np.float32)
        if self.datums.empty:
            return empty, empty.copy()
        sta  = self.datums["station"].values
        mhsw = self.datums["mhsw_m"].values.astype(np.float64)
        mlsw = self.datums["mlsw_m"].values.astype(np.float64)
        # Normalize station names for robust matching (case/whitespace insensitive)
        def _norm(s): return str(s).strip().lower()
        ok   = np.array([_norm(s) in self.station_xy or s in self.station_xy
                         for s in sta])
        if not ok.any():
            log.warning(
                "Year %d: no datum stations matched coordinates.\n"
                "  Datums stations : %s\n"
                "  Coords stations : %s\n"
                "  Fix: ensure station names in datums CSV match station CSV "
                "(case-insensitive, whitespace trimmed).",
                self.year,
                list(sta[:8]),
                list(self.station_xy.keys())[:8],
            )
            return empty, empty.copy()
        def _get(s):
            return self.station_xy.get(_norm(s), self.station_xy.get(s))
        src_xy = np.array([_get(s) for s in sta[ok]])
        mhsw = mhsw[ok]; mlsw = mlsw[ok]
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        tgt = _pixel_centres(rr.ravel(), cc.ravel(), self.study_grid.transform)
        CHUNK = 500_000
        mf = np.empty(H*W, dtype=np.float32); lf = np.empty(H*W, dtype=np.float32)
        for s in range(0, H*W, CHUNK):
            e = min(s+CHUNK, H*W)
            mf[s:e] = _idw(mhsw, src_xy, tgt[s:e], self.idw_power)
            lf[s:e] = _idw(mlsw, src_xy, tgt[s:e], self.idw_power)
        return mf.reshape(H, W), lf.reshape(H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IntertidalElevationProcessor:
    """
    Full pipeline: river mask → study grid → per-year elevation.

    Parameters
    ----------
    wse_dir           directory of WSE GeoTIFFs (*_wse.tif)
    datums_csv        mhsw_mlsw_results.csv
    station_coords    pd.DataFrame or CSV path with station positions
    output_dir        root for all outputs
    crs_wl            EPSG of station_coords  (default "EPSG:4326")
    idw_power         IDW exponent  (default 2)
    glob_pattern      filename glob  (default "*_wse.tif")
    years             specific years to process  (default: all found)
    overwrite         reprocess existing outputs
    buffer_m          land-side buffer for study grid  (default 3000 m)
    mask_sample_every use every N-th file for river mask  (default 1 = all)
    """

    def __init__(
        self,
        wse_dir,
        datums_csv,
        station_coords,
        output_dir,
        *,
        crs_wl="EPSG:4326",
        idw_power=2.0,
        glob_pattern="*_wse.tif",
        years=None,
        overwrite=False,
        buffer_m=STUDY_GRID_BUFFER_M,
        mask_sample_every=1,
    ):
        self.wse_dir           = Path(wse_dir)
        self.output_dir        = Path(output_dir)
        self.idw_power         = idw_power
        self.glob_pattern      = glob_pattern
        self.years             = years
        self.overwrite         = overwrite
        self.crs_wl            = crs_wl
        self.buffer_m          = buffer_m
        self.mask_sample_every = mask_sample_every

        self.datums_df = pd.read_csv(datums_csv)
        if "valid" not in self.datums_df.columns:
            self.datums_df["valid"] = True
        self.datums_df["valid"] = self.datums_df["valid"].astype(bool)

        self.coords_df = (
            pd.read_csv(station_coords)
            if isinstance(station_coords, (str, Path))
            else station_coords.copy()
        )
        self._station_xy: Optional[Dict] = None
        self._study_grid: Optional[StudyGrid] = None

    def run(self) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Build / load study grid ───────────────────────────────────────
        grid_json = self.output_dir / "study_grid_info.json"
        union_tif = self.output_dir / "river_mask_union_clipped.tif"

        if union_tif.exists() and grid_json.exists() and not self.overwrite:
            log.info("Loading existing study grid from %s", grid_json)
            ref_tif = sorted(self.wse_dir.glob(self.glob_pattern))[0]
            self._study_grid = StudyGrid.from_json(grid_json, union_tif)
            log.info("Study grid: %d x %d px (%.0f km²)",
                     self._study_grid.width, self._study_grid.height,
                     self._study_grid.area_km2)
        else:
            builder = RiverMaskBuilder(
                self.wse_dir, self.glob_pattern,
                sample_every=self.mask_sample_every,
                buffer_m=self.buffer_m,
            )
            self._study_grid, _ = builder.build(output_dir=self.output_dir)

        # ── Resolve station coordinates ───────────────────────────────────
        all_files = sorted(self.wse_dir.glob(self.glob_pattern))
        if not all_files:
            raise FileNotFoundError(f"No '{self.glob_pattern}' in {self.wse_dir}")

        with rasterio.open(all_files[0]) as src:
            crs_wkt = src.crs.to_wkt() if src.crs else ""
        self._station_xy = _resolve_station_xy(
            self.coords_df, crs_wkt, self.crs_wl
        )
        log.info("Station coords: %d stations", len(self._station_xy))

        # Datum status
        dry_ok = self.datums_df[
            (self.datums_df["season"] == "dry") & (self.datums_df["valid"] == True)
        ]
        if dry_ok.empty:
            log.warning("No valid dry-season datums. Method A will be NaN.")
        else:
            log.info("Dry datums: %d rows | hyd_year %d–%d | %d stations",
                     len(dry_ok), int(dry_ok["hyd_year"].min()),
                     int(dry_ok["hyd_year"].max()), dry_ok["station"].nunique())

        # ── Catalogue by year ──────────────────────────────────────────────
        by_year: Dict[int, List[Path]] = {}
        for f in all_files:
            dt = _parse_dt(f.name)
            if dt:
                by_year.setdefault(dt.year, []).append(f)

        target_years = self.years or sorted(by_year.keys())
        log.info("Processing %d year(s) across %d files",
                 len(target_years), len(all_files))

        # ── Per-year loop ──────────────────────────────────────────────────
        summary_rows = []
        it = (tqdm(target_years, desc="Intertidal v3") if _HAS_TQDM
              else target_years)

        for year in it:
            done = self.output_dir / str(year) / f"elevation_median_{year}.tif"
            if done.exists() and not self.overwrite:
                log.info("Year %d: skip.", year)
                summary_rows.append({"year": year, "status": "skipped"})
                continue

            files = by_year.get(year, [])
            if not files:
                continue

            proc = YearProcessor(
                year, files, self.datums_df,
                self._station_xy, self.output_dir,
                self._study_grid, self.idw_power,
            )
            try:
                out = proc.run()
                if not out:
                    summary_rows.append({"year": year, "status": "no_output"})
                    continue
                row = {"year": year, "n_images": len(files),
                       "n_dry": len(proc.dry_files), "status": "ok"}
                for method, ek in [("bt","elevation_bt"),
                                    ("median","elevation_median")]:
                    ep = out.get(ek); op = out.get("occurrence_dry")
                    if ep and Path(ep).exists() and op and Path(op).exists():
                        with rasterio.open(ep) as src: e = src.read(1)
                        with rasterio.open(op) as src: o = src.read(1)
                        inter = (o > 0) & (o < 1) & np.isfinite(e)
                        row[f"n_inter_{method}"]     = int(inter.sum())
                        row[f"elev_mean_{method}_m"] = (
                            float(np.nanmean(e[inter])) if inter.any() else np.nan)
                summary_rows.append(row)
            except Exception as exc:
                log.error("Year %d: %s", year, exc, exc_info=True)
                summary_rows.append({"year": year, "status": f"error: {exc}"})

        df = (pd.DataFrame(summary_rows).sort_values("year")
              .reset_index(drop=True))
        csv = self.output_dir / "intertidal_elevation_summary_v3.csv"
        df.to_csv(csv, index=False, float_format="%.4f")
        log.info("Summary → %s", csv)
        ok = (df["status"] == "ok").sum()
        log.info("Done: %d/%d years OK.", ok, len(target_years))
        print(df.to_string(index=False))
        return df

