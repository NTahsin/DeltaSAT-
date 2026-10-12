"""
intertidal_elevation.py  (v2)
==============================
Pixel-wise intertidal area and elevation from WSE GeoTIFFs and tidal datums.

Methodological improvements over v1
-------------------------------------
1. Dry-season occurrence only.
   Landsat stacks include monsoon images where fluvial flooding makes
   non-intertidal land look like river, inflating occurrence. Only
   November-May (dry-season) images define f_dry and the intertidal mask.

2. Two elevation methods produced as separate outputs.

   Method A  Bishop-Taylor (PRIMARY):
       z_A = MHSW - f_dry * (MHSW - MLSW)
       Reference: Bishop-Taylor et al. (2019) RSE Eq.1; Tseng et al. (2017).
       f=0 (never wet) → z ≈ MHSW  |  f=1 (always wet) → z ≈ MLSW
       Requires tidal datums (MHSW, MLSW) from gauge stations.

   Method B  Per-pixel waterline median (preferred):
       z_B = median{ WSE(i,j,t) | dry-season image AND pixel is wet at t }
       Uses real WSE observations directly. No tidal datum dependency.
       Robust to outlier WSEs. Consistent with the waterline method
       (Salameh et al. 2020; BishopTaylor et al. 2019; Chen et al. 2023).

3. Datum year mismatch fix.
   Dry season Nov(Y)-May(Y+1) is labelled hyd_year=Y in the CSV.
   Both hyd_year==year and hyd_year==year-1 are now searched; the row
   with the highest data completeness is kept per station.

4. Minimum observation threshold.
   At least MIN_DRY_OBS (default 2) valid dry-season images per pixel
   before classifying as intertidal.

Output GeoTIFFs (per year, in output_dir/YYYY/)
-------------------------------------------------
  occurrence_dry_YYYY.tif      dry-season inundation frequency f_dry(i,j)
  occurrence_all_YYYY.tif      all-image occurrence (diagnostic)
  h_min_YYYY.tif               minimum dry-season WSE when wet
  h_median_YYYY.tif            median  dry-season WSE when wet
  mhsw_YYYY.tif                IDW-interpolated dry-season MHSW (m)
  mlsw_YYYY.tif                IDW-interpolated dry-season MLSW (m)
  elevation_bt_YYYY.tif        Method A  Bishop-Taylor  z = MHSW − f·ΔTide  (PRIMARY)
  elevation_median_YYYY.tif    Method B  waterline median  (SUPPLEMENTARY)

Dependencies
------------
    pip install numpy pandas rasterio scipy pyproj tqdm
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as rasterio_xy

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

LAND_VALUE   =  999.0
RIVER_MARKER = -999.0
DRY_MONTHS   = [11, 12, 1, 2, 3, 4, 5]
WSE_RANGE    = (-30.0, 30.0)
MIN_DRY_OBS  = 2

_DATE_PATS = [
    (r"(\d{4}-\d{2}-\d{2})_(\d{6})", "%Y-%m-%d %H%M%S"),
    (r"(\d{4}-\d{2}-\d{2})",          "%Y-%m-%d"),
    (r"(\d{8})_(\d{6})",              "%Y%m%d %H%M%S"),
    (r"(\d{8})",                       "%Y%m%d"),
]


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


def _idw(values, src_xy, tgt_xy, power=2.0, min_dist=1.0):
    diff  = src_xy[:, np.newaxis, :] - tgt_xy[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    dists = np.where(dists < min_dist, min_dist, dists)
    w     = 1.0 / dists ** power
    res   = (w * values[:, np.newaxis]).sum(axis=0) / w.sum(axis=0)
    exact = dists.min(axis=0) < min_dist
    if exact.any():
        res[exact] = values[dists.argmin(axis=0)[exact]]
    return res


def _write_geotiff(data, path, profile, nodata=np.nan):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(np.float32), 1)


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




def diagnose_station_matching(
    datums_csv,
    station_csv,
    year=None,
    season="dry",
):
    """
    Print diagnostic: which datum stations match the coordinate CSV.
    Run this if Method A gives "no datum stations matched coordinates".

    Usage
    -----
    from gbm_delta.morphodynamics.elevation_v2 import diagnose_station_matching
    diagnose_station_matching(DATUMS_CSV, STATION_CSV, year=1996)
    """
    datums = pd.read_csv(datums_csv)
    coords = pd.read_csv(station_csv)

    print()
    print("=" * 65)
    print("  Station Matching Diagnostic")
    print("=" * 65)
    print(f"  Datums CSV columns : {list(datums.columns)}")
    print(f"  Coords CSV columns : {list(coords.columns)}")
    print()
    print("  Coords CSV (first 3 rows):")
    for _, row in coords.head(3).iterrows():
        print(f"    {dict(row)}")
    print()

    # Mirror the column detection logic from _resolve_station_xy
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

    def _looks_num(col):
        try:
            v = coords[col].dropna().astype(float)
            return len(v) > 0 and ((v.abs() < 200).mean() > 0.8)
        except Exception:
            return False

    if id_col is None:
        for col in coords.columns:
            if not _looks_num(col):
                id_col = col; break
    if id_col is None:
        id_col = coords.columns[0]

    print(f"  ID column selected : '{id_col}'")
    is_coord_col = _looks_num(id_col)
    if is_coord_col:
        print(f"  *** WARNING: '{id_col}' looks like a coordinate column, not names! ***")

    coord_names = sorted(coords[id_col].astype(str).str.strip().unique())
    coord_norms = {s.strip().lower(): s for s in coord_names}
    print(f"  Coord IDs (first 8): {coord_names[:8]}")
    print()

    if year is not None:
        d = datums[(datums["season"]==season) &
                   (datums["hyd_year"].isin([year, year-1]))]
        check = sorted(d["station"].astype(str).str.strip().unique())
    else:
        check = sorted(datums["station"].astype(str).str.strip().unique())

    print(f"  Matching {len(check)} datum stations  (year={year})")
    print("  " + "-" * 60)
    n_ok = 0
    for s in check:
        sn = s.strip().lower()
        if s in coord_names:
            status = "MATCH (exact)"; n_ok += 1
        elif sn in coord_norms:
            status = f"MATCH (normalised → {coord_norms[sn]})"; n_ok += 1
        else:
            status = "NO MATCH"
        print(f"    {s:<30}  {status}")

    print()
    print(f"  Result: {n_ok}/{len(check)} matched")

    if n_ok == 0:
        print()
        if is_coord_col:
            print("  ══ ROOT CAUSE ══════════════════════════════════════════")
            print(f"  The code selected '{id_col}' as the station ID column")
            print("  but it contains coordinate values (lat/lon), not names.")
            print()
            print("  FIX: Add a 'station' column to your coords CSV with the")
            print("  station names matching those in the datums CSV.")
            print()
            print("  QUICK FIX — run this once:")
            print("    import pandas as pd")
            print("    df = pd.read_csv(STATION_CSV)")
            print("    print(df.columns.tolist())  # see current columns")
            print("    # Then rename the correct column:")
            print("    df = df.rename(columns={'YourNameColumn': 'station'})")
            print("    df.to_csv(STATION_CSV, index=False)")
            print("  ════════════════════════════════════════════════════════")
        else:
            print("  Names in datums and coords do not match spelling.")
            print("  Edit one CSV to make names match (case-insensitive ok).")
    print("=" * 65)


class YearProcessor:
    def __init__(self, year, wse_files, datums_df, station_xy, output_dir, idw_power=2.0):
        self.year       = year
        self.output_dir = output_dir / str(year)
        self.idw_power  = idw_power
        self.station_xy = station_xy
        self.all_files  = sorted(wse_files, key=lambda p: _parse_dt(p.name))
        self.dry_files  = [f for f in self.all_files if _is_dry(_parse_dt(f.name))]

        # FIX: accept hyd_year==year OR hyd_year==year-1 to handle
        # the Nov-May dry season crossing the calendar year boundary.
        candidates = datums_df[
            (datums_df["season"]   == "dry")
            & (datums_df["hyd_year"].isin([year, year - 1]))
            & (datums_df["valid"]  == True)
        ].copy()

        if len(candidates) > 0 and "completeness_pct" in candidates.columns:
            self.datums = (
                candidates
                .sort_values("completeness_pct", ascending=False)
                .drop_duplicates(subset=["station"])
                .reset_index(drop=True)
            )
        else:
            self.datums = candidates.reset_index(drop=True)

        log.info(
            "Year %d: %d total / %d dry images | %d datum rows (hyd_year in [%d,%d])",
            year, len(self.all_files), len(self.dry_files),
            len(self.datums), year-1, year,
        )
        if self.datums.empty:
            log.warning("Year %d: no dry-season datums -> Method A will be NaN.", year)

    def run(self):
        if not self.all_files:
            log.warning("Year %d: no files.", self.year)
            return None

        use_dry = bool(self.dry_files)
        if not use_dry:
            log.warning("Year %d: no dry-season images; using all images.", self.year)

        stack_all, transform, profile = self._load_stack(self.all_files)
        H, W = profile["height"], profile["width"]
        stack_dry = self._load_stack(self.dry_files)[0] if use_dry else stack_all

        def river_mask(stack):
            return (
                np.isfinite(stack)
                & ~np.isclose(stack, LAND_VALUE,   atol=0.5)
                & ~np.isclose(stack, RIVER_MARKER, atol=0.5)
                & (stack > WSE_RANGE[0]) & (stack < WSE_RANGE[1])
            )

        is_river_dry = river_mask(stack_dry)
        is_valid_dry = np.isfinite(stack_dry)
        is_river_all = river_mask(stack_all)
        is_valid_all = np.isfinite(stack_all)

        with np.errstate(invalid="ignore", divide="ignore"):
            occ_dry = np.where(is_valid_dry.sum(0) > 0,
                               is_river_dry.sum(0) / is_valid_dry.sum(0).astype(float),
                               np.nan).astype(np.float32)
            occ_all = np.where(is_valid_all.sum(0) > 0,
                               is_river_all.sum(0) / is_valid_all.sum(0).astype(float),
                               np.nan).astype(np.float32)

        intertidal = (occ_dry > 0) & (occ_dry < 1) & (is_valid_dry.sum(0) >= MIN_DRY_OBS)
        log.info("Year %d: %d intertidal px (dry f in (0,1), >=%d obs)",
                 self.year, int(intertidal.sum()), MIN_DRY_OBS)

        wse_masked = np.where(is_river_dry, stack_dry, np.nan)
        any_wet    = is_river_dry.any(0)
        h_min    = np.full((H, W), np.nan, dtype=np.float32)
        h_median = np.full((H, W), np.nan, dtype=np.float32)
        if any_wet.any():
            cols_wet = wse_masked[:, any_wet]
            h_min[any_wet]    = np.nanmin   (cols_wet, axis=0).astype(np.float32)
            h_median[any_wet] = np.nanmedian(cols_wet, axis=0).astype(np.float32)

        mhsw_grid, mlsw_grid = self._interpolate_datums(H, W, transform)

        # Method A: Bishop-Taylor  z = MHSW - f_dry * (MHSW - MLSW)
        # Reference: Bishop-Taylor et al. (2019) RSE Eq.1; Tseng et al. (2017) Eq.3.
        elev_bt     = np.full((H, W), np.nan, dtype=np.float32)
        tidal_range = mhsw_grid - mlsw_grid
        mask_A = intertidal & np.isfinite(mhsw_grid) & np.isfinite(tidal_range) & (tidal_range > 0)
        if mask_A.any():
            elev_bt[mask_A] = mhsw_grid[mask_A] - occ_dry[mask_A] * tidal_range[mask_A]
            log.info("Year %d  Method A (Bishop-Taylor): %d px | z [%.3f, %.3f] m",
                     self.year, int(mask_A.sum()),
                     float(np.nanmin(elev_bt)), float(np.nanmax(elev_bt)))
        else:
            log.warning("Year %d  Method A: no valid pixels.", self.year)
        # Method B: Per-pixel waterline median (preferred)
        elev_median = np.full((H, W), np.nan, dtype=np.float32)
        mask_B = intertidal & np.isfinite(h_median)
        if mask_B.any():
            elev_median[mask_B] = h_median[mask_B]
            log.info("Year %d  Method B: %d px | z [%.3f, %.3f] m",
                     self.year, int(mask_B.sum()),
                     float(np.nanmin(elev_median)), float(np.nanmax(elev_median)))

        out_paths = {}
        for key, data, stem in [
            ("occurrence_dry",   occ_dry,     "occurrence_dry"),
            ("occurrence_all",   occ_all,     "occurrence_all"),
            ("h_min",            h_min,       "h_min_wse"),
            ("h_median",         h_median,    "h_median_wse"),
            ("mhsw",             mhsw_grid,   "mhsw"),
            ("mlsw",             mlsw_grid,   "mlsw"),
            ("elevation_bt",     elev_bt,     "elevation_bt"),
            ("elevation_median", elev_median, "elevation_median"),
        ]:
            p = self.output_dir / f"{stem}_{self.year}.tif"
            _write_geotiff(data, p, profile)
            out_paths[key] = p
        return out_paths

    def _load_stack(self, files):
        arrays, transform, profile = [], None, None
        for f in files:
            with rasterio.open(f) as src:
                data = src.read(1).astype(np.float32)
                nd   = src.nodata
                if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                    data[data == nd] = np.nan
                arrays.append(data)
                if transform is None:
                    transform = src.transform
                    profile   = src.profile.copy()
        return np.stack(arrays, axis=0), transform, profile

    def _interpolate_datums(self, H, W, transform):
        empty = np.full((H, W), np.nan, dtype=np.float32)
        if self.datums.empty:
            return empty, empty.copy()
        sta_ids  = self.datums["station"].values
        mhsw_val = self.datums["mhsw_m"].values.astype(np.float64)
        mlsw_val = self.datums["mlsw_m"].values.astype(np.float64)
        # Normalize station names for robust matching (case/whitespace insensitive)
        def _norm(s): return str(s).strip().lower()
        mask = np.array([_norm(s) in self.station_xy or s in self.station_xy
                         for s in sta_ids])
        if not mask.any():
            log.warning(
                "Year %d: no datum stations matched coordinates.\n"
                "  Datums stations : %s\n"
                "  Coords stations : %s\n"
                "  Fix: ensure station names in datums CSV match station CSV "
                "(case-insensitive, whitespace trimmed).",
                self.year,
                list(sta_ids[:8]),
                list(self.station_xy.keys())[:8],
            )
            return empty, empty.copy()
        def _get(s):
            return self.station_xy.get(_norm(s), self.station_xy.get(s))
        src_xy   = np.array([_get(s) for s in sta_ids[mask]])
        mhsw_val = mhsw_val[mask]; mlsw_val = mlsw_val[mask]
        rr, cc   = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        tgt_xy   = _pixel_centres(rr.ravel(), cc.ravel(), transform)
        CHUNK    = 500_000
        mf = np.empty(H*W, dtype=np.float32); lf = np.empty(H*W, dtype=np.float32)
        for s in range(0, H*W, CHUNK):
            e = min(s+CHUNK, H*W)
            mf[s:e] = _idw(mhsw_val, src_xy, tgt_xy[s:e], self.idw_power)
            lf[s:e] = _idw(mlsw_val, src_xy, tgt_xy[s:e], self.idw_power)
        return mf.reshape(H,W), lf.reshape(H,W)


class IntertidalElevationProcessor:
    """
    Derives pixel-wise intertidal elevation for each year in the WSE archive.

    Parameters: see module docstring.
    """

    def __init__(self, wse_dir, datums_csv, station_coords, output_dir, *,
                 crs_wl="EPSG:4326", idw_power=2.0, glob_pattern="*_wse.tif",
                 years=None, overwrite=False):
        self.wse_dir      = Path(wse_dir)
        self.output_dir   = Path(output_dir)
        self.idw_power    = idw_power
        self.glob_pattern = glob_pattern
        self.years        = years
        self.overwrite    = overwrite
        self.crs_wl       = crs_wl
        self.datums_df    = pd.read_csv(datums_csv)
        if "valid" not in self.datums_df.columns:
            self.datums_df["valid"] = True
        self.datums_df["valid"] = self.datums_df["valid"].astype(bool)
        self.coords_df    = (pd.read_csv(station_coords)
                             if isinstance(station_coords, (str, Path))
                             else station_coords.copy())
        self._station_xy  = None

    def run(self):
        all_files = sorted(self.wse_dir.glob(self.glob_pattern))
        if not all_files:
            raise FileNotFoundError(f"No '{self.glob_pattern}' in {self.wse_dir}")

        by_year = {}
        for f in all_files:
            dt = _parse_dt(f.name)
            if dt:
                by_year.setdefault(dt.year, []).append(f)

        target_years = self.years or sorted(by_year.keys())

        # Report datum status
        dry_ok = self.datums_df[
            (self.datums_df["season"] == "dry") & (self.datums_df["valid"] == True)
        ]
        if dry_ok.empty:
            log.warning("No valid dry-season datums. Method A will be NaN everywhere.")
        else:
            log.info("Dry datums: %d rows | hyd_year %d-%d | %d stations",
                     len(dry_ok), int(dry_ok["hyd_year"].min()),
                     int(dry_ok["hyd_year"].max()), dry_ok["station"].nunique())

        log.info("Processing %d years / %d total WSE files",
                 len(target_years), len(all_files))

        summary_rows = []
        iterable = tqdm(target_years, desc="Elevation v2") if _HAS_TQDM else target_years

        for year in iterable:
            done = self.output_dir / str(year) / f"elevation_median_{year}.tif"
            if done.exists() and not self.overwrite:
                log.info("Year %d: skip (overwrite=False).", year)
                summary_rows.append({"year": year, "status": "skipped"})
                continue

            files = by_year.get(year, [])
            if not files:
                continue

            if self._station_xy is None:
                with rasterio.open(files[0]) as src:
                    crs_wkt = src.crs.to_wkt() if src.crs else ""
                self._station_xy = _resolve_station_xy(
                    self.coords_df, crs_wkt, self.crs_wl)
                log.info("Station coords resolved: %d stations", len(self._station_xy))

            proc = YearProcessor(year, files, self.datums_df,
                                 self._station_xy, self.output_dir, self.idw_power)
            try:
                out = proc.run()
                if not out:
                    summary_rows.append({"year": year, "status": "no_output"})
                    continue

                row = {"year": year, "n_images": len(files),
                       "n_dry_images": len(proc.dry_files), "status": "ok"}

                for method, ek, ok_key in [
                    ("bt",     "elevation_bt",     "occurrence_dry"),
                    ("median", "elevation_median", "occurrence_dry"),
                ]:
                    ep, op = out.get(ek), out.get(ok_key)
                    if ep and Path(ep).exists() and op and Path(op).exists():
                        with rasterio.open(ep) as src: e = src.read(1)
                        with rasterio.open(op) as src: o = src.read(1)
                        inter = (o > 0) & (o < 1) & np.isfinite(e)
                        row[f"n_intertidal_{method}"] = int(inter.sum())
                        row[f"elev_mean_{method}_m"]  = (
                            float(np.nanmean(e[inter])) if inter.any() else np.nan)

                summary_rows.append(row)

            except Exception as exc:
                log.error("Year %d: %s", year, exc, exc_info=True)
                summary_rows.append({"year": year, "status": f"error: {exc}"})

        df = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
        csv = self.output_dir / "intertidal_elevation_summary_v2.csv"
        df.to_csv(csv, index=False, float_format="%.4f")
        log.info("Summary -> %s", csv)
        ok = (df["status"] == "ok").sum()
        log.info("Done: %d/%d years OK.", ok, len(target_years))
        print(df.to_string(index=False))
        return df
