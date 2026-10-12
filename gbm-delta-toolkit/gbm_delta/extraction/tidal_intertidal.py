"""
tide_intertidal_detector.py
===========================
Tide-aware detection of channelised rivers and intertidal areas,
for the case where tide-gauge stations are located INSIDE the image.

Extends IntertidalDetector (intertidal_detector.py) with three extra
constraints on top of the count-based + singularity-index (SI, per-image
OTSU thresholding) classification:

  1. WATER-LEVEL SURFACE per scene date
     For every scene date, the readings of the n tide gauges are
     interpolated to a full raster: a first-order plane (the dominant
     downstream slope) fitted to the n readings, plus inverse-distance-
     weighted (IDW) residuals so local anomalies are respected.

  2. PER-PIXEL MONOTONIC TIDAL TEST
     For every pixel, the water level h_i of every scene is known
     (from the surface of that date). With
         h_wet_min = min(h_i) over scenes where the pixel is WATER
         h_dry_max = max(h_i) over scenes where the pixel is LAND
     a switching pixel is only confirmed INTERDIDAL if its inundation
     tracks the water level monotonically:
         h_wet_min > h_dry_max        (all wet scenes higher than all dry)
     Non-monotonic switchers (floodplain, paddies, noise) become land.
     For confirmed intertidal pixels the elevation
         z = (h_dry_max + h_wet_min) / 2     [gauge datum]
     is exported as an extra GeoTIFF.

  3. CHANNEL CONNECTIVITY BUFFER
     The permanent river (always water + consistent high SI) is cleaned
     to its connected components (size >= min_channel_px) → the
     CHANNELISED network. Intertidal pixels are only kept within
     buffer_m metres of that network ("in and around the channelised
     part"); isolated switching patches elsewhere become land.

FINAL RULES per year
--------------------
    river      : always water AND high SI (per-image Otsu) in every wet
                 scene AND part of a channel-sized connected component
    intertidal : switches AND high SI when water (SI rules) AND
                 h_wet_min > h_dry_max (tidal monotonic) AND within
                 buffer_m of the channelised river
    land       : everything else
    nodata     : fewer than min_scenes valid observations

Fallbacks
---------
    - Scene dates without a matching water-level reading are excluded
      from the tidal test (nearest reading within tide_match_days is
      used if available).
    - If NO scene of a year has water-level data, the tidal test is
      skipped for that year (channel buffer + SI rules still apply).
    - If SI rasters are missing for a pixel, the count-based class is
      kept (same as IntertidalDetector).

Requirements
------------
    pip install numpy scipy tifffile
    GDAL (osgeo) or pyproj recommended — to project the WGS84 gauge
    coordinates into the raster CRS. Without either (or without CRS
    info in the rasters), gauge lon/lat are assumed to equal raster
    column/row (identity georeference) — only correct for rasters in
    geographic coordinates with 1-degree pixels.

Water-level file format
----------------------
    One CSV:  first column = date (and time); remaining columns = one
    per station, header = station name (matching the STATION_COORDS
    table below, case/spacing-insensitive). Missing values as blanks
    or NaN. Example:

        Date & Time,Chandpur,Char Ramdaspur,Rupsha,Meghna 03
        2005-01-15 04:30,1.85,1.62,1.43,1.71
        2005-01-15 05:30,1.91,1.70,1.50,1.78

    The acquisition time of each scene is resolved in this order:
      1. Landsat MTL metadata file (SCENE_CENTER_TIME, UTC) next to the
         raster,
      2. an optional scene-times CSV (scene_times_csv) matched by scene
         date (e.g. an EarthExplorer export of the <30% cloud scenes),
      3. a configurable default overpass time (default 10:30 local).
    The gauge series is then linearly interpolated to that exact local
    timestamp (mtl_utc_offset_hours, default +6 for Bangladesh).

Usage
-----
    from tide_intertidal_detector import TideIntertidalDetector

    det = TideIntertidalDetector(
        river_dir       = r'W:\\...\\02_River Extracted',
        output_dir      = r'W:\\...\\04_Intertidal',
        si_dir          = r'W:\\...\\03_Singularity Index',
        waterlevel_csv  = r'W:\\...\\waterlevels.csv',
        buffer_m        = 1000.0,     # intertidal within 1 km of channel
        dry_season_only = False,      # True: tidal test on Nov-Apr scenes
    )
    det.run()
"""

import os
import re
import csv
import datetime as dt
import warnings
from collections import defaultdict

import numpy as np
from scipy import ndimage

from gbm_delta.extraction.intertidal_base import (
    IntertidalDetector, LAND, RIVER, INTERTIDAL, NODATA,
)


DRY_MONTHS = {11, 12, 1, 2, 3, 4}


class TideIntertidalDetector(IntertidalDetector):
    """
    IntertidalDetector + tide-gauge validation + channel-connectivity
    buffer. See module docstring for the full rule set.

    Extra parameters (on top of IntertidalDetector)
    ------------------------------------------------
    waterlevel_csv   : str   CSV with a date(time) column + one column per
                           tide-gauge station (see module docstring).
    stations         : dict  optional override of the station table,
                           {name: (lat, lon)}. Default: the GBM gauge
                           network listed in STATION_COORDS.
    buffer_m         : float max distance (metres) between an intertidal
                           pixel and the channelised river network.
                           Default 1000.
    min_channel_px   : int   connected always-water components smaller than
                           this many pixels are not 'channel' (ponds,
                           noise) and are re-labelled land. Default 25.
    tide_match_days  : int   max days between the overpass timestamp and
                           the nearest gauge records for the scene to
                           still count as tide-informed. Default 2.
    use_tide         : bool  False = skip the tidal monotonic test
                           entirely (no water-level surface, no elevation
                           raster); SI rule + channel connectivity +
                           buffer still apply. Default True.
    dry_season_only  : bool  restrict the tidal monotonic test to
                           Nov-Apr scenes (excludes monsoon freshet).
                           Default False.
    monotonic_tol_m  : float tolerance for h_wet_min > h_dry_max.
                           Default 0.0 (strict). Slightly negative values
                           (e.g. -0.05) tolerate one marginally
                           mis-ranked scene.
    range_min_m      : float minimum water-level range (m) a pixel must
                           have observed for its tide evidence to be
                           ALLOWED TO REJECT it. Pixels observed over a
                           narrower range keep the count-based class
                           (patchy-data protection). Default 0.15.
    tide_pool_years  : int   pool tide evidence over +/- this many years
                           (0 = target year only). Default 0.
    mtl_utc_offset_hours : float local standard time offset of the
                           water-level series relative to UTC, in hours
                           (Bangladesh = +6). Used when converting the
                           Landsat SCENE_CENTER_TIME (UTC) to the
                           timestamps of the gauge records. Default 6.0.
    scene_times_csv  : str   optional CSV of scene acquisition times in
                           UTC, one row per scene (an EarthExplorer export
                           or a saved pandas DataFrame — an exported
                           row-index column is ignored, and comma /
                           semicolon / tab separators all work). Used when
                           no MTL file sits next to a raster; matched by
                           scene date. Default None.
    overpass_time    : str   fallback local satellite overpass time
                           'HH:MM' when no MTL metadata is found.
                           Default '10:30'.
    """

    STATION_COORDS = {
        'chandpur':        (23.22098, 90.651716),
        'char ramdaspur':  (22.818138, 90.595889),
        'chitalkhali':     (22.659409, 90.903213),
        'dasmina':         (22.235616, 90.551491),
        'dhulia':          (22.559863, 90.533287),
        'charchenga':      (22.641081, 90.862361)
    }
    """
        'char montaj':     (21.904494, 90.509741),
        'dhulasur':        (21.870721, 90.256854),
        'monpura 1030':    (22.25855, 90.961526),
        'tajumuddin 1031': (22.336107, 90.832967),
        'rupsha':          (22.696012, 90.379032),
        'meghna 03':       (22.778728, 90.650268),
    """

    def __init__(
        self,
        river_dir:        str,
        output_dir:       str,
        waterlevel_csv:   str,
        si_dir:           str   = None,
        stations:         dict  = None,
        buffer_m:         float = 1000.0,
        min_channel_px:   int   = 25,
        tide_match_days:  int   = 2,
        use_tide:         bool  = True,
        dry_season_only:  bool  = False,
        monotonic_tol_m:  float = 0.0,
        range_min_m:      float = 0.15,
        tide_pool_years:  int   = 0,
        mtl_utc_offset_hours: float = 6.0,
        scene_times_csv:  str   = None,
        overpass_time:    str   = '10:30',
        si_prefix:        str   = 'singularity_index_',
        si_threshold:     float = 0.5,
        min_scenes:       int   = 2,
        pixel_size_m:     float = 30.0,
        skip_existing:    bool  = True,
        nodata_val:       int   = 255,
    ):
        super().__init__(
            river_dir=river_dir,
            output_dir=output_dir,
            si_dir=si_dir,
            si_prefix=si_prefix,
            si_threshold=si_threshold,
            min_scenes=min_scenes,
            pixel_size_m=pixel_size_m,
            skip_existing=skip_existing,
            nodata_val=nodata_val,
        )

        if use_tide and not os.path.exists(waterlevel_csv):
            raise FileNotFoundError(f"Water-level file not found: {waterlevel_csv}")
        if not use_tide and waterlevel_csv and not os.path.exists(waterlevel_csv):
            warnings.warn(
                f"Water-level file not found ({waterlevel_csv}) — "
                f"the tidal test is disabled anyway (use_tide=False)."
            )

        self.waterlevel_csv  = waterlevel_csv
        self.buffer_m        = buffer_m
        self.min_channel_px  = min_channel_px
        self.tide_match_days = tide_match_days
        self.use_tide        = use_tide
        self.dry_season_only = dry_season_only
        self.monotonic_tol_m = monotonic_tol_m
        self.range_min_m     = range_min_m
        self.tide_pool_years = tide_pool_years
        self.mtl_utc_offset = mtl_utc_offset_hours
        self.scene_times_csv = scene_times_csv
        self.overpass_time   = overpass_time
        self._scene_times = None   # loaded lazily: {date_iso: utc_datetime}

        base = dict(self.STATION_COORDS)
        if stations:
            for k, v in stations.items():
                base[self._norm(k)] = (float(v[0]), float(v[1]))
        self.stations = base

        self._levels = None          # loaded lazily: [(datetime, {station: h})]
        self._surface_cache = {}     # date_str -> surface or None
        self._warned_identity_geo = False
        self._warned_unmatched = set()

        print(f"📁 Waterlevels: {os.path.abspath(waterlevel_csv)}")
        print(f"   Stations        : {len(self.stations)}")
        print(f"   Channel buffer  : {buffer_m} m  "
              f"(min component {min_channel_px} px)")
        print(f"   Tidal test      : h_wet_min > h_dry_max "
              f"(tol {monotonic_tol_m} m, "
              f"{'dry-season scenes only' if dry_season_only else 'all scenes'}; "
              f"range guard {range_min_m} m; pool {tide_pool_years} yr; "
              f"MTL times, UTC+{mtl_utc_offset_hours:g}h local)")

    # ─────────────────────────────────────────────────────────────────────────
    # Station / water-level handling
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(name: str) -> str:
        """Normalise a station name for matching (case/space-insensitive)."""
        return re.sub(r'\s+', ' ', str(name)).strip().lower()

    @staticmethod
    def _strip_id(name: str) -> str:
        """
        Remove trailing station-ID digits, e.g.
            'chandpur 1640'   -> 'chandpur'
            'monpura 1030'    -> 'monpura'
            'meghna 03'        -> 'meghna'
        Applied repeatedly so 'station 1030 2' also reduces fully.
        """
        prev = None
        cur = name
        while prev != cur:
            prev = cur
            cur = re.sub(r'[\s_]*\d+$', '', cur).strip()
        return cur

    @staticmethod
    def _compact(name: str) -> str:
        """Remove all whitespace/punctuation: 'char-ramdaspur' -> 'charramdaspur'."""
        return re.sub(r'[\s_\-]+', '', name)

    def _station_lookup(self) -> dict:
        """
        Build {variant_name: station_key} covering exact, ID-stripped and
        compact spellings of every station, so CSV columns like
        'Chandpur 1640' or 'Char-Ramdaspur 1620' still match.
        """
        if getattr(self, '_st_lookup', None) is None:
            lookup = {}
            for key in self.stations:
                stripped = self._strip_id(key)
                for variant in (key, stripped,
                                self._compact(key), self._compact(stripped)):
                    if variant and variant not in lookup:
                        lookup[variant] = key
            self._st_lookup = lookup
        return self._st_lookup

    def _match_station(self, column: str):
        """
        Match a CSV column header to a known station. Returns the station
        key or None. Tolerates appended station IDs ('Chandpur 1640'),
        missing IDs ('Monpura'), and punctuation/space variants
        ('Char-Ramdaspur').
        """
        key = self._norm(column)
        lookup = self._station_lookup()
        stripped = self._strip_id(key)
        for variant in (key, stripped,
                        self._compact(key), self._compact(stripped)):
            if variant in lookup:
                return lookup[variant]
        return None

    def _load_waterlevels(self) -> list:
        """
        Load the water-level CSV. Returns a sorted list of
        (datetime, {station_norm: h}) records.
        """
        if self._levels is not None:
            return self._levels

        records = []
        try:
            import pandas as pd
            df = pd.read_csv(self.waterlevel_csv)
            dt_col = df.columns[0]
            # Try ISO (YYYY-MM-DD) first; fall back to day-first parsing
            iso = pd.to_datetime(df[dt_col], format='%Y-%m-%d %H:%M:%S',
                                 errors='coerce')
            iso = iso.fillna(pd.to_datetime(df[dt_col], format='%Y-%m-%d %H:%M',
                                            errors='coerce'))
            iso = iso.fillna(pd.to_datetime(df[dt_col], format='%Y-%m-%d',
                                            errors='coerce'))
            if iso.isna().all():
                iso = pd.to_datetime(df[dt_col], dayfirst=True,
                                     errors='coerce')
            df[dt_col] = iso
            df = df.dropna(subset=[dt_col])
            for _, row in df.iterrows():
                levels = {}
                for col in df.columns[1:]:
                    st = self._match_station(col)
                    if st is None:
                        if self._norm(col) not in self._warned_unmatched:
                            self._warned_unmatched.add(self._norm(col))
                            warnings.warn(
                                f"Water-level column '{col}' does not match "
                                f"any known station — ignored."
                            )
                        continue
                    val = row[col]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    try:
                        levels[st] = float(val)
                    except (TypeError, ValueError):
                        continue
                if levels:
                    records.append((row[dt_col].to_pydatetime(), levels))
        except ImportError:
            with open(self.waterlevel_csv, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                for raw in reader:
                    if not raw or not raw[0].strip():
                        continue
                    ts = None
                    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
                                '%Y-%m-%d', '%d/%m/%Y %H:%M', '%d/%m/%Y'):
                        try:
                            ts = dt.datetime.strptime(raw[0].strip(), fmt)
                            break
                        except ValueError:
                            continue
                    if ts is None:
                        continue
                    levels = {}
                    for col, val in zip(header[1:], raw[1:]):
                        st = self._match_station(col)
                        if st is None:
                            if self._norm(col) not in self._warned_unmatched:
                                self._warned_unmatched.add(self._norm(col))
                                warnings.warn(
                                    f"Water-level column '{col}' does not "
                                    f"match any known station — ignored."
                                )
                            continue
                        val = val.strip()
                        if not val:
                            continue
                        try:
                            levels[st] = float(val)
                        except ValueError:
                            continue
                    if levels:
                        records.append((ts, levels))

        records.sort(key=lambda r: r[0])
        if not records:
            raise ValueError(
                f"No usable water-level records found in {self.waterlevel_csv}"
            )
        self._levels = records
        print(f"   Waterlevels     : {len(records)} records "
              f"({records[0][0]:%Y-%m-%d} → {records[-1][0]:%Y-%m-%d})")
        return records

    # ─────────────────────────────────────────────────────────────────────────
    # Scene acquisition time (from Landsat MTL metadata)
    # ─────────────────────────────────────────────────────────────────────────

    _MTL_TIME_RE = re.compile(
        r'SCENE_CENTER_TIME\s*=\s*"(\d{2}):(\d{2}):(\d+(?:\.\d+)?)Z?"?',
        re.IGNORECASE,
    )

    _TS_FORMATS = (
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d',
        '%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M', '%d-%m-%Y',
        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%d/%m/%Y',
    )

    @classmethod
    def _parse_timestamp(cls, raw: str):
        """
        Parse a date(time) string; ISO tried first, then day-first.
        Returns None for anything without a 19xx/20xx year (headers,
        bare times like '03:29', row indices).
        """
        raw = str(raw).strip()
        if not re.search(r'(19|20)\d{2}', raw):
            return None
        for fmt in cls._TS_FORMATS:
            try:
                return dt.datetime.strptime(raw, fmt)
            except ValueError:
                continue
        try:
            import pandas as pd
            return pd.to_datetime(raw, dayfirst=True).to_pydatetime()
        except Exception:
            return None

    def _load_scene_times(self) -> dict:
        """
        Load the optional scene-times CSV: a table of acquisition
        timestamps (UTC), one row per scene (e.g. an EarthExplorer
        export, or a pandas DataFrame saved with its row-index column).
        Every cell of every line is tried, so an exported integer index
        column, headers, and blank/unparseable cells are simply ignored;
        comma, semicolon and tab separators are all handled.
        Returns {date_iso: utc_datetime}.
        """
        if self._scene_times is not None:
            return self._scene_times

        times = {}
        if self.scene_times_csv and os.path.exists(self.scene_times_csv):
            with open(self.scene_times_csv, newline='') as f:
                for line in f:
                    for cell in re.split(r'[,;\t]', line.strip()):
                        cell = cell.strip().strip('"')
                        if not cell:
                            continue
                        ts = self._parse_timestamp(cell)
                        if ts is not None:
                            times[ts.strftime('%Y-%m-%d')] = ts

            if not times:
                warnings.warn(
                    f"No parseable timestamps found in "
                    f"{self.scene_times_csv}. Expected date(time) values "
                    f"such as '9-2-1996 03:29' or '1996-02-09 03:29'."
                )
            else:
                print(f"   Scene times     : {len(times)} timestamps from "
                      f"{os.path.basename(self.scene_times_csv)} (UTC, "
                      f"converted +{self.mtl_utc_offset:g}h)")

        self._scene_times = times
        return times

    def _scene_datetime(self, raster_path: str):
        """
        Local-standard-time acquisition datetime of a scene.

        Resolution order:
          1. SCENE_CENTER_TIME (UTC) from the Landsat MTL metadata file
             next to the raster (.MTL.txt / _MTL.txt / .txt / .MTL).
          2. The optional scene-times CSV (scene_times_csv), matched by
             scene date.
          3. The configured overpass_time on the scene date.
        UTC times are shifted by mtl_utc_offset_hours to match the gauge
        series' time zone.
        """
        date_str = self._date_from_filename(os.path.basename(raster_path))
        if not date_str:
            return None
        d = dt.datetime.strptime(date_str, '%Y-%m-%d').date()

        base = os.path.splitext(raster_path)[0]
        cands = [
            base + '.MTL.txt', base + '_MTL.txt', base + '.txt',
            base + '.MTL',
        ]
        for c in cands:
            if os.path.exists(c):
                try:
                    txt = open(c, encoding='utf-8', errors='ignore').read()
                except OSError:
                    continue
                m = self._MTL_TIME_RE.search(txt)
                if m:
                    hh, mm = int(m.group(1)), int(m.group(2))
                    ss = float(m.group(3))
                    utc = dt.datetime(d.year, d.month, d.day, hh, mm) \
                          + dt.timedelta(seconds=ss)
                    local = utc + dt.timedelta(hours=self.mtl_utc_offset)
                    return local

        # 2. scene-times CSV (matched by date)
        scene_times = self._load_scene_times()
        if scene_times:
            ts = scene_times.get(date_str)
            if ts is not None:
                return ts + dt.timedelta(hours=self.mtl_utc_offset)

        oh, om = map(int, self.overpass_time.split(':'))
        return dt.datetime.combine(d, dt.time(oh, om))

    def _interpolate_levels(self, when: dt.datetime):
        """
        Gauge levels at an exact timestamp, linearly interpolated from
        the loaded records (independently per station). Returns
        (levels_dict, matched_datetime) or (None, None) if no records lie
        within tide_match_days of `when`.
        """
        records = self._load_waterlevels()
        if not records:
            return None, None

        times = [r[0] for r in records]
        import bisect
        i = bisect.bisect_left(times, when)
        prev = records[i - 1] if i > 0 else None
        nxt  = records[i]     if i < len(records) else None

        max_gap = dt.timedelta(days=self.tide_match_days)
        if prev is None or (when - prev[0]) > max_gap:
            prev = None
        if nxt is None or (nxt[0] - when) > max_gap:
            nxt = None

        if prev is None and nxt is None:
            return None, None
        if prev is not None and nxt is not None and prev[0] == nxt[0]:
            nxt = None

        levels = {}
        stations = set((prev[1] if prev else {})) | set((nxt[1] if nxt else {}))
        for st in stations:
            v0 = prev[1].get(st) if prev else None
            v1 = nxt[1].get(st)  if nxt else None
            if v0 is not None and v1 is not None and prev is not None and nxt is not None:
                span = max((nxt[0] - prev[0]).total_seconds(), 1e-9)
                w = (when - prev[0]).total_seconds() / span
                levels[st] = float(v0 + (v1 - v0) * w)
            elif v0 is not None:
                levels[st] = float(v0)
            elif v1 is not None:
                levels[st] = float(v1)

        if not levels:
            return None, None
        return levels, when

    def _levels_for_scene(self, raster_path: str):
        """
        Water levels for a scene, interpolated at its exact local
        acquisition timestamp. Returns (levels, matched_datetime) or
        (None, None).
        """
        when = self._scene_datetime(raster_path)
        if when is None:
            return None, None
        return self._interpolate_levels(when)

    # ─────────────────────────────────────────────────────────────────────────
    # Georeferencing of gauge positions
    # ─────────────────────────────────────────────────────────────────────────

    def _get_geo(self, ref_path: str):
        """Return (geotransform, wkt) of a raster, or (None, None)."""
        try:
            from osgeo import gdal
            ds = gdal.Open(ref_path)
            return ds.GetGeoTransform(), ds.GetProjection()
        except (ImportError, AttributeError):
            return None, None

    def _station_pixel_coords(self, ref_path: str) -> dict:
        """
        Map station (lat, lon) to raster pixel (col, row).
        Uses GDAL/osr or pyproj to transform into the raster CRS; falls
        back to identity georeference (lon→col, lat→row) with a warning.
        """
        gt, wkt = self._get_geo(ref_path)
        if gt is None or gt[1] == 0 or gt[5] == 0:
            gt = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        transform = None
        if wkt:
            try:
                from osgeo import osr
                src = osr.SpatialReference()
                src.ImportFromEPSG(4326)
                if hasattr(src, 'SetAxisMappingStrategy'):
                    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst = osr.SpatialReference()
                dst.ImportFromWkt(wkt)
                if hasattr(dst, 'SetAxisMappingStrategy'):
                    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                transform = osr.CoordinateTransformation(src, dst).TransformPoint
            except (ImportError, AttributeError):
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(
                        'EPSG:4326', wkt, always_xy=True
                    )
                    transform = lambda lon, lat: transformer.transform(lon, lat)[:2]
                except Exception:
                    transform = None

        if transform is None:
            if not self._warned_identity_geo:
                self._warned_identity_geo = True
                warnings.warn(
                    "Could not determine/project the raster CRS "
                    "(no GDAL/pyproj or no CRS in the rasters). Assuming "
                    "identity georeference: gauge lon/lat are used directly "
                    "as raster column/row."
                )
            transform = lambda lon, lat: (lon, lat)

        coords = {}
        for name, (lat, lon) in self.stations.items():
            try:
                x, y = transform(lon, lat)
            except Exception:
                x, y = lon, lat
            if gt[1] == 0 or gt[5] == 0:
                warnings.warn(
                    f"Raster geotransform has zero pixel size "
                    f"({gt}); cannot map gauge coordinates to pixels. "
                    f"Check that the rasters carry a valid geotransform/CRS."
                )
                return {}
            col = (x - gt[0]) / gt[1]
            row = (y - gt[3]) / gt[5]
            coords[name] = (float(col), float(row))
        return coords

    # ─────────────────────────────────────────────────────────────────────────
    # Water-level surface (plane + IDW residuals)
    # ─────────────────────────────────────────────────────────────────────────

    def _level_surface(self, levels: dict, shape: tuple,
                       st_pix: dict) -> np.ndarray:
        """
        Interpolate gauge readings to a full water-level raster:
        first-order plane (downstream slope) + IDW (power 2) residuals.
        Returns float32 array or None if no usable stations.
        """
        names = [n for n in levels if n in st_pix]
        if not names:
            return None
        vals = np.array([levels[n] for n in names], dtype=np.float64)
        sx = np.array([st_pix[n][0] for n in names], dtype=np.float64)  # cols
        sy = np.array([st_pix[n][1] for n in names], dtype=np.float64)  # rows

        H, W = shape
        cols = np.arange(W, dtype=np.float64)
        rows = np.arange(H, dtype=np.float64)

        # ── plane trend ──
        coef = None
        if len(names) >= 3:
            A = np.column_stack([sx, sy, np.ones(len(names))])
            if np.linalg.matrix_rank(A) >= 3:
                coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
        if coef is not None:
            resid = vals - (sx * coef[0] + sy * coef[1] + coef[2])
            base = (coef[0] * cols[None, :] + coef[1] * rows[:, None]
                    + coef[2])
        else:
            resid = vals
            base = np.zeros(shape, dtype=np.float64)

        # ── IDW residuals, row-blocked to bound memory ──
        out = np.array(base, dtype=np.float64)
        block = 256
        for r0 in range(0, H, block):
            r1 = min(H, r0 + block)
            yr = rows[r0:r1][:, None]
            num = np.zeros((r1 - r0, W), dtype=np.float64)
            den = np.zeros((r1 - r0, W), dtype=np.float64)
            for k in range(len(names)):
                d2 = (cols[None, :] - sx[k]) ** 2 + (yr - sy[k]) ** 2
                w = 1.0 / (d2 + 1e-9)
                num += w * resid[k]
                den += w
            out[r0:r1] += num / den

        return out.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Tidal monotonic test
    # ─────────────────────────────────────────────────────────────────────────

    def _tide_extremes(self, scenes: list, shape: tuple, st_pix: dict):
        """
        Per-pixel water-level extremes over the scenes that have a
        water-level surface. `scenes` is a list of (raster_path, bin_array)
        pairs; extra pairs from adjacent years may be included for pooled
        tide evidence.

        Returns
        -------
        (h_wet_min, h_dry_max, h_max, h_min, n_tide_scenes, matched_dates)
            h_wet_min : float64 array, +inf where no wet tide scene
            h_dry_max : float64 array, -inf where no dry tide scene
            h_max/h_min: per-pixel highest/lowest water level observed
                         (over valid pixels), for the range guard
        """
        h_wet_min = np.full(shape, np.inf, dtype=np.float64)
        h_dry_max = np.full(shape, -np.inf, dtype=np.float64)
        h_max = np.full(shape, -np.inf, dtype=np.float64)
        h_min = np.full(shape, np.inf, dtype=np.float64)
        matched = {}

        n_tide = 0
        for fpath, arr in scenes:
            date_str = self._date_from_filename(os.path.basename(fpath))
            if not date_str:
                continue
            if self.dry_season_only:
                month = int(date_str[5:7])
                if month not in DRY_MONTHS:
                    continue

            if date_str in self._surface_cache:
                surface = self._surface_cache[date_str]
            else:
                levels, ts = self._levels_for_scene(fpath)
                surface = None
                if levels:
                    surface = self._level_surface(levels, shape, st_pix)
                    if surface is not None:
                        matched[date_str] = ts
                self._surface_cache[date_str] = surface

            if surface is None:
                continue

            valid = (arr != self.nodata_val)
            wet = (arr == 1) & valid
            dry = (arr == 0) & valid

            np.minimum(h_wet_min, surface, out=h_wet_min, where=wet)
            np.maximum(h_dry_max, surface, out=h_dry_max, where=dry)
            np.maximum(h_max, surface, out=h_max, where=valid)
            np.minimum(h_min, surface, out=h_min, where=valid)
            n_tide += 1

        return h_wet_min, h_dry_max, h_max, h_min, n_tide, matched

    # ─────────────────────────────────────────────────────────────────────────
    # Channel connectivity
    # ─────────────────────────────────────────────────────────────────────────

    def _channel_mask(self, river_mask: np.ndarray) -> np.ndarray:
        """
        Clean the always-water mask to channel-sized connected components
        (8-connectivity, size >= min_channel_px). Returns the channel mask.
        """
        if self.min_channel_px <= 1:
            return river_mask.copy()

        labels, n = ndimage.label(river_mask, structure=np.ones((3, 3), int))
        if n == 0:
            return np.zeros_like(river_mask)
        sizes = np.bincount(labels.ravel())
        keep = np.zeros(len(sizes), dtype=bool)
        keep[1:] = sizes[1:] >= self.min_channel_px
        return keep[labels] & river_mask

    # ─────────────────────────────────────────────────────────────────────────
    # Core classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_year(self, files: list):
        """
        Classify one year:

            river      : always water AND SI-high (Otsu) in every wet scene
                         AND in a channel-sized connected component
            intertidal : switches AND SI rules AND h_wet_min > h_dry_max
                         AND within buffer_m of the channelised river
            land       : everything else
            nodata     : valid_count < min_scenes

        Returns (result, valid_count, water_count, channel_mask,
                 elevation, tide_info)
        """
        if not files:
            raise ValueError("No files provided for classification.")

        first, _ = self._read_tif(files[0])
        shape = first.shape

        # Read all binary masks once for this year
        bin_cache = {}
        water_count = np.zeros(shape, dtype=np.uint16)
        valid_count = np.zeros(shape, dtype=np.uint16)
        for fpath in files:
            arr, _ = self._read_tif(fpath)
            if arr.shape != shape:
                warnings.warn(
                    f"Shape mismatch in {os.path.basename(fpath)}: "
                    f"{arr.shape} vs expected {shape}. Skipping."
                )
                continue
            bin_cache[fpath] = arr
            valid = (arr != self.nodata_val)
            water = (arr == 1) & valid
            valid_count += valid.astype(np.uint16)
            water_count += water.astype(np.uint16)

        used_files = [f for f in files if f in bin_cache]

        result = np.full(shape, NODATA, dtype=np.uint8)
        sufficient = valid_count >= self.min_scenes

        land_mask  = sufficient & (water_count == 0)
        river_mask = sufficient & (water_count == valid_count) & (water_count > 0)
        inter_mask = sufficient & (water_count > 0) & (water_count < valid_count)

        # ── SI validation (per-image Otsu) — same rules as parent ──────────
        if self.si_dir:
            reject_river, reject_inter = self._si_validation_masks(
                used_files, shape, water_count, valid_count
            )
            river_mask = river_mask & ~reject_river
            inter_mask = inter_mask & ~reject_inter
            land_mask  = land_mask | reject_river | reject_inter

        # ── Tidal monotonic test (skipped entirely when use_tide=False) ───
        n_tide = 0
        matched = {}
        h_wet_min = None
        h_dry_max = None
        if self.use_tide:
            st_pix = self._station_pixel_coords(files[0])

            # Scenes of the target year (switching evidence) — plus scenes of
            # adjacent years for POOLED tide evidence when tide_pool_years > 0
            scenes = [(f, bin_cache[f]) for f in used_files]
            if self.tide_pool_years > 0:
                year = int(self._date_from_filename(os.path.basename(files[0]))[:4])
                by_year = self._collect_by_year()
                for dy in range(1, self.tide_pool_years + 1):
                    for yy in (year - dy, year + dy):
                        for f in by_year.get(yy, []):
                            if f in bin_cache:
                                continue
                            arr2, _ = self._read_tif(f)
                            if arr2.shape == shape:
                                scenes.append((f, arr2))

            h_wet_min, h_dry_max, h_max, h_min, n_tide, matched = \
                self._tide_extremes(scenes, shape, st_pix)

            if n_tide > 0:
                has_tide = np.isfinite(h_wet_min) & np.isfinite(h_dry_max)
                # Range guard (patchy-data protection): a pixel observed only
                # over a NARROW water-level band carries no real tidal
                # information — its wet/dry labels there are dominated by
                # noise, so it keeps the count-based class.
                informative = (h_max - h_min) >= self.range_min_m
                monotonic = has_tide & (h_wet_min > h_dry_max + self.monotonic_tol_m)
                # Pixels WITH informative tide evidence must be monotonic;
                # pixels without informative evidence keep the candidacy.
                judged = has_tide & informative
                reject_switch = (inter_mask & judged & ~monotonic)
                inter_mask = inter_mask & ~reject_switch
                land_mask  = land_mask | reject_switch
                n_uninform = int((inter_mask & has_tide & ~informative).sum())
                print(f"   Tidal test      : {n_tide} tide scene(s) "
                      f"({', '.join(sorted(matched))}); "
                      f"{int(reject_switch.sum()):,} non-monotonic px → land; "
                      f"{n_uninform:,} px kept (observed range < {self.range_min_m:g} m)")
            else:
                print("   Tidal test      : no water-level data for this year "
                      "— skipped")

        else:
            print("   Tidal test      : disabled (use_tide=False)")

        # ── Channel connectivity + buffer ────────────────────────────────
        channel = self._channel_mask(river_mask)
        dropped = river_mask & ~channel
        river_mask = channel
        land_mask = land_mask | dropped
        print(f"   Channel network : {int(channel.sum()):,} px "
              f"({int(dropped.sum()):,} small always-water px → land)")

        if channel.any():
            dist_m = ndimage.distance_transform_edt(~channel) \
                     * self.pixel_size_m
            far = inter_mask & (dist_m > self.buffer_m)
            inter_mask = inter_mask & ~far
            land_mask = land_mask | far
            print(f"   Channel buffer  : {int(far.sum()):,} candidate px "
                  f"farther than {self.buffer_m:g} m → land")
        else:
            warnings.warn(
                "No channel-sized water component found; the buffer "
                "constraint removed ALL intertidal candidates."
            )
            land_mask = land_mask | inter_mask
            inter_mask = np.zeros_like(inter_mask)

        # ── Elevation estimate for intertidal pixels (gauge datum) ────────
        elevation = np.full(shape, np.nan, dtype=np.float32)
        if n_tide > 0 and inter_mask.any():
            z = (h_dry_max + h_wet_min) / 2.0
            elevation[inter_mask] = z[inter_mask].astype(np.float32)

        result[land_mask]  = LAND
        result[river_mask] = RIVER
        result[inter_mask] = INTERTIDAL

        tide_info = {
            'n_tide_scenes': n_tide,
            'matched_dates': matched,
            'h_wet_min': h_wet_min,
            'h_dry_max': h_dry_max,
        }
        return result, valid_count, water_count, channel, elevation, tide_info

    # ─────────────────────────────────────────────────────────────────────────
    # Extra GeoTIFF writer (float32, NaN nodata)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_tif_float(filepath: str, data: np.ndarray,
                         reference_path: str):
        """Write a float32 GeoTIFF with NaN nodata."""
        try:
            from osgeo import gdal, gdalconst
            ref_ds = gdal.Open(reference_path)
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                filepath, ref_ds.RasterXSize, ref_ds.RasterYSize, 1,
                gdalconst.GDT_Float32, options=['COMPRESS=LZW']
            )
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            band = out_ds.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))
            band.SetNoDataValue(float('nan'))
            out_ds.FlushCache()
            out_ds = ref_ds = None
            return
        except (ImportError, AttributeError):
            pass
        import tifffile
        tifffile.imwrite(filepath, data.astype(np.float32),
                         photometric='minisblack')

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_year(self, year: int) -> dict:
        """
        Process a single year. Writes:
            intertidal_YYYY.tif           (0/1/2/255)
            channel_YYYY.tif              (1 = channelised river, 0 else)
            intertidal_elevation_YYYY.tif (float, gauge datum, NaN elsewhere)
        Returns stats dict.
        """
        by_year = self._collect_by_year()

        if year not in by_year:
            print(f"⚠️  No river files found for year {year}")
            return {}

        files   = by_year[year]
        out_tif = os.path.join(self.output_dir, f'intertidal_{year}.tif')

        if self.skip_existing and os.path.exists(out_tif):
            print(f"   ⏭️  {year}: output exists — skipping")
            return {}

        print(f"\n── {year} ──  ({len(files)} scenes)")

        self._surface_cache = {}
        (classified, valid_count, water_count,
         channel, elevation, tide_info) = self._classify_year(files)

        stats = self._compute_stats(classified, year, len(files))
        stats['px_channel'] = int(channel.sum())
        stats['km2_channel'] = round(
            stats['px_channel'] * self.pixel_size_m ** 2 / 1e6, 3
        )

        self._write_tif(out_tif, classified, files[0])
        self._write_tif(
            os.path.join(self.output_dir, f'channel_{year}.tif'),
            channel.astype(np.uint8), files[0]
        )
        if np.isfinite(elevation).any():
            self._write_tif_float(
                os.path.join(self.output_dir, f'intertidal_elevation_{year}.tif'),
                elevation, files[0]
            )

        print(f"   Land       : {stats['px_land']:>8,} px  ({stats['km2_land']:.2f} km²)")
        print(f"   River      : {stats['px_river']:>8,} px  ({stats['km2_river']:.2f} km²)  "
              f"[channelised]")
        print(f"   Intertidal : {stats['px_intertidal']:>8,} px  ({stats['km2_intertidal']:.2f} km²)  "
              f"[{stats['pct_intertidal']:.1f}% of valid]")
        print(f"   NoData     : {stats['px_nodata']:>8,} px")
        print(f"   ✅ Saved → {out_tif}")

        return stats

    def _save_csv(self, all_stats: list) -> str:
        """Save summary stats (incl. channel) to CSV."""
        csv_path = os.path.join(self.output_dir, 'intertidal_summary.csv')
        fieldnames = [
            'year', 'n_scenes',
            'px_land', 'px_river', 'px_channel', 'px_intertidal',
            'px_nodata', 'px_total',
            'km2_land', 'km2_river', 'km2_channel', 'km2_intertidal',
            'pct_intertidal',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"📊 Summary CSV → {csv_path}")
        return csv_path