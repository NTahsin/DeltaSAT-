"""
river_surface_profiler.py  (v2 — phase-aware spatial interpolation)
====================================================================
Generates spatially distributed river water-surface elevation (WSE) grids
from land-river GeoTIFFs and reconstructed tidal-gauge time series.

Core change from v1
-------------------
The IDW (Euclidean-distance) interpolation has been replaced by a
*phase-aware along-channel* scheme that decomposes each gauge signal into
tidal harmonic components and a non-tidal residual (NTR):

    h(x, t) = Z₀(x) + Σ_k  A_k(x) · cos(ω_k·t − φ_k(x))  +  NTR(x, t)

    A_k(x), φ_k(x), Z₀(x)  — linearly interpolated in 1-D along-channel
                               distance between the two bracketing gauges.
    NTR(x, t)               — IDW-interpolated in 2-D (slowly varying;
                               Euclidean geometry is adequate).

The tidal wave travels upstream with a measurable phase lag; interpolating
amplitude and phase *separately* along the channel axis recovers the
propagating wave rather than averaging instantaneous gauge readings.

Along-channel coordinates
--------------------------
Station coordinates are projected onto the principal axis of the gauge
network (first PCA eigenvector of station XY positions).  River pixels
are projected the same way.  For most delta distributary systems where
gauges are placed along a main channel this is an accurate proxy.
If your gauges span multiple distributaries, supply explicit
``along_ch_dist_m`` in ``station_coords`` (see docstring).

Constituents fitted
--------------------
M2, S2, N2, K1, O1 (Rayleigh-filtered per station's record length).
NTR carries the monsoon river-discharge signal — no harmonic assumption.

Dependencies
------------
    pip install numpy pandas rasterio scipy tqdm pyproj

Author
------
    GBM Intertidal Morphodynamics Project (JGR Earth Surface)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import interp1d

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

LAND_VALUE   =  999.0
RIVER_MARKER = -999.0
EPOCH        = pd.Timestamp("1990-01-01 00:00:00")   # phase reference epoch

# Tidal constituent frequencies (cycles per hour)
TIDAL_FREQS_CPH: Dict[str, float] = {
    "M2": 1 / 12.42060,
    "S2": 1 / 12.00000,
    "N2": 1 / 12.65835,
    "K1": 1 / 23.93447,
    "O1": 1 / 25.81934,
}
DEFAULT_CONSTITUENTS = ["M2", "S2", "K1", "O1"]
RAYLEIGH_FACTOR = 1.0    # minimum resolved frequency separation × record length


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TidalComponent:
    name:       str
    amplitude:  float   # metres
    phase_rad:  float   # radians at EPOCH
    omega:      float   # rad / hour
    z0:         float   # mean level (m); only from the leading DC term


@dataclass
class StationTidalModel:
    """Complete tidal harmonic model for one gauge station."""
    station_id:  str
    z0:          float                         # mean water level (m)
    components:  Dict[str, TidalComponent]     # constituent name → params
    along_ch:    float = 0.0                   # along-channel distance (m)

    def tidal_h(self, t_h: float) -> float:
        """Predicted tidal level (m) at t_h hours since EPOCH."""
        return self.z0 + sum(
            c.amplitude * np.cos(c.omega * t_h - c.phase_rad)
            for c in self.components.values()
        )

    def tidal_h_vec(self, t_h: np.ndarray) -> np.ndarray:
        """Vectorised tidal prediction over a time array."""
        h = np.full_like(t_h, self.z0, dtype=float)
        for c in self.components.values():
            h += c.amplitude * np.cos(c.omega * t_h - c.phase_rad)
        return h


@dataclass
class StationCoords:
    station_id: str
    x: float
    y: float
    along_ch: Optional[float] = None   # user-supplied along-channel dist (m)


# ─────────────────────────────────────────────────────────────────────────────
# Phase estimator
# ─────────────────────────────────────────────────────────────────────────────

class PhaseEstimator:
    """
    Fits tidal harmonic components to a reconstructed water-level series
    using least-squares regression.  Phase is referenced to EPOCH so that
    values are comparable across stations.

    Parameters
    ----------
    constituents : list of constituent names to fit (default: M2 S2 K1 O1).
    rayleigh     : Rayleigh criterion factor (default 1.0).
    """

    def __init__(
        self,
        constituents: Optional[List[str]] = None,
        rayleigh: float = RAYLEIGH_FACTOR,
    ):
        self.constituents = constituents or DEFAULT_CONSTITUENTS
        self.rayleigh = rayleigh

    def fit(self, station_id: str, series: pd.Series) -> StationTidalModel:
        """
        Parameters
        ----------
        station_id : label for logging / output.
        series     : pd.Series with DatetimeIndex (UTC-naive) and WSE in metres.
                     May be the full multi-year reconstructed series.

        Returns
        -------
        StationTidalModel
        """
        valid = series.dropna()
        if len(valid) < 50:
            raise ValueError(
                f"Station '{station_id}': only {len(valid)} valid points; "
                "cannot estimate harmonic components."
            )

        # Hours since EPOCH  (consistent reference across stations)
        t_h = (valid.index.tz_localize(None) - EPOCH).total_seconds().values / 3600.0
        h   = valid.values.astype(np.float64)
        record_len_h = t_h[-1] - t_h[0]

        # Rayleigh-filtered constituent selection
        selected: List[Tuple[str, float]] = []    # (name, freq_cph)
        for name in self.constituents:
            if name not in TIDAL_FREQS_CPH:
                continue
            f = TIDAL_FREQS_CPH[name]
            resolvable = all(
                abs(f - TIDAL_FREQS_CPH[n]) * record_len_h >= self.rayleigh
                for n, _ in selected
            )
            if resolvable or not selected:
                selected.append((name, f))

        if not selected:
            raise ValueError(
                f"Station '{station_id}': no constituents pass Rayleigh "
                f"criterion for record length {record_len_h:.1f} h."
            )

        # Design matrix  [1, cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
        cols = [np.ones(len(t_h))]
        for _, f in selected:
            w = 2.0 * np.pi * f
            cols.extend([np.cos(w * t_h), np.sin(w * t_h)])
        A_mat = np.column_stack(cols)

        coeffs, *_ = np.linalg.lstsq(A_mat, h, rcond=None)

        z0 = float(coeffs[0])
        components: Dict[str, TidalComponent] = {}
        for i, (name, f) in enumerate(selected):
            a_cos = coeffs[1 + 2 * i]
            a_sin = coeffs[2 + 2 * i]
            amp   = float(np.sqrt(a_cos**2 + a_sin**2))
            phase = float(np.arctan2(a_sin, a_cos))   # h = amp·cos(ωt - phase)
            omega = float(2.0 * np.pi * f)
            components[name] = TidalComponent(
                name=name, amplitude=amp,
                phase_rad=phase, omega=omega, z0=z0,
            )

        log.debug(
            "  %s: Z₀=%.3f m | %s",
            station_id, z0,
            " | ".join(
                f"{n} A={c.amplitude:.3f}m φ={np.degrees(c.phase_rad):.1f}°"
                for n, c in components.items()
            ),
        )
        return StationTidalModel(
            station_id=station_id, z0=z0, components=components
        )


# ─────────────────────────────────────────────────────────────────────────────
# Along-channel axis
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAxis:
    """
    Computes 1-D along-channel coordinates via PCA of station positions,
    or accepts user-supplied distances.

    The axis direction is the first principal component of the station
    coordinate matrix (the direction of maximum variance, i.e. the main
    channel axis).  Positive direction points upstream (increasing distance
    from the tidal inlet / most downstream gauge).

    Parameters
    ----------
    station_xy  : (n_stations, 2) array of projected station coordinates.
    along_ch_m  : optional (n_stations,) array of user-supplied along-channel
                  distances (metres).  If given, PCA is skipped.
    """

    def __init__(
        self,
        station_xy: np.ndarray,
        along_ch_m: Optional[np.ndarray] = None,
    ):
        self.centroid = station_xy.mean(axis=0)

        if along_ch_m is not None:
            # User-supplied — coords are already absolute distances
            self._axis_vec      = None
            self._raw_ch_offset = 0.0
            self._station_ch    = np.asarray(along_ch_m, dtype=float)
        else:
            # PCA: first eigenvector of centred station positions
            centred = station_xy - self.centroid
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            self._axis_vec      = Vt[0]       # (2,) unit vector along channel
            raw                 = centred @ self._axis_vec
            # Store the raw minimum so pixel projections use the *same* offset.
            # Without this, batches of pixels that don't span the full station
            # range would compute a wrong local offset.
            self._raw_ch_offset = float(raw.min())
            self._station_ch    = raw - self._raw_ch_offset

        log.info(
            "Channel axis: station range %.0f – %.0f m (%.1f km span)",
            self._station_ch.min(),
            self._station_ch.max(),
            (self._station_ch.max() - self._station_ch.min()) / 1000,
        )

    @property
    def station_along_ch(self) -> np.ndarray:
        return self._station_ch

    def pixel_along_ch(self, pixel_xy: np.ndarray) -> np.ndarray:
        """
        Project pixel coordinates onto the channel axis.

        Parameters
        ----------
        pixel_xy : (n_pixels, 2)

        Returns
        -------
        (n_pixels,) along-channel distance (metres), same scale as stations.
        The offset is anchored to the station reference frame set at init time
        — independent of the spatial range of the pixel batch supplied.
        """
        if self._axis_vec is None:
            raise ValueError(
                "Cannot project pixels: channel axis was built from "
                "user-supplied distances, not station coordinates. "
                "Supply pixel along-channel distances explicitly."
            )
        centred = pixel_xy - self.centroid          # (n, 2)
        return centred @ self._axis_vec - self._raw_ch_offset   # (n,)


# ─────────────────────────────────────────────────────────────────────────────
# Phase-aware spatial interpolator
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareSpatialInterp:
    """
    Spatially distributes WSE using phase-aware along-channel interpolation.

    Algorithm (per pixel p, at image time t)
    -----------------------------------------
    1. Sort stations by along-channel distance (x).
    2. Find the two bracketing stations i (upstream), j (downstream).
    3. α = (x_p − x_i) / (x_j − x_i)   ← fractional along-channel position
    4. For each resolved tidal constituent k:
           A_k(p)  = A_k_i + α (A_k_j − A_k_i)
           Δφ_k   = wrap(φ_k_j − φ_k_i) to (−π, π)   ← handles 2π branch
           φ_k(p) = φ_k_i + α · Δφ_k
    5. Z₀(p) = Z₀_i + α (Z₀_j − Z₀_i)
    6. h_tidal(p, t) = Z₀(p) + Σ_k A_k(p) · cos(ω_k · t_h − φ_k(p))
    7. NTR_i(t)  = h_obs_i(t) − h_tidal_i(t)   (at each gauge)
    8. NTR(p, t) = IDW of {NTR_i(t)} across all stations
    9. h(p, t)   = h_tidal(p, t) + NTR(p, t)

    Pixels outside the station range (extrapolation) use the tidal model of
    the nearest edge station plus IDW NTR.

    Parameters
    ----------
    tidal_models   : list of StationTidalModel (one per station).
    channel_axis   : ChannelAxis instance.
    station_xy     : (n_stations, 2) projected coordinates (for IDW NTR).
    idw_power      : IDW exponent for NTR interpolation (default 2).
    """

    def __init__(
        self,
        tidal_models: List[StationTidalModel],
        channel_axis: ChannelAxis,
        station_xy:   np.ndarray,
        idw_power:    float = 2.0,
    ):
        self.idw_power = idw_power
        self._station_xy = station_xy

        # Sort everything by along-channel distance (downstream → upstream)
        s_ch = channel_axis.station_along_ch
        order = np.argsort(s_ch)
        self._sort_order = order                          # stored for obs_wl re-ordering
        self._s_ch    = s_ch[order]
        self._models  = [tidal_models[i] for i in order]
        self._s_xy    = station_xy[order]

        # Precompute per-station per-constituent arrays for vectorised ops
        self._all_names = []
        for m in self._models:
            for n in m.components:
                if n not in self._all_names:
                    self._all_names.append(n)

        self._channel_axis = channel_axis

    # ------------------------------------------------------------------

    def interpolate(
        self,
        t_dt:           datetime,
        pixel_xy:       np.ndarray,   # (n_pixels, 2)
        obs_wl:         np.ndarray,   # (n_stations,) observed WSE at t_dt
    ) -> np.ndarray:
        """
        Returns (n_pixels,) WSE array for a given datetime.

        Parameters
        ----------
        t_dt    : image acquisition datetime (timezone-naive).
        pixel_xy: projected (x, y) coordinates of river pixels.
        obs_wl  : observed / reconstructed WSE at each station at t_dt,
                  in the same order as tidal_models passed to __init__
                  (before internal sorting — this method handles reordering).
        """
        # Hours since epoch for this timestamp
        t_h = float((pd.Timestamp(t_dt) - EPOCH).total_seconds() / 3600.0)

        # Project pixels onto channel axis
        pix_ch = self._channel_axis.pixel_along_ch(pixel_xy)   # (n_pixels,)

        n_pix = len(pix_ch)
        n_sta = len(self._models)

        # Reorder obs_wl to match the sorted station order (stored at init time)
        obs_sorted = obs_wl[self._sort_order]

        # ── Step 1: Find bracketing stations ─────────────────────────────
        # bracket_lo[p] = index of the downstream-side station
        lo_idx = np.searchsorted(self._s_ch, pix_ch, side="right") - 1
        lo_idx = np.clip(lo_idx, 0, n_sta - 2)
        hi_idx = lo_idx + 1

        # Fractional along-channel position
        ch_span = self._s_ch[hi_idx] - self._s_ch[lo_idx]
        ch_span = np.where(ch_span < 1.0, 1.0, ch_span)   # avoid /0 (m)
        alpha   = np.clip((pix_ch - self._s_ch[lo_idx]) / ch_span, 0.0, 1.0)

        # ── Step 2: Tidal reconstruction at each pixel ────────────────────
        h_tidal_pix = np.zeros(n_pix)

        # Interpolate Z₀ along channel
        z0_lo = np.array([self._models[i].z0 for i in lo_idx])
        z0_hi = np.array([self._models[i].z0 for i in hi_idx])
        h_tidal_pix += z0_lo + alpha * (z0_hi - z0_lo)

        for name in self._all_names:
            amp_lo  = np.array([
                self._models[i].components[name].amplitude
                if name in self._models[i].components else 0.0
                for i in lo_idx
            ])
            amp_hi  = np.array([
                self._models[i].components[name].amplitude
                if name in self._models[i].components else 0.0
                for i in hi_idx
            ])
            phi_lo  = np.array([
                self._models[i].components[name].phase_rad
                if name in self._models[i].components else 0.0
                for i in lo_idx
            ])
            phi_hi  = np.array([
                self._models[i].components[name].phase_rad
                if name in self._models[i].components else 0.0
                for i in hi_idx
            ])
            omega   = TIDAL_FREQS_CPH.get(name, 0.0) * 2.0 * np.pi

            # Phase difference, wrapped to (−π, π) to avoid 2π branch cuts
            d_phi = phi_hi - phi_lo
            d_phi = ((d_phi + np.pi) % (2.0 * np.pi)) - np.pi

            amp_p = amp_lo + alpha * (amp_hi - amp_lo)
            phi_p = phi_lo + alpha * d_phi

            h_tidal_pix += amp_p * np.cos(omega * t_h - phi_p)

        # ── Step 3: NTR at each station ───────────────────────────────────
        ntr_sta = np.array([
            float(obs_sorted[i]) - self._models[i].tidal_h(t_h)
            for i in range(n_sta)
        ])

        # ── Step 4: IDW of NTR onto pixels ────────────────────────────────
        ntr_pix = _idw(ntr_sta, self._s_xy, pixel_xy, power=self.idw_power)

        return (h_tidal_pix + ntr_pix).astype(np.float32)

    def station_ntr(self, t_h: float, obs_sorted: np.ndarray) -> np.ndarray:
        """Return NTR array (n_stations,) at a given time, sorted along-channel."""
        return np.array([
            float(obs_sorted[i]) - self._models[i].tidal_h(t_h)
            for i in range(len(self._models))
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with visualizer)
# ─────────────────────────────────────────────────────────────────────────────

_DATE_PATTERNS: List[Tuple[str, str]] = [
    (r"(\d{4}-\d{2}-\d{2})_(\d{6})", "%Y-%m-%d %H%M%S"),
    (r"(\d{4}-\d{2}-\d{2})",          "%Y-%m-%d"),
    (r"(\d{8})_(\d{6})",              "%Y%m%d %H%M%S"),
    (r"(\d{8})",                       "%Y%m%d"),
]


def parse_datetime_from_name(fname: str) -> Optional[datetime]:
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, fname)
        if m:
            raw = " ".join(m.groups())
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
    return None


def _idw(
    values: np.ndarray,
    src_xy: np.ndarray,
    tgt_xy: np.ndarray,
    power:  float = 2.0,
    min_dist: float = 1.0,
) -> np.ndarray:
    """Inverse-distance weighting (vectorised, (n_src, n_tgt))."""
    diff  = src_xy[:, np.newaxis, :] - tgt_xy[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    dists_clamped = np.where(dists < min_dist, min_dist, dists)
    weights = 1.0 / dists_clamped ** power
    wsum    = weights.sum(axis=0)
    result  = (weights * values[:, np.newaxis]).sum(axis=0) / wsum
    # Exact coincidence: use station value directly
    exact = dists.min(axis=0) < min_dist
    if exact.any():
        sta_idx = dists.argmin(axis=0)
        result[exact] = values[sta_idx[exact]]
    return result


def pixel_centres(
    row_idx: np.ndarray, col_idx: np.ndarray, transform
) -> np.ndarray:
    xs, ys = rasterio.transform.xy(transform, row_idx, col_idx)
    return np.column_stack([xs, ys]).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Station utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_station_list(
    station_coords: Union[pd.DataFrame, Dict],
    wl_series:      Dict[str, pd.Series],
    raster_crs_wkt: str,
    crs_wl:         Optional[str],
) -> List[StationCoords]:
    """Parse station_coords, re-project if needed, return StationCoords list."""
    if isinstance(station_coords, dict):
        raw = {sid: (v[0], v[1]) for sid, v in station_coords.items()}
    else:
        df = station_coords
        id_col = next(
            (c for c in ["station_id","Station","ID","id"] if c in df.columns),
            df.columns[0],
        )
        x_col  = next(
            (c for c in ["along_ch_dist_m"] if c in df.columns), None
        )
        xy_x   = next(
            (c for c in ["x","X","Easting","Longitude","lon"] if c in df.columns),
            df.columns[1],
        )
        xy_y   = next(
            (c for c in ["y","Y","Northing","Latitude","lat"] if c in df.columns),
            df.columns[2],
        )
        raw = {
            str(row[id_col]): (float(row[xy_x]), float(row[xy_y]))
            for _, row in df.iterrows()
        }
        along_ch = (
            {str(row[id_col]): float(row[x_col]) for _, row in df.iterrows()}
            if x_col else {}
        )

    raw = {sid: xy for sid, xy in raw.items() if sid in wl_series}
    if not raw:
        raise ValueError(
            "No station_coords entries match any key in wl_series. "
            "Check station IDs are consistent."
        )

    needs_reproject = (
        crs_wl is not None
        and crs_wl.upper() not in ("NONE", "")
        and crs_wl != raster_crs_wkt
    )
    if needs_reproject:
        if not _HAS_PYPROJ:
            raise ImportError("pyproj required for coordinate reprojection.")
        tr = Transformer.from_crs(crs_wl, raster_crs_wkt, always_xy=True)
        raw = {sid: tr.transform(x, y) for sid, (x, y) in raw.items()}

    return [
        StationCoords(
            station_id=sid, x=xy[0], y=xy[1],
            along_ch=along_ch.get(sid) if "along_ch" in dir() else None,
        )
        for sid, xy in raw.items()
    ]


class _WLInterpolator:
    """Linear temporal interpolator for a single station's reconstructed series."""

    def __init__(self, station_id: str, series: pd.Series):
        self.station_id = station_id
        idx = series.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        valid = series.dropna()
        if len(valid) < 2:
            raise ValueError(
                f"Station '{station_id}' has < 2 non-NaN values."
            )
        t = (valid.index.tz_localize(None) - EPOCH).total_seconds().values
        v = valid.values.astype(np.float64)
        self._t_min, self._t_max = t.min(), t.max()
        self._fn = interp1d(t, v, kind="linear",
                            bounds_error=False, fill_value=(v[0], v[-1]))

    def __call__(self, dt: datetime) -> float:
        t_s = (pd.Timestamp(dt) - EPOCH).total_seconds()
        return float(self._fn(t_s))


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class RiverSurfaceProfiler:
    """
    Generates per-image water-surface elevation GeoTIFFs using
    phase-aware along-channel tidal interpolation.

    Parameters
    ----------
    geotiff_dir    : directory of land-river GeoTIFFs
                     (encoding: 999=land, -999=river, NaN=NoData).
    wl_series      : {station_id: pd.Series} or wide pd.DataFrame.
                     Reconstructed time series from water_level_processor.py.
    station_coords : pd.DataFrame or dict of station positions.
                     Optional column ``along_ch_dist_m`` for explicit
                     along-channel distances (metres from tidal inlet).
    output_dir     : directory for output WSE GeoTIFFs.
    idw_power      : IDW exponent for NTR interpolation (default 2).
    crs_wl         : EPSG of station_coords CRS (default EPSG:4326).
    constituents   : tidal constituents to resolve (default M2 S2 K1 O1).
    glob_pattern   : input file glob (default "*.tif").
    date_parser    : custom filename → datetime callable.
    overwrite      : overwrite existing output files.
    """

    def __init__(
        self,
        geotiff_dir:    Union[str, Path],
        wl_series:      Union[Dict[str, pd.Series], pd.DataFrame],
        station_coords: Union[pd.DataFrame, Dict],
        output_dir:     Union[str, Path],
        *,
        idw_power:      float = 2.0,
        crs_wl:         Optional[str] = "EPSG:4326",
        constituents:   Optional[List[str]] = None,
        glob_pattern:   str = "*.tif",
        date_parser:    Optional[Callable] = None,
        overwrite:      bool = False,
    ):
        self.geotiff_dir  = Path(geotiff_dir)
        self.output_dir   = Path(output_dir)
        self.idw_power    = idw_power
        self.crs_wl       = crs_wl
        self.glob_pattern = glob_pattern
        self.date_parser  = date_parser or parse_datetime_from_name
        self.overwrite    = overwrite

        # Normalise wl_series → dict
        if isinstance(wl_series, pd.DataFrame):
            self._wl_series = {
                str(col): wl_series[col] for col in wl_series.columns
            }
        else:
            self._wl_series = {str(k): v for k, v in wl_series.items()}

        # Temporal interpolators (one per station)
        self._wl_interp: Dict[str, _WLInterpolator] = {}
        for sid, s in self._wl_series.items():
            try:
                self._wl_interp[sid] = _WLInterpolator(sid, s)
            except ValueError as e:
                log.warning("Skip station '%s': %s", sid, e)

        if not self._wl_interp:
            raise ValueError("No valid station time series.")

        # Phase estimator — fit tidal models to full reconstructed series
        estimator = PhaseEstimator(constituents=constituents or DEFAULT_CONSTITUENTS)
        self._tidal_models: Dict[str, StationTidalModel] = {}
        for sid, s in self._wl_series.items():
            if sid not in self._wl_interp:
                continue
            try:
                self._tidal_models[sid] = estimator.fit(sid, s)
                log.info(
                    "Station '%s': fitted %d constituents (Z₀=%.2f m)",
                    sid,
                    len(self._tidal_models[sid].components),
                    self._tidal_models[sid].z0,
                )
            except ValueError as e:
                log.warning("Phase estimation failed for '%s': %s. "
                            "Station will use NTR only.", sid, e)

        # Lazy-initialised after first GeoTIFF CRS is known
        self._station_coords_raw = station_coords
        self._stations:  Optional[List[StationCoords]] = None
        self._station_xy: Optional[np.ndarray] = None
        self._interp_engine: Optional[PhaseAwareSpatialInterp] = None

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation (lazy — needs raster CRS)
    # ──────────────────────────────────────────────────────────────────────

    def _init_spatial(self, crs_wkt: str) -> None:
        if self._stations is not None:
            return

        self._stations = build_station_list(
            self._station_coords_raw,
            self._wl_interp,
            crs_wkt,
            self.crs_wl,
        )
        self._station_xy = np.array(
            [(s.x, s.y) for s in self._stations], dtype=np.float64
        )
        along_ch = np.array(
            [s.along_ch for s in self._stations], dtype=np.float64
        )
        user_along = (
            along_ch
            if not np.any(np.isnan(along_ch))
            else None
        )

        axis = ChannelAxis(self._station_xy, along_ch_m=user_along)

        # Assign along-channel distances to tidal models
        models_ordered = []
        for s in self._stations:
            if s.station_id in self._tidal_models:
                m = self._tidal_models[s.station_id]
                m.along_ch = float(
                    axis.station_along_ch[
                        [st.station_id for st in self._stations].index(s.station_id)
                    ]
                )
                models_ordered.append(m)

        if not models_ordered:
            raise RuntimeError("No tidal models available for any station.")

        self._interp_engine = PhaseAwareSpatialInterp(
            tidal_models=models_ordered,
            channel_axis=axis,
            station_xy=self._station_xy,
            idw_power=self.idw_power,
        )
        log.info(
            "Spatial engine ready: %d stations with tidal models, "
            "channel span %.1f km",
            len(models_ordered),
            (axis.station_along_ch.max() - axis.station_along_ch.min()) / 1000,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Per-image processing
    # ──────────────────────────────────────────────────────────────────────

    def _obs_wl_at(self, dt: datetime) -> np.ndarray:
        """Observed / reconstructed WSE at all stations at datetime dt."""
        return np.array(
            [self._wl_interp[s.station_id](dt) for s in self._stations],
            dtype=np.float64,
        )

    def _process_one(self, tif_path: Path) -> Optional[Path]:
        out_stem = tif_path.stem + "_wse" + tif_path.suffix
        out_path = self.output_dir / out_stem

        if out_path.exists() and not self.overwrite:
            log.info("Skip (exists): %s", out_path.name)
            return None

        dt = self.date_parser(tif_path.name)
        if dt is None:
            log.warning("Cannot parse datetime from '%s'; skipping.", tif_path.name)
            return None

        with rasterio.open(tif_path) as src:
            data      = src.read(1).astype(np.float32)
            profile   = src.profile.copy()
            transform = src.transform
            crs_wkt   = src.crs.to_wkt() if src.crs else None
            nd        = src.nodata

            if self._stations is None:
                if crs_wkt is None:
                    raise ValueError(
                        f"'{tif_path}' has no CRS; cannot build channel axis."
                    )
                self._init_spatial(crs_wkt)

        # Masks
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            data[data == nd] = np.nan
        nodata_mask = np.isnan(data)
        river_mask  = np.isclose(data, RIVER_MARKER, atol=1.0) & ~nodata_mask

        output = data.copy()

        n_river = int(river_mask.sum())
        if n_river > 0:
            row_i, col_i = np.where(river_mask)
            pix_xy  = pixel_centres(row_i, col_i, transform)
            obs_wl  = self._obs_wl_at(dt)

            wse = self._interp_engine.interpolate(
                t_dt=dt, pixel_xy=pix_xy, obs_wl=obs_wl
            )
            output[river_mask] = wse

        # Write output
        profile.update(
            dtype="float32", nodata=np.nan, count=1,
            compress="lzw", tiled=True, blockxsize=256, blockysize=256,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(output, 1)

        wse_vals = output[river_mask]
        log.info(
            "  %s → %d river px | WSE %.3f–%.3f m",
            out_path.name, n_river,
            float(wse_vals.min()) if n_river else np.nan,
            float(wse_vals.max()) if n_river else np.nan,
        )
        return out_path

    # ──────────────────────────────────────────────────────────────────────
    # Public API (unchanged from v1)
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        files: Optional[List[Union[str, Path]]] = None,
    ) -> pd.DataFrame:
        """Process all (or a supplied list of) GeoTIFFs."""
        tif_list = (
            [Path(f) for f in files]
            if files else
            sorted(self.geotiff_dir.glob(self.glob_pattern))
        )
        if not tif_list:
            log.warning("No files found matching '%s'.", self.glob_pattern)
            return pd.DataFrame()

        records = []
        iterable = tqdm(tif_list, desc="WSE (phase-aware)") if _HAS_TQDM else tif_list

        for tif in iterable:
            dt, status, out_path = self.date_parser(tif.name), "ok", None
            try:
                out_path = self._process_one(tif)
                if out_path is None:
                    status = "skipped"
            except Exception as exc:
                log.error("Error on '%s': %s", tif.name, exc, exc_info=True)
                status = f"error: {exc}"
            records.append({
                "input": str(tif), "output": str(out_path or ""),
                "datetime": dt, "status": status,
            })

        df = pd.DataFrame(records)
        log.info("Done: %d/%d OK.", (df.status == "ok").sum(), len(df))
        return df

    def run_single(
        self,
        tif_path: Union[str, Path],
        dt_override: Optional[datetime] = None,
    ) -> Optional[Path]:
        tif_path = Path(tif_path)
        if dt_override:
            orig = self.date_parser
            self.date_parser = lambda _: dt_override
            result = self._process_one(tif_path)
            self.date_parser = orig
            return result
        return self._process_one(tif_path)
