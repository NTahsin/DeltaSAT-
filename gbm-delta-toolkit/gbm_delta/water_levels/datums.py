# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:24:50 2026

@author: tahsi001

================================================================================
mhsw_mlsw_analysis.py
================================================================================
Derives Mean High Spring Water (MHSW) and Mean Low Spring Water (MLSW)
per station · per year · per season from reconstructed water level data
produced by tidal_gauge_reconstruction.py.

Designed for the GBM delta intertidal morphology study (JGR Earth Surface).

─── WHAT THIS SCRIPT CONSUMES ───────────────────────────────────────────────
Outputs of TidalGaugeReconstructor.run():

  water_level_ntr_reconstructed_YYYY_YYYY_Nmin.csv   (or _tidal_)
      Index  : DatetimeIndex (sub-daily, e.g. 5-min or 15-min)
      Columns: one per station  (e.g. "Khulna", "Mongla", …)

  tidal_constituents.csv
      Columns: station, constituent, amplitude_m, phase_deg
      Used to verify M2 amplitude and flag macro/meso-tidal stations.

  [optional] ntr_tier_report.csv
      Columns: station, tier, tier_label, max_gap_days
      Appended to results for transparency.

─── WHAT THIS SCRIPT PRODUCES ───────────────────────────────────────────────
  mhsw_mlsw_results.csv
      One row per (station × year × season) with:
        MHSW, MLSW, spring tidal range, N spring events,
        spring percentile threshold used, data completeness (%)

  mhsw_mlsw_diagnostics.png
      4-panel figure per station: time series + seasonal MHSW/MLSW trends

  mhsw_mlsw_summary_stats.csv
      Long-term mean MHSW/MLSW ± std per station per season

─── SEASON DEFINITION (GBM delta, literature-based) ────────────────────────
  Wet  (monsoon)    : June – October     (fluvial forcing dominant)
  Dry  (post-monsoon/pre-monsoon): November – May   (tidal forcing dominant)
  Reference: Auerbach et al. (2015, Nature Geoscience); consistent with
             DRY_MONTHS / MONSOON_MONTHS in tidal_gauge_reconstruction.py

─── SPRING TIDE IDENTIFICATION ──────────────────────────────────────────────
  Upper tercile (top 33 %) of tidal range within each seasonal window.
  Physically consistent with the spring-neap modulation resolved by UTide.
  Adjustable via SPRING_PERCENTILE below.

─── USAGE ───────────────────────────────────────────────────────────────────
  # Minimal
  python mhsw_mlsw_analysis.py \
      --wl       reconstructed/water_level_ntr_reconstructed_1996_2020_5min.csv \
      --output   results/

  # With constituent table and NTR tier report
  python mhsw_mlsw_analysis.py \
      --wl          reconstructed/water_level_ntr_reconstructed_1996_2020_5min.csv \
      --constituents reconstructed/tidal_constituents.csv \
      --ntr_tiers   reconstructed/ntr_tier_report.csv \
      --output      results/ \
      --spring_pct  67 \
      --min_completeness 80

  # Python API
  from mhsw_mlsw_analysis import MHSWMLSWAnalyser
  analyser = MHSWMLSWAnalyser(wl_csv='reconstructed/wl.csv', output_dir='results/')
  results  = analyser.run()

─── REQUIREMENTS ────────────────────────────────────────────────────────────
  pip install pandas numpy scipy matplotlib
================================================================================
"""

from __future__ import annotations

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (match tidal_gauge_reconstruction.py conventions)
# ════════════════════════════════════════════════════════════════════════════

#: Months defining each GBM season
WET_MONTHS = [6, 7, 8, 9, 10]        # June–October   (monsoon / fluvial)
DRY_MONTHS = [11, 12, 1, 2, 3, 4, 5] # November–May   (tidal dominant)

SEASON_LABEL = {
    'wet': 'Wet (Jun–Oct)',
    'dry': 'Dry (Nov–May)',
}

#: Minimum separation between successive HW or LW events (hours).
#  Set to slightly less than a semi-diurnal half-period (~6.2 h) to avoid
#  double-counting within one tidal cycle.
MIN_PEAK_SEPARATION_H = 6.0

#: Default spring-tide threshold: upper percentile of tidal range.
#  67 → top tercile; use 75 for a stricter definition.
DEFAULT_SPRING_PERCENTILE = 67

#: Minimum data completeness (%) required to compute MHSW/MLSW for a window.
#  Windows below this are flagged NaN.
DEFAULT_MIN_COMPLETENESS = 80


# ════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _season_of(month: int) -> str:
    """Return 'wet' or 'dry' for a calendar month integer."""
    return 'wet' if month in WET_MONTHS else 'dry'


def _season_year_key(ts: pd.Timestamp) -> tuple[str, int]:
    """
    Assign a (season, hydrological_year) key to a timestamp.

    The dry season spans Nov–May, which crosses a calendar year boundary.
    We assign the DRY season to the year in which it *starts* (November).
    So Nov 2005 – May 2006 → ('dry', 2005).
    The WET season stays within one calendar year → ('wet', year).
    """
    season = _season_of(ts.month)
    if season == 'dry' and ts.month in [1, 2, 3, 4, 5]:
        # Jan–May belong to the dry season that started in the previous November
        return 'dry', ts.year - 1
    return season, ts.year


def _detect_interval_minutes(index: pd.DatetimeIndex) -> float:
    """Estimate median time step in minutes from a DatetimeIndex."""
    if len(index) < 2:
        return 15.0
    diffs = pd.Series(index).diff().dropna()
    return diffs.median().total_seconds() / 60.0


def _completeness(series: pd.Series, interval_min: float) -> float:
    """
    Data completeness of a series window as a percentage.
    Expected n_obs = window_duration / interval_min.
    """
    if series.empty:
        return 0.0
    duration_min = (series.index[-1] - series.index[0]).total_seconds() / 60.0
    expected     = duration_min / interval_min + 1
    actual       = series.notna().sum()
    return min(100.0 * actual / expected, 100.0)


# ════════════════════════════════════════════════════════════════════════════
# PEAK / TROUGH DETECTION
# ════════════════════════════════════════════════════════════════════════════

def extract_hw_lw(
    series: pd.Series,
    interval_min: float,
    min_sep_h: float = MIN_PEAK_SEPARATION_H,
) -> tuple[pd.Series, pd.Series]:
    """
    Extract High Water (HW) and Low Water (LW) events from a continuous
    sub-daily water level series.

    Parameters
    ----------
    series       : water level time series (DatetimeIndex, metres)
    interval_min : time step in minutes (used to convert hour-separation → samples)
    min_sep_h    : minimum time between successive HW (or LW) events in hours

    Returns
    -------
    hw : pd.Series  high water levels with DatetimeIndex timestamps
    lw : pd.Series  low water levels with DatetimeIndex timestamps
    """
    # Fill short gaps by linear interpolation so find_peaks works cleanly
    filled = series.interpolate(method='time', limit=12)  # max 1 h gap at 5-min
    values = filled.values.astype(float)

    min_sep_samples = max(1, int(min_sep_h * 60.0 / interval_min))

    # Prominence filter: require peaks to stand out by at least 20 % of the
    # observed water level range within the window — suppresses noise spikes
    # and sub-peaks within a single tidal cycle.
    wl_range    = float(np.nanmax(values) - np.nanmin(values))
    prominence  = max(0.05, 0.10 * wl_range)   # at least 5 cm, ≥10 % of range

    # High water: local maxima
    hw_idx, _ = find_peaks(values,  distance=min_sep_samples, prominence=prominence)
    # Low water : local minima (find_peaks on negated series)
    lw_idx, _ = find_peaks(-values, distance=min_sep_samples, prominence=prominence)

    hw = pd.Series(values[hw_idx], index=series.index[hw_idx], name='HW')
    lw = pd.Series(values[lw_idx], index=series.index[lw_idx], name='LW')

    return hw, lw


# ════════════════════════════════════════════════════════════════════════════
# TIDAL RANGE PAIRING
# ════════════════════════════════════════════════════════════════════════════

def compute_tidal_ranges(hw, lw):
    if hw.empty or lw.empty:
        return pd.DataFrame(
            columns=['hw_time', 'hw_level', 'lw_time', 'lw_level', 'range_m']
        )

    rows = []
    # Convert to int64 (nanoseconds since epoch) so searchsorted works
    # regardless of numpy version — pd.Timestamp.value is also int64 ns
    lw_times = lw.index.astype("int64").values

    for hw_t, hw_val in hw.items():
        pos = np.searchsorted(lw_times, hw_t.value)   # ← .value here

        candidates = []
        if pos < len(lw_times):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)

        if not candidates:
            continue

        nearest = min(
            candidates,
            key=lambda i: abs(lw_times[i] - hw_t.value)   # ← .value here
        )
        lw_t   = lw.index[nearest]
        lw_val = lw.iloc[nearest]

        rows.append({
            'hw_time':  hw_t,
            'hw_level': hw_val,
            'lw_time':  lw_t,
            'lw_level': lw_val,
            'range_m':  hw_val - lw_val,
        })

    df = pd.DataFrame(rows)
    df = df[(df['range_m'] > 0) & (df['range_m'] < 15.0)].reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# SPRING TIDE IDENTIFICATION
# ════════════════════════════════════════════════════════════════════════════

def identify_spring_tides(
    ranges_df: pd.DataFrame,
    spring_percentile: float = DEFAULT_SPRING_PERCENTILE,
) -> pd.DataFrame:
    """
    Identify spring tidal cycles as those with range ≥ the spring_percentile
    threshold within the supplied DataFrame window.

    Returns the subset of ranges_df corresponding to spring conditions, with
    an additional column 'range_threshold_m' for reporting.
    """
    if ranges_df.empty:
        return ranges_df.copy()

    threshold = np.nanpercentile(ranges_df['range_m'].values, spring_percentile)
    spring_df = ranges_df[ranges_df['range_m'] >= threshold].copy()
    spring_df['range_threshold_m'] = threshold
    return spring_df


# ════════════════════════════════════════════════════════════════════════════
# MHSW / MLSW COMPUTATION  (single window)
# ════════════════════════════════════════════════════════════════════════════

def compute_mhsw_mlsw_window(
    series: pd.Series,
    interval_min: float,
    spring_percentile: float = DEFAULT_SPRING_PERCENTILE,
    min_completeness: float  = DEFAULT_MIN_COMPLETENESS,
) -> dict:
    """
    Compute MHSW and MLSW for a single seasonal window (one station, one
    season, one year).

    Parameters
    ----------
    series           : sub-daily water level series for the window
    interval_min     : time step in minutes
    spring_percentile: upper percentile cutoff defining spring conditions
    min_completeness : minimum data fill (%) to return a valid result

    Returns
    -------
    dict with keys:
        mhsw_m, mlsw_m, spring_range_m,
        n_spring_cycles, range_threshold_m,
        completeness_pct, n_hw_events, n_lw_events, valid
    """
    result = dict(
        mhsw_m=np.nan, mlsw_m=np.nan, spring_range_m=np.nan,
        n_spring_cycles=0, range_threshold_m=np.nan,
        completeness_pct=np.nan, n_hw_events=0, n_lw_events=0,
        valid=False,
    )

    if series is None or series.empty:
        return result

    completeness = _completeness(series, interval_min)
    result['completeness_pct'] = round(completeness, 1)

    if completeness < min_completeness:
        return result  # insufficient data coverage

    # --- Step 1: Extract HW / LW events
    hw, lw = extract_hw_lw(series, interval_min)
    result['n_hw_events'] = len(hw)
    result['n_lw_events'] = len(lw)

    if len(hw) < 5 or len(lw) < 5:
        return result  # too few events for a reliable estimate

    # --- Step 2: Compute tidal ranges per cycle
    ranges_df = compute_tidal_ranges(hw, lw)
    if ranges_df.empty:
        return result

    # --- Step 3: Identify spring tidal cycles
    spring_df = identify_spring_tides(ranges_df, spring_percentile)
    result['n_spring_cycles']  = len(spring_df)
    result['range_threshold_m'] = (
        round(spring_df['range_threshold_m'].iloc[0], 4) if not spring_df.empty else np.nan
    )

    if spring_df.empty or len(spring_df) < 3:
        return result

    # --- Step 4: Compute MHSW and MLSW
    result['mhsw_m']       = round(float(spring_df['hw_level'].mean()), 4)
    result['mlsw_m']       = round(float(spring_df['lw_level'].mean()), 4)
    result['spring_range_m'] = round(result['mhsw_m'] - result['mlsw_m'], 4)
    result['valid']        = True

    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSER CLASS
# ════════════════════════════════════════════════════════════════════════════

class MHSWMLSWAnalyser:
    """
    Derives MHSW and MLSW per station · per year · per season from the
    reconstructed water level DataFrame produced by TidalGaugeReconstructor.

    Parameters
    ----------
    wl_csv            : path to reconstructed water level CSV
                        (index = datetime, columns = station names)
    output_dir        : directory for output files
    constituents_csv  : [optional] tidal_constituents.csv from reconstructor
    ntr_tiers_csv     : [optional] ntr_tier_report.csv from reconstructor
    spring_percentile : upper percentile of tidal range defining spring tides
    min_completeness  : minimum % data coverage required per seasonal window
    """

    def __init__(
        self,
        wl_csv:            str,
        output_dir:        str  = '.',
        constituents_csv:  str | None = None,
        ntr_tiers_csv:     str | None = None,
        spring_percentile: float = DEFAULT_SPRING_PERCENTILE,
        min_completeness:  float = DEFAULT_MIN_COMPLETENESS,
    ):
        self.wl_csv            = wl_csv
        self.output_dir        = output_dir
        self.constituents_csv  = constituents_csv
        self.ntr_tiers_csv     = ntr_tiers_csv
        self.spring_percentile = spring_percentile
        self.min_completeness  = min_completeness

        os.makedirs(output_dir, exist_ok=True)

        # Loaded data
        self._wl:           pd.DataFrame | None = None
        self._constituents: pd.DataFrame | None = None
        self._ntr_tiers:    pd.DataFrame | None = None
        self._interval_min: float = 15.0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Full pipeline: load → derive MHSW/MLSW → save outputs → plot.

        Returns
        -------
        pd.DataFrame  results table (also saved to output_dir)
        """
        print("\n" + "═" * 70)
        print("  MHSW / MLSW ANALYSIS — GBM delta intertidal study")
        print("═" * 70)

        self._load_data()
        results = self._derive_all_stations()
        self._save_results(results)
        self._save_summary_stats(results)
        self._plot_diagnostics(results)

        print("\n✅  Analysis complete.")
        print(f"   Results   → {os.path.join(self.output_dir, 'mhsw_mlsw_results.csv')}")
        print(f"   Summary   → {os.path.join(self.output_dir, 'mhsw_mlsw_summary_stats.csv')}")
        print(f"   Figures   → {os.path.join(self.output_dir, 'mhsw_mlsw_diagnostics.png')}")

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Load
    # ──────────────────────────────────────────────────────────────────────

    def _load_data(self):
        print(f"\n📂  Loading reconstructed water levels …")
        print(f"    {self.wl_csv}")

        df = pd.read_csv(self.wl_csv, index_col=0, parse_dates=True)
        df = df.sort_index()

        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        self._wl           = df
        self._interval_min = _detect_interval_minutes(df.index)
        self._stations     = list(df.columns)

        print(f"    Stations   : {self._stations}")
        print(f"    Period     : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"    Time step  : {self._interval_min:.1f} min")
        print(f"    Rows       : {len(df):,}")

        # Optional: tidal constituents
        if self.constituents_csv and os.path.isfile(self.constituents_csv):
            self._constituents = pd.read_csv(self.constituents_csv)
            print(f"\n📂  Tidal constituents loaded: {self.constituents_csv}")
            self._print_m2_summary()

        # Optional: NTR tier report
        if self.ntr_tiers_csv and os.path.isfile(self.ntr_tiers_csv):
            self._ntr_tiers = pd.read_csv(self.ntr_tiers_csv)
            print(f"\n📂  NTR tier report loaded: {self.ntr_tiers_csv}")
            print(self._ntr_tiers.to_string(index=False))

    def _print_m2_summary(self):
        """Print M2 amplitude per station (tidal regime indicator)."""
        if self._constituents is None:
            return
        m2 = (self._constituents[self._constituents['constituent'] == 'M2']
              [['station', 'amplitude_m']])
        if not m2.empty:
            print("\n   M2 amplitude per station (tidal regime indicator):")
            for _, row in m2.iterrows():
                amp = row['amplitude_m']
                regime = ('Macro-tidal' if amp > 2.0
                          else 'Meso-tidal' if amp > 1.0
                          else 'Micro-tidal')
                print(f"   {str(row['station']):<30}  M2={amp:.3f} m  [{regime}]")

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Derive MHSW/MLSW for all stations
    # ──────────────────────────────────────────────────────────────────────

    def _derive_all_stations(self) -> pd.DataFrame:
        """Loop over stations and call _derive_one_station() for each."""
        all_rows = []
        for station in self._stations:
            print(f"\n{'─'*60}")
            print(f"  Station: {station}")
            print(f"{'─'*60}")
            rows = self._derive_one_station(station)
            all_rows.extend(rows)

        results = pd.DataFrame(all_rows)
        return results

    def _derive_one_station(self, station: str) -> list[dict]:
        """
        Derive MHSW/MLSW per year per season for one station.

        Seasonal windows are determined by the (season, hydrological_year)
        key returned by _season_year_key(), which correctly handles the
        Nov–May dry-season crossing the calendar year boundary.
        """
        series = self._wl[station].dropna()
        if series.empty:
            print(f"⚠️  No valid data {station}— skipping")
            return []

        # Group timestamps by (season, hyd_year)
        keys = pd.Series(
            [_season_year_key(t) for t in series.index],
            index=series.index,
        )

        rows = []
        for (season, hyd_year), idx in keys.groupby(keys).groups.items():
            window = series.loc[idx].sort_index()

            metrics = compute_mhsw_mlsw_window(
                series           = window,
                interval_min     = self._interval_min,
                spring_percentile= self.spring_percentile,
                min_completeness = self.min_completeness,
            )

            # Determine calendar year range for the dry-season window label
            if season == 'dry':
                year_label = f"{hyd_year}/{hyd_year + 1}"
            else:
                year_label = str(hyd_year)

            row = {
                'station':           station,
                'season':            season,
                'season_label':      SEASON_LABEL[season],
                'hyd_year':          hyd_year,
                'year_label':        year_label,
                'window_start':      window.index[0].date(),
                'window_end':        window.index[-1].date(),
                'spring_pct_used':   self.spring_percentile,
                **metrics,
            }

            # Append NTR tier if available
            if self._ntr_tiers is not None:
                match = self._ntr_tiers[self._ntr_tiers['station'] == station]
                if not match.empty:
                    row['ntr_tier']       = match.iloc[0].get('tier', np.nan)
                    row['ntr_tier_label'] = match.iloc[0].get('tier_label', '')

            status = '✓' if metrics['valid'] else '✗'
            print(f"  {status}  {season:<4}  {year_label:<9}  "
                  f"MHSW={metrics['mhsw_m']:>7.3f} m  "
                  f"MLSW={metrics['mlsw_m']:>7.3f} m  "
                  f"STR={metrics['spring_range_m']:>6.3f} m  "
                  f"N_spring={metrics['n_spring_cycles']:>3}  "
                  f"completeness={metrics['completeness_pct']:>5.1f}%")

            rows.append(row)

        return rows

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Save results
    # ──────────────────────────────────────────────────────────────────────

    def _save_results(self, results: pd.DataFrame):
        path = os.path.join(self.output_dir, 'mhsw_mlsw_results.csv')
        results.to_csv(path, index=False, float_format='%.4f')
        print(f"\n💾  Full results → {path}  ({len(results)} rows)")

    def _save_summary_stats(self, results: pd.DataFrame):
        """Long-term mean ± std per station per season (valid windows only)."""
        valid = results[results['valid']].copy()
        if valid.empty:
            print("  ⚠️  No valid windows — summary stats not produced")
            return

        summary = (
            valid.groupby(['station', 'season', 'season_label'])
            .agg(
                n_years         = ('hyd_year',      'count'),
                mhsw_mean_m     = ('mhsw_m',        'mean'),
                mhsw_std_m      = ('mhsw_m',        'std'),
                mlsw_mean_m     = ('mlsw_m',        'mean'),
                mlsw_std_m      = ('mlsw_m',        'std'),
                spring_range_mean_m = ('spring_range_m', 'mean'),
                spring_range_std_m  = ('spring_range_m', 'std'),
                mean_completeness   = ('completeness_pct','mean'),
            )
            .reset_index()
        )
        for col in ['mhsw_mean_m','mhsw_std_m','mlsw_mean_m','mlsw_std_m',
                    'spring_range_mean_m','spring_range_std_m','mean_completeness']:
            summary[col] = summary[col].round(4)

        path = os.path.join(self.output_dir, 'mhsw_mlsw_summary_stats.csv')
        summary.to_csv(path, index=False, float_format='%.4f')
        print(f"💾  Summary stats → {path}")

        # Print to console
        print("\n" + "═" * 70)
        print("  LONG-TERM SUMMARY  (mean ± std across valid years)")
        print("═" * 70)
        print(f"  {'Station':<25} {'Season':<8} {'N':>4}  "
              f"{'MHSW (m)':>12}  {'MLSW (m)':>12}  {'STR (m)':>10}")
        print("  " + "─" * 70)
        for _, r in summary.iterrows():
            print(f"  {r['station']:<25} {r['season']:<8} {int(r['n_years']):>4}  "
                  f"{r['mhsw_mean_m']:>6.3f}±{r['mhsw_std_m']:.3f}  "
                  f"{r['mlsw_mean_m']:>6.3f}±{r['mlsw_std_m']:.3f}  "
                  f"{r['spring_range_mean_m']:>6.3f}±{r['spring_range_std_m']:.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Diagnostic plots
    # ──────────────────────────────────────────────────────────────────────

    def _plot_diagnostics(self, results: pd.DataFrame):
        """
        Produce a multi-panel diagnostic figure per station:
          Panel A : Full reconstructed water level time series
          Panel B : Annual MHSW and MLSW — wet season
          Panel C : Annual MHSW and MLSW — dry season
          Panel D : Spring tidal range (MHSW−MLSW) both seasons
        """
        valid = results[results['valid']].copy()
        n_stations = len(self._stations)

        fig, axes = plt.subplots(
            nrows = 4, ncols = n_stations,
            figsize=(6 * n_stations, 14),
            squeeze=False,
        )
        fig.suptitle(
            'MHSW / MLSW Diagnostics — GBM Delta\n'
            f'Spring percentile: top {100 - self.spring_percentile}%  '
            f'| Min completeness: {self.min_completeness}%',
            fontsize=12, fontweight='bold', y=0.99,
        )

        colours = {'wet': '#1f77b4', 'dry': '#d62728'}

        for col_idx, station in enumerate(self._stations):
            wl    = self._wl[station].dropna()
            sv    = valid[valid['station'] == station]

            # ── Panel A: Full WL time series (monthly mean for clarity) ──
            ax = axes[0, col_idx]
            wl_monthly = wl.resample('ME').mean()
            ax.plot(wl_monthly.index, wl_monthly.values,
                    color='steelblue', lw=0.8, alpha=0.8)
            ax.set_title(station, fontsize=11, fontweight='bold')
            ax.set_ylabel('Water level (m)', fontsize=9)
            ax.set_xlabel('')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, ls=':', alpha=0.5)
            ax.set_title(f'{station}\n(monthly mean WL)', fontsize=10,
                         fontweight='bold')

            # ── Panel B: Annual MHSW & MLSW — wet season ──
            ax = axes[1, col_idx]
            for metric, label, marker in [
                ('mhsw_m', 'MHSW', 'o'),
                ('mlsw_m', 'MLSW', 's'),
            ]:
                sw = sv[sv['season'] == 'wet'].sort_values('hyd_year')
                if not sw.empty:
                    ax.plot(sw['hyd_year'], sw[metric],
                            color=colours['wet'], marker=marker,
                            lw=1.5, ms=5, label=label)
                    ax.fill_between(sw['hyd_year'], sw[metric],
                                    alpha=0.08, color=colours['wet'])
            ax.set_ylabel('Water level (m)', fontsize=9)
            ax.set_title('Wet season (Jun–Oct)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, ls=':', alpha=0.5)
            ax.tick_params(labelsize=8)

            # ── Panel C: Annual MHSW & MLSW — dry season ──
            ax = axes[2, col_idx]
            for metric, label, marker in [
                ('mhsw_m', 'MHSW', 'o'),
                ('mlsw_m', 'MLSW', 's'),
            ]:
                sd = sv[sv['season'] == 'dry'].sort_values('hyd_year')
                if not sd.empty:
                    ax.plot(sd['hyd_year'], sd[metric],
                            color=colours['dry'], marker=marker,
                            lw=1.5, ms=5, label=label)
                    ax.fill_between(sd['hyd_year'], sd[metric],
                                    alpha=0.08, color=colours['dry'])
            ax.set_ylabel('Water level (m)', fontsize=9)
            ax.set_title('Dry season (Nov–May)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, ls=':', alpha=0.5)
            ax.tick_params(labelsize=8)

            # ── Panel D: Spring tidal range — both seasons ──
            ax = axes[3, col_idx]
            for season, lbl in [('wet', 'Wet'), ('dry', 'Dry')]:
                ss = sv[sv['season'] == season].sort_values('hyd_year')
                if not ss.empty:
                    ax.plot(ss['hyd_year'], ss['spring_range_m'],
                            color=colours[season], marker='D',
                            lw=1.5, ms=4, label=f'{lbl} STR')
            ax.set_ylabel('Spring tidal range (m)', fontsize=9)
            ax.set_xlabel('Hydrological year', fontsize=9)
            ax.set_title('Spring tidal range (MHSW − MLSW)', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, ls=':', alpha=0.5)
            ax.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(self.output_dir, 'mhsw_mlsw_diagnostics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾  Diagnostic figure → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Derive MHSW / MLSW per station · per year · per season\n'
            'from TidalGaugeReconstructor output (GBM delta study).'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--wl', required=True,
        help='Reconstructed WL CSV from tidal_gauge_reconstruction.py\n'
             '(index=datetime, columns=station names)',
    )
    parser.add_argument(
        '--output', default='mhsw_mlsw_outputs/',
        help='Output directory (default: mhsw_mlsw_outputs/)',
    )
    parser.add_argument(
        '--constituents', default=None,
        help='[optional] tidal_constituents.csv from reconstructor',
    )
    parser.add_argument(
        '--ntr_tiers', default=None,
        help='[optional] ntr_tier_report.csv from reconstructor',
    )
    parser.add_argument(
        '--spring_pct', type=float, default=DEFAULT_SPRING_PERCENTILE,
        help=(f'Percentile threshold defining spring tides '
              f'(default: {DEFAULT_SPRING_PERCENTILE} → top tercile)'),
    )
    parser.add_argument(
        '--min_completeness', type=float, default=DEFAULT_MIN_COMPLETENESS,
        help=(f'Minimum data fill (%%) per seasonal window '
              f'(default: {DEFAULT_MIN_COMPLETENESS})'),
    )
    args = parser.parse_args()

    analyser = MHSWMLSWAnalyser(
        wl_csv            = args.wl,
        output_dir        = args.output,
        constituents_csv  = args.constituents,
        ntr_tiers_csv     = args.ntr_tiers,
        spring_percentile = args.spring_pct,
        min_completeness  = args.min_completeness,
    )
    analyser.run()


if __name__ == '__main__':
    main()