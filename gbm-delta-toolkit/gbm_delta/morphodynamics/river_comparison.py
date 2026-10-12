"""
river_intertidal_comparator.py
================================
Compares intertidal area extent, elevation, and Markov-derived lifespan
across multiple river channels, using shapefile(s) to define each river's
spatial domain.

Workflow
--------
  1. Load X river polygon/polyline shapefiles (or a single multi-feature file).
  2. For each year, clip occurrence and elevation GeoTIFFs to each river's
     bounding polygon (with an optional buffer) and compute per-river metrics.
  3. Derive per-river DTMC lifespan from the clipped occurrence stacks.
  4. Produce a multi-river comparison report, time-series figures, and
     spatial overview plots.

Shapefile input
---------------
Either:
  A) One shapefile per river — pass as a list of paths.
  B) A single shapefile with a name/ID attribute column that labels each
     river polygon separately.

The shapefile CRS must be provided or auto-detected; all clipping is done
in the raster's native CRS.

Output structure
----------------
  <output_dir>/rivers/
      <river_name>/
          occurrence_dry_YYYY_clipped.tif    — one per year per river
          elevation_median_YYYY_clipped.tif  — one per year per river
          pii_map_YYYY-YYYY.tif
          lifespan_map_YYYY-YYYY.tif
      comparison_timeseries.csv
      comparison_lifespan.csv
      figures/
          11a_river_intertidal_area_ts.pdf
          11b_river_elevation_ts.pdf
          11c_river_lifespan_comparison.pdf
          11d_river_spatial_overview.pdf

Dependencies
------------
    pip install numpy pandas rasterio fiona shapely scipy matplotlib tqdm
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds

try:
    import fiona
    from shapely.geometry import shape, mapping, box
    from shapely.ops import unary_union
    _HAS_FIONA = True
except ImportError:
    _HAS_FIONA = False
    warnings.warn("fiona/shapely not installed — shapefile reading unavailable. "
                  "Install with: pip install fiona shapely", stacklevel=2)

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.backends.backend_pdf import PdfPages
    _PLOT = True
except ImportError:
    _PLOT = False

from gbm_delta.morphodynamics.lifespan_markov import (
    occurrence_to_state, fit_global_dtmc, compute_pixelwise_pii,
    compute_lifespan_map, lifespan_from_pii, wilson_ci,
    STATE_NAMES, INTERTIDAL, LAND, RIVER,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Colour palette for rivers (up to 8) ──────────────────────────────────────
RIVER_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
]

# ── Pixel area (30 m Landsat) ─────────────────────────────────────────────────
PX_AREA_KM2 = 30.0**2 / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Shapefile loading
# ─────────────────────────────────────────────────────────────────────────────

def load_river_geometries(
    shapefiles: List[Union[str, Path]],
    name_field: Optional[str] = None,
    buffer_m:   float = 500.0,
) -> Dict[str, object]:
    """
    Load river geometries from one or more shapefiles.

    Parameters
    ----------
    shapefiles   list of shapefile paths, OR a single path to a multi-feature file
    name_field   attribute column to use as river name  (None = use filename stem
                 for single-feature files, or feature index for multi-feature)
    buffer_m     buffer added around each geometry in metres before clipping

    Returns
    -------
    dict {river_name: shapely_geometry (buffered union)}
    """
    if not _HAS_FIONA:
        raise ImportError("fiona and shapely are required. "
                          "Install with: pip install fiona shapely")

    rivers: Dict[str, object] = {}

    for shp_path in shapefiles:
        shp_path = Path(shp_path)
        if not shp_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        with fiona.open(shp_path) as src:
            log.info("Opened %s: %d features, CRS=%s",
                     shp_path.name, len(src), src.crs)
            features = list(src)
            shp_crs  = src.crs

        if len(features) == 0:
            log.warning("Empty shapefile: %s", shp_path)
            continue

        # One shapefile per river (single or merged geometry)
        if name_field is None or name_field not in (features[0]["properties"] or {}):
            river_name = shp_path.stem
            geoms = [shape(f["geometry"]) for f in features
                     if f["geometry"] is not None]
            merged = unary_union(geoms)
            if buffer_m > 0:
                merged = merged.buffer(buffer_m)
            rivers[river_name] = merged
            log.info("River '%s': %d features merged", river_name, len(geoms))

        else:
            # Multi-river file: split by name_field
            by_name: Dict[str, list] = {}
            for feat in features:
                n = str(feat["properties"].get(name_field, f"river_{len(by_name)}"))
                geom = shape(feat["geometry"]) if feat["geometry"] else None
                if geom:
                    by_name.setdefault(n, []).append(geom)
            for n, geoms in by_name.items():
                merged = unary_union(geoms)
                if buffer_m > 0:
                    merged = merged.buffer(buffer_m)
                rivers[n] = merged
                log.info("River '%s': %d features", n, len(geoms))

    return rivers


# ─────────────────────────────────────────────────────────────────────────────
# Per-river clipping
# ─────────────────────────────────────────────────────────────────────────────

def clip_raster_to_geometry(
    raster_path: Path,
    geometry,
    nodata: float = np.nan,
) -> Optional[tuple[np.ndarray, dict]]:
    """
    Clip a single-band GeoTIFF to a Shapely geometry.

    Returns (array, profile) or None if no pixels intersect.
    """
    if not _HAS_FIONA:
        raise ImportError("shapely required for clipping.")
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rio_mask(
                src, [mapping(geometry)],
                crop=True, nodata=nodata, filled=True,
            )
            out_profile = src.profile.copy()
            out_profile.update(
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                nodata=nodata,
                compress="lzw",
                dtype="float32",
            )
        arr = out_image[0].astype(np.float32)
        arr[arr == nodata] = np.nan
        if not np.isfinite(arr).any():
            return None
        return arr, out_profile
    except Exception as exc:
        log.warning("Clip failed for %s: %s", raster_path.name, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-river metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_river_annual_metrics(
    river_name:  str,
    geometry,
    output_dir:  Path,
    years:       List[int],
    occ_glob:    str = "occurrence_dry_{year}.tif",
    elev_glob:   str = "elevation_median_{year}.tif",
    river_dir:   Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute annual intertidal metrics for one river by clipping occurrence
    and elevation GeoTIFFs to the river's geometry.

    Returns a DataFrame with one row per year.
    """
    rows = []
    rd   = river_dir or (output_dir / "rivers" / river_name)
    rd.mkdir(parents=True, exist_ok=True)

    it = tqdm(years, desc=f"  {river_name}", leave=False) if _TQDM else years

    for year in it:
        occ_path  = output_dir / str(year) / occ_glob.format(year=year)
        elev_path = output_dir / str(year) / elev_glob.format(year=year)

        row = {"river": river_name, "year": year,
               "n_intertidal": 0, "area_it_km2": 0.0,
               "elev_mean_m": np.nan, "elev_std_m": np.nan,
               "elev_min_m": np.nan, "elev_max_m": np.nan,
               "mean_occ_f": np.nan, "n_land": 0, "n_river": 0}

        if not occ_path.exists():
            rows.append(row); continue

        clipped_occ = clip_raster_to_geometry(occ_path, geometry)
        if clipped_occ is None:
            rows.append(row); continue

        occ_arr, occ_prof = clipped_occ
        state = occurrence_to_state(occ_arr)
        valid = state != 255
        n_it  = int((state[valid] == INTERTIDAL).sum())

        row["n_intertidal"] = n_it
        row["area_it_km2"]  = round(n_it * PX_AREA_KM2, 4)
        row["n_land"]       = int((state[valid] == LAND).sum())
        row["n_river"]      = int((state[valid] == RIVER).sum())

        it_mask = (state == INTERTIDAL) & valid
        if it_mask.any():
            row["mean_occ_f"] = round(float(np.nanmean(occ_arr[it_mask])), 4)

        if elev_path.exists():
            clipped_elev = clip_raster_to_geometry(elev_path, geometry)
            if clipped_elev is not None:
                elev_arr, _ = clipped_elev
                if elev_arr.shape == occ_arr.shape and it_mask.any():
                    vals = elev_arr[it_mask & np.isfinite(elev_arr)]
                    if len(vals) > 0:
                        row["elev_mean_m"] = round(float(vals.mean()), 4)
                        row["elev_std_m"]  = round(float(vals.std()),  4)
                        row["elev_min_m"]  = round(float(vals.min()),  4)
                        row["elev_max_m"]  = round(float(vals.max()),  4)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_river_lifespan(
    river_name: str,
    geometry,
    output_dir: Path,
    years:      List[int],
    occ_glob:   str = "occurrence_dry_{year}.tif",
) -> dict:
    """
    Fit DTMC and compute lifespan for one river's clipped occurrence stack.
    Returns dict with P_II, lifespan, CI, pixel-wise maps.
    """
    arrays = []
    valid_years = []
    profile = None

    for year in years:
        occ_path = output_dir / str(year) / occ_glob.format(year=year)
        if not occ_path.exists():
            continue
        clipped = clip_raster_to_geometry(occ_path, geometry)
        if clipped is None:
            continue
        occ_arr, prof = clipped
        arrays.append(occurrence_to_state(occ_arr))
        valid_years.append(year)
        if profile is None:
            profile = prof

    if len(arrays) < 2:
        log.warning("River '%s': fewer than 2 years — lifespan not computed.",
                    river_name)
        return {"river": river_name, "P_II": np.nan, "lifespan": np.nan,
                "n_years": len(arrays)}

    # Pad to common shape (should all be same if geometry is fixed)
    H = max(a.shape[0] for a in arrays)
    W = max(a.shape[1] for a in arrays)
    padded = []
    for a in arrays:
        p = np.full((H, W), 255, dtype=np.uint8)
        p[:a.shape[0], :a.shape[1]] = a
        padded.append(p)

    stack = np.stack(padded, axis=0)
    P, N_df = fit_global_dtmc(stack)

    pii = float(P[INTERTIDAL, INTERTIDAL])
    L   = lifespan_from_pii(pii)

    # Wilson CI
    n_ii = int(N_df[(N_df["from_state"]=="Intertidal") &
                    (N_df["to_state"]  =="Intertidal")]["N_obs"].values[0]) \
           if len(N_df) > 0 else 0
    n_i  = int(N_df[N_df["from_state"]=="Intertidal"]["N_obs"].sum()) \
           if len(N_df) > 0 else 0
    ci_lo, ci_hi = wilson_ci(n_ii, n_i)

    pii_px   = compute_pixelwise_pii(stack)
    lifespan = compute_lifespan_map(pii_px)

    return {
        "river":      river_name,
        "n_years":    len(valid_years),
        "year_start": valid_years[0],
        "year_end":   valid_years[-1],
        "P_II":       round(pii, 6),
        "P_II_ci_lo": round(ci_lo, 6),
        "P_II_ci_hi": round(ci_hi, 6),
        "lifespan_yr":round(L, 3),
        "lifespan_ci_lo": round(lifespan_from_pii(ci_lo), 3),
        "lifespan_ci_hi": round(lifespan_from_pii(ci_hi), 3)
                          if not np.isinf(lifespan_from_pii(ci_hi)) else 999,
        "pii_map":    pii_px,
        "lifespan_map": lifespan,
        "profile":    profile,
        "P_matrix":   P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _savepdf(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_river_area_timeseries(ts_all: pd.DataFrame, out_path: Path):
    """Intertidal area time series, one line per river."""
    if not _PLOT: return
    rivers = ts_all["river"].unique()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for idx, river in enumerate(rivers):
        col = RIVER_COLOURS[idx % len(RIVER_COLOURS)]
        sub = ts_all[ts_all["river"] == river].sort_values("year")
        axes[0].plot(sub["year"], sub["area_it_km2"],
                     color=col, marker="o", ms=5, lw=1.5, label=river)
        axes[1].plot(sub["year"], sub["mean_occ_f"],
                     color=col, marker="s", ms=5, lw=1.5, ls="--", label=river)

    for ax, ylabel in zip(axes,
                           ["Intertidal area (km²)", "Mean inundation frequency f"]):
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, lw=0.4, alpha=0.4, ls=":")
        ax.tick_params(direction="in", labelsize=9)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    axes[0].legend(fontsize=9, ncol=min(4, len(rivers)),
                   loc="upper right", framealpha=0.8)
    axes[-1].set_xlabel("Year", fontsize=10)
    fig.suptitle("Intertidal Area Time Series — River Comparison",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


def plot_river_elevation_timeseries(ts_all: pd.DataFrame, out_path: Path):
    """Mean intertidal elevation time series per river."""
    if not _PLOT: return
    rivers = ts_all["river"].unique()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for idx, river in enumerate(rivers):
        col = RIVER_COLOURS[idx % len(RIVER_COLOURS)]
        sub = ts_all[ts_all["river"] == river].sort_values("year")
        axes[0].plot(sub["year"], sub["elev_mean_m"],
                     color=col, marker="o", ms=5, lw=1.5, label=river)
        if "elev_std_m" in sub.columns:
            axes[0].fill_between(
                sub["year"],
                sub["elev_mean_m"] - sub["elev_std_m"],
                sub["elev_mean_m"] + sub["elev_std_m"],
                color=col, alpha=0.12,
            )
        axes[1].plot(sub["year"], sub["elev_max_m"] - sub["elev_min_m"],
                     color=col, marker="^", ms=5, lw=1.5, ls="--")

    for ax, ylabel in zip(axes,
                           ["Mean elevation (m MSL)  ± std",
                            "Elevation range (m)"]):
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, lw=0.4, alpha=0.4, ls=":")
        ax.tick_params(direction="in", labelsize=9)
    axes[0].legend(fontsize=9, ncol=min(4, len(rivers)), framealpha=0.8)
    axes[-1].set_xlabel("Year", fontsize=10)
    fig.suptitle("Intertidal Elevation Time Series — River Comparison",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


def plot_river_lifespan_comparison(lifespan_rows: List[dict], out_path: Path):
    """Bar chart comparing P_II and lifespan across rivers."""
    if not _PLOT: return
    df = pd.DataFrame(lifespan_rows).dropna(subset=["P_II"])
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df))
    colours = [RIVER_COLOURS[i % len(RIVER_COLOURS)] for i in range(len(df))]

    # P_II with CI error bars
    lo_err = df["P_II"] - df["P_II_ci_lo"]
    hi_err = df["P_II_ci_hi"] - df["P_II"]
    axes[0].bar(x, df["P_II"], color=colours, alpha=0.82, edgecolor="white", lw=0.5)
    axes[0].errorbar(x, df["P_II"],
                     yerr=[lo_err.clip(lower=0), hi_err.clip(lower=0)],
                     fmt="none", color="black", capsize=4, lw=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["river"], rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("P_II  (intertidal self-transition)", fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_title("Intertidal Self-Transition Probability", fontsize=11)
    axes[0].grid(True, lw=0.3, alpha=0.3, ls=":", axis="y")

    # Lifespan with CI error bars
    l_lo = df["lifespan_yr"] - df["lifespan_ci_lo"]
    l_hi = df["lifespan_ci_hi"] - df["lifespan_yr"]
    l_hi = l_hi.clip(upper=30)
    axes[1].bar(x, df["lifespan_yr"].clip(upper=30),
                color=colours, alpha=0.82, edgecolor="white", lw=0.5)
    axes[1].errorbar(x, df["lifespan_yr"].clip(upper=30),
                     yerr=[l_lo.clip(lower=0), l_hi],
                     fmt="none", color="black", capsize=4, lw=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["river"], rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Expected lifespan L  (years)", fontsize=10)
    axes[1].set_title("Mean Intertidal Lifespan  L = 1/(1−P_II)", fontsize=11)
    axes[1].grid(True, lw=0.3, alpha=0.3, ls=":", axis="y")

    for ax in axes:
        ax.tick_params(labelsize=9)

    fig.suptitle("Markov Chain Lifespan Comparison — River Channels",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


def plot_river_lifespan_maps(lifespan_results: dict, out_path: Path):
    """Side-by-side spatial lifespan maps for each river."""
    if not _PLOT: return
    rivers = list(lifespan_results.keys())
    n = len(rivers)
    if n == 0:
        return
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows),
                              squeeze=False)

    for idx, river in enumerate(rivers):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        res = lifespan_results[river]
        lmap = res.get("lifespan_map")
        prof = res.get("profile")
        if lmap is None or prof is None:
            ax.set_visible(False); continue

        t = prof.get("transform") or rasterio.transform.from_bounds(0,0,1,1,1,1)
        try:
            ex = array_bounds(lmap.shape[0], lmap.shape[1], t)
            extent = [ex[0], ex[2], ex[1], ex[3]]
        except Exception:
            extent = None

        im = ax.imshow(lmap, origin="upper", extent=extent,
                       cmap="RdYlGn", vmin=1, vmax=15,
                       aspect="equal", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Lifespan (yr)")
        pii = res.get("P_II", np.nan)
        L   = res.get("lifespan_yr", np.nan)
        ax.set_title(f"{river}\nP_II={pii:.3f}  L={L:.1f} yr", fontsize=10)
        ax.set_xlabel("Easting (m)", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, rows*cols):
        r,c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Spatial Lifespan of Intertidal Zones — River Comparison",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RiverIntertidalComparator:
    """
    Compares intertidal dynamics across X river channels.

    Parameters
    ----------
    shapefiles      list of shapefile paths (one per river), OR a single path
                    to a multi-feature shapefile with a name_field attribute
    output_dir      root intertidal elevation output directory (has YYYY/ subfolders)
    years           years to analyse  (None = auto-detect)
    name_field      attribute column for river names in a multi-river shapefile
    buffer_m        spatial buffer around each river geometry (m)
    occ_glob        occurrence filename template (default: v3)
    elev_glob       elevation filename template  (default: v3 median)
    """

    def __init__(
        self,
        shapefiles:  List[Union[str, Path]],
        output_dir:  Union[str, Path],
        years:       Optional[List[int]] = None,
        name_field:  Optional[str] = None,
        buffer_m:    float = 500.0,
        occ_glob:    str = "occurrence_dry_{year}.tif",
        elev_glob:   str = "elevation_median_{year}.tif",
    ):
        self.shapefiles  = [Path(s) for s in shapefiles]
        self.output_dir  = Path(output_dir)
        self.name_field  = name_field
        self.buffer_m    = buffer_m
        self.occ_glob    = occ_glob
        self.elev_glob   = elev_glob
        self._years      = years

    def _discover_years(self):
        return sorted(
            int(d.name) for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
            and (d / self.occ_glob.format(year=int(d.name))).exists()
        )

    def run(self) -> dict:
        years = self._years or self._discover_years()
        if not years:
            raise FileNotFoundError(
                f"No occurrence rasters found under {self.output_dir}")

        # ── 1. Load river geometries ──────────────────────────────────────
        log.info("Loading river geometries from %d shapefile(s) …",
                 len(self.shapefiles))
        rivers = load_river_geometries(
            self.shapefiles,
            name_field=self.name_field,
            buffer_m=self.buffer_m,
        )
        if not rivers:
            raise ValueError("No river geometries loaded.")
        log.info("Rivers: %s", list(rivers.keys()))

        fig_dir = self.output_dir / "figures"
        yr_str  = f"{years[0]}-{years[-1]}"

        # ── 2. Per-river annual metrics ────────────────────────────────────
        all_ts  = []
        log.info("Computing annual metrics for %d river(s) × %d year(s) …",
                 len(rivers), len(years))
        for river_name, geom in rivers.items():
            log.info("  → %s", river_name)
            ts = compute_river_annual_metrics(
                river_name, geom, self.output_dir, years,
                self.occ_glob, self.elev_glob,
                river_dir=self.output_dir / "rivers" / river_name,
            )
            all_ts.append(ts)

        ts_all = pd.concat(all_ts, ignore_index=True)
        ts_csv = self.output_dir / f"comparison_timeseries_{yr_str}.csv"
        ts_all.to_csv(ts_csv, index=False, float_format="%.4f")
        log.info("Time series → %s", ts_csv)

        # ── 3. Per-river lifespan (DTMC) ──────────────────────────────────
        log.info("Computing lifespan for each river …")
        lifespan_results: Dict[str, dict] = {}
        lifespan_rows    = []
        for river_name, geom in rivers.items():
            log.info("  DTMC → %s", river_name)
            res = compute_river_lifespan(
                river_name, geom, self.output_dir, years, self.occ_glob)
            lifespan_results[river_name] = res
            # Summary row (exclude large arrays)
            row = {k: v for k, v in res.items()
                   if k not in ("pii_map","lifespan_map","profile","P_matrix")}
            lifespan_rows.append(row)
            print(f"  {river_name:20s}: P_II={res.get('P_II',np.nan):.4f}  "
                  f"L={res.get('lifespan_yr',np.nan):.1f} yr")

        ls_df  = pd.DataFrame(lifespan_rows)
        ls_csv = self.output_dir / f"comparison_lifespan_{yr_str}.csv"
        ls_df.to_csv(ls_csv, index=False, float_format="%.4f")
        log.info("Lifespan summary → %s", ls_csv)

        # Print comparison table
        print("\n" + "="*65)
        print(f"  River Intertidal Comparison  ({yr_str})")
        print("="*65)
        print(ls_df[["river","P_II","P_II_ci_lo","P_II_ci_hi",
                      "lifespan_yr","lifespan_ci_lo"]].to_string(index=False))
        print("="*65 + "\n")

        # ── 4. Figures ─────────────────────────────────────────────────────
        if _PLOT:
            plot_river_area_timeseries(
                ts_all, fig_dir/f"11a_river_intertidal_area_{yr_str}.pdf")
            plot_river_elevation_timeseries(
                ts_all, fig_dir/f"11b_river_elevation_{yr_str}.pdf")
            plot_river_lifespan_comparison(
                lifespan_rows, fig_dir/f"11c_river_lifespan_comparison_{yr_str}.pdf")
            plot_river_lifespan_maps(
                lifespan_results, fig_dir/f"11d_river_lifespan_maps_{yr_str}.pdf")

        return {
            "rivers":           list(rivers.keys()),
            "timeseries":       ts_all,
            "lifespan_summary": ls_df,
            "lifespan_results": lifespan_results,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(
        description="Compare intertidal dynamics across river channels.")
    p.add_argument("--shapefiles", nargs="+", required=True,
                   help="River shapefile paths (one per river or one multi-feature file)")
    p.add_argument("--output-dir", required=True,
                   help="Root intertidal elevation output directory")
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--name-field", default=None,
                   help="Attribute column for river names in multi-feature shapefile")
    p.add_argument("--buffer-m", type=float, default=500.0)
    args = p.parse_args()

    comp = RiverIntertidalComparator(
        shapefiles=args.shapefiles,
        output_dir=args.output_dir,
        years=args.years,
        name_field=args.name_field,
        buffer_m=args.buffer_m,
    )
    comp.run()
