"""
intertidal_lifespan_markov.py
==============================
Nazwa Tahsin (EngD WUR HWM)

Markov chain-based lifespan and temporal persistence analysis of intertidal
zones from annual occurrence GeoTIFF stacks based on the processings

Theory
------
Each 30 m resolution pixel is assigned one of three annual states from the dry-season
occurrence raster:

    LAND        f = 0          (never wet in the year's dry-season images)
    INTERTIDAL  0 < f < 1      (sometimes wet, sometimes dry)
    RIVER       f = 1          (always wet — permanent channel / subtidal)

Across Y years this gives a state sequence of length Y per pixel.
A Discrete-Time Markov Chain (DTMC) is fitted by maximum likelihood:

    P̂_ij = N_ij / Σ_j N_ij

where N_ij is the number of observed year-on-year transitions from state i
to state j across all pixels and all consecutive year pairs.

Expected lifespan of the intertidal state
    L = 1 / (1 - P_II)

where P_II is the intertidal self-transition probability.
This is the mean number of annual time steps a pixel stays intertidal
before transitioning to land or river.

Confidence intervals on P_II (and therefore L) use the Wilson (1927) score
interval propagated through the lifespan formula.

Spatial lifespan map
    Pixel-wise P_II and L are estimated from a sliding 3×3 neighbourhood
    (or from the per-pixel sequence directly) to produce spatially
    distributed lifespan GeoTIFFs.

Outputs
-------
    transition_matrix.csv         3×3 global DTMC  (rows=from, cols=to)
    lifespan_stats.csv            global L, P_II, CI, state frequencies
    pii_map_YYYY-YYYY.tif         pixel-wise P_II spatial map
    lifespan_map_YYYY-YYYY.tif    pixel-wise lifespan (years)
    state_sequence_YYYY-YYYY.tif  3D stack (1 band per year) of state codes
                                  0=land, 1=intertidal, 2=river, 255=nodata
    figures/10_lifespan_*.pdf     publication figures

Elevation input
---------------
    Reads elevation_bt_YYYY.tif (Method A, Bishop-Taylor) by default.
    Falls back to elevation_median_YYYY.tif (Method B) if bt file absent.
    Set elev_method='median' to force Method B.

Dependencies
------------
    pip install numpy pandas rasterio scipy matplotlib tqdm
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from scipy.ndimage import uniform_filter

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

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── State codes ───────────────────────────────────────────────────────────────
LAND        = 0
INTERTIDAL  = 1
RIVER       = 2
NODATA      = 255

STATE_NAMES = {LAND: "Land", INTERTIDAL: "Intertidal", RIVER: "River"}
STATE_COLS  = {LAND: "#C8B89A", INTERTIDAL: "#F28E2B", RIVER: "#4E79A7"}

# ── Thresholds ────────────────────────────────────────────────────────────────
F_LAND_MAX   = 1e-4   # f ≤ this → land
F_RIVER_MIN  = 1 - 1e-4   # f ≥ this → river


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _savepdf(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def _write_tif(data: np.ndarray, path: Path, profile: dict,
               nodata=None, dtype="float32"):
    p = profile.copy()
    p.update(dtype=dtype, count=1, nodata=nodata,
             compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(dtype), 1)


def occurrence_to_state(occ: np.ndarray) -> np.ndarray:
    """
    Convert a float32 occurrence array [0,1] to uint8 state codes.
    NaN → NODATA (255).
    """
    state = np.full(occ.shape, NODATA, dtype=np.uint8)
    valid = np.isfinite(occ)
    state[valid & (occ <= F_LAND_MAX)]  = LAND
    state[valid & (occ >= F_RIVER_MIN)] = RIVER
    it_mask = valid & (occ > F_LAND_MAX) & (occ < F_RIVER_MIN)
    state[it_mask] = INTERTIDAL
    return state


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a proportion p = k/n.
    Returns (lower, upper) at z-sigma confidence (~95 % for z=1.96).
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return float(max(0, centre - margin)), float(min(1, centre + margin))


def lifespan_from_pii(pii: float) -> float:
    """Expected sojourn time (years) in intertidal: L = 1/(1-P_II)."""
    if pii >= 1.0:
        return np.inf
    return 1.0 / (1.0 - pii)


# ─────────────────────────────────────────────────────────────────────────────
# State stack loader
# ─────────────────────────────────────────────────────────────────────────────

def load_state_stack(
    output_dir: Path,
    years: List[int],
    glob_occ: str = "occurrence_dry_{year}.tif",
) -> Tuple[np.ndarray, List[int], dict]:
    """
    Load annual occurrence GeoTIFFs and convert to a state-code stack.

    Parameters
    ----------
    output_dir : root intertidal elevation output directory
    years      : list of years to load
    glob_occ   : filename template  (default: v3 naming)

    Returns
    -------
    stack   : (n_years, H, W) uint8 array of state codes
    valid_years : years successfully loaded
    profile : rasterio profile of the first valid raster (for output)
    """
    arrays      = []
    valid_years = []
    profile     = None

    it = tqdm(years, desc="Loading occ") if _TQDM else years
    for year in it:
        p = output_dir / str(year) / glob_occ.format(year=year)
        if not p.exists():
            log.warning("Missing: %s — skipping year %d", p, year)
            continue
        with rasterio.open(p) as src:
            occ = src.read(1).astype(np.float32)
            nd  = src.nodata
            if nd is not None:
                occ[np.isclose(occ, nd, atol=1e-5)] = np.nan
            if profile is None:
                profile = src.profile.copy()
        arrays.append(occurrence_to_state(occ))
        valid_years.append(year)

    if not arrays:
        raise FileNotFoundError(
            f"No occurrence GeoTIFFs found in {output_dir}. "
            f"Run IntertidalElevationProcessor first.")

    stack = np.stack(arrays, axis=0)   # (T, H, W)
    log.info("State stack: %d years × %d × %d px", *stack.shape)
    return stack, valid_years, profile


# ─────────────────────────────────────────────────────────────────────────────
# Global DTMC fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_global_dtmc(
    stack: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fit a 3-state DTMC (Land / Intertidal / River) from all pixel-year
    transitions in the state stack.

    Parameters
    ----------
    stack : (T, H, W) uint8 array of state codes

    Returns
    -------
    P    : (3,3) float64 transition probability matrix
           rows = from-state, cols = to-state
    df   : DataFrame with transition counts and row-normalised probabilities
    """
    T   = stack.shape[0]
    states = [LAND, INTERTIDAL, RIVER]
    N = np.zeros((3, 3), dtype=np.int64)   # count matrix

    for t in range(T - 1):
        from_s = stack[t]
        to_s   = stack[t + 1]
        valid  = (from_s != NODATA) & (to_s != NODATA)
        for i, fs in enumerate(states):
            for j, ts in enumerate(states):
                N[i, j] += int(((from_s == fs) & (to_s == ts) & valid).sum())

    # Row-normalise
    row_sums = N.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)   # avoid /0
    P = N / row_sums.astype(float)

    rows = []
    for i, fs in enumerate(states):
        for j, ts in enumerate(states):
            rows.append({
                "from_state": STATE_NAMES[fs],
                "to_state":   STATE_NAMES[ts],
                "N_obs":      int(N[i, j]),
                "prob":       round(float(P[i, j]), 6),
            })
    df = pd.DataFrame(rows)
    return P, df


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-wise DTMC and lifespan
# ─────────────────────────────────────────────────────────────────────────────

def compute_pixelwise_pii(stack: np.ndarray) -> np.ndarray:
    """
    Compute pixel-wise intertidal self-transition probability P_II.

    For each pixel, counts transitions I→I vs I→(anything) over all
    consecutive year pairs.  Returns float32 array in [0, 1], NaN where
    the pixel was never intertidal.

    This is vectorised — no pixel loop.
    """
    T = stack.shape[0]

    # Masks for the intertidal from-state in each consecutive pair
    n_ii   = np.zeros(stack.shape[1:], dtype=np.int32)   # I→I count
    n_i_   = np.zeros(stack.shape[1:], dtype=np.int32)   # I→any valid count

    for t in range(T - 1):
        from_s = stack[t]
        to_s   = stack[t + 1]
        valid  = (from_s != NODATA) & (to_s != NODATA)
        from_i = (from_s == INTERTIDAL) & valid
        n_i_  += from_i.astype(np.int32)
        n_ii  += ((from_s == INTERTIDAL) & (to_s == INTERTIDAL) & valid).astype(np.int32)

    with np.errstate(invalid="ignore", divide="ignore"):
        pii = np.where(n_i_ > 0, n_ii / n_i_.astype(float), np.nan).astype(np.float32)

    log.info("P_II computed: %d pixels with ≥1 intertidal transition",
             int((n_i_ > 0).sum()))
    return pii


def compute_lifespan_map(pii: np.ndarray) -> np.ndarray:
    """
    Convert pixel-wise P_II to expected lifespan in years.
    L = 1 / (1 − P_II).  Clipped to 50 years maximum.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.where(np.isfinite(pii) & (pii < 1.0),
                     1.0 / (1.0 - pii),
                     np.where(pii >= 1.0, 50.0, np.nan))
    return np.clip(L, 0, 50).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_annual_timeseries(
    stack:       np.ndarray,
    valid_years: List[int],
    elev_dir:    Optional[Path] = None,
    elev_glob:   str = "elevation_bt_{year}.tif",
    elev_method: str = "bt",
) -> pd.DataFrame:
    """
    Per-year time series:
      - n_land, n_intertidal, n_river  pixel counts
      - area_intertidal_km2
      - mean / std elevation of intertidal pixels (loaded from GeoTIFF if available)
      - state stability index = P(I→I) from each year-pair

    Parameters
    ----------
    stack       : (T, H, W) state stack
    valid_years : year labels for each layer
    elev_dir    : root elevation output dir (None → skip elevation stats)
    elev_glob   : elevation filename template
    """
    rows = []

    # Pixel area in km² — assume 30 m Landsat
    px_area_km2 = 30.0**2 / 1e6

    for t, year in enumerate(valid_years):
        layer  = stack[t]
        valid  = layer != NODATA
        n_tot  = valid.sum()
        n_land = int((layer[valid] == LAND).sum())
        n_it   = int((layer[valid] == INTERTIDAL).sum())
        n_riv  = int((layer[valid] == RIVER).sum())

        row = {
            "year":            year,
            "n_land":          n_land,
            "n_intertidal":    n_it,
            "n_river":         n_riv,
            "area_it_km2":     round(n_it * px_area_km2, 4),
            "frac_intertidal": round(n_it / max(n_tot, 1), 6),
            "elev_mean_m":     np.nan,
            "elev_std_m":      np.nan,
        }

        # Load elevation for intertidal pixels in this year
        if elev_dir is not None:
            ep = Path(elev_dir) / str(year) / elev_glob.format(year=year)
            # Fallback: if bt not found, try median (and vice versa)
            if not ep.exists():
                fallback = ("elevation_median_{year}.tif" if "bt" in elev_glob
                            else "elevation_bt_{year}.tif")
                ep_fb = Path(elev_dir) / str(year) / fallback.format(year=year)
                if ep_fb.exists():
                    ep = ep_fb
                    log.debug("Year %d: using fallback elevation %s", year, ep_fb.name)
            if ep.exists():
                with rasterio.open(ep) as src:
                    elev = src.read(1).astype(np.float32)
                it_mask  = (layer == INTERTIDAL) & valid
                it_match = it_mask & np.isfinite(elev) & (elev.shape == layer.shape)
                if it_match.any():
                    row["elev_mean_m"] = round(float(np.nanmean(elev[it_match])), 4)
                    row["elev_std_m"]  = round(float(np.nanstd(elev[it_match])), 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_transition_matrix(P: np.ndarray, N_df: pd.DataFrame,
                            out_path: Path) -> None:
    """Heatmap of the 3×3 DTMC transition matrix."""
    if not _PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Probability matrix heatmap
    ax = axes[0]
    im = ax.imshow(P, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Transition probability")
    labels = [STATE_NAMES[s] for s in [LAND, INTERTIDAL, RIVER]]
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("To state", fontsize=10); ax.set_ylabel("From state", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if P[i,j] > 0.6 else "black")
    ax.set_title("Transition probability matrix  P̂", fontsize=11)

    # Bar chart: intertidal transitions
    ax2 = axes[1]
    it_row = N_df[N_df["from_state"] == "Intertidal"]
    if not it_row.empty:
        bars = ax2.bar(it_row["to_state"], it_row["prob"],
                       color=[STATE_COLS[k] for k in [LAND, INTERTIDAL, RIVER]],
                       edgecolor="white", alpha=0.85, lw=0.5)
        for bar, (_, row) in zip(bars, it_row.iterrows()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{row['prob']:.3f}\n(n={row['N_obs']:,})",
                     ha="center", va="bottom", fontsize=8)
        pii = it_row[it_row["to_state"]=="Intertidal"]["prob"].values
        pii = float(pii[0]) if len(pii) else 0.0
        L   = lifespan_from_pii(pii)
        ax2.set_ylim(0, min(1.2, max(it_row["prob"].max() * 1.35, 0.5)))
        ax2.set_title(
            f"Transitions from Intertidal\n"
            f"P_II = {pii:.3f}  →  L = {L:.1f} yr",
            fontsize=11)
        ax2.set_ylabel("Probability", fontsize=10)
        ax2.set_xlabel("Destination state", fontsize=10)
        ax2.grid(True, lw=0.4, alpha=0.4, ls=":")

    fig.suptitle("Discrete-Time Markov Chain — GBM delta intertidal states",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


def plot_lifespan_map(pii: np.ndarray, lifespan: np.ndarray,
                       profile: dict, out_path: Path) -> None:
    """Two-panel spatial map: P_II and lifespan."""
    from affine import Affine
    if not _PLOT:
        return
    gt = Affine.from_gdal(
        profile["transform"].c,
        profile["transform"].a,
        0,
        profile["transform"].f,
        0,
        profile["transform"].e,
  )
    extent = array_bounds(pii.shape[0], pii.shape[1],gt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, cmap, vmin, vmax, label in [
        (axes[0], pii,      "P_II — Intertidal self-transition prob.",
         "YlOrRd", 0, 1,  "P(I→I)"),
        (axes[1], lifespan, "Expected intertidal lifespan (years)",
         "RdYlGn", 1, 20, "Lifespan (yr)"),
    ]:
        im = ax.imshow(data, origin="upper",
                       extent=[extent[0], extent[2], extent[1], extent[3]],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("Spatial lifespan of intertidal zones — GBM delta",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


def plot_annual_timeseries(ts_df: pd.DataFrame, out_path: Path,
                            elev_method: str = "bt") -> None:
    """4-panel annual time series: area, elevation, composition, fraction."""
    if not _PLOT or ts_df.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # Label depends on which elevation method was used
    elev_label_map = {
        "bt":     "Mean elevation (m MSL) [Bishop-Taylor]",
        "median": "Mean elevation (m MSL) [waterline median]",
    }
    elev_label_map2 = {"bt": "Mean elevation (m MSL) [Bishop-Taylor]",
                       "median": "Mean elevation (m MSL) [waterline median]"}
    panels = [
        ("area_it_km2",     "Intertidal area (km²)",                              "#F28E2B"),
        ("elev_mean_m",     elev_label_map2.get(elev_method, "Mean elev (m MSL)"), "#E15759"),
        ("frac_intertidal", "Intertidal fraction of scene",      "#59A14F"),
        ("n_intertidal",    "Intertidal pixel count",            "#4E79A7"),
    ]
    letters = "abcd"
    for idx, (col, ylabel, col_c) in enumerate(panels):
        ax = axes[idx]
        if col in ts_df.columns:
            ax.plot(ts_df["year"], ts_df[col],
                    color=col_c, marker="o", ms=5, lw=1.4, clip_on=False)
            # Trend line
            mask = ts_df[col].notna()
            if mask.sum() >= 3:
                z = np.polyfit(ts_df.loc[mask,"year"], ts_df.loc[mask,col], 1)
                ax.plot(ts_df["year"], np.polyval(z, ts_df["year"]),
                        "--", color=col_c, lw=0.9, alpha=0.5,
                        label=f"trend {z[0]:+.3g}/yr")
                ax.legend(fontsize=8, frameon=False)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, lw=0.4, alpha=0.4, ls=":")
        ax.tick_params(direction="in", length=3, labelsize=9)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.text(0.02, 0.95, f"({letters[idx]})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

    axes[-1].set_xlabel("Year", fontsize=10)
    fig.suptitle("Annual intertidal time series — GBM delta",
                 fontsize=12, fontweight="medium")
    fig.tight_layout()
    _savepdf(fig, out_path)


def plot_state_transition_network(P: np.ndarray, out_path: Path) -> None:
    """
    Bubble-and-arrow state transition diagram.
    """
    if not _PLOT:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal"); ax.axis("off")

    pos = {LAND: np.array([0.2, 0.5]),
           INTERTIDAL: np.array([0.5, 0.85]),
           RIVER: np.array([0.8, 0.5])}

    # Draw nodes
    for state, xy in pos.items():
        circle = plt.Circle(xy, 0.10, color=STATE_COLS[state],
                             ec="white", lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(*xy, STATE_NAMES[state], ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    # Draw arrows
    states_ord = [LAND, INTERTIDAL, RIVER]
    for i, fs in enumerate(states_ord):
        for j, ts in enumerate(states_ord):
            p = P[i, j]
            if p < 0.01 or (i == j):
                continue
            start = pos[fs]
            end   = pos[ts]
            dx, dy = end - start
            norm   = np.sqrt(dx**2 + dy**2)
            offset = 0.12
            s = start + offset * np.array([dx, dy]) / norm
            e = end   - offset * np.array([dx, dy]) / norm
            ax.annotate("", xy=e, xytext=s,
                        arrowprops=dict(arrowstyle="-|>", color="grey",
                                        lw=max(0.5, p*4),
                                        mutation_scale=12))
            mid = (s + e) / 2
            ax.text(mid[0], mid[1], f"{p:.2f}",
                    ha="center", va="center", fontsize=8, color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

        # Self-loop for diagonal
        p_self = P[i, i]
        if p_self >= 0.01:
            xy = pos[fs]
            loop = plt.Circle(xy + np.array([0.0, 0.17]), 0.06,
                               fill=False, edgecolor=STATE_COLS[fs],
                               lw=max(0.8, p_self*3), zorder=2)
            ax.add_patch(loop)
            ax.text(xy[0], xy[1]+0.26, f"{p_self:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=STATE_COLS[fs])

    # Lifespan annotation
    pii = P[INTERTIDAL, INTERTIDAL]
    L   = lifespan_from_pii(pii)
    ax.text(0.5, 0.10,
            f"Intertidal  P_II = {pii:.3f}  →  L = {L:.1f} yr",
            transform=ax.transAxes, ha="center", fontsize=11,
            bbox=dict(facecolor="#FFF7E6", edgecolor="#F28E2B",
                      boxstyle="round,pad=0.4"))

    fig.suptitle("State Transition Network — GBM delta", fontsize=12)
    fig.tight_layout()
    _savepdf(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IntertidalLifespanAnalyser:
    """
    End-to-end Markov chain lifespan analysis.

    Parameters
    ----------
    output_dir      root intertidal elevation directory (contains YYYY/ subfolders)
    years           list of years to include  (None = auto-detect)
    occ_glob        occurrence filename template  (default: v3 naming)
    elev_glob       elevation filename template   (default: v3 median)
    overwrite       recompute even if outputs exist
    """

    def __init__(
        self,
        output_dir: str | Path,
        years: Optional[List[int]] = None,
        occ_glob:   str = "occurrence_dry_{year}.tif",
        elev_glob:  str = "elevation_bt_{year}.tif",
        elev_method: str = "bt",   # "bt" = Bishop-Taylor (primary); "median" = waterline
        overwrite:  bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.occ_glob    = occ_glob
        self.elev_glob   = elev_glob
        self.elev_method = elev_method
        self.overwrite   = overwrite
        self._years      = years

    def _discover_years(self) -> List[int]:
        years = sorted(
            int(d.name) for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
            and (d / self.occ_glob.format(year=int(d.name))).exists()
        )
        log.info("Auto-detected %d years: %s", len(years), years)
        return years

    def run(self) -> dict:
        years = self._years or self._discover_years()
        if not years:
            raise FileNotFoundError(
                f"No occurrence GeoTIFFs found under {self.output_dir}")

        yr_str = f"{years[0]}-{years[-1]}"
        fig_dir = self.output_dir / "figures"

        # ── Load state stack ──────────────────────────────────────────────
        log.info("Loading state stack for %d years …", len(years))
        stack, valid_years, profile = load_state_stack(
            self.output_dir, years, self.occ_glob)

        # ── Global DTMC ───────────────────────────────────────────────────
        log.info("Fitting global DTMC …")
        P, N_df = fit_global_dtmc(stack)
        pii_global = float(P[INTERTIDAL, INTERTIDAL])
        L_global   = lifespan_from_pii(pii_global)
        ci_lo, ci_hi = wilson_ci(
            int(N_df[(N_df["from_state"]=="Intertidal") &
                     (N_df["to_state"]  =="Intertidal")]["N_obs"].values[0]),
            int(N_df[N_df["from_state"]=="Intertidal"]["N_obs"].sum()),
        )
        L_ci = (lifespan_from_pii(ci_lo), lifespan_from_pii(ci_hi))

        print("\n" + "="*55)
        print(f"  DTMC Results  ({yr_str})")
        print("="*55)
        print(f"  P_II (global)        : {pii_global:.4f}")
        print(f"  95% Wilson CI        : [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Mean lifespan L      : {L_global:.2f} yr")
        print(f"  Lifespan 95% CI      : [{L_ci[0]:.2f}, {L_ci[1]:.2f}] yr")
        print("  Transition matrix:")
        labels = ["Land", "Inter.", "River"]
        print("         " + "  ".join(f"{l:>7}" for l in labels))
        for i, lab in enumerate(labels):
            print(f"  {lab:7s}" + "  ".join(f"{P[i,j]:7.4f}" for j in range(3)))
        print("="*55 + "\n")

        # ── Save global outputs ───────────────────────────────────────────
        N_df.to_csv(self.output_dir / "transition_matrix.csv", index=False)

        state_freq = {}
        for t in range(len(valid_years)):
            s = stack[t]
            v = s != NODATA
            for code, name in STATE_NAMES.items():
                state_freq.setdefault(name, []).append(
                    int((s[v] == code).sum()) / max(int(v.sum()), 1))

        stats = pd.DataFrame([{
            "years_analysed":   len(valid_years),
            "year_start":       valid_years[0],
            "year_end":         valid_years[-1],
            "P_II_global":      round(pii_global, 6),
            "P_II_ci_lo":       round(ci_lo, 6),
            "P_II_ci_hi":       round(ci_hi, 6),
            "lifespan_mean_yr": round(L_global, 3),
            "lifespan_ci_lo":   round(L_ci[0], 3),
            "lifespan_ci_hi":   round(L_ci[1], 3) if not np.isinf(L_ci[1]) else 999,
            "mean_frac_land":       round(np.mean(state_freq.get("Land",  [0])), 4),
            "mean_frac_intertidal": round(np.mean(state_freq.get("Intertidal", [0])), 4),
            "mean_frac_river":      round(np.mean(state_freq.get("River", [0])), 4),
        }])
        stats.to_csv(self.output_dir / "lifespan_stats.csv", index=False)
        log.info("Stats → %s", self.output_dir / "lifespan_stats.csv")

        # ── Pixel-wise P_II and lifespan maps ─────────────────────────────
        log.info("Computing pixel-wise P_II map …")
        pii_px   = compute_pixelwise_pii(stack)
        lifespan = compute_lifespan_map(pii_px)

        _write_tif(pii_px,   self.output_dir / f"pii_map_{yr_str}.tif",
                   profile, nodata=np.nan)
        _write_tif(lifespan, self.output_dir / f"lifespan_map_{yr_str}.tif",
                   profile, nodata=np.nan)
        log.info("Lifespan GeoTIFFs written.")

        # ── Annual time series ────────────────────────────────────────────
        log.info("Building annual time series …")
        ts_df = compute_annual_timeseries(
            stack, valid_years,
            elev_dir=self.output_dir, elev_glob=self.elev_glob,
            elev_method=self.elev_method)
        ts_df.to_csv(self.output_dir / f"annual_timeseries_{yr_str}.csv",
                     index=False, float_format="%.4f")
        log.info("Time series → %s", self.output_dir / f"annual_timeseries_{yr_str}.csv")

        # ── Figures ───────────────────────────────────────────────────────
        if _PLOT:
            log.info("Generating figures …")
            plot_transition_matrix(
                P, N_df,
                fig_dir / f"10a_transition_matrix_{yr_str}.pdf")
            plot_state_transition_network(
                P,
                fig_dir / f"10b_transition_network_{yr_str}.pdf")
            plot_lifespan_map(
                pii_px, lifespan, profile,
                fig_dir / f"10c_lifespan_map_{yr_str}.pdf")
            plot_annual_timeseries(
                ts_df,
                fig_dir / f"10d_annual_timeseries_{yr_str}.pdf",
                elev_method=self.elev_method)

        return {
            "P": P, "N_df": N_df, "stats": stats,
            "pii_map": pii_px, "lifespan_map": lifespan,
            "timeseries": ts_df, "years": valid_years,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(
        description="Markov chain intertidal lifespan analysis.")
    p.add_argument("--output-dir", required=True,
                   help="Root intertidal elevation output directory")
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    analyser = IntertidalLifespanAnalyser(
        output_dir=args.output_dir,
        years=args.years,
        overwrite=args.overwrite,
    )
    analyser.run()
