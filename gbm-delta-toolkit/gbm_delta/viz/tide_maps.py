# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:42:00 2026

@author: tahsi001
"""
# -*- coding: utf-8 -*-
"""
tide_maps.py  (AGU/JGR edition)
===============================
Per-year map panels (land grey / river white / intertidal blue) with a
tide diagram under each map: sensor-captured WL range (blue bar) vs
in-situ tidal range (rust double arrow).

AGU sizing
----------
* Full-page width 7.0 in (17.8 cm); figure created AT print size,
  included with width=\\textwidth in LaTeX -> fonts print 1:1.
* All text >= 12 pt (base 12, titles 13).
* <= 3 years -> single row; >= 4 years -> 2-column grid (2 x 2).

Inputs
------
    intertidal_filtered_{year}.tif   classes: 0 land / 1 river / 2 intertidal
    images CSV                       scene water levels per station
    observed WL CSV                  in-situ water levels (datetime index)
    optional elevation raster        intertidal_elev_{year}.tif — shades
                                     the intertidal class blue->green.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

log = logging.getLogger(__name__)

# ── AGU page geometry & type sizes ───────────────────────────────────────────
AGU_FULL_IN = 7.0          # full page width (inches)
FS_BASE   = 12             # minimum text size everywhere
FS_TITLE  = 13
FS_TICK   = 12
FS_ANNOT  = 12

# palette (ocean/earth)
LAND_GREY       = "#d9d9d9"
WATER           = "white"
INTERTIDAL_FLAT = "#3d7ea6"
BAR_BLUE        = "#0f4c5c"
ARROW_RUST      = "#8c6d46"
ELEV_CMAP = LinearSegmentedColormap.from_list(
    "blue_green", ["#0f4c5c", "#3d7ea6", "#4a9b8c", "#7fae7a"])


# ─────────────────────────────────────────────────────────────────────────────
# data helpers
# ─────────────────────────────────────────────────────────────────────────────
def scene_captured_range(images_csv: str, year: int
                         ) -> Tuple[float, float, int]:
    """
    Water-level range captured by the satellite scenes of one year.
    The images CSV has a date column and one WL column per station; the
    WL is averaged across stations per scene, then the year's min-max
    is taken. Returns (wl_min, wl_max, n_scenes).
    """
    df = pd.read_csv(images_csv, sep=None, engine="python")
    tcol = next(c for c in df.columns
                if any(k in c.lower() for k in ("date", "time", "scene")))
    try:
        df[tcol] = pd.to_datetime(df[tcol], dayfirst=True)
    except ValueError:
        df[tcol] = pd.to_datetime(df[tcol], format="mixed", dayfirst=True)
    df["year"] = df[tcol].dt.year
    sub = df[df["year"] == year]
    wl_cols = [c for c in df.columns
               if c not in (tcol, "year")
               and pd.api.types.is_float_dtype(df[c])]
    if sub.empty or not wl_cols:
        return np.nan, np.nan, 0
    scene_mean = sub[wl_cols].mean(axis=1).dropna()
    if scene_mean.empty:
        return np.nan, np.nan, 0
    return float(scene_mean.min()), float(scene_mean.max()), len(scene_mean)


def in_situ_range(obs_wl_csv: str, year: int,
                  station: Optional[str] = None) -> Tuple[float, float]:
    """
    Annual in-situ tidal range from the observed WL CSV
    (datetime index, station columns). All stations or one named.
    """
    df = pd.read_csv(obs_wl_csv, sep=None, engine="python",
                     index_col=0, parse_dates=True)
    if station is not None and station in df.columns:
        df = df[[station]]
    sub = df[df.index.year == year].dropna(how="all")
    if sub.empty:
        return np.nan, np.nan
    return float(np.nanmin(sub.to_numpy())), float(np.nanmax(sub.to_numpy()))


# ─────────────────────────────────────────────────────────────────────────────
# one map panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_map(ax, raster_path: Path,
                elev_path: Optional[Path]) -> None:
    """Land grey / river white / intertidal blue (or elevation-shaded)."""
    import rasterio
    with rasterio.open(raster_path) as src:
        cls = src.read(1)
        transform = src.transform
    rgb = np.zeros((*cls.shape, 3), dtype=np.float32)
    rgb[...] = to_rgb(LAND_GREY)                       # land base
    rgb[cls == 1] = to_rgb(WATER)                       # river

    m_int = cls == 2
    shaded = False
    if elev_path is not None and elev_path.exists():
        with rasterio.open(elev_path) as esrc:
            elev = esrc.read(1).astype(np.float32)
        h = min(elev.shape[0], cls.shape[0])
        w = min(elev.shape[1], cls.shape[1])
        ev = np.where(m_int[:h, :w], elev[:h, :w], np.nan)
        if np.isfinite(ev).sum() > 0:
            vmin, vmax = np.nanpercentile(ev[np.isfinite(ev)], [2, 98])
            normed = np.clip((ev - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            sel = m_int[:h, :w]
            rgb[:h, :w][sel] = ELEV_CMAP(normed)[..., :3][sel]
            shaded = True
    if not shaded:
        rgb[m_int] = to_rgb(INTERTIDAL_FLAT)

    ext = rasterio.transform.array_bounds(cls.shape[0], cls.shape[1],
                                          transform)
    ax.imshow(rgb, extent=[ext[0], ext[2], ext[1], ext[3]],
              interpolation="nearest")
    ax.set_xlim(ext[0], ext[2]); ax.set_ylim(ext[1], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#777777"); s.set_linewidth(0.6)

    # manual scale bar (bottom-left), 12 pt label
    x0 = ext[0] + (ext[2] - ext[0]) * 0.05
    y0 = ext[1] + (ext[1] - ext[3]) * 0.05
    L = (ext[2] - ext[0]) * 0.15
    ax.plot([x0, x0 + L], [y0, y0], "-", color="#333333", lw=2,
            zorder=5)
    ax.annotate(f"{L / 1000:.0f} km", (x0 + L / 2, y0),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=FS_ANNOT, color="#333333",
                zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# tide diagram (all text >= 12 pt; scene counts go in the caption)
# ─────────────────────────────────────────────────────────────────────────────
def _tide_diagram(ax, captured: Tuple[float, float, int],
                  in_situ: Tuple[float, float]) -> None:
    """Blue bar = sensor-captured WL range; rust arrow = in-situ range."""
    cap_lo, cap_hi, _ = captured
    ins_lo, ins_hi = in_situ
    if not np.isfinite([cap_lo, cap_hi, ins_lo, ins_hi]).all():
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                fontsize=FS_BASE, transform=ax.transAxes)
        return
    lo = min(cap_lo, ins_lo) - 0.4
    hi = max(cap_hi, ins_hi) + 0.4
    ax.set_xlim(lo, hi); ax.set_ylim(-0.9, 1.1)
    ax.set_yticks([])
    # in-situ range arrow
    ax.annotate("", xy=(ins_hi, 0.55), xytext=(ins_lo, 0.55),
                arrowprops=dict(arrowstyle="<->", color=ARROW_RUST,
                                lw=1.8))
    # sensor-captured range bar
    ax.barh(-0.30, cap_hi - cap_lo, left=cap_lo, height=0.75,
            color=BAR_BLUE, edgecolor="none")
    # labels
    ax.text((ins_lo + ins_hi) / 2, 0.85,
            f"{ins_lo:.1f} to {ins_hi:.1f} m", ha="center",
            fontsize=FS_ANNOT, color=ARROW_RUST)
    ax.text((cap_lo + cap_hi) / 2, -0.65,
            f"{cap_lo:.1f} to {cap_hi:.1f} m", ha="center",
            fontsize=FS_ANNOT, color=BAR_BLUE)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)


# ─────────────────────────────────────────────────────────────────────────────
# full figure
# ─────────────────────────────────────────────────────────────────────────────
def make_tide_map_series(
        years: Sequence[int],
        intertidal_dir: str,
        images_csv: str,
        obs_wl_csv: str,
        out: str = "tide_map_series.png",
        glob_pattern: str = "intertidal_filtered_{year}.tif",
        elev_glob: Optional[str] = None,   # e.g. "intertidal_elev_{year}.tif"
        station: Optional[str] = None,
        hspace: float = 0.55,              # map<->tide and cell-pair gap
        wspace: float = 0.10,
        suptitle: str = "Intertidal extent and tidal representativeness"
        ) -> plt.Figure:
    """
    years          : panel years (e.g. [1996, 2004, 2014, 2018])
    intertidal_dir : folder with intertidal_filtered_{year}.tif
    images_csv     : scene x station water levels (captured range)
    obs_wl_csv     : in-situ observed water levels (tidal range)
    elev_glob      : optional per-year elevation raster to shade the
                     intertidal class (blue->green ramp)
    hspace/wspace  : GridSpec spacing knobs (lower hspace = tighter
                     map-to-tide stacking)
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": FS_BASE, "font.family": "sans-serif",
        "axes.titlesize": FS_TITLE, "axes.labelsize": FS_BASE,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_BASE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    n = len(years)
    if n <= 3:
        ncols, nrows = n, 1
    else:
        ncols = 2
        nrows = (n + 1) // 2              # 4 years -> 2x2 grid

    panel_h, tide_h = 3.1, 0.9            # inches per panel
    fig_w = AGU_FULL_IN
    fig_h = nrows * (panel_h + tide_h) + 1.6   # suptitle + legend

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ONE gridspec for all axes: map/tide rows interleaved, shared spacing
    gs = fig.add_gridspec(
        2 * nrows, ncols,
        height_ratios=[panel_h, tide_h] * nrows,
        hspace=hspace, wspace=wspace,
        left=0.04, right=0.97, top=0.93, bottom=0.12)

    for idx, yr in enumerate(years):
        row, col = divmod(idx, ncols)
        raster = Path(intertidal_dir) / glob_pattern.format(year=yr)
        if not raster.exists():
            log.warning("missing %s", raster)
            continue
        elev = (Path(intertidal_dir) / elev_glob.format(year=yr)
                if elev_glob else None)

        ax_map = fig.add_subplot(gs[2 * row, col])           # map
        _render_map(ax_map, raster, elev)
        ax_map.set_title(str(yr), fontsize=FS_TITLE, pad=6)

        ax_tide = fig.add_subplot(gs[2 * row + 1, col])      # tide
        cap = scene_captured_range(images_csv, yr)
        ins = in_situ_range(obs_wl_csv, yr, station=station)
        _tide_diagram(ax_tide, cap, ins)
        ax_tide.set_xlabel("Water level (m)", fontsize=FS_BASE,
                           labelpad=2)
    '''
    # legend (>= 12 pt)
    handles = [
        mpatches.Patch(color=LAND_GREY, label="Land"),
        mpatches.Patch(facecolor=WATER, edgecolor="#999999",
                       label="River"),
        mpatches.Patch(color=INTERTIDAL_FLAT, label="Intertidal"),
        Line2D([], [], color=BAR_BLUE, lw=6, solid_capstyle="butt",
               label="Sensor captured range"),
        Line2D([], [], color=ARROW_RUST, lw=1.8, marker="<",
               markersize=7, label="In-situ tidal range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncols,
               fontsize=FS_BASE, frameon=False,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.2,
               handlelength=1.6, handletextpad=0.5)
    '''
    fig.suptitle(suptitle, fontsize=FS_TITLE + 1, y=0.995)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"saved {out}")
    return fig