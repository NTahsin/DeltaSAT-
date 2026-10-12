"""
channel_metrics_visualizer.py
==============================
Publication-grade reproduction of Figure-7 from the Meghna intertidal
morphodynamics paper:

    (left)  map of Main Channel & Side Channel polygons over a base map
    (mid)   four stacked time-series for a *reference* ROI (the aggregate):
                Migration (mm/yr), Area (km²), Elevation (m MSL), Lifespan (yr)
    (right) two boxed panels — "Main Channel" and "Side Channel(s)" —
            each carrying the same four metrics in miniature.

Input is the tidy DataFrame written by ``channel_metrics.ChannelMetricsAnalyzer``
(``channel_metrics_all_rois.csv``) plus the ROI geometries used to derive it.

Design notes
------------
 • Colour palette matches the existing figures in the paper (blue-teal family).
 • Uses matplotlib only — no seaborn dependency.
 • Output: PNG (raster) + PDF (vector) at 300 dpi.
 • Fonts and line weights follow the house style set in
   tidal_datum_visualizer._apply_pub_style().

Dependencies
------------
    pip install numpy pandas matplotlib geopandas contextily
    (contextily is optional; falls back to a plain white background)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

try:
    import contextily as cx
    _HAS_CX = True
except ImportError:
    _HAS_CX = False

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Styling — kept identical to tidal_datum_visualizer._apply_pub_style()
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pub_style() -> None:
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.linewidth":   0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "savefig.dpi":      300,
        "figure.dpi":       120,
    })


# Colour palette derived from Fig. 7 of the PDF (two blues + neutrals)
PALETTE = {
    "Main Channel":   "#8ECAE6",   # light blue
    "Side Channel":   "#264653",   # dark teal
    "Side Channels":  "#264653",
    "Side Channel(s)": "#264653",
    "_default":       "#4E79A7",
    "grid":           "#D0D4DB",
    "text":           "#1F2A33",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FigureConfig:
    """Controls the visual output of :class:`ChannelMetricsVisualizer`."""
    metric_order: List[str] = field(default_factory=lambda: [
        "migration_mean_mm_yr",
        "intertidal_area_km2",
        "intertidal_elevation_m",
        "lifespan_years",
    ])
    metric_labels: Dict[str, str] = field(default_factory=lambda: {
        "migration_mean_mm_yr":    "Migration (mm/yr)",
        "migration_mean_m_yr":     "Migration (m/yr)",
        "intertidal_area_km2":     "Area (km²)",
        "intertidal_elevation_m":  "Elevation (m MSL)",
        "lifespan_years":          "Lifespan (yr)",
    })
    figsize:   Tuple[float, float] = (14.0, 9.0)
    dpi:       int   = 300
    reference_roi: Optional[str] = None   # ROI for the large central panels
    smooth_window: int  = 3               # rolling mean for the "smoothed" look
    basemap_provider: Optional[str] = "OpenStreetMap.Mapnik"
    main_channel_names: Sequence[str] = ("Main Channel", "main_channel", "Main")
    side_channel_names: Sequence[str] = (
        "Side Channel", "Side Channels", "Side Channel(s)",
        "side_channel", "Side", "side")


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling(series: pd.Series, win: int) -> pd.Series:
    if win <= 1:
        return series
    return series.rolling(win, center=True, min_periods=1).mean()


def _colour_for(roi: str) -> str:
    return PALETTE.get(roi, PALETTE["_default"])


def _classify_rois(df_rois: Sequence[str],
                   cfg: FigureConfig) -> Tuple[List[str], List[str]]:
    """
    Split ROI names into (main, side) buckets based on the configured aliases.
    Unknown names go to 'side' so they still get plotted.
    """
    main, side = [], []
    for r in df_rois:
        rl = str(r).lower().strip()
        if any(rl == m.lower() or m.lower() in rl for m in cfg.main_channel_names):
            main.append(r)
        else:
            side.append(r)
    return main, side


# ─────────────────────────────────────────────────────────────────────────────
# Low-level plotting primitives
# ─────────────────────────────────────────────────────────────────────────────

def _plot_metric(ax: plt.Axes,
                 df: pd.DataFrame,
                 metric: str,
                 label: str,
                 colour: str,
                 smooth: int = 1,
                 *,
                 show_xlabel: bool = False,
                 title: Optional[str] = None,
                 compact: bool = False) -> None:
    df = df.sort_values("year")
    if metric not in df.columns or df[metric].dropna().empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=8)
    else:
        y_smooth = _rolling(df[metric], smooth)
        # Faint raw line under the smoothed line for context
        ax.plot(df["year"], df[metric], color=colour, alpha=0.25,
                lw=0.8, zorder=1)
        ax.plot(df["year"], y_smooth, color=colour, lw=1.8, zorder=2)
        ax.scatter(df["year"], df[metric], s=10, color=colour,
                   edgecolor="white", linewidth=0.4, zorder=3)

    # Compact panels: smaller label, moved further out, shorter units
    if compact:
        short = {
            "Migration (mm/yr)":   "Migration\n(mm/yr)",
            "Migration (m/yr)":    "Migration\n(m/yr)",
            "Area (km²)":          "Area\n(km²)",
            "Elevation (m MSL)":   "Elev.\n(m MSL)",
            "Lifespan (yr)":       "Lifespan\n(yr)",
        }
        ax.set_ylabel(short.get(label, label), fontsize=7, labelpad=2)
    else:
        ax.set_ylabel(label, fontsize=10)

    ax.grid(True, linestyle=":", linewidth=0.5, color=PALETTE["grid"],
            alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both",
                   labelsize=7 if compact else 9,
                   pad=1 if compact else 3)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Year", fontsize=8 if compact else 10)
    if title:
        ax.set_title(title, fontsize=10, pad=4)

    # Tidy y-axis
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3 if compact else 4))
    # Keep tick values from taking up half the panel
    if compact:
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 4))


def _draw_roi_map(ax: plt.Axes,
                  rois: "gpd.GeoDataFrame",
                  cfg: FigureConfig) -> None:
    """Base-mapped polygon panel on the left."""
    if not _HAS_GPD or rois is None or rois.empty:
        ax.text(0.5, 0.5, "ROI map unavailable", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.axis("off")
        return

    # Web Mercator for base maps
    plot_rois = rois.to_crs(3857) if rois.crs and _HAS_CX else rois.copy()

    main_names, side_names = _classify_rois(list(plot_rois["name"]), cfg)
    side_gdf = plot_rois[plot_rois["name"].isin(side_names)]
    main_gdf = plot_rois[plot_rois["name"].isin(main_names)]

    if not side_gdf.empty:
        side_gdf.plot(ax=ax, facecolor=PALETTE["Side Channel"],
                      edgecolor=PALETTE["Side Channel"], alpha=0.80, lw=0.8,
                      label="Side Channel(s)")
    if not main_gdf.empty:
        main_gdf.plot(ax=ax, facecolor=PALETTE["Main Channel"],
                      edgecolor=PALETTE["Main Channel"], alpha=0.85, lw=0.8,
                      label="Main Channel")

    # Basemap (optional)
    if _HAS_CX and cfg.basemap_provider:
        try:
            provider = cfg.basemap_provider
            if isinstance(provider, str):
                parts = provider.split(".")
                prov_obj = cx.providers
                for p in parts:
                    prov_obj = getattr(prov_obj, p)
                cx.add_basemap(ax, source=prov_obj, attribution_size=6)
            else:
                cx.add_basemap(ax, source=provider, attribution_size=6)
        except Exception as exc:
            log.warning("Basemap unavailable (%s) — continuing without.", exc)

    # Custom legend matches fig. 7
    handles = [
        Patch(facecolor=PALETTE["Main Channel"],  label="Main Channel",
              edgecolor="none"),
        Patch(facecolor=PALETTE["Side Channel"],  label="Side Channel(s)",
              edgecolor="none"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#888", fontsize=9)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#888")
        s.set_linewidth(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure class
# ─────────────────────────────────────────────────────────────────────────────

class ChannelMetricsVisualizer:
    """
    Assembles the Fig. 7 composite from a metrics DataFrame and ROI GeoDataFrame.
    """

    def __init__(
        self,
        metrics: Union[PathLike, pd.DataFrame],
        rois: Optional[Union[PathLike, "gpd.GeoDataFrame"]] = None,
        cfg: Optional[FigureConfig] = None,
    ) -> None:
        _apply_pub_style()
        self.cfg = cfg or FigureConfig()

        # Load metrics
        if isinstance(metrics, (str, Path)):
            self.df = pd.read_csv(metrics)
        else:
            self.df = metrics.copy()
        if "roi" not in self.df.columns:
            self.df["roi"] = "ALL"

        # Pick reference ROI for the central column
        if self.cfg.reference_roi is None:
            pick = next(
                (r for r in self.df["roi"].unique()
                 if any(m.lower() in str(r).lower()
                        for m in self.cfg.main_channel_names)),
                self.df["roi"].iloc[0],
            )
            self.cfg.reference_roi = pick

        # Load ROIs
        self.rois = None
        if rois is not None and _HAS_GPD:
            if isinstance(rois, (str, Path)):
                from gbm_delta.channel.metrics import load_rois   # defer import
                self.rois = load_rois(rois)
            else:
                self.rois = rois.copy()

    # ── compose the figure ──────────────────────────────────────────────
    def build(self) -> plt.Figure:
        """
        Layout:
            col 0  → ROI map (spans full height)
            col 1  → Main stacked time-series for reference ROI (full height)
            col 2  → TWO boxed sub-panels:
                       top  =  Main Channel   (4 metrics stacked)
                       bot  =  Side Channel(s) (4 metrics stacked)
        """
        cfg = self.cfg
        n   = len(cfg.metric_order)
        fig = plt.figure(figsize=cfg.figsize)

        # Outer grid: 1 row × 3 columns.  Equal height; column 2 will be
        # sub-divided into two labelled blocks.
        outer = GridSpec(
            nrows=1, ncols=3,
            width_ratios=[0.85, 1.15, 1.00],
            figure=fig,
            wspace=0.38,
            left=0.045, right=0.975, top=0.91, bottom=0.07,
        )

        # ── (A) ROI map ─────────────────────────────────────────────────
        ax_map = fig.add_subplot(outer[0, 0])
        _draw_roi_map(ax_map, self.rois, cfg)
        ax_map.set_title("Study reach", fontsize=11, pad=6, loc="left")

        # ── (B) Central stacked time-series for reference ROI ───────────
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        mid = GridSpecFromSubplotSpec(
            nrows=n, ncols=1, subplot_spec=outer[0, 1], hspace=0.25,
        )
        ref    = self.cfg.reference_roi
        ref_df = self.df[self.df["roi"] == ref].copy()
        ref_col = _colour_for(ref)
        for i, metric in enumerate(cfg.metric_order):
            ax = fig.add_subplot(mid[i, 0])
            _plot_metric(
                ax, ref_df, metric=metric,
                label=cfg.metric_labels.get(metric, metric),
                colour=ref_col,
                smooth=cfg.smooth_window,
                show_xlabel=(i == n - 1),
                title=ref if i == 0 else None,
            )

        # ── (C) Right column — two labelled boxes ───────────────────────
        right = GridSpecFromSubplotSpec(
            nrows=2, ncols=1, subplot_spec=outer[0, 2],
            hspace=0.32,
        )
        roi_names = list(self.df["roi"].unique())
        main_names, side_names = _classify_rois(roi_names, cfg)
        if not main_names and not side_names:
            main_names, side_names = roi_names[:1], roi_names[1:]

        self._draw_roi_block(fig, right[0, 0], main_names or roi_names,
                             title="Main Channel")
        self._draw_roi_block(fig, right[1, 0], side_names or main_names,
                             title="Side Channel(s)")

        fig.suptitle(
            "Channel migration, intertidal area, elevation and lifespan "
            f"({int(self.df['year'].min())}–{int(self.df['year'].max())})",
            fontsize=12, y=0.98,
        )
        return fig

    def _draw_roi_block(
        self,
        fig: plt.Figure,
        subplot_spec,
        roi_names: Sequence[str],
        title: str,
    ) -> None:
        """One titled, boxed block carrying all 4 metrics for one ROI group."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        cfg = self.cfg
        metrics = list(cfg.metric_order)
        inner = GridSpecFromSubplotSpec(
            nrows=len(metrics), ncols=1,
            subplot_spec=subplot_spec, hspace=0.75,
        )
        axes: List[plt.Axes] = []
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(inner[i, 0])
            for roi in roi_names:
                sub_df = self.df[self.df["roi"] == roi]
                _plot_metric(
                    ax, sub_df, metric=metric,
                    label=cfg.metric_labels.get(metric, metric),
                    colour=_colour_for(roi),
                    smooth=cfg.smooth_window,
                    show_xlabel=(i == len(metrics) - 1),
                    title=None,
                    compact=True,
                )
            axes.append(ax)

        # Multi-ROI legend inside the block, if needed
        if len(roi_names) > 1:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=_colour_for(r), lw=1.8, label=r)
                       for r in roi_names]
            axes[0].legend(handles=handles, loc="upper right",
                           frameon=False, fontsize=7,
                           bbox_to_anchor=(1.0, 1.55))

        # Title placed above the whole block using figure-level text
        if axes:
            top = axes[0].get_position()
            bot = axes[-1].get_position()
            fig.text(
                (top.x0 + top.x1) / 2,
                top.y1 + 0.020,
                title,
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=PALETTE["text"],
            )
            # Rounded-rect frame around the block
            from matplotlib.patches import FancyBboxPatch
            pad_x, pad_top, pad_bot = 0.010, 0.035, 0.015
            rect = FancyBboxPatch(
                (top.x0 - pad_x, bot.y0 - pad_bot),
                (top.x1 - top.x0) + 2 * pad_x,
                (top.y1 - bot.y0) + pad_top + pad_bot,
                boxstyle="round,pad=0.003",
                linewidth=0.9,
                edgecolor="#666",
                facecolor="none",
                transform=fig.transFigure,
                zorder=0,
                clip_on=False,
            )
            fig.patches.append(rect)

    # ── save ────────────────────────────────────────────────────────────
    def save(self,
             output_path: PathLike,
             formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = self.build()
        written: List[Path] = []
        for fmt in formats:
            p = out.with_suffix(f".{fmt}")
            fig.savefig(p, dpi=self.cfg.dpi, bbox_inches="tight",
                        facecolor="white")
            written.append(p)
            log.info("Wrote %s", p)
        plt.close(fig)
        return written
