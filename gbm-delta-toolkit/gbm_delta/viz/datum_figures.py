"""
tidal_datum_visualizer.py
=========================
Publication-quality figures for MHSW / MLSW tidal datum results.

Consumes
--------
  mhsw_mlsw_results.csv       — per (station × season × year) results
  mhsw_mlsw_summary_stats.csv — long-term mean ± std per station × season

Produces (all in output_dir)
--------
  01_annual_timeseries.pdf       — MHSW and MLSW annual time series per station,
                                   both seasons on shared axes
  02_seasonal_heatmap.pdf        — station × year heatmaps (MHSW, MLSW, STR)
  03_spring_tidal_range.pdf      — spring tidal range both seasons all stations
  04_station_comparison.pdf      — grouped bar chart: long-term mean ± std
  05_timeseries_interactive.html — Plotly interactive version of figure 01

Figure standards (JGR Earth Surface)
--------------------------------------
  Font       : Arial (falls back to DejaVu Sans if Arial not installed)
  Title      : 11 pt, weight 500
  Axis label : 10 pt
  Tick label :  9 pt
  Legend     :  9 pt
  Panel label:  9 pt bold  (a), (b), …
  Colourmap  : Seaborn muted palette (colour-blind-friendly)
  PDF backend: fonttype=42 → fonts embedded as TrueType (text not images)

Dependencies
------------
    pip install pandas numpy matplotlib seaborn plotly kaleido
"""

from __future__ import annotations

import os
import string
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Optional: Plotly for interactive figure ──────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ─────────────────────────────────────────────────────────────────────────────
# Publication style sheet
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pub_style() -> None:
    """Set matplotlib rcParams for publication-quality output."""
    matplotlib.rcParams.update({
        # ── Font ──────────────────────────────────────────────────────────
        "font.family":        "Arial",
        "font.sans-serif":    ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
        "mathtext.fontset":   "custom",
        "mathtext.rm":        "Arial",
        "mathtext.it":        "Arial:italic",
        "mathtext.bf":        "Arial:bold",
        # ── Sizes ─────────────────────────────────────────────────────────
        "font.size":          9,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.titlesize":   11,
        # ── PDF: embed fonts as TrueType (text, not outlines) ─────────────
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        # ── Lines & ticks ─────────────────────────────────────────────────
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,
        "lines.linewidth":    1.2,
        "lines.markersize":   5.0,
        # ── Layout ────────────────────────────────────────────────────────
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


# ── Colour constants (colour-blind-friendly via Seaborn "muted") ─────────────
_PALETTE  = sns.color_palette("muted")
C_WET     = _PALETTE[0]   # blue   – wet season
C_DRY     = _PALETTE[3]   # orange – dry season
C_MHSW    = _PALETTE[2]   # green  – MHSW
C_MLSW    = _PALETTE[4]   # purple – MLSW
C_STR     = _PALETTE[7]   # grey   – spring tidal range

# Season display labels
_SEAS_LABEL = {"wet": "Wet season (Jun–Oct)", "dry": "Dry season (Nov–May)"}


def _label_panel(ax, letter: str) -> None:
    """Add a bold panel label (a), (b), … in the upper-left corner."""
    ax.text(
        0.02, 0.97, f"({letter})",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="top", ha="left",
    )


def _savepdf(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save a figure to PDF using PdfPages (preserves font as text)."""
    with PdfPages(str(path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer class
# ─────────────────────────────────────────────────────────────────────────────

class TidalDatumVisualizer:
    """
    Publication-quality figures for MHSW / MLSW results.

    Parameters
    ----------
    results_csv  : path to mhsw_mlsw_results.csv
    summary_csv  : path to mhsw_mlsw_summary_stats.csv  (optional)
    output_dir   : directory where figures are written
    """

    def __init__(
        self,
        results_csv:  Union[str, Path],
        summary_csv:  Optional[Union[str, Path]] = None,
        output_dir:   Union[str, Path] = ".",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _apply_pub_style()

        # ── Load data ─────────────────────────────────────────────────────
        self.results = pd.read_csv(results_csv)
        self.results = self.results[self.results["valid"]].copy()
        self.stations = self.results["station"].unique().tolist()
        self.seasons  = ["wet", "dry"]

        if summary_csv and Path(summary_csv).exists():
            self.summary = pd.read_csv(summary_csv)
        else:
            self.summary = self._compute_summary()

        print(
            f"TidalDatumVisualizer ready — "
            f"{len(self.stations)} station(s), "
            f"{self.results['hyd_year'].nunique()} year(s)."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Produce all figures."""
        print("\nGenerating publication figures …")
        self.plot_annual_timeseries()
        self.plot_seasonal_heatmap()
        self.plot_spring_tidal_range()
        self.plot_station_comparison()
        if _HAS_PLOTLY:
            self.plot_interactive_timeseries()
        else:
            print("  [skip] Plotly not installed — no interactive HTML")
        print("Done.")

    # ── Figure 1: Annual time series ──────────────────────────────────────

    def plot_annual_timeseries(self) -> plt.Figure:
        """
        One row per station; two sub-panels per row (wet, dry season).
        Both MHSW and MLSW plotted as lines with markers and error bands
        (±0.5 × inter-annual std, as a stylistic range indicator).
        """
        n_sta  = len(self.stations)
        fig, axes = plt.subplots(
            n_sta, 2,
            figsize=(7.0, 2.2 * n_sta),
            sharex="col", squeeze=False,
        )
        fig.subplots_adjust(hspace=0.35, wspace=0.35)

        letters = list(string.ascii_lowercase)
        panel_n = 0

        for r, station in enumerate(self.stations):
            for c, season in enumerate(self.seasons):
                ax  = axes[r, c]
                sub = (
                    self.results[
                        (self.results["station"] == station)
                        & (self.results["season"]  == season)
                    ].sort_values("hyd_year")
                )

                if not sub.empty:
                    years = sub["hyd_year"].values
                    for metric, colour, marker, label in [
                        ("mhsw_m", C_MHSW, "o", "MHSW"),
                        ("mlsw_m", C_MLSW, "s", "MLSW"),
                    ]:
                        vals = sub[metric].values
                        ax.plot(
                            years, vals,
                            color=colour, marker=marker, lw=1.2,
                            ms=4, label=label, clip_on=False,
                        )
                        # Shaded ± 2-year rolling std as range indicator
                        s = pd.Series(vals, index=years)
                        roll = s.rolling(3, center=True, min_periods=1).std()
                        ax.fill_between(
                            years, vals - roll, vals + roll,
                            alpha=0.12, color=colour,
                        )

                ax.set_ylabel("Water level (m)", fontsize=10)
                ax.grid(True, lw=0.4, alpha=0.5, linestyle=":")
                ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                ax.tick_params(which="both", direction="in", length=3)

                _label_panel(ax, letters[panel_n])
                panel_n += 1

                # Column titles on top row only
                if r == 0:
                    ax.set_title(_SEAS_LABEL[season], fontsize=11, pad=4)

                # Station label as right-side ylabel on left column
                if c == 0:
                    ax.set_ylabel(
                        f"{station}\nWater level (m)", fontsize=9,
                    )

                # Legend on first subplot only
                if r == 0 and c == 0:
                    ax.legend(frameon=False, fontsize=9, loc="upper left")

        # Shared x-axis label
        for ax in axes[-1]:
            ax.set_xlabel("Hydrological year", fontsize=10)

        fig.suptitle(
            "Annual MHSW and MLSW — GBM delta gauge stations",
            fontsize=11, fontweight="medium", y=1.01,
        )

        path = self.output_dir / "01_annual_timeseries.pdf"
        _savepdf(fig, path)
        return fig

    # ── Figure 2: Seasonal heatmap ─────────────────────────────────────────

    def plot_seasonal_heatmap(self) -> plt.Figure:
        """
        Three heatmaps (MHSW, MLSW, spring tidal range), each split by season.
        Rows = stations, columns = hydrological years.
        """
        years    = sorted(self.results["hyd_year"].unique())
        metrics  = [
            ("mhsw_m",        "MHSW (m)",              "Blues"),
            ("mlsw_m",        "MLSW (m)",              "Greens"),
            ("spring_range_m","Spring tidal range (m)", "Oranges"),
        ]
        n_metrics = len(metrics)

        fig, axes = plt.subplots(
            n_metrics, 2,
            figsize=(max(8, len(years) * 0.55), 2.8 * n_metrics),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.55, wspace=0.3)

        letters = list(string.ascii_lowercase)
        panel_n = 0

        for row_idx, (metric, label, cmap_name) in enumerate(metrics):
            for col_idx, season in enumerate(self.seasons):
                ax  = axes[row_idx, col_idx]
                sub = self.results[self.results["season"] == season]

                # Pivot: rows=stations, columns=years
                pivot = sub.pivot_table(
                    index="station", columns="hyd_year", values=metric,
                )
                pivot = pivot.reindex(index=self.stations, columns=years)

                sns.heatmap(
                    pivot,
                    ax=ax, cmap=cmap_name,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    linewidths=0.3, linecolor="white",
                    cbar_kws={"label": label, "shrink": 0.8, "pad": 0.02},
                    vmin=float(self.results[metric].quantile(0.05)),
                    vmax=float(self.results[metric].quantile(0.95)),
                )
                ax.set_title(
                    f"{label} — {_SEAS_LABEL[season]}",
                    fontsize=10, pad=4,
                )
                ax.set_xlabel("Hydrological year", fontsize=10)
                ax.set_ylabel("" if col_idx else "Station", fontsize=10)
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                ax.tick_params(axis="y", labelrotation=0,  labelsize=8)

                _label_panel(ax, letters[panel_n])
                panel_n += 1

        fig.suptitle(
            "MHSW, MLSW and spring tidal range — station × year heatmaps",
            fontsize=11, fontweight="medium", y=1.01,
        )

        path = self.output_dir / "02_seasonal_heatmap.pdf"
        _savepdf(fig, path)
        return fig

    # ── Figure 3: Spring tidal range both seasons ──────────────────────────

    def plot_spring_tidal_range(self) -> plt.Figure:
        """
        Spring tidal range (MHSW − MLSW) timeseries for all stations,
        wet and dry season overlaid.  One panel per station.
        """
        n_sta = len(self.stations)
        ncols = min(3, n_sta)
        nrows = int(np.ceil(n_sta / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 3.0 * nrows),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.45, wspace=0.35)

        letters = list(string.ascii_lowercase)

        for idx, station in enumerate(self.stations):
            r, c = divmod(idx, ncols)
            ax   = axes[r][c]
            sub  = self.results[self.results["station"] == station]

            for season, colour, ls in [
                ("wet", C_WET, "-"),
                ("dry", C_DRY, "--"),
            ]:
                ss = sub[sub["season"] == season].sort_values("hyd_year")
                if not ss.empty:
                    ax.plot(
                        ss["hyd_year"], ss["spring_range_m"],
                        color=colour, ls=ls, marker="D", ms=4, lw=1.2,
                        label=_SEAS_LABEL[season].split(" ")[0],
                        clip_on=False,
                    )

            ax.set_title(station, fontsize=10, pad=3)
            ax.set_ylabel("Spring tidal range (m)", fontsize=10)
            ax.set_xlabel("Hydrological year",      fontsize=10)
            ax.grid(True, lw=0.4, alpha=0.5, ls=":")
            ax.tick_params(which="both", direction="in", length=3)
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            _label_panel(ax, letters[idx])

            if idx == 0:
                ax.legend(
                    frameon=False, fontsize=9,
                    title="Season", title_fontsize=9,
                )

        # Hide empty panels
        for idx in range(n_sta, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            "Annual spring tidal range (MHSW − MLSW)",
            fontsize=11, fontweight="medium",
        )

        path = self.output_dir / "03_spring_tidal_range.pdf"
        _savepdf(fig, path)
        return fig

    # ── Figure 4: Station comparison bar chart ─────────────────────────────

    def plot_station_comparison(self) -> plt.Figure:
        """
        Grouped bar chart showing long-term mean MHSW and MLSW ± std
        for each station × season combination.
        """
        df = self.summary.copy()
        # Build a tidy frame for seaborn
        rows = []
        for _, r in df.iterrows():
            for metric, mean_col, std_col, label in [
                ("MHSW", "mhsw_mean_m", "mhsw_std_m", "MHSW"),
                ("MLSW", "mlsw_mean_m", "mlsw_std_m", "MLSW"),
            ]:
                rows.append({
                    "station":    r["station"],
                    "season":     r["season"],
                    "seas_label": _SEAS_LABEL[r["season"]],
                    "datum":      label,
                    "mean":       r[mean_col],
                    "std":        r[std_col],
                    "group":      f"{r['station']} · {r['season'].capitalize()}",
                })
        tidy = pd.DataFrame(rows)

        fig, axes = plt.subplots(
            1, 2, figsize=(9.0, max(3.5, 0.5 * len(self.stations) + 2.0)),
            sharey=True,
        )

        palette = {"MHSW": C_MHSW, "MLSW": C_MLSW}

        for col_idx, (season, ax) in enumerate(zip(self.seasons, axes)):
            sub = tidy[tidy["season"] == season].copy()

            sns.barplot(
                data=sub,
                x="mean", y="station", hue="datum",
                palette=palette, edgecolor="white", linewidth=0.5,
                ax=ax, orient="h",
            )

            # Overlay error bars manually (seaborn barplot xerr not always clean)
            offsets = {"MHSW": -0.2, "MLSW": 0.2}
            for datum, grp in sub.groupby("datum"):
                grp = grp.sort_values("station")
                y_positions = [
                    list(self.stations).index(s) + offsets[datum]
                    for s in grp["station"]
                ]
                ax.errorbar(
                    grp["mean"], y_positions,
                    xerr=grp["std"],
                    fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=3, capthick=0.8,
                    zorder=5,
                )

            ax.set_title(_SEAS_LABEL[season], fontsize=11, pad=4)
            ax.set_xlabel("Water level (m)", fontsize=10)
            ax.set_ylabel("Station" if col_idx == 0 else "", fontsize=10)
            ax.axvline(0, color="black", lw=0.6, ls=":")
            ax.grid(True, axis="x", lw=0.4, alpha=0.5, ls=":")
            ax.tick_params(direction="in", length=3)
            _label_panel(ax, string.ascii_lowercase[col_idx])

            if col_idx == 1:
                ax.legend_.remove()  # keep legend only on left panel

        axes[0].legend(
            title="Datum", title_fontsize=9, frameon=False, fontsize=9,
        )

        fig.suptitle(
            "Long-term mean MHSW and MLSW ± inter-annual std",
            fontsize=11, fontweight="medium",
        )

        path = self.output_dir / "04_station_comparison.pdf"
        _savepdf(fig, path)
        return fig

    # ── Figure 5: Interactive Plotly timeseries ────────────────────────────

    def plot_interactive_timeseries(self) -> None:
        """
        Plotly multi-panel interactive HTML figure:
        one row per station, columns = wet / dry season.
        MHSW and MLSW as separate traces with hover info.
        """
        if not _HAS_PLOTLY:
            print("  Plotly not installed; skipping interactive figure.")
            return

        n_sta = len(self.stations)
        fig   = make_subplots(
            rows=n_sta, cols=2,
            shared_xaxes=False,
            subplot_titles=[
                f"{st} — {_SEAS_LABEL[se]}"
                for st in self.stations
                for se in self.seasons
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
        )

        _colour_map = {
            ("wet", "MHSW"): "#4878D0",
            ("wet", "MLSW"): "#EE854A",
            ("dry", "MHSW"): "#6ACC65",
            ("dry", "MLSW"): "#D65F5F",
        }
        shown_legends = set()

        for r_idx, station in enumerate(self.stations):
            for c_idx, season in enumerate(self.seasons):
                sub = (
                    self.results[
                        (self.results["station"] == station)
                        & (self.results["season"]  == season)
                    ].sort_values("hyd_year")
                )
                if sub.empty:
                    continue

                for metric, label in [("mhsw_m", "MHSW"), ("mlsw_m", "MLSW")]:
                    legend_key  = (season, label)
                    show_legend = legend_key not in shown_legends
                    if show_legend:
                        shown_legends.add(legend_key)

                    fig.add_trace(
                        go.Scatter(
                            x=sub["hyd_year"], y=sub[metric],
                            mode="lines+markers",
                            name=f"{label} ({season.capitalize()})",
                            line=dict(color=_colour_map[legend_key], width=1.5),
                            marker=dict(size=5),
                            legendgroup=legend_key,
                            showlegend=show_legend,
                            hovertemplate=(
                                f"<b>{station}</b><br>"
                                f"Year: %{{x}}<br>"
                                f"{label}: %{{y:.3f}} m<extra></extra>"
                            ),
                        ),
                        row=r_idx + 1, col=c_idx + 1,
                    )

        fig.update_layout(
            title=dict(
                text="Annual MHSW and MLSW — GBM delta gauge stations",
                font=dict(family="Arial", size=14),
            ),
            font=dict(family="Arial", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            height=320 * n_sta,
            template="plotly_white",
        )
        fig.update_xaxes(title_text="Hydrological year", title_font_size=11)
        fig.update_yaxes(title_text="Water level (m)",   title_font_size=11)

        path = self.output_dir / "05_timeseries_interactive.html"
        fig.write_html(str(path))
        print(f"  Saved: {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_summary(self) -> pd.DataFrame:
        """Compute summary stats from results if summary CSV not provided."""
        valid = self.results[self.results["valid"]].copy()
        return (
            valid.groupby(["station", "season"])
            .agg(
                n_years             = ("hyd_year",      "count"),
                mhsw_mean_m         = ("mhsw_m",        "mean"),
                mhsw_std_m          = ("mhsw_m",        "std"),
                mlsw_mean_m         = ("mlsw_m",        "mean"),
                mlsw_std_m          = ("mlsw_m",        "std"),
                spring_range_mean_m = ("spring_range_m","mean"),
                spring_range_std_m  = ("spring_range_m","std"),
            )
            .reset_index()
        )
