"""
gbm_delta.viz.style
===================
Shared publication figure style ("house style") for all GBM delta figures.

Provides the palette `C`, the `delta()` colour ramp, `apply_style()` /
`apply_pub_style()` rcParams helpers, and small panel/axis/legend helpers.
This module is the package version of the original `paper_style.py` helper
script and keeps the same API:

    apply_style, apply_pub_style, C, DELTA, delta,
    label_panel, style_ax, legend_ax
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── House palette ────────────────────────────────────────────────────────────
C = {
    # water / datum hierarchy (dark → light)
    "water_dark":  "#0F4C81",   # MHSW
    "water_mid":   "#4E8CB8",   # MSL
    "water_light": "#A6C8DC",   # MLSW
    # accents
    "extent_main": "#E07A2F",   # intertidal / main data series
    "ref_line":    "#666666",   # reference lines (γ = 1, trends)
    "muted":       "#8A8A8A",   # annotations, secondary text
    # categorical (Tableau-ish, used by river comparisons)
    "cat_1": "#4E79A7", "cat_2": "#F28E2B", "cat_3": "#E15759",
    "cat_4": "#76B7B2", "cat_5": "#59A14F", "cat_6": "#EDC948",
    "cat_7": "#B07AA1", "cat_8": "#FF9DA7",
}

CATEGORICAL = [C[f"cat_{i}"] for i in range(1, 9)]


# ── Colour ramp between the water palette endpoints ──────────────────────────
def delta(start: int = 0, stop: int = 1, n: int = 6):
    """
    Return `n` colours sampled from a ramp water_dark → water_light.
    `start`/`stop` (0–1) select the sub-range of the ramp, so
    ``delta(0, 3, 6)`` gives 3 ramp colours out of 6 steps.
    """
    import matplotlib.colors as mcolors
    ramp = mcolors.LinearSegmentedColormap.from_list(
        "delta_ramp", [C["water_dark"], C["water_light"]])
    ts = [start + (stop - start) * i / max(n - 1, 1) for i in range(n)]
    return [ramp(t) for t in ts]


#: backwards-compatible alias used by some scripts
DELTA = delta


# ── Global style ─────────────────────────────────────────────────────────────
def apply_style():
    """Publication rcParams: compact serif-free fonts, thin spines/ticks."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          9,
        "axes.titlesize":    10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "axes.grid":          False,
        "grid.linewidth":     0.4,
        "grid.alpha":        0.35,
        "grid.linestyle":    ":",
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })


def apply_pub_style():
    """Alias kept for scripts that call the alternative name."""
    apply_style()


# ── Panel helpers ────────────────────────────────────────────────────────────
def label_panel(ax, letter, x=0.02, y=0.97, fontsize=9):
    """Bold panel letter, e.g. label_panel(ax, 'a')."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


def style_ax(ax, xlabel=None, ylabel=None, title=None, fontsize=9):
    """Apply axis labels/title with a thin, publication-style frame."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, lw=0.4, alpha=0.35, ls=":")
    ax.tick_params(direction="out", length=3, labelsize=fontsize - 1)


def legend_ax(ax, loc="best", ncol=1, bbox_to_anchor=None, fontsize=8,
              frameon=False, **kw):
    """Consistent legend placement helper."""
    return ax.legend(loc=loc, ncol=ncol, fontsize=fontsize,
                     frameon=frameon,
                     bbox_to_anchor=bbox_to_anchor, **kw)
