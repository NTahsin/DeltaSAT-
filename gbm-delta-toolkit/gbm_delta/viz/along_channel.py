# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:22:37 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
along_channel.py
================
Along-channel profile figure (fig05-style): two columns — Lower Meghna
(main channel: Chandpur, Charchenga, Chitalkhali) and Tentulia (side
channel: Char Ramdaspur, Dhulia, Dasmina) — with distance along the
channel on the x-axis (upstream → seaward) and, per row:

    1. water levels  (MHSW / MSL / MLSW, m MSL)
    2. tide duration asymmetry  γ = T_fall / T_rise
       (γ > 1 → ebb-dominant, γ < 1 → flood-dominant)
       plus, optionally, the rate-asymmetry ratio R as open markers
    3. inter-annual: γ per year for each channel (x = Year)

Station positions on each channel axis are measured from the seaward end
and increase upstream? — NO, following fig05 the axis runs
"upstream → seaward", i.e. distance measured from the upstream reference
(Chandpur for the Meghna, Char Ramdaspur for the Tentulia).

Usage
-----
    from along_channel import along_channel_figure
    along_channel_figure(wl_csv="Reconstructed Water Level.csv",
                        out="fig05_asymmetry_profile.png")
"""

#from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gbm_delta.viz.style import (apply_style, C, DELTA, delta, label_panel,
                                  style_ax, legend_ax)
from gbm_delta.water_levels.tidal_range import load_water_levels
from gbm_delta.water_levels.stations import enrich, _resolve_name
from scipy.signal import find_peaks

# channel definitions: full CSV column label -> (channel, rank upstream)
# ranks: 0 = most upstream on that channel
MAIN_CHANNEL = {          # Lower Meghna
    "Chandpur 1640": 0,
    "Charchenga": 2,
    "Chitalkhali 1610": 1,
}
SIDE_CHANNEL = {          # Tentulia
    "Char Ramdaspur 1620": 0,
    "Dhulia 0820": 1,
    "Dasmina 0810": 2,
}
CHANNELS = [("Lower Meghna", MAIN_CHANNEL), ("Tentulia", SIDE_CHANNEL)]

# along-channel distance (km) per station, measured upstream → seaward
# (edit with your surveyed along-channel distances if different)
ALONG_KM = {
    "Chandpur 1640": 0.0,     # upstream end of the Meghna axis
    "Charchenga": 53.0,
    "Chitalkhali 1610": 65.0,
    "Char Ramdaspur 1620": 0.0,   # upstream end of the Tentulia axis
    "Dhulia 0820": 28.0,
    "Dasmina 0810": 63.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# duration asymmetry per station (γ = T_fall/T_rise) and levels
# ─────────────────────────────────────────────────────────────────────────────
def limb_durations(series: pd.Series) -> tuple[float, float]:
    """Mean flood (rising) and ebb (falling) limb duration in hours."""
    v = series.to_numpy(dtype=float)
    ok = np.isfinite(v)
    vf = np.where(ok, v, np.nan)
    hi, _ = find_peaks(np.nan_to_num(vf, nan=-1e9), distance=5, prominence=0.05)
    lo, _ = find_peaks(np.nan_to_num(-vf, nan=-1e9), distance=5,
                       prominence=0.05)
    pts = np.sort(np.concatenate([hi, lo]))
    if len(pts) < 4:
        return np.nan, np.nan
    t = series.index.to_numpy(dtype="datetime64[s]").astype(float) / 3.6e6
    durs, kinds = [], np.diff(np.isin(pts, hi).astype(int))
    # kinds[i] = +1 → rising from pts[i] to pts[i+1], -1 → falling
    for i, k in enumerate(kinds):
        durs.append((t[pts[i + 1]] - t[pts[i]]) * k)
    rise = [d for d in durs if d > 0]
    fall = [-d for d in durs if d < 0]
    return (np.mean(rise) if rise else np.nan,
            np.mean(fall) if fall else np.nan)


def station_asymmetry(wl: pd.DataFrame) -> pd.DataFrame:
    """Per-station γ (T_fall/T_rise), R ratio and spring/mean levels."""
    rows = {}
    for st in wl.columns:
        s = wl[st].dropna()
        if len(s) < 500:
            continue
        T_rise, T_fall = limb_durations(s)
        gamma = T_fall / T_rise if T_rise else np.nan
        # rate-based ratio for reference
        mean_rate = np.abs(np.diff(s.to_numpy(dtype=float))).mean() / \
            (1.0)   # per hour — wl index assumed hourly
        rows[st] = {"gamma": gamma, "T_rise_h": T_rise, "T_fall_h": T_fall}
    return pd.DataFrame(rows).T


def station_levels(wl: pd.DataFrame, months=None) -> pd.DataFrame:
    """MHSW / MSL / MLSW per station (optionally restricted to months)."""
    sub = wl
    if months:
        sub = wl[np.isin(wl.index.month, months)]
    rows = {}
    for st in sub.columns:
        s = sub[st].dropna()
        if len(s) < 100:
            continue
        hi, _ = find_peaks(s.to_numpy(), distance=5, prominence=0.05)
        lo, _ = find_peaks(-s.to_numpy(), distance=5, prominence=0.05)
        hw, lw = s.to_numpy()[hi], s.to_numpy()[lo]
        mhs = np.mean(hw[hw >= np.percentile(hw, 90)])
        mls = np.mean(lw[lw <= np.percentile(lw, 10)])
        rows[st] = {"MHSW": mhs, "MSL": s.mean(), "MLSW": mls}
    return pd.DataFrame(rows).T


def annual_gamma(wl: pd.DataFrame) -> pd.DataFrame:
    """γ per station per year (for the inter-annual row)."""
    out = {}
    for st in wl.columns:
        for yr, s in wl[st].dropna().groupby(wl[st].dropna().index.year):
            if len(s) < 2000:
                continue
            T_rise, T_fall = limb_durations(s)
            out[(st, yr)] = T_fall / T_rise if T_rise else np.nan
    return (pd.Series(out).rename("gamma")
            .rename_axis(["station", "year"]).reset_index())


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
def along_channel_figure(wl_csv: str,
                         out: str = "fig05_asymmetry_profile.png",
                         months=None):
    apply_style()
    wl = load_water_levels(wl_csv)
    asy = station_asymmetry(wl)
    lev = station_levels(wl, months=months)
    gamma_yr = annual_gamma(wl)

    fig, axes = plt.subplots(
        2, 2, figsize=(9.8, 6.4),
        gridspec_kw={"wspace": 0.28, "hspace": 0.55})

    C_MHS = C["water_dark"]; C_MSL = C["water_mid"]; C_MLS = C["water_light"]

    for col, (chan_name, members) in enumerate(CHANNELS):
        sts = [s for s in members if s in asy.index]
        sts.sort(key=lambda s: ALONG_KM[s])
        x = [ALONG_KM[s] for s in sts]

        # ── row 1: water levels (MHSW / MSL / MLSW) ─────────────────────
        ax = axes[0, col]
        ax.plot(x, lev.loc[sts, "MHSW"], "-o", ms=5, lw=1.2,
                color=C_MHS, label="MHSW")
        ax.plot(x, lev.loc[sts, "MSL"], "--", lw=1.1,
                color=C_MSL, label="MSL")
        ax.plot(x, lev.loc[sts, "MLSW"], "-o", ms=5, lw=1.2,
                color=C_MLS, label="MLSW")
        ax.set_xticks(x)
        ax.set_xticklabels([_resolve_name(s) for s in sts],
                           rotation=30, ha="right")
        style_ax(ax, xlabel="Distance along channel (km, "
                            "upstream → seaward)",
                 ylabel="Water level (m MSL)",
                 title=chan_name)
        ax.xaxis.labelpad = 10
        label_panel(ax, "ab"[col])

        # ── row 2: tide duration asymmetry γ ─────────────────────────────
        ax = axes[1, col]
        g = asy.loc[sts, "gamma"]
        ax.plot(x, g, "-s", ms=6, lw=1.3, color=C["extent_main"])
        ax.axhline(1.0, color=C["ref_line"], lw=0.9, ls=":",
                   label="γ = 1 (symmetric)")
        for xi, (st, gv) in zip(x, g.items()):
            if np.isfinite(gv):
                ax.annotate(f"{gv:.3f}", (xi, gv),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=6, color=C["muted"])
        ax.set_xticks(x)
        ax.set_xticklabels([_resolve_name(s) for s in sts],
                           rotation=30, ha="right")
        style_ax(ax, xlabel="Distance along channel (km, "
                            "upstream → seaward)",
                 ylabel="Duration asymmetry γ = T$_{fall}$/T$_{rise}$",
                 title=f"Tide duration asymmetry — {chan_name}")
        ax.xaxis.labelpad = 10
        label_panel(ax, "cd"[col])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42),
                  borderaxespad=0.0, frameon=False)

    # panel a legend outside
    axes[0, 0].legend(loc="upper center", ncol=3,
                      bbox_to_anchor=(1.28, -0.55), borderaxespad=0.0,
                      frameon=False)

    fig.suptitle("Along-channel tidal properties — Lower Meghna & Tentulia",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")

    # inter-annual γ per channel — separate small figure (like fig05 row 3)
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.8, 2.9),
                               gridspec_kw={"wspace": 0.28})
    for col, (chan_name, members) in enumerate(CHANNELS):
        ax = axes2[col]
        sts = [s for s in members if s in set(gamma_yr["station"])]
        cmap = dict(zip(sts, delta(0, 3, 6)))
        for st in sts:
            sub = (gamma_yr[gamma_yr["station"] == st]
                   .set_index("year")["gamma"])
            ax.plot(sub.index, sub.values, "-o", ms=3, lw=1.2,
                    color=cmap[st], label=_resolve_name(st))
        ax.axhline(1.0, color=C["ref_line"], lw=0.9, ls=":")
        style_ax(ax, xlabel="Year", ylabel="γ = T$_{fall}$/T$_{rise}$",
                 title=chan_name)
        label_panel(ax, "ef"[col])
        legend_ax(ax, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, -0.32), borderaxespad=0.0,
                  fontsize=6)
    out2 = out.replace(".png", "_annual.png")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2}")
    return asy, lev, gamma_yr