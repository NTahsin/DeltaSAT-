# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:39:26 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-

"""
annual_extremes.py
==================
Annual highest and lowest water-level readings per station and for the
region — GBM Delta Intertidal Morphology study.

Definitions
-----------
* ``annual_high`` / ``annual_low`` — the single highest / lowest hourly
  reading per calendar year, per station.
* ``region`` — for each year, the highest (lowest) of all stations' annual
  extremes (a regional envelope), and the cross-station mean of the annual
  extremes.
* ``event_dates`` — the timestamp of each extreme (for annotating bars).

Usage
-----
    from annual_extremes import summarize_extremes
    res = summarize_extremes("projection 6 stations 1996_2018.csv")
    res["annual_high"]        # year × station
    res["region_high"]       # year × [max, mean, station]
"""



import numpy as np
import pandas as pd

from gbm_delta.water_levels.tidal_range import load_water_levels


def _annual_extreme(wl: pd.DataFrame, kind: str) -> tuple[pd.DataFrame,
                                                          pd.DataFrame]:
    """Annual max ('high') or min ('low') values and their dates."""
    grouped = wl.groupby(wl.index.year)
    if kind == "high":
        values = grouped.max()
        # date of the annual maximum per station
        dates = pd.DataFrame(index=values.index, columns=wl.columns)
        for st in wl.columns:
            for yr, sub in wl[st].groupby(wl.index.year):
                dates.loc[yr, st] = sub.idxmax()
    else:
        values = grouped.min()
        dates = pd.DataFrame(index=values.index, columns=wl.columns)
        for st in wl.columns:
            for yr, sub in wl[st].groupby(wl.index.year):
                dates.loc[yr, st] = sub.idxmin()
    return values, dates


def annual_high(wl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values year × station, dates year × station)."""
    return _annual_extreme(wl, "high")


def annual_low(wl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values year × station, dates year × station)."""
    return _annual_extreme(wl, "low")


def region_extremes(high: pd.DataFrame, low: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                  pd.DataFrame]:
    """Year × [station, mean, max/min] regional envelopes."""
    reg_high = pd.DataFrame({
        "station_of_max": high.idxmax(axis=1),
        "mean_of_max": high.mean(axis=1),
        "max": high.max(axis=1),
    })
    reg_low = pd.DataFrame({
        "station_of_min": low.idxmin(axis=1),
        "mean_of_min": low.mean(axis=1),
        "min": low.min(axis=1),
    })
    return reg_high, reg_low


def summarize_extremes(path: str) -> dict:
    """One-call analysis: returns all extreme-value tables."""
    wl = load_water_levels(path)
    hv, hd = annual_high(wl)
    lv, ld = annual_low(wl)
    rh, rl = region_extremes(hv, lv)
    return {
        "water_levels": wl,
        "annual_high": hv,
        "annual_high_dates": hd,
        "annual_low": lv,
        "annual_low_dates": ld,
        "region_high": rh,
        "region_low": rl,
    }


def export_extremes(res: dict, out_csv: str = "annual_extremes_summary.csv"):
    frames = {
        "annual_high_m": res["annual_high"].round(3),
        "annual_low_m": res["annual_low"].round(3),
        "annual_high_dates": res["annual_high_dates"],
        "annual_low_dates": res["annual_low_dates"],
        "region_high": res["region_high"].round(3),
        "region_low": res["region_low"].round(3),
    }
    with open(out_csv, "w") as fh:
        for name, frame in frames.items():
            fh.write(f"# {name}\n")
            frame.to_csv(fh)
            fh.write("\n")
    return frames