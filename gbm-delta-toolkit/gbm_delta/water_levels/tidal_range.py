"""
gbm_delta.water_levels.tidal_range
==================================
Shared loader for reconstructed water-level CSVs.

The original `tidal_range.py` helper served one purpose: read the
"Reconstructed Water Level.csv" written by the water-level pipeline
(one datetime column + one column per station) and return a tidy
DataFrame indexed by datetime with station columns.

This module keeps that API:  ``load_water_levels(path) -> pd.DataFrame``
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_water_levels(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a reconstructed water-level CSV.

    Accepts either layout:
      * a `datetime`-like first column + one column per station
        (the pipeline export layout), or
      * long format with columns [datetime, station, water_level].

    Returns
    -------
    pd.DataFrame indexed by datetime, one column per station,
    values in metres (float).
    """
    path = Path(path)
    df = pd.read_csv(path)

    # long format → pivot
    lowered = {c.lower().strip(): c for c in df.columns}
    if {"station", "water_level"} <= set(lowered):
        dtcol = lowered.get("datetime") or df.columns[0]
        long = pd.DataFrame({
            "datetime": pd.to_datetime(df[dtcol]),
            "station": df[lowered["station"]].astype(str).str.strip(),
            "wl": pd.to_numeric(df[lowered["water_level"]], errors="coerce"),
        })
        out = (long.dropna(subset=["wl"])
                   .pivot_table(index="datetime", columns="station",
                                values="wl", dropna=False)
                   .sort_index())
        out.index.name = "datetime"
        return out

    # wide format: first parseable column is the index
    dt_col = None
    for cand in ("datetime", "date", "time", "Date & Time", "datetime_utc"):
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        dt_col = df.columns[0]

    out = df.set_index(pd.to_datetime(df[dt_col])).drop(columns=[dt_col])
    out = out.apply(pd.to_numeric, errors="coerce").sort_index()
    out.index.name = "datetime"
    out.columns = [str(c).strip() for c in out.columns]
    return out
