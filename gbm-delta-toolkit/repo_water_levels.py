#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
repo_water_levels.py
====================
Bridge between the GitHub hydro-morphological database

    https://github.com/NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho

and the ``gbm_delta`` water-level / morphodynamics modules.

Repository layout (folder ``meghna estuary/``)
---------------------------------------------
* ``<station>.csv``    hourly gauge records, semicolon-separated::

      Date & Time;WL (mMSL);;;;;;
      1-1-1996 00:00;-1.00509;;;;;;
      1-1-1996 07:00;;;;;;;          <- missing hours = empty cells

  Timestamps are day-first (``D-M-YYYY HH:MM``); values are water level
  in metres relative to MSL; files start with a UTF-8 BOM.
* ``stations.csv``     station metadata (semicolon-separated, WGS84
  lat/lon, plus BTM projected coordinates).

What this adapter does
----------------------
1. Parses every station CSV (BOM-safe, day-first dates, gaps -> NaN).
2. Labels each series with its full station name taken from
   ``stations.csv`` — e.g. ``chandpur.csv`` -> ``"Chandpur 1640"`` — the
   same naming used by the original "Reconstructed Water Level.csv" the
   toolkit was built around (so ``gbm_delta.water_levels.stations``
   prefix-matching works unchanged).
3. Writes one wide water-level CSV readable by
   ``gbm_delta.water_levels.tidal_range.load_water_levels()``.
4. Writes a station-coordinate CSV (station, latitude, longitude) for
   ``stations.station_frame()`` / the ``station_coords`` argument of the
   ``morphodynamics.elevation_v2/v3`` processors.

Usage
-----
    # 1. clone the database (or download the ZIP from GitHub)
    git clone https://github.com/NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho.git

    # 2. convert the repo -> toolkit CSVs
    python repo_water_levels.py /path/to/Ganges-Brahmaputra-Meghna-Hydro-morpho \
           --out water_levels

    # 3. continue with the toolkit (WF1 / WF8 in WORKFLOWS.md)
    from gbm_delta.water_levels.tidal_range import load_water_levels
    from gbm_delta.water_levels.stations import enrich
    wl = load_water_levels("water_levels/reconstructed_water_level.csv")
    sta = enrich(wl.columns)          # adds dist_coast_km, sorted landward->seaward

    # gauge QA / datums (v0.1.0 modules):
    #   processor.py -> cleaned series,  extremes.py -> HW/LW events,
    #   datums.py    -> MHW/MLW/MHSW/MLSW per station  -> datums.csv
    # surface_profiler.py -> along-channel water-surface profiles
    # viz/along_channel.along_channel_figure(wl_csv) -> asymmetry figure

    # the datums.csv + stations_from_repo.csv then feed the intertidal
    # elevation workflow (WORKFLOWS.md WF8):
    #   morphodynamics.elevation_v2.IntertidalElevationProcessor(
    #       wse_dir=..., datums_csv="datums.csv",
    #       station_coords="stations_from_repo.csv", output_dir=...)

API
---
    load_repo(repo_dir)            -> (wl_df, stations_df)
    load_station_csv(path)         -> pd.Series
    read_station_metadata(repo_dir)-> pd.DataFrame
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _norm(s: str) -> str:
    """lowercase, keep letters/digits only (for name matching)."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_data_dir(repo_dir: Path) -> Path:
    """Locate the 'meghna estuary' folder (name tolerant)."""
    repo_dir = Path(repo_dir)
    if repo_dir.is_file():                       # a single station CSV
        return repo_dir.parent
    for cand in [repo_dir, *repo_dir.iterdir()]:
        if cand.is_dir() and _norm(cand.name) == "meghnaestuary":
            return cand
    # fall back: any directory containing stations.csv
    for cand in sorted(repo_dir.rglob("stations.csv")):
        return cand.parent
    return repo_dir


def read_station_metadata(repo_dir: Path) -> pd.DataFrame:
    """Parse the repo's stations.csv -> station, latitude, longitude table."""
    meta_path = _find_data_dir(repo_dir) / "stations.csv"
    if not meta_path.exists():
        return pd.DataFrame(columns=["station", "latitude", "longitude"])
    df = pd.read_csv(meta_path, sep=";", encoding="utf-8-sig", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    name_col = next((c for c in df.columns
                     if "id & name" in c.lower()), df.columns[0])
    out = pd.DataFrame({
        "station":  df[name_col].astype(str).str.strip(),
        "latitude": pd.to_numeric(df["Latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df["Longitude"], errors="coerce"),
    })
    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def load_station_csv(path: Path) -> pd.Series:
    """Parse one repo station CSV -> Series of water level (m MSL), hourly index.

    Handles: semicolon separator, UTF-8 BOM, trailing empty columns,
    day-first timestamps ('1-1-1996 00:00'), gaps as empty cells.
    """
    path = Path(path)
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
    raw = raw.loc[:, [c for c in raw.columns if not str(c).startswith("Unnamed")]]
    if raw.shape[1] < 2:
        raise ValueError(f"{path.name}: expected >= 2 columns, got {raw.shape[1]}")

    dt = pd.to_datetime(raw.iloc[:, 0], dayfirst=True, errors="coerce")
    wl = pd.to_numeric(raw.iloc[:, 1], errors="coerce")          # '' -> NaN
    s = pd.Series(wl.to_numpy(), index=dt, name=path.stem)
    s = s[dt.notna()]
    s.index.name = "datetime"
    return s[~s.index.duplicated(keep="first")].sort_index()


def _display_name(stem: str, meta: pd.DataFrame) -> str:
    """chandpur -> 'Chandpur 1640' via the metadata table (prefix match)."""
    n = _norm(stem)
    for _, row in meta.iterrows():
        m = _norm(row["station"])
        if m == n or m.startswith(n):
            return str(row["station"])
    return stem.replace("_", " ").title()          # graceful fallback


# ---------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------

def load_repo(repo_dir):
    """Load the whole repository -> (wide water-level DataFrame, stations table).

    Returns
    -------
    wl_df : pd.DataFrame
        Hourly datetime index, one column per station, full gauge names
        as columns ('Chandpur 1640', 'Char Montaj', ...), NaN for gaps.
    stations_df : pd.DataFrame
        station / latitude / longitude (WGS84) from the repo metadata.
    """
    data_dir = _find_data_dir(Path(repo_dir))
    meta = read_station_metadata(data_dir)

    csvs = sorted(p for p in data_dir.glob("*.csv")
                  if _norm(p.stem) != "stations")
    if not csvs:
        raise FileNotFoundError(f"no station CSVs found under {data_dir}")

    series = {}
    for path in csvs:
        s = load_station_csv(path)
        s.name = _display_name(path.stem, meta)
        if s.name in series:                       # duplicate station files
            s.name = f"{s.name} ({path.stem})"
        series[s.name] = s

    wl = pd.concat(series.values(), axis=1, join="outer").sort_index()
    wl.index.name = "datetime"

    # keep only rows inside the joint observation window (kills the
    # all-NaN margins that the outer join can create)
    valid = wl.notna()
    wl = wl.loc[valid.any(axis=1)]

    stations_df = meta[meta["station"].isin(wl.columns)].reset_index(drop=True)
    return wl, stations_df


def coverage_report(wl: pd.DataFrame) -> pd.DataFrame:
    """Per-station date range and % of hours with data."""
    rows = []
    for col in wl.columns:
        s = wl[col].dropna()
        rows.append({
            "station": col,
            "n_hours": int(len(s)),
            "start": s.index.min(), "end": s.index.max(),
            "coverage_pct": round(100.0 * len(s) / max(len(wl), 1), 1),
        })
    return pd.DataFrame(rows)


def write_wide_csv(wl: pd.DataFrame, out_csv: Path) -> None:
    """Write the wide 'Reconstructed Water Level' CSV for the toolkit."""
    out = wl.copy()
    out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
    out.index.name = "Date & Time"
    out.to_csv(out_csv, index=True, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert the Ganges-Brahmaputra-Meghna-Hydro-morpho repo "
                    "into the wide water-level CSVs used by gbm_delta.")
    ap.add_argument("repo_dir",
                    help="path to the cloned repository (or the 'meghna estuary' folder)")
    ap.add_argument("--out", default="water_levels",
                    help="output folder (default: ./water_levels)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    wl, stations = load_repo(args.repo_dir)

    wide_csv = out_dir / "reconstructed_water_level.csv"
    write_wide_csv(wl, wide_csv)
    stations.to_csv(out_dir / "stations_from_repo.csv", index=False,
                    encoding="utf-8-sig")

    rep = coverage_report(wl)
    rep.to_csv(out_dir / "coverage_report.csv", index=False)

    print(f"stations : {len(wl.columns)}")
    print(f"period   : {wl.index.min()} -> {wl.index.max()}  ({len(wl)} hourly rows)")
    print("\ncoverage:")
    print(rep.to_string(index=False))
    print(f"\nwrote: {wide_csv}")
    print(f"wrote: {out_dir / 'stations_from_repo.csv'}")
    print(f"wrote: {out_dir / 'coverage_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
