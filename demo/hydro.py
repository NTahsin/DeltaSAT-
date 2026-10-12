"""
hydro.py — automatic water-level data for the DeltaSAT toolkit.

The demo never asks the user to upload gauge files. Instead it pulls the
water-level records it needs directly from the GitHub hydro-morphological
database of the Meghna estuary:

    https://github.com/NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho
    └── meghna estuary/
        ├── stations.csv          station metadata (semicolon-separated, WGS84)
        ├── chandpur.csv          hourly gauge records, semicolon-separated,
        ├── char_montaj.csv         day-first timestamps, WL in m MSL,
        ├── ...                    UTF-8 BOM, gaps = empty cells
        └── tajumuddin.csv

This module

  1. downloads ``stations.csv`` (1.5 kB, cached) and parses the 10 station
     coordinates (stdlib only — also powers the ``/stations`` map layer),
  2. on analysis, selects the stations inside (or near) the drawn ROI and
     downloads **only those** station CSVs (10–16 MB each, cached under
     ``WORK_DIR/_hydro`` so repeat runs are offline),
  3. parses them into ``{station name: pd.Series}`` with full display names
     ('Chandpur 1640', …) — the naming the gbm_delta water-level /
     morphodynamics modules match against.

Combined with ``gbm_delta.water_levels.datums.MHSWMLSWAnalyser`` (which turns
the series into the ``mhsw_mlsw_results.csv`` tidal-datums file) the whole
intertidal elevation chain runs without any user data.
"""
from __future__ import annotations

import csv
import io
import os
import unicodedata
import urllib.request
from pathlib import Path

REPO = "NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho"
FOLDER = "meghna%20estuary"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/{FOLDER}/"

CACHE = Path(os.getenv("WORK_DIR", "./_jobs")).resolve() / "_hydro"

_HEADERS = {"User-Agent": "DeltaSAT-demo/1.0 (hydro database fetcher)"}


# --------------------------------------------------------------------------- #
# download + cache
# --------------------------------------------------------------------------- #
def _download(fname: str, log=print) -> Path:
    """Download one database file into the cache (skips if already there)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / fname
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = RAW + urllib.request.quote(fname)
    log(f"hydro: downloading {fname} from the GitHub database …")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:   # nosec - fixed host
        data = r.read()
    if not data:
        raise RuntimeError(f"empty download for {fname}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    log(f"hydro: {fname} cached ({len(data)/1e6:.1f} MB)")
    return dest


# --------------------------------------------------------------------------- #
# station metadata (stdlib — usable without the scientific stack)
# --------------------------------------------------------------------------- #
def station_table(log=print) -> list[dict]:
    """Parsed ``stations.csv`` → list of {id, name, lat, lon, river, tidal}."""
    path = _download("stations.csv", log)
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    out = []
    for r in rows:
        try:
            lat = float(r["Latitude"])
            lon = float(r["Longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        name = (r.get("Station ID & Name") or r.get("Station Name") or
                r.get("Station ID") or "").strip()
        out.append({"id": name, "name": name, "lat": lat, "lon": lon,
                    "river": (r.get("River Name") or "").strip(),
                    "tidal": (r.get("Tidal Status") or "").strip()})
    return out


# --------------------------------------------------------------------------- #
# series parsing / ROI selection (pandas — live mode only)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _display_name(stem: str, meta: list[dict]) -> str:
    n = _norm(stem)
    for m in meta:
        mn = _norm(m["name"])
        if mn == n or mn.startswith(n):
            return m["name"]
    return stem.replace("_", " ").title()


def load_station_csv(path: Path):
    """One repo station CSV → pd.Series (hourly index, m MSL).

    Handles: semicolon separator, UTF-8 BOM, trailing empty columns,
    day-first timestamps ('1-1-1996 00:00'), gaps as empty cells.
    """
    import pandas as pd
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
    raw = raw.loc[:, [c for c in raw.columns
                      if not str(c).startswith("Unnamed")]]
    if raw.shape[1] < 2:
        raise ValueError(f"{path.name}: expected >= 2 columns")
    dt = pd.to_datetime(raw.iloc[:, 0], dayfirst=True, errors="coerce")
    wl = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    s = pd.Series(wl.to_numpy(), index=dt)
    s = s[dt.notna()]
    s.index.name = "datetime"
    return s[~s.index.duplicated(keep="first")].sort_index()


def series_near(roi: dict, margin_deg: float = 0.35, log=print):
    """Water-level series for the database stations inside/near the ROI.

    Returns (wl_series, coords):
      wl_series : {station display name: pd.Series of water level (m MSL)}
      coords    : pd.DataFrame [station_id, lon, lat]
    Raises RuntimeError when no station falls inside the box or the
    database cannot be reached.
    """
    import pandas as pd
    meta = station_table(log)
    if not meta:
        raise RuntimeError("hydro database: stations.csv unreadable")
    w, s_, e, n = roi["w"], roi["s"], roi["e"], roi["n"]
    near = [m for m in meta
            if (w - margin_deg) <= m["lon"] <= (e + margin_deg)
            and (s_ - margin_deg) <= m["lat"] <= (n + margin_deg)]
    if not near:
        raise RuntimeError(
            "no GBM hydro-database station inside/near the ROI "
            "(known stations: " + ", ".join(m["name"] for m in meta) + ")")

    wl_series, coords = {}, []
    for m in near:
        base = _norm(m["name"])                      # e.g. 'chandpur1640'
        fname = None
        for stem in _FILE_STEMS:
            ns = _norm(stem)                         # e.g. 'chandpur'
            if base == ns or base.startswith(ns) or ns.startswith(base):
                fname = f"{stem}.csv"
                break
        if fname is None:
            log(f"hydro: no CSV found for station {m['name']}, skipping")
            continue
        path = _download(fname, log)
        ser = load_station_csv(path)
        ser = ser.dropna()
        if ser.empty:
            log(f"hydro: {m['name']} has no readings, skipping")
            continue
        wl_series[m["name"]] = ser
        coords.append({"station_id": m["name"], "lon": m["lon"],
                        "lat": m["lat"]})
    if not wl_series:
        raise RuntimeError("no usable water-level series near the ROI")
    log(f"hydro: {len(wl_series)} station series ready "
        f"({', '.join(wl_series)})")
    return wl_series, pd.DataFrame(coords)


def write_wide_csv(wl_series: dict, path: Path) -> Path:
    """Wide water-level CSV (datetime column, one column per station) for
    ``gbm_delta.water_levels`` (tidal_range.load_water_levels /
    datums.MHSWMLSWAnalyser)."""
    import pandas as pd
    df = pd.DataFrame(wl_series)
    df.index.name = "datetime"
    df.to_csv(path)
    return path


# station file stems present in the database folder (order matters for
# prefix matching: 'dhulasur' is checked before its prefix 'dhulia')
_FILE_STEMS = ["charchenga", "char_ramdaspur", "char_montaj", "chandpur",
               "chitalkhali", "dasmina", "dhulasur", "dhulia", "monpura",
               "tajumuddin"]
