"""
geodata.py — GeoTIFF data fetcher + coverage for the DeltaSAT toolkit (no GEE).

The toolkit never asks the user to upload imagery. Water-index and
land/water GeoTIFFs of the Meghna estuary are pulled directly from the
GitHub hydro-morphological database — the same repository that holds the
tide-gauge records (demo/hydro.py):

    https://github.com/NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho
    └── geotiffs/
        ├── 00_MNDWI/                MNDWI index scenes   (mndwi_YYYY-MM-DD.tif)
        ├── 01_Fused index/          adapted index (MNDWI + AWEI)/2 scenes
        └── 02_Datewise land water/  ready-made binary land/water scenes

Files are downloaded once and cached under ``WORK_DIR/_geodata`` so repeat
runs are offline. ``coverage()`` reports, per folder, the WGS84 footprint
and the available scene years — this powers the dashed data-region outline
and the station layer on the Leaflet map (``GET /coverage``). Set
``GITHUB_TOKEN`` to raise the GitHub API rate limit.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho"

# toolkit data kind → folder in the database
FOLDERS = {
    "mndwi": "00_MNDWI",                  # MNDWI index GeoTIFFs
    "fused": "01_Fused index",            # adapted index (MNDWI + AWEI)/2 GeoTIFFs
    "binary": "02_Datewise land water",   # ready-made binary land/water masks
}

CACHE = Path(os.getenv("WORK_DIR", "./_jobs")).resolve() / "_geodata"

_HEADERS = {"User-Agent": "DeltaSAT-demo/1.0 (geotiff database fetcher)"}

_COVERAGE_CACHE: dict | None = None


def _hdrs() -> dict:
    h = dict(_HEADERS)
    tok = os.getenv("GITHUB_TOKEN", "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def folder(kind: str) -> Path:
    """Local cache folder for one data kind."""
    return CACHE / FOLDERS[kind].replace(" ", "_").lower()


def _year_of(name: str) -> int | None:
    m = re.search(r"(\d{4})-\d{2}-\d{2}", name) or re.search(r"(\d{4})\d{4}", name)
    return int(m.group(1)) if m else None


def _list_folder(kind: str) -> list[dict]:
    """GitHub contents-API listing of one geotiffs sub-folder."""
    url = (f"https://api.github.com/repos/{REPO}/contents/geotiffs/"
           f"{urllib.parse.quote(FOLDERS[kind])}")
    req = urllib.request.Request(url, headers=_hdrs())
    with urllib.request.urlopen(req, timeout=60) as r:   # nosec - fixed host
        entries = json.load(r)
    return [e for e in entries
            if e.get("type") == "file"
            and e["name"].lower().endswith((".tif", ".tiff"))]


def _download(entry: dict, dest_dir: Path, log=print) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["name"]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not entry.get("download_url"):
        raise RuntimeError(f"no download URL for {entry.get('name', '?')}")
    log(f"geodata: downloading {entry['name']} "
        f"({entry.get('size', 0) / 1e6:.1f} MB) …")
    req = urllib.request.Request(entry["download_url"], headers=_hdrs())
    with urllib.request.urlopen(req, timeout=600) as r:   # nosec - fixed host
        data = r.read()
    if not data:
        raise RuntimeError(f"empty download for {entry['name']}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    log(f"geodata: {dest.name} cached ({len(data) / 1e6:.1f} MB)")
    return dest


def _tif_bounds_wgs84(path: Path):
    """GeoTIFF bounds → Leaflet [[s, w], [n, e]] (None when ungeoreferenced)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds,
                                  densify_pts=21)
        return [[b[1], b[0]], [b[3], b[2]]]
    except Exception:
        return None


def fetch(kind: str, log=print) -> list[Path]:
    """Local paths of all GeoTIFFs of ``kind`` ('mndwi'|'fused'|'binary').

    Downloads anything missing from the GitHub database first (cached under
    ``WORK_DIR/_geodata``), so repeat runs are offline. Delete the cache
    folder to pick up newly pushed scenes.
    """
    if kind not in FOLDERS:
        raise ValueError(f"unknown data kind '{kind}' (use: {', '.join(FOLDERS)})")
    dest_dir = folder(kind)
    have = sorted(dest_dir.glob("*.tif")) if dest_dir.exists() else []
    if have:
        log(f"geodata: {len(have)} cached {kind} GeoTIFF(s) in {dest_dir}")
        return have
    entries = _list_folder(kind)
    if not entries:
        raise RuntimeError(f"no GeoTIFFs found in the database folder "
                           f"'geotiffs/{FOLDERS[kind]}' ({REPO})")
    return sorted(_download(e, dest_dir, log) for e in entries)


def coverage(log=print) -> dict:
    """Per-folder footprint + available years + scene counts, for the map.

    Only the smallest scene of each folder is downloaded (and cached) to
    read the footprint; years/counts come from the API listing itself.
    The result is cached in memory for the server's lifetime.
    """
    global _COVERAGE_CACHE
    if _COVERAGE_CACHE is not None:
        return _COVERAGE_CACHE
    out = {}
    for kind in FOLDERS:
        try:
            entries = _list_folder(kind)
            years = sorted({y for y in (_year_of(e["name"]) for e in entries)
                            if y is not None})
            bounds = None
            if entries:
                cached = sorted(folder(kind).glob("*.tif"))
                if cached:
                    sample = cached[0]
                else:
                    smallest = min(entries, key=lambda e: e.get("size", 0))
                    sample = _download(smallest, folder(kind), log)
                bounds = _tif_bounds_wgs84(sample)
            out[kind] = {"bounds": bounds, "years": years,
                         "n_scenes": len(entries)}
        except Exception as exc:
            log(f"geodata: coverage for '{kind}' unavailable ({exc})")
            out[kind] = {"bounds": None, "years": [], "n_scenes": 0,
                         "error": str(exc)}
    _COVERAGE_CACHE = out
    return out
