"""
geodata.py — GeoTIFF data fetcher for the DeltaSAT toolkit (no GEE).

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
runs are offline. The folder listing uses the GitHub contents API — set
``GITHUB_TOKEN`` if you hit its rate limit.
"""
from __future__ import annotations

import json
import os
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


def _hdrs() -> dict:
    h = dict(_HEADERS)
    tok = os.getenv("GITHUB_TOKEN", "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def folder(kind: str) -> Path:
    """Local cache folder for one data kind."""
    return CACHE / FOLDERS[kind].replace(" ", "_").lower()


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
    url = entry.get("download_url") or f"https://raw.githubusercontent.com/{REPO}/main/geotiffs/{urllib.parse.quote(FOLDERS.get('_root', ''))}"
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
