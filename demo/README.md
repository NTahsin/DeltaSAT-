# DeltaSAT backend — GeoTIFF edition (no Google Earth Engine)

Connects the `toolkit.html` WebGIS front-end to the **gbm-delta-toolkit**
(`gbm_delta` package) — and **serves that front-end itself**, so the whole
app is one server with zero configuration:

```bash
python run.py    # starts the backend and opens http://localhost:8000/toolkit.html
```

Everything runs on **local GeoTIFFs and gauge CSVs from the GitHub
hydro-morphological database** — no Earth Engine, no uploads:

1. **Water-index GeoTIFFs** — `NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho`
   → `geotiffs/00_MNDWI` (MNDWI) and `geotiffs/01_Fused index`
   (adapted index (MNDWI + AWEI)/2), downloaded + cached automatically by
   `demo/geodata.py` (under `WORK_DIR/_geodata`).
2. **Datewise land/water masks** — `geotiffs/02_Datewise land water` —
   used when a job has no freshly classified masks.
3. **Water levels** (10 stations) — same repository, `meghna estuary/`
   folder, via `demo/hydro.py` (cached under `WORK_DIR/_hydro`); gauge
   stations are plotted on the Leaflet map (`GET /stations`).

## Toolkit buttons → gbm_delta modules

| Button | Analysis id | Module |
|---|---|---|
| Land–water classification (MNDWI or adapted index) | `classify` | `extraction.index_quality.otsu_threshold` (Otsu or fixed 0) |
| Clean land–water → river channels | `clean` | `extraction.river_mask.RiverFromGeotiff` |
| Intertidal extent (area) — map + area time series | `intertidal` | `extraction.intertidal_detector.IntertidalDetector` |
| Lifespan analysis — map + lifespan time series | `lifespan` | `morphodynamics.lifespan_legacy.LifespanCalculator` |

The analyses chain automatically: `lifespan` runs the intertidal extent
first if no classified maps exist yet, and `clean` falls back to the
database's datewise land/water masks when the job has no classification.

## Quick start

```bash
cd demo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # scientific stack + FastAPI
python run.py
```

Open http://localhost:8000/toolkit.html — the status line shows
`connected · gbm_delta toolkit live` or `connected · MOCK mode` (when the
toolkit or its dependencies are missing; placeholder figures then render).

`MOCK_MODE=0` forces live (fails hard without the toolkit), `MOCK_MODE=1`
forces placeholders; default `auto` picks live whenever `gbm_delta`
imports. No GEE credentials of any kind are needed.

### Where the backend finds the toolkit

`pipeline.py` resolves the `gbm_delta` package automatically, in order:

1. env `TOOLKIT_DIR`
2. sibling `../gbm-delta-toolkit` (repo layout `DeltaSAT-/{demo,gbm-delta-toolkit}`)
3. sibling `../gbm-toolkit` (older folder name)
4. an installed `gbm_delta` (`pip install -e ../gbm-delta-toolkit`)

## Serving the front-end

The pages are served by the backend itself (`app.mount("/", StaticFiles(…,
html=True))`). Opening `toolkit.html` as a local file (`file://`) shows a
clear status-line hint instead of a working app.

## Job spec (what the front-end POSTs)

`POST /jobs` is a plain JSON body — no file parts:

```json
{
  "roi": {"w":90.4,"s":22.2,"e":91.1,"n":22.8},
  "index": "mndwi",
  "threshold": "otsu",
  "years": [1996, 2026],
  "minScenes": 2,
  "pixelSizeM": 30,
  "minComponentPx": 56,
  "elongation": 20,
  "analyses": ["classify", "clean", "intertidal", "lifespan"]
}
```

`roi` is optional (only crops the rendered figures). `analyses` accepts any
of `classify`, `clean`, `intertidal`, `lifespan`. Each result contains a
`status`, `files` (download URLs: GeoTIFFs + CSVs) and `layers` (rendered
map PNGs with WGS84 bounds for the Leaflet overlays, plus time-series
graphs).

## Caveats

- **Rate limits** — the GeoTIFF folder listing uses the GitHub contents API
  (60 req/h unauthenticated). Set `GITHUB_TOKEN` if you hit the limit;
  downloads go through `raw.githubusercontent.com` and are cached.
- **Intertidal extent** needs ≥ `minScenes` (default 2) binary scenes per
  year — pixels with fewer observations become NoData.
- **Lifespan** needs ≥ 2 years of classified maps; with one year the status
  says so and returns the maps only.
- **Classification threshold** — Otsu is computed per scene; the toolkit
  UI also offers the fixed index-zero threshold.
