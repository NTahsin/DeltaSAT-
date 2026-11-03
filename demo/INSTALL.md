# DeltaSAT demo — complete folder (gbm-delta-toolkit, no-GEE edition)

This is the **full `demo/` folder** of the DeltaSAT- web app: front-end
(HTML pages + `theme.css` + Leaflet WebGIS) and backend
(`app.py`, `pipeline.py`, `geodata.py`, `hydro.py`).

> If your repo still has a `demo/` folder, replace it wholesale with this
> folder (it contains everything, including the restored front-end).
> **Delete the old `demo/gee.py`** — the no-GEE edition no longer uses it.

## 1. Layout

```
DeltaSAT-/
├── demo/               ← this folder
│   ├── index.html            landing page
│   ├── toolkit.html          main WebGIS app (index choice, 4 analyses, gauges)
│   ├── intertidal.html, markov.html, migration.html,
│   │   river-extraction.html, singularity-index.html,
│   │   water-indices.html    topic pages
│   ├── theme.css             shared stylesheet
│   ├── app.py                FastAPI server — also serves the web pages
│   ├── run.py                one-command launcher (server + browser)
│   ├── pipeline.py           analysis wiring → gbm_delta toolkit (4 analyses)
│   ├── geodata.py            auto GeoTIFFs from the GitHub database (no GEE)
│   ├── hydro.py              auto water levels from the GitHub gauge database
│   ├── make_cloud_dataset.py, upload_dataset.py,
│   │   README_dataset.md     legacy cloud-dataset utilities (optional)
│   ├── requirements.txt
│   └── INSTALL.md            (this file)
└── gbm-delta-toolkit/  ← the gbm_delta Python package, NOT in this zip
```

## 2. Install

1. Place the **gbm-delta-toolkit** folder next to `demo/`
   (repo root: `DeltaSAT-/{demo, gbm-delta-toolkit}`). The pipeline finds
   it automatically; alternatives: `pip install -e /path/to/gbm-delta-toolkit`,
   or point `TOOLKIT_DIR` at it.
2. `pip install -r demo/requirements.txt`
   (numpy, pandas, rasterio, scikit-image, tifffile, FastAPI, uvicorn …).
3. No Earth Engine authentication — nothing else to configure.
   Optional: `GITHUB_TOKEN=…` to raise the GitHub API rate limit.

## 3. Run — one command

```bash
cd demo
python run.py        # starts the server AND opens http://localhost:8000/toolkit.html
```

That is the entire setup: the backend serves the web pages itself, so
there is no API URL to configure — the toolkit auto-checks the backend
status on page load and shows `connected · gbm_delta toolkit live` or
`connected · MOCK mode`.

Equivalent manual form:

```bash
uvicorn app:app --port 8000     # then open http://localhost:8000/toolkit.html
```

`MOCK_MODE=0` forces live (fails hard if the toolkit is not importable),
`MOCK_MODE=1` forces placeholder figures; default `auto` picks live when
the `gbm_delta` package imports.

## 4. What runs, with what data (no uploads, no GEE)

| Input | Source | Mechanism |
|---|---|---|
| Water-index GeoTIFFs (MNDWI · adapted (MNDWI+AEI)/2) | GitHub database `NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho` → `geotiffs/00_MNDWI`, `geotiffs/01_Fused index` | `demo/geodata.py` downloads + caches under `WORK_DIR/_geodata` |
| Datewise land/water masks | same database → `geotiffs/02_Datewise land water` | same, used when a job has no fresh classification |
| Water levels (10 stations) | same database → `meghna estuary/` | `demo/hydro.py` downloads + caches under `WORK_DIR/_hydro` |
| Basemap / WebGIS | Esri World Imagery satellite | Leaflet tile layer |
| Analyses | classification · channel cleaning · intertidal extent · lifespan | `gbm_delta` toolkit modules (`index_quality`, `river_mask`, `intertidal_detector`, `lifespan_legacy`) |

The job spec is a plain JSON body — `{roi?, index, threshold, years,
minScenes, pixelSizeM, minComponentPx, elongation, analyses}` — and every
result returns rendered map PNGs (with bounds, auto-overlaid in Leaflet),
time-series graphs, and download URLs for the GeoTIFF/CSV outputs.

## 5. Repo hygiene

`__pycache__/` is git-ignored via `demo/.gitignore` — please remove any
committed `__pycache__` folders:
`git rm -r --cached demo/__pycache__`.
