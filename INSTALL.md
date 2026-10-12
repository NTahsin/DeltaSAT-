# DeltaSAT demo — complete folder (gbm-toolkit v0.3.0)

This is the **full `demo/` folder** of the DeltaSAT- web app: front-end
(HTML pages + `theme.css` + Leaflet WebGIS) and backend
(`app.py`, `pipeline.py`, `gee.py`, dataset scripts).

> If your repo still has a `demo/` folder, replace it wholesale with this
> folder (it contains everything, including the restored front-end).
> If the folder was deleted, just copy this folder back as `DeltaSAT-/demo/`.

## 1. Layout

```
DeltaSAT-/
├── demo/               ← this folder
│   ├── index.html            landing page
│   ├── toolkit.html          main WebGIS app (ROI, analyses, gauges)
│   ├── intertidal.html, markov.html, migration.html,
│   │   river-extraction.html, singularity-index.html,
│   │   water-indices.html    topic pages
│   ├── theme.css             shared stylesheet
│   ├── app.py                FastAPI server (health, stations, preview, jobs)
│   ├── pipeline.py           analysis wiring → gbm_delta toolkit (single analysis)
│   ├── hydro.py              auto water levels from the GitHub gauge database
│   ├── gee.py                Google Earth Engine init + preview tiles
│   ├── make_cloud_dataset.py, upload_dataset.py,
│   │   README_dataset.md     cloud-dataset utilities
│   ├── requirements.txt
│   └── INSTALL.md            (this file)
└── gbm-toolkit/        ← the gbm_delta Python package (v0.3.0), NOT in this zip
```

## 2. Install

1. Place the **gbm-toolkit** folder next to `demo/`
   (repo root: `DeltaSAT-/{demo, gbm-toolkit}`). The pipeline finds it
   automatically; alternatives: `pip install -e /path/to/gbm-toolkit`,
   or point `TOOLKIT_DIR` at it.
2. `pip install -r demo/requirements.txt`
   (numpy, pandas, rasterio, tifffile, opencv-headless, FastAPI, uvicorn …).
3. GEE credentials (real mode only):
   `earthengine authenticate` and set `EE_PROJECT="your-gcp-project"`.

## 3. Run

```bash
cd demo
# default: auto — live GEE when credentials exist, MOCK otherwise
uvicorn app:app --reload --port 8000

# force live / force mock
MOCK_MODE=0 EE_PROJECT="your-gcp-project" uvicorn app:app --port 8000
MOCK_MODE=1 uvicorn app:app --reload --port 8000
```

Open http://localhost:8000 → **Toolkit** page. The server's toolkit
source (mock / gbm-toolkit / installed) is shown by the *status* badge
and in every job log.

## 4. What runs, with what data (no uploads anywhere)

| Input | Source | Mechanism |
|---|---|---|
| Landsat 5/8 imagery | Google Earth Engine Python API | imported + initialised automatically at job start (`pipeline.resolve_mode`) |
| Water levels (10 stations) | GitHub database `NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho` → `meghna estuary/` | `demo/hydro.py` downloads + caches under `WORK_DIR/_hydro` |
| Basemap / WebGIS | Esri World Imagery satellite | Leaflet tile layer, taller map |
| Analysis | **Intertidal extent & elevation only** | `gbm_delta` toolkit modules (detector → lifespan → datums → WSE → elevation → Markov) |

The job spec is a plain JSON body — `{roi, crs, export_crs, sensors:["landsat"],
years, maxCloud, analyses:["intertidal"]}`. Tidal datums (`mhsw_mlsw_results.csv`)
are derived on the fly from the auto-fetched gauge series, so the elevation
step needs no extra files.

## 5. Repo hygiene

`__pycache__/` is git-ignored via `demo/.gitignore` — please remove any
committed `__pycache__` folders:
`git rm -r --cached demo/__pycache__`.
