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
│   ├── app.py                FastAPI server (serves pages + /run API)
│   ├── pipeline.py            analysis wiring → gbm_delta toolkit
│   ├── gee.py                Google Earth Engine helpers
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
# MOCK mode — no GEE needed, exercises the full pipeline wiring
uvicorn app:app --reload --port 8000

# Real mode
MOCK_MODE=0 EE_PROJECT="your-gcp-project" uvicorn app:app --port 8000
```

Open http://localhost:8000 → **Toolkit** page. The server's toolkit
source (mock / gbm-toolkit / installed) is shown by the *status* badge
and in every job log.

## 4. What the front-end sends (`toolkit.html`)

The control panel builds a job spec consumed by `pipeline.py`; every
key now maps to a real toolkit parameter:

| UI control | Spec key | Toolkit use |
|---|---|---|
| Threshold method (Otsu/Sauvola/k-means) | `threshold` | GEE thresholding |
| Pixel size (m) | `scale`, `pixelSizeM` | exports, LifespanCalculator |
| Min patch removed (ha) | `minComponentPx` | RiverFromGeotiff refinement (converted ha→px) |
| Min elongation | `elongation` | river-shape filter |
| Min scenes / year | `minScenes` | IntertidalDetector |
| LSHA reconstruct toggle | `reconstruct` | water-level gap-filling before profiles |
| Server datums CSV path | `datumsCsv` | intertidal elevation validation |
| Server ROIs path | `rois` | bankline/migration per-ROI runs |

Gauge uploads (CSV/XLSX/NetCDF) remain available on the *Water-level data*
panel; with ≥ N gauges inside the ROI the water-surface profile unlocks.

## 5. Repo hygiene

`__pycache__/` is git-ignored via `demo/.gitignore` — please remove any
committed `__pycache__` folders:
`git rm -r --cached demo/__pycache__`.
