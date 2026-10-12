# DeltaSAT backend — Earth Engine connector (gbm-toolkit edition)

Connects the `toolkit.html` WebGIS front-end to the **gbm-toolkit**
(`gbm_delta` package, v0.3.0). The demo runs **one analysis —
intertidal extent & elevation — with zero user uploads**:

1. **Landsat imagery** — Landsat 5/8 surface reflectance via the
   **Google Earth Engine Python API**, imported and initialised
   automatically when a job starts (`pipeline.resolve_mode`).
2. **Water levels** — hourly records of the 10 Meghna-estuary gauges from
   the GitHub hydro-morphological database
   (`NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho`, folder
   `meghna estuary`), downloaded automatically by `demo/hydro.py`
   (cached under `WORK_DIR/_hydro`).
3. **Map** — Esri World Imagery satellite basemap; the gauge stations are
   plotted on the Leaflet map automatically (`GET /stations`).

Endpoints: `GET /health` · `GET /stations` · `POST /preview` ·
`POST /jobs` (JSON spec, no multipart) · `GET /jobs/{id}` ·
`GET /jobs/{id}/file?path=…`.

## Quick start (MOCK — test the wiring, no GEE needed)

```bash
cd demo
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]"
MOCK_MODE=1 uvicorn app:app --reload --port 8000
```

Open `toolkit.html` over HTTP (see below), draw an ROI, **Run**. Themed
placeholder figures appear in the results panel, proving the
front-end ↔ backend ↔ result-rendering path works.

## Real mode (auto GEE)

```bash
pip install -r requirements.txt          # full scientific stack + toolkit
earthengine authenticate                 # one-time, OR use a service account

# default (recommended): auto — live GEE when EE credentials work,
# MOCK with a clear log line otherwise
uvicorn app:app --port 8000

# force live / force mock
MOCK_MODE=0 EE_PROJECT="your-gcp-project" uvicorn app:app --port 8000
MOCK_MODE=1 uvicorn app:app --port 8000
```

Service-account auth instead of user auth:
```bash
EE_SERVICE_ACCOUNT="svc@project.iam.gserviceaccount.com" EE_KEY_FILE="/path/key.json"
```

### Where the backend finds the toolkit

`pipeline.py` resolves the `gbm_delta` package automatically, in order:

1. env `TOOLKIT_DIR` — e.g. `/abs/path/to/gbm-toolkit`
2. sibling `../gbm-toolkit` (repo layout `DeltaSAT-/{demo,gbm-toolkit}`)
3. sibling `../gbm-delta-toolkit` (older folder name)
4. an installed `gbm_delta` (`pip install -e ../gbm-toolkit`)

## Serving the front-end

The Esri satellite tiles and CDN libraries need HTTP (not `file://`).
From the folder with `toolkit.html` + `theme.css`:
```bash
python -m http.server 8080
# open http://localhost:8080/toolkit.html
```
In the toolkit, set the **API URL** field (top of the results panel) to
`http://localhost:8000` and click **Connect** — the gauge stations of the
hydro database are then plotted on the map; the ones inside your ROI are
highlighted.

## Job spec (what the front-end POSTs)

`POST /jobs` is a plain JSON body — no file parts:

```json
{
  "roi": {"w":90.40,"s":22.20,"e":91.10,"n":22.80},
  "crs": "4326",
  "export_crs": "EPSG:32646",
  "sensors": ["landsat"],
  "years": [2015, 2024],
  "maxCloud": 20,
  "analyses": ["intertidal"]
}
```

Optional keys (sensible defaults): `threshold` (otsu), `scale` (30 m),
`minScenes` (2), `minComponentPx` (500), `elongation` (20),
`pixelSizeM` (30). `analyses` accepts only `"intertidal"`.

## Processing chain (all verified against gbm_delta v0.3.0 signatures)

| Step | gbm-toolkit module |
|------|--------------------|
| River masks for the drawn ROI (MNDWI+AWEI fusion → MMSI → adaptive Otsu, server-side, `filterBounds` → any ROI) | `gee.rivermap.process_landsat_collection` |
| River-shape refinement (drop ponds / aquaculture) | `extraction.river_mask.RiverFromGeotiff` |
| Yearly intertidal extent + occurrence | `extraction.intertidal_detector.IntertidalDetector` |
| Lifespan (transition matrix) | `morphodynamics.lifespan_legacy.LifespanCalculator` |
| Gauge fetch (GitHub → series + coords) | `demo/hydro.py` (auto, cached) |
| Tidal datums MHSW/MLSW → `mhsw_mlsw_results.csv` | `water_levels.datums.MHSWMLSWAnalyser` |
| Water-surface elevation rasters | `water_levels.surface_profiler.RiverSurfaceProfiler` |
| Intertidal elevation per year | `morphodynamics.elevation_v3.IntertidalElevationProcessor` |
| Markov lifespan on the elevation grid | `morphodynamics.lifespan_markov.IntertidalLifespanAnalyser` |
| GEE preview tiles (MNDWI + JRC occurrence) | `demo/gee.py` |

### Caveats

- **EE auto-import**: in `auto` mode the job log states exactly why EE was
  unavailable (no module / not authenticated / no project). `MOCK_MODE=0`
  fails hard instead, for pipelines where silence is unacceptable.
- **Export is slow** — a multi-decade ROI takes minutes to hours; jobs are
  async for this reason.
- **Gauge coverage** — the hydro database has 10 Meghna-estuary stations;
  if none falls inside/near the ROI (±0.35°), the elevation step reports a
  clear `skipped` status and extent/lifespan results are still returned.
- Station CSVs (~10–16 MB each) are downloaded once and cached under
  `WORK_DIR/_hydro`; repeat runs are offline.
