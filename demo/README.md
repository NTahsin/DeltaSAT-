# DeltaSAT backend — Earth Engine connector

Connects the static `toolkit.html` front-end to Google Earth Engine and your
existing processing **Modules**. It does three things:

1. **Live map preview** — `/preview` returns GEE XYZ tiles (MNDWI water
   composite + JRC occurrence) for the drawn ROI, shown directly on the Leaflet
   map. Fast, no export.
2. **Async analysis jobs** — `/jobs` runs the heavy chain
   (`MNDWIExporter → SingularityIndexProcessor → OtsuBinaryClassifier →
   ChannelMaskRefiner` → `IntertidalDetector` / `ChannelMetricsAnalyzer` /
   `RiverSurfaceProfiler`) in the background and serves the output figures.
3. **File serving** — output PNG/PDF/CSV/TIF stream back into the toolkit's 16:9
   result slots.

## Quick start (MOCK — test the wiring first, no GEE/data needed)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]" python-multipart matplotlib
uvicorn app:app --reload --port 8000
```

Open `toolkit.html` (served over HTTP — see below), draw an ROI, pick analyses,
**Run**. You'll get themed placeholder figures in each slot, proving the
front-end ↔ backend ↔ result-rendering path works.

## Real mode (GEE + your Modules)

```bash
pip install -r requirements.txt          # full scientific stack
earthengine authenticate                 # one-time, OR set a service account
MOCK_MODE=0 \
EE_PROJECT="your-gcp-project" \
MODULES_DIR="/abs/path/to/codes/Modules" \
WORK_DIR="./_jobs" \
uvicorn app:app --port 8000
```

Service-account auth instead of user auth:
```bash
EE_SERVICE_ACCOUNT="svc@project.iam.gserviceaccount.com" EE_KEY_FILE="/path/key.json"
```

## Serving the front-end

The map tiles and CDN libraries need HTTP (not `file://`). From the folder with
`toolkit.html` + `theme.css`:
```bash
python -m http.server 8080
# open http://localhost:8080/toolkit.html
```
In the toolkit, set the **API URL** field (top of the results panel) to
`http://localhost:8000`.

## Job spec (what the front-end POSTs)

`POST /jobs` is multipart: a `spec` JSON field + optional `gauges` file.

```json
{
  "roi": {"w":90.27,"s":22.50,"e":90.97,"n":23.27},
  "crs": "utm",
  "export_crs": "EPSG:32646",
  "sensors": ["landsat","s1"],
  "years": [1996, 2019],
  "maxCloud": 30,
  "analyses": ["delineation","intertidal","migration","profile"],
  "minComponentPx": 500,
  "wrs": null,
  "rois": null,
  "gaugeCols": {"lat":"Latitude","lon":"Longitude","lvl":"WL","time":"datetime","id":"Station ID"}
}
```

## What is wired vs. what you must confirm (current status)

| Step | Status |
|------|--------|
| GEE preview tiles (MNDWI/water/JRC) | **Done**, runs on `ee` only |
| MNDWI→PSI→Otsu→Refiner (delineation, non-river removal) | **Wired** to your classes with verified signatures |
| `IntertidalDetector` (yearly extent) | **Wired** |
| Intertidal **elevation + lifespan** | **Stub** — uncomment `IntertidalElevationProcessor` / `IntertidalLifespanAnalyser` once you confirm their constructor args |
| `ChannelMetricsAnalyzer` (migration) | **Wired**, but needs ROI polygon(s) + bankline/occurrence/elevation dirs (your `05_Banklines`, `intertidal_elevation`, …). Returns a clear "needs input" until supplied |
| `RiverSurfaceProfiler` (WSE) | **Wired** via a minimal adapter from the uploaded gauge file. For full harmonic reconstruction, switch to `WaterLevelPipeline + StationLocations` (your Excel/station layout) |

### Important caveats ATM

- **Global ROIs vs WRS path/row.** `MNDWIExporter` filters by explicit
  `wrs_path`/`wrs_row`. For arbitrary deltas the backend subclasses it to use
  `filterBounds` only (`BoundsOnlyExporter`). Pass `spec.wrs=[path,row]` to keep
  the strict (faster, scene-aligned) filter for a known delta.
- **Export is slow.** `export_all()` downloads each scene as a local GeoTIFF via
  `geemap`; a multi-decade ROI is minutes-to-hours and large on disk. Jobs are
  async for this reason. For production, export to GCS/Drive and read from there.
- **Migration & profile need more than a bbox.** Migration needs banklines + ROI
  polygons; the rigorous profile needs gauges distributed *along* the channel
  with overlapping time records — a count-in-box is necessary but not sufficient.
- **NetCDF gauges** are not parsed in the minimal adapter; provide CSV/Excel, or
  tell me your `.nc` variable names and I'll add a CF reader.
- I could not execute your Modules here (no GEE creds, GDAL, or your data), so
  the wired-but-not-mock paths should be smoke-tested on your machine; the
  table above flags exactly where to verify.
```
