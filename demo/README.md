# DeltaSAT backend — Earth Engine connector (gbm-toolkit edition)

Connects the static `toolkit.html` front-end to Google Earth Engine and the
**gbm-toolkit** (`gbm_delta` package, v0.3.0). It does three things:

1. **Live map preview** — `/preview` returns GEE XYZ tiles (MNDWI water
   composite + JRC occurrence) for the drawn ROI, shown directly on the
   Leaflet map. Fast, no export.
2. **Async analysis jobs** — `/jobs` runs the heavy chain in the background
   and serves the output figures: river delineation, intertidal extent +
   lifespan, intertidal elevation + Markov lifespan, channel migration
   (banklines → metrics), WSE profiles.
3. **File serving** — output PNG/PDF/CSV/TIF stream back into the toolkit's
   16:9 result slots.

## Quick start (MOCK — test the wiring first, no GEE/data needed)

```bash
cd demo
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]" python-multipart matplotlib
uvicorn app:app --reload --port 8000
```

Open `toolkit.html` over HTTP (see below), draw an ROI, pick analyses,
**Run**. You'll get themed placeholder figures in each slot, proving the
front-end ↔ backend ↔ result-rendering path works.

## Real mode (GEE + gbm-toolkit)

```bash
pip install -r requirements.txt          # full scientific stack
earthengine authenticate                 # one-time, OR set a service account

MOCK_MODE=0 \
EE_PROJECT="your-gcp-project" \
WORK_DIR="./_jobs" \
uvicorn app:app --port 8000
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

Or simply: `pip install -e ../gbm-toolkit` and drop it from the search
list entirely.

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
  "sensors": ["landsat"],
  "years": [1996, 2019],
  "maxCloud": 30,
  "analyses": ["delineation","intertidal","migration","profile"],
  "minComponentPx": 500,
  "threshold": "otsu",
  "scale": 30,
  "elongation": 20.0,
  "wrs": null,
  "rois": null,
  "datumsCsv": null,
  "reconstruct": false,
  "gaugeCols": {"lat":"Latitude","lon":"Longitude","lvl":"WL","time":"datetime","id":"Station ID"}
}
```

New keys (v0.3.0): `threshold` (otsu/sauvola/kmeans — GEE adaptive method),
`scale` (export resolution, m), `elongation` (RiverFromGeotiff shape filter),
`reconstruct` (true → LSHA harmonic gap-filling of the uploaded gauges before
the WSE profile / elevation steps), `datumsCsv` (path to a datums table such as
`mhsw_mlsw_results.csv` — required only for the elevation step).

## What is wired (current status — all against verified gbm_delta signatures)

| Step | Status | gbm-toolkit module |
|------|--------|--------------------|
| GEE preview tiles (MNDWI/water/JRC) | **Done**, runs on `ee` only | `demo/gee.py` |
| Delineation (water masks → river channels) | **Wired**: `rivermap.process_landsat_collection` (server-side MNDWI+AWEI → MMSI → adaptive Otsu; arbitrary ROI via filterBounds) → per-year mosaic export → `RiverFromGeotiff` shape refinement | `gee.rivermap`, `extraction.river_mask` |
| Sentinel-2 delineation | **Wired**: `gee.export_s2_mndwi` scenes → local Otsu (`utils.otsu_threshold`) → `RiverFromGeotiff` | `demo/gee.py`, `utils`, `extraction.river_mask` |
| `IntertidalDetector` (yearly extent) | **Wired** | `extraction.intertidal_detector` |
| Lifespan (count-based) | **Wired** | `morphodynamics.lifespan_legacy` |
| Intertidal **elevation** | **Wired** (no longer a stub): `RiverSurfaceProfiler` → WSE rasters → `IntertidalElevationProcessor(wse_dir, datums_csv, station_coords, output_dir, crs_wl, years, overwrite)` — needs gauge upload + `datumsCsv` | `water_levels.surface_profiler`, `morphodynamics.elevation_v3` |
| **Markov lifespan** | **Wired**: `IntertidalLifespanAnalyser(output_dir, years, elev_method="bt")` on the elevation grid | `morphodynamics.lifespan_markov` |
| Migration | **Wired**: banklines are now **auto-extracted** from the river masks (`BanklineExtractor`, needs opencv), then `ChannelMetricsAnalyzer(MetricsConfig(...), rois).run(years)`; needs ROI polygon(s) via `spec.rois` | `channel.banklines`, `channel.metrics` |
| WSE profile | **Wired** via the gauge adapter; `reconstruct: true` switches on LSHA harmonic gap-filling | `water_levels.surface_profiler`, `water_levels.processor` |

### Important caveats ATM

- **Arbitrary ROIs.** The old demo required WRS path/row for Landsat. The
  toolkit chain uses `filterBounds`, so any ROI works; set `spec.wrs` only if
  you want the stricter scene-aligned filter (GBMPipeline fast-path).
- **Export is slow.** A multi-decade ROI is minutes-to-hours and large on
  disk. Jobs are async for this reason. For production, export to GCS/Drive
  and read from there.
- **Elevation needs a datums table.** The `datumsCsv` input (e.g.
  `mhsw_mlsw_results.csv`) comes from the toolkit's datums analysis
  (`gbm_delta.water_levels.datums`) on your gauge record.
- **Migration needs ROI polygons** (`spec.rois`, path to shapefile/GeoJSON);
  a plain bbox is not enough.
- **NetCDF gauges** are not parsed in the minimal adapter; provide CSV/Excel.
- The toolkit import itself (numpy/pandas/rasterio) must be installed for
  `MOCK_MODE=0`; the backend reports the exact resolution attempts in the
  job log when it fails.

## Module mapping (old demo flat chain → gbm-toolkit)

| Old demo (flat Modules) | gbm-toolkit |
|---|---|
| `MNDWIExporter` | `gee.rivermap.process_landsat_collection` / `gee.landsat.GBMPipeline` |
| `SingularityIndexProcessor` + `OtsuBinaryClassifier` | server-side MMSI + adaptive Otsu in `process_landsat_collection`; local `utils.otsu_threshold` for S2 |
| `ChannelMaskRefiner` | `extraction.river_mask.RiverFromGeotiff` |
| `intertidal_detector` | `extraction.intertidal_detector` |
| `lifespan_calculator` | `morphodynamics.lifespan_legacy` |
| `intertidal_grided_elevation` | `morphodynamics.elevation_v3` |
| `intertidal_lifespan_markov` | `morphodynamics.lifespan_markov` |
| `river_surface_profiler` | `water_levels.surface_profiler` |
| `channel_metrics` | `channel.metrics` |
| `landsat_rivermap_pipeline` | `gee.rivermap` |
| `water_level_processor` | `water_levels.processor` (QC → gap-fill → LSHA) |
