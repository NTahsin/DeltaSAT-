# DeltaSAT backend — GeoTIFF edition (no Google Earth Engine)

Connects the `toolkit.html` WebGIS front-end to the **gbm-delta-toolkit**
(`gbm_delta` package) — and serves that front-end itself:

```bash
python run.py    # starts the backend and opens http://localhost:8000/toolkit.html
```

The map shows the **data region** (dashed GeoTIFF coverage outline + the
tide-gauge stations with water-level readings, `GET /coverage`). The user
draws a **polygon ROI** inside it, selects the **years**, and runs one of
three analyses (or all). Everything is fetched from
`NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho` — no Earth Engine, no uploads.

## Buttons → gbm_delta modules (all clipped to the polygon ROI)

| Button | Analysis id | Chain |
|---|---|---|
| a · Intertidal area extents (maps + km² time series) | `intertidal` | `index_quality.otsu_threshold` → `river_mask.RiverFromGeotiff` → `IntertidalDetector` |
| b · Intertidal lifespan hotspot (map + time series) | `lifespan` | intertidal chain → `lifespan_v2.run_analysis` (windowed P_II hotspot maps + bootstrap CIs) |
| c · Channel migration rate (time series + bank-movement map) | `migration` | intertidal chain → `channel.migration.ChannelMigrationAnalyzer` (+ `bankline_from_classes` for the bank overlay map) |

## Job spec

```json
{
  "roi": {"type": "Polygon", "coordinates": [[[90.4,22.2],[91.1,22.2],[91.1,22.8],[90.4,22.8],[90.4,22.2]]]},
  "index": "mndwi",
  "threshold": "otsu",
  "years": [1996, 2026],
  "minScenes": 2, "pixelSizeM": 30,
  "minComponentPx": 56, "elongation": 20,
  "analyses": ["intertidal", "lifespan", "migration"]
}
```

`roi` is optional GeoJSON (WGS84); without it the full scene footprint is
used. Results contain `status`, `files` (GeoTIFF/CSV download URLs) and
`layers` (map PNGs with bounds — auto-overlaid in Leaflet — and
time-series graphs).

## Quick start

```bash
cd demo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

`MOCK_MODE=0` forces live, `MOCK_MODE=1` placeholders; default `auto` is
live whenever `gbm_delta` imports. Optional `GITHUB_TOKEN` raises the
GitHub API rate limit. The toolkit resolves the package from
`../gbm-delta-toolkit` (or `TOOLKIT_DIR`, or an installed `gbm_delta`).

## Caveats

- **Lifespan / migration need ≥ 2 years** of classified maps; with fewer,
  the status line says so and returns what exists.
- **Intertidal extent needs ≥ `minScenes`** (default 2) scenes per year.
- Scenes of different Landsat rows have different grids — draw the ROI
  within one row's footprint or extend scenes in the database.
