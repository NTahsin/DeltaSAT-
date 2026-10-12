# DeltaSAT

Organised, cloud-compatible Python modules for **Ganges–Brahmaputra–Meghna
(GBM) delta intertidal research**: tide-gauge analysis, satellite water-index
extraction, river/bankline/centerline mapping, and intertidal
morphodynamics (elevation, accommodation space, tidal prism, sediment mass,
DTMC lifespan).

This is a **drop-in update** for the previously released toolkit (v0.1.0).
Unzip this archive over your existing `gbm-delta-toolkit` folder — replaced
files overwrite cleanly, and modules unchanged from v0.1.0 (GEE download,
banklines, etc.) keep working without modification.

---

## 1. Package layout

gbm_delta/
├── config.py                 shared constants (class codes, seasons, CRS, stations)
├── utils.py                  small shared helpers (dates, Otsu, GeoTIFF I/O, IDW)
│
├── water_levels/             tide-gauge data analysis
│   ├── processor.py          *(v0.1.0, unchanged)* cleaning & QA
│   ├── extremes.py            *(v0.1.0, unchanged)* high/low water events
│   ├── datums.py              *(v0.1.0, unchanged)* tidal datums (MHW/MLW/…)
│   ├── surface_profiler.py    *(v0.1.0, unchanged)* water-surface profiles
│   ├── tidal_range.py         load_water_levels() CSV loader (refreshed)
│   └── stations.py            GBM gauge coordinates + coast distance (UPDATED)
│
├── gee/                      *(v0.1.0, unchanged)* Landsat/GEE download &
│                             index generation (cloud side only)
│
├── extraction/               water / river / intertidal extraction (local rasters)
│   ├── refine_pipeline.py     *(v0.1.0, unchanged)* index fusion pipeline
│   ├── interannual.py         *(v0.1.0, unchanged)*
│   ├── tidal_intertidal.py    *(v0.1.0, unchanged)*
│   ├── index_quality.py       *(v0.1.0, unchanged)*
│   ├── river_mask.py          water masks → river channels (UPDATED)
│   ├── intertidal_detector.py per-year intertidal classification (REAL IMPL.)
│   ├── intertidal_base.py     compat alias for IntertidalDetector (refreshed)
│   ├── intertidal_filter.py   connectivity filter for intertidal pixels
│   ├── validation.py          Otsu vs Planet NDWI validation (refreshed)
│   ├── planet_anyindex.py     Planet reference from *any* index (NEW)
│   └── planet_mosaic.py       Planet SR tile mosaic → water fraction (NEW)
│
├── channel/                  *(v0.1.0, unchanged)* banklines, centerlines,
│                             migration metrics
│
├── morphodynamics/           NEW SUBPACKAGE — intertidal morphodynamics
│   ├── elevation_v2.py        occurrence + elevation from WSE stacks
│   ├── elevation_v3.py        study-grid version (river masks, windowed reads)
│   ├── reconciler.py          frequency + JRC + datum reconciliation
│   ├── accommodation.py       accommodation depth & fill ratio
│   ├── accommodation_roi.py   per-ROI accommodation
│   ├── tidal_prism.py         intertidal tidal prism + Sen's slope
│   ├── sediment_mass.py       year-on-year sediment mass from dz
│   ├── stats.py               per-year class-area statistics
│   ├── stats_roi.py           per-ROI class-area statistics
│   ├── lifespan_markov.py     pooled DTMC + pixel-wise lifespan maps
│   ├── lifespan_v2.py         interval-based persistence + CIs + trends
│   ├── lifespan_roi.py        per-ROI interval analysis
│   ├── lifespan_local.py      annual transition matrices + lifespan maps
│   ├── lifespan_legacy.py     oldest Markov calculator (legacy encoding!)
│   ├── stationarity.py        Anderson–Goodman stationarity test
│   └── river_comparison.py    multi-river comparison & figures
│
└── viz/                      publication figures
    ├── datum_figures.py       *(v0.1.0, unchanged)*
    ├── channel_figures.py     *(v0.1.0, unchanged)*
    ├── tide_maps.py           *(v0.1.0, unchanged)*
    ├── style.py               house style — API of paper_style (refreshed)
    ├── along_channel.py       along-channel levels & asymmetry (UPDATED)
    └── map_gif.py             map GIF / subplot panels (NEW, from notebook)


*(Items marked "unchanged" are not duplicated in this archive — they are
already in your v0.1.0 tree.)*

---

## 2. What changed in v0.2.0

### Replaced (drop-in, same role, improved)
| Module | Change |
|---|---|
| `water_levels/stations.py` | 12-station table, prefix matching, haversine coast distance |
| `viz/along_channel.py` | package imports (no `sys.path` hacks), γ = T_fall/T_rise panels |
| `extraction/river_mask.py` | was `river_from_geotiff.py`; area + elongation river extraction |

### Real implementations replacing shims
| Module | Change |
|---|---|
| `extraction/intertidal_detector.py` | v0.1.0 shipped a reconstructed shim; this is the real per-year, SI-validated Otsu classifier |
| `extraction/intertidal_base.py` | compat re-export kept; subclasses see the real `_classify_year()` 3-tuple API |

### New: `morphodynamics` subpackage (16 modules)
The complete intertidal chain: elevation (2 variants) → reconciliation →
accommodation / prism / sediment → statistics → DTMC lifespan (4 variants) →
stationarity testing → multi-river comparison.

### Refreshed, API-compatible
`config.py`, `utils.py`, `viz/style.py` (was `paper_style.py`),
`water_levels/tidal_range.py`, `extraction/validation.py` (was
`otsu_validation.py`). All old function names and signatures are preserved;
only the import path changes to `gbm_delta.*`.

### Bug fixes applied
- Removed all `sys.path` hacks / flat imports → absolute package imports.
- `extraction/planet_anyindex.py` imported from the old flat
  `otsu_validation` module → now `gbm_delta.extraction.validation`.
- `morphodynamics/lifespan_local.py` was **truncated mid-statement** in the
  uploaded file → completed the annual-lifespan-map plotting loop and
  `return stats_df`.
- `viz/map_gif.py` (from the `export_map_overlay` notebook): fixed `draw`
  undefined when no overlay year, replaced undefined `colormaps[...]`
  lookups with `_resolve_cmap()` (cmocean optional, matplotlib fallback),
  and removed the deprecated `cm.get_cmap` call.
- `extraction/planet_mosaic.py`: pandas import moved to module top.
- Docstring escape-sequence warnings (Windows paths) fixed with raw strings.

### v0.2.1 — complete build + CSV ingestion
* **This archive is complete** — the v0.1.0 modules are included, so no
  overlay onto an older release is needed.
* `water_levels/processor.py` (drop-in update): `load_csv()` and
  `load_csv_folder()` on `WaterLevelIngestor` and `WaterLevelPipeline`
  ingest the raw per-station CSVs of the hydro-morpho GitHub database
  (NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho): semicolon-separated,
  UTF-8 BOM, day-first timestamps, gaps → NaN. Station display names are
  resolved automatically from the sibling `stations.csv`
  (`chandpur.csv` → `Chandpur 1640`). The Excel path is unchanged.
* `StationLocations` reads the database `stations.csv` as-is (auto
  separator, Latitude/Longitude, optional BTM_X/BTM_Y).
* `repo_water_levels.py` (toolkit root): bridge script — converts the
  database into the wide `reconstructed_water_level.csv` + station CSVs
  consumed by the analysis modules (`load_water_levels`, datums,
  along-channel figures) **without** re-running reconstruction.

---

## 3. Conventions (package-wide)

| Convention | Value |
|---|---|
| Class codes (rasters) | `0` = land, `1` = river, `2` = intertidal, `255` = nodata |
| WSE rasters | `999.0` = land marker, `NaN` = river |
| Seasons | wet = Jun–Oct, dry = Nov–May |
| Projected CRS | EPSG:32646 (UTM 46N) |
| Import side-effects | **none** — no module initialises GEE, plots, or touches the network at import time |

⚠ **Legacy exception:** `morphodynamics/lifespan_legacy.py` uses
`1 = intertidal, 0 = land, 2 = river`. It is kept for reference only —
prefer `lifespan_markov.py` / `lifespan_v2.py`.

---

## 4. Cloud vs. local

Every module runs in **both** environments:

| Task | Cloud (GEE/Codespaces) | Local laptop |
|---|---|---|
| Landsat index download | `gbm_delta.gee` (needs `earthengine-api`) | — download once, then local |
| Water/river/intertidal extraction | `gbm_delta.extraction` on downloaded rasters | same |
| Tide-gauge analysis | `gbm_delta.water_levels` (pure pandas) | same |
| Banklines / centerlines / migration | `gbm_delta.channel` (local GeoTIFFs) | same |
| Morphodynamics chain | `gbm_delta.morphodynamics` (pure numpy/rasterio) | same |
| Figures / GIFs | `gbm_delta.viz` | same |

Only the GEE download step requires the cloud (or authenticated
`earthengine-api`); **everything else is filesystem-based** and identical in
both environments.

---

## 5. Installation

```bash
pip install -r requirements.txt          # core
pip install -e ".[viz]"                  # figure extras
pip install -e ".[gee]"                  # Google Earth Engine side
```

or without installing, add the toolkit root to `PYTHONPATH` and import
`gbm_delta.*`.

---

## 6. Quick start

```python
from gbm_delta.water_levels.stations import enrich
from gbm_delta.water_levels.tidal_range import load_water_levels
from gbm_delta.extraction.river_mask import RiverFromGeotiff
from gbm_delta.morphodynamics.reconciler import IntertidalReconciler
from gbm_delta.morphodynamics.lifespan_v2 import Config, run_analysis

# 1) tide gauges
wl = load_water_levels("Reconstructed Water Level.csv")
sta = enrich(wl.columns)                       # coords + distance to coast

# 2) rivers from binary water masks
rx = RiverFromGeotiff(input_dir="water_masks", output_dir="river_masks",
                      min_area_px=56, elongation=20)
rx.run()

# 3) intertidal reconciliation (frequency + JRC + datums)
rec = IntertidalReconciler(channel_masks_dir="02_River_Extracted",
                           elevation_dir="bishop_taylor",
                           mhsw_dir="mhsw_grids", mlsw_dir="mlsw_grids",
                           output_dir="07_intertidal_reconciled")
rec.run(years=range(1996, 2021))

# 4) DTMC lifespan with CIs
cfg = Config()                                  # codes Land=0 River=1 Intertidal=2
run_analysis(Path("07_intertidal_reconciled"), Path("08_lifespan"),
             years=[1996, 1998, 2000, 2005, 2010, 2015, 2018], cfg=cfg)
```

Numbered end-to-end workflows: see **WORKFLOWS.md**.

---

## 7. Module index (new / updated modules)

### `gbm_delta.water_levels`
| Symbol | Purpose |
|---|---|
| `stations.STATIONS` | 12-gauge coordinate table (indexed by name) |
| `stations.station_frame(cols)` | coordinates for columns of a WL dataframe |
| `stations.enrich(cols)` | + `dist_coast_km`, sorted seaward→upstream |
| `stations.haversine(...)` | great-circle distance (km) |
| `tidal_range.load_water_levels(path)` | wide/long CSV → datetime-indexed DataFrame |

### `gbm_delta.extraction`
| Symbol | Purpose |
|---|---|
| `river_mask.RiverFromGeotiff(input_dir, output_dir, min_area_px=56, elongation=20.0, morph_open=1, skip_existing=True)` | water masks → river channels (`.run(year=None)`, `.process_file(f)`, `.summary()`) |
| `intertidal_detector.IntertidalDetector(river_dir, output_dir, si_dir=None, si_prefix='singularity_index_', si_threshold=0.5, min_scenes=2, ...)` | per-year intertidal classification, SI-validated Otsu |
| `intertidal_base.IntertidalDetector` | compat re-export (subclassing: `_classify_year()` → `(result, valid_count, water_count)`) |
| `intertidal_filter.filter_connected_intertidal(...)` / `.process_file` / `.process_directory` | drop disconnected intertidal pixels |
| `validation.otsu(arr)` | Otsu threshold |
| `validation.confusion_matrix(pred, ref)` / `binary_metrics(cm)` | accuracy assessment (PA/UA/F1/OA/κ) |
| `validation.block_bootstrap_ci(pred, ref, block_px, n_boot, valid=None)` | spatially blocked CIs |
| `validation.validate_fused_index(planet_ndwi, fused_index, ...)` | full Otsu-vs-Planet validation → `(cm, metrics)` |
| `planet_anyindex.validate_with_any_index(planet_raster, fused_index, ...)` | Planet reference from any index direction (midpoint/Otsu guidance) |
| `planet_mosaic.build_reference(sr_tiles, class_grid_path, ...)` | overlap-safe Planet mosaic → water fraction + coverage |

### `gbm_delta.morphodynamics`

**Elevation & extent**
| Symbol | Purpose |
|---|---|
| `elevation_v2.IntertidalElevationProcessor(wse_dir, datums_csv, station_coords, output_dir, *, crs_wl='EPSG:4326', idw_power=2.0, ...)` | per-year occurrence + Bishop-Taylor / waterline elevation |
| `elevation_v2.diagnose_station_matching(...)` | station↔raster match diagnostics |
| `elevation_v3.IntertidalElevationProcessor(..., buffer_m=..., mask_sample_every=1)` | study-grid version with river masks, windowed reads |
| `reconciler.IntertidalReconciler(channel_masks_dir, elevation_dir, mhsw_dir, mlsw_dir, output_dir, jrc_occurrence=None, min_scenes=5, ...)` | frequency + JRC + datum reconciliation → `intertidal_YYYY.tif` / `intertidal_h_YYYY.tif` |

**Volumes & budgets**
| Symbol | Purpose |
|---|---|
| `accommodation.AccommodationSpaceCalculator(AccommodationConfig)` | accommodation depth `a = max(MHSW − z, 0)`, fill ratio |
| `accommodation_roi.ROIAccommodationCalculator(ROIAccommodationConfig)` | per-ROI version (needs `roi_shapefile`, `name_field='name'`) |
| `tidal_prism.TidalPrismCalculator(TidalPrismConfig)` | intertidal tidal prism + Sen's slope trend; `reference_year` for normalised prism |
| `sediment_mass.SedimentMassCalculator(SedimentMassConfig)` | dz stacks → sediment mass (`rho_s=2650`, `porosity`, `sigma_z=0.25`) |

**Statistics**
| Symbol | Purpose |
|---|---|
| `stats.summarize_rasters(folder)` / `export_raster_summary(df, csv)` | per-year class-area table |
| `stats_roi.summarize_rois(raster_folder, roi_path)` / `intertidal_wide(df)` | per-ROI class areas, pivot table |

**Lifespan (DTMC residence time)**
| Symbol | Purpose |
|---|---|
| `lifespan_markov.IntertidalLifespanAnalyser(output_dir, years=None, occ_glob='occurrence_dry_{year}.tif', elev_glob='elevation_bt_{year}.tif', elev_method='bt', overwrite=False)` | pooled DTMC, pixel-wise P_II & lifespan maps, transition network figures |
| `lifespan_v2.Config` + `run_analysis(input_dir, output_dir, years, cfg)` | interval-based annualised persistence, Wilson + block-bootstrap CIs, trend tests |
| `lifespan_roi.run_roi_analysis(input_dir, roi_path, output_dir, ...)` | per-ROI interval analysis |
| `lifespan_local` | annual transition matrices + annual lifespan maps (repaired file) |
| `lifespan_legacy.LifespanCalculator(classified_dir, output_dir, pixel_size_m=30)` | legacy — encoding `1=intertidal` |
| `stationarity.run_stationarity(StationarityConfig, rois=None, roi_groups=None)` | Anderson–Goodman χ²/G² test of time-homogeneity (full delta / per ROI / channel groups) |

**Comparisons**
| Symbol | Purpose |
|---|---|
| `river_comparison.RiverIntertidalComparator(shapefiles, output_dir, years=None, name_field=None, buffer_m=500.0, occ_glob=..., elev_glob=...)` | clip per-river, per-river DTMC lifespan, time series + figures |

### `gbm_delta.viz`
| Symbol | Purpose |
|---|---|
| `style.apply_style()` / `style_ax()` / `label_panel()` / `legend_ax()` / palette `C` / `DELTA` | house style (API of old `paper_style`) |
| `along_channel.along_channel_figure(wl_csv, out=..., months=None)` | levels (MHSW/MSL/MLSW), γ asymmetry, annual γ — Lower Meghna vs Tentulia |
| `along_channel.limb_durations(series)` | mean rise/fall limb durations (h) |
| `map_gif.create_tif_gif_bar(...)` / `make_subplot(...)` / `create_subplot_figure(...)` | animated map GIFs + subplot panels, fixed colorbar, scale bar (cmocean optional) |

---

## 8. Compatibility notes

* **v0.1.0 → v0.2.0 renames** (import paths only, APIs preserved):
  `paper_style` → `gbm_delta.viz.style`; `otsu_validation` →
  `gbm_delta.extraction.validation`; `river_from_geotiff` →
  `gbm_delta.extraction.river_mask`; `intertidal_*` scripts →
  `gbm_delta.morphodynamics.*` (see layout above).
* Old flat-import notebooks keep working if you add the shim lines:
  ```python
  import gbm_delta.viz.style as paper_style
  import gbm_delta.extraction.validation as otsu_validation
  ```
* `intertidal_base.IntertidalDetector` subclasses: the real detector's
  `_classify_year()` returns a 3-tuple — update overrides accordingly.

## 9. License / citation
Research code for the GBM delta intertidal study. Cite the underlying
datasets: Landsat (USGS/NASA), PlanetScope (Planet Labs), JRC surface water
(Pekel et al. 2016), Bishop-Taylor et al. (2018/2019) intertidal methods.
