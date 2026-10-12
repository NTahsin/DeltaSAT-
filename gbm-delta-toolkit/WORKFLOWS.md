# WORKFLOWS.md — numbered end-to-end workflows (v0.2.0)

Workflows are numbered **in processing order**. WF1–WF7 are unchanged from
the first release; WF8–WF13 are new in v0.2.0 and build on their outputs.
Every step after the GEE download (WF2) is **local-filesystem based** —
identical on cloud and laptop.

> Shared conventions: class codes `0` land / `1` river / `2` intertidal /
> `255` nodata; WSE rasters `999` = land, `NaN` = river; wet season
> Jun–Oct, dry season Nov–May; CRS EPSG:32646.

---

## WF1 — Tide-gauge record → datums & extremes *(v0.1.0)*

```
raw gauge CSV → processor.py (QA/cleaning) → extremes.py (HW/LW events)
             → datums.py (MHW, MLW, MSL, MHSW, MLSW per station)
             → surface_profiler.py (along-channel water-surface profiles)
             → stations.enrich(columns)  (coords + distance to coast)
```

* Inputs: gauge CSVs (e.g. *Reconstructed Water Level.csv*).
* Outputs: per-station datum table (used by WF8), event tables.
* Cloud/local: pure pandas — identical anywhere.
* **Raw database → reconstructed series (v0.2.1).** The pipeline that
  produced the reconstructed record is `processor.py`: QC → smart
  gap-fill → **LSHA harmonic reconstruction** (`HarmonicReconstructor`,
  identical mathematics to UTide — Rayleigh-criterion constituent
  selection + robust IRLS fit, pure NumPy/SciPy, no compiled
  dependency) → validation (RMSE/MAE/R², scatter & KDE plots) → 5-min
  export. It now ingests the NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho
  CSVs directly:

  ```python
  from gbm_delta.water_levels.processor import (WaterLevelPipeline,
                                                StationLocations)
  locs = StationLocations("<repo>/meghna estuary/stations.csv")
  pipe = WaterLevelPipeline(locs, output_dir="output")
  pipe.load_csv_folder("<repo>/meghna estuary")   # auto-names from stations.csv
  rec = pipe.run()          # {station: reconstructed 5-min series}
  ```

  To skip re-running the reconstruction and just use the already
  reconstructed record, run `python repo_water_levels.py <repo> --out
  water_levels` instead.

## WF2 — Landsat via GEE → per-scene water indices *(v0.1.0, cloud)*

```
AOI + date range → gbm_delta.gee → per-scene NDWI/MNDWI/AWEI/etc. rasters
```

* Requires `earthengine-api` + authentication; download once, work locally
  afterwards.
* Outputs: per-scene index GeoTIFFs feeding WF3–WF5.

## WF3 — Index quality & fusion *(v0.1.0)*

```
per-scene indices → index_quality.py → refine_pipeline.py / interannual.py
```

* Chooses the best-performing index per scene/year; fuses to a single
  water/land product per year.

## WF4 — Binary water masks → river channels *(updated in v0.2.0)*

```
water_mask_YYYY-MM-DD.tif → gbm_delta.extraction.river_mask.RiverFromGeotiff
                          → river_YYYY-MM-DD.tif
```

```python
from gbm_delta.extraction.river_mask import RiverFromGeotiff
rx = RiverFromGeotiff(input_dir="water_masks", output_dir="river_masks",
                      min_area_px=56,      # ≈0.05 km² at 30 m
                      elongation=20.0)     # perimeter²/area; keep elongated blobs
rx.run()              # or rx.run(year=2005) / rx.process_file(...) / rx.summary()
```

* Removes ponds/lakes (round blobs) and small noise; keeps channel-like
  water.

## WF5 — Intertidal detection per year *(real implementation in v0.2.0)*

```
river masks + per-scene indices + singularity index
    → gbm_delta.extraction.intertidal_detector.IntertidalDetector
    → per-year intertidal classification
    → intertidal_filter (connectivity to permanent water)
    → intertidal_filtered_{year}.tif   (0/1/2, 255 nodata)
```

```python
from gbm_delta.extraction.intertidal_detector import IntertidalDetector
det = IntertidalDetector(river_dir="04_River_Extracted",
                         output_dir="05_Intertidal",
                         si_dir="singularity_index",
                         si_threshold=0.5, min_scenes=2)
```

## WF6 — Banklines, centerlines, migration *(v0.1.0)*

```
river masks per year → gbm_delta.channel → banklines, centerlines,
                       migration rates, width / sinuosity metrics
```

## WF7 — Validation against Planet *(updated in v0.2.0)*

```
Planet SR tiles (+UDM2) ─┬→ planet_mosaic.build_reference → water fraction
                         └→ planet_anyindex.validate_with_any_index
fused index (Otsu) ────────→ validation.validate_fused_index
                            → confusion matrix, PA/UA/F1/OA/κ, block-bootstrap CIs
```

```python
from gbm_delta.extraction.validation import validate_fused_index
cm, met = validate_fused_index(planet_ndwi="planet_ndwi.tif",
                               fused_index="fused_2019.tif",
                               out_prefix="val_2019")
```

* `planet_mosaic` builds an overlap-safe NDWI water-fraction reference on
  the classification grid from many tiles.
* `planet_anyindex` handles rasters where water is *not* simply
  "index > 0" — it learns the direction and threshold from Landsat-water
  cores (note the partial circularity; inspect the exported masks).

---

# NEW in v0.2.0 — intertidal morphodynamics chain

## WF8 — Intertidal elevation from WSE stacks

```
per-scene WSE rasters (999=land, NaN=river) + datum table (WF1)
    → morphodynamics.elevation_v2.IntertidalElevationProcessor
      (Bishop-Taylor primary, waterline-median supplementary)
    or elevation_v3 (study grid + river masks, windowed reads — large sites)
    → occurrence_dry_{year}.tif, elevation_bt_{year}.tif, mhsw/mlsw grids
```

```python
from gbm_delta.morphodynamics.elevation_v2 import IntertidalElevationProcessor
proc = IntertidalElevationProcessor(wse_dir="03_WSE", datums_csv="datums.csv",
                                     station_coords=..., output_dir="06_Elevation")
```

* Use `diagnose_station_matching()` first if datum stations don't line up
  with the raster grid (CRS issues).

## WF9 — Reconciliation → definitive intertidal maps

```
channel masks (WF4) + elevation (WF8) + MHSW/MLSW grids + JRC occurrence (optional)
    → morphodynamics.reconciler.IntertidalReconciler
    → intertidal_{year}.tif (0/1) + intertidal_h_{year}.tif (elevation)
    → intertidal_summary.csv
```

* Rules: ≥`min_scenes` observations, inundation frequency in
  [`f_low`, `f_high`], JRC occurrence < threshold, and MLSW ≤ z ≤ MHSW.
* Then apply `extraction.intertidal_filter` for connectivity if desired →
  `intertidal_filtered_{year}.tif` (0/1/2) used by WF12–WF13.

## WF10 — Accommodation space & tidal prism

```
intertidal_h + mhsw/mlsw + occurrence (WF8/WF9)
    → accommodation.AccommodationSpaceCalculator   (a = max(MHSW − z, 0))
    → accommodation_roi.ROIAccommodationCalculator (per ROI polygon)
    → tidal_prism.TidalPrismCalculator (V_prism, normalised P, Sen's slope)
```

```python
from gbm_delta.morphodynamics.accommodation import (AccommodationConfig,
                                                    AccommodationSpaceCalculator)
from gbm_delta.morphodynamics.tidal_prism import (TidalPrismConfig,
                                                  TidalPrismCalculator)
acc = AccommodationSpaceCalculator(AccommodationConfig(
    elevation_dir="07_intertidal_reconciled", output_dir="08_accommodation"))
acc.run()
prism = TidalPrismCalculator(TidalPrismConfig(
    elevation_dir="07_intertidal_reconciled", output_dir="09_prism",
    reference_year=1996))
prism.run()
```

* Prism ≡ accommodation volume numerically; interpret hydrodynamically
  (water volume exchanged per tide).

## WF11 — Sediment mass budget

```
intertidal_h stacks (WF9) + occurrence masks
    → sediment_mass.SedimentMassCalculator
    → dz_{year}.tif, sediment mass per interval, cumulative CSV
```

* `rho_s=2650 kg/m³` default; set `porosity` for wet bulk density or
  `density_mode="grain"` for dry grain mass; `sigma_z=0.25` m elevation
  uncertainty propagated.

## WF12 — Intertidal lifespan (DTMC) + stationarity test

```
intertidal_filtered_{year}.tif (0/1/2)
    ├─ lifespan_markov.IntertidalLifespanAnalyser   pooled DTMC, P_II maps
    ├─ lifespan_v2.run_analysis                     intervals + Wilson/
    │                                                block-bootstrap CIs + trends
    ├─ lifespan_roi.run_roi_analysis                per-ROI intervals
    ├─ lifespan_local                               annual transition matrices
    └─ stationarity.run_stationarity                Anderson–Goodman test
```

```python
from gbm_delta.morphodynamics.lifespan_v2 import Config, run_analysis
from gbm_delta.morphodynamics.stationarity import (StationarityConfig,
                                                   run_stationarity)
cfg = Config()          # Land=0 River=1 Intertidal=2, window 15 px, 500 boots
run_analysis(Path("07_intertidal_reconciled"), Path("10_lifespan"),
             years=[1996, 1998, 2000, 2005, 2010, 2015, 2018], cfg=cfg)

res = run_stationarity(StationarityConfig(
    raster_dir=Path("07_intertidal_reconciled"),
    years=[1996, 1998, 2000, 2005, 2010, 2015, 2018],
    out_dir=Path("11_stationarity")),
    rois=rois, roi_groups={"Main channel": ["roi_a"],
                           "Side channel": ["roi_b"]})
```

* Stationarity output: χ²/G², df, p, Cramér's V + verdict per region —
  justifies (or rejects) the time-homogeneous Markov assumption used by
  the lifespan modules.

## WF13 — Multi-river comparison & area statistics

```
channel polygons (WF6) + occurrence/elevation stacks (WF8)
    → river_comparison.RiverIntertidalComparator   per-river clip, DTMC
    → stats.summarize_rasters / stats_roi.summarize_rois
    → along_channel_figure / map_gif (viz)
```

```python
from gbm_delta.morphodynamics.river_comparison import RiverIntertidalComparator
cmp_ = RiverIntertidalComparator(shapefiles=["meghna.shp", "tentulia.shp"],
                                 output_dir="12_rivers",
                                 name_field="name", buffer_m=500)
cmp_.run()
```

* Produces per-river area/elevation time series, per-river lifespan maps,
  and comparison figures.

---

## Figure layer (any point in the chain)

| Figure | Module |
|---|---|
| Along-channel levels & γ asymmetry | `viz.along_channel.along_channel_figure(wl_csv, out)` |
| Animated annual map GIF / subplot panels | `viz.map_gif.create_tif_gif_bar(...)` / `create_subplot_figure(...)` |
| Datums, channels, tide maps | `viz.datum_figures` / `channel_figures` / `tide_maps` *(v0.1.0)* |

All figure modules share `viz.style` (the old `paper_style` API).
