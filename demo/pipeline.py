"""
pipeline.py — orchestrates the existing DeltaSAT processing Modules.

It maps a toolkit job spec to the real classes, using the exact constructor
signatures found in your code:

  GEE (landsat_rivermap_pipeline.py):
    MNDWIExporter(polygon, wrs_path, wrs_row, start_date, end_date, output_dir,
                  cloud_cover_threshold, common_crs, common_scale)
    SingularityIndexProcessor(mndwi_dir, output_dir)
    OtsuBinaryClassifier(psi_dir, output_dir, width_dir, ...)
    ChannelMaskRefiner(binary_dir, output_dir, min_component_px, ...)
  Local:
    IntertidalDetector(river_dir, output_dir, min_scenes, pixel_size_m, ...)
    ChannelMetricsAnalyzer(MetricsConfig(...), rois).run(years)
    WaterLevelPipeline(StationLocations(...)).run() + RiverSurfaceProfiler(...)

Two execution modes (env MOCK_MODE):
  MOCK_MODE=1 (default)  → no GEE / no heavy deps; writes placeholder PNGs so the
                           front-end wiring can be tested end-to-end immediately.
  MOCK_MODE=0            → imports your Modules from MODULES_DIR and runs for real.

Honest notes are inline where a step needs inputs a single web request cannot
supply (ROI shapefile for migration, station/Excel layout for the harmonic
water-level reconstruction). Those return a clear "needs input" status rather
than crashing.
"""
from __future__ import annotations
import os
import sys
import glob
import traceback
from pathlib import Path

MOCK = os.getenv("MOCK_MODE", "1") != "0"
MODULES_DIR = os.getenv("MODULES_DIR", "")
WORK_ROOT = Path(os.getenv("WORK_DIR", "./_jobs")).resolve()

if MODULES_DIR and MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def workspace(job_id: str) -> Path:
    d = WORK_ROOT / job_id
    for sub in ("00_mndwi", "01_psi", "02_binary", "03_river",
                "04_intertidal", "05_banklines", "06_metrics",
                "07_profile", "08_waterlevel", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _collect(out_dir: Path, exts=(".png", ".pdf", ".csv", ".tif")) -> list[str]:
    files = []
    for e in exts:
        files += [str(p) for p in out_dir.rglob(f"*{e}")]
    return sorted(files)


def _placeholder_png(path: Path, title: str, subtitle: str = "") -> None:
    """Tidal-Earth themed placeholder so the UI has something to render in MOCK mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 9), dpi=60)
        fig.patch.set_facecolor("#f5f0e6")
        ax.set_facecolor("#f5f0e6")
        ax.axis("off")
        ax.text(0.5, 0.56, title, ha="center", va="center",
                fontsize=30, color="#2f3b3a", family="sans-serif", weight="bold")
        ax.text(0.5, 0.42, subtitle or "MOCK output — connect GEE + Modules for real results",
                ha="center", va="center", fontsize=15, color="#5f7b86", family="sans-serif")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        path.write_text(f"{title}\n{subtitle}\n")  # last-resort text stub


# --------------------------------------------------------------------------- #
# GEE export → river delineation  (land–channel + non-river water removal)
# --------------------------------------------------------------------------- #
def _bounds_only_exporter(spec, ws, log):
    """
    Build river_*.tif for the ROI. MNDWI scenes come from Sentinel-2 (default,
    10 m) or Landsat (30 m) depending on spec['sensors']; the rest of the chain
    (singularity → Otsu → refine) is shared.

    NOTE on global ROIs: the Landsat MNDWIExporter filters by an explicit WRS
    path/row. For arbitrary deltas we drop that filter (filterBounds only) via a
    subclass. Set spec['wrs']=[path,row] to keep the strict filter. Sentinel-2
    has no WRS — it always uses filterBounds.
    """
    import ee
    from landsat_rivermap_pipeline import (
        MNDWIExporter, SingularityIndexProcessor,
        OtsuBinaryClassifier, ChannelMaskRefiner,
    )
    b = spec["roi"]
    geom = ee.Geometry.Rectangle([b["w"], b["s"], b["e"], b["n"]])
    y0, y1 = spec["years"]
    sensors = spec.get("sensors", ["landsat", "s2"])
    mndwi_dir = str(ws / "00_mndwi")
    cloud = int(spec.get("maxCloud", 30))

    if "s2" in sensors:
        log("GEE: exporting Sentinel-2 MNDWI scenes …")
        import gee as _gee
        _gee.export_s2_mndwi(b, y0, y1, mndwi_dir, max_cloud=cloud, scale=10)
    else:
        log("GEE: exporting Landsat MNDWI scenes …")
        common_crs = spec.get("export_crs", "EPSG:32646")

        class BoundsOnlyExporter(MNDWIExporter):
            def _base_filter(self, collection_id):
                return (ee.ImageCollection(collection_id)
                        .filterDate(self.start_date, self.end_date)
                        .filterBounds(self.polygon)
                        .filter(ee.Filter.lt("CLOUD_COVER", self.cloud_cover_threshold)))

        wrs = spec.get("wrs")
        Exporter = MNDWIExporter if wrs else BoundsOnlyExporter
        wrs_path, wrs_row = (wrs or [0, 0])
        exp = Exporter(polygon=geom, wrs_path=wrs_path, wrs_row=wrs_row,
                       start_date=f"{y0}-01-01", end_date=f"{y1}-12-31",
                       output_dir=mndwi_dir, cloud_cover_threshold=cloud,
                       common_crs=common_crs, common_scale=30)
        exp.export_all()

    log("Singularity index …")
    SingularityIndexProcessor(mndwi_dir=mndwi_dir,
                              output_dir=str(ws / "01_psi")).process_all()
    log("Otsu binary classification …")
    OtsuBinaryClassifier(psi_dir=str(ws / "01_psi" / "psi"),
                         output_dir=str(ws / "02_binary"),
                         width_dir=str(ws / "01_psi" / "width")).process_all()
    log("Channel-mask refine (remove non-river water) …")
    ChannelMaskRefiner(binary_dir=str(ws / "02_binary"),
                       output_dir=str(ws / "03_river"),
                       min_component_px=int(spec.get("minComponentPx", 500))).process_all()
    return ws / "03_river"


# --------------------------------------------------------------------------- #
# per-analysis runners
# --------------------------------------------------------------------------- #
def run_delineation(spec, ws, log):
    out = ws / "out" / "delineation"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        _placeholder_png(out / "river_mask.png", "Land–channel delineation",
                         "River retained · non-river water removed")
        return {"status": "ok (mock)", "files": _collect(out)}
    river_dir = _bounds_only_exporter(spec, ws, log)
    return {"status": "ok", "files": _collect(river_dir, (".tif",))[:50] + _collect(out)}


def run_intertidal(spec, ws, log, gauge_file=None):
    """
    Intertidal dynamics = extent (+lifespan) always; elevation (+Markov lifespan)
    when gauge water levels and tidal datums are available.

      extent    : IntertidalDetector(river_dir) → classified + occurrence per year
      lifespan  : LifespanCalculator(classified_dir) → transition/lifespan (no gauges)
      elevation : RiverSurfaceProfiler → *_wse.tif, then v3
                  IntertidalElevationProcessor(wse_dir, datums_csv, station_coords)
                  → occurrence_dry_/elevation_bt_/elevation_median_<year>.tif
      markov    : IntertidalLifespanAnalyser(output_dir=elevation_dir) on that grid
    """
    out = ws / "out" / "intertidal"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        for t in ("extent", "lifespan", "elevation"):
            _placeholder_png(out / f"intertidal_{t}.png", f"Intertidal {t}")
        return {"status": "ok (mock)", "files": _collect(out)}

    from intertidal_detector import IntertidalDetector
    from lifespan_calculator import LifespanCalculator

    river_dir = ws / "03_river"
    if not any(river_dir.glob("*.tif")):
        river_dir = _bounds_only_exporter(spec, ws, log)
    years = list(range(spec["years"][0], spec["years"][1] + 1))
    classified_dir = ws / "04_intertidal"

    # 1) extent + per-year occurrence
    log("IntertidalDetector.run() …")
    IntertidalDetector(river_dir=str(river_dir), output_dir=str(classified_dir),
                       min_scenes=int(spec.get("minScenes", 2))).run(years=years)

    # 2) lifespan (count-based, no gauges needed)
    status = ["extent: ok"]
    try:
        log("LifespanCalculator.run() …")
        LifespanCalculator(classified_dir=str(classified_dir),
                           output_dir=str(ws / "04b_lifespan"),
                           pixel_size_m=float(spec.get("pixelSizeM", 30.0))).run(years=years)
        status.append("lifespan: ok")
    except Exception as exc:
        status.append(f"lifespan: {exc}")

    # 3) elevation — needs WSE (gauges) + tidal datums CSV
    datums_csv = spec.get("datumsCsv")        # mhsw_mlsw_results.csv (from mhsw_mlsw_analysis.py)
    cols = spec.get("gaugeCols") or {}
    if gauge_file and datums_csv and {"lat", "lon", "lvl", "time"} <= set(cols):
        try:
            import pandas as pd
            from river_surface_profiler import RiverSurfaceProfiler
            from intertidal_grided_elevation import IntertidalElevationProcessor

            df = (pd.read_csv(gauge_file)
                  if str(gauge_file).lower().endswith((".csv", ".tsv"))
                  else pd.read_excel(gauge_file))
            sid = cols.get("id") or "station_id"
            if sid not in df.columns:
                df[sid] = "station_1"
            coords = (df[[sid, cols["lon"], cols["lat"]]].drop_duplicates()
                      .rename(columns={sid: "station_id", cols["lon"]: "lon", cols["lat"]: "lat"}))
            wl_series = {str(s): pd.Series(g[cols["lvl"]].values,
                                           index=pd.to_datetime(g[cols["time"]])).sort_index()
                         for s, g in df.groupby(sid)}
            wse_dir = ws / "07_profile"
            log("RiverSurfaceProfiler.run() → WSE …")
            RiverSurfaceProfiler(geotiff_dir=str(river_dir), wl_series=wl_series,
                                 station_coords=coords, output_dir=str(wse_dir),
                                 crs_wl="EPSG:4326", overwrite=True).run()
            station_coords_csv = wse_dir / "station_coords.csv"
            coords.to_csv(station_coords_csv, index=False)

            elev_dir = ws / "04c_elevation"
            log("IntertidalElevationProcessor.run() …")
            IntertidalElevationProcessor(
                wse_dir=str(wse_dir), datums_csv=str(datums_csv),
                station_coords=str(station_coords_csv), output_dir=str(elev_dir),
                crs_wl="EPSG:4326", years=years, overwrite=True).run()
            status.append("elevation: ok")

            # 4) Markov lifespan refinement on the elevation grid
            try:
                from intertidal_lifespan_markov import IntertidalLifespanAnalyser
                log("IntertidalLifespanAnalyser.run() …")
                IntertidalLifespanAnalyser(output_dir=str(elev_dir), years=years,
                                           elev_method="bt").run()
                status.append("markov-lifespan: ok")
            except Exception as exc:
                status.append(f"markov-lifespan: {exc}")
        except Exception as exc:
            status.append(f"elevation: error {exc}")
    else:
        status.append("elevation: needs gauge upload + datums CSV "
                      "(mhsw_mlsw_results.csv) + mapped lat/lon/level/time columns")

    files = (_collect(classified_dir) + _collect(ws / "04b_lifespan")
             + _collect(ws / "04c_elevation") + _collect(out))
    return {"status": "; ".join(status), "files": files}


def run_migration(spec, ws, log):
    out = ws / "out" / "migration"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        _placeholder_png(out / "migration_rate.png", "Channel migration",
                         "Transect-based lateral rate (m/yr)")
        return {"status": "ok (mock)", "files": _collect(out)}
    # Needs banklines + a ROI definition. Banklines come from BanklineExtractor;
    # ChannelMetricsAnalyzer then needs occurrence/elevation/lifespan dirs too.
    if not spec.get("rois"):
        return {"status": "needs ROI polygon(s) (shapefile/geojson) and bankline/"
                          "occurrence/elevation dirs — see README", "files": []}
    from channel_metrics import ChannelMetricsAnalyzer, MetricsConfig
    cfg = MetricsConfig(
        bankline_dir=ws / "05_banklines",
        occurrence_dir=ws / "04_intertidal",
        elevation_dir=ws / "04_intertidal",
        lifespan_dir=ws / "04_intertidal",
        output_dir=ws / "06_metrics",
    )
    ana = ChannelMetricsAnalyzer(cfg, rois=spec["rois"])
    df = ana.run(years=range(spec["years"][0], spec["years"][1] + 1))
    df.to_csv(out / "channel_metrics.csv", index=False)
    return {"status": "ok", "files": _collect(ws / "06_metrics") + _collect(out)}


def run_profile(spec, ws, gauge_file, log):
    out = ws / "out" / "profile"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        _placeholder_png(out / "wse_profile.png", "Water surface profile",
                         "Water level vs along-channel distance")
        return {"status": "ok (mock)", "files": _collect(out)}
    if not gauge_file:
        return {"status": "needs uploaded gauge dataset (CSV/Excel/NetCDF)", "files": []}
    import pandas as pd
    from river_surface_profiler import RiverSurfaceProfiler
    # Minimal adapter: build wl_series + station_coords from the uploaded gauge file.
    # (The full harmonic reconstruction path uses WaterLevelPipeline + StationLocations
    #  with your station/Excel layout — see README to switch to that.)
    df = pd.read_csv(gauge_file) if str(gauge_file).lower().endswith((".csv", ".tsv")) \
        else pd.read_excel(gauge_file)
    cols = spec.get("gaugeCols", {})
    sid = cols.get("id") or "station_id"
    if sid not in df.columns:
        df[sid] = "station_1"
    coords = (df[[sid, cols["lon"], cols["lat"]]]
              .drop_duplicates()
              .rename(columns={sid: "station_id",
                               cols["lon"]: "lon", cols["lat"]: "lat"}))
    wl_series = {}
    for s, g in df.groupby(sid):
        ser = pd.Series(g[cols["lvl"]].values,
                        index=pd.to_datetime(g[cols["time"]]))
        wl_series[str(s)] = ser.sort_index()
    river_dir = ws / "03_river"
    if not any(river_dir.glob("*.tif")):
        river_dir = _bounds_only_exporter(spec, ws, log)
    prof = RiverSurfaceProfiler(geotiff_dir=str(river_dir), wl_series=wl_series,
                                station_coords=coords, output_dir=str(out),
                                crs_wl="EPSG:4326", overwrite=True)
    prof.run()
    return {"status": "ok", "files": _collect(out)}


# --------------------------------------------------------------------------- #
# entry point used by the job runner
# --------------------------------------------------------------------------- #
def run_job(job_id: str, spec: dict, gauge_file: str | None, log) -> dict:
    ws = workspace(job_id)
    results = {}
    runners = {
        "delineation": lambda: run_delineation(spec, ws, log),
        "intertidal":  lambda: run_intertidal(spec, ws, log, gauge_file),
        "migration":   lambda: run_migration(spec, ws, log),
        "profile":     lambda: run_profile(spec, ws, gauge_file, log),
    }
    for a in spec.get("analyses", []):
        if a not in runners:
            continue
        try:
            log(f"▶ {a} starting (mock={MOCK})")
            results[a] = runners[a]()
            log(f"✓ {a} done: {results[a]['status']}")
        except Exception as exc:
            log(f"✗ {a} failed: {exc}")
            results[a] = {"status": f"error: {exc}",
                          "trace": traceback.format_exc(), "files": []}
    return results
