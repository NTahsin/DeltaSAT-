"""
pipeline.py — orchestrates the DeltaSAT analyses with the **gbm-toolkit**
(`gbm_delta` package, v0.3.0).

The backend resolves the toolkit automatically, in this order:

  1. env ``TOOLKIT_DIR``          (e.g. /abs/path/to/gbm-toolkit)
  2. sibling  ``../gbm-toolkit``  (repo layout: DeltaSAT-/{demo,gbm-toolkit})
  3. sibling  ``../gbm-delta-toolkit``   (older folder name)
  4. an installed ``gbm_delta``    (pip install -e ../gbm-toolkit)

Toolkit APIs used (all constructor signatures verified against v0.3.0):

  GEE   : gbm_delta.gee.rivermap.process_landsat_collection(
              start_date, end_date, roi, satellite, threshold_method, ...)
          → per-scene river_mask ImageCollection (MNDWI+AWEI fusion → MMSI →
            adaptive Otsu → post-process, all server-side; filterBounds,
            works for arbitrary ROIs — no WRS needed)
          gbm_delta.gee.landsat.GBMPipeline(polygon, wrs_path, wrs_row,
              start_date, end_date, years, index, cloud_cover_threshold,
              common_crs, common_scale)         (WRS fast-path, optional)
  Local : gbm_delta.extraction.river_mask.RiverFromGeotiff(
              input_dir, output_dir, min_area_px, elongation, ...)
          gbm_delta.extraction.intertidal_detector.IntertidalDetector(
              river_dir, output_dir, si_dir=None, min_scenes=2, ...)
          gbm_delta.morphodynamics.lifespan_legacy.LifespanCalculator(
              classified_dir, output_dir, pixel_size_m=30)
          gbm_delta.morphodynamics.elevation_v3.IntertidalElevationProcessor(
              wse_dir, datums_csv, station_coords, output_dir, *,
              crs_wl="EPSG:4326", years=None, overwrite=False)
          gbm_delta.morphodynamics.lifespan_markov.IntertidalLifespanAnalyser(
              output_dir, years=None, occ_glob=..., elev_glob=...,
              elev_method="bt", overwrite=False)
          gbm_delta.channel.banklines.BanklineExtractor(BanklineConfig(...))
          gbm_delta.channel.metrics.ChannelMetricsAnalyzer(
              MetricsConfig(...), rois=None, roi_name_field="name").run(years)
          gbm_delta.water_levels.surface_profiler.RiverSurfaceProfiler(
              geotiff_dir, wl_series, station_coords, output_dir, *,
              crs_wl="EPSG:4326", overwrite=False)
          gbm_delta.water_levels.processor.HarmonicReconstructor(
              min_coverage_pct=30, n_irls_iter=5, rayleigh_factor=1.0)

Two execution modes (env MOCK_MODE):
  MOCK_MODE=1 (default) → no GEE / no heavy deps; writes placeholder PNGs so
                          the front-end wiring can be tested end-to-end.
  MOCK_MODE=0            → imports gbm_delta and runs for real.

Honest "needs input" statuses are returned where a single web request cannot
supply the inputs (datums CSV for elevation, ROI polygons for migration).
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

MOCK = os.getenv("MOCK_MODE", "1") != "0"
TOOLKIT_DIR = os.getenv("TOOLKIT_DIR", "")
WORK_ROOT = Path(os.getenv("WORK_DIR", "./_jobs")).resolve()


# --------------------------------------------------------------------------- #
# toolkit resolution
# --------------------------------------------------------------------------- #
def _toolkit_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    cands = []
    if TOOLKIT_DIR:
        cands.append(Path(TOOLKIT_DIR))
    cands += [here.parent / "gbm-toolkit",           # repo layout
              here.parent / "gbm-delta-toolkit",     # older folder name
              here / "gbm-toolkit"]                  # inside demo/
    return [c for c in cands if (c / "gbm_delta" / "__init__.py").exists()]


_API = None


def _api() -> SimpleNamespace:
    """Import the toolkit once; return a namespace of verified symbols.

    Raises a RuntimeError with the exact resolution attempt list when the
    package cannot be found, so the job log states precisely what to fix.
    """
    global _API
    if _API is not None:
        return _API

    tried = [str(c) for c in _toolkit_candidates()]
    for c in _toolkit_candidates():                       # sibling first
        p = str(c)
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import gbm_delta
        from gbm_delta.extraction.intertidal_detector import IntertidalDetector
        from gbm_delta.extraction.river_mask import RiverFromGeotiff
        from gbm_delta.morphodynamics.lifespan_legacy import LifespanCalculator
        from gbm_delta.morphodynamics.elevation_v3 import IntertidalElevationProcessor
        from gbm_delta.morphodynamics.lifespan_markov import IntertidalLifespanAnalyser
        from gbm_delta.channel.metrics import ChannelMetricsAnalyzer, MetricsConfig
        from gbm_delta.channel.banklines import BanklineExtractor, BanklineConfig
        from gbm_delta.water_levels.surface_profiler import RiverSurfaceProfiler
        from gbm_delta.water_levels.processor import HarmonicReconstructor
        from gbm_delta.utils import otsu_threshold
        from gbm_delta.gee import rivermap

        _API = SimpleNamespace(
            source=f"gbm_delta {getattr(gbm_delta, '__version__', '?')} "
                   f"({', '.join(tried) or 'installed'})",
            IntertidalDetector=IntertidalDetector,
            RiverFromGeotiff=RiverFromGeotiff,
            LifespanCalculator=LifespanCalculator,
            IntertidalElevationProcessor=IntertidalElevationProcessor,
            IntertidalLifespanAnalyser=IntertidalLifespanAnalyser,
            ChannelMetricsAnalyzer=ChannelMetricsAnalyzer,
            MetricsConfig=MetricsConfig,
            BanklineExtractor=BanklineExtractor,
            BanklineConfig=BanklineConfig,
            RiverSurfaceProfiler=RiverSurfaceProfiler,
            HarmonicReconstructor=HarmonicReconstructor,
            otsu_threshold=otsu_threshold,
            rivermap=rivermap,
        )
        return _API
    except ImportError as exc:
        raise RuntimeError(
            f"gbm-toolkit not importable (looked in: {tried or 'sys.path'}; "
            f"last error: {exc}). Set TOOLKIT_DIR or place the gbm-toolkit "
            "folder next to demo/, or `pip install -e /path/to/gbm-toolkit`. "
            "All toolkit dependencies (numpy/pandas/rasterio/…) must be "
            "installed for MOCK_MODE=0."
        ) from exc


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
        ax.text(0.5, 0.42, subtitle or "MOCK output — connect GEE + gbm-toolkit for real results",
                ha="center", va="center", fontsize=15, color="#5f7b86", family="sans-serif")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        path.write_text(f"{title}\n{subtitle}\n")  # last-resort text stub


def _refine_river(spec, ws, log, input_dir: Path) -> Path:
    """Shape-based river refinement — replaces the old ChannelMaskRefiner.

    gbm_delta.extraction.river_mask.RiverFromGeotiff keeps elongated
    (channel-like) water bodies and drops ponds / small noise.
    """
    api = _api()
    log("RiverFromGeotiff: area + elongation refinement …")
    rx = api.RiverFromGeotiff(
        input_dir=str(input_dir),
        output_dir=str(ws / "03_river"),
        min_area_px=int(spec.get("minComponentPx", 500)),
        elongation=float(spec.get("elongation", 20.0)),
        skip_existing=False,
    )
    rx.run()
    return ws / "03_river"


def _read_gauges(spec, gauge_file):
    """Parse the uploaded gauge CSV/Excel into (wl_series, coords) — unchanged
    minimal adapter; the LSHA reconstruction path (spec.reconstruct) is
    applied afterwards by _reconstruct()."""
    import pandas as pd
    df = (pd.read_csv(gauge_file)
          if str(gauge_file).lower().endswith((".csv", ".tsv"))
          else pd.read_excel(gauge_file))
    cols = spec.get("gaugeCols") or {}
    missing = [c for c in ("lat", "lon", "lvl", "time") if c not in cols]
    if missing:
        raise ValueError(f"gaugeCols missing keys: {missing} "
                         f"(got {list(cols)})")
    sid = cols.get("id") or "station_id"
    if sid not in df.columns:
        df[sid] = "station_1"
    coords = (df[[sid, cols["lon"], cols["lat"]]].drop_duplicates()
              .rename(columns={sid: "station_id",
                               cols["lon"]: "lon", cols["lat"]: "lat"}))
    wl_series = {str(s): pd.Series(g[cols["lvl"]].values,
                                   index=pd.to_datetime(g[cols["time"]])).sort_index()
                 for s, g in df.groupby(sid)}
    wl_series = {s: ser[ser.index.notna()] for s, ser in wl_series.items()}
    return wl_series, coords


def _reconstruct(spec, wl_series, log):
    """Optional LSHA harmonic gap-filling of the uploaded gauges
    (spec['reconstruct'] = true). Uses gbm_delta.water_levels.processor.
    HarmonicReconstructor — the same UTide-equivalent least-squares fit the
    toolkit's WaterLevelPipeline uses internally."""
    if not spec.get("reconstruct"):
        return wl_series
    api = _api()
    import pandas as pd
    rec = api.HarmonicReconstructor(
        min_coverage_pct=float(spec.get("minCoveragePct", 30.0)))
    out = {}
    for name, ser in wl_series.items():
        parts, diag_all = [], []
        for year in sorted({ts.year for ts in ser.index}):
            recon, diag = rec.fit_reconstruct(ser, year=int(year),
                                              station=name)
            if diag.get("status") == "ok":
                parts.append(recon)
            diag_all.append(f"{year}:{diag.get('status')}")
        if parts:
            out[name] = pd.concat(parts).sort_index()
            log(f"  LSHA {name}: {'; '.join(diag_all)}")
        else:
            out[name] = ser
    return out


# --------------------------------------------------------------------------- #
# GEE export → river delineation (land–channel, non-river water removal)
# --------------------------------------------------------------------------- #
def _export_landsat_masks(spec, ws, log):
    """Per-year water masks for an arbitrary ROI — the old four-class chain
    (MNDWIExporter → SingularityIndexProcessor → OtsuBinaryClassifier) is one
    server-side call in the toolkit: gbm_delta.gee.rivermap.
    process_landsat_collection fuses MNDWI+AWEI, applies MMSI, adaptive
    Otsu and post-processing on the server, filtered by bounds.

    Landsat 5 serves years ≤ 2012, Landsat 8 years ≥ 2013 (no L7).
    """
    import ee
    import geemap
    api = _api()

    b = spec["roi"]
    y0, y1 = spec["years"]
    y0 = max(int(y0), 1984)
    y1 = min(int(y1), 2024)
    geom = ee.Geometry.Rectangle([b["w"], b["s"], b["e"], b["n"]])
    scale = int(spec.get("scale", 30))
    method = spec.get("threshold", "otsu")
    masks_dir = ws / "00_mndwi"
    masks_dir.mkdir(parents=True, exist_ok=True)

    col5 = (api.rivermap.process_landsat_collection(
        f"{y0}-01-01", f"{min(y1, 2012)}-12-31", roi=geom,
        satellite="LANDSAT_5", threshold_method=method)
        if y0 <= 2012 else None)
    col8 = (api.rivermap.process_landsat_collection(
        f"{max(y0, 2013)}-01-01", f"{y1}-12-31", roi=geom,
        satellite="LANDSAT_8", threshold_method=method)
        if y1 >= 2013 else None)

    for year in range(y0, y1 + 1):
        parts = []
        if col5 is not None and year <= 2012:
            parts.append(col5.filter(ee.Filter.calendarRange(year, year, "year")))
        if col8 is not None and year >= 2013:
            parts.append(col8.filter(ee.Filter.calendarRange(year, year, "year")))
        if not parts:
            continue
        year_col = parts[0] if len(parts) == 1 else parts[0].merge(parts[1])
        img = year_col.mosaic().unmask(0)
        out = masks_dir / f"water_mask_{year}.tif"
        log(f"  GEE: exporting {out.name} (scale {scale} m) …")
        geemap.ee_to_geotiff(ee_object=img, output=str(out),
                             resolution=scale,
                             bbox=[b["w"], b["s"], b["e"], b["n"]])
    return masks_dir


def _s2_binary_masks(spec, ws, log):
    """Sentinel-2 path: demo/gee.py exports MNDWI scenes; this binarises them
    locally with the toolkit's Otsu (gbm_delta.utils.otsu_threshold).
    """
    import numpy as np
    import rasterio
    api = _api()
    b = spec["roi"]
    y0, y1 = spec["years"]
    mndwi_dir = ws / "00_mndwi"
    mndwi_dir.mkdir(parents=True, exist_ok=True)

    import gee as _gee
    log("GEE: exporting Sentinel-2 MNDWI scenes …")
    _gee.export_s2_mndwi(b, y0, y1, str(mndwi_dir),
                         max_cloud=int(spec.get("maxCloud", 30)),
                         scale=int(spec.get("scale", 10)))

    binary_dir = ws / "02_binary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    for tif in sorted(mndwi_dir.glob("*.tif")):
        with rasterio.open(tif) as src:
            arr = src.read(1).astype("float32")
            profile = src.profile
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        thr = api.otsu_threshold(finite)
        binary = np.where(np.isfinite(arr) & (arr > thr), 1, 0).astype("uint8")
        profile.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(binary_dir / tif.name, "w", **profile) as dst:
            dst.write(binary, 1)
    return binary_dir


def _bounds_only_exporter(spec, ws, log) -> Path:
    """Build river_*.tif for the ROI (kept name for app-compatibility)."""
    sensors = spec.get("sensors", ["landsat"])
    if "s2" in sensors and "landsat" not in sensors:
        src = _s2_binary_masks(spec, ws, log)
    else:
        src = _export_landsat_masks(spec, ws, log)
    return _refine_river(spec, ws, log, src)


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
    log(f"toolkit: {_api().source}")
    river_dir = _bounds_only_exporter(spec, ws, log)
    return {"status": "ok", "files": _collect(river_dir, (".tif",))[:50] + _collect(out)}


def run_intertidal(spec, ws, log, gauge_file=None):
    """
    Intertidal dynamics = extent (+lifespan) always; elevation (+Markov
    lifespan) when gauge water levels and a tidal-datums CSV are available.

      extent    : IntertidalDetector(river_dir) → classified + occurrence per year
      lifespan  : LifespanCalculator(classified_dir) → transition/lifespan (no gauges)
      elevation : RiverSurfaceProfiler → *_wse.tif, then
                  IntertidalElevationProcessor(wse_dir, datums_csv,
                  station_coords, output_dir) → occurrence_dry_ /
                  elevation_bt_ / elevation_median_<year>.tif
      markov    : IntertidalLifespanAnalyser(output_dir=elevation_dir) on that grid
    """
    out = ws / "out" / "intertidal"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        for t in ("extent", "lifespan", "elevation"):
            _placeholder_png(out / f"intertidal_{t}.png", f"Intertidal {t}")
        return {"status": "ok (mock)", "files": _collect(out)}

    api = _api()
    log(f"toolkit: {api.source}")
    river_dir = ws / "03_river"
    if not any(river_dir.glob("*.tif")):
        river_dir = _bounds_only_exporter(spec, ws, log)
    years = list(range(spec["years"][0], spec["years"][1] + 1))
    classified_dir = ws / "04_intertidal"

    # 1) extent + per-year occurrence
    log("IntertidalDetector.run() …")
    api.IntertidalDetector(river_dir=str(river_dir),
                           output_dir=str(classified_dir),
                           min_scenes=int(spec.get("minScenes", 2))).run(years=years)

    # 2) lifespan (count-based, no gauges needed)
    status = ["extent: ok"]
    try:
        log("LifespanCalculator.run() …")
        api.LifespanCalculator(classified_dir=str(classified_dir),
                               output_dir=str(ws / "04b_lifespan"),
                               pixel_size_m=float(spec.get("pixelSizeM", 30.0))
                               ).run(years=years)
        status.append("lifespan: ok")
    except Exception as exc:
        status.append(f"lifespan: {exc}")

    # 3) elevation — needs WSE (gauges) + tidal datums CSV
    datums_csv = spec.get("datumsCsv")     # e.g. mhsw_mlsw_results.csv
    cols = spec.get("gaugeCols") or {}
    if gauge_file and datums_csv and {"lat", "lon", "lvl", "time"} <= set(cols):
        try:
            wl_series, coords = _read_gauges(spec, gauge_file)
            wl_series = _reconstruct(spec, wl_series, log)
            wse_dir = ws / "07_profile"
            log("RiverSurfaceProfiler.run() → WSE …")
            api.RiverSurfaceProfiler(geotiff_dir=str(river_dir),
                                     wl_series=wl_series,
                                     station_coords=coords,
                                     output_dir=str(wse_dir),
                                     crs_wl="EPSG:4326", overwrite=True).run()
            station_coords_csv = wse_dir / "station_coords.csv"
            coords.to_csv(station_coords_csv, index=False)

            elev_dir = ws / "04c_elevation"
            log("IntertidalElevationProcessor.run() …")
            api.IntertidalElevationProcessor(
                wse_dir=str(wse_dir), datums_csv=str(datums_csv),
                station_coords=str(station_coords_csv), output_dir=str(elev_dir),
                crs_wl="EPSG:4326", years=years, overwrite=True).run()
            status.append("elevation: ok")

            # 4) Markov lifespan refinement on the elevation grid
            try:
                log("IntertidalLifespanAnalyser.run() …")
                api.IntertidalLifespanAnalyser(output_dir=str(elev_dir),
                                               years=years,
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
    """Banklines (auto-extracted from the river masks) → channel metrics
    per ROI polygon."""
    out = ws / "out" / "migration"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        _placeholder_png(out / "migration_rate.png", "Channel migration",
                         "Transect-based lateral rate (m/yr)")
        return {"status": "ok (mock)", "files": _collect(out)}
    if not spec.get("rois"):
        return {"status": "needs ROI polygon(s) (shapefile/geojson) — "
                          "pass their path(s) via spec.rois", "files": []}

    api = _api()
    log(f"toolkit: {api.source}")
    river_dir = ws / "03_river"
    if not any(river_dir.glob("*.tif")):
        river_dir = _bounds_only_exporter(spec, ws, log)

    status = []
    # 1) banklines from the river masks (needs opencv)
    bankline_dir = ws / "05_banklines"
    if not any(bankline_dir.glob("*.tif")):
        try:
            cfg = api.BanklineConfig(
                input_dir=river_dir, output_dir=bankline_dir,
                min_component_pixels=int(spec.get("minComponentPx", 500)))
            log("BanklineExtractor.run() …")
            api.BanklineExtractor(cfg, rois=spec["rois"]).run()
            status.append("banklines: ok")
        except ImportError as exc:
            status.append(f"banklines: opencv missing ({exc})")
        except Exception as exc:
            status.append(f"banklines: {exc}")

    if not any(bankline_dir.glob("*.tif")):
        return {"status": "; ".join(status + ["no banklines produced"]),
                "files": _collect(out)}

    # 2) metrics — occurrence/elevation/lifespan dirs point at the intertidal
    #    outputs when they exist; ChannelMetricsAnalyzer skips years whose
    #    files are missing.
    try:
        elev_dir = ws / "04c_elevation"
        occ_dir = (ws / "04_intertidal"
                   if any((ws / "04_intertidal").glob("occurrence*.tif"))
                   else river_dir)
        life_dir = (elev_dir if any(elev_dir.glob("*.tif")) else river_dir)
        cfg = api.MetricsConfig(
            bankline_dir=bankline_dir,
            occurrence_dir=occ_dir,
            elevation_dir=elev_dir if any(elev_dir.glob("*.tif")) else occ_dir,
            lifespan_dir=life_dir,
            output_dir=ws / "06_metrics",
        )
        ana = api.ChannelMetricsAnalyzer(cfg, rois=spec["rois"])
        df = ana.run(years=range(spec["years"][0], spec["years"][1] + 1))
        df.to_csv(out / "channel_metrics.csv", index=False)
        status.append("metrics: ok")
    except Exception as exc:
        status.append(f"metrics: {exc}")

    return {"status": "; ".join(status) or "ok",
            "files": _collect(ws / "06_metrics") + _collect(out)}


def run_profile(spec, ws, gauge_file, log):
    """Water-surface elevation profile from uploaded gauges.
    spec.reconstruct = true → LSHA harmonic gap-filling first."""
    out = ws / "out" / "profile"
    out.mkdir(parents=True, exist_ok=True)
    if MOCK:
        _placeholder_png(out / "wse_profile.png", "Water surface profile",
                         "Water level vs along-channel distance")
        return {"status": "ok (mock)", "files": _collect(out)}
    if not gauge_file:
        return {"status": "needs uploaded gauge dataset (CSV/Excel)", "files": []}

    api = _api()
    log(f"toolkit: {api.source}")
    wl_series, coords = _read_gauges(spec, gauge_file)
    wl_series = _reconstruct(spec, wl_series, log)

    river_dir = ws / "03_river"
    if not any(river_dir.glob("*.tif")):
        river_dir = _bounds_only_exporter(spec, ws, log)
    api.RiverSurfaceProfiler(geotiff_dir=str(river_dir), wl_series=wl_series,
                             station_coords=coords, output_dir=str(out),
                             crs_wl="EPSG:4326", overwrite=True).run()
    note = " (+LSHA reconstruction)" if spec.get("reconstruct") else ""
    return {"status": "ok" + note, "files": _collect(out)}


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
