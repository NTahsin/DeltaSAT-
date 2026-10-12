"""
pipeline.py — DeltaSAT analysis orchestrator (gbm-toolkit edition).

The demo runs **one** analysis — *Intertidal extent & elevation* — using
the ``gbm_delta`` toolkit modules. No user uploads: imagery comes from
the Landsat collections on Google Earth Engine (imported and initialised
automatically when a job starts), water levels come from the GitHub
hydro-morphological database of the Meghna estuary
(NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho, folder ``meghna estuary``)
via demo/hydro.py.

Toolkit resolution (env ``TOOLKIT_DIR`` → sibling ``../gbm-toolkit`` →
sibling ``../gbm-delta-toolkit`` → installed ``gbm_delta``).

Toolkit APIs used (signatures verified against v0.3.0):

  gbm_delta.gee.rivermap.process_landsat_collection(
      start_date, end_date, roi, satellite, threshold_method, ...)
      → per-scene river_mask ImageCollection (MNDWI+AWEI fusion → MMSI →
        adaptive Otsu → post-processing, all server-side)
  gbm_delta.extraction.river_mask.RiverFromGeotiff(
      input_dir, output_dir, min_area_px, elongation, ...)
  gbm_delta.extraction.intertidal_detector.IntertidalDetector(
      river_dir, output_dir, si_dir=None, min_scenes=2, ...)
  gbm_delta.morphodynamics.lifespan_legacy.LifespanCalculator(
      classified_dir, output_dir, pixel_size_m=30)
  gbm_delta.water_levels.datums.MHSWMLSWAnalyser(wl_csv, output_dir, ...)
      → mhsw_mlsw_results.csv (tidal datums per station/year/season)
  gbm_delta.water_levels.surface_profiler.RiverSurfaceProfiler(
      geotiff_dir, wl_series, station_coords, output_dir, *, crs_wl, ...)
  gbm_delta.morphodynamics.elevation_v3.IntertidalElevationProcessor(
      wse_dir, datums_csv, station_coords, output_dir, *, crs_wl, years)
  gbm_delta.morphodynamics.lifespan_markov.IntertidalLifespanAnalyser(
      output_dir, years=None, elev_method="bt", ...)

Execution modes (env ``MOCK_MODE``):
  "1"        → always MOCK   (placeholder PNGs, no GEE / no heavy deps)
  "0"        → always live   (job fails hard without EE credentials)
  unset/"auto" → try to import + initialise Earth Engine at job start;
                 live when it succeeds, MOCK with a log note otherwise.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

TOOLKIT_DIR = os.getenv("TOOLKIT_DIR", "")
WORK_ROOT = Path(os.getenv("WORK_DIR", "./_jobs")).resolve()


# --------------------------------------------------------------------------- #
# mode resolution (auto-import Earth Engine when a job starts)
# --------------------------------------------------------------------------- #
def _ee_ready(log) -> bool:
    """Import and initialise the Earth Engine Python API once."""
    try:
        import gee                                   # demo helper
        gee.init_ee()
        log("Earth Engine Python API imported and initialised ✓")
        return True
    except Exception as exc:
        log(f"Earth Engine unavailable ({exc.__class__.__name__}: {exc})")
        return False


def resolve_mode(log) -> str:
    """mock | live — per job, from MOCK_MODE (default: auto-probe EE)."""
    forced = os.getenv("MOCK_MODE", "auto").strip().lower()
    if forced == "1":
        return "mock"
    if forced == "0":
        _ee_ready(log)          # fail fast / show the exact error
        return "live"
    return "live" if _ee_ready(log) else "mock"


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
    """Import the toolkit once; return a namespace of verified symbols."""
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
        from gbm_delta.water_levels.surface_profiler import RiverSurfaceProfiler
        from gbm_delta.water_levels.datums import MHSWMLSWAnalyser
        from gbm_delta.gee import rivermap

        _API = SimpleNamespace(
            source=f"gbm_delta {getattr(gbm_delta, '__version__', '?')} "
                   f"({', '.join(tried) or 'installed'})",
            IntertidalDetector=IntertidalDetector,
            RiverFromGeotiff=RiverFromGeotiff,
            LifespanCalculator=LifespanCalculator,
            IntertidalElevationProcessor=IntertidalElevationProcessor,
            IntertidalLifespanAnalyser=IntertidalLifespanAnalyser,
            RiverSurfaceProfiler=RiverSurfaceProfiler,
            MHSWMLSWAnalyser=MHSWMLSWAnalyser,
            rivermap=rivermap,
        )
        return _API
    except ImportError as exc:
        raise RuntimeError(
            f"gbm-toolkit not importable (looked in: {tried or 'sys.path'}; "
            f"last error: {exc}). Set TOOLKIT_DIR or place the gbm-toolkit "
            "folder next to demo/, or `pip install -e /path/to/gbm-toolkit`. "
            "All toolkit dependencies (numpy/pandas/rasterio/…) must be "
            "installed for live runs."
        ) from exc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def workspace(job_id: str) -> Path:
    d = WORK_ROOT / job_id
    for sub in ("00_mndwi", "03_river", "04_intertidal", "04b_lifespan",
                "04c_elevation", "04d_datums", "07_profile", "out"):
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


# --------------------------------------------------------------------------- #
# GEE export → river masks for the ROI
# --------------------------------------------------------------------------- #
def _export_landsat_masks(spec, ws, log):
    """Per-year river masks for the drawn ROI via the toolkit's GEE module.

    gbm_delta.gee.rivermap.process_landsat_collection fuses MNDWI+AWEI,
    applies MMSI, adaptive Otsu and post-processing on the server, filtered
    by bounds. Landsat 5 serves years ≤ 2012, Landsat 8 years ≥ 2013.
    (Earth Engine is already imported + initialised by resolve_mode().)
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


def _refine_river(spec, ws, log, input_dir: Path) -> Path:
    """Shape-based river refinement: keep elongated (channel-like) water,
    drop ponds / aquaculture / noise."""
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


def _bounds_only_exporter(spec, ws, log) -> Path:
    """Build river_*.tif for the drawn ROI."""
    src = _export_landsat_masks(spec, ws, log)
    return _refine_river(spec, ws, log, src)


# --------------------------------------------------------------------------- #
# the one analysis: intertidal extent & elevation
# --------------------------------------------------------------------------- #
def run_intertidal(spec, ws, log, mock: bool):
    """
    extent    : IntertidalDetector(river_dir) → classified + occurrence per year
    lifespan  : LifespanCalculator(classified_dir) → transition/lifespan
    elevation : water levels from the GitHub hydro database (hydro.py) →
                MHSWMLSWAnalyser → tidal datums;
                RiverSurfaceProfiler → *_wse.tif;
                IntertidalElevationProcessor → occurrence_dry_ /
                elevation_bt_ / elevation_median_<year>.tif
    markov    : IntertidalLifespanAnalyser on the elevation grid
    """
    out = ws / "out" / "intertidal"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
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

    # 2) lifespan (transition-matrix, no gauges needed)
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

    # 3) elevation — water levels auto-fetched from the GitHub database
    try:
        import hydro
        wl_series, coords = hydro.series_near(spec["roi"], log=log)

        # 3a) tidal datums (MHSW/MLSW) from the auto-fetched series
        datums_dir = ws / "04d_datums"
        wl_csv = hydro.write_wide_csv(wl_series, datums_dir / "water_levels.csv")
        log("MHSWMLSWAnalyser.run() → tidal datums …")
        api.MHSWMLSWAnalyser(wl_csv=str(wl_csv),
                            output_dir=str(datums_dir)).run()
        datums_csv = datums_dir / "mhsw_mlsw_results.csv"

        # 3b) water-surface elevation (WSE) grids
        wse_dir = ws / "07_profile"
        log("RiverSurfaceProfiler.run() → WSE …")
        api.RiverSurfaceProfiler(geotiff_dir=str(river_dir),
                                 wl_series=wl_series,
                                 station_coords=coords,
                                 output_dir=str(wse_dir),
                                 crs_wl="EPSG:4326", overwrite=True).run()
        station_coords_csv = wse_dir / "station_coords.csv"
        coords.to_csv(station_coords_csv, index=False)

        # 3c) intertidal elevation per year
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
        status.append(f"elevation: skipped ({exc})")

    files = (_collect(classified_dir) + _collect(ws / "04b_lifespan")
             + _collect(ws / "04c_elevation") + _collect(out))
    return {"status": "; ".join(status), "files": files}


# --------------------------------------------------------------------------- #
# entry point used by the job runner
# --------------------------------------------------------------------------- #
def run_job(job_id: str, spec: dict, log) -> dict:
    ws = workspace(job_id)
    mock = resolve_mode(log) == "mock"
    results = {}
    for a in spec.get("analyses", ["intertidal"]):
        if a != "intertidal":
            continue
        try:
            log(f"▶ {a} starting (mock={mock})")
            results[a] = run_intertidal(spec, ws, log, mock)
            log(f"✓ {a} done: {results[a]['status']}")
        except Exception as exc:
            log(f"✗ {a} failed: {exc}")
            results[a] = {"status": f"error: {exc}",
                          "trace": traceback.format_exc(), "files": []}
    return results
