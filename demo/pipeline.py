"""
pipeline.py — DeltaSAT analysis orchestrator (no-GEE, polygon-ROI edition).

All imagery is **local-filesystem based**: no Google Earth Engine, no user
uploads. Water-index and land/water GeoTIFFs of the Meghna estuary live in
the GitHub hydro-morphological database and are auto-downloaded + cached
by demo/geodata.py; tide-gauge records come from the same repository via
demo/hydro.py. The user draws a polygon ROI inside the data region (or none
at all) and every analysis is clipped to it.

Analyses (spec["analyses"] — the three toolkit buttons):

  intertidal  per-year intertidal area extents (km²): classified maps +
              area time series
                chain: Otsu classification (index_quality.otsu_threshold)
                       → RiverFromGeotiff channel cleaning
                       → IntertidalDetector
  lifespan    intertidal lifespan hotspot maps over the period + interval
              time series (windowed P_II maps, bootstrap CIs)
                → morphodynamics.lifespan_v2.run_analysis
  migration   channel migration-rate time series + map of bank movement
                → channel.migration.ChannelMigrationAnalyzer
                  (+ bankline_from_classes for the bank-movement map)

Toolkit resolution (env ``TOOLKIT_DIR`` → sibling ``../gbm-delta-toolkit``
→ sibling ``../gbm-toolkit`` → installed ``gbm_delta``).

Execution modes (env ``MOCK_MODE``):
  "1" → always MOCK (placeholder PNGs) · "0" → force live ·
  unset/"auto" → live when the toolkit imports, MOCK with a log note.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

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
    cands += [here.parent / "gbm-delta-toolkit",   # repo layout
              here.parent / "gbm-toolkit",          # older folder name
              here / "gbm-delta-toolkit"]           # inside demo/
    return [c for c in cands if (c / "gbm_delta" / "__init__.py").exists()]


_API = None


def _api() -> SimpleNamespace:
    """Import the toolkit once; return a namespace of verified symbols."""
    global _API
    if _API is not None:
        return _API
    tried = [str(c) for c in _toolkit_candidates()]
    for c in _toolkit_candidates():
        p = str(c)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import gbm_delta
        from gbm_delta.extraction.index_quality import otsu_threshold
        from gbm_delta.extraction.river_mask import RiverFromGeotiff
        from gbm_delta.extraction.intertidal_detector import IntertidalDetector
        from gbm_delta.morphodynamics.lifespan_v2 import Config, run_analysis
        from gbm_delta.channel.migration import (MetricsConfig,
                                                  ChannelMigrationAnalyzer,
                                                  bankline_from_classes)

        _API = SimpleNamespace(
            source=f"gbm_delta {getattr(gbm_delta, '__version__', '?')} "
                   f"({', '.join(tried) or 'installed'})",
            otsu_threshold=otsu_threshold,
            RiverFromGeotiff=RiverFromGeotiff,
            IntertidalDetector=IntertidalDetector,
            lifespan_Config=Config,
            lifespan_run=run_analysis,
            MetricsConfig=MetricsConfig,
            ChannelMigrationAnalyzer=ChannelMigrationAnalyzer,
            bankline_from_classes=bankline_from_classes,
        )
        return _API
    except ImportError as exc:
        raise RuntimeError(
            f"gbm-delta-toolkit not importable (looked in: {tried or 'sys.path'}; "
            f"last error: {exc}). Set TOOLKIT_DIR, place the gbm-delta-toolkit "
            "folder next to demo/, or `pip install -e /path/to/gbm-delta-toolkit`. "
            "All toolkit dependencies (numpy/pandas/rasterio/…) must be "
            "installed for live runs."
        ) from exc


def toolkit_available() -> bool:
    try:
        _api()
        return True
    except Exception:
        return False


def resolve_mode(log) -> str:
    """mock | live — no Earth Engine involved, only toolkit importability."""
    forced = os.getenv("MOCK_MODE", "auto").strip().lower()
    if forced == "1":
        return "mock"
    if forced == "0":
        _api()          # fail fast with the real import error
        return "live"
    try:
        _api()
        return "live"
    except Exception as exc:
        log(f"gbm_delta toolkit unavailable ({exc.__class__.__name__}: {exc}) "
            "— running in MOCK mode")
        return "mock"


# --------------------------------------------------------------------------- #
# workspace + small helpers
# --------------------------------------------------------------------------- #
def workspace(job_id: str) -> Path:
    d = WORK_ROOT / job_id
    for sub in ("01_index", "01_binary", "02_staged", "03_river",
                "03_river_bin", "04_intertidal", "05_lifespan",
                "06_migration", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _collect(out_dir: Path, exts=(".png", ".pdf", ".csv", ".tif")) -> list[str]:
    files = []
    for e in exts:
        files += [str(p) for p in out_dir.rglob(f"*{e}")]
    return sorted(files)


def _placeholder_png(path: Path, title: str, subtitle: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 9), dpi=60)
        fig.patch.set_facecolor("#f5f0e6")
        ax.set_facecolor("#f5f0e6")
        ax.axis("off")
        ax.text(0.5, 0.56, title, ha="center", va="center",
                fontsize=30, color="#2f3b3a", family="sans-serif",
                weight="bold")
        ax.text(0.5, 0.42, subtitle or "MOCK output — install the gbm_delta "
                "toolkit (and its deps) for real results",
                ha="center", va="center", fontsize=15, color="#5f7b86",
                family="sans-serif")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        path.write_text(f"{title}\n{subtitle}\n")


def _date_of(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return stem


def _year_of(name: str) -> int:
    """First 4-digit year in a filename. Works for full dates
    (binary_1996-02-09.tif) AND yearly maps (intertidal_1996.tif) —
    do NOT use ``_date_of(...)[:4]`` for the latter ('inte' → crash)."""
    return int(re.search(r"(\d{4})", Path(name).stem).group(1))


def _bounds_wgs84(path):
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds,
                                  densify_pts=21)
        return [[b[1], b[0]], [b[3], b[2]]]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# polygon ROI clipping
# --------------------------------------------------------------------------- #
def _clip_raster(src_path, dst_path, geom):
    """Clip one GeoTIFF to a WGS84 GeoJSON Polygon (rasterio.mask).

    Float rasters get NaN outside the polygon; uint8 rasters get 255.
    """
    import numpy as np
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import transform_geom

    with rasterio.open(src_path) as src:
        g = (transform_geom("EPSG:4326", src.crs, geom)
             if src.crs is not None else geom)
        if np.issubdtype(src.dtypes[0], np.floating):
            arr, tfm = mask(src, [g], crop=True, filled=False)
            a = arr[0].filled(np.nan).astype("float32")
            profile = src.profile.copy()
            profile.update(dtype="float32", nodata=np.nan, transform=tfm,
                           width=a.shape[1], height=a.shape[0])
        else:
            nd = src.nodata if src.nodata is not None else 255
            arr, tfm = mask(src, [g], crop=True, filled=True, nodata=nd)
            a = arr[0]
            profile = src.profile.copy()
            profile.update(nodata=nd, transform=tfm,
                           width=a.shape[1], height=a.shape[0])
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(a, 1)
    return dst_path


def _maybe_clip(src_path, dst_path, roi):
    """Clip when a polygon ROI is given, else copy."""
    if roi and roi.get("type") == "Polygon":
        return _clip_raster(src_path, dst_path, roi)
    shutil.copy2(src_path, dst_path)
    return dst_path


# --------------------------------------------------------------------------- #
# figure rendering
# --------------------------------------------------------------------------- #
def _render_map_png(tif_path, out_png, kind="classified", roi=None, title=""):
    """GeoTIFF → themed PNG (+ WGS84 bounds for the Leaflet overlay)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import rasterio

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
    b = _bounds_wgs84(tif_path)

    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")

    if kind == "continuous":            # float lifespan maps (NaN = nodata)
        v = arr[np.isfinite(arr)]
        vmin, vmax = (np.nanpercentile(v, [2, 98]) if v.size else (0, 1))
        cmap = plt.get_cmap("terrain").copy()
        cmap.set_bad("#efe9dc")
        if b:
            extent = [b[0][1], b[1][1], b[0][0], b[1][0]]
            im = ax.imshow(arr.astype(float), cmap=cmap, vmin=vmin,
                           vmax=vmax, extent=extent, origin="upper")
            if roi:
                w, s_, e, n = roi["w"], roi["s"], roi["e"], roi["n"]
                if (e > b[0][1] and w < b[1][1] and n > b[0][0]
                        and s_ < b[1][0]):
                    ax.set_xlim(w, e); ax.set_ylim(s_, n)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        else:
            im = ax.imshow(arr.astype(float), cmap=cmap, vmin=vmin,
                           vmax=vmax, origin="upper")
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=.035, label="Lifespan (yr)")
    else:
        if kind == "classified":        # 0 land · 1 river · 2 intertidal · 255
            disp = np.where(arr == 255, 3, arr)
            cmap = ListedColormap(["#d9c7a3", "#5F7B86", "#7BA05B",
                                   "#efe9dc"])
            vmax = 3
        else:                           # binary: 0 land · 1 water · 255
            disp = np.where(arr == 255, 2, arr)
            cmap = ListedColormap(["#d9c7a3", "#5F7B86", "#efe9dc"])
            vmax = 2
        if b:
            extent = [b[0][1], b[1][1], b[0][0], b[1][0]]
            ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, extent=extent,
                      origin="upper")
            if roi:
                w, s_, e, n = roi["w"], roi["s"], roi["e"], roi["n"]
                if (e > b[0][1] and w < b[1][1] and n > b[0][0]
                        and s_ < b[1][0]):
                    ax.set_xlim(w, e); ax.set_ylim(s_, n)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        else:
            ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
            ax.set_xlabel("x"); ax.set_ylabel("y")

    ax.set_title(title or Path(tif_path).name, fontsize=10, color="#5C6664")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png),
            "bounds": b, "kind": "map"}


def _render_area_png(csv_path, out_png):
    """intertidal_summary.csv → per-year area (km²) time-series figure."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path).sort_values("year")
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")
    if "km2_intertidal" in df:
        ax.plot(df["year"], df["km2_intertidal"], "o-", color="#7BA05B",
                label="Intertidal")
    if "km2_river" in df:
        ax.plot(df["year"], df["km2_river"], "o-", color="#5F7B86",
                label="Permanent river")
    if "km2_land" in df:
        ax.plot(df["year"], df["km2_land"], "o-", color="#C2AE92",
                label="Land")
    ax.set_xlabel("Year"); ax.set_ylabel("Area (km²)")
    ax.set_title("Intertidal area extents — yearly area (km²)", fontsize=11,
                 color="#2F3B3A")
    ax.grid(alpha=.3); ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png), "kind": "graph"}


def _render_lifespan_csv_png(stats_csv, out_png):
    """lifespan_v2 interval_lifespan_stats.csv → time series with CIs."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(stats_csv)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")
    ax.errorbar(df["mid_year"], df["lifespan_yr"],
                yerr=[df["lifespan_yr"] - df["lifespan_boot_lo"],
                      df["lifespan_boot_hi"] - df["lifespan_yr"]],
                fmt="o-", color="#9E5A41", capsize=3,
                label="Lifespan (block-bootstrap 95%)")
    ax.set_yscale("log")
    ax.set_xlabel("Interval mid-year")
    ax.set_ylabel("Lifespan 1/(1-P_annual) [yr]")
    ax.set_title("Intertidal lifespan — year intervals", fontsize=11,
                 color="#2F3B3A")
    ax.grid(alpha=.3, which="both"); ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png), "kind": "graph"}


def _render_migration_csv_png(csv_path, out_png):
    """migration_all_rois.csv → migration-rate time series."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path).sort_values("year")
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")
    if "migration_mean_m_yr" in df:
        ax.plot(df["year"], df["migration_mean_m_yr"], "o-",
                color="#9E5A41", label="Mean")
    if "migration_median_m_yr" in df:
        ax.plot(df["year"], df["migration_median_m_yr"], "s--",
                color="#C2AE92", label="Median")
    if "migration_p95_m_yr" in df:
        ax.plot(df["year"], df["migration_p95_m_yr"], "^:",
                color="#5F7B86", label="p95")
    ax.set_xlabel("Year"); ax.set_ylabel("Migration rate (m/yr)")
    ax.set_title("Channel migration rate", fontsize=11, color="#2F3B3A")
    ax.grid(alpha=.3); ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png), "kind": "graph"}


def _render_bank_overlay_png(tif_a, tif_b, out_png, label, roi=None):
    """Bank movement map: banks of year A (blue) vs year B (red) on a muted
    classified base — via channel.migration.bankline_from_classes."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import rasterio

    api = _api()
    with rasterio.open(tif_a) as src:
        a = src.read(1)
    with rasterio.open(tif_b) as src:
        b = src.read(1)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    bank_a = api.bankline_from_classes(a)
    bank_b = api.bankline_from_classes(b)

    def rgb(hexs):
        return [int(hexs[i:i + 2], 16) / 255 for i in (1, 3, 5)]

    canvas = np.ones((h, w, 3))
    for val, col in ((0, "#efe9dc"), (1, "#d7e2e6"), (2, "#e6eedd"),
                     (255, "#ffffff")):
        canvas[b == val] = rgb(col)
    canvas[bank_a == 1] = rgb("#5F7B86")
    canvas[bank_b == 1] = rgb("#9E5A41")

    bnds = _bounds_wgs84(tif_b)
    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")
    if bnds:
        extent = [bnds[0][1], bnds[1][1], bnds[0][0], bnds[1][0]]
        ax.imshow(canvas, extent=extent, origin="upper")
        if roi:
            w_, s_, e_, n_ = roi["w"], roi["s"], roi["e"], roi["n"]
            if (e_ > bnds[0][1] and w_ < bnds[1][1] and n_ > bnds[0][0]
                    and s_ < bnds[1][0]):
                ax.set_xlim(w_, e_); ax.set_ylim(s_, n_)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    else:
        ax.imshow(canvas, origin="upper")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(handles=[Patch(color="#5F7B86", label=f"Banks {label[0]}"),
                      Patch(color="#9E5A41", label=f"Banks {label[1]}")],
              frameon=False, loc="lower right")
    ax.set_title(f"Bank movement {label[0]} → {label[1]}", fontsize=10,
                 color="#5C6664")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png),
            "bounds": bnds, "kind": "map"}


# --------------------------------------------------------------------------- #
# internal chain: classify (Otsu) → clean channels → intertidal detection
# --------------------------------------------------------------------------- #
def _classify_step(spec, ws, log, api):
    """Download the chosen index GeoTIFFs, clip to the polygon ROI and
    classify them (Otsu or fixed) → binary_{date}.tif in 01_binary."""
    import geodata
    import numpy as np
    import rasterio

    index = spec.get("index", "mndwi")
    files = geodata.fetch(index, log, years=spec.get("years"))
    log(f"classify: {len(files)} '{index}' GeoTIFF(s) from the GitHub database"
        + (f" for years {spec['years'][0]}–{spec['years'][1]}"
           if spec.get("years") else ""))
    roi = spec.get("roi")          # GeoJSON Polygon or None

    bindir = ws / "01_binary"
    rows = []
    for f in files:
        date = _date_of(f.name)
        clipped = _maybe_clip(f, ws / "01_index" / f.name, roi)
        arr, profile = None, None
        with rasterio.open(clipped) as src:
            profile = src.profile.copy()
            a = src.read(1).astype("float32")
            nod = src.nodata
            if nod is not None:
                a[a == nod] = np.nan
        valid = np.isfinite(a)
        fixed = spec.get("threshold", "otsu")
        if str(fixed) not in ("otsu", "auto", "", "None"):
            thr, how = float(fixed), f"fixed {float(fixed):g}"
        else:
            thr, how = float(api.otsu_threshold(a[valid])), "Otsu"
        binary = np.full(a.shape, 255, dtype="uint8")
        binary[valid] = 0
        binary[valid & (a > thr)] = 1
        dst = bindir / f"binary_{date}.tif"
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(dst, "w", **profile) as r:
            r.write(binary, 1)
        nw = int((binary == 1).sum())
        rows.append(dict(date=date, index=index, method=how,
                         threshold=round(thr, 4), n_water=nw,
                         pct_water=round(100 * nw / max(binary.size, 1), 2)))
        log(f"  {date}: {how} thr={thr:.4f}  water={nw:,} px")
    return bindir, rows


def _clean_step(spec, ws, log, api, bindir):
    """RiverFromGeotiff: keep channel-like water, drop ponds/noise."""
    stage = ws / "02_staged"
    pairs = []
    for f in sorted(bindir.glob("binary_*.tif")):
        date = _date_of(f.name)
        dst = stage / f"water_mask_{date}.tif"
        shutil.copy2(f, dst)
        pairs.append((f, dst, date))
    if not pairs:
        raise RuntimeError("no classified masks to clean")
    api.RiverFromGeotiff(
        input_dir=str(stage),
        output_dir=str(ws / "03_river"),
        min_area_px=int(spec.get("minComponentPx", 56)),
        elongation=float(spec.get("elongation", 20.0)),
        skip_existing=False,
    ).run()

    det_dir = ws / "03_river_bin"
    det_dir.mkdir(exist_ok=True)
    for orig, staged, date in pairs:
        river_tif = ws / "03_river" / f"river_{date}.tif"
        if not river_tif.exists():
            continue
        _ensure_georef(river_tif, staged)      # repair CRS if GDAL missing
        shutil.copy2(river_tif, det_dir / f"binary_river_{date}.tif")
    return det_dir


def _ensure_georef(out_tif: Path, ref_tif: Path) -> None:
    import rasterio
    with rasterio.open(out_tif) as src:
        if src.crs is not None:
            return
        arr = src.read(1)
    with rasterio.open(ref_tif) as ref:
        profile = ref.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr.astype("uint8"), 1)


def _intertidal_step(spec, ws, log, api):
    """classify → clean → IntertidalDetector → intertidal_{year}.tif maps."""
    bindir, rows = _classify_step(spec, ws, log, api)
    det_dir = _clean_step(spec, ws, log, api, bindir)
    if not any(det_dir.glob("binary_*.tif")):
        det_dir = bindir          # cleaning kept nothing → raw masks
        log("intertidal: channel cleaning kept nothing — using raw masks")

    years_data = sorted({_year_of(p.name)
                         for p in det_dir.glob("binary_*.tif")})
    y0, y1 = spec.get("years", [None, None])
    target = [y for y in years_data
              if (not y0 or y >= int(y0)) and (not y1 or y <= int(y1))]
    if not target:
        if y0 or y1:
            raise RuntimeError(
                f"no scenes in {y0 or '…'}–{y1 or '…'} — available years: "
                f"{', '.join(map(str, years_data)) or 'none'}. Widen the "
                f"year range in the toolkit panel.")
        log("intertidal: no scenes found — nothing to analyse")
        raise RuntimeError("no scenes available in the database")
    log(f"intertidal: years {target}")

    out_dir = ws / "04_intertidal"
    api.IntertidalDetector(
        river_dir=str(det_dir),
        output_dir=str(out_dir),
        min_scenes=int(spec.get("minScenes", 2)),
        pixel_size_m=float(spec.get("pixelSizeM", 30.0)),
        skip_existing=False,
    ).run(years=target)
    return out_dir, target, det_dir, rows


def _roi_bbox(roi):
    """Leaflet-style bbox dict for the renderers (from GeoJSON Polygon)."""
    if not roi or roi.get("type") != "Polygon":
        return None
    ring = roi["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return {"w": min(lons), "s": min(lats), "e": max(lons), "n": max(lats)}


# --------------------------------------------------------------------------- #
# analysis a) intertidal area extents
# --------------------------------------------------------------------------- #
def run_intertidal(spec, ws, log, mock):
    out = ws / "out" / "intertidal"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        m, g = out / "intertidal_map.png", out / "intertidal_area.png"
        _placeholder_png(m, "Intertidal extent (map)")
        _placeholder_png(g, "Intertidal area (km²) time series")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "intertidal map (mock)", "png": str(m),
                            "kind": "map"},
                           {"name": "area time series (mock)", "png": str(g),
                            "kind": "graph"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    det_dir, target, _, _ = _intertidal_step(spec, ws, log, api)
    maps = sorted(det_dir.glob("intertidal_*.tif"))
    if not maps:
        raise RuntimeError("IntertidalDetector produced no maps — see the "
                           "server log")

    roi = _roi_bbox(spec.get("roi"))
    layers = []
    for t in maps:
        year = _year_of(t.name)
        layers.append(_render_map_png(t, out / f"intertidal_map_{year}.tif.png",
                                      "classified", roi=roi,
                                      title=f"Intertidal classes {year}"))
    summary_csv = det_dir / "intertidal_summary.csv"
    if summary_csv.exists():
        layers.append(_render_area_png(summary_csv, out / "intertidal_area.png"))
    else:
        raise RuntimeError("IntertidalDetector produced no summary CSV")

    cls_csv = out / "classification_summary.csv"
    cls_csv.write_text(
        "date,index,method,threshold,n_water,pct_water\n" +
        "\n".join(f"{r['date']},{r['index']},{r['method']},{r['threshold']},"
                  f"{r['n_water']},{r['pct_water']}" for r in
                  (spec.get("_cls_rows") or [])), encoding="utf-8")

    return {"status": f"ok — intertidal maps + area (km²) series for "
                      f"{len(target)} year(s): "
                      f"{', '.join(str(y) for y in target)}",
            "files": _collect(out) + _collect(det_dir),
            "layers": layers}


# --------------------------------------------------------------------------- #
# analysis b) intertidal lifespan hotspot
# --------------------------------------------------------------------------- #
def run_lifespan(spec, ws, log, mock):
    out = ws / "out" / "lifespan"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        m, g = out / "lifespan_hotspot.png", out / "lifespan_series.png"
        _placeholder_png(m, "Lifespan hotspot (map)")
        _placeholder_png(g, "Lifespan time series")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "hotspot map (mock)", "png": str(m),
                            "kind": "map"},
                           {"name": "lifespan time series (mock)",
                            "png": str(g), "kind": "graph"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    det_dir = ws / "04_intertidal"
    if not any(det_dir.glob("intertidal_*.tif")):
        log("lifespan: no classified maps in this job — running the "
            "intertidal extent first …")
        _intertidal_step(spec, ws, log, api)

    maps = sorted(det_dir.glob("intertidal_*.tif"))
    years = sorted({_year_of(p.name) for p in maps})
    roi = _roi_bbox(spec.get("roi"))
    if len(years) < 2:
        layers = [_render_map_png(t,
                                  out / f"classified_{_year_of(t.name)}.tif.png",
                                  "classified", roi=roi) for t in maps]
        return {"status": f"needs ≥ 2 years of intertidal maps — found "
                          f"{len(years)} ({', '.join(map(str, years)) or 'none'}). "
                          "Add more scenes to the database or widen the year "
                          "range, then run again.",
                "files": _collect(out) + _collect(det_dir),
                "layers": layers}

    ls_dir = ws / "05_lifespan"
    cfg = api.lifespan_Config(glob_pattern="intertidal_{year}.tif")
    api.lifespan_run(det_dir, ls_dir, years=years, cfg=cfg)

    layers = []
    period = ls_dir / "lifespan_win_period_mean.tif"
    if period.exists():
        layers.append(_render_map_png(period, out / "lifespan_hotspot_mean.png",
                                      "continuous", roi=roi,
                                      title="Lifespan hotspot — period mean (yr)"))
    for t in sorted(ls_dir.glob("lifespan_win_*.tif")):
        if t.name == period.name:
            continue
        layers.append(_render_map_png(t, out / f"hotspot_{t.name}.png",
                                      "continuous", roi=roi))
    stats_csv = ls_dir / "interval_lifespan_stats.csv"
    if stats_csv.exists():
        layers.append(_render_lifespan_csv_png(stats_csv,
                                               out / "lifespan_series.png"))
    else:
        raise RuntimeError("lifespan_v2 produced no interval statistics "
                           "(no intertidal pixels?)")

    return {"status": f"ok — lifespan hotspots + interval series over "
                      f"{len(years)} year(s)",
            "files": _collect(out) + _collect(ls_dir) + _collect(det_dir),
            "layers": layers}


# --------------------------------------------------------------------------- #
# analysis c) channel migration rate + bank movement map
# --------------------------------------------------------------------------- #
def run_migration(spec, ws, log, mock):
    out = ws / "out" / "migration"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        m, g = out / "bank_movement.png", out / "migration_series.png"
        _placeholder_png(m, "Bank movement (map)")
        _placeholder_png(g, "Migration rate time series")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "bank movement map (mock)",
                            "png": str(m), "kind": "map"},
                           {"name": "migration time series (mock)",
                            "png": str(g), "kind": "graph"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    det_dir = ws / "04_intertidal"
    if not any(det_dir.glob("intertidal_*.tif")):
        log("migration: no classified maps in this job — running the "
            "intertidal extent first …")
        _intertidal_step(spec, ws, log, api)

    maps = sorted(det_dir.glob("intertidal_*.tif"))
    years = sorted({_year_of(p.name) for p in maps})
    roi = _roi_bbox(spec.get("roi"))
    if len(years) < 2:
        return {"status": f"needs ≥ 2 years to measure migration — found "
                          f"{len(years)} ({', '.join(map(str, years)) or 'none'}). "
                          "Add more scenes to the database or widen the year "
                          "range, then run again.",
                "files": _collect(out) + _collect(det_dir), "layers": []}

    mig_dir = ws / "06_migration"
    cfg = api.MetricsConfig(
        bankline_dir=str(ws / "03_river_bin"),      # none → intertidal fallback
        intertidal_dir=str(det_dir),
        output_dir=str(mig_dir),
        intertidal_glob="intertidal_{year}.tif",
        pixel_size_m=float(spec.get("pixelSizeM", 30.0)),
        use_intertidal_fallback=True,
    )
    df = api.ChannelMigrationAnalyzer(cfg, rois=None).run(years=years)

    layers = []
    tif_a, tif_b = maps[0], maps[-1]
    layers.append(_render_bank_overlay_png(
        tif_a, tif_b, out / f"bank_movement_{years[0]}-{years[-1]}.png",
        (str(years[0]), str(years[-1])), roi=roi))

    mig_csv = mig_dir / "migration_all_rois.csv"
    if mig_csv.exists() and df is not None and not df.empty:
        layers.append(_render_migration_csv_png(mig_csv,
                                                out / "migration_series.png"))
    else:
        raise RuntimeError("ChannelMigrationAnalyzer produced no statistics")

    return {"status": f"ok — migration rates {years[0]}–{years[-1]} "
                      f"+ bank movement map",
            "files": _collect(out) + _collect(mig_dir) + _collect(det_dir),
            "layers": layers}


# --------------------------------------------------------------------------- #
# entry point used by the job runner
# --------------------------------------------------------------------------- #
ANALYSES = {"intertidal": run_intertidal,
            "lifespan": run_lifespan,
            "migration": run_migration}


def run_job(job_id: str, spec: dict, log) -> dict:
    ws = workspace(job_id)
    mock = resolve_mode(log) == "mock"
    results = {}
    for a in spec.get("analyses", ["intertidal"]):
        fn = ANALYSES.get(a)
        if fn is None:
            continue
        try:
            log(f"▶ {a} starting (mock={mock})")
            results[a] = fn(spec, ws, log, mock)
            log(f"✓ {a} done: {results[a]['status']}")
        except Exception as exc:
            log(f"✗ {a} failed: {exc}")
            results[a] = {"status": f"error: {exc}",
                          "trace": traceback.format_exc(), "files": []}
    return results
