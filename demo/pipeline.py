"""
pipeline.py — DeltaSAT analysis orchestrator (no-GEE, GeoTIFF edition).

All imagery is **local-filesystem based**: no Google Earth Engine, no user
uploads. Water-index and land/water GeoTIFFs of the Meghna estuary live in
the GitHub hydro-morphological database and are auto-downloaded + cached
by demo/geodata.py:

  geotiffs/00_MNDWI/                MNDWI index scenes  (mndwi_YYYY-MM-DD.tif)
  geotiffs/01_Fused index/          adapted index (MNDWI + AWEI)/2 scenes
  geotiffs/02_Datewise land water/  ready-made binary land/water masks

Tide-gauge records come from the same repository via demo/hydro.py.

Analyses (spec["analyses"] — each is one button of the toolkit):

  classify    Otsu land/water classification of the chosen index GeoTIFFs
                → gbm_delta.extraction.index_quality.otsu_threshold
  clean       land/water masks → clean river-channel masks (ponds, noise)
                → gbm_delta.extraction.river_mask.RiverFromGeotiff
  intertidal  per-year intertidal extent (area): classified maps +
              intertidal_summary.csv → area time-series figure
                → gbm_delta.extraction.intertidal_detector.IntertidalDetector
  lifespan    transition-matrix Markov lifespan of the intertidal class
                → gbm_delta.morphodynamics.lifespan_legacy.LifespanCalculator

Toolkit resolution (env ``TOOLKIT_DIR`` → sibling ``../gbm-delta-toolkit``
→ sibling ``../gbm-toolkit`` → installed ``gbm_delta``).

Execution modes (env ``MOCK_MODE``):
  "1"          → always MOCK (placeholder PNGs, no heavy deps)
  "0"          → always live (fails hard if the toolkit is not importable)
  unset/"auto" → live when the toolkit imports, MOCK with a log note otherwise
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
        from gbm_delta.morphodynamics.lifespan_legacy import LifespanCalculator

        _API = SimpleNamespace(
            source=f"gbm_delta {getattr(gbm_delta, '__version__', '?')} "
                   f"({', '.join(tried) or 'installed'})",
            otsu_threshold=otsu_threshold,
            RiverFromGeotiff=RiverFromGeotiff,
            IntertidalDetector=IntertidalDetector,
            LifespanCalculator=LifespanCalculator,
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
    """True when the gbm_delta package can be imported (powers /health)."""
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
    for sub in ("01_binary", "02_staged", "03_river", "03_river_bin",
                "04_intertidal", "05_lifespan", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _collect(out_dir: Path, exts=(".png", ".pdf", ".csv", ".tif")) -> list[str]:
    files = []
    for e in exts:
        files += [str(p) for p in out_dir.rglob(f"*{e}")]
    return sorted(files)


def _placeholder_png(path: Path, title: str, subtitle: str = "") -> None:
    """Themed placeholder so the UI has something to render in MOCK mode."""
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
        ax.text(0.5, 0.42, subtitle or "MOCK output — install the gbm_delta "
                "toolkit (and its deps) for real results",
                ha="center", va="center", fontsize=15, color="#5f7b86",
                family="sans-serif")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        path.write_text(f"{title}\n{subtitle}\n")  # last-resort text stub


def _date_of(name: str) -> str:
    """Extract a YYYY-MM-DD (or YYYYMMDD) date from a filename."""
    stem = Path(name).stem
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return stem


def _read_band(path):
    """Single-band GeoTIFF → (float32 array with nodata→NaN, rasterio profile)."""
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr, profile


def _bounds_wgs84(path):
    """GeoTIFF bounds → Leaflet [[s, w], [n, e]] (None when ungeoreferenced)."""
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
# figure rendering (map PNGs + time-series graphs for the results panel)
# --------------------------------------------------------------------------- #
def _render_map_png(tif_path, out_png, kind="classified", roi=None):
    """GeoTIFF → themed PNG figure (+ WGS84 bounds for the Leaflet overlay)."""
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

    if kind == "classified":          # 0 land · 1 river · 2 intertidal · 255 nodata
        disp = np.where(arr == 255, 3, arr)
        cmap = ListedColormap(["#d9c7a3", "#5F7B86", "#7BA05B", "#efe9dc"])
        vmax = 3
    else:                             # binary: 0 land · 1 water · 255 nodata
        disp = np.where(arr == 255, 2, arr)
        cmap = ListedColormap(["#d9c7a3", "#5F7B86", "#efe9dc"])
        vmax = 2

    if b:
        extent = [b[0][1], b[1][1], b[0][0], b[1][0]]      # w, e, s, n
        ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, extent=extent,
                  origin="upper")
        if roi:                       # crop the view to the drawn ROI
            w, s_, e, n = roi["w"], roi["s"], roi["e"], roi["n"]
            if (e > b[0][1] and w < b[1][1] and n > b[0][0] and s_ < b[1][0]):
                ax.set_xlim(w, e)
                ax.set_ylim(s_, n)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    else:
        ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.set_title(Path(tif_path).name, fontsize=10, color="#5C6664")
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
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.set_title("Intertidal extent — yearly area", fontsize=11,
                 color="#2F3B3A")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png), "kind": "graph"}


def _render_lifespan_png(lifespans: dict, out_png):
    """{interval-year: mean intertidal lifespan} → time-series figure."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = sorted(lifespans)
    vals = [float(lifespans[y]) for y in years]
    finite = [v for v in vals if np.isfinite(v)]
    cap = (max(finite) if finite else 1.0) or 1.0

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")
    capped = [min(v, cap * 1.15) if np.isfinite(v) else cap * 1.15 for v in vals]
    ax.bar([str(y) for y in years], capped, color="#9E5A41", width=.55)
    for x, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(x, v + cap * .03, f"{v:.2f}", ha="center", fontsize=8,
                    color="#2F3B3A")
        else:
            ax.text(x, capped[x] + cap * .03, "∞", ha="center", fontsize=11,
                    color="#9E5A41")
    ax.set_ylabel("Mean intertidal lifespan (years)")
    ax.set_xlabel("Start of year interval")
    ax.set_title("Intertidal lifespan (Markov, per interval)", fontsize=11,
                 color="#2F3B3A")
    ax.grid(axis="y", alpha=.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png), "kind": "graph"}


def _render_transition_png(matrix_csv, out_png, label=""):
    """transition_{y1}_{y2}.csv → annotated transition-matrix figure."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.loadtxt(matrix_csv, delimiter=",")
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=110)
    fig.patch.set_facecolor("#f5f0e6")
    im = ax.imshow(m, cmap="Blues")
    classes = ["Land", "Intertidal", "River"]     # legacy LifespanCalculator order
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center",
                    fontsize=8, color="#2F3B3A")
    ax.set_xlabel("→ next year class")
    ax.set_ylabel("current year class")
    ax.set_title(f"Transition matrix {label}", fontsize=10, color="#2F3B3A")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"name": Path(out_png).name, "png": str(out_png), "kind": "graph"}


def _ensure_georef(out_tif: Path, ref_tif: Path) -> None:
    """RiverFromGeotiff may write without CRS (no GDAL) — repair from input."""
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


# --------------------------------------------------------------------------- #
# 1) classify — land/water classification of the chosen index GeoTIFFs
# --------------------------------------------------------------------------- #
def run_classify(spec, ws, log, mock):
    import geodata

    out = ws / "out" / "classify"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        p = out / "classify_map.png"
        _placeholder_png(p, "Land–water classification")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "classification (mock)", "png": str(p),
                            "kind": "map"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    index = spec.get("index", "mndwi")          # 'mndwi' | 'fused'
    files = geodata.fetch(index, log)
    log(f"classify: {len(files)} '{index}' GeoTIFF(s) from the GitHub database")

    import numpy as np
    import rasterio

    bindir = ws / "01_binary"
    rows, layers = [], []
    for f in files:
        arr, profile = _read_band(f)
        valid = np.isfinite(arr)
        fixed = spec.get("threshold", "otsu")
        if str(fixed) not in ("otsu", "auto", "", "None"):
            thr = float(fixed)
            how = f"fixed {thr:g}"
        else:
            thr = float(api.otsu_threshold(arr[valid]))
            how = "Otsu"
        binary = np.full(arr.shape, 255, dtype="uint8")
        binary[valid] = 0
        binary[valid & (arr > thr)] = 1

        date = _date_of(f.name)
        dst = bindir / f"binary_{date}.tif"
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(dst, "w", **profile) as r:
            r.write(binary, 1)

        nw = int((binary == 1).sum())
        nl = int((binary == 0).sum())
        rows.append(dict(date=date, index=index, threshold=round(thr, 4),
                         method=how, n_water=nw, n_land=nl,
                         pct_water=round(100 * nw / max(nw + nl, 1), 2)))
        log(f"  {date}: {how} thr={thr:.4f}  water={nw:,} px "
            f"({rows[-1]['pct_water']}%)")
        layers.append(_render_map_png(dst, out / f"classify_{date}.png",
                                      "binary", roi=spec.get("roi")))

    csv_path = out / "classification_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return {"status": f"ok — {len(files)} scene(s) classified "
                      f"(index={index}, {how} thresholding)",
            "files": _collect(out) + [str(p) for p in bindir.glob("*.tif")],
            "layers": layers}


# --------------------------------------------------------------------------- #
# 2) clean — land/water masks → clean river-channel masks
# --------------------------------------------------------------------------- #
def _stage_for_clean(src_dir, stage: Path):
    """RiverFromGeotiff expects water_mask_*.tif — stage renamed copies."""
    stage.mkdir(parents=True, exist_ok=True)
    pairs = []
    for f in sorted(Path(src_dir).glob("binary_*.tif")):
        date = _date_of(f.name)
        dst = stage / f"water_mask_{date}.tif"
        shutil.copy2(f, dst)
        pairs.append((f, dst, date))
    return pairs


def run_clean(spec, ws, log, mock):
    import geodata

    out = ws / "out" / "clean"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        p = out / "clean_map.png"
        _placeholder_png(p, "Clean river-channel masks")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "cleaned channels (mock)", "png": str(p),
                            "kind": "map"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    src = ws / "01_binary"
    if not any(src.glob("binary_*.tif")):
        log("clean: no classified masks in this job — using the database's "
            "datewise land/water GeoTIFFs")
        src = geodata.fetch("binary", log)[0].parent
    pairs = _stage_for_clean(src, ws / "02_staged")
    if not pairs:
        raise RuntimeError("no binary land/water masks found — run the "
                           "classification (or add scenes to '02_Datewise "
                           "land water') first")

    rx = api.RiverFromGeotiff(
        input_dir=str(ws / "02_staged"),
        output_dir=str(ws / "03_river"),
        min_area_px=int(spec.get("minComponentPx", 56)),
        elongation=float(spec.get("elongation", 20.0)),
        skip_existing=False,
    )
    rx.run()

    riverdir = ws / "03_river"
    det_dir = ws / "03_river_bin"          # detector-compatible copies
    det_dir.mkdir(exist_ok=True)
    layers, rows = [], []
    for orig, staged, date in pairs:
        river_tif = riverdir / f"river_{date}.tif"
        if not river_tif.exists():
            continue
        _ensure_georef(river_tif, staged)  # repair CRS if GDAL was missing
        det = det_dir / f"binary_river_{date}.tif"
        shutil.copy2(river_tif, det)

        import numpy as np
        import rasterio
        with rasterio.open(staged) as r:
            before = int((r.read(1) == 1).sum())
        with rasterio.open(det) as r:
            after = int((r.read(1) == 1).sum())
        rows.append(dict(date=date, water_px_before=before, river_px_after=after,
                         pct_retained=round(100 * after / max(before, 1), 1)))
        log(f"  {date}: water {before:,} px → channel {after:,} px "
            f"({rows[-1]['pct_retained']}%)")
        layers.append(_render_map_png(det, out / f"clean_{date}.png",
                                      "binary", roi=spec.get("roi")))

    csv_path = out / "clean_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return {"status": f"ok — {len(layers)} scene(s) cleaned to channel masks "
                      f"(min {spec.get('minComponentPx', 56)} px · elongation "
                      f"{spec.get('elongation', 20)})",
            "files": _collect(out) + [str(p) for p in det_dir.glob("*.tif")],
            "layers": layers}


# --------------------------------------------------------------------------- #
# 3) intertidal — yearly intertidal extent (area)
# --------------------------------------------------------------------------- #
def _binary_inputs(ws, log) -> Path:
    """Best available binary_*.tif folder for the detector, in chain order."""
    import geodata
    for cand, note in ((ws / "03_river_bin", "cleaned channel masks"),
                       (ws / "01_binary", "classified land/water masks")):
        if any(cand.glob("binary_*.tif")):
            log(f"intertidal: using {note}")
            return cand
    files = geodata.fetch("binary", log)
    log("intertidal: using the database's datewise land/water GeoTIFFs")
    return files[0].parent


def run_intertidal(spec, ws, log, mock):
    out = ws / "out" / "intertidal"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        m, g = out / "intertidal_map.png", out / "intertidal_area.png"
        _placeholder_png(m, "Intertidal extent (map)")
        _placeholder_png(g, "Intertidal area time series")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "intertidal map (mock)", "png": str(m),
                            "kind": "map"},
                           {"name": "area time series (mock)", "png": str(g),
                            "kind": "graph"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    bindir = _binary_inputs(ws, log)
    if not any(Path(bindir).glob("binary_*.tif")):
        raise RuntimeError("no land/water masks available — run the "
                           "classification or cleaning first")

    years_data = sorted({int(_date_of(p.name)[:4])
                         for p in Path(bindir).glob("binary_*.tif")})
    y0, y1 = spec.get("years", [None, None])
    target = [y for y in years_data
              if (not y0 or y >= int(y0)) and (not y1 or y <= int(y1))]
    if not target:
        log(f"intertidal: no scenes in {y0}–{y1}; using all available years")
        target = years_data
    log(f"intertidal: years {target} from {bindir}")

    det_dir = ws / "04_intertidal"
    api.IntertidalDetector(
        river_dir=str(bindir),
        output_dir=str(det_dir),
        min_scenes=int(spec.get("minScenes", 2)),
        pixel_size_m=float(spec.get("pixelSizeM", 30.0)),
        skip_existing=False,
    ).run(years=target)

    layers = []
    for tif in sorted(det_dir.glob("intertidal_*.tif")):
        year = _date_of(tif.name)
        layers.append(_render_map_png(tif, out / f"intertidal_map_{year}.tif.png",
                                      "classified", roi=spec.get("roi")))
    summary_csv = det_dir / "intertidal_summary.csv"
    if summary_csv.exists():
        layers.append(_render_area_png(summary_csv,
                                       out / "intertidal_area.png"))
    else:
        raise RuntimeError("IntertidalDetector produced no summary CSV")

    return {"status": f"ok — intertidal maps + area series for {len(target)} "
                      f"year(s): {', '.join(str(y) for y in target)}",
            "files": _collect(out) + _collect(det_dir),
            "layers": layers}


# --------------------------------------------------------------------------- #
# 4) lifespan — transition matrices + Markov lifespans of the intertidal class
# --------------------------------------------------------------------------- #
def run_lifespan(spec, ws, log, mock):
    import numpy as np
    import rasterio

    out = ws / "out" / "lifespan"
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        m, g = out / "lifespan_map.png", out / "lifespan_series.png"
        _placeholder_png(m, "Intertidal lifespan (map)")
        _placeholder_png(g, "Lifespan time series")
        return {"status": "ok (mock)", "files": _collect(out),
                "layers": [{"name": "classified maps (mock)", "png": str(m),
                            "kind": "map"},
                           {"name": "lifespan time series (mock)", "png": str(g),
                            "kind": "graph"}]}

    api = _api()
    log(f"toolkit: {api.source}")
    det_dir = ws / "04_intertidal"
    if not any(det_dir.glob("intertidal_*.tif")):
        log("lifespan: no classified maps in this job — running the "
            "intertidal extent first …")
        run_intertidal(spec, ws, log, mock)

    maps = sorted(det_dir.glob("intertidal_*.tif"))
    layers = [_render_map_png(t, out / f"lifespan_map_{_date_of(t.name)}.tif.png",
                              "classified", roi=spec.get("roi"))
              for t in maps]
    years = sorted(int(_date_of(p.name)[:4]) for p in maps)
    if len(years) < 2:
        return {"status": f"needs ≥ 2 years of intertidal maps — found "
                          f"{len(years)} ({', '.join(map(str, years)) or 'none'}). "
                          "Add more scenes to the database or widen the year "
                          "range, then run again.",
                "files": _collect(out) + _collect(det_dir),
                "layers": layers}

    # LifespanCalculator uses the legacy class order 0=land, 1=intertidal,
    # 2=river — remap the detector's 0=land, 1=river, 2=intertidal.
    ls_in = ws / "05_lifespan_input"
    ls_in.mkdir(exist_ok=True)
    for p in maps:
        with rasterio.open(p) as src:
            arr = src.read(1)
            profile = src.profile.copy()
        remapped = np.where(arr == 1, 2,
                            np.where(arr == 2, 1, arr)).astype("uint8")
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(ls_in / p.name, "w", **profile) as dst:
            dst.write(remapped, 1)
    log("lifespan: class codes remapped to the legacy convention "
        "(0=land, 1=intertidal, 2=river)")

    lifespans = api.LifespanCalculator(
        classified_dir=str(ls_in),
        output_dir=str(ws / "05_lifespan"),
        pixel_size_m=float(spec.get("pixelSizeM", 30.0)),
    ).run(years=years)

    csv_path = out / "lifespan_summary.csv"
    rows = []
    for i, y in enumerate(years[:-1]):
        v = float(lifespans.get(y, float("nan")))
        rows.append(dict(interval=f"{y}–{years[i + 1]}",
                         lifespan_y=round(v, 3) if np.isfinite(v) else "inf"))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["interval", "lifespan_y"])
        w.writeheader()
        w.writerows(rows)

    layers.append(_render_lifespan_png(lifespans, out / "lifespan_series.png"))
    for t in sorted((ws / "05_lifespan").glob("transition_*.csv")):
        label = t.stem.replace("transition_", "").replace("_", "–")
        layers.append(_render_transition_png(t, out / f"transition_{label}.png",
                                             label))

    return {"status": f"ok — lifespans for {len(rows)} interval(s) over "
                      f"{len(years)} year(s)",
            "files": _collect(out) + _collect(ws / "05_lifespan")
                     + _collect(det_dir),
            "layers": layers}


# --------------------------------------------------------------------------- #
# entry point used by the job runner
# --------------------------------------------------------------------------- #
ANALYSES = {"classify": run_classify,
            "clean": run_clean,
            "intertidal": run_intertidal,
            "lifespan": run_lifespan}


def run_job(job_id: str, spec: dict, log) -> dict:
    ws = workspace(job_id)
    mock = resolve_mode(log) == "mock"
    results = {}
    for a in spec.get("analyses", ["classify"]):
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
