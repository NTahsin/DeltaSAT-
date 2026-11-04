# DeltaSAT demo — install (no-GEE edition)

## 1. Layout

```
DeltaSAT-/
├── demo/               ← this folder
│   ├── toolkit.html          main WebGIS app (data region, polygon ROI, 3 analyses)
│   ├── app.py                FastAPI server — serves the web pages
│   ├── run.py                one-command launcher
│   ├── pipeline.py           analysis wiring → gbm_delta toolkit
│   ├── geodata.py            GeoTIFFs + coverage from the GitHub database
│   ├── hydro.py              water levels + stations from the same database
│   └── …                     topic pages, theme.css, requirements.txt
└── gbm-delta-toolkit/  ← the gbm_delta Python package (sibling folder)
```

**Delete the old `demo/gee.py`** — nothing imports it anymore.

## 2. Install

1. Place **gbm-delta-toolkit** next to `demo/` (or `pip install -e` it, or
   set `TOOLKIT_DIR`).
2. `pip install -r demo/requirements.txt`
3. No Earth Engine authentication. Optional: `GITHUB_TOKEN=…`.

## 3. Run

```bash
cd demo
python run.py        # → http://localhost:8000/toolkit.html
```

The map draws the data region automatically; draw a polygon inside it,
pick the years, press a button:

- **a · Intertidal area extents** — per-year maps + area (km²) time series
- **b · Intertidal lifespan hotspot** — windowed P_II hotspot maps + interval time series
- **c · Channel migration rate** — migration-rate time series + bank-movement map
