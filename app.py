"""
app.py — DeltaSAT backend API (FastAPI).

Endpoints
---------
GET  /health                      → {ok, mock, mode, ee}
GET  /stations                    → Meghna-estuary gauge stations from the
                                     GitHub hydro database (auto-downloaded)
POST /preview      {roi,years,...} → live GEE tile layers for the map
POST /jobs         (JSON spec)    → start an async analysis job → {job_id}
GET  /jobs/{id}                    → status, log, per-analysis results
GET  /jobs/{id}/file?path=...      → download an output file (png/pdf/csv/tif)

No user uploads: Landsat imagery comes from the Earth Engine Python API
(imported automatically when a job starts — see pipeline.resolve_mode),
water levels from NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho
("meghna estuary") via demo/hydro.py.

Run:
    pip install -r requirements.txt
    # default: auto — live GEE when credentials exist, MOCK otherwise
    uvicorn app:app --reload --port 8000
    # force:
    MOCK_MODE=0 EE_PROJECT="your-gcp-project" uvicorn app:app --port 8000
    MOCK_MODE=1 uvicorn app:app --port 8000
"""
from __future__ import annotations
import os
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import pipeline
import hydro

MODE = os.getenv("MOCK_MODE", "auto").strip().lower()   # auto | 0 | 1
app = FastAPI(title="DeltaSAT backend")

# Allow the static toolkit page (any origin during dev) to call the API.
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# In-memory job store. Swap for Redis/DB in production.
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

# Earth Engine availability probe (background so startup is instant).
_EE_OK: bool | None = None


def _probe_ee():
    global _EE_OK
    _EE_OK = pipeline._ee_ready(lambda msg: None)


threading.Thread(target=_probe_ee, daemon=True).start()


def _log(job_id):
    def add(msg):
        with _LOCK:
            JOBS[job_id]["log"].append(str(msg))
    return add


@app.get("/health")
def health():
    mock = True if MODE == "1" else False if MODE == "0" else (_EE_OK is not True)
    return {"ok": True, "mode": MODE, "mock": mock, "ee": _EE_OK,
            "toolkit_dir": os.getenv("TOOLKIT_DIR", "") or None}


@app.get("/stations")
def stations():
    """Gauge stations of the Meghna-estuary hydro database (for the map)."""
    try:
        return {"ok": True, "stations": hydro.station_table()}
    except Exception as exc:
        raise HTTPException(503, f"hydro database unreachable: {exc}")


@app.post("/preview")
def preview(spec: dict):
    """Live map tiles (MNDWI water composite + JRC occurrence)."""
    if _EE_OK is not True:
        return {"mock": True, "layers": [],
                "note": "Earth Engine not initialised (auto mode). "
                        "Authenticate (earthengine authenticate) and/or set "
                        "MOCK_MODE=0 + EE_PROJECT."}
    try:
        import gee
        layers = gee.preview_layers(spec["roi"], spec["years"][0], spec["years"][1],
                                    sensors=spec.get("sensors", ["landsat"]),
                                    max_cloud=spec.get("maxCloud", 20))
        return {"mock": False, "layers": layers}
    except Exception as exc:
        raise HTTPException(500, f"preview failed: {exc}")


def _run_job(job_id, spec):
    with _LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        res = pipeline.run_job(job_id, spec, _log(job_id))
        with _LOCK:
            JOBS[job_id]["results"] = res
            JOBS[job_id]["status"] = "done"
    except Exception as exc:
        with _LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["log"].append(f"FATAL: {exc}")


@app.post("/jobs")
async def create_job(spec: dict):
    """Start an analysis job. The spec is a plain JSON body — there are no
    file uploads; water levels and imagery are fetched by the backend."""
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        JOBS[job_id] = {"status": "queued", "log": [], "results": {}, "spec": spec}
    threading.Thread(target=_run_job, args=(job_id, spec),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _LOCK:
        j = JOBS.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        # expose files as API-relative paths the front-end can fetch
        results = {}
        for a, r in j["results"].items():
            results[a] = {**r, "files": [_rel(job_id, p) for p in r.get("files", [])]}
        return {"job_id": job_id, "status": j["status"],
                "log": j["log"], "results": results}


def _rel(job_id, abspath):
    root = (pipeline.WORK_ROOT / job_id).resolve()
    try:
        rel = Path(abspath).resolve().relative_to(root)
        return f"/jobs/{job_id}/file?path={rel.as_posix()}"
    except Exception:
        return None


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str, path: str):
    root = (pipeline.WORK_ROOT / job_id).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)) or not target.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(str(target))
