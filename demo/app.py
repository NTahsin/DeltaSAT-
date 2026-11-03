"""
app.py — DeltaSAT backend API (FastAPI, no-GEE edition).

Endpoints
---------
GET  /health                      → {ok, mock, mode}
GET  /stations                    → Meghna-estuary gauge stations from the
                                     GitHub hydro database (auto-downloaded)
POST /jobs         (JSON spec)    → start an async analysis job → {job_id}
GET  /jobs/{id}                    → status, log, per-analysis results
GET  /jobs/{id}/file?path=...      → download an output file (png/pdf/csv/tif)

No Google Earth Engine and no user uploads: water-index GeoTIFFs come from
NTahsin/Ganges-Brahmaputra-Meghna-Hydro-morpho ("geotiffs") via demo/geodata.py,
water levels from the same repository ("meghna estuary") via demo/hydro.py.

Run:
    pip install -r requirements.txt
    uvicorn app:app --port 8000
"""
from __future__ import annotations
import os
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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


def _log(job_id):
    def add(msg):
        with _LOCK:
            JOBS[job_id]["log"].append(str(msg))
    return add


@app.get("/health")
def health():
    mock = True if MODE == "1" else (False if MODE == "0"
                                     else not pipeline.toolkit_available())
    return {"ok": True, "mode": MODE, "mock": mock,
            "toolkit_dir": os.getenv("TOOLKIT_DIR", "") or None}


@app.get("/stations")
def stations():
    """Gauge stations of the Meghna-estuary hydro database (for the map)."""
    try:
        return {"ok": True, "stations": hydro.station_table()}
    except Exception as exc:
        raise HTTPException(503, f"hydro database unreachable: {exc}")


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
    file uploads; GeoTIFFs and water levels are fetched by the backend."""
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
        # expose files & layer PNGs as API-relative paths the front-end can fetch
        results = {a: _rewrite(job_id, r) for a, r in j["results"].items()}
        return {"job_id": job_id, "status": j["status"],
                "log": j["log"], "results": results}


def _rewrite(job_id, r: dict) -> dict:
    root = (pipeline.WORK_ROOT / job_id).resolve()

    def rel(p):
        try:
            return (f"/jobs/{job_id}/file?path="
                    f"{Path(p).resolve().relative_to(root).as_posix()}")
        except Exception:
            return None

    out = {**r}
    if "files" in r:
        out["files"] = [rel(p) for p in r.get("files", [])]
    if "layers" in r:
        out["layers"] = [{**ly, "png": rel(ly.get("png"))}
                         for ly in r.get("layers", []) if ly.get("png")]
    return out


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str, path: str):
    root = (pipeline.WORK_ROOT / job_id).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)) or not target.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(str(target))


# --------------------------------------------------------------------------- #
# the front-end IS served by this backend — one server, same origin,
# zero configuration: open http://localhost:8000 after starting it
# --------------------------------------------------------------------------- #
app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent),
                           html=True), name="site")
