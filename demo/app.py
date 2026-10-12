"""
app.py — DeltaSAT backend API (FastAPI).

Endpoints
---------
GET  /health                      → {ok, mock}
POST /preview      {roi,years,...} → live GEE tile layers for the map
POST /jobs         (multipart)     → start an async analysis job → {job_id}
GET  /jobs/{id}                    → status, log, per-analysis results
GET  /jobs/{id}/file?path=...      → download an output file (png/pdf/csv/tif)

Run:
    pip install -r requirements.txt
    # MOCK (no GEE, test wiring):  default
    uvicorn app:app --reload --port 8000
    # REAL:
    MOCK_MODE=0 MODULES_DIR="/path/to/codes/Modules" EE_PROJECT="your-gcp-project" \
        uvicorn app:app --port 8000
"""
from __future__ import annotations
import os
import json
import uuid
import shutil
import threading
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import pipeline

MOCK = os.getenv("MOCK_MODE", "1") != "0"
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
    return {"ok": True, "mock": MOCK,
            "modules_dir": os.getenv("MODULES_DIR", "") or None}


@app.post("/preview")
def preview(spec: dict):
    """Live map tiles (MNDWI water composite + JRC occurrence)."""
    if MOCK:
        return {"mock": True, "layers": [],
                "note": "MOCK mode: no GEE tiles. Start with MOCK_MODE=0 + EE auth."}
    try:
        import gee
        layers = gee.preview_layers(spec["roi"], spec["years"][0], spec["years"][1],
                                    sensors=spec.get("sensors", ["landsat"]),
                                    max_cloud=spec.get("maxCloud", 20))
        return {"mock": False, "layers": layers}
    except Exception as exc:
        raise HTTPException(500, f"preview failed: {exc}")


def _run_job(job_id, spec, gauge_path):
    with _LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        res = pipeline.run_job(job_id, spec, gauge_path, _log(job_id))
        with _LOCK:
            JOBS[job_id]["results"] = res
            JOBS[job_id]["status"] = "done"
    except Exception as exc:
        with _LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["log"].append(f"FATAL: {exc}")


@app.post("/jobs")
async def create_job(spec: str = Form(...),
                     gauges: UploadFile | None = File(default=None)):
    try:
        spec_obj = json.loads(spec)
    except Exception:
        raise HTTPException(400, "spec must be JSON")
    job_id = uuid.uuid4().hex[:12]

    gauge_path = None
    if gauges is not None:
        ws = pipeline.workspace(job_id)
        gauge_path = str(ws / "08_waterlevel" / gauges.filename)
        with open(gauge_path, "wb") as f:
            shutil.copyfileobj(gauges.file, f)

    with _LOCK:
        JOBS[job_id] = {"status": "queued", "log": [], "results": {}, "spec": spec_obj}
    threading.Thread(target=_run_job, args=(job_id, spec_obj, gauge_path),
                     daemon=True).start()
    return {"job_id": job_id, "mock": MOCK}


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
