# DeltaSAT demo — update instructions (gbm-toolkit v0.3.0)

1. Replace `demo/pipeline.py`, `demo/README.md`, `demo/requirements.txt`
   with the versions in this folder (app.py, gee.py and the .html files are
   unchanged — no edits needed).
2. Place the **gbm-toolkit** folder next to `demo/`
   (repo root: `DeltaSAT-/{demo, gbm-toolkit}`), or
   `pip install -e /path/to/gbm-toolkit`, or set `TOOLKIT_DIR`.
3. `pip install -r requirements.txt` (adds opencv-python-headless for the
   bankline/migration analysis).
4. MOCK wiring test: `uvicorn app:app --reload --port 8000`
   Real mode: `MOCK_MODE=0 EE_PROJECT="your-gcp-project" uvicorn app:app`
