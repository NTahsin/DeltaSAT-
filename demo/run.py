#!/usr/bin/env python3
"""
run.py — one-command DeltaSAT launcher.

    python run.py    # → http://localhost:8000/toolkit.html

Options via environment variables (all optional):
    PORT=8000            server port
    MOCK_MODE=1|0|auto   1 = placeholder figures · 0 = force live ·
                         auto (default) = live when the gbm_delta toolkit
                         imports, MOCK with a log note otherwise
    GITHUB_TOKEN=...     optional — raises the GitHub API rate limit for the
                         GeoTIFF / gauge downloads
"""
from __future__ import annotations

import os
import threading
import time
import webbrowser

import uvicorn

PORT = int(os.getenv("PORT", "8000"))
OPEN_AFTER = float(os.getenv("OPEN_AFTER", "1.5"))


def _open_browser() -> None:
    time.sleep(OPEN_AFTER)
    try:
        webbrowser.open(f"http://localhost:{PORT}/toolkit.html")
    except Exception:
        pass  # headless — the URL is printed below anyway


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    print("\n  DeltaSAT backend starting …")
    print(f"  ➜ Toolkit:   http://localhost:{PORT}/toolkit.html")
    print(f"  ➜ Home page: http://localhost:{PORT}/")
    print("  ➜ Data: GeoTIFFs + water levels auto-fetched from the GitHub "
          "database (no Google Earth Engine)")
    print("  ➜ Stop with Ctrl+C\n")
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)
