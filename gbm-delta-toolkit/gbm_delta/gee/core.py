"""
gbm_delta.gee.core
==================
Cloud (Google Earth Engine) initialisation and export helpers.

Use this at the top of any GEE session (Colab, Jupyter, local):

    from gbm_delta.gee.core import initialize_ee
    initialize_ee()            # or initialize_ee(project="my-gee-project")

Nothing in this module runs at import time, so local-only users can import
the rest of the toolkit without an Earth Engine account.
"""

from __future__ import annotations

from typing import Optional


def initialize_ee(project: Optional[str] = None) -> None:
    """
    Initialise the Earth Engine Python API.

    Tries a silent `ee.Initialize()` (uses stored credentials); on failure,
    runs the interactive `ee.Authenticate()` flow first (browser / device
    code), then initialises again. An optional GEE *project* can be supplied
    (required for newer GEE accounts).
    """
    import ee

    kwargs = {"project": project} if project else {}
    try:
        ee.Initialize(**kwargs)
    except Exception:
        ee.Authenticate()
        ee.Initialize(**kwargs)


def export_to_drive(image, description: str, folder: str = "gbm_delta",
                    region=None, scale: float = 30, max_pixels: float = 1e13):
    """Start a Drive export task for an ee.Image and return the task."""
    import ee  # noqa: F401  (image is already an ee.Image)

    task_kwargs = dict(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        scale=scale,
        maxPixels=max_pixels,
    )
    if region is not None:
        task_kwargs["region"] = region

    import ee.batch  # type: ignore[attr-defined]
    task = ee.batch.Export.image.toDrive(**task_kwargs)
    task.start()
    print(f"Export task started: {description}")
    return task


def export_to_local(image, filename: str, region, scale: float = 30,
                    crs: str = "EPSG:32646") -> str:
    """
    Download an ee.Image directly to a local GeoTIFF via geemap
    (no Google Drive round-trip). Returns the filename.
    """
    try:
        import geemap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "geemap is required for local export:  pip install geemap"
        ) from exc

    geemap.ee_export_image(image, filename=filename, scale=scale,
                           region=region, crs=crs, file_per_band=False)
    print(f"Saved: {filename}")
    return filename
