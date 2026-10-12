"""
gbm_delta.gee — Google Earth Engine (cloud) modules
===================================================

Requires: earthengine-api (+ geemap for local GeoTIFF export).
These modules import `ee` at module level; the rest of the toolkit works
without GEE installed.

  landsat.py        Landsat 5/8 collection builder, water indices,
                    Otsu thresholding, dry-season composites, exporters
  adaptive_index.py Cloud-masked NDWI/MNDWI/AWEI + fused water index
  rivermap.py       GEE-side river delineation (MMSI + adaptive threshold)
  core.py           ee initialisation + Drive/local export helpers

Run `from gbm_delta.gee.core import initialize_ee; initialize_ee()` once
per session (Colab, Jupyter, or local with a GEE account).
"""

from .core import initialize_ee, export_to_drive  # noqa: F401

__all__ = ["core", "landsat", "adaptive_index", "rivermap",
           "initialize_ee", "export_to_drive"]
