"""
gbm_delta.channel — banklines, centerlines, migration, metrics
==============================================================

  banklines.py    Canny banklines from binary land/water GeoTIFFs (+ROI)
  centerlines.py  RivaMap ψ-max composites → NMS → hysteresis centerlines
                  → width-expanded channel platform
  metrics.py      annual migration / intertidal area / elevation / DTMC
                  lifespan per ROI
  migration.py    lighter migration-only analyser using bankline +
                  intertidal rasters
"""

__all__ = ["banklines", "centerlines", "metrics", "migration"]
