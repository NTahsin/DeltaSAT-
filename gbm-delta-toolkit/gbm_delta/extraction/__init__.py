"""
gbm_delta.extraction — water / river / intertidal extraction (local rasters)
============================================================================

  river_mask.py         binary water masks -> river channels (size/elongation)
  intertidal_detector.py  count + per-image Otsu SI-validated intertidal
                        classification per year (the real implementation)
  intertidal_base.py   compat alias re-exporting IntertidalDetector
  intertidal_filter.py drop intertidal pixels not hydrologically connected
                        to permanent water
  validation.py        Otsu water delineation vs. Planet NDWI (accuracy,
                        confusion matrix, block-bootstrap CIs)
  planet_anyindex.py   Planet reference from *any* index direction (auto
                        midpoint/Otsu threshold guidance)
  planet_mosaic.py     overlap-safe Planet SR tile mosaic -> water-fraction
                        reference layer on the classification grid

  (refine_pipeline.py, interannual.py, tidal_intertidal.py, index_quality.py
  are part of the previous toolkit release and remain unchanged.)
"""

__all__ = ["river_mask", "intertidal_detector", "intertidal_base",
           "intertidal_filter", "validation", "planet_anyindex",
           "planet_mosaic"]
