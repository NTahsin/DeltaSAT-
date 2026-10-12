"""
gbm_delta.extraction.intertidal_base
====================================
Compatibility alias for the intertidal detector base class.

Earlier toolkit releases shipped a minimal reconstructed
``IntertidalDetector`` here (the original ``intertidal_detector.py``
was not available at the time).  The full, real implementation now
lives in ``gbm_delta.extraction.intertidal_detector``; this module
simply re-exports it so existing imports keep working:

    from gbm_delta.extraction.intertidal_base import (
        IntertidalDetector, LAND, RIVER, INTERTIDAL, NODATA)

Note for subclasses (e.g. TideIntertidalDetector): in the real
implementation ``_classify_year()`` returns a 3-tuple
``(result, valid_count, water_count)`` — check overrides against this
signature if you port code written against the old reconstructed base.
"""

from gbm_delta.extraction.intertidal_detector import (  # noqa: F401
    IntertidalDetector,
    LAND, RIVER, INTERTIDAL, NODATA,
)

__all__ = ["IntertidalDetector", "LAND", "RIVER", "INTERTIDAL", "NODATA"]
