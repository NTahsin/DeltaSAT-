"""
gbm_delta — GBM delta intertidal morphodynamics toolkit
=======================================================

Organised, cloud-compatible modules for:

  * tide-gauge data analysis        -> gbm_delta.water_levels
  * Landsat water indices (GEE)     -> gbm_delta.gee          (previous release)
  * water / river / intertidal
    extraction                      -> gbm_delta.extraction
  * banklines, centerlines,
    migration & metrics             -> gbm_delta.channel       (previous release)
  * intertidal elevation, accommodation,
    sediment mass, tidal prism,
    lifespan (DTMC)                 -> gbm_delta.morphodynamics
  * publication figures             -> gbm_delta.viz

See README.md for the module index and WORKFLOWS.md for the numbered
end-to-end workflows (cloud vs. local).
"""

__version__ = "0.2.1"

from . import config, utils  # noqa: F401

__all__ = ["config", "utils", "water_levels", "extraction", "morphodynamics",
           "viz", "gee", "channel", "__version__"]
