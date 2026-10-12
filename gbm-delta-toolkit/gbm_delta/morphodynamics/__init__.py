"""
gbm_delta.morphodynamics — intertidal morphodynamics analysis chain
====================================================================

Elevation & extent
  elevation_v2.py       per-year occurrence + elevation from WSE stacks
                        (Bishop-Taylor primary, waterline median supplementary)
  elevation_v3.py       study-grid version: river masks + compact domain,
                        windowed reads, same two elevation methods
  reconciler.py         frequency-of-inundation + JRC + datum reconciliation
                        -> intertidal_YYYY.tif / intertidal_h_YYYY.tif
  intertidal_filter.py lives in gbm_delta.extraction (post-filtering)

Volumes & budgets
  accommodation.py      accommodation depth a = max(MHSW - z, 0), fill ratio
  accommodation_roi.py  per-ROI version
  tidal_prism.py        intertidal tidal prism (numerically = V_acc,
                        hydrodynamic interpretation) + Sen's slope trend
  sediment_mass.py      year-on-year sediment mass from dz stacks

Statistics
  stats.py              per-year class-area table from intertidal maps
  stats_roi.py          per-ROI class-area statistics

Lifespan (residence time)
  lifespan_markov.py     pooled DTMC + pixel-wise P_II / lifespan maps + figures
  lifespan_v2.py         interval-based annualised persistence with
                        Wilson + block-bootstrap CIs and trend tests
  lifespan_roi.py        per-ROI version of the interval analysis
  lifespan_local.py      annual transition matrices + annual lifespan maps
  lifespan_legacy.py     oldest simple Markov calculator (class encoding:
                        1 = intertidal!) — kept for reference
  stationarity.py        Anderson–Goodman time-homogeneity test (full delta,
                        per ROI, per channel group)

Comparisons
  river_comparison.py    multi-river comparison: clip rasters per channel,
                        per-river DTMC lifespan, time series + figures
"""

__all__ = ["elevation_v2", "elevation_v3", "reconciler",
           "accommodation", "accommodation_roi", "tidal_prism",
           "sediment_mass", "stats", "stats_roi",
           "lifespan_markov", "lifespan_v2", "lifespan_roi",
           "lifespan_local", "lifespan_legacy", "stationarity",
           "river_comparison"]
