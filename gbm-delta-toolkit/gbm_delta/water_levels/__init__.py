"""
gbm_delta.water_levels — tide-gauge data analysis
=================================================

  stations.py      GBM gauge coordinates + along-coast distance helpers
  tidal_range.py    load_water_levels() CSV loader (shared)
  processor.py     full pipeline: QC -> smart gap-fill -> LSHA harmonic
                   reconstruction (UTide-equivalent, pure NumPy/SciPy) ->
                   validation -> 5-min export; ingests multi-sheet Excel
                   and per-station CSV folders (load_excel / load_csv /
                   load_csv_folder), incl. the NTahsin/Ganges-Brahmaputra-
                   Meghna-Hydro-morpho database format

  (extremes.py, datums.py, surface_profiler.py are unchanged from the
  first release.)
"""

__all__ = ["stations", "tidal_range"]
