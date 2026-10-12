"""
intertidal_detector.py
======================
Detect intertidal zones per year from a stack of binary river/land GeoTIFFs.

For each year, loads all binary river/land scenes and classifies every pixel:

    0 = Permanent Land    — never water (or fails the SI rules)
    1 = Permanent River   — always water AND consistently high SI
    2 = Intertidal Zone   — switches AND consistently high SI when water
  255 = NoData            — insufficient valid observations

Classification rules (with singularity index, SI, validation ON)
----------------------------------------------------------------
'High' SI values are determined PER IMAGE via OTSU thresholding of each
singularity index GeoTIFF (no fixed threshold).

    permanent river : always water in the binaries AND its SI is in the
                      HIGH class (>= that image's Otsu threshold) in EVERY
                      image in which the pixel is water
    intertidal      : switches between water and land AND its SI is in
                      the HIGH class (>= that image's Otsu threshold) in
                      EVERY image in which the pixel is water
    everything else : land (including always-water pixels and switching
                      pixels whose SI is not consistently in the high
                      class when they are water)

    - If SI evidence is missing for a pixel (no valid SI values),
      the count-based classification is kept as a fallback.
    - Without si_dir, classification is purely count-based:
      river = always water, intertidal = switches, land = never water.
    - si_threshold is only a FALLBACK for images where Otsu cannot be
      computed (constant or empty images).

Outputs per year
----------------
  intertidal_YYYY.tif        — classified GeoTIFF (uint8, values 0/1/2/255)
  intertidal_summary.csv     — pixel counts + areas for each year and class

Requirements
------------
    pip install numpy tifffile
    (Optional but recommended: pip install gdal   — for correct CRS/georef)

Usage
-----
    from gbm_delta.extraction.intertidal_detector import IntertidalDetector

    det = IntertidalDetector(
        river_dir        = r'W:\\...\\02_River Extracted',
        output_dir       = r'W:\\...\\04_Intertidal',
        si_dir           = r'W:\\...\\03_Singularity Index',
    )

    det.summary()               # preview available files per year
    det.run()                   # all years
    det.run(years=[2005, 2010]) # specific years only
    det.run_year(2005)          # single year

Notes
-----
- Input files must start with 'binary_' and contain the acquisition date
  as YYYY-MM-DD (or YYYYMMDD), e.g.
      binary_psi_2005-01-15.tif
      binary_fused_index_1996-02-09_LT05_137044_19960209.tif
- Singularity index files are matched BY DATE: any .tif in si_dir whose
  filename contains the same date (dashed or compact) as a binary mask
  is used. The si_prefix is only a preference when several files match.
- Pixel value convention matches river_from_geotiff.py:
    1 = river/water,  0 = land,  255 = nodata
- With SI validation ON, 'high' is per-image OTSU thresholding:
      river      : always water AND SI >= image Otsu threshold in EVERY
                   scene where the pixel is water
      intertidal : switches AND SI >= image Otsu threshold in EVERY
                   scene where the pixel is water
      land       : everything else that has enough valid observations
- Set min_scenes to require a minimum number of valid observations
  before classifying a pixel (default 2).
- The output CSV reports pixel counts and km² areas (assuming 30m
  pixels unless pixel_size_m is changed).
"""

import os
import re
import glob
import warnings
from collections import defaultdict

import numpy as np


# ── Class labels ─────────────────────────────────────────────────────────────
LAND       = 0
RIVER      = 1
INTERTIDAL = 2
NODATA     = 255


class IntertidalDetector:
    """
    Detect permanent land, permanent river, and intertidal zones per year
    from stacks of binary river GeoTIFFs.

    Parameters
    ----------
    river_dir      : str    folder containing binary river mask files, e.g.
                            binary_fused_index_1996-02-09_LT05_137044_19960209.tif
                            (flat or with year subfolders)
    output_dir     : str    where to save intertidal_YYYY.tif + CSV
    si_dir         : str    optional folder containing pre-computed
                            singularity index GeoTIFFs (one per date).
                            Files are matched BY DATE: any .tif whose
                            filename contains the same date (YYYY-MM-DD or
                            YYYYMMDD) as a binary mask is used. When given,
                            the SI rules apply, with 'high' SI decided per
                            image by OTSU thresholding:
                              river      : always water AND SI >= image
                                           Otsu threshold in EVERY wet scene
                              intertidal : switches AND SI >= image Otsu
                                           threshold in EVERY wet scene
                              land       : everything else
                            Pixels whose SI evidence is missing keep the
                            count-based class.
    si_prefix      : str    preferred filename prefix of the singularity
                            index files when several files match a date.
                            Default 'singularity_index_'.
    si_threshold   : float  FALLBACK 'high' SI value, used ONLY for images
                            where the Otsu threshold cannot be computed
                            (constant or empty images). Default 0.5.
    min_scenes     : int    minimum number of valid (non-nodata) observations
                            per pixel per year to make any classification.
                            Pixels below this threshold → NoData in output.
                            Default 2 (need at least 2 clear scenes/year).

                            Classification uses absolute counts — NOT fractions
                            — because only 2–5 cloud-free Landsat scenes/year
                            are typically available over the GBM delta. A pixel
                            wet in 1 of 2 scenes = 50% fraction but could just
                            be a single flood event, making fraction thresholds
                            unreliable at this sample size. Rules:

                              permanent land  : N_water == 0
                              permanent river : N_water == N_valid
                              intertidal      : 0 < N_water < N_valid
    pixel_size_m   : float  pixel size in metres for area calculations.
                            Default 30 (Landsat).
    skip_existing  : bool   skip years whose output tif already exists.
    nodata_val     : int    nodata value in input files. Default 255.
    """

    def __init__(
        self,
        river_dir:      str,
        output_dir:     str,
        si_dir:         str   = None,
        si_prefix:      str   = 'singularity_index_',
        si_threshold:   float = 0.5,
        min_scenes:     int   = 2,
        pixel_size_m:   float = 30.0,
        skip_existing:  bool  = True,
        nodata_val:     int   = 255,
    ):
        self.river_dir      = river_dir
        self.output_dir     = output_dir
        self.si_dir         = si_dir
        self.si_prefix      = si_prefix
        self.si_threshold   = si_threshold
        self.min_scenes     = min_scenes
        self.pixel_size_m   = pixel_size_m
        self.skip_existing  = skip_existing
        self.nodata_val     = nodata_val

        os.makedirs(output_dir, exist_ok=True)

        print(f"📁 River dir  : {os.path.abspath(river_dir)}")
        print(f"📁 Output dir : {os.path.abspath(output_dir)}")
        if si_dir:
            print(f"📁 SI dir     : {os.path.abspath(si_dir)}")
            print(f"   SI rules        : per-image OTSU thresholding; "
                  f"river = always water & high SI in every wet scene; "
                  f"intertidal = switches & high SI (>= image Otsu) in "
                  f"every wet scene; else land")
            print(f"   SI fallback thr : {si_threshold} (only when Otsu undefined)")
        else:
            print(f"   SI validation  : off (no si_dir given)")
        print(f"   Min scenes      : {min_scenes}")
        print(f"   Classification  : absolute count (not fraction-based)")
        print(f"   Pixel size      : {pixel_size_m} m")

    # ─────────────────────────────────────────────────────────────────────────
    # I/O helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_tif(filepath: str):
        """
        Read a GeoTIFF. Returns (array uint8, reference_ds_or_None).
        Tries GDAL first (preserves CRS), falls back to tifffile.
        """
        try:
            from osgeo import gdal
            ds   = gdal.Open(filepath)
            arr  = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
            return arr, ds          # keep ds open for georef copying
        except (ImportError, AttributeError):
            pass

        try:
            import tifffile
            arr = tifffile.imread(filepath)
            if arr.ndim == 3:
                arr = arr[0]
            return arr.astype(np.uint8), None
        except ImportError:
            raise ImportError("Install tifffile:  pip install tifffile")

    @staticmethod
    def _read_tif_float(filepath: str):
        """
        Read a GeoTIFF as float32 (for singularity index rasters).
        Returns (array float32, reference_ds_or_None) or (None, None)
        if the file cannot be read.
        """
        try:
            from osgeo import gdal
            ds   = gdal.Open(filepath)
            arr  = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            return arr, ds
        except (ImportError, AttributeError):
            pass

        try:
            import tifffile
            arr = tifffile.imread(filepath)
            if arr.ndim == 3:
                arr = arr[0]
            return arr.astype(np.float32), None
        except Exception:
            return None, None

    @staticmethod
    def _write_tif(filepath: str, data: np.ndarray, reference_path: str):
        """
        Write uint8 array as GeoTIFF, copying CRS + geotransform from
        reference_path. Uses GDAL if available, falls back to tifffile.
        """
        try:
            from osgeo import gdal, gdalconst
            ref_ds = gdal.Open(reference_path)
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                filepath,
                ref_ds.RasterXSize,
                ref_ds.RasterYSize,
                1,
                gdalconst.GDT_Byte,
                options=['COMPRESS=LZW']
            )
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            band = out_ds.GetRasterBand(1)
            band.WriteArray(data.astype(np.uint8))
            band.SetNoDataValue(NODATA)
            out_ds.FlushCache()
            out_ds = ref_ds = None
            return

        except (ImportError, AttributeError):
            pass

        # Fallback: tifffile (no CRS, still viewable + correct pixels)
        try:
            import tifffile
            tifffile.imwrite(
                filepath,
                data.astype(np.uint8),
                photometric='minisblack'
            )
        except ImportError:
            raise ImportError("Install tifffile:  pip install tifffile")

    # ─────────────────────────────────────────────────────────────────────────
    # Otsu thresholding
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _otsu_threshold(values: np.ndarray, fallback: float) -> float:
        """
        Compute the OTSU threshold of the finite values in `values`.

        Uses a 256-bin histogram and maximises the between-class variance
        (classic Otsu, 1979). Returns `fallback` when the threshold is
        undefined (no finite values, or all values identical).
        """
        v = values[np.isfinite(values)]
        if v.size == 0:
            return float(fallback)

        vmin, vmax = float(v.min()), float(v.max())
        if vmax <= vmin:
            return float(fallback)

        hist, edges = np.histogram(v, bins=256, range=(vmin, vmax))
        hist    = hist.astype(np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0

        w0  = np.cumsum(hist)                       # class-0 weight per bin
        w1  = w0[-1] - w0                           # class-1 weight per bin
        csum = np.cumsum(hist * centers)

        valid = (w0 > 0) & (w1 > 0)
        if not valid.any():
            return float(fallback)

        total = csum[-1]
        mu0 = csum / np.maximum(w0, 1e-12)
        mu1 = (total - csum) / np.maximum(w1, 1e-12)

        var_between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, 0.0)
        if var_between.max() <= 0:
            return float(fallback)

        return float(centers[int(np.argmax(var_between))])

    # ─────────────────────────────────────────────────────────────────────────
    # File discovery
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_by_year(self) -> dict:
        """
        Scan river_dir for binary river mask files, e.g.
            binary_psi_2005-01-15.tif
            binary_fused_index_1996-02-09_LT05_137044_19960209.tif
        Any file starting with 'binary_' and containing a YYYY-MM-DD (or
        YYYYMMDD) date is accepted. Returns {year_int: [sorted list of paths]}.
        """
        patterns = [
            os.path.join(self.river_dir, '**', 'binary_*.tif'),
            os.path.join(self.river_dir, 'binary_*.tif'),
        ]
        all_files = []
        for p in patterns:
            all_files.extend(glob.glob(p, recursive=True))
        all_files = sorted(set(all_files))

        by_year = defaultdict(list)

        for f in all_files:
            date_str = self._date_from_filename(os.path.basename(f))
            if date_str:
                year = int(date_str[:4])
                by_year[year].append(f)

        # Sort each year's list chronologically
        for yr in by_year:
            by_year[yr] = sorted(by_year[yr])

        return dict(sorted(by_year.items()))

    @staticmethod
    def _date_from_filename(basename: str) -> str:
        """
        Extract the acquisition date from a filename.
        Accepts dashed (YYYY-MM-DD) or compact (YYYYMMDD) dates.
        Returns 'YYYY-MM-DD' or '' if no date found.
        """
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', basename)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        m = re.search(r'(\d{4})(\d{2})(\d{2})', basename)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        return ''

    def _si_path_for_date(self, date_str: str) -> str:
        """
        Return the singularity index path for a YYYY-MM-DD date, or '' if
        none exists. Any .tif in si_dir (or its subfolders) whose filename
        contains the date — dashed (YYYY-MM-DD) or compact (YYYYMMDD) —
        is accepted, e.g.:
            singularity_index_1996-02-09.tif
            fused_index_1996-02-09_LT05_137044_19960209.tif
        """
        if not self.si_dir:
            return ''

        compact = date_str.replace('-', '')

        candidates = glob.glob(
            os.path.join(self.si_dir, '**', '*.tif'), recursive=True
        )
        candidates += glob.glob(os.path.join(self.si_dir, '*.tif'))

        # Prefer files matching the configured prefix first, then any match
        prefix_matches = [
            c for c in candidates
            if os.path.basename(c).lower().startswith(self.si_prefix.lower())
            and (date_str in os.path.basename(c) or compact in os.path.basename(c))
        ]
        if prefix_matches:
            return sorted(prefix_matches)[0]

        any_matches = [
            c for c in candidates
            if date_str in os.path.basename(c) or compact in os.path.basename(c)
        ]
        return sorted(any_matches)[0] if any_matches else ''

    def _collect_si_by_year(self) -> dict:
        """
        Scan si_dir for singularity index GeoTIFFs (any .tif containing a
        date in its filename). Returns {year_int: [sorted list of paths]}.
        """
        if not self.si_dir:
            return {}

        patterns = [
            os.path.join(self.si_dir, '**', '*.tif'),
            os.path.join(self.si_dir, '*.tif'),
        ]
        all_files = []
        for p in patterns:
            all_files.extend(glob.glob(p, recursive=True))
        all_files = sorted(set(all_files))

        by_year = defaultdict(list)

        for f in all_files:
            date_str = self._date_from_filename(os.path.basename(f))
            if date_str:
                year = int(date_str[:4])
                by_year[year].append(f)

        for yr in by_year:
            by_year[yr] = sorted(by_year[yr])

        return dict(sorted(by_year.items()))

    # ─────────────────────────────────────────────────────────────────────────
    # Core classification
    # ─────────────────────────────────────────────────────────────────────────

    def _si_validation_masks(self, files: list, shape: tuple,
                             water_count: np.ndarray,
                             valid_count: np.ndarray):
        """
        Singularity index (SI) validation of river and intertidal candidates,
        with 'high' SI values decided PER IMAGE by OTSU thresholding.

        Rules (with SI evidence available):

            river      : pixel always water AND its SI is >= the Otsu
                         threshold of that image in EVERY image where the
                         pixel is water (consistently in the high class)
            intertidal : pixel switches AND the SAME high-SI rule — SI >=
                         the image's Otsu threshold in EVERY image where
                         the pixel is water
            land       : everything else

        SI values that are NaN or non-finite are excluded. A pixel is only
        SI-filtered when it has usable SI evidence; pixels without SI
        evidence keep the count-based classification (fallback).

        Parameters
        ----------
        files        : list of binary river mask paths for the year
                       (names must embed the YYYY-MM-DD date)
        shape        : (rows, cols) of the binary masks
        water_count  : uint16 array, N_water per pixel
        valid_count  : uint16 array, N_valid per pixel

        Returns
        -------
        (reject_river, reject_inter) : bool arrays
            reject_river : always-water pixels FAILING the consistency rule
                           (should become land)
            reject_inter : switching pixels FAILING the high-SI rule
                           (should become land)
        """
        wet_si_n     = np.zeros(shape, dtype=np.uint16)
        wet_high_n   = np.zeros(shape, dtype=np.uint16)

        n_used = 0
        for fpath in files:
            date_str = self._date_from_filename(os.path.basename(fpath))
            if not date_str:
                continue

            si_path = self._si_path_for_date(date_str)
            if not si_path:
                continue

            si, _ = self._read_tif_float(si_path)
            if si is None or si.shape != shape:
                continue

            bin_arr, _ = self._read_tif(fpath)
            if bin_arr.shape != shape:
                continue

            # Per-image Otsu threshold — decides what 'high' SI means here
            thr = self._otsu_threshold(si, self.si_threshold)
            print(f"   Otsu threshold {date_str}: {thr:.4f}")

            valid_mask = (bin_arr != self.nodata_val)
            water_mask = (bin_arr == 1) & valid_mask

            wet_ok    = water_mask & np.isfinite(si)
            wet_high  = wet_ok & (si >= thr)

            wet_si_n   += wet_ok.astype(np.uint16)
            wet_high_n += wet_high.astype(np.uint16)

            n_used += 1

        reject_river = np.zeros(shape, dtype=bool)
        reject_inter = np.zeros(shape, dtype=bool)

        if n_used == 0:
            return reject_river, reject_inter

        has_wet_evidence = wet_si_n > 0

        # ── River rule ───────────────────────────────────────────────────────
        # Always-water pixels: SI must be in the high class (>= that image's
        # Otsu threshold) in EVERY image where the pixel is water.
        # Pixels with no wet SI evidence keep the count-based class.
        always_water = (water_count > 0) & (water_count == valid_count)
        river_ok     = has_wet_evidence & (wet_high_n == wet_si_n)
        reject_river = always_water & has_wet_evidence & ~river_ok

        # ── Intertidal rule ────────────────────────────────────────────────────
        # Switching pixels: SAME rule as river — SI must be in the high
        # class (>= that image's Otsu threshold) in EVERY image where the
        # pixel is water. Only pixels with wet SI evidence are SI-filtered;
        # the rest keep the count-based intertidal class.
        switches     = (water_count > 0) & (water_count < valid_count)
        inter_ok     = has_wet_evidence & (wet_high_n == wet_si_n)
        reject_inter = switches & has_wet_evidence & ~inter_ok

        return reject_river, reject_inter

    def _classify_year(self, files: list) -> np.ndarray:
        """
        Stack all river masks for one year and classify per pixel.

        Classification is count-based (not fraction-based), which is correct
        for sparse Landsat archives (2–5 scenes/yr in GBM delta):

            permanent land    N_water == 0               (never wet)
            permanent river   N_water == N_valid         (always wet)
            intertidal        0 < N_water < N_valid      (state changes)
            NoData            N_valid < min_scenes        (insufficient obs)

        If a singularity index folder is configured, the SI rules refine the
        count-based classes, with 'high' SI decided per image by OTSU
        thresholding:

            river      : always water AND SI >= image Otsu threshold in
                         EVERY scene where the pixel is water
            intertidal : switches AND SI >= image Otsu threshold in
                         EVERY scene where the pixel is water
            land       : everything else (failed river/intertidal candidates)

        Pixels without SI evidence keep the count-based class as fallback.

        Returns uint8 array with values:
            LAND (0), RIVER (1), INTERTIDAL (2), NODATA (255)
        """
        if not files:
            raise ValueError("No files provided for classification.")

        # Read first file to get shape
        first, _ = self._read_tif(files[0])
        shape = first.shape

        # Accumulators — use uint16 to handle up to 65535 scenes safely
        water_count = np.zeros(shape, dtype=np.uint16)   # N_water
        valid_count = np.zeros(shape, dtype=np.uint16)   # N_valid

        for fpath in files:
            arr, _ = self._read_tif(fpath)

            if arr.shape != shape:
                warnings.warn(
                    f"Shape mismatch in {os.path.basename(fpath)}: "
                    f"{arr.shape} vs expected {shape}. Skipping."
                )
                continue

            valid_mask = (arr != self.nodata_val)        # True where data exists
            water_mask = (arr == 1) & valid_mask         # True where river water

            valid_count += valid_mask.astype(np.uint16)
            water_count += water_mask.astype(np.uint16)

        # ── Classification — purely count-based ──────────────────────────────
        # Start with NODATA everywhere
        result = np.full(shape, NODATA, dtype=np.uint8)

        # Pixels with sufficient valid observations
        sufficient = valid_count >= self.min_scenes

        # Permanent land  : wet in 0 out of N_valid scenes
        land_mask  = sufficient & (water_count == 0)

        # Permanent river : wet in ALL N_valid scenes
        river_mask = sufficient & (water_count == valid_count)

        # Intertidal      : wet in at least 1 scene but not all
        #                   → state changed at least once during the year
        inter_mask = sufficient & (water_count > 0) & (water_count < valid_count)

        # ── SI validation of river and intertidal candidates ──────────────────
        # river      : always water AND SI in the high (Otsu) class of
        #              EVERY image where the pixel is water
        # intertidal : switches AND SI >= image Otsu threshold in EVERY
        #              scene where the pixel is water
        # everything else (with sufficient obs) : land
        if self.si_dir:
            reject_river, reject_inter = self._si_validation_masks(
                files, shape, water_count, valid_count
            )

            # Failed candidates — river and intertidal alike — become LAND
            river_mask = river_mask & ~reject_river
            inter_mask = inter_mask & ~reject_inter
            land_mask  = land_mask | reject_river | reject_inter

        result[land_mask]  = LAND
        result[river_mask] = RIVER
        result[inter_mask] = INTERTIDAL

        return result, valid_count, water_count

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_stats(self, classified: np.ndarray, year: int,
                       n_scenes: int) -> dict:
        """Return dict of pixel counts and km² areas per class."""
        px_m2 = self.pixel_size_m ** 2
        px_km2 = px_m2 / 1e6

        n_land      = int((classified == LAND).sum())
        n_river     = int((classified == RIVER).sum())
        n_inter     = int((classified == INTERTIDAL).sum())
        n_nodata    = int((classified == NODATA).sum())
        n_total     = classified.size

        return {
            'year':             year,
            'n_scenes':         n_scenes,
            'px_land':          n_land,
            'px_river':         n_river,
            'px_intertidal':    n_inter,
            'px_nodata':        n_nodata,
            'px_total':         n_total,
            'km2_land':         round(n_land    * px_km2, 3),
            'km2_river':        round(n_river   * px_km2, 3),
            'km2_intertidal':   round(n_inter   * px_km2, 3),
            'pct_intertidal':   round(n_inter / max(n_total - n_nodata, 1) * 100, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_year(self, year: int) -> dict:
        """
        Process a single year. Returns stats dict.

        Example
        -------
            det.run_year(2005)
        """
        by_year = self._collect_by_year()

        if year not in by_year:
            print(f"⚠️  No river files found for year {year}")
            return {}

        files   = by_year[year]
        out_tif = os.path.join(self.output_dir, f'intertidal_{year}.tif')

        if self.skip_existing and os.path.exists(out_tif):
            print(f"   ⏭️  {year}: output exists — skipping")
            return {}

        print(f"\n── {year} ──  ({len(files)} scenes)")
        print(f"   Files: {[os.path.basename(f) for f in files]}")

        if self.si_dir:
            si_by_year = self._collect_si_by_year()
            n_si = len(si_by_year.get(year, []))
            if n_si:
                print(f"   SI validation: {n_si} singularity index scenes found for {year} "
                      f"(per-image Otsu thresholding)")
            else:
                print(f"   SI validation: no SI files found for {year} "
                      f"— count-based classification only")

        classified, valid_count, water_count = self._classify_year(files)
        stats = self._compute_stats(classified, year, len(files))

        # Write GeoTIFF
        self._write_tif(out_tif, classified, files[0])

        print(f"   Land       : {stats['px_land']:>8,} px  ({stats['km2_land']:.2f} km²)")
        print(f"   River      : {stats['px_river']:>8,} px  ({stats['km2_river']:.2f} km²)")
        print(f"   Intertidal : {stats['px_intertidal']:>8,} px  ({stats['km2_intertidal']:.2f} km²)  "
              f"[{stats['pct_intertidal']:.1f}% of valid]")
        print(f"   NoData     : {stats['px_nodata']:>8,} px")
        print(f"   ✅ Saved → {out_tif}")

        return stats

    def run(self, years: list = None) -> list:
        """
        Process all years (or a specified list).

        Parameters
        ----------
        years : list of int, optional
            e.g. [2000, 2005, 2010]. If None, processes all available years.

        Returns
        -------
        list of dicts — one stats dict per year, also saved to CSV.

        Example
        -------
            det.run()
            det.run(years=[2000, 2005, 2010])
        """
        by_year = self._collect_by_year()

        if not by_year:
            print(f"⚠️  No binary_*.tif files found in: {self.river_dir}")
            return []

        target_years = sorted(years) if years else sorted(by_year.keys())

        print(f"\n{'='*60}")
        print(f"🌊  Intertidal Detection")
        print(f"   Years to process : {target_years}")
        print(f"   Total years found: {len(by_year)}")
        print(f"   Output dir       : {self.output_dir}")
        if self.si_dir:
            print(f"   SI validation   : ON — per-image OTSU thresholding ({self.si_dir})")
        print(f"{'='*60}")

        all_stats = []
        for yr in target_years:
            if yr not in by_year:
                print(f"⚠️  {yr}: no files found — skipping")
                continue
            stats = self.run_year(yr)
            if stats:
                all_stats.append(stats)

        if all_stats:
            self._save_csv(all_stats)

        # Print summary table
        print(f"\n{'='*60}")
        print(f"{'Year':>6}  {'Scenes':>7}  {'Land km²':>10}  "
              f"{'River km²':>10}  {'Intertidal km²':>15}  {'Intertidal %':>13}")
        print('-' * 68)
        for s in all_stats:
            print(f"{s['year']:>6}  {s['n_scenes']:>7}  {s['km2_land']:>10.2f}  "
                  f"{s['km2_river']:>10.2f}  {s['km2_intertidal']:>15.2f}  "
                  f"{s['pct_intertidal']:>12.1f}%")
        print(f"{'='*60}\n")

        return all_stats

    def summary(self, years: list = None) -> None:
        """
        Print a summary of available river files per year without processing.

        Example
        -------
            det.summary()
            det.summary(years=[2000, 2005])
        """
        by_year = self._collect_by_year()

        if not by_year:
            print(f"⚠️  No binary_*.tif files found in: {self.river_dir}")
            return

        target_years = sorted(years) if years else sorted(by_year.keys())

        print(f"\n{'Year':>6}  {'Scenes':>7}  {'Output exists':>14}  {'Date range'}")
        print('-' * 65)

        for yr in target_years:
            if yr not in by_year:
                print(f"{yr:>6}  {'—':>7}")
                continue

            files   = by_year[yr]
            out_tif = os.path.join(self.output_dir, f'intertidal_{yr}.tif')
            exists  = '✓' if os.path.exists(out_tif) else '✗'

            # Extract date range from filenames
            dates = []
            for f in files:
                date_str = self._date_from_filename(os.path.basename(f))
                if date_str:
                    dates.append(date_str)
            date_range = f"{dates[0]}  →  {dates[-1]}" if dates else '?'

            print(f"{yr:>6}  {len(files):>7}  {exists:>14}  {date_range}")

        total_files = sum(len(v) for v in by_year.values())
        print(f"\nTotal: {len(by_year)} years,  {total_files} scenes")

    # ─────────────────────────────────────────────────────────────────────────
    # CSV output
    # ─────────────────────────────────────────────────────────────────────────

    def _save_csv(self, all_stats: list) -> str:
        """Save summary stats to CSV."""
        import csv
        csv_path = os.path.join(self.output_dir, 'intertidal_summary.csv')

        fieldnames = [
            'year', 'n_scenes',
            'px_land', 'px_river', 'px_intertidal', 'px_nodata', 'px_total',
            'km2_land', 'km2_river', 'km2_intertidal',
            'pct_intertidal',
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)

        print(f"📊 Summary CSV → {csv_path}")
        return csv_path