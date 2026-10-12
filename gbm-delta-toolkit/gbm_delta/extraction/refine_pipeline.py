"""
landsat_rivermap_pipeline.py
============================
Four fully independent, step-by-step classes for Landsat-based river
channel mapping in the Ganges-Brahmaputra Delta (or any region).

Each class is called independently. Outputs from one stage become the
inputs to the next. Nothing runs automatically between stages.

Step-by-step workflow
---------------------

  STEP 1  [GEE — requires internet + Google Earth Engine account]
  ----------------------------------------------------------------
  MNDWIExporter
    • Builds a cloud-masked, SR-corrected Landsat 5 + 8 collection in GEE
    • Computes MNDWI = (Green − SWIR1) / (Green + SWIR1) for every scene
    • Downloads one MNDWI GeoTIFF per acquisition date to a local folder
      via geemap (no Google Drive needed)
    Output files: <mndwi_dir>/mndwi_YYYY-MM-DD.tif

  STEP 2  [Local — no internet needed]
  -------------------------------------
  SingularityIndexProcessor
    • Reads the locally saved MNDWI GeoTIFFs from Step 1
    • Applies the RivaMap multiscale singularity index (MMSI)
    • Saves psi, width, and orientation GeoTIFFs per date
    Output files: <output_dir>/psi/psi_YYYY-MM-DD.tif
                  <output_dir>/width/width_YYYY-MM-DD.tif
                  <output_dir>/orient/orient_YYYY-MM-DD.tif

  STEP 3  [Local — no internet needed]
  -------------------------------------
  OtsuBinaryClassifier
    • Reads the psi GeoTIFFs from Step 2
    • Computes a per-date Otsu threshold on the psi values
    • Saves a binary GeoTIFF per date:
        river pixels → NaN
        land  pixels → 999
    Output files: <output_dir>/binary_YYYY-MM-DD.tif

  STEP 4  [Local — no internet needed]
  -------------------------------------
  ChannelMaskRefiner
    • Reads the binary GeoTIFFs from Step 3
    • Runs connected-component labelling and removes small patches
      (standing water, noise) below min_component_px
    • Saves a refined channel mask GeoTIFF per date:
        channelised river → NaN
        land / small patches → 999
    Output files: <output_dir>/channel_mask_YYYY-MM-DD.tif

Dependencies
------------
    Step 1:  pip install earthengine-api geemap
    Step 2:  pip install rivamap opencv-python numpy rasterio
    Steps 3/4: pip install numpy rasterio scikit-image

Example usage
-------------
    import ee
    ee.Initialize()

    from landsat_rivermap_pipeline import (
        MNDWIExporter,
        SingularityIndexProcessor,
        OtsuBinaryClassifier,
        ChannelMaskRefiner,
    )

    polygon = ee.Geometry.Rectangle([88.0, 21.5, 90.5, 23.5])

    # ── Step 1 ──────────────────────────────────────────────────
    exporter = MNDWIExporter(
        polygon    = polygon,
        wrs_path   = 137,
        wrs_row    = 44,
        start_date = '2000-01-01',
        end_date   = '2020-01-01',
        output_dir = r'C:/outputs/mndwi',
    )
    exporter.export_all()

    # ── Step 2 ──────────────────────────────────────────────────
    si = SingularityIndexProcessor(
        mndwi_dir  = r'C:/outputs/mndwi',
        output_dir = r'C:/outputs/singularity',
    )
    si.process_all()

    # ── Step 3 ──────────────────────────────────────────────────
    otsu = OtsuBinaryClassifier(
        psi_dir    = r'C:/outputs/singularity/psi',
        output_dir = r'C:/outputs/binary',
    )
    otsu.process_all()

    # ── Step 4 ──────────────────────────────────────────────────
    refiner = ChannelMaskRefiner(
        binary_dir       = r'C:/outputs/binary',
        output_dir       = r'C:/outputs/channel_mask',
        min_component_px = 500,
    )
    refiner.process_all()
"""

from __future__ import annotations

import os
import glob
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Shared local file-system helpers  (used by Steps 2, 3, 4)
# =============================================================================

def _read_band(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read a single-band GeoTIFF into a float32 numpy array.
    NoData values are replaced with NaN.

    Returns
    -------
    array   : 2-D float32 ndarray
    profile : rasterio profile dict  (pass this to _write_geotiff)
    """
    import rasterio
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        data    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
    return data, profile


def _write_geotiff(
    path:    str,
    array:   np.ndarray,
    profile: dict,
    nodata:  Optional[float] = None,
) -> None:
    """
    Write a 2-D float32 array to a deflate-compressed GeoTIFF,
    preserving the CRS and geotransform from *profile*.
    Parent directories are created automatically.
    """
    import rasterio
    out_profile = profile.copy()
    out_profile.update(
        dtype    = rasterio.float32,
        count    = 1,
        nodata   = nodata,
        compress = 'deflate',
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def _extract_date_tag(filename: str) -> str:
    """
    Pull a YYYY-MM-DD string from *filename*.
    Falls back to the file stem (name without extension) if no date found.
    """
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else Path(filename).stem


def _list_tifs(directory: str, pattern: str = '*.tif') -> list[str]:
    """
    Sorted list of GeoTIFF paths in *directory* matching *pattern*.
    Tries lowercase .tif first, then uppercase .TIF as fallback.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(directory, pattern.upper())))
    return paths


# =============================================================================
# STEP 1 — MNDWIExporter  (GEE)
# =============================================================================

class MNDWIExporter:
    """
    Build a cloud-masked, surface-reflectance-corrected Landsat 5 + 8
    collection in Google Earth Engine, compute MNDWI for every scene,
    and download each MNDWI image as a local GeoTIFF.

    What happens inside GEE before export
    --------------------------------------
    1. Filter by WRS path/row, date range, cloud cover threshold.
    2. Rename bands to harmonised names (Blue, Green, Red, NIR, SWIR1, …).
    3. Apply Landsat C02 surface-reflectance scale factors:
         optical = DN × 0.0000275 − 0.2   → reflectance in [−0.2, 0.5]
    4. Pixel-level cloud / shadow / saturation mask via QA_PIXEL bitmask.
    5. Reproject all scenes to a common CRS and scale.
    6. Compute MNDWI = normalizedDifference(Green, SWIR1) → range [−1, 1].
    7. Export the single-band MNDWI image to local disk via geemap.

    Landsat 7 is intentionally excluded (SLC-off stripe artefacts).

    Parameters
    ----------
    polygon               : ee.Geometry   Area of interest
    wrs_path              : int
    wrs_row               : int
    start_date            : str           'YYYY-MM-DD'
    end_date              : str           'YYYY-MM-DD'
    output_dir            : str           Local folder for MNDWI GeoTIFFs
    cloud_cover_threshold : int           Image-level pre-filter (default 30)
    common_crs            : str           EPSG string (default 'EPSG:32646')
    common_scale          : int           Resolution in metres (default 30)

    Output files
    ------------
    <output_dir>/mndwi_YYYY-MM-DD.tif   — one file per acquisition date
    """

    # ── Landsat band definitions ──────────────────────────────────────────────
    _L5_BANDS  = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7',
                  'ST_B6','QA_PIXEL','QA_RADSAT']
    _L8_BANDS  = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
                  'ST_B10','QA_PIXEL','QA_RADSAT']
    _OUT_BANDS = ['Blue','Green','Red','NIR','SWIR1','SWIR2',
                  'TIR','QA_PIXEL','QA_RADSAT']

    def __init__(
        self,
        polygon,                          # ee.Geometry
        wrs_path:              int,
        wrs_row:               int,
        start_date:            str,
        end_date:              str,
        output_dir:            str,
        cloud_cover_threshold: int = 30,
        common_crs:            str = 'EPSG:32646',
        common_scale:          int = 30,
    ):
        self.polygon               = polygon
        self.wrs_path              = wrs_path
        self.wrs_row               = wrs_row
        self.start_date            = start_date
        self.end_date              = end_date
        self.output_dir            = output_dir
        self.cloud_cover_threshold = cloud_cover_threshold
        self.common_crs            = common_crs
        self.common_scale          = common_scale
        os.makedirs(output_dir, exist_ok=True)

    # ── private GEE helpers ───────────────────────────────────────────────────

    def _base_filter(self, collection_id: str):
        """Filter a GEE collection by path/row, date, bounds, cloud cover."""
        import ee
        return (
            ee.ImageCollection(collection_id)
            .filter(ee.Filter.eq('WRS_PATH', self.wrs_path))
            .filter(ee.Filter.eq('WRS_ROW',  self.wrs_row))
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.polygon)
            .filter(ee.Filter.lt('CLOUD_COVER', self.cloud_cover_threshold))
        )

    @staticmethod
    def _apply_scale_factors(image):
        """
        Convert Landsat C02 T1_L2 integer DNs to surface reflectance.
        Optical: SR = DN × 0.0000275 − 0.2
        Runs after band renaming so harmonised names are used.
        """
        optical = (
            image.select(['Blue','Green','Red','NIR','SWIR1','SWIR2'])
                 .multiply(0.0000275).add(-0.2)
        )
        thermal = (
            image.select(['TIR'])
                 .multiply(0.00341802).add(149.0)
        )
        return (
            image.addBands(optical, None, True)
                 .addBands(thermal, None, True)
        )

    @staticmethod
    def _mask_clouds(image):
        """
        Pixel-level cloud / shadow / dilation / saturation mask
        using QA_PIXEL and QA_RADSAT bitmask bands.
        Works for both L5 C02 and L8 C02 after band renaming.
        """
        qa   = image.select('QA_PIXEL')
        mask = (
            qa.bitwiseAnd(1 << 1).eq(0)              # dilated cloud
              .And(qa.bitwiseAnd(1 << 3).eq(0))       # cloud shadow
              .And(qa.bitwiseAnd(1 << 4).eq(0))       # cloud
        )
        sat = image.select('QA_RADSAT').eq(0)
        return (
            image.updateMask(mask)
                 .updateMask(sat)
                 .copyProperties(image, ['system:time_start'])
        )

    def _rename(self, col, src_bands: list):
        """Rename src_bands to harmonised OUT_BANDS."""
        import ee
        return col.map(lambda img: img.select(src_bands, self._OUT_BANDS))

    def _align(self, image):
        """Reproject to common CRS and pixel scale."""
        return image.reproject(crs=self.common_crs, scale=self.common_scale)

    @staticmethod
    def _add_mndwi(image):
        """
        Add MNDWI = normalizedDifference(Green, SWIR1) as a band.
        Result range: [−1, 1].  Higher values → more water.
        """
        mndwi = image.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')
        return image.addBands(mndwi)

    def _build_collection(self):
        """Build the merged, processed L5 + L8 collection with MNDWI band."""
        # Landsat 5
        l5 = self._base_filter('LANDSAT/LT05/C02/T1_L2')
        l5 = self._rename(l5, self._L5_BANDS)
        l5 = l5.map(self._apply_scale_factors)
        l5 = l5.map(self._mask_clouds)

        # Landsat 8
        l8 = self._base_filter('LANDSAT/LC08/C02/T1_L2')
        l8 = self._rename(l8, self._L8_BANDS)
        l8 = l8.map(self._apply_scale_factors)
        l8 = l8.map(self._mask_clouds)

        merged = (
            l5.merge(l8)
              .map(self._align)
              .map(self._add_mndwi)
              .sort('system:time_start')
        )
        return merged

    @staticmethod
    def _get_geemap():
        try:
            import geemap
            return geemap
        except ImportError:
            raise ImportError(
                "geemap is required for local export.\n"
                "Install with:  pip install geemap"
            )

    # ── public ───────────────────────────────────────────────────────────────

    def export_all(self) -> list[str]:
        """
        Compute MNDWI for every scene in the collection and download each
        as a local GeoTIFF named  mndwi_YYYY-MM-DD.tif.

        Returns
        -------
        list[str] : paths of all saved MNDWI GeoTIFFs
        """
        import ee
        geemap = self._get_geemap()

        collection = self._build_collection()
        n          = collection.size().getInfo()
        image_list = collection.toList(n)
        saved      = []

        print(f"\n📋 MNDWIExporter — {n} scene(s) to export")
        print(f"   Output : {self.output_dir}\n")

        for i in range(n):
            image    = ee.Image(image_list.get(i))
            date_str = (ee.Date(image.get('system:time_start'))
                          .format('YYYY-MM-dd').getInfo())
            mndwi    = image.select('MNDWI')

            filename = os.path.join(self.output_dir, f'mndwi_{date_str}.tif')

            print(f"── [{i+1}/{n}] {date_str}  ⬇  {filename}")
            geemap.ee_export_image(
                mndwi,
                filename      = filename,
                scale         = self.common_scale,
                region        = self.polygon,
                crs           = self.common_crs,
                file_per_band = False,
            )
            saved.append(filename)
            print(f"   ✓ saved")

        print(f"\n✅ MNDWI export complete — {n} files in: {self.output_dir}")
        return saved

    def collection_size(self) -> int:
        """Return the number of scenes in the collection (makes a GEE call)."""
        return self._build_collection().size().getInfo()


# =============================================================================
# STEP 2 — SingularityIndexProcessor  (local)
# =============================================================================

class SingularityIndexProcessor:
    """
    Apply the RivaMap multiscale singularity index (MMSI) to every MNDWI
    GeoTIFF saved by MNDWIExporter and write the results to local GeoTIFFs.

    Input
    -----
    Folder of MNDWI GeoTIFFs from Step 1.
    Files must contain a YYYY-MM-DD string in their name:
        mndwi_2005-01-14.tif

    Output
    ------
    <output_dir>/psi/psi_YYYY-MM-DD.tif        — singularity index response
    <output_dir>/width/width_YYYY-MM-DD.tif    — per-pixel channel width
    <output_dir>/orient/orient_YYYY-MM-DD.tif  — channel orientation (rad)

    The psi folder is the input for OtsuBinaryClassifier (Step 3).

    Parameters
    ----------
    mndwi_dir     : str   Folder containing MNDWI GeoTIFFs
    output_dir    : str   Root output folder; sub-folders created automatically
    mndwi_pattern : str   Glob pattern (default 'mndwi_*.tif')

    Requires
    --------
        pip install rivamap opencv-python rasterio numpy
    """

    def __init__(
        self,
        mndwi_dir:     str,
        output_dir:    str,
        mndwi_pattern: str = 'mndwi_*.tif',
    ):
        self.mndwi_dir     = mndwi_dir
        self.output_dir    = output_dir
        self.mndwi_pattern = mndwi_pattern

        self.psi_dir    = os.path.join(output_dir, 'psi')
        self.width_dir  = os.path.join(output_dir, 'width')
        self.orient_dir = os.path.join(output_dir, 'orient')
        for d in (self.psi_dir, self.width_dir, self.orient_dir):
            os.makedirs(d, exist_ok=True)

        # RivaMap filter bank — created once on first use
        self._filters           = None
        self._singularity_index = None

    # ── private ──────────────────────────────────────────────────────────────

    def _load_rivamap(self):
        """Import RivaMap and initialise the MMSI filter bank (cached)."""
        if self._filters is None:
            try:
                from rivamap import singularity_index as si
                self._singularity_index = si
                self._filters = si.SingularityIndexFilters()
            except ImportError:
                raise ImportError(
                    "rivamap is required for SingularityIndexProcessor.\n"
                    "Install with:  pip install rivamap opencv-python"
                )
        return self._filters

    @staticmethod
    def _prepare_for_rivamap(mndwi: np.ndarray) -> np.ndarray:
        """
        RivaMap's applyMMSI expects values in [0, 1].
        MNDWI from GEE is in [−1, 1] → rescale linearly.
        NaN pixels (NoData / cloud mask) are set to 0 before passing to
        RivaMap and re-masked afterwards.
        """
        rescaled = (mndwi + 1.0) / 2.0
        return np.where(np.isnan(rescaled), 0.0, rescaled)

    # ── public ───────────────────────────────────────────────────────────────

    def process_single(self, mndwi_path: str, date_tag: str) -> dict[str, str]:
        """
        Run MMSI on one MNDWI GeoTIFF and save psi / width / orient outputs.

        Parameters
        ----------
        mndwi_path : str   Path to the MNDWI GeoTIFF
        date_tag   : str   Date string used in output filenames

        Returns
        -------
        dict with keys 'psi', 'width', 'orient' → paths of saved GeoTIFFs
        """
        filters = self._load_rivamap()

        mndwi, profile = _read_band(mndwi_path)
        nan_mask       = np.isnan(mndwi)

        psi, width_map, orient = self._singularity_index.applyMMSI(
            self._prepare_for_rivamap(mndwi), filters
        )

        # Restore the original NoData mask
        psi[nan_mask]       = np.nan
        width_map[nan_mask] = np.nan
        orient[nan_mask]    = np.nan

        psi_path    = os.path.join(self.psi_dir,    f'psi_{date_tag}.tif')
        width_path  = os.path.join(self.width_dir,  f'width_{date_tag}.tif')
        orient_path = os.path.join(self.orient_dir, f'orient_{date_tag}.tif')

        _write_geotiff(psi_path,    psi,       profile, nodata=np.nan)
        _write_geotiff(width_path,  width_map, profile, nodata=np.nan)
        _write_geotiff(orient_path, orient,    profile, nodata=np.nan)

        print(f"   [SI] {date_tag} → psi, width, orient saved")
        return {'psi': psi_path, 'width': width_path, 'orient': orient_path}

    def process_all(self) -> list[dict[str, str]]:
        """
        Process every MNDWI GeoTIFF found in mndwi_dir.

        Returns
        -------
        list of dicts, one per date, each with keys 'psi', 'width', 'orient'
        """
        files = _list_tifs(self.mndwi_dir, self.mndwi_pattern)
        if not files:
            raise FileNotFoundError(
                f"No MNDWI GeoTIFFs found in '{self.mndwi_dir}' "
                f"matching pattern '{self.mndwi_pattern}'.\n"
                f"Run MNDWIExporter (Step 1) first."
            )

        n       = len(files)
        results = []
        print(f"\n📋 SingularityIndexProcessor — {n} date(s) to process")
        print(f"   Input  : {self.mndwi_dir}")
        print(f"   Output : {self.output_dir}\n")

        for i, path in enumerate(files, 1):
            tag = _extract_date_tag(os.path.basename(path))
            print(f"── [{i}/{n}] {tag}")
            results.append(self.process_single(path, tag))

        print(f"\n✅ Singularity index complete — outputs in: {self.output_dir}")
        return results


# =============================================================================
# STEP 3 — OtsuBinaryClassifier  (local)
# =============================================================================

class OtsuBinaryClassifier:
    """
    Classify each psi GeoTIFF (from Step 2) into a binary river / land
    image using a per-date Otsu threshold, and save as a GeoTIFF.

    Encoding convention
    -------------------
    river pixels  → NaN   (float32 NoData — renders transparent in GIS)
    land  pixels  → 999   (float32 sentinel value)

    How the Otsu threshold is computed
    -----------------------------------
    scikit-image's threshold_otsu maximises between-class variance across
    the histogram of valid (non-NaN) psi pixels — the same principle as the
    GEE-side Otsu in the original WaterExtractor class, now applied locally
    on the downloaded psi raster.

    Input
    -----
    Folder of psi GeoTIFFs from Step 2:
        <singularity_dir>/psi/psi_YYYY-MM-DD.tif

    Output
    ------
    <output_dir>/binary_YYYY-MM-DD.tif

    Parameters
    ----------
    psi_dir     : str   Folder containing psi GeoTIFFs
    output_dir  : str   Where to write binary GeoTIFFs
    psi_pattern : str   Glob pattern (default 'psi_*.tif')
    land_value  : float Sentinel value assigned to land pixels (default 999)

    Requires
    --------
        pip install scikit-image rasterio numpy
    """

    def __init__(
        self,
        psi_dir:     str,
        output_dir:  str,
        psi_pattern: str   = 'psi_*.tif',
        land_value:  float = 999.0,
    ):
        self.psi_dir     = psi_dir
        self.output_dir  = output_dir
        self.psi_pattern = psi_pattern
        self.land_value  = float(land_value)
        os.makedirs(output_dir, exist_ok=True)

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_otsu(psi: np.ndarray) -> float:
        """Compute the Otsu threshold from the valid (non-NaN) psi values."""
        from skimage.filters import threshold_otsu
        valid = psi[~np.isnan(psi)].ravel()
        if valid.size == 0:
            raise ValueError(
                "psi image contains only NaN — cannot compute Otsu threshold.\n"
                "Check that the MNDWI export and singularity index ran correctly."
            )
        return float(threshold_otsu(valid))

    def _make_binary(
        self,
        psi:         np.ndarray,
        threshold:   float,
        nodata_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Threshold psi to produce a binary float32 array.

        psi > threshold  → river → NaN
        psi ≤ threshold  → land  → self.land_value
        original NoData  → NaN   (preserved)
        """
        binary                  = np.full_like(psi, self.land_value, dtype=np.float32)
        binary[psi > threshold] = np.nan    # river pixels
        binary[nodata_mask]     = np.nan    # original NoData
        return binary

    # ── public ───────────────────────────────────────────────────────────────

    def process_single(
        self,
        psi_path: str,
        date_tag: str,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Classify one psi GeoTIFF and save the binary result.

        Parameters
        ----------
        psi_path : str   Path to the psi GeoTIFF
        date_tag : str   Date string used in the output filename

        Returns
        -------
        binary    : float32 ndarray  (NaN = river, 999 = land)
        threshold : float            Otsu threshold that was applied
        out_path  : str              Path of the saved binary GeoTIFF
        """
        psi, profile = _read_band(psi_path)
        nodata_mask  = np.isnan(psi)

        threshold = self._compute_otsu(psi)
        binary    = self._make_binary(psi, threshold, nodata_mask)

        out_path = os.path.join(self.output_dir, f'binary_{date_tag}.tif')
        _write_geotiff(out_path, binary, profile, nodata=np.nan)

        print(f"   [Otsu] {date_tag}  threshold={threshold:.5f}  → {out_path}")
        return binary, threshold, out_path

    def process_all(self) -> list[dict]:
        """
        Classify all psi GeoTIFFs found in psi_dir.

        Returns
        -------
        list of dicts, one per date:
            'date'      : str
            'threshold' : float
            'path'      : str   path to the saved binary GeoTIFF
        """
        files = _list_tifs(self.psi_dir, self.psi_pattern)
        if not files:
            raise FileNotFoundError(
                f"No psi GeoTIFFs found in '{self.psi_dir}' "
                f"matching pattern '{self.psi_pattern}'.\n"
                f"Run SingularityIndexProcessor (Step 2) first."
            )

        n       = len(files)
        results = []
        print(f"\n📋 OtsuBinaryClassifier — {n} date(s) to classify")
        print(f"   Input  : {self.psi_dir}")
        print(f"   Output : {self.output_dir}\n")

        for i, path in enumerate(files, 1):
            tag = _extract_date_tag(os.path.basename(path))
            print(f"── [{i}/{n}] {tag}")
            _, thr, out_path = self.process_single(path, tag)
            results.append({'date': tag, 'threshold': thr, 'path': out_path})

        print(f"\n✅ Otsu classification complete — outputs in: {self.output_dir}")
        return results


# =============================================================================
# STEP 4 — ChannelMaskRefiner  (local)
# =============================================================================

class ChannelMaskRefiner:
    """
    Remove small, isolated water patches from the binary images (Step 3)
    using connected-component labelling, retaining only large channelised
    river bodies.

    Why this is needed
    ------------------
    The Otsu binary image contains all pixels where psi > threshold,
    including small standing-water bodies, tidal flats, and noise.
    Connected-component filtering keeps only spatially coherent river
    channels above a minimum size, discarding everything else.

    Algorithm
    ---------
    1. Read binary GeoTIFF  (NaN = river, 999 = land).
    2. Build a boolean river mask  (True where pixel is NaN).
    3. Label connected components (scikit-image).
    4. Discard components with fewer than min_component_px pixels.
    5. Write refined binary GeoTIFF with the same NaN / 999 encoding.

    Input
    -----
    Folder of binary GeoTIFFs from Step 3:
        binary_YYYY-MM-DD.tif

    Output
    ------
    <output_dir>/channel_mask_YYYY-MM-DD.tif

    Parameters
    ----------
    binary_dir       : str   Folder containing binary GeoTIFFs
    output_dir       : str   Where to write refined channel mask GeoTIFFs
    min_component_px : int   Minimum connected-component size to keep.
                             At 30 m Landsat resolution:
                               100 px ≈ 0.09 km²
                               500 px ≈ 0.45 km²  ← default
                              1000 px ≈ 0.90 km²
                             Increase if small wetland patches remain.
    binary_pattern   : str   Glob pattern (default 'binary_*.tif')
    land_value       : float Must match OtsuBinaryClassifier (default 999)
    connectivity     : int   1 = 4-connected, 2 = 8-connected (default 2)

    Requires
    --------
        pip install scikit-image rasterio numpy
    """

    def __init__(
        self,
        binary_dir:       str,
        output_dir:       str,
        min_component_px: int   = 500,
        binary_pattern:   str   = 'binary_*.tif',
        land_value:       float = 999.0,
        connectivity:     int   = 2,
    ):
        self.binary_dir       = binary_dir
        self.output_dir       = output_dir
        self.min_component_px = min_component_px
        self.binary_pattern   = binary_pattern
        self.land_value       = float(land_value)
        self.connectivity     = connectivity
        os.makedirs(output_dir, exist_ok=True)

    # ── private ──────────────────────────────────────────────────────────────

    def _filter_components(self, river_bool: np.ndarray) -> np.ndarray:
        """
        Label connected river components and return a boolean mask keeping
        only those >= min_component_px pixels in area.
        """
        from skimage.measure import label, regionprops

        labelled = label(river_bool, connectivity=self.connectivity)
        refined  = np.zeros_like(river_bool, dtype=bool)

        for region in regionprops(labelled):
            if region.area >= self.min_component_px:
                refined[labelled == region.label] = True

        n_before = int(river_bool.sum())
        n_after  = int(refined.sum())
        print(
            f"      river px: {n_before:,} → {n_after:,}  "
            f"(removed {n_before - n_after:,} small-patch px)"
        )
        return refined

    # ── public ───────────────────────────────────────────────────────────────

    def process_single(self, binary_path: str, date_tag: str) -> str:
        """
        Refine one binary GeoTIFF using connected-component filtering.

        Parameters
        ----------
        binary_path : str   Path to the binary GeoTIFF
        date_tag    : str   Date string used in the output filename

        Returns
        -------
        str : path of the saved channel mask GeoTIFF
        """
        binary, profile = _read_band(binary_path)

        # NaN pixels in the binary image = river (as set in Step 3)
        river_bool   = np.isnan(binary)
        refined_bool = self._filter_components(river_bool)

        # Rebuild with the same NaN / land_value encoding
        refined                  = np.full_like(binary, self.land_value, dtype=np.float32)
        refined[refined_bool]    = np.nan    # retained river channels

        out_path = os.path.join(self.output_dir, f'channel_mask_{date_tag}.tif')
        _write_geotiff(out_path, refined, profile, nodata=np.nan)

        print(f"   [CC]   {date_tag} → {out_path}")
        return out_path

    def process_all(self) -> list[str]:
        """
        Refine all binary GeoTIFFs found in binary_dir.

        Returns
        -------
        list[str] : paths to all saved channel mask GeoTIFFs
        """
        files = _list_tifs(self.binary_dir, self.binary_pattern)
        if not files:
            raise FileNotFoundError(
                f"No binary GeoTIFFs found in '{self.binary_dir}' "
                f"matching pattern '{self.binary_pattern}'.\n"
                f"Run OtsuBinaryClassifier (Step 3) first."
            )

        n     = len(files)
        saved = []
        print(f"\n📋 ChannelMaskRefiner — {n} date(s) to refine")
        print(f"   Input            : {self.binary_dir}")
        print(f"   Output           : {self.output_dir}")
        print(f"   min_component_px : {self.min_component_px}\n")

        for i, path in enumerate(files, 1):
            tag = _extract_date_tag(os.path.basename(path))
            print(f"── [{i}/{n}] {tag}")
            out_path = self.process_single(path, tag)
            saved.append(out_path)

        print(f"\n✅ Channel refinement complete — outputs in: {self.output_dir}")
        return saved
