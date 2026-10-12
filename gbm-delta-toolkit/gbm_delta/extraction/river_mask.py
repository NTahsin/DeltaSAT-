"""
river_from_geotiff.py
=====================
Post-process locally saved binary land/water GeoTIFFs (from Otsu thresholding)
to extract river channels by removing ponds, isolated patches, and noise.

Works entirely locally — no GEE connection needed.
Input  : binary GeoTIFF  (1=water, 0=land)
Output : binary GeoTIFF  (1=river water, 0=land/non-river)

Methods
-------
1. Connected Components (scipy.ndimage)
   - Labels all water blobs
   - Removes blobs smaller than min_area_pixels
   - Removes blobs with low elongation (circular = pond/lake)

2. Morphological cleaning
   - Binary opening to remove thin noise
   - Optional skeletonisation for channel centrelines

Requirements
------------
    pip install numpy scipy scikit-image tifffile

Usage
-----
    from gbm_delta.extraction.river_mask import RiverFromGeotiff

    rx = RiverFromGeotiff(
        input_dir    = r'W:\\...\\water_masks',
        output_dir   = r'W:\\...\\river_masks',
        min_area_px  = 56,        # ~0.05 km² at 30m resolution
        elongation   = 20,        # perimeter²/area — lower = more inclusive
    )

    rx.run()                      # all files in input_dir
    rx.run(year=2005)             # one year subfolder
    rx.process_file('water_mask_1996-11-14.tif')   # single file
    rx.summary()                  # preview what's in input_dir
"""

import os
import glob
import numpy as np


class RiverFromGeotiff:
    """
    Extracts river channels from binary water mask GeoTIFFs saved locally.

    Parameters
    ----------
    input_dir    : str    folder containing water_mask_YYYY-MM-dd.tif files
                          Can contain year subfolders or flat structure.
    output_dir   : str    where to save river_YYYY-MM-dd.tif outputs
    min_area_px  : int    minimum connected water blob size in pixels
                          default 56 ≈ 0.05 km² at 30m resolution
                          Use 556 for 0.5 km², 6 for 0.005 km²
    elongation   : float  perimeter²/area threshold — blobs ABOVE this are
                          kept (elongated = river-like).
                          Lower values = more inclusive (keeps more channels)
                          Higher values = stricter (only very elongated shapes)
                          Default 20. Try 10–50 depending on your area.
    morph_open   : int    morphological opening kernel size to remove thin
                          noise before CC labelling. 0 = skip. Default 1.
    skip_existing: bool   skip output files that already exist. Default True.
    nodata_val   : int    nodata value in input files. Default 0 (treated as land).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_area_px: int    = 56,
        elongation: float   = 20.0,
        morph_open: int     = 1,
        skip_existing: bool = True,
        nodata_val: int     = 255,
    ):
        self.input_dir     = input_dir
        self.output_dir    = output_dir
        self.min_area_px   = min_area_px
        self.elongation    = elongation
        self.morph_open    = morph_open
        self.skip_existing = skip_existing
        self.nodata_val    = nodata_val

        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Input  : {os.path.abspath(input_dir)}")
        print(f"📁 Output : {os.path.abspath(output_dir)}")
        print(f"   Min area     : {min_area_px} px  (~{min_area_px*30*30/1e6:.4f} km² at 30m)")
        print(f"   Elongation   : {elongation}")

    # ----------------------------------------------------------
    # I/O helpers
    # ----------------------------------------------------------

    @staticmethod
    def _read_tif(filepath: str):
        """
        Read a GeoTIFF and return (array, profile) using tifffile + manual
        georef, falling back to a simple read if needed.
        Returns (data: np.ndarray, meta: dict)
        """
        try:
            import tifffile
            data = tifffile.imread(filepath)
            # Read geotransform from raw TIFF tags
            meta = {'filepath': filepath}
            return data, meta
        except ImportError:
            raise ImportError("pip install tifffile")

    @staticmethod
    def _write_tif(filepath: str, data: np.ndarray, reference_path: str):
        """
        Write array as GeoTIFF, copying georeferencing from reference file.
        Uses tifffile for the array write, then patches in georeference via
        shutil copy of tags if available.
        """
        try:
            import tifffile
            # Copy raw file first to preserve all TIFF tags (georef etc.)
            import shutil
            shutil.copy2(reference_path, filepath)
            # Overwrite pixel data only
            with tifffile.TiffFile(filepath, mode='r+b') as tif:
                pass  # tifffile doesn't support in-place edit easily

            # Use gdal if available for clean georef write
            try:
                from osgeo import gdal, gdalconst
                ref_ds  = gdal.Open(reference_path)
                driver  = gdal.GetDriverByName('GTiff')
                out_ds  = driver.Create(
                    filepath,
                    ref_ds.RasterXSize,
                    ref_ds.RasterYSize,
                    1,
                    gdalconst.GDT_Byte,
                    options=['COMPRESS=LZW']
                )
                out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
                out_ds.SetProjection(ref_ds.GetProjection())
                out_ds.GetRasterBand(1).WriteArray(data.astype(np.uint8))
                out_ds.GetRasterBand(1).SetNoDataValue(255)
                out_ds.FlushCache()
                out_ds = None
                ref_ds = None
            except ImportError:
                # Fallback: tifffile with no georef (still viewable, just no CRS)
                tifffile.imwrite(filepath, data.astype(np.uint8),
                                 photometric='minisblack')
        except ImportError:
            raise ImportError("pip install tifffile")

    # ----------------------------------------------------------
    # Core processing
    # ----------------------------------------------------------

    def _process_array(self, water: np.ndarray) -> np.ndarray:
        """
        Given a binary water array (1=water, 0=land),
        return a binary river array (1=river, 0=other).
        """
        from scipy import ndimage
        from skimage import morphology

        # --- Step 1: morphological opening to remove single-pixel noise ---
        if self.morph_open > 0:
            selem = morphology.square(self.morph_open * 2 + 1)
            water = morphology.binary_opening(water, selem).astype(np.uint8)

        # --- Step 2: label connected components ---
        structure = np.ones((3, 3), dtype=int)   # 8-connectivity
        labeled, n_blobs = ndimage.label(water, structure=structure)

        if n_blobs == 0:
            return np.zeros_like(water, dtype=np.uint8)

        # --- Step 3: compute area and perimeter per blob ---
        blob_ids   = np.arange(1, n_blobs + 1)
        areas      = ndimage.sum(water, labeled, blob_ids)

        # Perimeter: erode then subtract → edge pixels
        eroded     = ndimage.binary_erosion(water, structure=structure)
        edge       = water.astype(bool) & ~eroded
        perimeters = ndimage.sum(edge.astype(np.uint8), labeled, blob_ids)

        # --- Step 4: elongation index = perimeter² / area ---
        elongations = np.where(
            areas > 0,
            (perimeters ** 2) / (areas + 1e-10),
            0
        )

        # --- Step 5: keep blobs that are large AND elongated ---
        keep_ids = blob_ids[
            (areas      >= self.min_area_px) &
            (elongations >= self.elongation)
        ]

        if len(keep_ids) == 0:
            return np.zeros_like(water, dtype=np.uint8)

        # Build output mask
        river = np.isin(labeled, keep_ids).astype(np.uint8)
        return river

    # ----------------------------------------------------------
    # File-level processing
    # ----------------------------------------------------------

    def process_file(self, input_path: str, output_path: str = None) -> str:
        """
        Process a single water mask GeoTIFF and save river mask.

        Parameters
        ----------
        input_path  : str  full path to input water_mask_*.tif
        output_path : str  optional output path. If None, auto-generated
                           in output_dir with 'river_' prefix.

        Returns
        -------
        str : path to saved output file
        """
        fname = os.path.basename(input_path)

        # Auto-generate output path
        if output_path is None:
            out_name    = fname.replace('water_mask_', 'river_')
            # Mirror year subfolder structure if present
            parent      = os.path.basename(os.path.dirname(input_path))
            if parent.isdigit() and len(parent) == 4:
                year_dir    = os.path.join(self.output_dir, parent)
                os.makedirs(year_dir, exist_ok=True)
                output_path = os.path.join(year_dir, out_name)
            else:
                output_path = os.path.join(self.output_dir, out_name)

        if self.skip_existing and os.path.exists(output_path):
            print(f"   ⏭️  Skipping (exists): {os.path.basename(output_path)}")
            return output_path

        # Read
        data, meta = self._read_tif(input_path)

        # Handle multi-band or squeezed arrays
        if data.ndim == 3:
            data = data[0]

        # Binarise: treat 1 as water, everything else as land
        # Also handle nodata (255 or masked areas)
        water = ((data == 1)).astype(np.uint8)

        n_water = water.sum()
        if n_water == 0:
            print(f"   ⚠️  No water pixels in {fname} — saving empty mask")
            river = np.zeros_like(water)
        else:
            river = self._process_array(water)

        n_river = river.sum()
        pct     = (n_river / n_water * 100) if n_water > 0 else 0
        print(f"   Water px: {n_water:>8,}  →  River px: {n_river:>8,}  ({pct:.1f}% retained)")

        # Write
        self._write_tif(output_path, river, input_path)
        return output_path

    # ----------------------------------------------------------
    # Batch processing
    # ----------------------------------------------------------

    def _collect_files(self, year: int = None) -> list:
        """Collect all water_mask_*.tif files, optionally filtered by year."""
        if year is not None:
            # Look in year subfolder first, then flat
            patterns = [
                os.path.join(self.input_dir, str(year), 'water_mask_*.tif'),
                os.path.join(self.input_dir, f'water_mask_{year}-*.tif'),
            ]
        else:
            patterns = [
                os.path.join(self.input_dir, '**', 'water_mask_*.tif'),
                os.path.join(self.input_dir, 'water_mask_*.tif'),
            ]

        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))

        # Deduplicate and sort by date
        files = sorted(set(files))
        return files

    def run(self, year: int = None) -> dict:
        """
        Process all water mask GeoTIFFs in input_dir.

        Parameters
        ----------
        year : int  optional — process only files from this year

        Returns
        -------
        dict  {input_path: output_path}

        Examples
        --------
            rx.run()            # all files
            rx.run(year=2005)   # one year only
        """
        files = self._collect_files(year=year)

        if not files:
            print(f"⚠️  No water_mask_*.tif files found in: {self.input_dir}")
            return {}

        label = str(year) if year else 'all years'
        print(f"\n{'='*55}")
        print(f"🌊 River Extraction from GeoTIFFs")
        print(f"   Files     : {len(files)}")
        print(f"   Period    : {label}")
        print(f"   Min area  : {self.min_area_px} px")
        print(f"   Elongation: {self.elongation}")
        print(f"{'='*55}\n")

        results   = {}
        processed = 0
        skipped   = 0
        failed    = 0
        empty     = 0

        for i, fpath in enumerate(files):
            fname    = os.path.basename(fpath)
            date_str = fname.replace('water_mask_', '').replace('.tif', '')
            print(f"[{i+1:>3}/{len(files)}] {date_str}")

            try:
                out = self.process_file(fpath)
                results[fpath] = out

                if out and os.path.exists(out):
                    processed += 1
                else:
                    empty += 1

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[fpath] = None
                failed += 1

        print(f"\n{'='*55}")
        print(f"✅ Complete")
        print(f"   Processed : {processed}")
        print(f"   Skipped   : {skipped}")
        print(f"   Empty     : {empty}")
        print(f"   Failed    : {failed}")
        print(f"   Output    : {self.output_dir}")
        print(f"{'='*55}\n")
        return results

    def summary(self, year: int = None) -> None:
        """
        Print a summary of available input files and output status.

        Example
        -------
            rx.summary(year=2005)
        """
        files = self._collect_files(year=year)
        print(f"\n{'File':<35} {'Water px':>10}  {'Output exists'}")
        print('-' * 60)

        for fpath in files:
            fname    = os.path.basename(fpath)
            date_str = fname.replace('water_mask_', '').replace('.tif', '')

            # Quick pixel count
            try:
                import tifffile
                data    = tifffile.imread(fpath)
                if data.ndim == 3:
                    data = data[0]
                n_water = int((data == 1).sum())
            except Exception:
                n_water = -1

            # Check output
            out_name   = fname.replace('water_mask_', 'river_')
            parent     = os.path.basename(os.path.dirname(fpath))
            if parent.isdigit():
                out_path = os.path.join(self.output_dir, parent, out_name)
            else:
                out_path = os.path.join(self.output_dir, out_name)

            exists = '✓' if os.path.exists(out_path) else '✗'
            print(f"{date_str:<35} {n_water:>10,}  {exists}")

        print(f"\nTotal: {len(files)} files")
