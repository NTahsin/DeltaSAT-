"""
landsat_gbm.py
==============
Object-oriented Google Earth Engine pipeline for Landsat-based
water extraction in the Ganges-Brahmaputra Delta.

Excludes Landsat 7 (SLC-off artefacts).
Provides pixel-level cloud masking, surface reflectance scaling,
MNDWI/NDWI computation, Otsu thresholding, and dry-season compositing.

Usage in another script:
    from landsat_gbm import LandsatCollection, WaterExtractor, DrySeasonCompositor
"""

import ee

# ============================================================
# 1. LANDSAT COLLECTION BUILDER
# ============================================================

class LandsatCollection:
    """
    Builds a merged, cloud-masked, scaled Landsat 5 + 8 collection
    for a given WRS path/row and date range.

    Parameters
    ----------
    polygon    : ee.Geometry  — area of interest
    wrs_path   : int
    wrs_row    : int
    start_date : str  'YYYY-MM-DD'
    end_date   : str  'YYYY-MM-DD'
    cloud_cover_threshold : int  (image-level pre-filter, default 30)
    common_crs : str  EPSG string (default 'EPSG:32646')
    common_scale : int  metres (default 30)
    """

    # Band rename map → harmonised names used downstream
    L5_BANDS = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','ST_B6',
                'QA_PIXEL','QA_RADSAT']
    L8_BANDS = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10',
                'QA_PIXEL','QA_RADSAT']
    OUT_BANDS = ['Blue','Green','Red','NIR','SWIR1','SWIR2','TIR',
                 'QA_PIXEL','QA_RADSAT']

    def __init__(
        self,
        polygon,
        wrs_path: int,
        wrs_row: int,
        start_date: str,
        end_date: str,
        cloud_cover_threshold: int = 30,
        common_crs: str = 'EPSG:32646',
        common_scale: int = 30,
    ):
        self.polygon = polygon
        self.wrs_path = wrs_path
        self.wrs_row = wrs_row
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover_threshold = cloud_cover_threshold
        self.common_crs = common_crs
        self.common_scale = common_scale

        self.collection = self._build()

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _base_filter(self, collection_id: str) -> ee.ImageCollection:
        return (
            ee.ImageCollection(collection_id)
            .filter(ee.Filter.eq('WRS_PATH', self.wrs_path))
            .filter(ee.Filter.eq('WRS_ROW', self.wrs_row))
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.polygon)
            .filter(ee.Filter.lt('CLOUD_COVER', self.cloud_cover_threshold))
        )

    @staticmethod
    def _mask_clouds(image: ee.Image) -> ee.Image:
        """
        Pixel-level cloud/shadow/dilation mask via QA_PIXEL bitmask.
        Works for both L5 C02 and L8 C02 after band renaming.
        """
        qa = image.select('QA_PIXEL')
        mask = (
            qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
             .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud shadow
             .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud
        )
        sat = image.select('QA_RADSAT').eq(0)
        return (
            image.updateMask(mask)
                 .updateMask(sat)
                 .copyProperties(image, ['system:time_start'])
        )

    @staticmethod
    def _apply_scale_factors(image: ee.Image) -> ee.Image:
        """
        Convert C02 T1_L2 integer DN to surface reflectance [0,1].
        Runs AFTER band renaming, so selects harmonised names.
        Optical bands: Blue, Green, Red, NIR, SWIR1, SWIR2
        Thermal band : TIR
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

    def _rename_and_harmonise(
        self, col: ee.ImageCollection, src_bands: list
    ) -> ee.ImageCollection:
        return col.map(lambda img: img.select(src_bands, self.OUT_BANDS))

    def _align(self, image: ee.Image) -> ee.Image:
        return image.reproject(crs=self.common_crs, scale=self.common_scale)

    # ----------------------------------------------------------
    # Build
    # ----------------------------------------------------------

    def _build(self) -> ee.ImageCollection:
        # --- Landsat 5 ---
        l5 = self._base_filter('LANDSAT/LT05/C02/T1_L2')
        l5 = self._rename_and_harmonise(l5, self.L5_BANDS)   # rename first
        l5 = l5.map(self._apply_scale_factors)                # scale on harmonised names
        l5 = l5.map(self._mask_clouds)                        # mask last

        # --- Landsat 8 ---
        l8 = self._base_filter('LANDSAT/LC08/C02/T1_L2')
        l8 = self._rename_and_harmonise(l8, self.L8_BANDS)   # rename first
        l8 = l8.map(self._apply_scale_factors)                # scale on harmonised names
        l8 = l8.map(self._mask_clouds)                        # mask last

        merged = l5.merge(l8).map(self._align).sort('system:time_start')
        return merged

    # ----------------------------------------------------------
    # Public
    # ----------------------------------------------------------

    def size(self) -> int:
        return self.collection.size().getInfo()

    def get(self) -> ee.ImageCollection:
        return self.collection

    def __repr__(self):
        return (
            f"LandsatCollection(path={self.wrs_path}, row={self.wrs_row}, "
            f"{self.start_date} → {self.end_date})"
        )


# ============================================================
# 2. WATER INDEX CALCULATOR
# ============================================================

class WaterExtractor:
    """
    Adds water indices to each image in a collection and derives
    a land/water binary mask using an Otsu threshold on MNDWI.

    Parameters
    ----------
    collection : ee.ImageCollection  (harmonised band names expected)
    index      : str  'MNDWI' | 'NDWI'  (default 'MNDWI')
    """

    INDEX_BANDS = {
        'MNDWI': ('Green', 'SWIR1'),   # Modified NDWI — best for turbid delta water
        'NDWI':  ('Green', 'NIR'),     # Classic NDWI
    }

    def __init__(self, collection: ee.ImageCollection, index: str = 'MNDWI'):
        if index not in self.INDEX_BANDS:
            raise ValueError(f"index must be one of {list(self.INDEX_BANDS.keys())}")
        self.index = index
        self.collection = collection.map(self._add_indices)

    # ----------------------------------------------------------
    # Index computation
    # ----------------------------------------------------------

    def _add_indices(self, image: ee.Image) -> ee.Image:
        """Add MNDWI, NDWI, NDVI, and AWEInsh bands."""
        mndwi = image.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')
        ndwi  = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')
        ndvi  = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
        awei  = image.expression(
            '4*(G - S1) - (0.25*N + 2.75*S2)',
            {
                'G':  image.select('Green'),
                'S1': image.select('SWIR1'),
                'N':  image.select('NIR'),
                'S2': image.select('SWIR2'),
            }
        ).rename('AWEInsh')
        return image.addBands([mndwi, ndwi, ndvi, awei])

    # ----------------------------------------------------------
    # Otsu threshold
    # ----------------------------------------------------------

    @staticmethod
    def _otsu(histogram: ee.List) -> ee.Number:
        """
        Compute Otsu optimal threshold from a GEE fixedHistogram result.

        fixedHistogram returns a List of [midpoint, count] pairs:
            [[-1.0, 0], [-0.99, 5], ..., [0.99, 3]]

        ee.Array.slice uses keyword args: slice(axis, start, end)
        """
        arr    = ee.Array(histogram)                                    # shape [N, 2]
        means  = arr.slice(axis=1, start=0, end=1).project([0])        # shape [N]
        counts = arr.slice(axis=1, start=1, end=2).project([0])        # shape [N]

        size       = means.length().get([0])
        total      = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        total_mean = (means.multiply(counts)
                           .reduce(ee.Reducer.sum(), [0]).get([0])
                           .divide(total))

        def between_var(i):
            i  = ee.Number(i).toInt()
            w0 = (counts.slice(axis=0, start=0, end=i)
                        .reduce(ee.Reducer.sum(), [0]).get([0])
                        .divide(total))
            w1 = ee.Number(1).subtract(w0)
            m0 = (means.slice(axis=0, start=0, end=i)
                       .multiply(counts.slice(axis=0, start=0, end=i))
                       .reduce(ee.Reducer.sum(), [0]).get([0])
                       .divide(total.multiply(w0).add(1e-10)))
            m1 = total_mean.subtract(w0.multiply(m0)).divide(w1.add(1e-10))
            return w0.multiply(w1).multiply(m0.subtract(m1).pow(2))

        # Pair each between-class variance with its corresponding threshold mean.
        # Sort descending by variance and take the top threshold — avoids argmax()
        # indexing issues entirely.
        bv_list    = ee.List.sequence(1, size.subtract(1)).map(between_var)
        means_sub  = means.slice(axis=0, start=1)          # means at split points 1..N-1
        means_list = means_sub.toList()

        # Zip [variance, mean] pairs, sort by variance descending, return best mean
        pairs = bv_list.zip(means_list)                    # [[bv0, m0], [bv1, m1], ...]
        sorted_pairs = pairs.sort(
            pairs.map(lambda p: ee.Number(ee.List(p).get(0)))
        )
        best_pair = ee.List(sorted_pairs.reverse().get(0))
        return ee.Number(best_pair.get(1))

    def threshold_otsu(
        self,
        image: ee.Image,
        region: ee.Geometry,
        histogram_scale: int = 300,
    ) -> dict:
        """
        Compute Otsu threshold for a single image.

        Returns dict with keys:
            threshold   : ee.Number
            water_mask  : ee.Image  (1=water, 0=land)
            index_image : ee.Image
        """
        band      = self.index
        index_img = image.select(band)

        histogram = index_img.reduceRegion(
            reducer   = ee.Reducer.fixedHistogram(-1, 1, 200),
            geometry  = region,
            scale     = histogram_scale,
            maxPixels = 1e10,
        ).get(band)

        threshold  = self._otsu(histogram)
        water_mask = index_img.gt(threshold).rename('water').toInt8()

        return {
            'threshold':   threshold,
            'water_mask':  water_mask,
            'index_image': index_img,
        }

    def threshold_fixed(self, image: ee.Image, value: float = 0.0) -> ee.Image:
        """
        Apply a fixed threshold to the index band.
        Returns binary water mask (1=water, 0=land).
        """
        return image.select(self.index).gt(value).rename('water').toInt8()

    # ----------------------------------------------------------
    # Apply to full collection
    # ----------------------------------------------------------

    def apply_fixed_threshold(self, value: float = 0.0) -> ee.ImageCollection:
        """Map a fixed threshold across the entire collection."""
        return self.collection.map(
            lambda img: img.addBands(self.threshold_fixed(img, value))
        )

    def get(self) -> ee.ImageCollection:
        return self.collection


# ============================================================
# 3. DRY SEASON COMPOSITOR
# ============================================================

class DrySeasonCompositor:
    """
    Creates annual dry-season median composites from a collection.
    Default dry season for GBD: November–March (avoids monsoon cloud cover).

    Parameters
    ----------
    collection        : ee.ImageCollection  (with indices already added)
    years             : list[int]
    dry_start_month   : int  (default 11 = November)
    dry_end_month     : int  (default 3  = March, next year)
    reducer           : str  'median' | 'mean' | 'min' | 'max'
    """

    def __init__(
        self,
        collection: ee.ImageCollection,
        years: list,
        dry_start_month: int = 11,
        dry_end_month: int = 3,
        reducer: str = 'median',
    ):
        self.collection       = collection
        self.years            = years
        self.dry_start_month  = dry_start_month
        self.dry_end_month    = dry_end_month
        self.reducer          = getattr(ee.Reducer, reducer)()
        self.composites       = self._build()

    def _make_composite(self, year: int) -> ee.Image:
        start = ee.Date.fromYMD(year, self.dry_start_month, 1)
        # If end month < start month the season spans new year
        end_year = year + 1 if self.dry_end_month < self.dry_start_month else year
        end = ee.Date.fromYMD(end_year, self.dry_end_month, 28)

        composite = (
            self.collection
            .filterDate(start, end)
            .reduce(self.reducer)
        )
        # Rename bands: remove reducer suffix added by GEE
        old_names = composite.bandNames()
        new_names = old_names.map(
            lambda b: ee.String(b).replace('_(median|mean|min|max)$', '', 'g')
        )
        return (
            composite.rename(new_names)
                     .set('year', year)
                     .set('system:time_start', start.millis())
        )

    def _build(self) -> ee.ImageCollection:
        images = [self._make_composite(y) for y in self.years]
        return ee.ImageCollection(images)

    def get(self) -> ee.ImageCollection:
        return self.composites

    def size(self) -> int:
        return self.composites.size().getInfo()

    def get_year(self, year: int) -> ee.Image:
        return self.composites.filter(ee.Filter.eq('year', year)).first()


# ============================================================
# 4. CONVENIENCE WRAPPER — one call to rule them all
# ============================================================

class GBMPipeline:
    """
    High-level wrapper that chains LandsatCollection → WaterExtractor
    → DrySeasonCompositor in one object.

    Example
    -------
    >>> import ee; ee.Initialize()
    >>> from landsat_gbm import GBMPipeline
    >>>
    >>> polygon = ee.Geometry.Rectangle([88.0, 21.5, 90.5, 23.5])
    >>> pipe = GBMPipeline(
    ...     polygon    = polygon,
    ...     wrs_path   = 137,
    ...     wrs_row    = 44,
    ...     start_date = '1996-01-01',
    ...     end_date   = '2019-01-01',
    ...     years      = list(range(1996, 2019)),
    ... )
    >>> composites = pipe.composites.get()
    >>> water_1996 = pipe.get_water_mask(year=1996)
    """

    def __init__(
        self,
        polygon: ee.Geometry,
        wrs_path: int,
        wrs_row: int,
        start_date: str,
        end_date: str,
        years: list,
        index: str = 'MNDWI',
        cloud_cover_threshold: int = 30,
        common_crs: str = 'EPSG:32646',
        common_scale: int = 30,
        dry_start_month: int = 11,
        dry_end_month: int = 3,
    ):
        print("⏳ Building Landsat collection (L5 + L8, no L7)...")
        self.raw = LandsatCollection(
            polygon=polygon,
            wrs_path=wrs_path,
            wrs_row=wrs_row,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_threshold=cloud_cover_threshold,
            common_crs=common_crs,
            common_scale=common_scale,
        )
        print(f"   ✓ {self.raw.size()} images after cloud masking")

        print(f"⏳ Computing {index} + other indices...")
        self.water = WaterExtractor(self.raw.get(), index=index)

        print("⏳ Building dry-season composites...")
        self.composites = DrySeasonCompositor(
            collection=self.water.get(),
            years=years,
            dry_start_month=dry_start_month,
            dry_end_month=dry_end_month,
        )
        print(f"   ✓ {self.composites.size()} annual composites ready")

        self._polygon = polygon
        self._index   = index
        self._extractor = self.water

    def get_water_mask(
        self,
        year: int,
        method: str = 'otsu',
        fixed_threshold: float = 0.0,
    ) -> ee.Image:
        """
        Return binary water mask for a given year.

        Parameters
        ----------
        year             : int
        method           : 'otsu' | 'fixed'
        fixed_threshold  : float  (used only when method='fixed')
        """
        image = self.composites.get_year(year)
        if method == 'otsu':
            result = self._extractor.threshold_otsu(image, self._polygon)
            print(f"   Otsu threshold for {year}: {result['threshold'].getInfo():.4f}")
            return result['water_mask']
        else:
            return self._extractor.threshold_fixed(image, fixed_threshold)

    def get_all_water_masks(self, method: str = 'otsu') -> list:
        """
        Return a list of (year, ee.Image) tuples — one binary water mask per year
        from annual dry-season composites. Each thresholded independently.
        """
        results = []
        for year in self.composites.years:
            mask = self.get_water_mask(year=year, method=method)
            mask = mask.set('year', year)
            results.append((year, mask))
        return results

    def get_scene_water_mask(self, image: ee.Image, method: str = 'otsu') -> ee.Image:
        """
        Apply Otsu or fixed threshold to a single individual scene image.
        Returns binary water mask (1=water, 0=land).
        """
        if method == 'otsu':
            result = self._extractor.threshold_otsu(image, self._polygon)
            return result['water_mask']
        else:
            return self._extractor.threshold_fixed(image)

    def get_all_scene_masks(self, method: str = 'otsu') -> list:
        """
        Return a list of (date_str, ee.Image) tuples — one binary water mask
        per individual Landsat scene in the full collection (~153 images).
        Each scene gets its own independent Otsu threshold.

        Returns
        -------
        list of (date_str, ee.Image) e.g. ('1996-11-14', ee.Image)
        """
        col        = self.water.get()
        n          = col.size().getInfo()
        image_list = col.toList(n)
        results    = []

        print(f"📋 Processing {n} individual scenes...")
        for i in range(n):
            image    = ee.Image(image_list.get(i))
            # Get acquisition date as string YYYY-MM-DD
            date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            mask     = self.get_scene_water_mask(image, method=method)
            mask     = mask.set('system:time_start', image.get('system:time_start'))                            .set('date', date_str)
            results.append((date_str, mask))
            print(f"   [{i+1}/{n}] {date_str} ✓")

        return results


# ============================================================
# 5. EXPORTER — save water masks to local disk
# ============================================================

class Exporter:
    """
    Downloads GEE ee.Image objects directly to local disk as GeoTIFF.
    Uses geemap.ee_export_image — no Google Drive needed.

    Requirements
    ------------
        pip install geemap

    Parameters
    ----------
    output_dir : str   path to local folder (created if missing)
    scale      : int   export resolution in metres (default 30)
    crs        : str   EPSG string (default 'EPSG:32646')
    """

    def __init__(
        self,
        output_dir: str,
        scale: int = 30,
        crs: str = 'EPSG:32646',
    ):
        import os
        self.output_dir = output_dir
        self.scale      = scale
        self.crs        = crs
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {os.path.abspath(output_dir)}")

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

    def save_image(
        self,
        image: ee.Image,
        filename: str,
        region: ee.Geometry,
    ) -> str:
        """
        Export a single ee.Image to a local GeoTIFF.

        Parameters
        ----------
        image    : ee.Image
        filename : str   e.g. 'water_mask_2005'  (.tif added if missing)
        region   : ee.Geometry

        Returns
        -------
        str : full path to the saved file
        """
        import os
        geemap = self._get_geemap()

        if not filename.endswith('.tif'):
            filename += '.tif'
        filepath = os.path.join(self.output_dir, filename)

        print(f"⬇️  Downloading → {filepath} ...")
        geemap.ee_export_image(
            image,
            filename      = filepath,
            scale         = self.scale,
            region        = region,
            crs           = self.crs,
            file_per_band = False,
        )
        print(f"   ✓ Saved: {filepath}")
        return filepath

    def save_collection(
        self,
        image_collection: ee.ImageCollection,
        region: ee.Geometry,
        prefix: str = 'water_mask',
        year_property: str = 'year',
    ) -> list:
        """
        Export every image in a collection to separate GeoTIFFs,
        labelled by a property value (default: 'year').

        Parameters
        ----------
        image_collection : ee.ImageCollection
        region           : ee.Geometry
        prefix           : str   filename prefix
        year_property    : str   image property used for file labelling

        Returns
        -------
        list : paths of all saved files
        """
        n          = image_collection.size().getInfo()
        image_list = image_collection.toList(n)
        saved      = []

        print(f"📦 Exporting {n} images to local disk...")
        for i in range(n):
            image = ee.Image(image_list.get(i))
            label = image.get(year_property).getInfo()
            path  = self.save_image(image, f"{prefix}_{label}", region)
            saved.append(path)

        print(f"\n✅ Done — {n} files saved to: {self.output_dir}")
        return saved


# ============================================================
# 6. UPDATED GBMPipeline — export methods added
# ============================================================
# Monkey-patch export capability onto GBMPipeline so it stays callable
# from other scripts without touching the class definition above.

def _export_single(self, year: int, output_dir: str, method: str = 'otsu',
                   scale: int = 30, crs: str = 'EPSG:32646') -> str:
    """
    Convenience: get water mask for one year and save to disk.

    Example
    -------
        pipe.export_year(2005, output_dir='./water_masks')
    """
    exporter = Exporter(output_dir=output_dir, scale=scale, crs=crs)
    mask = self.get_water_mask(year=year, method=method)
    return exporter.save_image(mask, f'water_mask_{year}', self._polygon)


def _export_all(self, output_dir: str, method: str = 'otsu',
                scale: int = 30, crs: str = 'EPSG:32646') -> list:
    """
    Export one binary water mask per year, each thresholded independently
    using that year's own Otsu threshold computed from that year's
    dry-season composite. Loops in Python so each year gets its own threshold.

    Example
    -------
        pipe.export_all(output_dir=r'./water_masks')
    """
    exporter = Exporter(output_dir=output_dir, scale=scale, crs=crs)
    saved    = []
    years    = self.composites.years
    n        = len(years)

    print(f"📦 Exporting {n} annual binary masks (one Otsu threshold per year)...")
    for i, year in enumerate(years, 1):
        print(f"── [{i}/{n}] Year {year} ──")
        mask = self.get_water_mask(year=year, method=method)
        mask = mask.set('year', year)
        path = exporter.save_image(mask, f'water_mask_{year}', self._polygon)
        saved.append(path)

    print(f"✅ All {n} masks saved to: {output_dir}")
    return saved


GBMPipeline.export_year       = _export_single
GBMPipeline.export_all        = _export_all


def _export_all_scenes(self, output_dir: str, method: str = 'otsu',
                       scale: int = 30, crs: str = 'EPSG:32646') -> list:
    """
    Export one binary water mask per individual Landsat scene (~153 images).
    Each scene gets its own independent Otsu threshold.
    Files are named: water_mask_YYYY-MM-dd.tif

    Example
    -------
        pipe.export_all_scenes(output_dir=path, method='otsu')
    """
    exporter = Exporter(output_dir=output_dir, scale=scale, crs=crs)
    scenes   = self.get_all_scene_masks(method=method)
    saved    = []
    n        = len(scenes)

    print(f"📦 Saving {n} scene binary masks to disk...")
    for i, (date_str, mask) in enumerate(scenes, 1):
        print(f"── [{i}/{n}] {date_str} ──")
        path = exporter.save_image(mask, f'water_mask_{date_str}', self._polygon)
        saved.append(path)

    print(f"✅ All {n} scene masks saved to: {output_dir}")
    return saved


GBMPipeline.export_all_scenes = _export_all_scenes


# ==========================================================