from __future__ import annotations

import os
import ee
import geemap
from datetime import datetime


# --- Constants ---
# Custom ROI for the study area
CUSTOM_ROI = ee.Geometry.Polygon(
    [[[90.3102134, 23.2663197],
      [90.7690719, 23.2427192],
      [90.9689139, 22.5936647],
      [90.9309718, 22.5035358],
      [90.2752803, 22.5223133],
      [90.3102134, 23.2663197]]]
)

# Landsat band names for Level-2 (L2) data
LANDSAT_8_L2_BANDS = {
    'green': 'SR_B3',
    'red': 'SR_B4',
    'nir': 'SR_B5',
    'swir1': 'SR_B6',
    'swir2': 'SR_B7'
}

LANDSAT_5_L2_BANDS = {
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7'
}

# --- Helper Functions ---
def apply_scale_factors(image, satellite):
    """Applies scaling factors to Landsat L2 SR and ST bands."""
    if satellite == 'LANDSAT_8':
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B10').multiply(0.00341802).add(149.0)
    elif satellite == 'LANDSAT_5':
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B6').multiply(0.00341802).add(149.0)
    else:
        raise ValueError(f"Unknown satellite: {satellite}")

    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

def mask_landsat8_l2(image):
    """Cloud mask for Landsat 8 L2 using QA_PIXEL."""
    qa = image.select('QA_PIXEL').toInt()
    cloud_bit = ee.Number(2).pow(5).toInt()
    cloud_shadow_bit = ee.Number(2).pow(3).toInt()
    cloud_mask = qa.bitwiseAnd(cloud_bit).neq(0)
    cloud_shadow_mask = qa.bitwiseAnd(cloud_shadow_bit).neq(0)
    mask = cloud_mask.Not().And(cloud_shadow_mask.Not())
    return image.updateMask(mask)

def add_nd_indices(image, bands):
    """Add NDWI, MNDWI, and AWEI to a Landsat image."""
    green = image.select(bands['green']).toFloat()
    red = image.select(bands['red']).toFloat()
    nir = image.select(bands['nir']).toFloat()
    swir1 = image.select(bands['swir1']).toFloat()

    # NDWI
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')

    # MNDWI
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')

    # AWEI
    awei = ee.Image(4).multiply(green.subtract(swir1)).subtract(nir.add(red)).rename('AWEI')

    return image.addBands([ndwi, mndwi, awei])

def process_landsat_collection(start_date, end_date, roi=CUSTOM_ROI, satellite='LANDSAT_8', cloud_cover_threshold=30, valid_pixel_threshold=70):
    """
    Preprocesses Landsat L2 scenes and outputs an ImageCollection of fused water index images,
    filtered by cloud cover and valid pixel percentage.
    """
    if satellite == 'LANDSAT_8':
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        bands = LANDSAT_8_L2_BANDS
        mask_func = mask_landsat8_l2
    elif satellite == 'LANDSAT_5':
        collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
        bands = LANDSAT_5_L2_BANDS
        mask_func = None  # No QA band for Landsat 5 L2
    else:
        raise ValueError(f"Unknown satellite: {satellite}")

    # Filter by date, ROI, and cloud cover
    filtered_collection = (
        collection
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_threshold))
    )

    # Function to process each image
    def process_image(image):
        # Apply scaling factors
        scaled_image = apply_scale_factors(image, satellite)

        # Apply cloud mask (only for Landsat 8)
        if mask_func:
            cloud_masked = mask_func(scaled_image)
        else:
            cloud_masked = scaled_image  # Skip cloud masking for Landsat 5 L2

        # Add spectral indices
        with_indices = add_nd_indices(cloud_masked, bands)

        mndwi = with_indices.select('MNDWI')
        awei = with_indices.select('AWEI')

        # Fused Water Index
        fused_index = mndwi.add(awei).divide(2).rename('fused_water_index')

        return fused_index.set(
            'system:time_start', image.get('system:time_start'),
            'date', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        )

    # Apply processing to each image
    processed_collection = filtered_collection.map(process_image)

    return processed_collection