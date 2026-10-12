"""
landsat_rivermap_pipeline.py
============================
River Delineation Module for Landsat 5/8.

Key Features:
- Computes MNDWI and AWEI for robust water detection.
- Applies MMSI for multiscale edge enhancement.
- Uses adaptive thresholding (Sauvola, Otsu, or k-means).
- Filters objects using size, shape, and DEM context.
- Supports Landsat 5 and Landsat 8.

Dependencies:
- earthengine-api
- geemap (for visualization and local export)
- numpy
- matplotlib (optional, for visualization)

Installation:
    pip install earthengine-api geemap

Authentication:
    Run `ee.Authenticate()` in a Python environment before using this module.
    Initialize with `ee.Initialize()`.

Usage:
    python gee_river_delineation.py --start_date 2020-01-01 --end_date 2020-12-31 --output_dir ./output

"""

from __future__ import annotations

import os
import argparse
import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



# --- Constants ---
# Lower Meghna-Tentulia Region (GBM Delta) - Example ROI
LOWER_MEGHNA_TENTULIA = ee.Geometry.Polygon(
    [[[90.5, 22.0], [90.5, 23.0], [91.5, 23.0], [91.5, 22.0]]]
)

# Landsat band names
LANDSAT_8_BANDS = {
    'green': 'B3',
    'red': 'B4',
    'nir': 'B5',
    'swir1': 'B6',
    'swir2': 'B7'
}

LANDSAT_5_BANDS = {
    'green': 'B2',
    'red': 'B3',
    'nir': 'B4',
    'swir1': 'B5',
    'swir2': 'B7'
}


# --- Helper Functions ---
def mask_landsat8_sr(image):
    """Cloud mask for Landsat 8/9 TOA/SR using QA_PIXEL."""
    qa = image.select('QA_PIXEL').toInt()
    cloud_bit = ee.Number(2).pow(5)
    cloud_shadow_bit = ee.Number(2).pow(3)
    cloud_mask = qa.bitwiseAnd(cloud_bit).neq(0)
    cloud_shadow_mask = qa.bitwiseAnd(cloud_shadow_bit).neq(0)
    mask = cloud_mask.Not().And(cloud_shadow_mask.Not())
    return image.updateMask(mask)

def mask_landsat5_sr(image):
    """Cloud mask for Landsat 5 TOA using BQA."""
    qa = image.select('BQA')
    mask = qa.lt(16)
    return image.updateMask(mask)
#####

def add_nd_indices(image, bands):
    """Add NDWI, MNDWI, and AWEI to a Landsat image."""
    green = image.select(bands['green'])
    red = image.select(bands['red'])
    nir = image.select(bands['nir'])
    swir1 = image.select(bands['swir1'])
    
    # NDWI
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    # MNDWI
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')
    
    # AWEI
    awei = ee.Image(4).multiply(green.subtract(swir1)).subtract(nir.add(red)).rename('AWEI')
    
    return image.addBands([ndwi, mndwi, awei])
#####

def compute_mmsi_gee(image, scales=[1, 2, 4, 8]):
    """
    Compute Multiscale Singularity Index (MMSI) in GEE using a custom Laplacian kernel.
    """
    # Define the Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian_kernel = ee.Kernel.fixed(3, 3, [[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    mmsi = ee.Image(0)
    for sigma in scales:
        # Apply Gaussian blur
        blurred = image.convolve(ee.Kernel.gaussian(sigma, sigma))
        # Compute Laplacian using the kernel
        laplacian = blurred.convolve(laplacian_kernel)
        mmsi = mmsi.add(laplacian.abs())
    return mmsi.divide(len(scales))

#####
def adaptive_threshold_gee(image, method='sauvola', **kwargs):
    """Apply adaptive thresholding in GEE (server-side)."""
    if method == 'otsu':
        roi = kwargs.get('roi', image.geometry())
        # Simplified Otsu (GEE does not natively support Otsu)
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,   # orimage.geometry(),
            scale=30,
            maxPixels=1e9
        ).get('mean')
        return image.gt(ee.Number(mean).multiply(1.1))  # 10% above mean

    elif method == 'sauvola':
        # Sauvola thresholding (server-side)
        window_size = kwargs.get('window_size', 25)
        mean = image.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.square(window_size)
        )
        std_dev = image.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(),
            kernel=ee.Kernel.square(window_size)
        )
        # Use ee.Number for all constants
        k = ee.Number(0.2)
        one = ee.Number(1)
        threshold = mean.multiply(
            one.add(
                k.multiply(
                    std_dev.divide(mean).subtract(one)
                )
            )
        )
        return image.gt(threshold)

    else:
        raise ValueError(f"Unsupported thresholding method: {method}")

#####
def connected_components_gee(binary_image):
    """
    Apply Connected Component Analysis (CCA) in GEE (simplified).
    Note: GEE's connectedComponents does not support size/elongation filtering.
    For full CCA, export the binary mask and use skimage.measure.regionprops locally.
    """
    # Label connected components (no size filtering in GEE)
    labeled = binary_image.connectedComponents(ee.Kernel.square(1),2)
    return labeled.gte(1)  # Retain all labeled objects

####
def post_process_gee(binary_image, dem_image=None, kernel_size=2):
    """
    Apply morphological operations and DEM masking in GEE.
    
    Args:
        binary_image: Input binary ee.Image.
        dem_image: Optional DEM ee.Image for masking.
        kernel_size: Size of the morphological kernel.
    
    Returns:
        ee.Image: Post-processed binary mask.
    """
    # Morphological operations (dilation followed by erosion)
    kernel = ee.Kernel.square(kernel_size)
    dilated = binary_image.focal_median(kernel_size, kernel, 'square')
    eroded = dilated.focal_median(kernel_size, kernel, 'square')
    
    # Apply DEM mask if provided
    if dem_image:
        eroded = eroded.updateMask(dem_image.lte(5))  # Mask for elevation <= 5m
    
    return eroded


def process_landsat_collection(start_date, end_date, roi=LOWER_MEGHNA_TENTULIA, 
                               satellite='LANDSAT_8', threshold_method='otsu',
                               dem_image=None,
                               min_area=10, max_elongation=5.0):
    """
    Note: min_area / max_elongation object filters are applied locally
    (e.g. with gbm_delta.extraction.river_mask) after export — GEE's
    connectedComponents does not support size/shape filtering. They are
    accepted here so CLI args flow through unchanged.
    """
    
    # 1. Choose the collection, band map, AND the correct mask function here:
    if satellite == 'LANDSAT_8':
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
        bands = LANDSAT_8_BANDS
        mask_func = mask_landsat8_sr  # <-- Reference L8 function
    elif satellite == 'LANDSAT_5':
        collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA')
        bands = LANDSAT_5_BANDS
        mask_func = mask_landsat5_sr  # <-- Reference L5 function
    else:
        raise ValueError(f"Unknown satellite: {satellite}")
    
    filtered_collection = collection.filterDate(start_date, end_date).filterBounds(roi)
    
    def process_image(image):
        # 2. Use the client-side assigned function cleanly here:
        cloud_masked = mask_func(image)
        
        with_indices = add_nd_indices(cloud_masked, bands)
        
        mndwi = with_indices.select('MNDWI')
        awei = with_indices.select('AWEI')
        fused_index = mndwi.add(awei).divide(2)
        
        mmsi = compute_mmsi_gee(fused_index)
        
        # Keep the explicit ROI fix we discussed previously!
        binary_mask = adaptive_threshold_gee(mmsi, method=threshold_method, roi=roi)
        
        #cca_mask = connected_components_gee(binary_mask)
        #final_mask = post_process_gee(cca_mask, dem_image=dem_image)
        final_mask = post_process_gee(binary_mask, dem_image=dem_image)
        
        return final_mask.rename('river_mask').set(
            'system:time_start', image.get('system:time_start')
        )
    
    river_masks = filtered_collection.map(process_image)
    return river_masks



def export_to_drive(image, description, folder='river_masks', region=None, scale=30):
    """
    Export an ee.Image to Google Drive.
    
    Args:
        image: ee.Image to export.
        description: Name for the exported file.
        folder: Google Drive folder name.
        region: Region to export (ee.Geometry).
        scale: Export scale in meters.
    """
    if region is None:
        region = LOWER_MEGHNA_TENTULIA
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        maxPixels=1e13
    )
    task.start()
    print(f"Export task started: {description}")
    return task


# Inside your Modules/landsat_rivermap_pipeline.py file:
def export_to_local(ee_object, output):
    geemap.ee_to_geotiff(
        ee_object=ee_object,
        output=output,
        resolution=30,                     # Set explicit spatial resolution
        bbox=[90.5, 22.0, 91.5, 23.0]  # Force precise clipping boundaries
    )


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description='GEE River Delineation for Landsat 5/8'
    )
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--satellite', type=str, default='LANDSAT_8', 
                        choices=['LANDSAT_8', 'LANDSAT_5'], 
                        help='Satellite (default: LANDSAT_8)')
    parser.add_argument('--threshold', type=str, default='otsu',
                        choices=['otsu', 'sauvola', 'kmeans'],
                        help='Thresholding method (default: otsu)')
    parser.add_argument('--min_area', type=int, default=10, 
                        help='Minimum object area for CCA filtering (default: 10)')
    parser.add_argument('--max_elongation', type=float, default=5,
                        help='Maximum eccentricity for CCA filtering (default: 5)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Local output directory (default: ./output)')
    parser.add_argument('--export_drive', action='store_true',
                        help='Export results to Google Drive')
    parser.add_argument('--export_local', action='store_true',
                        help='Export results to local directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DEM (optional)
    dem_image = ee.Image('USGS/SRTMGL1_003')  # SRTM 30m DEM
    
    # Process Landsat collection
    river_masks = process_landsat_collection(
        start_date=args.start_date,
        end_date=args.end_date,
        satellite=args.satellite,
        threshold_method=args.threshold,
        min_area=args.min_area,
        max_elongation=args.max_elongation,
        dem_image=dem_image
    )
    
    # Export results
    if args.export_drive:
        # Export each image in the collection to Drive
        collection_list = river_masks.toList(river_masks.size())
        for i in range(collection_list.size().getInfo()):
            image = ee.Image(collection_list.get(i))
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-DD').getInfo()
            export_to_drive(
                image=image,
                description=f'river_mask_{date}',
                folder='river_masks'
            )
    
    if args.export_local:
        # Export the first image to local for testing
        first_image = river_masks.first()
        date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-DD').getInfo()
        filename = os.path.join(args.output_dir, f'river_mask_{date}.tif')
        export_to_local(first_image, filename)
    
    # Print summary
    print(f"Processed {river_masks.size().getInfo()} images.")
    if args.export_drive:
        print("Results exported to Google Drive (folder: river_masks).")
    if args.export_local:
        print(f"First image exported to: {filename}")


if __name__ == '__main__':
    main()