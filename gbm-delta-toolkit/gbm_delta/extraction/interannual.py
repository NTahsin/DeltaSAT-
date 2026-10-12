# -*- coding: utf-8 -*-
"""
interannual_water.py

Annual intertidal mapping from paired water classification GeoTIFFs
(binary water) and water index GeoTIFFs (e.g. NDWI/MNDWI).

The index raster is used purely as an observation mask: a pixel is
"observed" on a date if the index raster has valid data there
(nodata = cloud / cloud shadow / sensor gap).

For each year YYYY, every pixel is classified:

    water in all observed images      -> permanent water
    water in >= 1 but < all observed  -> intertidal
    water in 0 observed images        -> land
    never observed (always nodata)   -> land

Output classes:
    0 = land
    1 = permanent water
    2 = intertidal
"""

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


# ---------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------

def extract_date(path):
    """
    Extract date (YYYY-MM-DD) from a filename.

    Supports:
        binary_fused_index_1996-02-09_LT05_137044_19960209.tif
        water_1996_02_09.tif
        19960209.tif
    """
    name = Path(path).stem

    match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if match:
        return match.group(1)

    match = re.search(r"(\d{4})-_-_", name)
    if match:
        y, m, d = match.groups()
        return f"{y}-{m}-{d}"

    match = re.search(r"(?<!\d)(\d{8})(?!\d)", name)
    if match:
        d = match.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    return None


def group_pairs_by_year(water_dir, index_dir, pattern="*.tif"):
    """
    Match water and index rasters by date.

    Returns {year: [(water_path, index_path, date), ...]}
    and prints diagnostics for unmatched files.
    """
    water_files = {}
    index_files = {}

    for path in sorted(Path(water_dir).glob(pattern)):
        date = extract_date(path)
        if date is None:
            print(f"Could not find date in water file: {path.name}")
            continue
        water_files[date] = path

    for path in sorted(Path(index_dir).glob(pattern)):
        date = extract_date(path)
        if date is None:
            print(f"Could not find date in index file: {path.name}")
            continue
        index_files[date] = path

    by_year = {}
    unmatched = []

    for date in sorted(set(water_files) | set(index_files)):

        w = water_files.get(date)
        i = index_files.get(date)

        if w is not None and i is not None:
            by_year.setdefault(date[:4], []).append((w, i, date))
        else:
            missing = []
            if w is None:
                missing.append("water")
            if i is None:
                missing.append("index")
            unmatched.append((date, missing))

    print("\n================================================")
    print("FILE PAIRING")
    print("================================================")
    print(f"Paired dates : {sum(len(v) for v in by_year.values())}")
    print(f"Unpaired dates: {len(unmatched)}")

    for date, missing in unmatched:
        print(f"  {date}: missing {', '.join(missing)}")

    return by_year


# ---------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------

def read_aligned(path, ref_src, resampling=None):
    """
    Read a raster warped onto the reference grid.

    If it already matches the reference grid exactly, read as-is.
    Nearest-neighbour resampling is used for the categorical water
    raster; bilinear for the continuous index raster.
    """
    if resampling is None:
        resampling = Resampling.nearest

    with rasterio.open(path) as src:

        already_aligned = (
            src.crs == ref_src.crs
            and src.transform == ref_src.transform
            and src.width == ref_src.width
            and src.height == ref_src.height
        )

        if already_aligned:
            return src.read(1), src.nodata

        print(f"  Aligning {Path(path).name} to reference grid")

        with WarpedVRT(
            src,
            crs=ref_src.crs,
            transform=ref_src.transform,
            width=ref_src.width,
            height=ref_src.height,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1), src.nodata


def choose_reference(paths):
    """Reference grid = raster with the largest area, to cover the
    entire region captured by the GeoTIFFs."""
    best = None
    best_area = -1

    for path in paths:
        with rasterio.open(path) as src:
            area = src.width * src.height
            if area > best_area:
                best_area = area
                best = path

    return best


# ---------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------

def classify_year(pairs, ref_path=None, water_value=1):
    """
    pairs : list of (water_path, index_path, date)

    For each pixel:
        observed_count = number of dates where the INDEX raster
                         had valid (non-nodata) data
        water_count    = number of dates where the WATER raster
                         was water AND the pixel was observed

        water_count == observed_count > 0  -> permanent water
        0 < water_count < observed_count   -> intertidal
        water_count == 0                   -> land
        observed_count == 0                -> land (never seen)

    Returns (classified, profile)
    """

    pairs = [(Path(w), Path(i), d) for w, i, d in pairs]

    if ref_path is None:
        # Use the largest water raster as the reference grid
        ref_path = choose_reference([w for w, i, d in pairs])

    ref_path = Path(ref_path)

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()

    height = profile["height"]
    width = profile["width"]

    observed_count = np.zeros((height, width), dtype=np.uint16)
    water_count = np.zeros((height, width), dtype=np.uint16)

    with rasterio.open(ref_path) as ref:

        for water_path, index_path, date in pairs:

            water, _ = read_aligned(
                water_path, ref, resampling=Resampling.nearest
            )

            index, index_nodata = read_aligned(
                index_path, ref, resampling=Resampling.bilinear
            )

            # Observed = index raster has valid data at this pixel
            observed = np.isfinite(index)

            if index_nodata is not None:
                if np.isnan(index_nodata):
                    observed &= ~np.isnan(index)
                else:
                    observed &= index != index_nodata

            water = (water == water_value) & observed

            observed_count += observed.astype(np.uint16)
            water_count += water.astype(np.uint16)

    # -------------------------------------------------------------
    # Classify
    # -------------------------------------------------------------

    classified = np.zeros((height, width), dtype=np.uint8)

    observed = observed_count > 0
    permanent = observed & (water_count == observed_count)
    intertidal = observed & (water_count > 0) & (water_count < observed_count)

    classified[permanent] = 1
    classified[intertidal] = 2
    # land (incl. never observed) stays 0

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        predictor=2,
    )

    return classified, profile


def process_year(pairs, output_path, ref_path=None, water_value=1):
    """Classify one year and write the result."""
    output_path = Path(output_path)

    classified, profile = classify_year(
        pairs, ref_path=ref_path, water_value=water_value
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classified, 1)
        dst.set_band_description(1, "Land / Permanent water / Intertidal")
        dst.write_colormap(
            1,
            {
                0: (210, 180, 140, 255),  # land - tan
                1: (0, 90, 200, 255),     # permanent water - blue
                2: (0, 200, 120, 255),    # intertidal - green
            },
        )

    return {
        "output": str(output_path),
        "n_dates": len(pairs),
        "pixels": {
            "land": int(np.sum(classified == 0)),
            "permanent_water": int(np.sum(classified == 1)),
            "intertidal": int(np.sum(classified == 2)),
            "total": int(classified.size),
        },
    }


def process_directory(
    water_dir,
    index_dir,
    output_dir,
    pattern="*.tif",
    water_value=1,
    overwrite=False,
):
    """
    Process all years found in water_dir (paired with index_dir).

    Returns summary dict {year: stats}.
    """
    output_dir = Path(output_dir)

    by_year = group_pairs_by_year(water_dir, index_dir, pattern=pattern)

    print("\n================================================")
    print("YEARS FOUND (paired water + index)")
    print("================================================")

    for year in sorted(by_year):
        print(f"{year}: {len(by_year[year])} dates")

    results = {}

    print("\n================================================")
    print("PROCESSING")
    print("================================================")

    for year in sorted(by_year):

        pairs = by_year[year]

        output_path = output_dir / f"intertidal_{year}.tif"

        print("\n--------------------------------------------")
        print(f"Year: {year}  ({len(pairs)} dates)")
        print("--------------------------------------------")

        if output_path.exists() and not overwrite:
            print("Output already exists -- skipping.")
            continue

        try:
            stats = process_year(pairs, output_path, water_value=water_value)
            results[year] = stats

            px = stats["pixels"]
            print(
                f"SUCCESS: "
                f"land={px['land']}, permanent={px['permanent_water']}, "
                f"intertidal={px['intertidal']}"
            )

        except Exception as e:
            print(f"ERROR processing {year}:")
            print(e)

    print("\n================================================")
    print("DONE")
    print("================================================")

    return results