# -*- coding: utf-8 -*-
"""
channel_geometry.py

Annual channel geometry from RivaMap psi / orientation / width
rasters -- no water mask required.

For each year:
    1. All psi images of the year are stacked per pixel.
    2. Per pixel, the MAXIMUM psi is kept, together with the
       orientation and width of the image that produced it.
       (Maximum psi is robust to date-level noise; a channel is
       present in a pixel if it was ever strongly singular there.)
    3. Directional NMS -> hysteresis -> centerlines.
    4. Centerlines are expanded by width to give the channel
       platform corridor.

Outputs per year:
    channel_platform_YYYY.tif  (uint8, 1 = channel platform)
    centerlines_YYYY.tif        (uint8, 1 = centerline)   [optional]
    nms_YYYY.tif               (float32)                 [optional]
"""

import re
from pathlib import Path
from collections import deque

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------
# Date / year extraction
# ---------------------------------------------------------------------

def extract_year(path):
    """Extract year (YYYY) from a filename, tolerant of any
    separator characters (hyphen, underscore, dash variants)."""
    name = Path(path).stem

    # Any non-digit separator (including unicode dashes) or nothing
    match = re.search(
        r"(?<!\d)(\d{4})\D{0,1}(\d{2})\D{0,1}(\d{2})(?!\d)",
        name,
    )
    if match:
        return match.group(1)

    # Fallback: first standalone 4-digit group (1996-2100 range)
    match = re.search(r"(?<!\d)(19[9]\d|20[0-2]\d)(?!\d)", name)
    if match:
        return match.group(1)

    return None


def group_files_by_year(folder, pattern="*.tif"):
    """Return {year: [paths]} for all GeoTIFFs in folder."""
    by_year = {}

    for path in sorted(Path(folder).glob(pattern)):
        year = extract_year(path)

        if year is None:
            print(f"Could not find year in: {path.name} -- skipped")
            continue

        by_year.setdefault(year, []).append(path)

    return by_year


# ---------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------

def read_aligned(path, ref_src, resampling=None):
    """Read a raster warped onto the reference grid."""
    if resampling is None:
        resampling = Resampling.bilinear

    with rasterio.open(path) as src:

        already_aligned = (
            src.crs == ref_src.crs
            and src.transform == ref_src.transform
            and src.width == ref_src.width
            and src.height == ref_src.height
        )

        if already_aligned:
            return src.read(1)

        print(
            f"  Aligning {Path(path).name} "
            f"({src.height}x{src.width}) to reference grid"
        )

        with WarpedVRT(
            src,
            crs=ref_src.crs,
            transform=ref_src.transform,
            width=ref_src.width,
            height=ref_src.height,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1)


def choose_reference(paths):
    """Reference grid = raster with the largest area."""
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
# NMS and hysteresis (same as channel_extraction.py)
# ---------------------------------------------------------------------

def directional_sample_nearest(image, row, col, dr, dc):
    rr = np.rint(row + dr).astype(np.int32)
    cc = np.rint(col + dc).astype(np.int32)

    valid = (
        (rr >= 0) & (rr < image.shape[0])
        & (cc >= 0) & (cc < image.shape[1])
    )

    result = np.full(row.shape, np.nan, dtype=np.float32)
    result[valid] = image[rr[valid], cc[valid]]

    return result


def nonmax_suppression(psi, orient, valid, angle_units="degrees"):
    """RivaMap-style directional NMS along the orientation."""
    psi = np.asarray(psi, dtype=np.float32)
    orient = np.asarray(orient, dtype=np.float32)

    if angle_units == "degrees":
        theta = np.deg2rad(orient)
    elif angle_units == "radians":
        theta = orient
    else:
        raise ValueError("angle_units must be 'degrees' or 'radians'")

    rows, cols = np.indices(psi.shape)

    dr = np.cos(theta)
    dc = np.sin(theta)

    p1 = directional_sample_nearest(psi, rows, cols, dr, dc)
    p2 = directional_sample_nearest(psi, rows, cols, -dr, -dc)

    nms = np.zeros_like(psi, dtype=np.float32)

    keep = (
        valid
        & np.isfinite(psi)
        & np.isfinite(p1)
        & np.isfinite(p2)
        & (psi >= p1)
        & (psi >= p2)
        & (psi > 0)
    )

    nms[keep] = psi[keep]

    return nms


def hysteresis_threshold(
    image,
    low_threshold,
    high_threshold,
    valid,
    connectivity=8,
):
    """Continuity-preserving hysteresis thresholding."""
    if high_threshold < low_threshold:
        raise ValueError("high_threshold must be >= low_threshold")

    weak = valid & np.isfinite(image) & (image >= low_threshold)
    strong = valid & np.isfinite(image) & (image >= high_threshold)

    result = np.zeros(image.shape, dtype=bool)

    if connectivity == 4:
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbours = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    queue = deque()

    strong_rows, strong_cols = np.where(strong)

    for r, c in zip(strong_rows, strong_cols):
        result[r, c] = True
        queue.append((r, c))

    rows, cols = image.shape

    while queue:
        r, c = queue.popleft()

        for dr, dc in neighbours:
            rr = r + dr
            cc = c + dc

            if rr < 0 or rr >= rows:
                continue
            if cc < 0 or cc >= cols:
                continue
            if not weak[rr, cc]:
                continue
            if result[rr, cc]:
                continue

            result[rr, cc] = True
            queue.append((rr, cc))

    return result


# ---------------------------------------------------------------------
# Annual compositing
# ---------------------------------------------------------------------

def build_annual_max(pairs, ref_src):
    """
    Per-pixel annual maximum composite.

    pairs : list of (psi_path, orient_path, width_path)

    Returns (psi_max, orient_at_max, width_at_max).
    The orientation and width are taken from the image that
    produced the maximum psi at each pixel.
    """
    shape = (ref_src.height, ref_src.width)

    psi_max = np.full(shape, -np.inf, dtype=np.float32)
    orient_at_max = np.full(shape, np.nan, dtype=np.float32)
    width_at_max = np.full(shape, np.nan, dtype=np.float32)

    for psi_path, orient_path, width_path in pairs:

        psi = read_aligned(psi_path, ref_src, Resampling.bilinear)
        orient = read_aligned(orient_path, ref_src, Resampling.nearest)
        width = read_aligned(width_path, ref_src, Resampling.bilinear)

        # Replace nodata with -inf so it never wins the max
        psi_finite = np.where(np.isfinite(psi), psi, -np.inf)

        better = psi_finite > psi_max

        psi_max = np.where(better, psi_finite, psi_max)
        orient_at_max = np.where(better, orient, orient_at_max)
        width_at_max = np.where(better, width, width_at_max)

    psi_max[psi_max == -np.inf] = np.nan

    return psi_max, orient_at_max, width_at_max


# ---------------------------------------------------------------------
# Width expansion
# ---------------------------------------------------------------------

def expand_centerlines(centerlines, width_px, width_factor=0.5,
                        min_width_pixels=1.0, max_width_pixels=None):
    """Expand centerlines into the channel platform corridor."""
    radius = np.asarray(width_px, dtype=np.float32) * width_factor

    radius = np.where(np.isfinite(radius), radius, 0)
    radius = np.maximum(radius, min_width_pixels)

    if max_width_pixels is not None:
        radius = np.minimum(radius, max_width_pixels)

    distance = distance_transform_edt(~centerlines)

    corridor = distance <= radius
    corridor |= centerlines

    return corridor


# ---------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------

def process_year(pairs, output_dir, year,
                 psi_low=0.012,
                 psi_high=0.11,
                 angle_units="degrees",
                 width_units="pixels",
                 width_factor=0.5,
                 min_width_pixels=1.0,
                 max_width_pixels=None,
                 connectivity=8,
                 save_intermediate=False):
    """
    Build the annual channel platform for one year.

    pairs : list of (psi_path, orient_path, width_path)
    """

    pairs = [(Path(p), Path(o), Path(w)) for p, o, w in pairs]
    output_dir = Path(output_dir)

    ref_path = choose_reference([p for p, o, w in pairs])

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()

        psi, orient, width = build_annual_max(pairs, ref)

    height, width_px_shape = psi.shape

    valid = np.isfinite(psi) & np.isfinite(orient)

    nms = nonmax_suppression(
        psi, orient, valid, angle_units=angle_units
    )

    centerlines = hysteresis_threshold(
        nms,
        low_threshold=psi_low,
        high_threshold=psi_high,
        valid=valid,
        connectivity=connectivity,
    )

    # Convert width to pixels if given in map units
    if width_units == "map":
        transform = profile["transform"]
        pixel_size = (abs(transform.a) + abs(transform.e)) / 2.0
        width_use = np.asarray(width) / pixel_size
    else:
        width_use = width

    corridor = expand_centerlines(
        centerlines,
        width_use,
        width_factor=width_factor,
        min_width_pixels=min_width_pixels,
        max_width_pixels=max_width_pixels,
    )

    # -------------------------------------------------------------
    # Write platform
    # -------------------------------------------------------------

    platform_profile = profile.copy()
    platform_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        predictor=2,
        nodata=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    platform_path = output_dir / f"channel_platform_{year}.tif"

    with rasterio.open(platform_path, "w", **platform_profile) as dst:
        dst.write(corridor.astype(np.uint8), 1)
        dst.set_band_description(1, "Channel platform")

    # -------------------------------------------------------------
    # Optional intermediates
    # -------------------------------------------------------------

    if save_intermediate:

        cl_path = output_dir / f"centerlines_{year}.tif"

        with rasterio.open(cl_path, "w", **platform_profile) as dst:
            dst.write(centerlines.astype(np.uint8), 1)
            dst.set_band_description(1, "Channel centerlines")

        nms_profile = profile.copy()
        nms_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            compress="deflate",
            nodata=0,
        )

        nms_path = output_dir / f"nms_{year}.tif"

        with rasterio.open(nms_path, "w", **nms_profile) as dst:
            dst.write(nms.astype(np.float32), 1)
            dst.set_band_description(
                1, "Annual-max directional NMS response"
            )

    return {
        "platform": str(platform_path),
        "n_images": len(pairs),
        "pixels": {
            "platform": int(np.sum(corridor)),
            "centerline": int(np.sum(centerlines)),
        },
    }


# ---------------------------------------------------------------------
# Directory-level driver (used by the runner)
# ---------------------------------------------------------------------

def process_directory(
    psi_dir,
    orient_dir,
    width_dir,
    output_dir,
    pattern="*.tif",
    psi_low=0.012,
    psi_high=0.11,
    angle_units="degrees",
    width_units="pixels",
    width_factor=0.5,
    min_width_pixels=1.0,
    max_width_pixels=None,
    connectivity=8,
    save_intermediate=False,
    overwrite=False,
):
    """
    Process all years found across the psi / orient / width folders.

    Files are matched per year (not per date): every psi image of a
    year contributes to that year's maximum composite.
    """

    psi_by_year = group_files_by_year(psi_dir, pattern)
    orient_by_year = group_files_by_year(orient_dir, pattern)
    width_by_year = group_files_by_year(width_dir, pattern)

    print("\n================================================")
    print("YEARS FOUND")
    print("================================================")
    print(f"Psi:    {sorted(psi_by_year)}")
    print(f"Orient: {sorted(orient_by_year)}")
    print(f"Width:  {sorted(width_by_year)}")

    common_years = (
        set(psi_by_year)
        & set(orient_by_year)
        & set(width_by_year)
    )

    print(f"\nComplete years: {sorted(common_years)}")

    results = {}

    for year in sorted(common_years):

        # Match orient/width counts to psi images per year by
        # position in the sorted lists (they are date-parallel)
        psi_paths = psi_by_year[year]
        orient_paths = orient_by_year[year]
        width_paths = width_by_year[year]

        n = min(len(psi_paths), len(orient_paths), len(width_paths))

        if n < len(psi_paths):
            print(
                f"WARNING: year {year} has unequal file counts "
                f"(psi={len(psi_paths)}, orient={len(orient_paths)}, "
                f"width={len(width_paths)}); using first {n}."
            )

        pairs = list(
            zip(psi_paths[:n], orient_paths[:n], width_paths[:n])
        )

        platform_path = Path(output_dir) / f"channel_platform_{year}.tif"

        print("\n--------------------------------------------")
        print(f"Year: {year}  ({n} images)")
        print("--------------------------------------------")

        if platform_path.exists() and not overwrite:
            print("Output already exists -- skipping.")
            continue

        try:
            stats = process_year(
                pairs,
                output_dir,
                year,
                psi_low=psi_low,
                psi_high=psi_high,
                angle_units=angle_units,
                width_units=width_units,
                width_factor=width_factor,
                min_width_pixels=min_width_pixels,
                max_width_pixels=max_width_pixels,
                connectivity=connectivity,
                save_intermediate=save_intermediate,
            )
            results[year] = stats

            print(
                f"SUCCESS: platform pixels = "
                f"{stats['pixels']['platform']}, "
                f"centerline pixels = "
                f"{stats['pixels']['centerline']}"
            )

        except Exception as e:
            print(f"ERROR processing {year}:")
            print(e)

    print("\n================================================")
    print("DONE")
    print("================================================")

    return results