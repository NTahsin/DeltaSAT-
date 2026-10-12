# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:15:43 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
intertidal_filter.py

Remove intertidal pixels that are not hydrologically connected to
permanent water.

Input rasters (from interannual_water.py):
    0 = land, 1 = permanent water, 2 = intertidal

Rule:
    An intertidal pixel is KEPT only if it belongs to a connected
    component (8-connectivity, of permanent + intertidal pixels
    together) that contains at least one permanent-water pixel.
    Otherwise it becomes land (0).

    Permanent water pixels are always kept.

Output uses the same class encoding as the input.
"""

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


# Class encoding
LAND = 0
PERMANENT = 1
INTERTIDAL = 2


def filter_connected_intertidal(
    classified,
    connectivity=8,
    keep_unconnected_intertidal=False,
):
    """
    Filter intertidal pixels by connectivity to permanent water.

    Parameters
    ----------
    classified : 2D uint8 array (0 land, 1 perm, 2 intertidal)
    connectivity : 4 or 8
    keep_unconnected_intertidal : bool
        If True, isolated intertidal components are kept (no-op).
        Included so you can quantify the effect by running both.

    Returns
    -------
    filtered : 2D uint8 array (same encoding)
    n_removed : int (intertidal pixels turned into land)
    n_components_removed : int
    """

    if keep_unconnected_intertidal:
        return classified.copy(), 0, 0

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        structure = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    perm = classified == PERMANENT
    intertidal = classified == INTERTIDAL

    # Connected components of permanent + intertidal together.
    # Intertidal chains linking to permanent water belong to the
    # same component as that permanent water.
    wet_zone = perm | intertidal

    labels, n_components = ndimage.label(wet_zone, structure=structure)

    if n_components == 0:
        return classified.copy(), 0, 0

    # Components that contain at least one permanent-water pixel
    perm_labels = np.unique(labels[perm])

    # Label 0 is background -- drop it if present
    perm_labels = perm_labels[perm_labels != 0]

    # Boolean lookup: connected[i] True if component i touches
    # permanent water
    connected = np.zeros(n_components + 1, dtype=bool)
    connected[perm_labels] = True

    # Intertidal pixels in components WITHOUT permanent water
    isolated = intertidal & ~connected[labels]

    filtered = classified.copy()
    filtered[isolated] = LAND

    n_removed = int(np.sum(isolated))

    # Number of distinct components removed
    isolated_labels = np.unique(labels[isolated])
    n_components_removed = int(len(isolated_labels))

    return filtered, n_removed, n_components_removed


def process_file(
    input_path,
    output_path,
    connectivity=8,
    overwrite=False,
):
    """
    Filter one annual intertidal map.

    Returns stats dict.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print("Output already exists -- skipping.")
        return None

    with rasterio.open(input_path) as src:
        classified = src.read(1)
        profile = src.profile.copy()

    filtered, n_removed, n_comp_removed = filter_connected_intertidal(
        classified, connectivity=connectivity
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filtered, 1)
        dst.set_band_description(
            1, "Land / Permanent water / Intertidal (connected)"
        )
        try:
            dst.write_colormap(
                1,
                {
                    0: (210, 180, 140, 255),  # land - tan
                    1: (0, 90, 200, 255),     # permanent water - blue
                    2: (0, 200, 120, 255),    # intertidal - green
                },
            )
        except Exception:
            pass  # colormap optional; not all drivers accept it here

    return {
        "input": str(input_path),
        "output": str(output_path),
        "pixels": {
            "intertidal_before": int(np.sum(classified == INTERTIDAL)),
            "intertidal_after": int(np.sum(filtered == INTERTIDAL)),
            "intertidal_removed": n_removed,
            "permanent_water": int(np.sum(filtered == PERMANENT)),
            "land": int(np.sum(filtered == LAND)),
        },
        "components_removed": n_comp_removed,
    }


def process_directory(
    input_dir,
    output_dir,
    pattern="intertidal_*.tif",
    connectivity=8,
    overwrite=False,
):
    """
    Filter all annual intertidal maps in input_dir.

    Returns {year: stats}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(input_dir.glob(pattern))

    if not files:
        print(f"No files matching {pattern} in {input_dir}")
        return {}

    print("\n================================================")
    print("FILES FOUND")
    print("================================================")
    for f in files:
        print(f"  {f.name}")

    results = {}

    print("\n================================================")
    print("PROCESSING")
    print("================================================")

    for input_path in files:

        # intertidal_YYYY.tif -> YYYY
        year = input_path.stem.split("_")[-1]

        output_path = output_dir / f"intertidal_filtered_{year}.tif"

        print("\n--------------------------------------------")
        print(f"Year: {year}")
        print(f"Input : {input_path.name}")
        print(f"Output: {output_path.name}")
        print("--------------------------------------------")

        try:
            stats = process_file(
                input_path,
                output_path,
                connectivity=connectivity,
                overwrite=overwrite,
            )

            if stats is None:
                continue

            results[year] = stats

            px = stats["pixels"]
            removed_pct = (
                100.0 * px["intertidal_removed"]
                / max(px["intertidal_before"], 1)
            )
            print(
                f"SUCCESS: intertidal "
                f"{px['intertidal_before']} -> {px['intertidal_after']} "
                f"pixels ({removed_pct:.1f}% removed, "
                f"{stats['components_removed']} isolated components)"
            )

        except Exception as e:
            print(f"ERROR processing {input_path.name}:")
            print(e)

    print("\n================================================")
    print("DONE")
    print("================================================")

    return results