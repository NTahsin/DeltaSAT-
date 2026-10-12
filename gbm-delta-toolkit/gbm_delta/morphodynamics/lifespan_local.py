# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:57:06 2026

@author: tahsi001
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    _PLOT = True
except ImportError:
    _PLOT = False

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# State codes
LAND = 0
INTERTIDAL = 2
RIVER = 1
NODATA = 255

STATE_NAMES = {LAND: "Land", INTERTIDAL: "Intertidal", RIVER: "River"}
STATE_COLS = {LAND: "#C8B89A", INTERTIDAL: "#F28E2B", RIVER: "#4E79A7"}

# --- Utilities ---
def _savepdf(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

def _write_tif(data: np.ndarray, path: Path, profile: dict, nodata=None, dtype="float32"):
    p = profile.copy()
    p.update(dtype=dtype, count=1, nodata=nodata, compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(dtype), 1)

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0, centre - margin)), float(min(1, centre + margin))

def lifespan_from_pii(pii: float) -> float:
    if pii >= 1.0:
        return np.inf
    return 1.0 / (1.0 - pii)

# --- Load State Stack ---
def load_state_stack(input_dir: Path, years: List[int], glob_pattern: str = "intertidal_filtered_{year}.tif") -> Tuple[np.ndarray, List[int], dict]:
    arrays = []
    valid_years = []
    profile = None

    it = tqdm(years, desc="Loading GeoTIFFs") if _TQDM else years
    for year in it:
        p = input_dir / glob_pattern.format(year=year)
        if not p.exists():
            log.warning("Missing: %s — skipping year %d", p, year)
            continue
        with rasterio.open(p) as src:
            # Assume the GeoTIFF already contains classified states (0=Land, 1=Intertidal, 2=River)
            state = src.read(1).astype(np.uint8)
            nd = src.nodata
            if nd is not None:
                state[np.isclose(state, nd, atol=1e-5)] = NODATA
            if profile is None:
                profile = src.profile.copy()
            # Debug: Print unique values in the state array
            log.info(f"Year {year}: Unique values in state array: {np.unique(state)}")
        arrays.append(state)
        valid_years.append(year)

    if not arrays:
        raise FileNotFoundError(f"No GeoTIFFs found in {input_dir}.")

    stack = np.stack(arrays, axis=0)  # (T, H, W)
    log.info("State stack: %d years × %d × %d px", *stack.shape)
    return stack, valid_years, profile

# --- Annual Transition Matrices ---
def compute_annual_transitions(stack: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Compute annual transition matrices for each pair of consecutive years.
    """
    T = stack.shape[0]
    states = [LAND, INTERTIDAL, RIVER]
    annual_P = []
    year_pairs = []

    for t in range(T - 1):
        from_s = stack[t]
        to_s = stack[t + 1]
        valid = (from_s != NODATA) & (to_s != NODATA)
        N = np.zeros((3, 3), dtype=np.int64)

        for i, fs in enumerate(states):
            for j, ts in enumerate(states):
                N[i, j] += int(((from_s == fs) & (to_s == ts) & valid).sum())

        row_sums = N.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        P = N / row_sums.astype(float)

        annual_P.append(P)
        year_pairs.append((t, t + 1))

    return annual_P, year_pairs

# --- Annual P_II and Lifespan Maps ---
def compute_annual_pii_lifespan(stack: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Compute annual P_II (self-transition probability for Intertidal) and lifespan maps.
    P_II is calculated as the probability of a pixel remaining Intertidal from t-1 to t.
    Lifespan is derived from P_II: L = 1 / (1 - P_II).
    """
    T = stack.shape[0]
    annual_pii_maps = []
    annual_lifespan_maps = []
    valid_years = []

    for t in range(1, T):  # Start from t=1 to compute t-1 to t
        from_s = stack[t - 1]  # State at t-1
        to_s = stack[t]        # State at t
        valid = (from_s != NODATA) & (to_s != NODATA)

        # Count transitions from Intertidal at t-1 to Intertidal at t
        from_intertidal = (from_s == INTERTIDAL) & valid
        to_intertidal = (to_s == INTERTIDAL) & valid

        n_ii = ((from_intertidal) & (to_intertidal)).astype(np.int32)  # I→I transitions
        n_i = from_intertidal.astype(np.int32)  # Total Intertidal pixels at t-1

        with np.errstate(invalid="ignore", divide="ignore"):
            pii = np.where(n_i > 0, n_ii / n_i.astype(float), np.nan).astype(np.float32)

        # Compute lifespan from P_II
        lifespan = np.where(
            np.isfinite(pii) & (pii < 1.0),
            1.0 / (1.0 - pii),
            np.where(pii >= 1.0, 50.0, np.nan)
        )
        lifespan = np.clip(lifespan, 0, 50).astype(np.float32)

        annual_pii_maps.append(pii)
        annual_lifespan_maps.append(lifespan)
        valid_years.append(t)  # Year of the "to" state (t)

    return annual_pii_maps, annual_lifespan_maps, valid_years

# --- Main Orchestrator ---
def run_analysis(input_dir: Path, output_dir: Path, years: List[int]):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load state stack
    stack, valid_years, profile = load_state_stack(input_dir, years, glob_pattern="intertidal_filtered_{year}.tif")
    if len(valid_years) < 2:
        raise ValueError("At least 2 years of data are required for transition analysis.")

    # Compute annual transition matrices
    annual_P, year_pairs = compute_annual_transitions(stack)

    # Compute annual P_II and lifespan maps
    annual_pii_maps, annual_lifespan_maps, lifespan_years = compute_annual_pii_lifespan(stack)

    # Save annual transition matrices
    for idx, (P, (y1, y2)) in enumerate(zip(annual_P, year_pairs)):
        df_rows = []
        for i, fs in enumerate([LAND, INTERTIDAL, RIVER]):
            for j, ts in enumerate([LAND, INTERTIDAL, RIVER]):
                df_rows.append({
                    "from_state": STATE_NAMES[fs],
                    "to_state": STATE_NAMES[ts],
                    "prob": round(float(P[i, j]), 6),
                    "year_from": valid_years[y1],
                    "year_to": valid_years[y2],
                })
        df = pd.DataFrame(df_rows)
        df.to_csv(output_dir / f"transition_matrix_{valid_years[y1]}-{valid_years[y2]}.csv", index=False)

    # Save annual P_II and lifespan maps
    for idx, (pii_map, lifespan_map, year) in enumerate(zip(annual_pii_maps, annual_lifespan_maps, lifespan_years)):
        year_from = valid_years[year - 1]
        year_to = valid_years[year]
        _write_tif(pii_map, output_dir / f"pii_map_{year_from}-{year_to}.tif", profile, nodata=np.nan)
        _write_tif(lifespan_map, output_dir / f"lifespan_map_{year_from}-{year_to}.tif", profile, nodata=np.nan)

    # Save global stats
    global_stats = []
    for idx, (P, (y1, y2)) in enumerate(zip(annual_P, year_pairs)):
        pii_global = float(P[INTERTIDAL, INTERTIDAL])
        L_global = lifespan_from_pii(pii_global)
        global_stats.append({
            "year_from": valid_years[y1],
            "year_to": valid_years[y2],
            "P_II_global": round(pii_global, 6),
            "lifespan_mean_yr": round(L_global, 3),
        })

    stats_df = pd.DataFrame(global_stats)
    stats_df.to_csv(output_dir / "annual_lifespan_stats.csv", index=False)

    # Generate figures (if matplotlib is available)
    if _PLOT:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # Plot annual transition matrices
        for idx, (P, (y1, y2)) in enumerate(zip(annual_P, year_pairs)):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(P, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Transition probability")
            labels = [STATE_NAMES[s] for s in [LAND, INTERTIDAL, RIVER]]
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("To state", fontsize=10)
            ax.set_ylabel("From state", fontsize=10)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{P[i,j]:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if P[i,j] > 0.6 else "black")
            ax.set_title(f"Transition Matrix: {valid_years[y1]}-{valid_years[y2]}", fontsize=11)
            _savepdf(fig, fig_dir / f"transition_matrix_{valid_years[y1]}-{valid_years[y2]}.pdf")

        # Plot annual lifespan maps
        for idx, (lifespan_map, year) in enumerate(zip(annual_lifespan_maps, lifespan_years)):
            year_from = valid_years[year - 1]
            year_from = valid_years[year - 1]
            year_to   = valid_years[year]
            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = ax.imshow(lifespan_map, vmin=0, vmax=50, cmap="viridis",
                           interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Lifespan (yr)")
            ax.set_title(f"Intertidal lifespan: {year_from}-{year_to}",
                         fontsize=11)
            _savepdf(fig, fig_dir / f"lifespan_map_{year_from}-{year_to}.pdf")

    # Return the summary table
    return stats_df
