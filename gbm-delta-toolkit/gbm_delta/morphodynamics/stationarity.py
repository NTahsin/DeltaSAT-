# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:58:14 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
markov_stationarity.py
======================
Test the stationarity (time-homogeneity) assumption of the discrete-time
Markov chain used in the intertidal lifespan analysis.

Method (standard Markov-chain stationarity test, cf. Anderson & Goodman
1957):
    1. Split the record into B time blocks (early / mid / late).
    2. For each block b and state pair (i, j), count transitions N_b(i→j).
    3. Pooled estimate:  P̂(i→j) = Σ_b N_b(i→j) / Σ_b N_i·
    4. Under H0 (time-homogeneous), the statistic
           χ² = Σ_b Σ_i Σ_j  N_b(i→j) · ln( P_b(i→j) / P̂(i→j) )
       (likelihood-ratio form, df = (B−1) · Σ_i (k_i − 1))
       — or the classic Pearson form —
           χ² = Σ_b Σ_i Σ_j (N_b(i→j) − N_b(i·)·P̂(i→j))² /
                            (N_b(i·)·P̂(i→j))
       is χ²-distributed.
    5. Evaluated on (a) the full delta, (b) each ROI polygon, and
       (c) separately for the main-channel and side-channel polygons.

Blocks are defined on the *interval* mid-years so that irregular year
gaps (1999→2004 etc.) are handled correctly; each transition t→t+1 is
assigned to the block of its mid-year. By default blocks are cut at the
terciles of the mid-year distribution.

Inputs: the intertidal_filtered_{year}.tif rasters (0 land / 1 river /
2 intertidal), same convention as intertidal_lifespan_v2.py.
"""

#from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

log = logging.getLogger(__name__)

NODATA = 255
STATES = [0, 1, 2]                       # land, river, intertidal
STATE_NAMES = ["Land", "River", "Intertidal"]


# ─────────────────────────────────────────────────────────────────────────────
# state reading (same convention as intertidal_lifespan_v2)
# ─────────────────────────────────────────────────────────────────────────────
def read_state(path: Path) -> Optional[np.ndarray]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.uint8)
    arr[~np.isin(arr, STATES)] = NODATA
    return arr


def transition_counts_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """N(i→j) between two consecutive state rasters (3×3)."""
    valid = (a != NODATA) & (b != NODATA)
    N = np.zeros((3, 3), dtype=np.int64)
    for i, si in enumerate(STATES):
        fa = (a == si) & valid
        for j, sj in enumerate(STATES):
            N[i, j] = int((fa & (b == sj)).sum())
    return N


# ─────────────────────────────────────────────────────────────────────────────
# the test
# ─────────────────────────────────────────────────────────────────────────────
def block_assignment(years_pairs: Sequence[Tuple[int, int]],
                     n_blocks: int = 3) -> np.ndarray:
    """Assign each interval (y0, y1) to a block by mid-year terciles."""
    mid = np.array([(y0 + y1) / 2 for y0, y1 in years_pairs])
    cuts = np.quantile(mid, np.linspace(0, 1, n_blocks + 1)[1:-1])
    return np.digitize(mid, cuts)          # 0..n_blocks-1


def stationarity_test(counts_by_block: List[np.ndarray]
                      ) -> Dict[str, float]:
    """
    Anderson–Goodman test of time-homogeneous transition probabilities.

    counts_by_block : list (length B) of 3×3 count matrices N_b(i→j)
    """
    B = len(counts_by_block)
    stacked = np.stack(counts_by_block)               # (B, 3, 3)
    row_tot_b = stacked.sum(axis=2)                   # (B, 3)  N_b(i·)
    pooled = stacked.sum(axis=0)                      # (3, 3)
    row_tot_p = pooled.sum(axis=1)                    # (3,)

    with np.errstate(invalid="ignore", divide="ignore"):
        P_pool = pooled / row_tot_p[:, None]          # (3, 3)
        P_b = stacked / row_tot_b[:, :, None]          # (B, 3, 3)

    # degrees of freedom: (B-1) * Σ_i (k_i - 1), k_i = #reachable j from i
    df = 0
    for i in range(3):
        reachable = pooled[i] > 0
        if reachable.sum() > 1:
            df += (reachable.sum() - 1)
    df *= (B - 1)

    # ── Pearson statistic ────────────────────────────────────────────────────
    exp = row_tot_b[:, :, None] * P_pool[None, :, :]  # (B, 3, 3)
    mask = exp > 0
    chi2 = float((((stacked - exp)[mask] ** 2) / exp[mask]).sum())

    # ── likelihood-ratio (G²) statistic ──────────────────────────────────────
    m = mask & (stacked > 0)
    G2 = float((stacked[m] *
                np.log(stacked[m] / exp[m])).sum() * 2.0)

    p_chi2 = float(stats.chi2.sf(chi2, df)) if df > 0 else np.nan
    p_G2 = float(stats.chi2.sf(G2, df)) if df > 0 else np.nan
    CramersV = (float(np.sqrt(chi2 /
                    (stacked.sum() * (min(3, B) - 1))))
                if stacked.sum() > 0 else np.nan)

    return dict(chi2=chi2, G2=G2, df=df,
                p_chi2=p_chi2, p_G2=p_G2,
                n_transitions=int(stacked.sum()), CramersV=CramersV,
                B=B,
                verdict=("stationary (time-homogeneous) — H0 not rejected"
                         if p_chi2 > 0.05 else
                         "NON-stationary — transition probabilities vary "
                         "between time blocks"))


def transition_table(counts_by_block: List[np.ndarray]) -> pd.DataFrame:
    """Readable per-block transition probability table."""
    frames = []
    for b, N in enumerate(counts_by_block):
        P = N / N.sum(axis=1, keepdims=True)
        df = pd.DataFrame(P, index=[f"from_{s}" for s in STATE_NAMES],
                          columns=[f"to_{s}" for s in STATE_NAMES])
        df["block"] = b
        frames.append(df)
    return pd.concat(frames).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# ROI clipping
# ─────────────────────────────────────────────────────────────────────────────
def clip_to_roi(arr: np.ndarray, raster_path: Path, roi_geom) -> np.ndarray:
    """Clip a state raster to an ROI (geometry in the raster's CRS),
    cropped to the same window across years (aligned to first year)."""
    from rasterio.mask import mask
    from shapely.geometry import mapping
    with rasterio.open(raster_path) as src:
        try:
            clipped, _ = mask(src, [mapping(roi_geom)], crop=True,
                              filled=True, nodata=NODATA)
            return clipped[0].astype(np.uint8)
        except ValueError:
            return np.zeros((0, 0), dtype=np.uint8)


def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


# ─────────────────────────────────────────────────────────────────────────────
# orchestrator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StationarityConfig:
    raster_dir: Path
    years: Sequence[int]
    out_dir: Path
    n_blocks: int = 3
    glob_pattern: str = "intertidal_filtered_{year}.tif"


def run_stationarity(cfg: StationarityConfig,
                     rois: Optional["gpd.GeoDataFrame"] = None,
                     roi_groups: Optional[Dict[str, List[str]]] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test Markov stationarity for the full delta, each ROI, and named
    channel groups.

    roi_groups: e.g. {"Main channel": ["roi_a"], "Side channel": ["roi_b"]}
    """
    cfg.out_dir = Path(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for y in cfg.years:
        p = Path(cfg.raster_dir) / cfg.glob_pattern.format(year=y)
        if p.exists():
            files[y] = p
    yrs = sorted(files)
    log.info("Stationarity test on %d years: %s", len(yrs), yrs)

    intervals = list(zip(yrs[:-1], yrs[1:]))
    blocks = block_assignment(intervals, cfg.n_blocks)
    log.info("Block assignment: %s",
             {b: [f"{y0}-{y1}" for (y0, y1), bb in zip(intervals, blocks)
                  if bb == b] for b in range(cfg.n_blocks)})

    regions = [("Full delta", None)]
    if rois is not None:
        regions += [(n, g) for n, g in
                    zip(rois["name"], rois.geometry)]
        if roi_groups:
            for gname, members in roi_groups.items():
                sub = rois[rois["name"].isin(members)]
                # union the member geometries
                from shapely.ops import unary_union
                regions.append((f"Group: {gname}",
                                unary_union(list(sub.geometry))))

    results, tables = [], []
    for name, geom in regions:
        counts = [np.zeros((3, 3), dtype=np.int64)
                  for _ in range(cfg.n_blocks)]
        prev = None
        for (y0, y1), blk in zip(intervals, blocks):
            a = read_state(files[y0]) if geom is None \
                else clip_to_roi(read_state(files[y0]), files[y0], geom)
            b = read_state(files[y1]) if geom is None \
                else clip_to_roi(read_state(files[y1]), files[y1], geom)
            if a.size == 0 or b.size == 0:
                continue
            a, b = _align(a, b)
            counts[blk] += transition_counts_block(a, b)
        if sum(c.sum() for c in counts) == 0:
            log.warning("Region %s: no transitions", name)
            continue
        res = stationarity_test(counts)
        res["region"] = name
        results.append(res)
        t = transition_table(counts)
        t["region"] = name
        tables.append(t)
        log.info("%s: χ²=%.1f df=%d p=%.4g → %s",
                 name, res["chi2"], res["df"], res["p_chi2"],
                 res["verdict"])

    df_res = pd.DataFrame(results)
    df_tab = pd.concat(tables, ignore_index=True)
    df_res.to_csv(cfg.out_dir / "markov_stationarity_results.csv",
                  index=False)
    df_tab.to_csv(cfg.out_dir / "markov_stationarity_transitions.csv",
                  index=False)
    return df_res, df_tab


# optional geopandas import (only needed for ROIs)
try:
    import geopandas as gpd
except ImportError:
    gpd = None