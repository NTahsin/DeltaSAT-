# -*- coding: utf-8 -*-
"""
intertidal_sediment_mass.py
===========================
Year-on-year intertidal sediment mass captured  (GBM delta, JGR Earth Surface).

    M_sed,y→y+1  =  rho * dA * sum_(i,j in stable mask) [z_{y+1} - z_y]

Default rho = grain density 2650 kg/m^3 (GBM quartz/feldspar; Ando et al. 2019).
Set density_mode='bulk' with porosity to use rho_d = rho_s * (1 - phi_p).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio

log = logging.getLogger("sediment_mass")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None and not np.isnan(nd):
            a[a == nd] = np.nan
        return a, src.profile.copy()


def _write(arr: np.ndarray, path: Path, profile: dict) -> None:
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.float32(np.nan),
             compress="deflate", predictor=2, tiled=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)


def _pixel_area_m2(profile: dict) -> float:
    t = profile["transform"]
    return abs(t.a * t.e)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SedimentMassConfig:
    output_dir: Path                       # where dz_*.tif + summary CSV are written
    elev_dir: Optional[Path] = None        # folder with intertidal_h_{year}.tif
    occ_dir: Optional[Path] = None         # folder with occurrence_dry_{year}.tif
    elev_glob: str = "intertidal_h_{year}.tif"
    occ_glob: str = "occurrence_dry_{year}.tif"
    years: Optional[List[int]] = None
    rho_s: float = 2650.0
    porosity: Optional[float] = None
    density_mode: str = "grain"
    occ_low: float = 0.0
    occ_high: float = 1.0
    sigma_z: float = 0.25
    out_subdir: str = "sediment_mass"
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
class SedimentMassCalculator:
    """Year-on-year sediment-mass budget from intertidal elevation rasters."""

    def __init__(self, cfg: SedimentMassConfig):
        self.cfg = cfg
        self.cfg.output_dir = Path(self.cfg.output_dir)
        self.cfg.elev_dir = Path(self.cfg.elev_dir) if self.cfg.elev_dir else self.cfg.output_dir
        self.cfg.occ_dir = Path(self.cfg.occ_dir) if self.cfg.occ_dir else self.cfg.elev_dir
        self.out = self.cfg.output_dir / self.cfg.out_subdir
        self.out.mkdir(parents=True, exist_ok=True)

        if cfg.density_mode == "bulk":
            if cfg.porosity is None:
                raise ValueError("density_mode='bulk' requires `porosity`.")
            self.rho = cfg.rho_s * (1.0 - cfg.porosity)
            log.info("Using BULK dry density rho_d = %.0f kg/m^3 "
                     "(rho_s=%.0f, phi_p=%.2f)",
                     self.rho, cfg.rho_s, cfg.porosity)
        else:
            self.rho = cfg.rho_s
            log.info("Using GRAIN density rho_s = %.0f kg/m^3", self.rho)

    # -------------------------------------------------------------------
    def _discover_years(self) -> List[int]:
        pat = re.escape(self.cfg.elev_glob).replace(re.escape("{year}"), r"(\d+)")
        rx = re.compile(pat + r"$")

        ys: List[int] = []
        for p in self.cfg.elev_dir.iterdir():
            if p.is_file():
                m = rx.match(p.name)
                if m:
                    try:
                        ys.append(int(m.group(1)))
                    except ValueError:
                        pass
        if ys:
            return sorted(set(ys))

        for d in self.cfg.elev_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                y = int(d.name)
                if (d / self.cfg.elev_glob.format(year=y)).exists():
                    ys.append(y)
        return sorted(set(ys))

    # -------------------------------------------------------------------
    def _paths(self, year: int) -> Tuple[Path, Path]:
        z_flat = self.cfg.elev_dir / self.cfg.elev_glob.format(year=year)
        o_flat = self.cfg.occ_dir / self.cfg.occ_glob.format(year=year)
        if z_flat.exists():
            return z_flat, o_flat
        ydir = self.cfg.elev_dir / f"{year}"
        return (ydir / self.cfg.elev_glob.format(year=year),
                ydir / self.cfg.occ_glob.format(year=year))

    # -------------------------------------------------------------------
    def _process_pair(self, y0: int, y1: int) -> Optional[Dict[str, float]]:
        z0_p, o0_p = self._paths(y0)
        z1_p, o1_p = self._paths(y1)

        for p in (z0_p, z1_p):
            if not p.exists():
                log.warning("%d->%d: missing %s -- skipped.", y0, y1, p.name)
                return None

        z0, prof = _read(z0_p)
        z1, _ = _read(z1_p)

        if z0.shape != z1.shape:
            log.error("%d->%d: grid mismatch %s vs %s -- skipped.",
                      y0, y1, z0.shape, z1.shape)
            return None

        m0 = np.isfinite(z0)
        m1 = np.isfinite(z1)
        if o0_p.exists() and o1_p.exists():
            o0, _ = _read(o0_p)
            o1, _ = _read(o1_p)
            m0 &= (o0 > self.cfg.occ_low) & (o0 < self.cfg.occ_high)
            m1 &= (o1 > self.cfg.occ_low) & (o1 < self.cfg.occ_high)
        mask = m0 & m1
        N = int(mask.sum())
        if N == 0:
            log.warning("%d->%d: empty stable mask -- skipped.", y0, y1)
            return None

        dz = np.full(z0.shape, np.nan, dtype=np.float32)
        dz[mask] = z1[mask] - z0[mask]

        out_tif = self.out / f"dz_{y0}_{y1}.tif"
        if (not out_tif.exists()) or self.cfg.overwrite:
            _write(dz, out_tif, prof)
            log.info("%d->%d: wrote %s", y0, y1, out_tif.name)

        px = _pixel_area_m2(prof)
        dV = float(np.nansum(dz[mask]) * px)
        sigmaV = px * self.cfg.sigma_z * np.sqrt(2.0 * N)
        M_kg = self.rho * dV
        sigmaM = self.rho * sigmaV

        dz_vals = dz[mask]
        A_km2 = N * px / 1e6
        dt = float(y1 - y0)
        r_acc = float(np.nanmean(dz_vals)) / dt

        return dict(
            year_start=int(y0),
            year_end=int(y1),
            dt_yr=dt,
            n_pixels=N,
            area_km2=A_km2,
            mean_dz_m=float(np.nanmean(dz_vals)),
            median_dz_m=float(np.nanmedian(dz_vals)),
            std_dz_m=float(np.nanstd(dz_vals)),
            dV_m3=dV,
            sigma_dV_m3=sigmaV,
            M_sed_kg=M_kg,
            M_sed_Mt=M_kg / 1e9,
            sigma_M_Mt=sigmaM / 1e9,
            r_acc_mm_yr=r_acc * 1e3,
            rho_used=self.rho,
            density_mode=self.cfg.density_mode,
        )

    # -------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        years = self.cfg.years or self._discover_years()
        if len(years) < 2:
            log.error("Need >= 2 years; found %s", years)
            return pd.DataFrame()

        rows = []
        for y0, y1 in zip(years[:-1], years[1:]):
            r = self._process_pair(y0, y1)
            if r is not None:
                rows.append(r)

        df = pd.DataFrame(rows)
        if not df.empty:
            csv = self.out / "sediment_mass_summary.csv"
            df.to_csv(csv, index=False, float_format="%.4f")
            log.info("Summary -> %s", csv)
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(
        description="Annual intertidal sediment-mass budget (GBM delta).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--rho-s", type=float, default=2650.0)
    p.add_argument("--porosity", type=float, default=None)
    p.add_argument("--density-mode", choices=["grain", "bulk"], default="grain")
    p.add_argument("--sigma-z", type=float, default=0.25)
    p.add_argument("--elev-glob", default="intertidal_h_{year}.tif")
    p.add_argument("--occ-glob", default="occurrence_dry_{year}.tif")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    cfg = SedimentMassConfig(
        output_dir=Path(a.output_dir),
        elev_glob=a.elev_glob,
        occ_glob=a.occ_glob,
        years=a.years,
        rho_s=a.rho_s,
        porosity=a.porosity,
        density_mode=a.density_mode,
        sigma_z=a.sigma_z,
        overwrite=a.overwrite,
    )
    df = SedimentMassCalculator(cfg).run()
    if not df.empty:
        print("\n" + df[["year_start", "year_end", "area_km2",
                         "mean_dz_m", "M_sed_Mt", "sigma_M_Mt",
                         "r_acc_mm_yr"]].to_string(index=False))


if __name__ == "__main__":
    _cli()
