# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:19:05 2026

@author: tahsi001
"""

# -*- coding: utf-8 -*-
"""
intertidal_lifespan_v2.py

Residence-time ("lifespan") analysis of the Intertidal state from a series of
classified rasters.

Method
------
For each pair of consecutive available years (t0 -> t1, gap dt = t1 - t0):
  1. p_gap   = N(I->I) / N(I)                 pooled over all valid pixels
  2. p_annual = p_gap ** (1/dt)               annual equivalent (fixes 3-yr gaps)
  3. L        = 1 / (1 - p_annual)            expected residence time, years
Uncertainty:
  * Wilson CI (pixel-independent; too narrow because of spatial autocorrelation)
  * Spatial block-bootstrap CI (resamples blocks of pixels; more realistic)
Trend:
  * Done on logit(p_annual) against interval mid-year (NOT on L, which explodes
    as p -> 1). Theil-Sen, Mann-Kendall (tau-b) and OLS, all two-sided.
Maps:
  * Per-pixel P_II is only 0 or 1 (meaningless as a lifespan), so maps use
    P_II pooled over a moving window (default 15x15 px) with a minimum count.

Assumption: exit from Intertidal is memoryless (geometric). L is the expected
residence time of a pixel that is Intertidal, not the age of the feature.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from scipy.ndimage import uniform_filter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False

log = logging.getLogger(__name__)
NODATA = 255


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # !! VERIFY against your rasters (your old code was inconsistent:
    #    constants said Intertidal=2, a comment said Intertidal=1).
    codes: Dict[str, int] = None            # name -> raster value
    glob_pattern: str = "intertidal_filtered_{year}.tif"
    window_px: int = 15                      # window for lifespan maps
    min_window_n: int = 30                   # min Intertidal px in window
    lifespan_cap_yr: float = 100.0           # display cap for maps only
    block_px: int = 50                       # bootstrap block size (pixels)
    n_boot: int = 500
    seed: int = 42
    write_maps: bool = True

    def __post_init__(self):
        if self.codes is None:
            self.codes = {"Land": 0, "River": 1, "Intertidal": 2}

    @property
    def I(self) -> int:
        return self.codes["Intertidal"]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def annualise(p_gap, dt):
    """Annual equivalent of a persistence probability measured over dt years."""
    return np.power(p_gap, 1.0 / dt)


def lifespan(p_annual):
    """Expected residence time 1/(1-p); inf if p >= 1."""
    p = np.asarray(p_annual, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(p < 1.0, 1.0 / (1.0 - p), np.inf)
    return out if out.ndim else float(out)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(max(0, c - m)), float(min(1, c + m))


def _block_sum(mask: np.ndarray, b: int) -> np.ndarray:
    H, W = mask.shape
    ph, pw = (-H) % b, (-W) % b
    m = np.pad(mask, ((0, ph), (0, pw))) if (ph or pw) else mask
    Hb, Wb = m.shape[0] // b, m.shape[1] // b
    return m.reshape(Hb, b, Wb, b).sum(axis=(1, 3), dtype=np.int64).ravel()


def _window_sum(x: np.ndarray, w: int) -> np.ndarray:
    return uniform_filter(x.astype(np.float32), size=w, mode="constant") * (w * w)


def _write_tif(data, path: Path, profile: dict):
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="lzw",
             tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype("float32"), 1)


# ----------------------------------------------------------------------------
# Raster reading (one year at a time, no giant stack)
# ----------------------------------------------------------------------------
def read_state(path: Path, cfg: Config):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nd = src.nodata
        profile = src.profile.copy()
        meta = (src.shape, src.transform, src.crs)
    bad = ~np.isin(arr, list(cfg.codes.values()))
    if nd is not None:
        bad |= np.isnan(arr) if np.isnan(nd) else np.isclose(arr, nd)
    vals, cnts = np.unique(arr[~bad], return_counts=True)
    log.info("%s: values %s counts %s | invalid px: %d",
             path.name, vals.tolist(), cnts.tolist(), int(bad.sum()))
    if cfg.I not in vals:
        log.warning("%s: Intertidal code %d not present! Check Config.codes.",
                    path.name, cfg.I)
    state = np.where(bad, NODATA, arr).astype(np.uint8)
    return state, profile, meta


# ----------------------------------------------------------------------------
# Core analysis over intervals
# ----------------------------------------------------------------------------
def analyse_intervals(input_dir: Path, output_dir: Path, years: Sequence[int],
                      cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    files = {}
    for y in years:
        p = Path(input_dir) / cfg.glob_pattern.format(year=y)
        if p.exists():
            files[y] = p
        else:
            log.warning("Missing %s -> year %d skipped", p, y)
    yrs = sorted(files)
    if len(yrs) < 2:
        raise ValueError("Need >= 2 available years.")
    log.info("Years used: %s", yrs)

    names = list(cfg.codes)
    codes = [cfg.codes[n] for n in names]
    rows, mats = [], []
    acc_num = acc_den = None
    profile0 = None

    prev, profile0, meta0 = read_state(files[yrs[0]], cfg)
    for y0, y1 in zip(yrs[:-1], yrs[1:]):
        cur, _, meta = read_state(files[y1], cfg)
        if meta[0] != meta0[0] or meta[1] != meta0[1]:
            raise ValueError(f"Raster {y1} grid differs from {yrs[0]} (shape/transform).")
        dt = y1 - y0
        valid = (prev != NODATA) & (cur != NODATA)

        # full transition count matrix (kept for reference)
        N = np.zeros((3, 3), dtype=np.int64)
        for i, a in enumerate(codes):
            fa = (prev == a) & valid
            for j, b in enumerate(codes):
                N[i, j] = int((fa & (cur == b)).sum())
        pd.DataFrame(N, index=names, columns=names).to_csv(
            Path(output_dir) / f"transition_counts_{y0}-{y1}.csv")

        from_i = (prev == cfg.I) & valid
        stay = from_i & (cur == cfg.I)
        n_i, n_ii = int(from_i.sum()), int(stay.sum())
        if n_i == 0:
            log.warning("%d-%d: no Intertidal pixels; skipped.", y0, y1)
            prev = cur
            continue

        p_gap = n_ii / n_i
        p_ann = float(annualise(p_gap, dt))
        lo, hi = wilson_ci(n_ii, n_i)

        # spatial block bootstrap
        bi, bii = _block_sum(from_i, cfg.block_px), _block_sum(stay, cfg.block_px)
        keep = bi > 0
        bi, bii = bi[keep], bii[keep]
        idx = rng.integers(0, len(bi), size=(cfg.n_boot, len(bi)))
        pb = bii[idx].sum(1) / bi[idx].sum(1)
        blo, bhi = np.percentile(pb, [2.5, 97.5])

        rows.append(dict(
            year_from=y0, year_to=y1, gap_yr=dt, n_I=n_i, n_II=n_ii,
            P_II_gap=p_gap, P_II_annual=p_ann,
            P_annual_wilson_lo=float(annualise(lo, dt)),
            P_annual_wilson_hi=float(annualise(hi, dt)),
            P_annual_boot_lo=float(annualise(blo, dt)),
            P_annual_boot_hi=float(annualise(bhi, dt)),
            lifespan_yr=float(lifespan(p_ann)),
            lifespan_boot_lo=float(lifespan(annualise(blo, dt))),
            lifespan_boot_hi=float(lifespan(annualise(bhi, dt))),
            mid_year=(y0 + y1) / 2.0,
        ))
        log.info("%d-%d dt=%d  P_gap=%.4f P_ann=%.4f L=%.1f yr",
                 y0, y1, dt, p_gap, p_ann, lifespan(p_ann))

        # windowed maps
        if cfg.write_maps:
            w = cfg.window_px
            Wi, Wii = _window_sum(from_i, w), _window_sum(stay, w)
            ok = Wi >= cfg.min_window_n
            with np.errstate(invalid="ignore", divide="ignore"):
                pw_ann = np.where(ok, annualise(np.clip(Wii / Wi, 0, 1), dt), np.nan)
            L_map = np.where(np.isfinite(pw_ann),
                             np.minimum(lifespan(np.nan_to_num(pw_ann)), cfg.lifespan_cap_yr),
                             np.nan)
            _write_tif(pw_ann, Path(output_dir) / f"pii_annual_win_{y0}-{y1}.tif", profile0)
            _write_tif(L_map, Path(output_dir) / f"lifespan_win_{y0}-{y1}.tif", profile0)
            if acc_num is None:
                acc_num = np.zeros_like(Wi, dtype=np.float64)
                acc_den = np.zeros_like(Wi, dtype=np.float64)
            acc_num += np.where(ok, np.nan_to_num(pw_ann) * Wi, 0)
            acc_den += np.where(ok, Wi, 0)
        prev = cur

    if cfg.write_maps and acc_den is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = np.where(acc_den > 0, acc_num / acc_den, np.nan)
        Lm = np.where(np.isfinite(pm),
                      np.minimum(lifespan(np.nan_to_num(pm)), cfg.lifespan_cap_yr), np.nan)
        _write_tif(pm, Path(output_dir) / "pii_annual_win_period_mean.tif", profile0)
        _write_tif(Lm, Path(output_dir) / "lifespan_win_period_mean.tif", profile0)

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "interval_lifespan_stats.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# Overall summary
# ----------------------------------------------------------------------------
def overall_summary(df: pd.DataFrame) -> dict:
    """Geometric-mean annual persistence weighted by years spanned."""
    w = df["gap_yr"].to_numpy(float)
    logp = np.log(df["P_II_annual"].to_numpy(float))
    p_mean = float(np.exp((w * logp).sum() / w.sum()))
    return dict(total_years=float(w.sum()),
                cumulative_P=float(np.exp((w * logp).sum())),
                geo_mean_annual_P=p_mean,
                lifespan_yr=float(lifespan(p_mean)))


# ----------------------------------------------------------------------------
# Trend analysis (on logit of annual P_II)
# ----------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def trend_analysis(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    x = df["mid_year"].to_numpy(float)
    p = df["P_II_annual"].to_numpy(float)
    y = _logit(p)
    n = len(x)
    if n < 4:
        raise ValueError("Need >= 4 intervals for a trend test.")

    ts_slope, ts_int, ts_lo, ts_hi = stats.theilslopes(y, x, alpha=1 - alpha)
    tau, p_mk = stats.kendalltau(x, y)              # two-sided
    ols = stats.linregress(x, y)                    # two-sided
    resid = y - (ols.intercept + ols.slope * x)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 3 else np.nan

    def L_at(xx):
        return float(lifespan(_expit(ts_int + ts_slope * xx)))

    L0, L1 = L_at(x.min()), L_at(x.max())
    if ts_lo > 0:
        verdict = "significant INCREASE in persistence/lifespan (Theil-Sen CI > 0)"
    elif ts_hi < 0:
        verdict = "significant DECREASE in persistence/lifespan (Theil-Sen CI < 0)"
    else:
        verdict = "no significant monotonic trend (Theil-Sen CI includes 0)"

    out = pd.DataFrame([dict(
        n_intervals=n, x_min=x.min(), x_max=x.max(),
        theilsen_slope_logit_per_yr=ts_slope, theilsen_ci_lo=ts_lo, theilsen_ci_hi=ts_hi,
        kendall_tau=tau, kendall_p_two_sided=p_mk,
        ols_slope_logit_per_yr=ols.slope, ols_p_two_sided=ols.pvalue,
        resid_lag1_autocorr=r1,
        fitted_lifespan_start_yr=L0, fitted_lifespan_end_yr=L1,
        verdict=verdict)])
    log.info("TREND: %s | tau=%.3f p=%.3f | fitted L %.1f -> %.1f yr | resid r1=%.2f",
             verdict, tau, p_mk, L0, L1, r1)
    if abs(r1) > 0.4:
        log.warning("Residual autocorrelation is high; p-values are optimistic.")
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, trend: Optional[pd.DataFrame], out_dir: Path):
    if not _PLOT:
        return
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)
    x = df["mid_year"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].errorbar(x, df["P_II_annual"],
                   yerr=[df["P_II_annual"] - df["P_annual_boot_lo"],
                         df["P_annual_boot_hi"] - df["P_II_annual"]],
                   fmt="o", color="tab:orange", capsize=3, label="annual P_II (block-bootstrap 95%)")
    if trend is not None:
        t = trend.iloc[0]
        xs = np.linspace(x.min(), x.max(), 50)
        ys = _expit(stats.theilslopes(_logit(df["P_II_annual"].to_numpy()), x.to_numpy())[1]
                    + t.theilsen_slope_logit_per_yr * xs)
        ax[0].plot(xs, ys, "k--", label="Theil-Sen (logit scale)")
    ax[0].set_xlabel("Interval mid-year"); ax[0].set_ylabel("Annual P_II"); ax[0].legend()
    ax[0].grid(alpha=.3)

    ax[1].errorbar(x, df["lifespan_yr"],
                   yerr=[df["lifespan_yr"] - df["lifespan_boot_lo"],
                         df["lifespan_boot_hi"] - df["lifespan_yr"]],
                   fmt="o", color="tab:blue", capsize=3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Interval mid-year"); ax[1].set_ylabel("Lifespan 1/(1-P_annual) [yr]")
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "lifespan_timeseries.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def run_analysis(input_dir, output_dir, years: Sequence[int],
                 cfg: Optional[Config] = None):
    cfg = cfg or Config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = analyse_intervals(Path(input_dir), output_dir, years, cfg)
    summ = overall_summary(df)
    log.info("OVERALL: %s", summ)
    pd.DataFrame([summ]).to_csv(output_dir / "overall_summary.csv", index=False)

    trend = None
    try:
        trend = trend_analysis(df)
        trend.to_csv(output_dir / "trend_results.csv", index=False)
    except ValueError as e:
        log.warning("Trend skipped: %s", e)

    make_figures(df, trend, output_dir)
    return df, summ, trend