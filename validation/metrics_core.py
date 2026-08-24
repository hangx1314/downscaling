#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core validation metric formulas for warm-season coastal China downscaling.

Self-contained reference implementation (no path config, no plotting runners).
Convention: u = model/prediction, x = observation/reference.

Formulas
--------
- Bias   = mean(u - x)
- MAE    = mean(|u - x|)
- RMSE   = sqrt(mean((u - x)^2))
- R2     = 1 - SS_res / SS_tot  with y_true=x, y_pred=u
- Cor    = Pearson correlation between u and x
- SDR    = std(u) / std(x)
- PSS    = sum(min(p_u, p_x)) over shared histogram bins
- Pk_Bias = percentile_k(u) - percentile_k(x)

Heatwave / occurrence indices use a calendar-day sliding window:
  threshold(d) = P90 over baseline years in [d-7, d+7] (15-day window).
Occurrence count = days exceeding the fixed threshold in the eval period.

DTR validation:
  DTR_obs  = station Tmax - station Tmin
  DTR_pred = model Tmax - model Tmin  (same stations, nearest cell)
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

DEFAULT_PSS_BINS = 120


def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 = 1 - SS_res / SS_tot. May be negative."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def safe_nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1))


def compute_pss(u: np.ndarray, x: np.ndarray, bins: int = DEFAULT_PSS_BINS) -> float:
    """Pattern similarity score from overlapping histogram mass."""
    u = np.asarray(u, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(u) & np.isfinite(x)
    u = u[valid]
    x = x[valid]
    if u.size < 5:
        return np.nan

    lo = min(float(np.nanmin(u)), float(np.nanmin(x)))
    hi = max(float(np.nanmax(u)), float(np.nanmax(x)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan

    edges = np.linspace(lo, hi, int(bins) + 1)
    hu, _ = np.histogram(u, bins=edges)
    hx, _ = np.histogram(x, bins=edges)
    if hu.sum() == 0 or hx.sum() == 0:
        return np.nan

    pu = hu / hu.sum()
    px = hx / hx.sum()
    return float(np.sum(np.minimum(pu, px)))


def compute_metrics(u: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """Scalar skill metrics for paired samples (u=pred, x=obs)."""
    u = np.asarray(u, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(u) & np.isfinite(x)
    u = u[valid]
    x = x[valid]

    out = {
        "N": 0,
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Bias": np.nan,
        "Cor": np.nan,
        "SDR": np.nan,
        "PSS": np.nan,
        "P90_Bias": np.nan,
        "P95_Bias": np.nan,
        "P99_Bias": np.nan,
    }
    if u.size == 0:
        return out

    diff = u - x
    out["N"] = int(u.size)
    out["MAE"] = float(np.mean(np.abs(diff)))
    out["RMSE"] = float(np.sqrt(np.mean(diff**2)))
    out["Bias"] = float(np.mean(diff))

    u_mean = float(np.mean(u))
    x_mean = float(np.mean(x))
    num = float(np.sum((u - u_mean) * (x - x_mean)))
    den = float(np.sqrt(np.sum((u - u_mean) ** 2) * np.sum((x - x_mean) ** 2)))
    out["Cor"] = float(num / den) if den > 0 else np.nan
    out["R2"] = coefficient_of_determination(x, u)

    std_u = safe_nanstd(u)
    std_x = safe_nanstd(x)
    out["SDR"] = float(std_u / std_x) if np.isfinite(std_u) and np.isfinite(std_x) and std_x > 0 else np.nan
    out["PSS"] = compute_pss(u, x)

    out["P90_Bias"] = float(np.nanpercentile(u, 90) - np.nanpercentile(x, 90))
    out["P95_Bias"] = float(np.nanpercentile(u, 95) - np.nanpercentile(x, 95))
    out["P99_Bias"] = float(np.nanpercentile(u, 99) - np.nanpercentile(x, 99))
    return out


def compute_station_metrics(
    model_values: np.ndarray,
    ref_values: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    model_name: str,
    pss_batch: int = 2048,
) -> pd.DataFrame:
    """Per-station metrics along time axis 0. Shapes: (time, n_stations)."""
    u = np.asarray(model_values, dtype=np.float64)
    x = np.asarray(ref_values, dtype=np.float64)
    if u.shape != x.shape:
        raise ValueError("model_values and ref_values must share shape (time, station).")

    _, n_sta = u.shape
    valid = np.isfinite(u) & np.isfinite(x)
    diff = np.where(valid, u - x, np.nan)

    mae = np.nanmean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.nanmean(diff**2, axis=0))
    bias = np.nanmean(diff, axis=0)

    u_mean = np.nanmean(np.where(valid, u, np.nan), axis=0)
    x_mean = np.nanmean(np.where(valid, x, np.nan), axis=0)
    uc = np.where(valid, u - u_mean, np.nan)
    xc = np.where(valid, x - x_mean, np.nan)
    num = np.nansum(uc * xc, axis=0)
    den = np.sqrt(np.nansum(uc**2, axis=0) * np.nansum(xc**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cor = np.where(den > 0, num / den, np.nan)
        ss_res = np.nansum(diff**2, axis=0)
        ss_tot = np.nansum(xc**2, axis=0)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
        std_u = np.nanstd(u, axis=0, ddof=1)
        std_x = np.nanstd(x, axis=0, ddof=1)
        sdr = np.where(np.isfinite(std_u) & np.isfinite(std_x) & (std_x > 0), std_u / std_x, np.nan)

    uu = np.where(valid, u, np.nan)
    xx = np.where(valid, x, np.nan)
    p90b = np.nanpercentile(uu, 90, axis=0) - np.nanpercentile(xx, 90, axis=0)
    p95b = np.nanpercentile(uu, 95, axis=0) - np.nanpercentile(xx, 95, axis=0)
    p99b = np.nanpercentile(uu, 99, axis=0) - np.nanpercentile(xx, 99, axis=0)

    pss = np.full(n_sta, np.nan, dtype=np.float64)
    bs = max(1, int(pss_batch))
    for start in range(0, n_sta, bs):
        end = min(n_sta, start + bs)
        for j in range(start, end):
            pss[j] = compute_pss(u[:, j], x[:, j])

    return pd.DataFrame(
        {
            "model": model_name,
            "station_id": np.arange(n_sta, dtype=np.int64),
            "lat": np.asarray(lat_values, dtype=np.float64),
            "lon": np.asarray(lon_values, dtype=np.float64),
            "N": np.sum(valid, axis=0),
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "R2": r2,
            "Cor": cor,
            "SDR": sdr,
            "PSS": pss,
            "P90_Bias": p90b,
            "P95_Bias": p95b,
            "P99_Bias": p99b,
        }
    )


def compute_doy365_array(time_index: Iterable) -> np.ndarray:
    """365-day calendar index; Feb 29 mapped to Feb 28."""
    ti = pd.DatetimeIndex(time_index)
    doy = np.asarray(ti.dayofyear, dtype=np.int32)
    leap_after_feb = np.asarray(ti.is_leap_year) & (np.asarray(ti.month) > 2)
    return doy - leap_after_feb.astype(np.int32)


def build_window_doys(doy: int, half_window: int, n_days: int = 365) -> np.ndarray:
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int32)
    return ((doy - 1 + offsets) % n_days) + 1


def sliding_percentile_threshold(
    baseline_values: np.ndarray,
    baseline_doys365: np.ndarray,
    percentile: float = 90.0,
    half_window: int = 7,
) -> np.ndarray:
    """Fixed calendar-day thresholds shape (365,) from baseline daily series."""
    thr = np.full(365, np.nan, dtype=np.float64)
    for d in range(1, 366):
        window = build_window_doys(d, half_window)
        mask = np.isin(baseline_doys365, window) & np.isfinite(baseline_values)
        vals = baseline_values[mask]
        thr[d - 1] = float(np.nanpercentile(vals, percentile)) if vals.size else np.nan
    return thr


def compute_event_stats_1d(
    exceed_1d: np.ndarray,
    inten_1d: np.ndarray,
    min_dur: int,
) -> Tuple[int, int, int, float, float, float]:
    """Run-length stats for exceedance events (duration >= min_dur)."""
    x = exceed_1d.astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    lens = ends - starts
    ok = lens >= min_dur
    if not np.any(ok):
        return 0, 0, 0, 0.0, 0.0, 0.0

    starts = starts[ok]
    ends = ends[ok]
    lens = lens[ok]
    n_events = int(lens.size)
    n_days = int(lens.sum())
    max_dur = int(lens.max())

    mask = np.zeros_like(exceed_1d, dtype=bool)
    for st, en in zip(starts, ends):
        mask[st:en] = True
    inten_sel = inten_1d[mask]
    cum_int = float(np.sum(inten_sel)) if inten_sel.size else 0.0
    mean_int = float(cum_int / n_days) if n_days > 0 else 0.0
    max_int = float(np.max(inten_sel)) if inten_sel.size else 0.0
    return n_events, n_days, max_dur, cum_int, mean_int, max_int


def compute_yearly_occurrence_metrics(
    values: np.ndarray,
    doys365: np.ndarray,
    threshold365: np.ndarray,
    min_duration: int = 3,
) -> Dict[str, float]:
    """Occurrence + event metrics for one station-year series."""
    thr = threshold365[doys365 - 1]
    exceed = np.isfinite(values) & np.isfinite(thr) & (values > thr)
    inten = np.where(exceed, values - thr, 0.0)
    n_hot = int(np.sum(exceed))
    n_events, n_hw_days, max_dur, cum_int, mean_int, max_int = compute_event_stats_1d(
        exceed.astype(np.int8), inten, min_duration
    )
    mean_dur = float(n_hw_days / n_events) if n_events > 0 else np.nan
    return {
        "n_hot_days": float(n_hot),
        "n_events": float(n_events),
        "n_hw_days": float(n_hw_days),
        "max_duration": float(max_dur),
        "mean_duration": mean_dur,
        "mean_intensity": mean_int if n_hw_days > 0 else np.nan,
        "max_intensity": max_int if n_hw_days > 0 else np.nan,
        "cum_intensity": cum_int,
    }


def compute_dtr_metrics(
    tmax_model: np.ndarray,
    tmin_model: np.ndarray,
    tmax_ref: np.ndarray,
    tmin_ref: np.ndarray,
) -> Dict[str, float]:
    """DTR skill: model DTR vs station DTR, same alignment along time."""
    dtr_pred = tmax_model - tmin_model
    dtr_obs = tmax_ref - tmin_ref
    return compute_metrics(dtr_pred, dtr_obs)


def compute_stratified_station_metrics(
    obs_yearly: np.ndarray,
    mod_yearly: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-station Bias/MAE/RMSE across years. Arrays (n_year, n_sta)."""
    diff = mod_yearly - obs_yearly
    with np.errstate(invalid="ignore"):
        bias = np.nanmean(diff, axis=0)
        mae = np.nanmean(np.abs(diff), axis=0)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
    return bias, mae, rmse


def aggregate_subcells_to_parent(
    high_res_values: np.ndarray,
    flat_target: np.ndarray,
    nlat_parent: int,
    nlon_parent: int,
) -> np.ndarray:
    """Mean of valid high-res subcells within each parent grid cell."""
    out = np.full((nlat_parent, nlon_parent), np.nan, dtype=np.float64)
    counts = np.zeros((nlat_parent, nlon_parent), dtype=np.int32)
    sums = np.zeros((nlat_parent, nlon_parent), dtype=np.float64)

    flat = high_res_values.ravel()
    for idx, parent in enumerate(flat_target.ravel()):
        if parent < 0 or not np.isfinite(flat[idx]):
            continue
        row, col = divmod(int(parent), nlon_parent)
        sums[row, col] += float(flat[idx])
        counts[row, col] += 1

    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out.astype(np.float32)
