#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal hold-out evaluation utilities (2016-2025 by default)."""

import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from split_config import EVAL_END_YEAR, EVAL_START_YEAR
from clim_month_tools import time_mask_for_output_months, output_months_attr


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def find_coord_name(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Cannot find coordinate among candidates: {candidates}")


def find_time_var(ds: xr.Dataset, preferred=None):
    if preferred is None:
        preferred = []
    for name in preferred:
        if name in ds.data_vars or name in ds.coords:
            return name
    for name in ["time", "date", "Time"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError("Cannot find time coordinate.")


def pick_data_var(ds: xr.Dataset, preferred_names=None):
    if preferred_names is None:
        preferred_names = []
    for name in preferred_names:
        if name in ds.data_vars:
            return name
    for name, da in ds.data_vars.items():
        if da.ndim >= 3:
            return name
    raise KeyError("Cannot find a suitable data variable.")


def get_years_from_time(time_values: np.ndarray) -> np.ndarray:
    if np.issubdtype(time_values.dtype, np.datetime64):
        return pd.to_datetime(time_values).year.astype(np.int32).values
    raise ValueError("Time coordinate must be datetime64.")


def compute_holdout_metrics(
    pred_path: str,
    ref_path: str,
    out_csv: str,
    eval_start_year: int = EVAL_START_YEAR,
    eval_end_year: int = EVAL_END_YEAR,
    pred_var: Optional[str] = None,
    ref_var: Optional[str] = None,
    pred_preferred=None,
    ref_preferred=None,
) -> dict:
    if pred_preferred is None:
        pred_preferred = ["tmax_001_final", "tmax_005_final", "tmin_001_final", "tmin_005_final", "tmax", "tmin"]
    if ref_preferred is None:
        ref_preferred = ["tmax", "Tmax", "tmin", "Tmin", "clim_tmax", "clim_tmin"]

    log(f"Loading prediction file: {pred_path}")
    ds_pred = xr.open_dataset(pred_path)
    time_pred = find_time_var(ds_pred)
    lat_pred = find_coord_name(ds_pred, ["lat", "latitude", "y"])
    lon_pred = find_coord_name(ds_pred, ["lon", "longitude", "x"])
    pred_name = pred_var or pick_data_var(ds_pred, pred_preferred)
    da_pred = ds_pred[pred_name]

    log(f"Loading reference file: {ref_path}")
    ds_ref = xr.open_dataset(ref_path)
    time_ref = find_time_var(ds_ref)
    lat_ref = find_coord_name(ds_ref, ["lat", "latitude", "y"])
    lon_ref = find_coord_name(ds_ref, ["lon", "longitude", "x"])
    ref_name = ref_var or pick_data_var(ds_ref, ref_preferred)
    da_ref = ds_ref[ref_name]

    years_pred = get_years_from_time(ds_pred[time_pred].values)
    years_ref = get_years_from_time(ds_ref[time_ref].values)
    hold_pred = (years_pred >= eval_start_year) & (years_pred <= eval_end_year)
    hold_ref = (years_ref >= eval_start_year) & (years_ref <= eval_end_year)

    da_pred_h = da_pred.isel({time_pred: hold_pred})
    da_ref_h = da_ref.isel({time_ref: hold_ref})

    month_mask_pred = time_mask_for_output_months(da_pred_h[time_pred].values)
    month_mask_ref = time_mask_for_output_months(da_ref_h[time_ref].values)
    da_pred_h = da_pred_h.isel({time_pred: month_mask_pred})
    da_ref_h = da_ref_h.isel({time_ref: month_mask_ref})

    if da_pred_h.sizes[time_pred] != da_ref_h.sizes[time_ref]:
        raise ValueError(
            f"Hold-out time length mismatch: pred={da_pred_h.sizes[time_pred]}, ref={da_ref_h.sizes[time_ref]}"
        )

    if da_pred_h.sizes.get(lat_pred) != da_ref_h.sizes.get(lat_ref) or da_pred_h.sizes.get(lon_pred) != da_ref_h.sizes.get(lon_ref):
        log("Regridding reference to prediction grid (nearest)")
        da_ref_h = da_ref_h.interp({lat_ref: da_pred_h[lat_pred], lon_ref: da_pred_h[lon_pred]}, method="nearest")

    pred_vals = da_pred_h.transpose(time_pred, lat_pred, lon_pred).values.astype(np.float64)
    ref_vals = da_ref_h.transpose(time_pred, lat_pred, lon_pred).values.astype(np.float64)
    valid = np.isfinite(pred_vals) & np.isfinite(ref_vals)
    if not np.any(valid):
        ds_pred.close()
        ds_ref.close()
        raise ValueError("No valid overlapping cells in temporal hold-out period.")

    diff = pred_vals[valid] - ref_vals[valid]
    metrics = {
        "pred_file": pred_path,
        "ref_file": ref_path,
        "pred_var": pred_name,
        "ref_var": ref_name,
        "eval_start_year": int(eval_start_year),
        "eval_end_year": int(eval_end_year),
        "output_calendar_months": output_months_attr(),
        "n_days": int(da_pred_h.sizes[time_pred]),
        "n_valid": int(valid.sum()),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    ensure_parent_dir(out_csv)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    ds_pred.close()
    ds_ref.close()
    log(
        f"Hold-out {eval_start_year}-{eval_end_year}: "
        f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, bias={metrics['bias']:.4f}"
    )
    log(f"Metrics saved: {out_csv}")
    return metrics
