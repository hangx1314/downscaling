#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Warm-season monthly climatology training helpers (May–Sep by default)."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

import pandas as pd

from split_config import TRAIN_MONTHS, TRAIN_MONTHS_STR

OUTPUT_MONTHS = TRAIN_MONTHS
OUTPUT_MONTHS_STR = TRAIN_MONTHS_STR


def parse_train_months(spec: Optional[str] = None) -> List[int]:
    text = TRAIN_MONTHS_STR if spec is None or str(spec).strip() == "" else str(spec)
    months = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not months:
        raise ValueError("train-months must contain at least one month.")
    for m in months:
        if m < 1 or m > 12:
            raise ValueError(f"Invalid calendar month in train-months: {m}")
    return months


def resolve_train_months(
    months_available: Sequence[int],
    spec: Optional[str] = None,
) -> List[int]:
    requested = parse_train_months(spec)
    avail = {int(m) for m in months_available}
    selected = [m for m in requested if m in avail]
    if not selected:
        raise ValueError(
            f"No overlap between requested train months {requested} "
            f"and available months {sorted(avail)}."
        )
    return selected


def find_coord_name(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Cannot find coordinate among candidates: {candidates}")


def find_monthly_var(
    ds: xr.Dataset,
    month_name: str,
    lat_name: str,
    lon_name: str,
    preferred_names=None,
) -> str:
    if preferred_names is None:
        preferred_names = []
    for name in preferred_names:
        if name in ds.data_vars:
            da = ds[name]
            if month_name in da.dims and lat_name in da.dims and lon_name in da.dims:
                return name
    for name, da in ds.data_vars.items():
        if month_name in da.dims and lat_name in da.dims and lon_name in da.dims:
            return name
    raise KeyError("Cannot find a monthly 3D variable with month/lat/lon dims.")


def sort_da_by_latlon(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})
    if da[lon_name].values[0] > da[lon_name].values[-1]:
        da = da.isel({lon_name: slice(None, None, -1)})
    return da


def load_source_clim_on_grid(
    clim_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    preferred_names=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (clim_cube[12, nlat, nlon], month_values[12])."""
    if preferred_names is None:
        preferred_names = [
            "clim_tmax",
            "clim_tmin",
            "tmax_clim",
            "tmin_clim",
            "tmax",
            "tmin",
            "Band1",
        ]
    ds = xr.open_dataset(clim_path)
    month_name = find_coord_name(ds, ["month"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])
    var_name = find_monthly_var(ds, month_name, lat_name, lon_name, preferred_names)
    da = sort_da_by_latlon(ds[var_name], lat_name, lon_name)
    da = da.interp({lat_name: lat, lon_name: lon}, method="linear")
    month_values = np.arange(1, 13, dtype=np.int32)
    cube = np.full((12, lat.size, lon.size), np.nan, dtype=np.float32)
    for im, m in enumerate(month_values):
        if int(m) in set(int(x) for x in da[month_name].values):
            cube[im] = da.sel({month_name: m}).values.astype(np.float32)
    ds.close()
    return cube, month_values


def find_first_model_path(model_dir: str, prefix: str, months: Iterable[int]) -> str:
    for month in months:
        for ext in (".joblib", ".pt"):
            path = os.path.join(model_dir, f"{prefix}_month_{int(month):02d}{ext}")
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        f"No monthly model found under {model_dir} with prefix {prefix} "
        f"for months {list(months)}."
    )


def apply_source_clim_fallback(
    clim_cube: np.ndarray,
    train_months: Sequence[int],
    source_cube: np.ndarray,
    log_fn=print,
) -> None:
    train_set = {int(m) for m in train_months}
    for im in range(12):
        month = im + 1
        if month in train_set:
            continue
        if np.any(np.isfinite(clim_cube[im])):
            continue
        clim_cube[im] = source_cube[im]
        if log_fn is not None:
            log_fn(f"Month {month:02d}: no ML model — filled from source monthly climatology")


def output_months_list(spec: Optional[str] = None) -> List[int]:
    return parse_train_months(spec)


def output_months_attr(months: Optional[Sequence[int]] = None) -> str:
    months = output_months_list() if months is None else [int(m) for m in months]
    return ",".join(str(m) for m in months)


def time_mask_for_output_months(
    time_values: np.ndarray,
    months: Optional[Sequence[int]] = None,
) -> np.ndarray:
    months = output_months_list() if months is None else [int(m) for m in months]
    month_arr = pd.DatetimeIndex(pd.to_datetime(time_values)).month
    return np.isin(month_arr, months)


def filter_monthly_dataset_to_output_months(
    ds: xr.Dataset,
    month_name: str = "month",
    months: Optional[Sequence[int]] = None,
) -> xr.Dataset:
    months = output_months_list() if months is None else [int(m) for m in months]
    if month_name not in ds.dims and month_name not in ds.coords:
        month_name = find_coord_name(ds, ["month"])
    out = ds.sel({month_name: months})
    out.attrs = dict(ds.attrs)
    out.attrs["output_calendar_months"] = output_months_attr(months)
    return out


def add_dataarrays_on_same_grid(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> xr.DataArray:
    """
    Element-wise add on the same physical grid.

    xarray's ``+`` aligns on coordinate *labels*; interpolated products often
    differ by ~1e-6 in lat/lon metadata, which collapses the result to a tiny
    inner join. Use numpy addition after a tolerance check instead.
    """
    if da_a.dims != da_b.dims:
        raise ValueError(f"Dimension mismatch: {da_a.dims} vs {da_b.dims}")
    for dim in da_a.dims:
        if da_a.sizes[dim] != da_b.sizes[dim]:
            raise ValueError(
                f"Size mismatch on {dim}: {da_a.sizes[dim]} vs {da_b.sizes[dim]}"
            )
    spatial_coords = ("lat", "lon", "latitude", "longitude", "y", "x")
    for coord in spatial_coords:
        if coord not in da_a.coords or coord not in da_b.coords:
            continue
        if not np.allclose(da_a[coord].values, da_b[coord].values, rtol=rtol, atol=atol):
            raise ValueError(
                f"Coordinate '{coord}' differs beyond tolerance "
                f"(max abs diff={np.max(np.abs(da_a[coord].values - da_b[coord].values)):.3e})."
            )
    out = da_a.copy(deep=False)
    out.values = (da_a.values + da_b.values).astype(np.float32)
    return out


def filter_dataset_to_output_months(
    ds: xr.Dataset,
    months: Optional[Sequence[int]] = None,
) -> xr.Dataset:
    months = output_months_list() if months is None else [int(m) for m in months]
    time_name = find_coord_name(ds, ["time", "date"])
    mask = time_mask_for_output_months(ds[time_name].values, months)
    if not np.any(mask):
        raise ValueError(
            f"No time steps left after filtering to calendar months {months}."
        )
    out = ds.isel({time_name: mask})
    out.attrs = dict(ds.attrs)
    out.attrs["output_calendar_months"] = output_months_attr(months)
    return out


def subset_monthly_cube_to_output_months(
    clim_cube_12: np.ndarray,
    months: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select rows from a 12-month cube (index 0=Jan) for output months only."""
    months = output_months_list() if months is None else sorted({int(m) for m in months})
    out = np.stack([clim_cube_12[int(m) - 1] for m in months], axis=0).astype(np.float32)
    return out, np.asarray(months, dtype=np.int32)


def load_predicted_clim_on_grid(
    clim_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    preferred_names=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ML-predicted monthly clim (may contain only output months)."""
    if preferred_names is None:
        preferred_names = [
            "clim_tmax_rf",
            "clim_tmax_xgb",
            "clim_tmax_unet",
            "clim_tmin_rf",
            "clim_tmin_xgb",
            "clim_tmin_unet",
            "clim_tmax",
            "clim_tmin",
            "tmax_clim",
            "tmin_clim",
        ]
    ds = xr.open_dataset(clim_path)
    month_name = find_coord_name(ds, ["month"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])
    var_name = find_monthly_var(ds, month_name, lat_name, lon_name, preferred_names)
    da = sort_da_by_latlon(ds[var_name], lat_name, lon_name)
    if not (np.allclose(da[lat_name].values, lat) and np.allclose(da[lon_name].values, lon)):
        da = da.interp({lat_name: lat, lon_name: lon}, method="linear")
    month_values = np.asarray(da[month_name].values, dtype=np.int32)
    cube = da.transpose(month_name, lat_name, lon_name).values.astype(np.float32)
    ds.close()
    return cube, month_values


def assemble_full_year_clim_on_grid(
    predict_clim_path: str,
    source_clim_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    output_months: Optional[Sequence[int]] = None,
    preferred_predict=None,
    preferred_source=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 12-month climatology on the target grid for daily expansion.
    Output months use ML prediction; other months use bilinear source clim.
    """
    output_months = output_months_list() if output_months is None else [int(m) for m in output_months]
    output_set = set(output_months)
    source_cube, _ = load_source_clim_on_grid(
        source_clim_path, lat, lon, preferred_names=preferred_source
    )
    full_cube = source_cube.copy()
    try:
        pred_cube, pred_months = load_predicted_clim_on_grid(
            predict_clim_path, lat, lon, preferred_names=preferred_predict
        )
        month_to_row = {int(m): i for i, m in enumerate(pred_months)}
        for m in output_months:
            if m not in month_to_row:
                raise ValueError(
                    f"Predicted climatology missing output month {m:02d} "
                    f"(available={sorted(month_to_row)})."
                )
            full_cube[m - 1] = pred_cube[month_to_row[m]]
    except FileNotFoundError:
        pass
    month_values = np.arange(1, 13, dtype=np.int32)
    if output_set != set(month_values):
        # When only warm-season ML exists, non-output months remain from source clim.
        pass
    return full_cube, month_values
