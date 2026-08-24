#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_make_clim_anom_025.py

Build monthly climatology and daily anomaly from 0.25-degree daily Tmax.

Input:
- CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc

Outputs:
- tmax_025_clim_monthly.nc
- tmax_025_anom_daily.nc
"""

import os
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import sys
_HERE = os.path.abspath(os.path.dirname(__file__))
_ZCN_COMMON = None
_p = _HERE
for _ in range(8):
    _cand = os.path.join(_p, "common")
    if os.path.isdir(_cand):
        _ZCN_COMMON = _cand
        break
    _p = os.path.dirname(_p)
if _ZCN_COMMON is None:
    raise RuntimeError("Cannot locate common/ directory from %s" % _HERE)
if _ZCN_COMMON not in sys.path:
    sys.path.insert(0, _ZCN_COMMON)
from clim_month_tools import filter_dataset_to_output_months, filter_monthly_dataset_to_output_months
from split_config import CLIM_START_YEAR, CLIM_END_YEAR


def get_years_from_time_values(time_values):
    import pandas as pd
    if np.issubdtype(time_values.dtype, np.datetime64):
        return pd.to_datetime(time_values).year.astype(np.int32).values
    raise ValueError("Time values are not decoded as datetime64.")



# =========================================================
# Default paths
# =========================================================
DEFAULT_TMAX_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc"
DEFAULT_CLIM_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/tmax_025_clim_monthly.nc"
DEFAULT_ANOM_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/tmax_025_anom_daily.nc"

# Optional manual override
TARGET_VAR_NAME = None


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


def find_3d_target_var(ds: xr.Dataset, time_name: str, lat_name: str, lon_name: str, preferred_names=None):
    if preferred_names is None:
        preferred_names = []

    for name in preferred_names:
        if name in ds.data_vars:
            da = ds[name]
            if time_name in da.dims and lat_name in da.dims and lon_name in da.dims:
                return name

    for name, da in ds.data_vars.items():
        if time_name in da.dims and lat_name in da.dims and lon_name in da.dims:
            return name

    raise KeyError("Cannot find a 3D target variable with time/lat/lon dims.")


def to_ordered_da(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str) -> xr.DataArray:
    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values

    if lat_vals[0] > lat_vals[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    if lon_vals[0] > lon_vals[-1]:
        da = da.isel({lon_name: slice(None, None, -1)})

    da = da.transpose(time_name, lat_name, lon_name)
    return da


def main():
    parser = argparse.ArgumentParser(description="Build monthly climatology and daily anomaly from 0.25-degree Tmax.")
    parser.add_argument("--input", default=DEFAULT_TMAX_PATH, help="Input daily Tmax NetCDF")
    parser.add_argument("--clim-out", default=DEFAULT_CLIM_OUT, help="Output monthly climatology NetCDF")
    
    parser.add_argument("--clim-start-year", type=int, default=CLIM_START_YEAR, help="Start year for climatology baseline and yearly predictor aggregation")
    parser.add_argument("--clim-end-year", type=int, default=CLIM_END_YEAR, help="End year for climatology baseline and yearly predictor aggregation")
    parser.add_argument("--anom-out", default=DEFAULT_ANOM_OUT, help="Output daily anomaly NetCDF")
    parser.add_argument("--var", default=None, help="Optional target variable name inside input NetCDF")
    args = parser.parse_args()

    ensure_parent_dir(args.clim_out)
    ensure_parent_dir(args.anom_out)

    log(f"Loading input dataset: {args.input}")
    ds = xr.open_dataset(args.input)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if args.var is not None:
        if args.var not in ds.data_vars:
            raise KeyError(f"--var '{args.var}' not found in input dataset.")
        var_name = args.var
    elif TARGET_VAR_NAME is not None:
        if TARGET_VAR_NAME not in ds.data_vars:
            raise KeyError(f"TARGET_VAR_NAME '{TARGET_VAR_NAME}' not found in input dataset.")
        var_name = TARGET_VAR_NAME
    else:
        var_name = find_3d_target_var(
            ds,
            time_name,
            lat_name,
            lon_name,
            preferred_names=["tmax", "Tmax", "tmx", "tasmax", "Band1"]
        )

    log(f"Using target variable: {var_name}")

    da = ds[var_name]
    da = to_ordered_da(da, time_name, lat_name, lon_name)
    da = da.astype(np.float32)

    years = get_years_from_time_values(da[time_name].values)
    clim_mask = (years >= args.clim_start_year) & (years <= args.clim_end_year)
    if not np.any(clim_mask):
        raise ValueError("No time steps found in climatology baseline period.")
    log(f"Computing monthly climatology from {args.clim_start_year}-{args.clim_end_year}")
    clim = da.isel({time_name: clim_mask}).groupby(f"{time_name}.month").mean(time_name, skipna=True)
    clim.name = "clim_tmax"

    log("Computing daily anomaly")
    anom = da.groupby(f"{time_name}.month") - clim
    anom.name = "anom_tmax"

    clim.attrs = {
        "long_name": "Monthly climatology of Tmax on 0.25-degree grid",
        "units": da.attrs.get("units", ""),
        "source_variable": var_name,
        "aggregation": "train-period monthly mean",
    }

    anom.attrs = {
        "long_name": "Daily anomaly of Tmax relative to monthly climatology on 0.25-degree grid",
        "units": da.attrs.get("units", ""),
        "source_variable": var_name,
        "definition": "daily_tmax - monthly_climatology",
    }

    ds_clim = xr.Dataset(
        {"clim_tmax": clim},
        attrs={
            "title": "Monthly climatology of 0.25-degree Tmax",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": args.input,
            "source_variable": var_name,
        },
    )

    ds_anom = xr.Dataset(
        {"anom_tmax": anom},
        attrs={
            "title": "Daily anomaly of 0.25-degree Tmax",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": args.input,
            "source_variable": var_name,
        },
    )

    ds_clim["month"].attrs = {"long_name": "calendar month", "units": "1"}
    ds_clim["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_clim["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    ds_anom["time"].attrs = ds[time_name].attrs
    ds_anom["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_anom["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    clim_encoding = {"clim_tmax": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    anom_encoding = {"anom_tmax": {"zlib": True, "complevel": 4, "dtype": "float32"}}

    ds_clim = filter_monthly_dataset_to_output_months(ds_clim)
    ds_anom = filter_dataset_to_output_months(ds_anom)

    log(f"Saving climatology: {args.clim_out}")
    ds_clim.to_netcdf(args.clim_out, encoding=clim_encoding)

    log(f"Saving anomaly: {args.anom_out}")
    ds_anom.to_netcdf(args.anom_out, encoding=anom_encoding)

    ds.close()
    log("Done.")


if __name__ == "__main__":
    main()