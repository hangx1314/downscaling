#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_correct_tmin005.py

Apply interpolated 0.25->0.05 residual correction to the initial
0.05-degree daily Tmin field.

Inputs:
- tmin_005_raw.nc
- residual_005_interp.nc

Output:
- tmin_005_corr.nc
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


DEFAULT_RAW005_PATH = "./outputs/tmin/xgb/xgb005/interim/tmin_005_raw.nc"
DEFAULT_RES005_PATH = "./outputs/tmin/xgb/xgb005/interim/residual_005_interp.nc"
DEFAULT_OUT_PATH = "./outputs/tmin/xgb/xgb005/interim/tmin_005_corr.nc"

RAW_VAR_NAME = None
RES_VAR_NAME = None


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


def find_3d_var(ds: xr.Dataset, time_name: str, lat_name: str, lon_name: str, preferred_names=None):
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

    raise KeyError("Cannot find a 3D variable with time/lat/lon dims.")


def sort_da_by_latlon(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})
    if da[lon_name].values[0] > da[lon_name].values[-1]:
        da = da.isel({lon_name: slice(None, None, -1)})
    return da


def load_field(path: str, default_names, override_var_name=None, fallback_global=None):
    log(f"Loading field: {path}")
    ds = xr.open_dataset(path)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if override_var_name is not None:
        if override_var_name not in ds.data_vars:
            raise KeyError(f"Variable '{override_var_name}' not found in {path}")
        var_name = override_var_name
    elif fallback_global is not None:
        if fallback_global not in ds.data_vars:
            raise KeyError(f"Variable '{fallback_global}' not found in {path}")
        var_name = fallback_global
    else:
        var_name = find_3d_var(
            ds,
            time_name,
            lat_name,
            lon_name,
            preferred_names=default_names
        )

    da = ds[var_name]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(time_name, lat_name, lon_name).astype(np.float32)

    return ds, da, time_name, lat_name, lon_name, var_name


def main():
    parser = argparse.ArgumentParser(description="Apply interpolated residual correction to 0.05-degree daily Tmin.")
    parser.add_argument("--raw005", default=DEFAULT_RAW005_PATH, help="Input 0.05-degree raw Tmin NetCDF")
    parser.add_argument("--res005", default=DEFAULT_RES005_PATH, help="Input 0.05-degree interpolated residual NetCDF")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output corrected 0.05-degree Tmin NetCDF")
    parser.add_argument("--raw-var", default=None, help="Optional raw Tmin variable name")
    parser.add_argument("--res-var", default=None, help="Optional residual variable name")
    args = parser.parse_args()

    ensure_parent_dir(args.out)

    ds_raw, da_raw, raw_time_name, raw_lat_name, raw_lon_name, raw_var_name = load_field(
        args.raw005,
        default_names=["tmin_005_raw", "tmin", "Band1"],
        override_var_name=args.raw_var,
        fallback_global=RAW_VAR_NAME,
    )

    ds_res, da_res, res_time_name, res_lat_name, res_lon_name, res_var_name = load_field(
        args.res005,
        default_names=["residual_tmin_005_interp", "residual_tmin_005", "residual", "Band1"],
        override_var_name=args.res_var,
        fallback_global=RES_VAR_NAME,
    )

    if not np.array_equal(da_raw[raw_time_name].values, da_res[res_time_name].values):
        raise ValueError("Time coordinates of raw field and residual field do not match.")
    if not np.allclose(da_raw[raw_lat_name].values, da_res[res_lat_name].values):
        raise ValueError("Latitude coordinates of raw field and residual field do not match.")
    if not np.allclose(da_raw[raw_lon_name].values, da_res[res_lon_name].values):
        raise ValueError("Longitude coordinates of raw field and residual field do not match.")

    da_raw_std = da_raw.rename(
        {
            raw_time_name: "time",
            raw_lat_name: "lat",
            raw_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")

    da_res_std = da_res.rename(
        {
            res_time_name: "time",
            res_lat_name: "lat",
            res_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")

    log("Applying residual correction")
    da_corr = (da_raw_std + da_res_std).astype(np.float32)
    da_corr.name = "tmin_005_corr"
    da_corr.attrs = {
        "long_name": "Residual-corrected daily Tmin on 0.05-degree grid",
        "units": da_raw.attrs.get("units", da_res.attrs.get("units", "")),
        "definition": "tmin_005_raw + residual_tmin_005_interp",
        "raw_file": args.raw005,
        "raw_variable": raw_var_name,
        "residual_file": args.res005,
        "residual_variable": res_var_name,
    }

    ds_out = xr.Dataset(
        {"tmin_005_corr": da_corr},
        attrs={
            "title": "Residual-corrected daily Tmin on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "raw_file": args.raw005,
            "raw_variable": raw_var_name,
            "residual_file": args.res005,
            "residual_variable": res_var_name,
        },
    )

    ds_out["time"].attrs = ds_raw[raw_time_name].attrs
    ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    log(f"Saving output: {args.out}")
    ds_out.to_netcdf(
        args.out,
        encoding={"tmin_005_corr": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    ds_raw.close()
    ds_res.close()
    log("Done.")


if __name__ == "__main__":
    main()