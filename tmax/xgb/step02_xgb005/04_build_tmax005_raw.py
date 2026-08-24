#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_build_tmax005_raw.py

Build initial 0.05-degree daily Tmax by combining:
- RF-predicted monthly climatology on 0.05-degree grid
- interpolated daily anomaly on 0.05-degree grid

Inputs:
- tmax_005_clim_rf.nc
- tmax_005_anom_interp.nc

Output:
- tmax_005_raw.nc
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
from clim_month_tools import assemble_full_year_clim_on_grid, filter_dataset_to_output_months, filter_monthly_dataset_to_output_months


# =========================================================
# Default paths
# =========================================================
DEFAULT_SOURCE_CLIM = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/tmax_025_clim_monthly.nc"
DEFAULT_CLIM005_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005//interim/tmax_005_clim_xgb.nc"
DEFAULT_ANOM005_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/tmax_005_anom_interp.nc"
DEFAULT_OUT_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/tmax_005_raw.nc"

# Optional manual overrides
CLIM_VAR_NAME = None
ANOM_VAR_NAME = None


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


def find_monthly_var(ds: xr.Dataset, month_name: str, lat_name: str, lon_name: str, preferred_names=None):
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


def find_time_var(ds: xr.Dataset, time_name: str, lat_name: str, lon_name: str, preferred_names=None):
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

    raise KeyError("Cannot find a time-varying 3D variable with time/lat/lon dims.")


def sort_da_by_latlon(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})
    if da[lon_name].values[0] > da[lon_name].values[-1]:
        da = da.isel({lon_name: slice(None, None, -1)})
    return da


def load_clim(clim_path: str, clim_var_name: str = None):
    log(f"Loading climatology dataset: {clim_path}")
    ds = xr.open_dataset(clim_path)

    month_name = find_coord_name(ds, ["month"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if clim_var_name is not None:
        if clim_var_name not in ds.data_vars:
            raise KeyError(f"clim_var_name '{clim_var_name}' not found in climatology dataset.")
        var_name = clim_var_name
    elif CLIM_VAR_NAME is not None:
        if CLIM_VAR_NAME not in ds.data_vars:
            raise KeyError(f"CLIM_VAR_NAME '{CLIM_VAR_NAME}' not found in climatology dataset.")
        var_name = CLIM_VAR_NAME
    else:
        var_name = find_monthly_var(
            ds,
            month_name,
            lat_name,
            lon_name,
            preferred_names=["clim_tmax_xgb", "clim_tmax", "tmax_clim_xgb", "Band1"]
        )

    da = ds[var_name]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(month_name, lat_name, lon_name).astype(np.float32)

    return ds, da, month_name, lat_name, lon_name, var_name


def load_anom(anom_path: str, anom_var_name: str = None):
    log(f"Loading interpolated anomaly dataset: {anom_path}")
    ds = xr.open_dataset(anom_path)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if anom_var_name is not None:
        if anom_var_name not in ds.data_vars:
            raise KeyError(f"anom_var_name '{anom_var_name}' not found in anomaly dataset.")
        var_name = anom_var_name
    elif ANOM_VAR_NAME is not None:
        if ANOM_VAR_NAME not in ds.data_vars:
            raise KeyError(f"ANOM_VAR_NAME '{ANOM_VAR_NAME}' not found in anomaly dataset.")
        var_name = ANOM_VAR_NAME
    else:
        var_name = find_time_var(
            ds,
            time_name,
            lat_name,
            lon_name,
            preferred_names=["anom_tmax_interp", "anom_tmax", "tmax_anom_interp", "Band1"]
        )

    da = ds[var_name]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(time_name, lat_name, lon_name).astype(np.float32)

    return ds, da, time_name, lat_name, lon_name, var_name


def main():
    parser = argparse.ArgumentParser(description="Build initial 0.05-degree daily Tmax from monthly climatology and anomaly.")
    parser.add_argument("--clim005", default=DEFAULT_CLIM005_PATH, help="Input RF-predicted 0.05-degree monthly climatology NetCDF")
    parser.add_argument("--anom005", default=DEFAULT_ANOM005_PATH, help="Input interpolated 0.05-degree daily anomaly NetCDF")
    parser.add_argument("--source-clim", default=DEFAULT_SOURCE_CLIM, help="Source monthly climatology for non-ML months when expanding to daily")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output 0.05-degree daily raw Tmax NetCDF")
    parser.add_argument("--clim-var", default=None, help="Optional climatology variable name")
    parser.add_argument("--anom-var", default=None, help="Optional anomaly variable name")
    args = parser.parse_args()

    ensure_parent_dir(args.out)

    ds_clim, _, month_name, clim_lat_name, clim_lon_name, clim_var_name = load_clim(
        args.clim005, clim_var_name=args.clim_var
    )
    clim_cube, month_coords = assemble_full_year_clim_on_grid(
        args.clim005,
        args.source_clim,
        ds_clim[clim_lat_name].values,
        ds_clim[clim_lon_name].values,
    )
    da_clim = xr.DataArray(
        clim_cube,
        coords={
            month_name: month_coords,
            clim_lat_name: ds_clim[clim_lat_name].values,
            clim_lon_name: ds_clim[clim_lon_name].values,
        },
        dims=(month_name, clim_lat_name, clim_lon_name),
        name=clim_var_name,
    )
    ds_anom, da_anom, time_name, anom_lat_name, anom_lon_name, anom_var_name = load_anom(
        args.anom005, anom_var_name=args.anom_var
    )

    if not np.allclose(da_clim[clim_lat_name].values, da_anom[anom_lat_name].values):
        raise ValueError("Latitude coordinates of climatology and anomaly datasets do not match.")
    if not np.allclose(da_clim[clim_lon_name].values, da_anom[anom_lon_name].values):
        raise ValueError("Longitude coordinates of climatology and anomaly datasets do not match.")

    month_indexer = xr.DataArray(
        da_anom[time_name].dt.month.values,
        coords={time_name: da_anom[time_name].values},
        dims=(time_name,),
        name="month_indexer"
    )

    log("Expanding monthly climatology to daily time axis")
    da_clim_daily = da_clim.sel({month_name: month_indexer})
    
    # Vectorized month selection already creates the time dimension.
    # Do not rename month -> time again.
    if month_name in da_clim_daily.coords:
        da_clim_daily = da_clim_daily.drop_vars(month_name)
    
    da_clim_daily = da_clim_daily.assign_coords({time_name: da_anom[time_name].values})
    da_clim_daily = da_clim_daily.transpose(time_name, clim_lat_name, clim_lon_name)
    
    da_anom_std = da_anom.rename(
        {
            time_name: "time",
            anom_lat_name: "lat",
            anom_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")
    
    da_clim_daily_std = da_clim_daily.rename(
        {
            clim_lat_name: "lat",
            clim_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")
    
    log("Combining climatology and anomaly to build initial daily Tmax")
    da_raw = da_clim_daily_std + da_anom_std

    da_raw.name = "tmax_005_raw"
    da_raw.attrs = {
        "long_name": "Initial daily Tmax on 0.05-degree grid before residual correction",
        "units": da_clim.attrs.get("units", da_anom.attrs.get("units", "")),
        "definition": "monthly_climatology_rf + interpolated_daily_anomaly",
        "climatology_file": args.clim005,
        "climatology_variable": clim_var_name,
        "anomaly_file": args.anom005,
        "anomaly_variable": anom_var_name,
    }

    ds_out = xr.Dataset(
        {"tmax_005_raw": da_raw.astype(np.float32)},
        attrs={
            "title": "Initial daily Tmax on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "climatology_file": args.clim005,
            "climatology_variable": clim_var_name,
            "anomaly_file": args.anom005,
            "anomaly_variable": anom_var_name,
        },
    )

    ds_out["time"].attrs = ds_anom[time_name].attrs
    ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    encoding = {
        "tmax_005_raw": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    ds_out = filter_dataset_to_output_months(ds_out)

    log(f"Saving output: {args.out}")
    ds_out.to_netcdf(args.out, encoding=encoding)

    ds_clim.close()
    ds_anom.close()
    log("Done.")


if __name__ == "__main__":
    main()