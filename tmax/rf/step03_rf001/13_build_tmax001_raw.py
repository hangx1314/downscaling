#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18_build_tmax001_raw.py

Build initial 0.01-degree daily Tmax by combining:
- RF-predicted monthly climatology on 0.01-degree grid
- interpolated daily anomaly on 0.01-degree grid

Inputs:
- tmax_001_clim_rf.nc
- tmax_001_anom_interp.nc

Output:
- tmax_001_raw.nc
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
from build_raw_chunk_tools import build_daily_raw_netcdf_time_chunked
from clim_month_tools import assemble_full_year_clim_on_grid, filter_dataset_to_output_months, filter_monthly_dataset_to_output_months


DEFAULT_SOURCE_CLIM = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/03random_f/02exp/rf001/interim/tmax_005_clim_monthly.nc"
DEFAULT_CLIM001_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/03random_f/02exp/rf001/interim/tmax_001_clim_rf.nc"
DEFAULT_ANOM001_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/03random_f/02exp/rf001/interim/tmax_001_anom_interp.nc"
DEFAULT_OUT_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/03random_f/02exp/rf001/interim/tmax_001_raw.nc"

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
            preferred_names=["clim_tmax_rf", "clim_tmax", "tmax_clim_rf", "Band1"]
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
    parser = argparse.ArgumentParser(description="Build initial 0.01-degree daily Tmax from monthly climatology and anomaly.")
    parser.add_argument("--clim001", default=DEFAULT_CLIM001_PATH, help="Input RF-predicted 0.01-degree monthly climatology NetCDF")
    parser.add_argument("--anom001", default=DEFAULT_ANOM001_PATH, help="Input interpolated 0.01-degree daily anomaly NetCDF")
    parser.add_argument("--source-clim", default=DEFAULT_SOURCE_CLIM, help="Source monthly climatology for non-ML months when expanding to daily")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output 0.01-degree daily raw Tmax NetCDF")
    parser.add_argument("--clim-var", default=None, help="Optional climatology variable name")
    parser.add_argument("--anom-var", default=None, help="Optional anomaly variable name")
    parser.add_argument("--time-chunk", type=int, default=30, help="Time chunk size for 0.01° raw build")
    args = parser.parse_args()

    ensure_parent_dir(args.out)

    ds_clim, _, month_name, clim_lat_name, clim_lon_name, clim_var_name = load_clim(
        args.clim001, clim_var_name=args.clim_var
    )
    clim_cube, month_coords = assemble_full_year_clim_on_grid(
        args.clim001,
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
        args.anom001, anom_var_name=args.anom_var
    )

    if not np.allclose(da_clim[clim_lat_name].values, da_anom[anom_lat_name].values):
        raise ValueError("Latitude coordinates of climatology and anomaly datasets do not match.")
    if not np.allclose(da_clim[clim_lon_name].values, da_anom[anom_lon_name].values):
        raise ValueError("Longitude coordinates of climatology and anomaly datasets do not match.")

    log(f"Building initial daily Tmax in time chunks of {args.time_chunk}")
    out_var = "tmax_001_raw"
    var_attrs = {
        "long_name": "Initial daily Tmax on 0.01-degree grid before residual correction",
        "units": da_clim.attrs.get("units", da_anom.attrs.get("units", "")),
        "definition": "monthly_climatology + interpolated_daily_anomaly",
        "climatology_file": args.clim001,
        "climatology_variable": clim_var_name,
        "anomaly_file": args.anom001,
        "anomaly_variable": anom_var_name,
    }
    ds_attrs = {
        "title": "Initial daily Tmax on 0.01-degree grid",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "climatology_file": args.clim001,
        "climatology_variable": clim_var_name,
        "anomaly_file": args.anom001,
        "anomaly_variable": anom_var_name,
    }
    time_attrs = ds_anom[time_name].attrs
    lat_attrs = {"long_name": "latitude", "units": "degrees_north"}
    lon_attrs = {"long_name": "longitude", "units": "degrees_east"}
    encoding = {
        out_var: {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    build_daily_raw_netcdf_time_chunked(
        da_clim=da_clim,
        month_name=month_name,
        lat_name=clim_lat_name,
        lon_name=clim_lon_name,
        da_anom=da_anom,
        time_name=time_name,
        out_path=args.out,
        out_var=out_var,
        var_attrs=var_attrs,
        ds_attrs=ds_attrs,
        time_attrs=time_attrs,
        lat_attrs=lat_attrs,
        lon_attrs=lon_attrs,
        encoding=encoding,
        time_chunk=args.time_chunk,
        log_fn=log,
    )
    log(f"Saved output: {args.out}")

    ds_clim.close()
    ds_anom.close()
    log("Done.")


if __name__ == "__main__":
    main()