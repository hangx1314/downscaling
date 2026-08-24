#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_interp_residual025_to_005.py

Interpolate 0.25-degree daily residual to the 0.05-degree template grid.

Inputs:
- residual_025_daily.nc
- land_mask0p05d.nc

Output:
- residual_005_interp.nc
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
from interp_chunk_tools import apply_coastal_mask_and_nearest_fill
from clim_month_tools import filter_dataset_to_output_months, filter_monthly_dataset_to_output_months


DEFAULT_RES025_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/03random_f/02exp/rf005/interim/residual_025_daily.nc"
DEFAULT_MASK005_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_masks/coastal005mask.nc"
DEFAULT_OUT_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/03random_f/02exp/rf005/interim/residual_005_interp.nc"

RES_VAR_NAME = None
MASK_VAR_NAME = None


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


def find_2d_var(ds: xr.Dataset, lat_name: str, lon_name: str, preferred_names=None):
    if preferred_names is None:
        preferred_names = []

    for name in preferred_names:
        if name in ds.data_vars:
            da = ds[name]
            if lat_name in da.dims and lon_name in da.dims:
                return name

    for name, da in ds.data_vars.items():
        if lat_name in da.dims and lon_name in da.dims:
            return name

    raise KeyError("Cannot find a 2D variable with lat/lon dims.")


def sort_da_by_latlon(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})
    if da[lon_name].values[0] > da[lon_name].values[-1]:
        da = da.isel({lon_name: slice(None, None, -1)})
    return da


def load_residual(path: str, var_name: str = None):
    log(f"Loading residual dataset: {path}")
    ds = xr.open_dataset(path)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if var_name is not None:
        if var_name not in ds.data_vars:
            raise KeyError(f"residual var '{var_name}' not found.")
        target_var = var_name
    elif RES_VAR_NAME is not None:
        if RES_VAR_NAME not in ds.data_vars:
            raise KeyError(f"RES_VAR_NAME '{RES_VAR_NAME}' not found.")
        target_var = RES_VAR_NAME
    else:
        target_var = find_3d_var(
            ds, time_name, lat_name, lon_name,
            preferred_names=["residual_tmin_025", "residual", "Band1"]
        )

    da = ds[target_var]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(time_name, lat_name, lon_name).astype(np.float32)

    return ds, da, time_name, lat_name, lon_name, target_var


def load_mask(path: str, var_name: str = None):
    log(f"Loading 0.05-degree mask: {path}")
    ds = xr.open_dataset(path)

    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if var_name is not None:
        if var_name not in ds.data_vars:
            raise KeyError(f"mask var '{var_name}' not found.")
        mask_var = var_name
    elif MASK_VAR_NAME is not None:
        if MASK_VAR_NAME not in ds.data_vars:
            raise KeyError(f"MASK_VAR_NAME '{MASK_VAR_NAME}' not found.")
        mask_var = MASK_VAR_NAME
    else:
        mask_var = find_2d_var(
            ds, lat_name, lon_name,
            preferred_names=["land_mask", "mask", "lsm", "land"]
        )

    da = ds[mask_var]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(lat_name, lon_name)

    if np.issubdtype(da.dtype, np.floating):
        mask = xr.where(np.isfinite(da) & (da > 0), 1.0, 0.0).astype(np.float32)
    else:
        mask = xr.where(da > 0, 1.0, 0.0).astype(np.float32)

    return ds, mask, lat_name, lon_name, mask_var


def main():
    parser = argparse.ArgumentParser(description="Interpolate 0.25-degree residual to 0.05-degree grid.")
    parser.add_argument("--res025", default=DEFAULT_RES025_PATH, help="Input 0.25-degree residual NetCDF")
    parser.add_argument("--mask005", default=DEFAULT_MASK005_PATH, help="Input 0.05-degree mask template NetCDF")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output 0.05-degree interpolated residual NetCDF")
    parser.add_argument("--res-var", default=None, help="Optional residual variable name")
    parser.add_argument("--mask-var", default=None, help="Optional mask variable name")
    args = parser.parse_args()

    ensure_parent_dir(args.out)

    ds_res, da_res, time_name, res_lat_name, res_lon_name, res_var_name = load_residual(
        args.res025, var_name=args.res_var
    )
    ds_mask, land_mask, mask_lat_name, mask_lon_name, mask_var_name = load_mask(
        args.mask005, var_name=args.mask_var
    )

    tgt_lat = land_mask[mask_lat_name]
    tgt_lon = land_mask[mask_lon_name]

    log("Interpolating residual from 0.25 degree to 0.05 degree using linear interpolation")
    da_linear = da_res.interp(
        {
            res_lat_name: tgt_lat.values,
            res_lon_name: tgt_lon.values,
        },
        method="linear"
    )

    log("Filling edge/remaining NaN using nearest interpolation")
    da_nearest = da_res.interp(
        {
            res_lat_name: tgt_lat.values,
            res_lon_name: tgt_lon.values,
        },
        method="nearest"
    )

    da_interp = da_linear.where(np.isfinite(da_linear), da_nearest)

    da_interp = da_interp.rename(
        {
            time_name: "time",
            res_lat_name: "lat",
            res_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")

    land_mask_std = land_mask.rename({mask_lat_name: "lat", mask_lon_name: "lon"}).transpose("lat", "lon")
    da_interp = apply_coastal_mask_and_nearest_fill(
        da_interp,
        land_mask_std,
        time_name="time",
        time_chunk=getattr(args, "time_chunk", 30),
        log_fn=log,
    )

    da_interp.name = "residual_tmin_005_interp"
    da_interp.attrs = {
        "long_name": "Daily residual interpolated from 0.25-degree to 0.05-degree grid",
        "units": da_res.attrs.get("units", ""),
        "source_file": args.res025,
        "source_variable": res_var_name,
        "interpolation_method": "linear_then_nearest_fill_then_mask_nearest",
        "target_mask_file": args.mask005,
        "target_mask_variable": mask_var_name,
    }

    ds_out = xr.Dataset(
        {"residual_tmin_005_interp": da_interp.astype(np.float32)},
        attrs={
            "title": "Interpolated daily residual on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_residual_file": args.res025,
            "source_residual_variable": res_var_name,
            "target_mask_file": args.mask005,
            "target_mask_variable": mask_var_name,
            "interpolation_method": "linear_then_nearest_fill_then_mask_nearest",
        }
    )

    ds_out["time"].attrs = ds_res[time_name].attrs
    ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    log(f"Saving output: {args.out}")
    ds_out.to_netcdf(
        args.out,
        encoding={"residual_tmin_005_interp": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    ds_res.close()
    ds_mask.close()
    log("Done.")


if __name__ == "__main__":
    main()