#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_compute_residual025.py

Aggregate 0.05-degree daily raw Tmax back to the 0.25-degree grid,
then compute daily residual on the 0.25-degree grid.

Inputs:
- tmax_005_raw.nc
- CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc

Outputs:
- agg025_from_tmax005_raw.nc
- residual_025_daily.nc
"""

import os
os.environ["PROJ_LIB"] = "/public/home/ggao001/anaconda3/envs/cn/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
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


DEFAULT_RAW005_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/tmax_005_raw.nc"
DEFAULT_TMAX025_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc"
DEFAULT_AGG_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/agg025_from_tmax005_raw.nc"
DEFAULT_RES_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb005/interim/residual_025_daily.nc"

RAW_VAR_NAME = None
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


def compute_grid_edges(coords: np.ndarray, outer_mode: str = "clip"):
    res = float(np.median(np.abs(np.diff(coords))))
    edges = np.empty(coords.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])

    if outer_mode == "half":
        edges[0] = coords[0] - 0.5 * res
        edges[-1] = coords[-1] + 0.5 * res
    elif outer_mode == "clip":
        edges[0] = coords[0]
        edges[-1] = coords[-1]
    else:
        raise ValueError(f"Unsupported outer_mode: {outer_mode}")

    return edges


def load_raw005(path: str, var_name: str = None):
    log(f"Loading 0.05-degree raw Tmax: {path}")
    ds = xr.open_dataset(path)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if var_name is not None:
        if var_name not in ds.data_vars:
            raise KeyError(f"raw var '{var_name}' not found.")
        target_var = var_name
    elif RAW_VAR_NAME is not None:
        if RAW_VAR_NAME not in ds.data_vars:
            raise KeyError(f"RAW_VAR_NAME '{RAW_VAR_NAME}' not found.")
        target_var = RAW_VAR_NAME
    else:
        target_var = find_3d_var(
            ds, time_name, lat_name, lon_name,
            preferred_names=["tmax_005_raw", "tmax", "Band1"]
        )

    da = ds[target_var]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(time_name, lat_name, lon_name).astype(np.float32)

    return ds, da, time_name, lat_name, lon_name, target_var


def load_target025(path: str, var_name: str = None):
    log(f"Loading 0.25-degree target Tmax: {path}")
    ds = xr.open_dataset(path)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    if var_name is not None:
        if var_name not in ds.data_vars:
            raise KeyError(f"target var '{var_name}' not found.")
        target_var = var_name
    elif TARGET_VAR_NAME is not None:
        if TARGET_VAR_NAME not in ds.data_vars:
            raise KeyError(f"TARGET_VAR_NAME '{TARGET_VAR_NAME}' not found.")
        target_var = TARGET_VAR_NAME
    else:
        target_var = find_3d_var(
            ds, time_name, lat_name, lon_name,
            preferred_names=["tmax", "Tmax", "tasmax", "Band1"]
        )

    da = ds[target_var]
    da = sort_da_by_latlon(da, lat_name, lon_name)
    da = da.transpose(time_name, lat_name, lon_name).astype(np.float32)

    return ds, da, time_name, lat_name, lon_name, target_var


def build_subcell_mapping(src_lat: np.ndarray, src_lon: np.ndarray, tgt_lat: np.ndarray, tgt_lon: np.ndarray):
    nlat_t = tgt_lat.size
    nlon_t = tgt_lon.size
    ncell_t = nlat_t * nlon_t

    lat_edges = compute_grid_edges(tgt_lat, outer_mode="clip")
    lon_edges = compute_grid_edges(tgt_lon, outer_mode="clip")

    src_lat2d, src_lon2d = np.meshgrid(src_lat, src_lon, indexing="ij")
    row_idx = np.searchsorted(lat_edges, src_lat2d, side="right") - 1
    col_idx = np.searchsorted(lon_edges, src_lon2d, side="right") - 1

    inside = (
        (row_idx >= 0) & (row_idx < nlat_t) &
        (col_idx >= 0) & (col_idx < nlon_t)
    )

    flat_target = np.full(src_lat2d.shape, -1, dtype=np.int64)
    flat_target[inside] = (row_idx[inside] * nlon_t + col_idx[inside]).astype(np.int64)

    return flat_target.reshape(-1), ncell_t


def aggregate_chunk_to_target(chunk_3d: np.ndarray, flat_target: np.ndarray, nlat_t: int, nlon_t: int):
    """
    chunk_3d: [ntime_chunk, nlat_src, nlon_src]
    returns:
    - agg_chunk: [ntime_chunk, nlat_t, nlon_t]
    """
    ntime_chunk = chunk_3d.shape[0]
    ncell_t = nlat_t * nlon_t
    agg_out = np.full((ntime_chunk, ncell_t), np.nan, dtype=np.float32)

    src_flat = chunk_3d.reshape(ntime_chunk, -1)

    target_valid = flat_target >= 0
    target_ids_all = flat_target[target_valid]

    for it in range(ntime_chunk):
        vals = src_flat[it]
        valid = target_valid & np.isfinite(vals)

        if not np.any(valid):
            continue

        target_ids = flat_target[valid]
        values = vals[valid].astype(np.float64)

        counts = np.bincount(target_ids, minlength=ncell_t).astype(np.float64)
        sums = np.bincount(target_ids, weights=values, minlength=ncell_t)

        mean = np.full(ncell_t, np.nan, dtype=np.float64)
        has = counts > 0
        mean[has] = sums[has] / counts[has]

        agg_out[it, :] = mean.astype(np.float32)

    return agg_out.reshape(ntime_chunk, nlat_t, nlon_t)


def main():
    parser = argparse.ArgumentParser(description="Aggregate 0.05-degree raw Tmax to 0.25-degree and compute residual.")
    parser.add_argument("--raw005", default=DEFAULT_RAW005_PATH, help="Input 0.05-degree raw Tmax NetCDF")
    parser.add_argument("--target025", default=DEFAULT_TMAX025_PATH, help="Input original 0.25-degree Tmax NetCDF")
    parser.add_argument("--agg-out", default=DEFAULT_AGG_OUT, help="Output aggregated 0.25-degree NetCDF")
    parser.add_argument("--res-out", default=DEFAULT_RES_OUT, help="Output residual 0.25-degree NetCDF")
    parser.add_argument("--raw-var", default=None, help="Optional 0.05-degree raw Tmax variable name")
    parser.add_argument("--target-var", default=None, help="Optional 0.25-degree target Tmax variable name")
    parser.add_argument("--time-chunk", type=int, default=180, help="Time chunk size for aggregation")
    args = parser.parse_args()

    ensure_parent_dir(args.agg_out)
    ensure_parent_dir(args.res_out)

    ds_raw, da_raw, raw_time_name, raw_lat_name, raw_lon_name, raw_var_name = load_raw005(
        args.raw005, var_name=args.raw_var
    )
    ds_tgt, da_tgt, tgt_time_name, tgt_lat_name, tgt_lon_name, tgt_var_name = load_target025(
        args.target025, var_name=args.target_var
    )

    raw_time = da_raw[raw_time_name].values
    tgt_time = da_tgt[tgt_time_name].values
    if not np.array_equal(raw_time, tgt_time):
        if not np.all(np.isin(raw_time, tgt_time)):
            raise ValueError(
                "Raw time coordinates are not a subset of target time coordinates."
            )
        log("Subsetting target to raw time coordinates (warm-season intermediate product).")
        da_tgt = da_tgt.sel({tgt_time_name: raw_time})
        tgt_time = da_tgt[tgt_time_name].values


    src_lat = da_raw[raw_lat_name].values
    src_lon = da_raw[raw_lon_name].values
    tgt_lat = da_tgt[tgt_lat_name].values
    tgt_lon = da_tgt[tgt_lon_name].values

    ntime = da_raw.sizes[raw_time_name]
    nlat_t = tgt_lat.size
    nlon_t = tgt_lon.size

    log("Building 0.05 -> 0.25 subcell mapping")
    flat_target, _ = build_subcell_mapping(src_lat, src_lon, tgt_lat, tgt_lon)

    agg_cube = np.full((ntime, nlat_t, nlon_t), np.nan, dtype=np.float32)

    chunk_size = int(args.time_chunk)
    log(f"Aggregating in time chunks of {chunk_size}")

    for start in range(0, ntime, chunk_size):
        end = min(ntime, start + chunk_size)
        log(f"Aggregating time slice {start}:{end}")

        chunk = da_raw.isel({raw_time_name: slice(start, end)}).values.astype(np.float32)
        agg_chunk = aggregate_chunk_to_target(chunk, flat_target, nlat_t, nlon_t)
        agg_cube[start:end, :, :] = agg_chunk

    agg_da = xr.DataArray(
        agg_cube,
        coords={
            "time": raw_time,
            "lat": tgt_lat,
            "lon": tgt_lon,
        },
        dims=("time", "lat", "lon"),
        name="agg_tmax_025_from_005_raw",
        attrs={
            "long_name": "0.05-degree raw Tmax aggregated back to 0.25-degree grid",
            "units": da_raw.attrs.get("units", ""),
            "source_file": args.raw005,
            "source_variable": raw_var_name,
            "aggregation_method": "mean_over_valid_005_subcells",
            "grid_edge_mode": "clip",
        }
    )

    tgt_std = da_tgt.rename(
        {
            tgt_time_name: "time",
            tgt_lat_name: "lat",
            tgt_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")

    res_da = (tgt_std - agg_da).astype(np.float32)
    res_da.name = "residual_tmax_025"
    res_da.attrs = {
        "long_name": "Daily residual on 0.25-degree grid",
        "units": da_tgt.attrs.get("units", ""),
        "definition": "original_tmax_025 - aggregated_tmax_from_005_raw",
        "target_file": args.target025,
        "target_variable": tgt_var_name,
        "aggregated_file": args.agg_out,
        "grid_edge_mode": "clip",
    }

    ds_agg = xr.Dataset(
        {"agg_tmax_025_from_005_raw": agg_da},
        attrs={
            "title": "Aggregated 0.25-degree Tmax from 0.05-degree raw Tmax",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_raw005_file": args.raw005,
            "source_raw005_variable": raw_var_name,
            "target025_file": args.target025,
            "target025_variable": tgt_var_name,
            "aggregation_method": "mean_over_valid_005_subcells",
            "grid_edge_mode": "clip",
        }
    )

    ds_res = xr.Dataset(
        {"residual_tmax_025": res_da},
        attrs={
            "title": "Daily 0.25-degree residual from raw 0.05-degree Tmax",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target025_file": args.target025,
            "target025_variable": tgt_var_name,
            "aggregated025_file": args.agg_out,
            "aggregated025_variable": "agg_tmax_025_from_005_raw",
            "definition": "target025 - aggregated025",
            "grid_edge_mode": "clip",
        }
    )

    for ds_out in [ds_agg, ds_res]:
        ds_out["time"].attrs = ds_tgt[tgt_time_name].attrs
        ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
        ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    ds_agg = filter_dataset_to_output_months(ds_agg)
    ds_res = filter_dataset_to_output_months(ds_res)

    log(f"Saving aggregated file: {args.agg_out}")
    ds_agg.to_netcdf(
        args.agg_out,
        encoding={"agg_tmax_025_from_005_raw": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    log(f"Saving residual file: {args.res_out}")
    ds_res.to_netcdf(
        args.res_out,
        encoding={"residual_tmax_025": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    ds_raw.close()
    ds_tgt.close()
    log("Done.")


if __name__ == "__main__":
    main()