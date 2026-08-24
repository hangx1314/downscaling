#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_consistency_correct_005.py

Apply strict 0.25-degree consistency correction to the residual-corrected
0.05-degree daily Tmax field.

Steps:
1) Aggregate tmax_005_corr back to 0.25-degree grid
2) Compute delta_025 = target025 - aggregated025
3) Distribute delta_025 equally to all valid 0.05-degree subcells inside
   each 0.25-degree parent cell
4) Save delta_025 and final tmax_005_final

Inputs:
- tmax_005_corr.nc
- CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc

Outputs:
- delta_025_consistency.nc
- tmax_005_final.nc
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
from clim_month_tools import filter_dataset_to_output_months
from mask_fill import fill_nan_inside_mask_chunk
from mask_loader import load_land_mask_2d


DEFAULT_MASK005_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_masks/coastal005mask.nc"
DEFAULT_CORR005_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/05unet/02exp/unet005/interim/tmax_005_corr.nc"
DEFAULT_TMAX025_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc"
DEFAULT_DELTA_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/05unet/02exp/unet005/interim/delta_025_consistency.nc"
DEFAULT_FINAL_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/05unet/02exp/unet005/outputs/tmax_005_final.nc"

CORR_VAR_NAME = None
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
    ntime_chunk = chunk_3d.shape[0]
    ncell_t = nlat_t * nlon_t
    agg_out = np.full((ntime_chunk, ncell_t), np.nan, dtype=np.float32)

    src_flat = chunk_3d.reshape(ntime_chunk, -1)
    target_valid = flat_target >= 0

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


def distribute_delta_chunk(chunk_3d: np.ndarray, delta_3d: np.ndarray, flat_target: np.ndarray, nlat_t: int, nlon_t: int):
    """
    chunk_3d: [ntime_chunk, nlat_src, nlon_src]
    delta_3d: [ntime_chunk, nlat_t, nlon_t]
    """
    ntime_chunk, nlat_s, nlon_s = chunk_3d.shape
    src_flat = chunk_3d.reshape(ntime_chunk, -1)
    delta_flat = delta_3d.reshape(ntime_chunk, nlat_t * nlon_t)

    out_flat = np.array(src_flat, copy=True)
    target_valid = flat_target >= 0

    for it in range(ntime_chunk):
        vals = src_flat[it]
        valid = target_valid & np.isfinite(vals)

        if not np.any(valid):
            continue

        target_ids = flat_target[valid]
        add_vals = delta_flat[it, target_ids].astype(np.float32)
        out_flat[it, valid] = vals[valid] + add_vals

    return out_flat.reshape(ntime_chunk, nlat_s, nlon_s)


def main():
    parser = argparse.ArgumentParser(description="Apply strict 0.25-degree consistency correction to 0.05-degree Tmax.")
    parser.add_argument("--corr005", default=DEFAULT_CORR005_PATH, help="Input residual-corrected 0.05-degree Tmax NetCDF")
    parser.add_argument("--target025", default=DEFAULT_TMAX025_PATH, help="Input original 0.25-degree Tmax NetCDF")
    parser.add_argument("--delta-out", default=DEFAULT_DELTA_OUT, help="Output 0.25-degree consistency delta NetCDF")
    parser.add_argument("--final-out", default=DEFAULT_FINAL_OUT, help="Output final 0.05-degree Tmax NetCDF")
    parser.add_argument("--corr-var", default=None, help="Optional corrected 0.05-degree variable name")
    parser.add_argument("--target-var", default=None, help="Optional target 0.25-degree variable name")
    parser.add_argument("--time-chunk", type=int, default=180, help="Time chunk size")
    parser.add_argument("--mask005", default=DEFAULT_MASK005_PATH, help="Coastal mask NetCDF for nearest fill")
    args = parser.parse_args()

    ensure_parent_dir(args.delta_out)
    ensure_parent_dir(args.final_out)

    ds_corr, da_corr, corr_time_name, corr_lat_name, corr_lon_name, corr_var_name = load_field(
        args.corr005,
        default_names=["tmax_005_corr", "tmax", "Band1"],
        override_var_name=args.corr_var,
        fallback_global=CORR_VAR_NAME,
    )

    ds_tgt, da_tgt, tgt_time_name, tgt_lat_name, tgt_lon_name, tgt_var_name = load_field(
        args.target025,
        default_names=["tmax", "Tmax", "tasmax", "Band1"],
        override_var_name=args.target_var,
        fallback_global=TARGET_VAR_NAME,
    )

    corr_time = da_corr[corr_time_name].values
    tgt_time = da_tgt[tgt_time_name].values
    if not np.array_equal(corr_time, tgt_time):
        if not np.all(np.isin(corr_time, tgt_time)):
            raise ValueError(
                "Corrected time coordinates are not a subset of target time coordinates."
            )
        log("Subsetting target to corrected time coordinates (warm-season intermediate product).")
        da_tgt = da_tgt.sel({tgt_time_name: corr_time})
        tgt_time = da_tgt[tgt_time_name].values


    src_lat = da_corr[corr_lat_name].values
    src_lon = da_corr[corr_lon_name].values
    tgt_lat = da_tgt[tgt_lat_name].values
    tgt_lon = da_tgt[tgt_lon_name].values

    ntime = da_corr.sizes[corr_time_name]
    nlat_s = src_lat.size
    nlon_s = src_lon.size
    nlat_t = tgt_lat.size
    nlon_t = tgt_lon.size

    log("Building 0.05 -> 0.25 subcell mapping")
    flat_target, _ = build_subcell_mapping(src_lat, src_lon, tgt_lat, tgt_lon)

    corr_std = da_corr.rename(
        {
            corr_time_name: "time",
            corr_lat_name: "lat",
            corr_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")

    tgt_std = da_tgt.rename(
        {
            tgt_time_name: "time",
            tgt_lat_name: "lat",
            tgt_lon_name: "lon",
        }
    ).transpose("time", "lat", "lon")

    delta_cube = np.full((ntime, nlat_t, nlon_t), np.nan, dtype=np.float32)
    final_cube = np.full((ntime, nlat_s, nlon_s), np.nan, dtype=np.float32)

    chunk_size = int(args.time_chunk)

    mask2d = load_land_mask_2d(args.mask005, src_lat, src_lon)
    log(f"Coastal mask cells: {int(mask2d.sum())}")
    log(f"Processing in time chunks of {chunk_size}")

    for start in range(0, ntime, chunk_size):
        end = min(ntime, start + chunk_size)
        log(f"Consistency correction time slice {start}:{end}")

        corr_chunk = corr_std.isel(time=slice(start, end)).values.astype(np.float32)
        tgt_chunk = tgt_std.isel(time=slice(start, end)).values.astype(np.float32)

        agg_chunk = aggregate_chunk_to_target(corr_chunk, flat_target, nlat_t, nlon_t)
        delta_chunk = (tgt_chunk - agg_chunk).astype(np.float32)
        final_chunk = distribute_delta_chunk(corr_chunk, delta_chunk, flat_target, nlat_t, nlon_t)
        final_chunk = fill_nan_inside_mask_chunk(final_chunk, mask2d)

        delta_cube[start:end, :, :] = delta_chunk
        final_cube[start:end, :, :] = final_chunk

    delta_da = xr.DataArray(
        delta_cube,
        coords={
            "time": tgt_std["time"].values,
            "lat": tgt_std["lat"].values,
            "lon": tgt_std["lon"].values,
        },
        dims=("time", "lat", "lon"),
        name="delta_025_consistency",
        attrs={
            "long_name": "Consistency delta on 0.25-degree grid",
            "units": da_tgt.attrs.get("units", ""),
            "definition": "target_tmax_025 - aggregated_tmax_025_from_tmax_005_corr",
            "source_corr_file": args.corr005,
            "source_corr_variable": corr_var_name,
            "target_file": args.target025,
            "target_variable": tgt_var_name,
            "grid_edge_mode": "clip",
        }
    )

    final_da = xr.DataArray(
        final_cube,
        coords={
            "time": corr_std["time"].values,
            "lat": corr_std["lat"].values,
            "lon": corr_std["lon"].values,
        },
        dims=("time", "lat", "lon"),
        name="tmax_005_final",
        attrs={
            "long_name": "Final daily Tmax on 0.05-degree grid after residual and consistency correction",
            "units": da_corr.attrs.get("units", da_tgt.attrs.get("units", "")),
            "definition": "tmax_005_corr + distributed_delta_025_consistency",
            "source_corr_file": args.corr005,
            "source_corr_variable": corr_var_name,
            "target025_file": args.target025,
            "target025_variable": tgt_var_name,
            "distribution_method": "equal_addition_to_all_valid_005_subcells_within_each_025_parent",
            "grid_edge_mode": "clip",
        }
    )

    ds_delta = xr.Dataset(
        {"delta_025_consistency": delta_da},
        attrs={
            "title": "0.25-degree consistency delta for 0.05-degree Tmax",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "corr005_file": args.corr005,
            "corr005_variable": corr_var_name,
            "target025_file": args.target025,
            "target025_variable": tgt_var_name,
            "grid_edge_mode": "clip",
        }
    )

    ds_final = xr.Dataset(
        {"tmax_005_final": final_da},
        attrs={
            "title": "Final daily Tmax on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "corr005_file": args.corr005,
            "corr005_variable": corr_var_name,
            "target025_file": args.target025,
            "target025_variable": tgt_var_name,
            "grid_edge_mode": "clip",
        }
    )

    ds_delta = filter_dataset_to_output_months(ds_delta)
    ds_final = filter_dataset_to_output_months(ds_final)

    for ds_out in [ds_delta, ds_final]:
        ds_out["time"].attrs = ds_corr[corr_time_name].attrs
        ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
        ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    log(f"Saving delta file: {args.delta_out}")
    ds_delta.to_netcdf(
        args.delta_out,
        encoding={"delta_025_consistency": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    log(f"Saving final file: {args.final_out}")
    ds_final.to_netcdf(
        args.final_out,
        encoding={"tmax_005_final": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    ds_corr.close()
    ds_tgt.close()
    log("Done.")


if __name__ == "__main__":
    main()