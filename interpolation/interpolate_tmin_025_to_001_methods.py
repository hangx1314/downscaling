#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interpolate_tmin_025_to_001_methods.py

Interpolate CN_YANHAI_DOWN Tmin from 0.25 degree to high-resolution grid constrained by coastal001mask.

Methods:
1. Bilinear
2. IDW
3. Local ordinary kriging

Input:
    ./data/coastal_daily_025/CN05.1_Tmin_1961_2025_daily_025x025_coastal.nc

Mask constraint:
    ./data/coastal_masks/coastal001mask.nc

Outputs:
    ./outputs/interpolation/tmin/01bilinear/tmin_001_bilinear.nc
    ./outputs/interpolation/tmin/02kriging/tmin_001_kriging.nc
    ./outputs/interpolation/tmin/03IDW/tmin_001_idw.nc

Notes:
    - Bilinear, IDW, and kriging weights are precomputed once.
    - Daily fields are processed by time chunks.
    - Output is restricted by the target mask.
    - For bilinear interpolation, NaN cells inside the target mask are filled by nearest valid source cell (cKDTree).
"""

import os
import argparse
from datetime import datetime
from typing import List, Tuple

import numpy as np
import xarray as xr

try:
    import netCDF4 as nc
except Exception as exc:
    nc = None
    NETCDF4_ERROR = exc
else:
    NETCDF4_ERROR = None

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import cKDTree
except Exception as exc:
    RegularGridInterpolator = None
    cKDTree = None
    SCIPY_ERROR = exc
else:
    SCIPY_ERROR = None

try:
    from joblib import Parallel, delayed
except Exception as exc:
    Parallel = None
    delayed = None
    JOBLIB_ERROR = exc
else:
    JOBLIB_ERROR = None


# =========================================================
# Default paths
# =========================================================
DEFAULT_INPUT = "./data/coastal_daily_025/CN05.1_Tmin_1961_2025_daily_025x025_coastal.nc"
DEFAULT_MASK = "./data/coastal_masks/coastal001mask.nc"

DEFAULT_BILINEAR_OUT = "./outputs/interpolation/tmin/01bilinear/tmin_001_bilinear.nc"
DEFAULT_KRIGING_OUT = "./outputs/interpolation/tmin/02kriging/tmin_001_kriging.nc"
DEFAULT_IDW_OUT = "./outputs/interpolation/tmin/03IDW/tmin_001_idw.nc"

DEFAULT_VAR = "tmin"


# =========================================================
# Basic helpers
# =========================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def find_coord_name(ds: xr.Dataset, candidates: List[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Cannot find coordinate among candidates: {candidates}")


def find_3d_var(ds: xr.Dataset, time_name: str, lat_name: str, lon_name: str, preferred: List[str]) -> str:
    for name in preferred:
        if name in ds.data_vars:
            da = ds[name]
            if time_name in da.dims and lat_name in da.dims and lon_name in da.dims:
                return name

    for name, da in ds.data_vars.items():
        if time_name in da.dims and lat_name in da.dims and lon_name in da.dims:
            return name

    raise KeyError("Cannot find a 3D time-lat-lon variable.")


def find_2d_var(ds: xr.Dataset, lat_name: str, lon_name: str, preferred: List[str]) -> str:
    for name in preferred:
        if name in ds.data_vars:
            da = ds[name]
            if lat_name in da.dims and lon_name in da.dims:
                return name

    for name, da in ds.data_vars.items():
        if lat_name in da.dims and lon_name in da.dims:
            return name

    raise KeyError("Cannot find a 2D lat-lon variable.")


def sort_ds_by_latlon(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.Dataset:
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})

    if ds[lon_name].values[0] > ds[lon_name].values[-1]:
        ds = ds.isel({lon_name: slice(None, None, -1)})

    return ds


def datetime64_to_days(time_values: np.ndarray, origin: str = "1961-01-01") -> Tuple[np.ndarray, str]:
    if not np.issubdtype(time_values.dtype, np.datetime64):
        return np.asarray(time_values), ""

    base = np.datetime64(origin)
    days = (time_values.astype("datetime64[D]") - base).astype(np.int32)
    units = f"days since {origin} 00:00:00"
    return days, units


# =========================================================
# Data loading
# =========================================================
def load_time_lat_lon_field(path: str, preferred_names: List[str], var_name: str = None):
    log(f"Loading source file: {path}")

    ds = xr.open_dataset(path, decode_times=True)

    time_name = find_coord_name(ds, ["time", "date"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    ds = sort_ds_by_latlon(ds, lat_name, lon_name)

    if var_name is None:
        var_name = find_3d_var(ds, time_name, lat_name, lon_name, preferred_names)

    if var_name not in ds.data_vars:
        var_name = find_3d_var(ds, time_name, lat_name, lon_name, preferred_names)

    da = ds[var_name].transpose(time_name, lat_name, lon_name).astype(np.float32)

    time_values = da[time_name].values
    lat = da[lat_name].values.astype(np.float64)
    lon = da[lon_name].values.astype(np.float64)

    log(f"Source variable: {var_name}")
    log(f"Source shape: time={time_values.size}, lat={lat.size}, lon={lon.size}")

    return ds, da, time_name, lat_name, lon_name, var_name, time_values, lat, lon


def load_mask(mask_path: str):
    log(f"Loading mask file: {mask_path}")

    ds = xr.open_dataset(mask_path)

    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    ds = sort_ds_by_latlon(ds, lat_name, lon_name)

    var_name = find_2d_var(ds, lat_name, lon_name, ["land_mask", "mask", "lsm", "land", "Band1"])
    da = ds[var_name].transpose(lat_name, lon_name)

    mask = np.isfinite(da.values) & (da.values > 0)
    lat = ds[lat_name].values.astype(np.float64)
    lon = ds[lon_name].values.astype(np.float64)

    ds.close()

    log(f"Mask variable: {var_name}")
    log(f"Mask shape: lat={lat.size}, lon={lon.size}, valid={int(mask.sum())}")

    return mask.astype(bool), lat, lon


# =========================================================
# Target grid and mask
# =========================================================
def build_target_grid_from_mask(mask: np.ndarray, mask_lat: np.ndarray, mask_lon: np.ndarray):
    """
    Use the coordinate grid of coastal001mask directly as target grid.
    """
    valid_rows, valid_cols = np.where(mask)

    if valid_rows.size == 0:
        raise ValueError("Mask has no valid cells.")

    lat = mask_lat.astype(np.float64)
    lon = mask_lon.astype(np.float64)

    log(f"Target grid from mask: lat={lat.size}, lon={lon.size}")
    log(f"Target lat range: {lat[0]:.6f} to {lat[-1]:.6f}")
    log(f"Target lon range: {lon[0]:.6f} to {lon[-1]:.6f}")

    return lat, lon


def save_target_mask(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray, out_path: str):
    mask_out = os.path.join(os.path.dirname(out_path), "target_mask.nc")
    ensure_parent_dir(mask_out)

    ds = xr.Dataset(
        {"mask": (("lat", "lon"), mask.astype(np.int8))},
        coords={"lat": lat, "lon": lon},
        attrs={
            "title": "Target mask used by interpolation",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    ds["mask"].attrs = {
        "long_name": "target valid mask",
        "definition": "copied from coastal001mask valid cells",
    }

    ds.to_netcdf(mask_out, encoding={"mask": {"zlib": True, "complevel": 4, "dtype": "i1"}})
    ds.close()

    log(f"Saved target mask: {mask_out}")


# =========================================================
# NetCDF output
# =========================================================
def create_time_lat_lon_netcdf(
    out_path: str,
    var_name: str,
    time_values: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    attrs: dict,
    var_attrs: dict,
    time_chunk: int,
    complevel: int = 4,
):
    if nc is None:
        raise ImportError(f"netCDF4 is required. Original error: {NETCDF4_ERROR}")

    ensure_parent_dir(out_path)

    if os.path.exists(out_path):
        os.remove(out_path)

    ds = nc.Dataset(out_path, "w", format="NETCDF4")

    ds.createDimension("time", len(time_values))
    ds.createDimension("lat", len(lat_values))
    ds.createDimension("lon", len(lon_values))

    time_var = ds.createVariable("time", "i4", ("time",))
    time_num, time_units = datetime64_to_days(time_values)
    time_var[:] = time_num

    if time_units:
        time_var.units = time_units
        time_var.calendar = "standard"

    time_var.long_name = "time"

    lat_var = ds.createVariable("lat", "f8", ("lat",))
    lon_var = ds.createVariable("lon", "f8", ("lon",))

    lat_var[:] = lat_values.astype(np.float64)
    lon_var[:] = lon_values.astype(np.float64)

    lat_var.long_name = "latitude"
    lat_var.units = "degrees_north"

    lon_var.long_name = "longitude"
    lon_var.units = "degrees_east"

    chunksizes = (
        min(int(time_chunk), len(time_values)),
        min(128, len(lat_values)),
        min(128, len(lon_values)),
    )

    data_var = ds.createVariable(
        var_name,
        "f4",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=int(complevel),
        fill_value=np.float32(np.nan),
        chunksizes=chunksizes,
    )

    for k, v in var_attrs.items():
        try:
            setattr(data_var, k, v)
        except Exception:
            setattr(data_var, k, str(v))

    for k, v in attrs.items():
        try:
            setattr(ds, k, v)
        except Exception:
            setattr(ds, k, str(v))

    return ds, data_var


# =========================================================
# Geometry helpers
# =========================================================
def make_source_points(source_lat: np.ndarray, source_lon: np.ndarray):
    lat2d, lon2d = np.meshgrid(source_lat, source_lon, indexing="ij")
    points = np.column_stack([lat2d.reshape(-1), lon2d.reshape(-1)]).astype(np.float64)
    return points


def make_target_points(target_lat: np.ndarray, target_lon: np.ndarray, target_mask: np.ndarray):
    lat2d, lon2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    all_points = np.column_stack([lat2d.reshape(-1), lon2d.reshape(-1)]).astype(np.float64)
    valid_flat = target_mask.reshape(-1)
    valid_points = all_points[valid_flat]
    return all_points, valid_points, valid_flat


# =========================================================
# Bilinear
# =========================================================
def make_bilinear_grid_points(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
):
    lat2d, lon2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    target_points = np.column_stack([lat2d.reshape(-1), lon2d.reshape(-1)]).astype(np.float64)

    src_lat2d, src_lon2d = np.meshgrid(source_lat, source_lon, indexing="ij")
    source_points = np.column_stack([src_lat2d.reshape(-1), src_lon2d.reshape(-1)]).astype(np.float64)

    return target_points, source_points


def fill_bilinear_gaps_with_nearest(
    pred: np.ndarray,
    arr2d: np.ndarray,
    target_mask: np.ndarray,
    target_points: np.ndarray,
    source_points: np.ndarray,
):
    if cKDTree is None:
        raise ImportError(f"scipy is required. Original error: {SCIPY_ERROR}")

    nan_inside = target_mask & (~np.isfinite(pred))
    if not np.any(nan_inside):
        return pred

    values_flat = arr2d.reshape(-1).astype(np.float32)
    finite_mask = np.isfinite(values_flat)
    if not np.any(finite_mask):
        return pred

    tree = cKDTree(source_points[finite_mask])
    valid_values = values_flat[finite_mask]
    gap_points = target_points[nan_inside.reshape(-1)]

    _, nearest_idx = tree.query(gap_points, k=1)

    pred_flat = pred.reshape(-1)
    pred_flat[nan_inside.reshape(-1)] = valid_values[nearest_idx]
    return pred_flat.reshape(pred.shape).astype(np.float32)


def bilinear_one_day(
    arr2d: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
    target_points: np.ndarray = None,
    source_points: np.ndarray = None,
):
    if RegularGridInterpolator is None:
        raise ImportError(f"scipy is required. Original error: {SCIPY_ERROR}")

    if target_points is None or source_points is None:
        target_points, source_points = make_bilinear_grid_points(
            source_lat,
            source_lon,
            target_lat,
            target_lon,
        )

    linear_interp = RegularGridInterpolator(
        (source_lat, source_lon),
        arr2d.astype(np.float32),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pred = linear_interp(target_points).reshape(target_lat.size, target_lon.size).astype(np.float32)
    pred[~target_mask] = np.nan

    pred = fill_bilinear_gaps_with_nearest(
        pred,
        arr2d,
        target_mask,
        target_points,
        source_points,
    )
    pred[~target_mask] = np.nan

    return pred


def run_bilinear(
    da_src,
    time_name: str,
    time_values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
    out_path: str,
    out_var_name: str,
    time_chunk: int,
    source_file: str = "",
    mask_file: str = "",
):
    log("=" * 100)
    log("Running bilinear interpolation")
    log("=" * 100)

    target_points, source_points = make_bilinear_grid_points(
        source_lat,
        source_lon,
        target_lat,
        target_lon,
    )

    src_meta = source_file or DEFAULT_INPUT
    mask_meta = mask_file or DEFAULT_MASK
    is_tmin = str(out_var_name).lower() == "tmin"
    title = (
        "Daily Tmin interpolated from 0.25 degree using bilinear interpolation"
        if is_tmin
        else "Daily Tmax interpolated from 0.25 degree using bilinear interpolation"
    )
    long_name = "Daily minimum temperature" if is_tmin else "Daily maximum temperature"

    ds_out, var_out = create_time_lat_lon_netcdf(
        out_path=out_path,
        var_name=out_var_name,
        time_values=time_values,
        lat_values=target_lat,
        lon_values=target_lon,
        attrs={
            "title": title,
            "method": "bilinear",
            "source_file": src_meta,
            "constraint_mask": mask_meta,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        var_attrs={
            "long_name": long_name,
            "method": "bilinear interpolation with cKDTree nearest-neighbor filling for NaN cells inside target mask",
            "units": da_src.attrs.get("units", ""),
        },
        time_chunk=time_chunk,
    )

    try:
        ntime = da_src.sizes[time_name]

        for start in range(0, ntime, time_chunk):
            end = min(ntime, start + time_chunk)
            log(f"Bilinear processing time slice {start}:{end}")

            src_chunk = da_src.isel({time_name: slice(start, end)}).values.astype(np.float32)
            out_chunk = np.full((end - start, target_lat.size, target_lon.size), np.nan, dtype=np.float32)

            for i in range(end - start):
                out_chunk[i] = bilinear_one_day(
                    src_chunk[i],
                    source_lat,
                    source_lon,
                    target_lat,
                    target_lon,
                    target_mask,
                    target_points,
                    source_points,
                )

            var_out[start:end, :, :] = out_chunk.astype(np.float32)

    finally:
        ds_out.close()

    save_target_mask(target_mask, target_lat, target_lon, out_path)
    log(f"Saved bilinear output: {out_path}")


# =========================================================
# IDW
# =========================================================
def build_idw_weights(
    src_points: np.ndarray,
    target_points: np.ndarray,
    k: int,
    power: float,
    eps: float,
    n_jobs: int,
):
    if cKDTree is None:
        raise ImportError(f"scipy is required. Original error: {SCIPY_ERROR}")

    log(f"Building IDW weights: k={k}, power={power}")

    tree = cKDTree(src_points)
    dist, idx = tree.query(target_points, k=int(k), workers=int(n_jobs))

    if int(k) == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    weights = 1.0 / np.maximum(dist, eps) ** float(power)

    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / np.maximum(weights_sum, eps)

    return idx.astype(np.int64), weights.astype(np.float32)


def apply_weights_one_day(
    values_flat: np.ndarray,
    idx: np.ndarray,
    weights: np.ndarray,
    target_valid_flat: np.ndarray,
    out_shape: Tuple[int, int],
):
    valid_values = values_flat.astype(np.float32)

    pred_valid = np.sum(valid_values[idx] * weights, axis=1).astype(np.float32)

    out_flat = np.full(out_shape[0] * out_shape[1], np.nan, dtype=np.float32)
    out_flat[target_valid_flat] = pred_valid

    return out_flat.reshape(out_shape).astype(np.float32)


def run_weighted_method(
    method_name: str,
    da_src,
    time_name: str,
    time_values: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
    out_path: str,
    out_var_name: str,
    idx: np.ndarray,
    weights: np.ndarray,
    source_valid_mask: np.ndarray,
    time_chunk: int,
    source_file: str = "",
    mask_file: str = "",
):
    log("=" * 100)
    log(f"Running {method_name}")
    log("=" * 100)

    src_meta = source_file or DEFAULT_INPUT
    mask_meta = mask_file or DEFAULT_MASK
    is_tmin = str(out_var_name).lower() == "tmin"
    title = (
        f"Daily Tmin interpolated from 0.25 degree using {method_name}"
        if is_tmin
        else f"Daily Tmax interpolated from 0.25 degree using {method_name}"
    )
    long_name = "Daily minimum temperature" if is_tmin else "Daily maximum temperature"

    ds_out, var_out = create_time_lat_lon_netcdf(
        out_path=out_path,
        var_name=out_var_name,
        time_values=time_values,
        lat_values=target_lat,
        lon_values=target_lon,
        attrs={
            "title": title,
            "method": method_name,
            "source_file": src_meta,
            "constraint_mask": mask_meta,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        var_attrs={
            "long_name": long_name,
            "method": method_name,
            "units": da_src.attrs.get("units", ""),
        },
        time_chunk=time_chunk,
    )

    try:
        ntime = da_src.sizes[time_name]
        out_shape = (target_lat.size, target_lon.size)

        for start in range(0, ntime, time_chunk):
            end = min(ntime, start + time_chunk)
            log(f"{method_name} processing time slice {start}:{end}")

            src_chunk = da_src.isel({time_name: slice(start, end)}).values.astype(np.float32)
            out_chunk = np.full((end - start, target_lat.size, target_lon.size), np.nan, dtype=np.float32)

            for i in range(end - start):
                values_flat_all = src_chunk[i].reshape(-1)
                values_flat = values_flat_all[source_valid_mask]
                out_chunk[i] = apply_weights_one_day(
                    values_flat,
                    idx,
                    weights,
                    target_mask.reshape(-1),
                    out_shape,
                )

            var_out[start:end, :, :] = out_chunk.astype(np.float32)

    finally:
        ds_out.close()

    save_target_mask(target_mask, target_lat, target_lon, out_path)
    log(f"Saved {method_name} output: {out_path}")


# =========================================================
# Kriging
# =========================================================
def spherical_covariance(distance, range_param=1.0, sill=1.0):
    """
    Spherical covariance model.
    """
    h = np.asarray(distance, dtype=np.float64)
    r = float(range_param)

    if r <= 0:
        raise ValueError("range_param must be positive.")

    hr = np.clip(h / r, 0.0, None)

    cov = np.zeros_like(hr, dtype=np.float64)
    inside = hr < 1.0
    cov[inside] = sill * (1.0 - 1.5 * hr[inside] + 0.5 * hr[inside] ** 3)

    return cov


def build_kriging_weights(
    src_points,
    target_points,
    k=12,
    range_param=1.0,
    nugget=1.0e-6,
    chunk_size=50000,
    n_jobs=1,
):
    """
    Build local ordinary kriging weights for all valid target cells.

    src_points:
        Array with shape [n_source, 2], columns are lat and lon.

    target_points:
        Array with shape [n_target, 2], columns are lat and lon.

    return:
        neighbor_idx: [n_target, k]
        weights:      [n_target, k]
    """
    if cKDTree is None:
        raise ImportError(f"scipy is required. Original error: {SCIPY_ERROR}")

    src_points = np.asarray(src_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    n_source = src_points.shape[0]
    n_target = target_points.shape[0]
    k_eff = min(int(k), int(n_source))

    log(f"Building local ordinary kriging weights: k={k_eff}, range={range_param}, nugget={nugget}")

    tree = cKDTree(src_points)
    dist, neighbor_idx = tree.query(target_points, k=k_eff, workers=int(n_jobs))

    if k_eff == 1:
        dist = dist[:, None]
        neighbor_idx = neighbor_idx[:, None]

    neighbor_idx = neighbor_idx.astype(np.int64)
    weights = np.full((n_target, k_eff), np.nan, dtype=np.float32)

    for start in range(0, n_target, int(chunk_size)):
        end = min(n_target, start + int(chunk_size))
        log(f"Solving kriging weights target cells {start}:{end}")

        idx_chunk = neighbor_idx[start:end]
        tgt_chunk = target_points[start:end]
        n_chunk = end - start

        for i in range(n_chunk):
            ids = idx_chunk[i]
            pts = src_points[ids]
            tgt = tgt_chunk[i]

            d_src = np.sqrt(
                (pts[:, 0, None] - pts[None, :, 0]) ** 2
                + (pts[:, 1, None] - pts[None, :, 1]) ** 2
            )

            d_tgt = np.sqrt(
                (pts[:, 0] - tgt[0]) ** 2
                + (pts[:, 1] - tgt[1]) ** 2
            )

            cov_src = spherical_covariance(d_src, range_param=range_param, sill=1.0)
            cov_tgt = spherical_covariance(d_tgt, range_param=range_param, sill=1.0)

            A = np.empty((k_eff + 1, k_eff + 1), dtype=np.float64)
            A[:k_eff, :k_eff] = cov_src
            A[:k_eff, :k_eff] += np.eye(k_eff, dtype=np.float64) * float(nugget)
            A[:k_eff, k_eff] = 1.0
            A[k_eff, :k_eff] = 1.0
            A[k_eff, k_eff] = 0.0

            b = np.empty((k_eff + 1,), dtype=np.float64)
            b[:k_eff] = cov_tgt
            b[k_eff] = 1.0

            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]

            weights[start + i, :] = sol[:k_eff].astype(np.float32)

    return neighbor_idx, weights


# =========================================================
# Main
# =========================================================
def parse_methods(methods_text: str) -> List[str]:
    methods = [item.strip().lower() for item in str(methods_text).split(",") if item.strip()]
    allowed = {"all", "bilinear", "idw", "kriging"}

    for method in methods:
        if method not in allowed:
            raise ValueError(f"Unsupported method: {method}")

    if "all" in methods:
        return ["bilinear", "idw", "kriging"]

    return methods


def main():
    parser = argparse.ArgumentParser(description="Interpolate Tmin 0.25 degree to target grid using bilinear, IDW, and kriging.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--mask", default=DEFAULT_MASK)
    parser.add_argument("--var", default=DEFAULT_VAR)
    parser.add_argument("--methods", default="all", help="all,bilinear,idw,kriging")
    parser.add_argument("--out-bilinear", default=DEFAULT_BILINEAR_OUT)
    parser.add_argument("--out-idw", default=DEFAULT_IDW_OUT)
    parser.add_argument("--out-kriging", default=DEFAULT_KRIGING_OUT)
    parser.add_argument("--time-chunk", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=256)
    parser.add_argument("--idw-k", type=int, default=12)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--kriging-k", type=int, default=12)
    parser.add_argument("--kriging-range", type=float, default=1.0)
    parser.add_argument("--kriging-nugget", type=float, default=1.0e-6)
    parser.add_argument("--weight-chunk", type=int, default=50000)
    args = parser.parse_args()

    methods = parse_methods(args.methods)

    log("=" * 100)
    log(f"Interpolation from 0.25 degree to target grid (variable={args.var})")
    log("=" * 100)
    log(f"Methods: {', '.join(methods)}")
    log(f"CPU jobs: {args.n_jobs}")
    log(f"Time chunk: {args.time_chunk}")

    ds_src, da_src, time_name, lat_name, lon_name, src_var, time_values, source_lat, source_lon = load_time_lat_lon_field(
        args.input,
        preferred_names=[args.var, "tmax", "Tmax", "tasmax", "tmin", "Tmin", "tasmin", "Band1"],
        var_name=args.var,
    )

    target_mask, target_lat, target_lon = load_mask(args.mask)
    target_lat, target_lon = build_target_grid_from_mask(target_mask, target_lat, target_lon)

    source_points_all = make_source_points(source_lat, source_lon)
    target_points_all, target_points_valid, target_valid_flat = make_target_points(target_lat, target_lon, target_mask)

    first_field = da_src.isel({time_name: 0}).values.astype(np.float32)
    source_valid_mask = np.isfinite(first_field.reshape(-1))

    source_points_valid = source_points_all[source_valid_mask]

    log(f"Valid source cells: {int(source_valid_mask.sum())}")
    log(f"Valid target cells: {int(target_valid_flat.sum())}")

    try:
        if "bilinear" in methods:
            run_bilinear(
                da_src=da_src,
                time_name=time_name,
                time_values=time_values,
                source_lat=source_lat,
                source_lon=source_lon,
                target_lat=target_lat,
                target_lon=target_lon,
                target_mask=target_mask,
                out_path=args.out_bilinear,
                out_var_name=args.var,
                time_chunk=args.time_chunk,
                source_file=args.input,
                mask_file=args.mask,
            )

        if "idw" in methods:
            idw_idx, idw_w = build_idw_weights(
                src_points=source_points_valid,
                target_points=target_points_valid,
                k=args.idw_k,
                power=args.idw_power,
                eps=1.0e-12,
                n_jobs=args.n_jobs,
            )

            run_weighted_method(
                method_name="idw",
                da_src=da_src,
                time_name=time_name,
                time_values=time_values,
                target_lat=target_lat,
                target_lon=target_lon,
                target_mask=target_mask,
                out_path=args.out_idw,
                out_var_name=args.var,
                idx=idw_idx,
                weights=idw_w,
                source_valid_mask=source_valid_mask,
                time_chunk=args.time_chunk,
                source_file=args.input,
                mask_file=args.mask,
            )

        if "kriging" in methods:
            krig_idx, krig_w = build_kriging_weights(
                src_points=source_points_valid,
                target_points=target_points_valid,
                k=args.kriging_k,
                range_param=args.kriging_range,
                nugget=args.kriging_nugget,
                chunk_size=args.weight_chunk,
                n_jobs=args.n_jobs,
            )

            run_weighted_method(
                method_name="kriging",
                da_src=da_src,
                time_name=time_name,
                time_values=time_values,
                target_lat=target_lat,
                target_lon=target_lon,
                target_mask=target_mask,
                out_path=args.out_kriging,
                out_var_name=args.var,
                idx=krig_idx,
                weights=krig_w,
                source_valid_mask=source_valid_mask,
                time_chunk=args.time_chunk,
                source_file=args.input,
                mask_file=args.mask,
            )

    finally:
        ds_src.close()

    log("=" * 100)
    log("All requested interpolation methods completed")
    log("=" * 100)


if __name__ == "__main__":
    main()