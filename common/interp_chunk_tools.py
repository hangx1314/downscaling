#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory-safe lat/lon interpolation for large target grids (e.g. 0.01°)."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Sequence

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from mask_fill import fill_nan_inside_mask_2d


def _time_units_and_calendar(time_vals: np.ndarray, time_attrs: Dict) -> tuple[str, str]:
    units = time_attrs.get("units")
    calendar = time_attrs.get("calendar", "proleptic_gregorian")
    if not units:
        t0 = pd.Timestamp(time_vals[0])
        if t0.tz is not None:
            t0 = t0.tz_localize(None)
        units = f"days since {t0.strftime('%Y-%m-%d %H:%M:%S')}"
    return units, calendar


def _encode_time_values(
    time_vals: np.ndarray,
    t0: int,
    t1: int,
    units: str,
    calendar: str,
) -> np.ndarray:
    dt_index = pd.to_datetime(time_vals[t0:t1])
    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_localize(None)
    return netCDF4.date2num(dt_index.to_pydatetime(), units=units, calendar=calendar)


def _set_ncattrs(obj, attrs: Dict, skip: Optional[set[str]] = None) -> None:
    skip = skip or set()
    for key, value in attrs.items():
        if key in skip:
            continue
        try:
            obj.setncattr(key, value)
        except Exception:
            pass


def _interp_chunk_values(
    da_sub: xr.DataArray,
    lat_name: str,
    lon_name: str,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    da_linear = da_sub.interp(
        {lat_name: tgt_lat, lon_name: tgt_lon},
        method="linear",
    )
    da_nearest = da_sub.interp(
        {lat_name: tgt_lat, lon_name: tgt_lon},
        method="nearest",
    )
    return da_linear.where(np.isfinite(da_linear), da_nearest).astype(np.float32).values


def apply_coastal_mask_and_nearest_fill(
    da: xr.DataArray,
    land_mask: xr.DataArray,
    *,
    time_name: str = "time",
    time_chunk: int = 30,
    log_fn: Optional[Callable[[str], None]] = print,
) -> xr.DataArray:
    """Mask to coastal domain and fill interior NaN from nearest valid neighbor."""
    mask2d = np.asarray(land_mask.values, dtype=np.float32) > 0

    if time_name not in da.dims and da.ndim == 2:
        arr = np.asarray(da.values, dtype=np.float32)
        before = int(np.sum(mask2d & (~np.isfinite(arr))))
        out2d = fill_nan_inside_mask_2d(arr, mask2d)
        remaining = int(np.sum(mask2d & (~np.isfinite(out2d))))
        if log_fn is not None:
            log_fn(
                f"Nearest-neighbor filled {before} NaN cells inside mask; "
                f"remaining={remaining}"
            )
        return xr.DataArray(out2d, coords=da.coords, dims=da.dims, attrs=da.attrs)

    if time_chunk < 1:
        raise ValueError("time_chunk must be >= 1")

    ntime = int(da.sizes[time_name])
    total_before = 0
    total_remaining = 0
    out_chunks = []

    for t0 in range(0, ntime, time_chunk):
        t1 = min(t0 + time_chunk, ntime)
        if log_fn is not None:
            log_fn(f"Mask nearest-fill time slice {t0}:{t1} / {ntime}")
        sub = np.asarray(
            da.isel({time_name: slice(t0, t1)}).transpose(time_name, ...).values,
            dtype=np.float32,
        )
        if sub.ndim == 2:
            sub = sub[None, ...]
        out_sub = np.empty_like(sub)
        for i in range(sub.shape[0]):
            total_before += int(np.sum(mask2d & (~np.isfinite(sub[i]))))
            out_sub[i] = fill_nan_inside_mask_2d(sub[i], mask2d)
            total_remaining += int(np.sum(mask2d & (~np.isfinite(out_sub[i]))))
        out_chunks.append(
            xr.DataArray(
                out_sub,
                coords=da.isel({time_name: slice(t0, t1)}).coords,
                dims=da.isel({time_name: slice(t0, t1)}).dims,
                attrs=da.attrs,
            )
        )

    if log_fn is not None:
        log_fn(
            f"Nearest-neighbor filled {total_before} NaN cells inside mask; "
            f"remaining={total_remaining}"
        )
    if len(out_chunks) == 1:
        return out_chunks[0]
    return xr.concat(out_chunks, dim=time_name)


def write_interp_to_target_grid_with_mask_fill_netcdf(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    time_name: str,
    tgt_lat: Sequence[float],
    tgt_lon: Sequence[float],
    land_mask: xr.DataArray,
    out_path: str,
    out_var: str,
    *,
    time_chunk: int = 30,
    var_attrs: Optional[Dict] = None,
    ds_attrs: Optional[Dict] = None,
    time_attrs: Optional[Dict] = None,
    lat_attrs: Optional[Dict] = None,
    lon_attrs: Optional[Dict] = None,
    encoding: Optional[Dict] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> None:
    """Interp + mask nearest-fill + NetCDF write without holding full 0.01° cube in RAM."""
    if time_chunk < 1:
        raise ValueError("time_chunk must be >= 1")

    mask2d = np.asarray(land_mask.values, dtype=np.float32) > 0
    tgt_lat_arr = np.asarray(tgt_lat, dtype=np.float64)
    tgt_lon_arr = np.asarray(tgt_lon, dtype=np.float64)

    var_attrs = dict(var_attrs or {})
    ds_attrs = dict(ds_attrs or {})
    time_attrs = dict(time_attrs or {})
    lat_attrs = dict(lat_attrs or {"long_name": "latitude", "units": "degrees_north"})
    lon_attrs = dict(lon_attrs or {"long_name": "longitude", "units": "degrees_east"})
    encoding = encoding or {}
    enc = encoding.get(out_var, {})
    use_zlib = bool(enc.get("zlib", True))
    complevel = int(enc.get("complevel", 4))

    if os.path.exists(out_path):
        os.remove(out_path)

    time_vals = da[time_name].values
    ntime = int(time_vals.size)
    nlat = int(tgt_lat_arr.size)
    nlon = int(tgt_lon_arr.size)
    time_units, time_calendar = _time_units_and_calendar(time_vals, time_attrs)

    total_before = 0
    total_remaining = 0

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)

        v_time = ds.createVariable("time", "f8", ("time",))
        v_lat = ds.createVariable("lat", "f4", ("lat",))
        v_lon = ds.createVariable("lon", "f4", ("lon",))
        v_data = ds.createVariable(
            out_var,
            "f4",
            ("time", "lat", "lon"),
            zlib=use_zlib,
            complevel=complevel,
            chunksizes=(
                min(time_chunk, ntime),
                min(256, nlat),
                min(256, nlon),
            ),
            fill_value=np.float32(np.nan),
        )

        v_time.units = time_units
        v_time.calendar = time_calendar
        _set_ncattrs(v_time, time_attrs, skip={"units", "calendar"})
        _set_ncattrs(v_lat, lat_attrs)
        _set_ncattrs(v_lon, lon_attrs)
        _set_ncattrs(v_data, var_attrs)
        _set_ncattrs(ds, ds_attrs)

        v_lat[:] = tgt_lat_arr.astype(np.float32)
        v_lon[:] = tgt_lon_arr.astype(np.float32)

        for t0 in range(0, ntime, time_chunk):
            t1 = min(t0 + time_chunk, ntime)
            if log_fn is not None:
                log_fn(
                    f"Interpolating+mask-fill time slice {t0}:{t1} / {ntime}"
                )
            da_sub = da.isel({time_name: slice(t0, t1)})
            chunk = _interp_chunk_values(da_sub, lat_name, lon_name, tgt_lat_arr, tgt_lon_arr)
            if chunk.ndim == 2:
                chunk = chunk[None, ...]
            out_chunk = np.empty_like(chunk)
            for i in range(chunk.shape[0]):
                total_before += int(np.sum(mask2d & (~np.isfinite(chunk[i]))))
                out_chunk[i] = fill_nan_inside_mask_2d(chunk[i], mask2d)
                total_remaining += int(np.sum(mask2d & (~np.isfinite(out_chunk[i]))))
            v_time[t0:t1] = _encode_time_values(
                time_vals, t0, t1, time_units, time_calendar
            )
            v_data[t0:t1, :, :] = out_chunk
            ds.sync()

    if log_fn is not None:
        log_fn(
            f"Nearest-neighbor filled {total_before} NaN cells inside mask; "
            f"remaining={total_remaining}"
        )
        log_fn(f"Saved: {out_path}")


def interp_to_target_grid_time_chunked(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    time_name: str,
    tgt_lat: Sequence[float],
    tgt_lon: Sequence[float],
    time_chunk: int = 30,
    log_fn: Optional[Callable[[str], None]] = print,
) -> xr.DataArray:
    """Linear interp to target grid with nearest fill; process time in chunks."""
    if time_chunk < 1:
        raise ValueError("time_chunk must be >= 1")

    tgt_lat_arr = np.asarray(tgt_lat)
    tgt_lon_arr = np.asarray(tgt_lon)
    ntime = int(da.sizes[time_name])
    chunks = []
    for t0 in range(0, ntime, time_chunk):
        t1 = min(t0 + time_chunk, ntime)
        if log_fn is not None:
            log_fn(f"Interpolating time slice {t0}:{t1} / {ntime}")
        da_sub = da.isel({time_name: slice(t0, t1)})
        da_linear = da_sub.interp(
            {lat_name: tgt_lat_arr, lon_name: tgt_lon_arr},
            method="linear",
        )
        da_nearest = da_sub.interp(
            {lat_name: tgt_lat_arr, lon_name: tgt_lon_arr},
            method="nearest",
        )
        da_chunk = da_linear.where(np.isfinite(da_linear), da_nearest).astype(np.float32)
        chunks.append(da_chunk)
    if len(chunks) == 1:
        return chunks[0]
    return xr.concat(chunks, dim=time_name)
