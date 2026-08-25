#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory-safe daily raw build on fine grids (e.g. 0.01°)."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr


def _time_units_and_calendar(
    time_vals: np.ndarray,
    time_attrs: Dict,
) -> tuple[str, str]:
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


def build_daily_raw_netcdf_time_chunked(
    da_clim: xr.DataArray,
    month_name: str,
    lat_name: str,
    lon_name: str,
    da_anom: xr.DataArray,
    time_name: str,
    out_path: str,
    out_var: str,
    var_attrs: Dict,
    ds_attrs: Dict,
    time_attrs: Dict,
    lat_attrs: Dict,
    lon_attrs: Dict,
    encoding: Dict,
    time_chunk: int = 30,
    log_fn: Optional[Callable[[str], None]] = print,
) -> None:
    if time_chunk < 1:
        raise ValueError("time_chunk must be >= 1")

    if os.path.exists(out_path):
        os.remove(out_path)

    enc = encoding.get(out_var, {})
    use_zlib = bool(enc.get("zlib", True))
    complevel = int(enc.get("complevel", 4))

    clim_arr = (
        da_clim.transpose(month_name, lat_name, lon_name).astype(np.float32).values
    )
    lat = np.asarray(da_clim[lat_name].values)
    lon = np.asarray(da_clim[lon_name].values)
    time_vals = da_anom[time_name].values
    month_idx = pd.to_datetime(time_vals).month.values.astype(np.int32) - 1
    ntime = int(time_vals.size)
    nlat = int(lat.size)
    nlon = int(lon.size)

    anom_da = da_anom.transpose(time_name, lat_name, lon_name).astype(np.float32)
    time_units, time_calendar = _time_units_and_calendar(time_vals, time_attrs)

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

        v_lat[:] = lat.astype(np.float32)
        v_lon[:] = lon.astype(np.float32)

        for t0 in range(0, ntime, time_chunk):
            t1 = min(t0 + time_chunk, ntime)
            if log_fn is not None:
                log_fn(f"Building raw time slice {t0}:{t1} / {ntime}")

            clim_slice = clim_arr[month_idx[t0:t1]]
            anom_slice = anom_da.isel({time_name: slice(t0, t1)}).values
            raw_slice = (clim_slice + anom_slice).astype(np.float32)
            time_chunk_vals = _encode_time_values(
                time_vals, t0, t1, time_units, time_calendar
            )

            v_time[t0:t1] = time_chunk_vals
            v_data[t0:t1, :, :] = raw_slice
            ds.sync()
