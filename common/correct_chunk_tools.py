#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory-safe residual correction write on fine grids."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from build_raw_chunk_tools import _encode_time_values, _set_ncattrs, _time_units_and_calendar
from mask_fill import fill_nan_inside_mask_2d


def _check_same_spatial_grid(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> None:
    if da_a.dims != da_b.dims:
        raise ValueError(f"Dimension mismatch: {da_a.dims} vs {da_b.dims}")
    for dim in da_a.dims:
        if da_a.sizes[dim] != da_b.sizes[dim]:
            raise ValueError(
                f"Size mismatch on {dim}: {da_a.sizes[dim]} vs {da_b.sizes[dim]}"
            )
    for coord in ("lat", "lon"):
        if coord not in da_a.coords or coord not in da_b.coords:
            continue
        if not np.allclose(da_a[coord].values, da_b[coord].values, rtol=rtol, atol=atol):
            raise ValueError(
                f"Coordinate '{coord}' differs beyond tolerance "
                f"(max abs diff={np.max(np.abs(da_a[coord].values - da_b[coord].values)):.3e})."
            )


def write_corrected_sum_netcdf_time_chunked(
    out_path: str,
    da_raw: xr.DataArray,
    da_res: xr.DataArray,
    out_var: str,
    *,
    time_chunk: int = 30,
    var_attrs: Optional[Dict] = None,
    ds_attrs: Optional[Dict] = None,
    time_attrs: Optional[Dict] = None,
    lat_attrs: Optional[Dict] = None,
    lon_attrs: Optional[Dict] = None,
    encoding: Optional[Dict] = None,
    mask2d: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> None:
    """Write ``da_raw + da_res`` without xarray label alignment or full-cube RAM."""
    if time_chunk < 1:
        raise ValueError("time_chunk must be >= 1")

    _check_same_spatial_grid(da_raw, da_res)

    if os.path.exists(out_path):
        os.remove(out_path)

    var_attrs = dict(var_attrs or {})
    ds_attrs = dict(ds_attrs or {})
    time_attrs = dict(time_attrs or {})
    lat_attrs = dict(lat_attrs or {"long_name": "latitude", "units": "degrees_north"})
    lon_attrs = dict(lon_attrs or {"long_name": "longitude", "units": "degrees_east"})
    encoding = encoding or {}
    enc = encoding.get(out_var, {})
    use_zlib = bool(enc.get("zlib", True))
    complevel = int(enc.get("complevel", 4))

    lat = np.asarray(da_raw["lat"].values, dtype=np.float32)
    lon = np.asarray(da_raw["lon"].values, dtype=np.float32)
    time_vals = da_raw["time"].values
    ntime = int(time_vals.size)
    nlat = int(lat.size)
    nlon = int(lon.size)
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

        v_lat[:] = lat
        v_lon[:] = lon

        for t0 in range(0, ntime, time_chunk):
            t1 = min(t0 + time_chunk, ntime)
            if log_fn is not None:
                log_fn(f"Correcting time slice {t0}:{t1} / {ntime}")

            raw_slice = da_raw.isel(time=slice(t0, t1)).load().values.astype(np.float32)
            res_slice = da_res.isel(time=slice(t0, t1)).load().values.astype(np.float32)
            corr_slice = (raw_slice + res_slice).astype(np.float32)
            if mask2d is not None:
                for i in range(corr_slice.shape[0]):
                    corr_slice[i] = fill_nan_inside_mask_2d(corr_slice[i], mask2d)

            v_time[t0:t1] = _encode_time_values(
                time_vals, t0, t1, time_units, time_calendar
            )
            v_data[t0:t1, :, :] = corr_slice
            ds.sync()
