#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load coastal mask aligned to a lat/lon grid."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _find_coord_name(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Cannot find coordinate among candidates: {candidates}")


def _find_2d_var(ds: xr.Dataset, lat_name: str, lon_name: str):
    for name in ["land_mask", "mask", "lsm", "land"]:
        if name in ds.data_vars:
            da = ds[name]
            if lat_name in da.dims and lon_name in da.dims:
                return name
    for name, da in ds.data_vars.items():
        if lat_name in da.dims and lon_name in da.dims:
            return name
    raise KeyError("Cannot find a 2D mask variable with lat/lon dims.")


def load_land_mask_2d(mask_path: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return boolean mask on (lat, lon) grid; True = inside coastal domain."""
    ds = xr.open_dataset(mask_path)
    try:
        lat_name = _find_coord_name(ds, ["lat", "latitude", "y"])
        lon_name = _find_coord_name(ds, ["lon", "longitude", "x"])
        var_name = _find_2d_var(ds, lat_name, lon_name)
        da = ds[var_name].transpose(lat_name, lon_name)
        if not np.allclose(da[lat_name].values, lat, rtol=0, atol=1e-4):
            da = da.interp({lat_name: lat, lon_name: lon}, method="nearest")
        values = np.asarray(da.values)
        if np.issubdtype(values.dtype, np.floating):
            return (np.isfinite(values) & (values > 0)).astype(bool)
        return (values > 0).astype(bool)
    finally:
        ds.close()
