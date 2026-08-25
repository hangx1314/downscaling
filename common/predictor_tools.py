#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictor loading helpers for zcn climatology ML (static + yearly fields).

RF, XGBoost, and U-Net all use the 22-variable whitelist in predictor_names.yaml.
land_mask is read only as a spatial mask, not as a model input.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from split_config import CLIM_END_YEAR, CLIM_START_YEAR

_HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_PREDICTOR_NAMES_PATH = os.path.join(_HERE, "predictor_names.yaml")
EXPECTED_N_PREDICTORS = 22


def load_predictor_whitelist(path: Optional[str] = None) -> List[str]:
    yaml_path = path or DEFAULT_PREDICTOR_NAMES_PATH
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError("Predictor whitelist not found: %s" % yaml_path)
    names = []
    with open(yaml_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.endswith(":"):
                continue
            if line.startswith("- "):
                names.append(line[2:].strip().strip("'").strip('"'))
    if len(names) != EXPECTED_N_PREDICTORS:
        raise ValueError(
            "Expected %d predictors in %s, got %d: %s"
            % (EXPECTED_N_PREDICTORS, yaml_path, len(names), names)
        )
    return names


def require_whitelist_feature_names(names) -> List[str]:
    expected = load_predictor_whitelist()
    got = list(names)
    if got != expected:
        extra = [n for n in got if n not in expected]
        missing = [n for n in expected if n not in got]
        raise ValueError(
            "Checkpoint is not the 22-predictor whitelist. "
            "Retrain RF/XGBoost/U-Net; 23-channel weights that include land_mask are not supported.\n"
            "expected (%d): %s\n"
            "got (%d): %s\n"
            "extra: %s\n"
            "missing: %s"
            % (len(expected), expected, len(got), got, extra, missing)
        )
    return expected


def find_coord_name(ds: xr.Dataset, candidates: List[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Cannot find coordinate among candidates: {candidates}")


def sort_ds_by_latlon(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.Dataset:
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})
    if ds[lon_name].values[0] > ds[lon_name].values[-1]:
        ds = ds.isel({lon_name: slice(None, None, -1)})
    return ds


def map_predictor_year(input_year: int, available_years: np.ndarray) -> int:
    years = {int(y) for y in available_years.tolist()}
    if input_year <= 1985 and 1985 in years:
        return 1985
    if input_year <= 1990 and 1990 in years:
        return 1990
    if input_year in years:
        return int(input_year)
    lower = [y for y in years if y <= input_year]
    if lower:
        return int(max(lower))
    return int(min(years))


def fit_feature_stats(feature_maps: np.ndarray, mask: np.ndarray) -> Tuple[List[float], List[float]]:
    means = []
    stds = []
    for i in range(feature_maps.shape[0]):
        arr = feature_maps[i]
        valid = mask & np.isfinite(arr)
        if not np.any(valid):
            means.append(0.0)
            stds.append(1.0)
            continue
        mean = float(np.mean(arr[valid]))
        std = float(np.std(arr[valid]))
        if std < 1e-6:
            std = 1.0
        means.append(mean)
        stds.append(std)
    return means, stds


def apply_feature_stats(
    feature_maps: np.ndarray, means: List[float], stds: List[float]
) -> np.ndarray:
    out = np.empty_like(feature_maps, dtype=np.float32)
    for i in range(feature_maps.shape[0]):
        std = float(stds[i]) if abs(float(stds[i])) >= 1e-6 else 1.0
        arr = feature_maps[i].astype(np.float32)
        out[i] = np.where(np.isfinite(arr), (arr - float(means[i])) / std, 0.0).astype(np.float32)
    return out


def aggregate_yearly_predictor_for_clim(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    clim_start_year: int = CLIM_START_YEAR,
    clim_end_year: int = CLIM_END_YEAR,
) -> np.ndarray:
    da_y = da.transpose("year", lat_name, lon_name)
    available_years = da_y["year"].values.astype(np.int32)
    year_to_idx = {int(y): i for i, y in enumerate(available_years.tolist())}
    snapshots = []
    for y in range(int(clim_start_year), int(clim_end_year) + 1):
        mapped = map_predictor_year(y, available_years)
        snapshots.append(da_y.isel(year=year_to_idx[mapped]).values.astype(np.float32))
    return np.nanmean(np.stack(snapshots, axis=0), axis=0).astype(np.float32)


def load_clim_predictor_maps(
    pred_path: str,
    exclude_vars: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    stats: Optional[Tuple[List[float], List[float]]] = None,
    clim_start_year: int = CLIM_START_YEAR,
    clim_end_year: int = CLIM_END_YEAR,
) -> Tuple[np.ndarray, List[str], List[float], List[float], np.ndarray, np.ndarray, np.ndarray]:
    del exclude_vars  # kept for call-site compatibility; whitelist is authoritative
    ds = xr.open_dataset(pred_path)
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])
    ds = sort_ds_by_latlon(ds, lat_name, lon_name)
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    if "land_mask" in ds.data_vars:
        mask = ds["land_mask"].transpose(lat_name, lon_name).values > 0
    else:
        mask = np.ones((lat.size, lon.size), dtype=bool)

    if feature_names is None:
        names = load_predictor_whitelist()
    else:
        names = require_whitelist_feature_names(feature_names)
    if len(names) != EXPECTED_N_PREDICTORS:
        raise ValueError(
            "Expected %d predictors, got %d: %s" % (EXPECTED_N_PREDICTORS, len(names), names)
        )

    maps = []
    for name in names:
        if name not in ds.data_vars:
            raise KeyError(f"Predictor variable '{name}' not found in {pred_path}")
        da = ds[name]
        if "year" in da.dims:
            arr = aggregate_yearly_predictor_for_clim(
                da, lat_name, lon_name, clim_start_year=clim_start_year, clim_end_year=clim_end_year
            )
        else:
            arr = da.transpose(lat_name, lon_name).values.astype(np.float32)
        maps.append(arr)

    raw = np.stack(maps, axis=0).astype(np.float32)
    if stats is None:
        means, stds = fit_feature_stats(raw, mask)
    else:
        means, stds = stats
        if len(means) != EXPECTED_N_PREDICTORS or len(stds) != EXPECTED_N_PREDICTORS:
            raise ValueError(
                "feature_means/feature_stds must have length %d, got %d/%d"
                % (EXPECTED_N_PREDICTORS, len(means), len(stds))
            )
    out = apply_feature_stats(raw, means, stds)
    ds.close()
    return out, names, means, stds, mask.astype(bool), lat, lon


def build_feature_matrix_from_maps(feature_maps: np.ndarray):
    nfeat, nlat, nlon = feature_maps.shape
    if nfeat != EXPECTED_N_PREDICTORS:
        raise ValueError(
            "Expected %d predictor maps, got %d" % (EXPECTED_N_PREDICTORS, nfeat)
        )
    x_flat = feature_maps.reshape(nfeat, nlat * nlon).T.astype(np.float32)
    valid_x = np.all(np.isfinite(x_flat), axis=1)
    row_ids = np.repeat(np.arange(nlat, dtype=np.int32), nlon)
    col_ids = np.tile(np.arange(nlon, dtype=np.int32), nlat)
    return x_flat, valid_x, row_ids, col_ids
