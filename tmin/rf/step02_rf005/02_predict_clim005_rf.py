#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_predict_clim005_rf.py

Predict 0.05-degree monthly climatology using trained RF-005 monthly models.

Inputs:
- predictors_005_static.nc
- rf005_month_01.joblib ... rf005_month_12.joblib

Output:
- tmin_005_clim_rf.nc
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
from joblib import load
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
from split_config import CLIM_START_YEAR, CLIM_END_YEAR, EVAL_START_YEAR, EVAL_END_YEAR, TRAIN_MONTHS_STR
from clim_month_tools import (
    resolve_train_months,
    find_first_model_path,
    load_source_clim_on_grid,
    subset_monthly_cube_to_output_months,
    output_months_attr,
)
from predictor_tools import (
    load_clim_predictor_maps,
    build_feature_matrix_from_maps,
    load_predictor_whitelist,
    require_whitelist_feature_names,
)



# =========================================================
# Default paths
# =========================================================
DEFAULT_PRED_PATH = "./data/coastal_static/predictors_005_static.nc"
DEFAULT_MODEL_DIR = "./outputs/tmin/rf/rf005/models"
DEFAULT_SOURCE_CLIM = "./outputs/tmin/rf/rf005/interim/tmin_025_clim_monthly.nc"
DEFAULT_OUT_PATH = "./outputs/tmin/rf/rf005/interim/tmin_005_clim_rf.nc"

PRED_LAT_NAME = None
PRED_LON_NAME = None


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


def to_numpy_2d_from_da(da: xr.DataArray, lat_name: str, lon_name: str) -> np.ndarray:
    dims = list(da.dims)

    if da.ndim > 2:
        squeeze_indexers = {}
        for d in dims:
            if d not in (lat_name, lon_name):
                if da.sizes[d] != 1:
                    raise ValueError(
                        f"Variable {da.name} has extra dim '{d}' with size {da.sizes[d]}; "
                        f"please manually select a 2D variable."
                    )
                squeeze_indexers[d] = 0
        da = da.isel(**squeeze_indexers)

    da = da.transpose(lat_name, lon_name)
    return da.values


def restore_orientation(arr2d: np.ndarray, lat_ascending: bool, lon_ascending: bool) -> np.ndarray:
    out = arr2d
    if not lon_ascending:
        out = out[:, ::-1]
    if not lat_ascending:
        out = out[::-1, :]
    return out


def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_serializable(v) for v in obj.tolist()]
    if obj is None:
        return None
    return obj


def load_predictor_dataset(pred_path: str, lat_name_override=None, lon_name_override=None):
    log(f"Loading predictor dataset: {pred_path}")
    ds = xr.open_dataset(pred_path)

    if lat_name_override is not None:
        lat_name = lat_name_override
    elif PRED_LAT_NAME is not None:
        lat_name = PRED_LAT_NAME
    else:
        lat_name = find_coord_name(ds, ["lat", "latitude", "y"])

    if lon_name_override is not None:
        lon_name = lon_name_override
    elif PRED_LON_NAME is not None:
        lon_name = PRED_LON_NAME
    else:
        lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    lat_raw = ds[lat_name].values
    lon_raw = ds[lon_name].values

    lat_ascending = np.all(np.diff(lat_raw) > 0)
    lon_ascending = np.all(np.diff(lon_raw) > 0)

    lat_proc = lat_raw.copy()
    lon_proc = lon_raw.copy()

    if not lat_ascending:
        lat_proc = lat_proc[::-1]
    if not lon_ascending:
        lon_proc = lon_proc[::-1]

    vars_proc = {}
    var_attrs = {}

    for name, da in ds.data_vars.items():
        if lat_name not in da.dims or lon_name not in da.dims:
            continue

        arr = to_numpy_2d_from_da(da, lat_name, lon_name)

        if not lat_ascending:
            arr = arr[::-1, :]
        if not lon_ascending:
            arr = arr[:, ::-1]

        vars_proc[name] = arr.astype(np.float32)
        var_attrs[name] = dict(da.attrs)

    global_attrs = dict(ds.attrs)
    ds.close()

    log(f"Predictor dataset loaded: {len(vars_proc)} gridded variables found")

    return {
        "lat_name": lat_name,
        "lon_name": lon_name,
        "lat_raw": lat_raw,
        "lon_raw": lon_raw,
        "lat_proc": lat_proc,
        "lon_proc": lon_proc,
        "lat_ascending": lat_ascending,
        "lon_ascending": lon_ascending,
        "vars_proc": vars_proc,
        "var_attrs": var_attrs,
        "global_attrs": global_attrs,
    }


def get_model_path(model_dir: str, month_int: int) -> str:
    return os.path.join(model_dir, f"rf005_month_{month_int:02d}.joblib")


def build_feature_matrix_from_names(vars_proc: dict, feature_names):
    missing = [name for name in feature_names if name not in vars_proc]
    if len(missing) > 0:
        raise KeyError("Missing predictor variables required by model: " + ", ".join(missing))

    feature_arrays = []
    for name in feature_names:
        arr = vars_proc[name].astype(np.float32)
        feature_arrays.append(arr)

    x_stack = np.stack(feature_arrays, axis=0)
    nfeat, nlat, nlon = x_stack.shape
    x_flat = x_stack.reshape(nfeat, nlat * nlon).T
    valid_x = np.all(np.isfinite(x_flat), axis=1)

    return x_stack, x_flat, valid_x


def resolve_models_from_payload(payload, model_path):
    if not isinstance(payload, dict):
        raise TypeError(
            f"Unexpected model payload type in {model_path}. "
            "Expected a dict containing model metadata."
        )

    if "feature_names" not in payload:
        raise KeyError(f"Model payload missing 'feature_names' key: {model_path}")

    feature_names = require_whitelist_feature_names(payload["feature_names"])

    if "model" in payload:
        models = [payload["model"]]
        mode = "single_final_model"
    elif "models" in payload and isinstance(payload["models"], list) and len(payload["models"]) > 0:
        models = payload["models"]
        mode = "ensemble_mean"
    else:
        raise KeyError(f"Model payload missing 'model' or non-empty 'models' key: {model_path}")

    return models, feature_names, mode


def predict_with_models(models, x_pred):
    pred_sum = None
    for model in models:
        pred = model.predict(x_pred).astype(np.float32)
        if pred_sum is None:
            pred_sum = pred
        else:
            pred_sum += pred
    return pred_sum / float(len(models))


def main():
    parser = argparse.ArgumentParser(description="Predict 0.05-degree monthly climatology using trained RF-005 models.")
    parser.add_argument("--pred", default=DEFAULT_PRED_PATH, help="Input predictor NetCDF")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory containing monthly model files")
    parser.add_argument("--source-clim", default=DEFAULT_SOURCE_CLIM, help="Source monthly climatology for non-trained months (bilinear to target grid)")
    parser.add_argument("--train-months", default=TRAIN_MONTHS_STR, help="Months with trained ML models (others use --source-clim)")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output NetCDF for predicted monthly climatology")
    parser.add_argument("--pred-lat", default=None, help="Optional predictor latitude coordinate name")
    parser.add_argument("--pred-lon", default=None, help="Optional predictor longitude coordinate name")
    args = parser.parse_args()

    ensure_parent_dir(args.out)

    train_months = resolve_train_months(range(1, 13), args.train_months)
    first_model_path = find_first_model_path(args.model_dir, "rf005", train_months)
    first_payload = load(first_model_path)
    feature_names_ref = require_whitelist_feature_names(first_payload["feature_names"])
    feature_means = first_payload.get("feature_means")
    feature_stds = first_payload.get("feature_stds")
    clim_start_year = int(first_payload.get("clim_start_year", CLIM_START_YEAR))
    clim_end_year = int(first_payload.get("clim_end_year", CLIM_END_YEAR))
    x_maps, _, _, _, pred_mask_arr, lat_proc, lon_proc = load_clim_predictor_maps(
        args.pred,
        feature_names=feature_names_ref,
        stats=(feature_means, feature_stds) if feature_means is not None else None,
        clim_start_year=clim_start_year,
        clim_end_year=clim_end_year,
    )
    x_flat_all, valid_x_all, _, _ = build_feature_matrix_from_maps(x_maps)
    nlat = lat_proc.size
    nlon = lon_proc.size
    lat_raw = lat_proc.copy()
    lon_raw = lon_proc.copy()
    lat_ascending = bool(np.all(np.diff(lat_proc) > 0))
    lon_ascending = bool(np.all(np.diff(lon_proc) > 0))
    land_mask_flat = pred_mask_arr.reshape(-1)
    has_land_mask = True
    log(f"Using feature names: {', '.join(feature_names_ref)}")

    month_values = np.arange(1, 13, dtype=np.int32)
    clim_cube_proc = np.full((12, nlat, nlon), np.nan, dtype=np.float32)

    month_param_summary = []
    prediction_modes = []

    for im, month_value in enumerate(month_values):
        month_int = int(month_value)
        model_path = get_model_path(args.model_dir, month_int)
        if not os.path.exists(model_path):
            log(f"Month {month_int:02d}: model missing ({model_path}); will use source clim fallback")
            continue

        log(f"Loading model for month {month_int:02d}: {model_path}")
        payload = load(model_path)
        models, feature_names, mode = resolve_models_from_payload(payload, model_path)
        prediction_modes.append(mode)

        if feature_names != feature_names_ref:
            raise ValueError(
                f"Feature names mismatch in month {month_int:02d} model.\n"
                f"Reference: {feature_names_ref}\n"
                f"Current:   {feature_names}"
            )

        x_flat, valid_x = x_flat_all, valid_x_all

        if has_land_mask:
            valid_land = np.isfinite(land_mask_flat) & (land_mask_flat > 0)
            valid_pred = valid_x & valid_land
        else:
            valid_pred = valid_x

        if int(valid_pred.sum()) == 0:
            raise ValueError(f"No valid predictor cells found for month {month_int:02d} prediction.")

        log(
            f"Predicting month {month_int:02d} on {int(valid_pred.sum())} valid cells "
            f"(mode={mode}, models={len(models)})"
        )

        y_pred_flat = np.full((nlat * nlon,), np.nan, dtype=np.float32)
        y_pred_flat[valid_pred] = predict_with_models(models, x_flat[valid_pred])
        clim_cube_proc[im, :, :] = y_pred_flat.reshape(nlat, nlon)

        month_param_summary.append({
            "month": month_int,
            "best_params": payload.get("best_params", {}),
            "cv_strategy": payload.get("cv_strategy", ""),
            "cv_folds": payload.get("cv_folds", ""),
            "cv_repeats": payload.get("cv_repeats", ""),
            "spatial_block_size": payload.get("spatial_block_size", ""),
            "prediction_mode": mode,
            "n_models": len(models),
        })

    clim_cube_out = np.full_like(clim_cube_proc, np.nan, dtype=np.float32)
    for im in range(12):
        clim_cube_out[im, :, :] = restore_orientation(
            clim_cube_proc[im, :, :],
            lat_ascending=lat_ascending,
            lon_ascending=lon_ascending,
        )


    clim_out, output_month_values = subset_monthly_cube_to_output_months(
        clim_cube_out, train_months
    )
    if not np.any(np.isfinite(clim_out)):
        raise ValueError("Predicted climatology has no valid values for output months.")

    ds_out = xr.Dataset(
        data_vars={
            "clim_tmin_rf": (("month", "lat", "lon"), clim_out.astype(np.float32))
        },
        coords={
            "month": output_month_values,
            "lat": lat_raw,
            "lon": lon_raw,
        },
        attrs={
            "title": "RF-005 predicted monthly climatology on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictor_file": args.pred,
            "model_dir": args.model_dir,
            "feature_names": ",".join(feature_names_ref) if feature_names_ref is not None else "",
            "source_grid": "0.05_degree",
            "prediction_type": "monthly_climatology",
            "output_calendar_months": output_months_attr(train_months),
            "has_land_mask_constraint": str(bool(has_land_mask)),
            "prediction_modes": ",".join(sorted(set(prediction_modes))),
            "month_param_summary": json.dumps(
                to_json_serializable(month_param_summary),
                ensure_ascii=False
            ),
        },
    )

    ds_out["clim_tmin_rf"].attrs = {
        "long_name": "RF-005 predicted monthly climatology of Tmin on 0.05-degree grid",
        "units": "",
    }
    ds_out["month"].attrs = {"long_name": "calendar month", "units": "1"}
    ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    encoding = {
        "clim_tmin_rf": {"zlib": True, "complevel": 4, "dtype": "float32"}
    }

    log(f"Saving output: {args.out}")
    ds_out.to_netcdf(args.out, encoding=encoding)
    log("Done.")


if __name__ == "__main__":
    main()
