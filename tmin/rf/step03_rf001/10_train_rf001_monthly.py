#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_train_rf001_monthly.py

Train monthly RF-001 models with repeated spatial block cross-validation:
- X: 0.05-degree predictors aggregated from 0.01-degree static predictors
- y: 0.05-degree monthly climatology of Tmin

Inputs:
- predictors_005_static.nc
- tmin_005_clim_monthly.nc

Outputs:
- rf001_month_01.joblib ... rf001_month_12.joblib
- rf001_train_metrics.csv
- rf001_feature_importance.csv
- tmin_005_clim_rf001_fit.nc
"""

import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import sys

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
from clim_month_tools import resolve_train_months
from predictor_tools import (
    load_clim_predictor_maps,
    build_feature_matrix_from_maps,
    load_predictor_whitelist,
)

from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid


# =========================================================
# Default paths
# =========================================================
DEFAULT_PRED_PATH = "./data/coastal_static/predictors_005_static.nc"
DEFAULT_CLIM_PATH = "./outputs/tmin/rf/rf001/interim/tmin_005_clim_monthly.nc"

DEFAULT_MODEL_DIR = "./outputs/tmin/rf/rf001/models"
DEFAULT_METRICS_CSV = "./outputs/tmin/rf/rf001/models/rf001_train_metrics.csv"
DEFAULT_IMPORTANCE_CSV = "./outputs/tmin/rf/rf001/models/rf001_feature_importance.csv"
DEFAULT_FIT_NC = "./outputs/tmin/rf/rf001/interim/tmin_005_clim_rf001_fit.nc"

TARGET_VAR_NAME = None

DEFAULT_N_ESTIMATORS_GRID = "400,800"
DEFAULT_MAX_DEPTH_GRID = "12,None"
DEFAULT_MIN_SAMPLES_LEAF_GRID = "1,5"
DEFAULT_MAX_FEATURES_GRID = "sqrt"
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_REPEATS = 3
DEFAULT_SPATIAL_BLOCK_SIZE = 4


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_coord_name(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Cannot find coordinate among candidates: {candidates}")


def find_monthly_target_var(ds: xr.Dataset, month_name: str, lat_name: str, lon_name: str, preferred_names=None):
    if preferred_names is None:
        preferred_names = []

    for name in preferred_names:
        if name in ds.data_vars:
            da = ds[name]
            if month_name in da.dims and lat_name in da.dims and lon_name in da.dims:
                return name

    for name, da in ds.data_vars.items():
        if month_name in da.dims and lat_name in da.dims and lon_name in da.dims:
            return name

    raise KeyError("Cannot find a monthly target variable with month/lat/lon dims.")


def sort_dataset_2d_vars(ds: xr.Dataset, lat_name: str, lon_name: str):
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    if lat_vals[0] > lat_vals[-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})

    if lon_vals[0] > lon_vals[-1]:
        ds = ds.isel({lon_name: slice(None, None, -1)})

    return ds


def load_predictors(pred_path: str):
    log(f"Loading predictors: {pred_path}")
    ds = xr.open_dataset(pred_path)

    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])
    ds = sort_dataset_2d_vars(ds, lat_name, lon_name)

    predictor_vars = load_predictor_whitelist()
    missing = [name for name in predictor_vars if name not in ds.data_vars]
    if missing:
        raise KeyError("Missing predictor variables: " + ", ".join(missing))

    return ds, lat_name, lon_name, predictor_vars


def load_target(clim_path: str):
    log(f"Loading monthly climatology target: {clim_path}")
    ds = xr.open_dataset(clim_path)

    month_name = find_coord_name(ds, ["month"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])
    ds = sort_dataset_2d_vars(ds, lat_name, lon_name)

    if TARGET_VAR_NAME is not None:
        if TARGET_VAR_NAME not in ds.data_vars:
            raise KeyError(f"TARGET_VAR_NAME '{TARGET_VAR_NAME}' not found in target dataset.")
        target_var = TARGET_VAR_NAME
    else:
        target_var = find_monthly_target_var(
            ds,
            month_name,
            lat_name,
            lon_name,
            preferred_names=["clim_tmin", "tmin_clim", "tmin", "Band1"],
        )

    return ds, month_name, lat_name, lon_name, target_var


def build_feature_matrix(ds_pred: xr.Dataset, predictor_names, lat_name: str, lon_name: str):
    feature_arrays = []
    for name in predictor_names:
        arr = ds_pred[name].transpose(lat_name, lon_name).values.astype(np.float32)
        feature_arrays.append(arr)

    x_stack = np.stack(feature_arrays, axis=0)
    nfeat, nlat, nlon = x_stack.shape
    x_flat = x_stack.reshape(nfeat, nlat * nlon).T
    valid_x = np.all(np.isfinite(x_flat), axis=1)

    row_ids = np.repeat(np.arange(nlat, dtype=np.int32), nlon)
    col_ids = np.tile(np.arange(nlon, dtype=np.int32), nlat)

    return x_stack, x_flat, valid_x, row_ids, col_ids


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    if y_true.size < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def save_csv(rows, out_csv):
    ensure_parent_dir(out_csv)
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_or_none_list(text: str):
    values = []
    for item in str(text).split(","):
        s = item.strip()
        if not s:
            continue
        if s.lower() in ("none", "null"):
            values.append(None)
        else:
            values.append(int(s))
    if not values:
        raise ValueError("Parsed empty integer/None grid.")
    return values


def parse_max_features_list(text: str):
    values = []
    for item in str(text).split(","):
        s = item.strip()
        if not s:
            continue
        low = s.lower()
        if low in ("none", "null"):
            values.append(None)
        elif low in ("sqrt", "log2"):
            values.append(low)
        else:
            values.append(float(s))
    if not values:
        raise ValueError("Parsed empty max_features grid.")
    return values


def build_param_grid(args):
    param_dict = {
        "n_estimators": parse_int_or_none_list(args.n_estimators_grid),
        "max_depth": parse_int_or_none_list(args.max_depth_grid),
        "min_samples_leaf": parse_int_or_none_list(args.min_samples_leaf_grid),
        "max_features": parse_max_features_list(args.max_features_grid),
    }
    grid = list(ParameterGrid(param_dict))
    if len(grid) == 0:
        raise ValueError("Parameter grid is empty.")
    return grid


def build_spatial_cv_splits(
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    valid_mask: np.ndarray,
    n_splits: int,
    n_repeats: int,
    block_size: int,
    random_state: int,
):
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        raise ValueError("No valid samples available for CV split generation.")

    row_valid = row_ids[valid_idx]
    col_valid = col_ids[valid_idx]
    rng = np.random.default_rng(random_state)

    split_records = []
    for repeat_idx in range(n_repeats):
        row_shift = int(rng.integers(0, max(1, block_size)))
        col_shift = int(rng.integers(0, max(1, block_size)))

        group_row = (row_valid + row_shift) / block_size
        group_col = (col_valid + col_shift) / block_size
        n_group_col = int(group_col.max()) + 1
        groups = group_row * n_group_col + group_col

        unique_groups = np.unique(groups)
        effective_splits = min(int(n_splits), int(unique_groups.size))
        if effective_splits < 2:
            raise ValueError(
                f"Not enough spatial groups for GroupKFold: only {unique_groups.size} groups found."
            )

        splitter = GroupKFold(n_splits=effective_splits)
        for fold_idx, (train_idx_local, test_idx_local) in enumerate(splitter.split(valid_idx, groups=groups)):
            if train_idx_local.size == 0 or test_idx_local.size == 0:
                continue

            split_records.append({
                "repeat_idx": repeat_idx,
                "fold_idx": fold_idx,
                "row_shift": row_shift,
                "col_shift": col_shift,
                "train_idx": train_idx_local.astype(np.int32),
                "test_idx": test_idx_local.astype(np.int32),
            })

    if len(split_records) == 0:
        raise ValueError("Failed to build any valid spatial CV splits.")

    return split_records


def evaluate_param_grid(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    split_records,
    param_grid,
    base_random_state: int,
):
    cv_rows = []

    for grid_idx, params in enumerate(param_grid, start=1):
        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        log(
            "Evaluating params "
            f"{grid_idx}/{len(param_grid)}: "
            f"n_estimators={params['n_estimators']}, "
            f"max_depth={params['max_depth']}, "
            f"min_samples_leaf={params['min_samples_leaf']}, "
            f"max_features={params['max_features']}"
        )

        for split in split_records:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            model = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=params["max_depth"],
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params["max_features"],
                random_state=base_random_state + split["repeat_idx"] * 100 + split["fold_idx"],
                n_jobs=-1,
                bootstrap=True,
                oob_score=False,
            )

            model.fit(x_valid[train_idx], y_valid[train_idx])
            pred = model.predict(x_valid[test_idx])

            fold_mae.append(float(mean_absolute_error(y_valid[test_idx], pred)))
            fold_rmse.append(rmse_np(y_valid[test_idx], pred))
            fold_r2.append(safe_r2(y_valid[test_idx], pred))

        row = {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": params["max_depth"],
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "max_features": params["max_features"],
            "cv_mae_mean": float(np.mean(fold_mae)),
            "cv_mae_std": float(np.std(fold_mae, ddof=0)),
            "cv_rmse_mean": float(np.mean(fold_rmse)),
            "cv_rmse_std": float(np.std(fold_rmse, ddof=0)),
            "cv_r2_mean": float(np.nanmean(fold_r2)),
            "cv_r2_std": float(np.nanstd(fold_r2, ddof=0)),
            "n_splits_total": int(len(split_records)),
        }
        cv_rows.append(row)

    cv_rows.sort(
        key=lambda r: (
            r["cv_rmse_mean"],
            r["cv_mae_mean"],
            -9999.0 if np.isnan(r["cv_r2_mean"]) else -r["cv_r2_mean"],
            r["min_samples_leaf"],
            r["n_estimators"],
        )
    )
    return cv_rows


def train_final_model(x_valid, y_valid, best_params, random_state):
    model = RandomForestRegressor(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=best_params["max_depth"],
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        max_features=best_params["max_features"],
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    model.fit(x_valid, y_valid)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train monthly RF-001 models with repeated spatial block CV.")
    parser.add_argument("--pred", default=DEFAULT_PRED_PATH, help="Input predictor NetCDF")
    parser.add_argument("--clim", default=DEFAULT_CLIM_PATH, help="Input monthly climatology NetCDF")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Output directory for monthly RF models")
    parser.add_argument("--metrics-csv", default=DEFAULT_METRICS_CSV, help="Output CSV for monthly training metrics")
    parser.add_argument("--importance-csv", default=DEFAULT_IMPORTANCE_CSV, help="Output CSV for feature importance")
    parser.add_argument("--fit-nc", default=DEFAULT_FIT_NC, help="Output NetCDF for fitted monthly climatology on training grid")
    parser.add_argument("--target-var", default=None, help="Optional monthly target variable name")
    parser.add_argument("--n-estimators-grid", default=DEFAULT_N_ESTIMATORS_GRID, help="Comma-separated n_estimators search grid")
    parser.add_argument("--max-depth-grid", default=DEFAULT_MAX_DEPTH_GRID, help="Comma-separated max_depth search grid, use None for unlimited depth")
    parser.add_argument("--min-samples-leaf-grid", default=DEFAULT_MIN_SAMPLES_LEAF_GRID, help="Comma-separated min_samples_leaf search grid")
    parser.add_argument("--max-features-grid", default=DEFAULT_MAX_FEATURES_GRID, help="Comma-separated max_features search grid, supports sqrt/log2/float/None")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="Number of folds in GroupKFold for each repeat")
    parser.add_argument("--cv-repeats", type=int, default=DEFAULT_CV_REPEATS, help="Number of repeated spatial block offsets")
    parser.add_argument("--spatial-block-size", type=int, default=DEFAULT_SPATIAL_BLOCK_SIZE, help="Spatial block size in grid cells")
    
    parser.add_argument("--clim-start-year", type=int, default=CLIM_START_YEAR, help="Start year for climatology baseline and yearly predictor aggregation")
    parser.add_argument("--train-months", default=TRAIN_MONTHS_STR, help="Comma-separated calendar months to train (default warm season May–Sep)")
    parser.add_argument("--clim-end-year", type=int, default=CLIM_END_YEAR, help="End year for climatology baseline and yearly predictor aggregation")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    args = parser.parse_args()

    global TARGET_VAR_NAME
    if args.target_var is not None:
        TARGET_VAR_NAME = args.target_var

    ensure_dir(args.model_dir)
    ensure_parent_dir(args.metrics_csv)
    ensure_parent_dir(args.importance_csv)
    ensure_parent_dir(args.fit_nc)

    x_maps, feature_names, feature_means, feature_stds, pred_mask, pred_lat, pred_lon = load_clim_predictor_maps(
        args.pred,
        clim_start_year=args.clim_start_year,
        clim_end_year=args.clim_end_year,
    )
    ds_tgt, month_name, tgt_lat_name, tgt_lon_name, target_var = load_target(args.clim)

    tgt_lat = ds_tgt[tgt_lat_name].values
    tgt_lon = ds_tgt[tgt_lon_name].values

    if pred_lat.size != tgt_lat.size or pred_lon.size != tgt_lon.size:
        raise ValueError("Predictor grid shape does not match target grid shape.")
    if not np.allclose(pred_lat, tgt_lat):
        raise ValueError("Predictor latitude coordinates do not match target latitude coordinates.")
    if not np.allclose(pred_lon, tgt_lon):
        raise ValueError("Predictor longitude coordinates do not match target longitude coordinates.")

    predictor_names = feature_names
    if len(predictor_names) != 22:
        raise ValueError("Expected 22 predictors from common/predictor_names.yaml, got %d: %s" % (len(predictor_names), predictor_names))

    log(f"Target variable: {target_var}")
    log(f"Predictor variables ({len(predictor_names)}): {', '.join(predictor_names)}")
    log(f"Yearly predictors aggregated over {args.clim_start_year}-{args.clim_end_year}")

    param_grid = build_param_grid(args)
    log(f"Hyperparameter combinations: {len(param_grid)}")

    x_stack = x_maps
    x_flat, valid_x, row_ids, col_ids = build_feature_matrix_from_maps(x_maps)
    _, nlat, nlon = x_stack.shape
    months = ds_tgt[month_name].values

    fit_cube = np.full((len(months), nlat, nlon), np.nan, dtype=np.float32)
    metrics_rows = []
    importance_rows = []

    y_da = ds_tgt[target_var].transpose(month_name, tgt_lat_name, tgt_lon_name).astype(np.float32)

    train_months = resolve_train_months([int(m) for m in months], args.train_months)
    log(f"Training months: {train_months}")

    for im, month_value in enumerate(months):
        month_int = int(month_value)
        if month_int not in train_months:
            log(f"Skipping month {month_int:02d} (not in --train-months)")
            continue
        log(f"Training month {month_int:02d}")

        y2d = y_da.sel({month_name: month_value}).values.astype(np.float32)
        y_flat = y2d.reshape(-1)

        valid_y = np.isfinite(y_flat)
        valid_mask = valid_x & valid_y

        n_valid = int(valid_mask.sum())
        if n_valid < max(20, args.cv_folds * 2):
            raise ValueError(
                f"Month {month_int:02d} has too few valid samples ({n_valid}) for repeated CV training."
            )

        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]

        split_records = build_spatial_cv_splits(
            row_ids=row_ids,
            col_ids=col_ids,
            valid_mask=valid_mask,
            n_splits=args.cv_folds,
            n_repeats=args.cv_repeats,
            block_size=max(1, int(args.spatial_block_size)),
            random_state=args.random_state + month_int,
        )
        log(f"Month {month_int:02d} spatial CV splits: {len(split_records)}")

        cv_rows = evaluate_param_grid(
            x_valid=x_valid,
            y_valid=y_valid,
            split_records=split_records,
            param_grid=param_grid,
            base_random_state=args.random_state + month_int * 1000,
        )
        best_row = cv_rows[0]
        best_params = {
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": best_row["max_depth"],
            "min_samples_leaf": int(best_row["min_samples_leaf"]),
            "max_features": best_row["max_features"],
        }

        log(
            f"Month {month_int:02d} best params -> "
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"min_samples_leaf={best_params['min_samples_leaf']}, "
            f"max_features={best_params['max_features']} | "
            f"cv_rmse={best_row['cv_rmse_mean']:.4f}, cv_r2={best_row['cv_r2_mean']:.4f}"
        )

        final_model = train_final_model(
            x_valid=x_valid,
            y_valid=y_valid,
            best_params=best_params,
            random_state=args.random_state + month_int,
        )

        pred_valid = final_model.predict(x_valid).astype(np.float32)
        final_train_mae = float(mean_absolute_error(y_valid, pred_valid))
        final_train_rmse = rmse_np(y_valid, pred_valid)
        final_train_r2 = safe_r2(y_valid, pred_valid)
        oob_score = getattr(final_model, "oob_score_", np.nan)

        model_path = os.path.join(args.model_dir, f"rf001" + f"_month_{month_int:02d}.joblib")
        payload = {
            "model": final_model,
            "feature_names": predictor_names,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "clim_start_year": int(args.clim_start_year),
            "clim_end_year": int(args.clim_end_year),
            "month": month_int,
            "target_variable": target_var,
            "predictor_file": args.pred,
            "target_file": args.clim,
            "random_state": args.random_state,
            "cv_strategy": "repeated_spatial_block_groupkfold",
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "spatial_block_size": int(args.spatial_block_size),
            "best_params": best_params,
            "best_cv_metrics": {
                "cv_mae_mean": float(best_row["cv_mae_mean"]),
                "cv_mae_std": float(best_row["cv_mae_std"]),
                "cv_rmse_mean": float(best_row["cv_rmse_mean"]),
                "cv_rmse_std": float(best_row["cv_rmse_std"]),
                "cv_r2_mean": float(best_row["cv_r2_mean"]),
                "cv_r2_std": float(best_row["cv_r2_std"]),
                "n_splits_total": int(best_row["n_splits_total"]),
            },
            "cv_results": cv_rows,
            "training_summary": {
                "n_valid": int(n_valid),
                "final_train_mae": final_train_mae,
                "final_train_rmse": final_train_rmse,
                "final_train_r2": final_train_r2,
                "oob_score": float(oob_score) if oob_score is not None else np.nan,
            },
        }
        dump(payload, model_path)

        y_fit_full = np.full(y_flat.shape, np.nan, dtype=np.float32)
        y_fit_full[valid_mask] = pred_valid
        fit_cube[im, :, :] = y_fit_full.reshape(nlat, nlon)

        metrics_rows.append({
            "month": month_int,
            "n_samples_total": n_valid,
            "cv_strategy": "repeated_spatial_block_groupkfold",
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "n_splits_total": int(best_row["n_splits_total"]),
            "spatial_block_size": int(args.spatial_block_size),
            "cv_mae_mean": float(best_row["cv_mae_mean"]),
            "cv_mae_std": float(best_row["cv_mae_std"]),
            "cv_rmse_mean": float(best_row["cv_rmse_mean"]),
            "cv_rmse_std": float(best_row["cv_rmse_std"]),
            "cv_r2_mean": float(best_row["cv_r2_mean"]),
            "cv_r2_std": float(best_row["cv_r2_std"]),
            "final_train_mae": final_train_mae,
            "final_train_rmse": final_train_rmse,
            "final_train_r2": final_train_r2,
            "oob_score": float(oob_score) if oob_score is not None else np.nan,
            "best_n_estimators": int(best_params["n_estimators"]),
            "best_max_depth": "" if best_params["max_depth"] is None else int(best_params["max_depth"]),
            "best_min_samples_leaf": int(best_params["min_samples_leaf"]),
            "best_max_features": best_params["max_features"],
            "model_path": model_path,
        })

        importances = final_model.feature_importances_
        for feat_name, feat_imp in zip(predictor_names, importances):
            importance_rows.append({
                "month": month_int,
                "feature": feat_name,
                "importance": float(feat_imp),
            })

        log(
            f"Month {month_int:02d} done | "
            f"cv_rmse={best_row['cv_rmse_mean']:.4f}, "
            f"final_train_rmse={final_train_rmse:.4f}, "
            f"oob={float(oob_score) if oob_score is not None else float('nan'):.4f}"
        )

    save_csv(metrics_rows, args.metrics_csv)
    save_csv(importance_rows, args.importance_csv)

    ds_fit = xr.Dataset(
        {
            "clim_tmin_rf001_fit": (
                ("month", "lat", "lon"),
                fit_cube.astype(np.float32),
            )
        },
        coords={
            "month": ds_tgt[month_name].values,
            "lat": ds_tgt[tgt_lat_name].values,
            "lon": ds_tgt[tgt_lon_name].values,
        },
        attrs={
            "title": "RF-001 fitted monthly climatology on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictor_file": args.pred,
            "target_file": args.clim,
            "target_variable": target_var,
            "feature_count": len(predictor_names),
            "feature_names": ",".join(predictor_names),
            "cv_strategy": "repeated_spatial_block_groupkfold",
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "spatial_block_size": int(args.spatial_block_size),
            "search_grid": json.dumps({
                "n_estimators_grid": parse_int_or_none_list(args.n_estimators_grid),
                "max_depth_grid": parse_int_or_none_list(args.max_depth_grid),
                "min_samples_leaf_grid": parse_int_or_none_list(args.min_samples_leaf_grid),
                "max_features_grid": parse_max_features_list(args.max_features_grid),
            }, ensure_ascii=False),
        },
    )

    ds_fit["clim_tmin_rf001_fit"].attrs = {
        "long_name": "RF-001 fitted monthly climatology on training grid",
        "units": ds_tgt[target_var].attrs.get("units", ""),
    }
    ds_fit["month"].attrs = {"long_name": "calendar month", "units": "1"}
    ds_fit["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_fit["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    log(f"Saving fitted monthly field: {args.fit_nc}")
    ds_fit.to_netcdf(
        args.fit_nc,
        encoding={"clim_tmin_rf001_fit": {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )

    ds_tgt.close()
    log(f"Metrics saved: {args.metrics_csv}")
    log(f"Feature importance saved: {args.importance_csv}")
    log("Done.")


if __name__ == "__main__":
    main()
