#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_train_unet001_monthly.py

Train monthly UNet-001 models:
- X: 0.05-degree predictors aggregated from 0.01-degree static predictors
- y: 0.05-degree monthly climatology of Tmin

Inputs:
- predictors_005_from001.nc
- tmin_005_clim_monthly.nc

Outputs:
- unet001_month_01.pt ... unet001_month_12.pt
- unet001_train_metrics.csv
- unet001_feature_importance_proxy.csv
- tmin_005_clim_unet001_fit.nc
"""

import os
import csv
import math
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import torch
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
from clim_month_tools import resolve_train_months, filter_monthly_dataset_to_output_months
from predictor_tools import load_clim_predictor_maps, build_feature_matrix_from_maps

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# =========================================================
# Default paths
# =========================================================
DEFAULT_PRED005_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_static/predictors_005_static.nc"
DEFAULT_CLIM005_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/05unet/02exp/unet001/interim/tmin_005_clim_monthly.nc"

DEFAULT_MODEL_DIR = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/05unet/02exp/unet001/models"
DEFAULT_METRICS_CSV = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/05unet/02exp/unet001/models/unet001_train_metrics.csv"
DEFAULT_IMPORTANCE_CSV = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/05unet/02exp/unet001/models/unet001_feature_importance_proxy.csv"
DEFAULT_FIT_NC = "/public/home/ggao001/users/xhang/Projects/zcn/tmin/05unet/02exp/unet001/interim/tmin_005_clim_unet001_fit.nc"

TARGET_VAR_NAME = None


# =========================================================
# Utilities
# =========================================================
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


def find_monthly_target_var(
    ds: xr.Dataset,
    month_name: str,
    lat_name: str,
    lon_name: str,
    preferred_names=None,
):
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

    predictor_vars = []
    for name, da in ds.data_vars.items():
        if lat_name in da.dims and lon_name in da.dims and da.ndim == 2:
            predictor_vars.append(name)

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


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def save_csv(rows, out_csv):
    ensure_parent_dir(out_csv)
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




# =========================================================
# Basic UNet architecture
# Replaced with the user's specified U-Net structure.
# =========================================================
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        if d5.shape[-2:] != e4.shape[-2:]:
            d5 = F.interpolate(d5, size=e4.shape[-2:], mode="bilinear", align_corners=True)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if d4.shape[-2:] != e3.shape[-2:]:
            d4 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=True)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if d3.shape[-2:] != e2.shape[-2:]:
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=True)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=True)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out



# =========================================================
# Data preparation
# =========================================================
def build_input_tensor(
    ds_pred: xr.Dataset,
    predictor_names,
    lat_name: str,
    lon_name: str,
    target_2d: np.ndarray,
):
    target_land_mask = np.isfinite(target_2d)

    x_list = []
    means = []
    stds = []

    for name in predictor_names:
        arr = ds_pred[name].transpose(lat_name, lon_name).values.astype(np.float32)

        valid = np.isfinite(arr) & target_land_mask
        if valid.sum() == 0:
            mean_val = 0.0
            std_val = 1.0
        else:
            mean_val = float(np.mean(arr[valid]))
            std_val = float(np.std(arr[valid]))
            if std_val < 1e-6:
                std_val = 1.0

        arr_std = np.where(np.isfinite(arr), (arr - mean_val) / std_val, 0.0).astype(np.float32)

        x_list.append(arr_std)
        means.append(mean_val)
        stds.append(std_val)

    x = np.stack(x_list, axis=0).astype(np.float32)

    target_mean = float(np.mean(target_2d[target_land_mask]))
    target_std = float(np.std(target_2d[target_land_mask]))
    if target_std < 1e-6:
        target_std = 1.0

    y = np.full_like(target_2d, 0.0, dtype=np.float32)
    y[target_land_mask] = ((target_2d[target_land_mask] - target_mean) / target_std).astype(np.float32)

    return x, y, target_land_mask.astype(np.bool_), means, stds, target_mean, target_std


def build_train_test_masks(
    land_mask: np.ndarray,
    test_fraction: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    land_indices = np.flatnonzero(land_mask.reshape(-1))
    if land_indices.size == 0:
        raise ValueError("No valid land cells found in target mask.")

    n_test = int(round(land_indices.size * test_fraction))
    if n_test < 1:
        n_test = 1
    if n_test >= land_indices.size:
        n_test = max(1, land_indices.size - 1)

    test_idx = rng.choice(land_indices, size=n_test, replace=False)
    train_flat = land_mask.reshape(-1).copy()
    test_flat = np.zeros_like(train_flat, dtype=bool)

    train_flat[test_idx] = False
    test_flat[test_idx] = True

    train_mask = train_flat.reshape(land_mask.shape)
    test_mask = test_flat.reshape(land_mask.shape)

    return train_mask, test_mask


def masked_l1_l2_loss(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    diff = pred[mask] - target[mask]
    l1 = torch.mean(torch.abs(diff))
    l2 = torch.mean(diff ** 2)
    return l1 + 0.5 * l2


# =========================================================
# Training
# =========================================================
def train_single_month(
    month_int: int,
    x_np: np.ndarray,
    y_np: np.ndarray,
    train_mask_np: np.ndarray,
    test_mask_np: np.ndarray,
    feature_names,
    args,
    device,
):
    in_channels, height, width = x_np.shape

    model = U_Net(in_ch=in_channels, out_ch=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = torch.from_numpy(x_np[None, ...]).to(device)
    y = torch.from_numpy(y_np[None, None, ...]).to(device)
    train_mask = torch.from_numpy(train_mask_np[None, None, ...]).to(device)
    test_mask = torch.from_numpy(test_mask_np[None, None, ...]).to(device)

    best_state = None
    best_score = math.inf
    best_epoch = -1
    patience_counter = 0

    train_mask_bool = train_mask > 0
    test_mask_bool = test_mask > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(x)
        loss = masked_l1_l2_loss(pred, y, train_mask_bool)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(x)

            train_loss_eval = masked_l1_l2_loss(pred_eval, y, train_mask_bool).item()

            if test_mask_bool.sum() > 0:
                test_loss_eval = masked_l1_l2_loss(pred_eval, y, test_mask_bool).item()
                score = test_loss_eval
            else:
                test_loss_eval = float("nan")
                score = train_loss_eval

        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
        else:
            patience_counter += 1

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            log(
                f"Month {month_int:02d} | "
                f"epoch={epoch:04d} | "
                f"train_loss={train_loss_eval:.6f} | "
                f"test_loss={test_loss_eval if np.isfinite(test_loss_eval) else float('nan'):.6f}"
            )

        if patience_counter >= args.patience:
            log(f"Month {month_int:02d} early stopped at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Month {month_int:02d} failed to save a best model state.")

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_best = model(x).detach().cpu().numpy()[0, 0]

    return model, pred_best, best_state


def first_layer_weight_proxy(model: nn.Module, feature_names):
    weight = model.Conv1.conv[0].weight.detach().cpu().numpy()
    score = np.mean(np.abs(weight), axis=(0, 2, 3))
    rows = []
    for feat, val in zip(feature_names, score):
        rows.append((feat, float(val)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train monthly UNet-001 models.")
    parser.add_argument("--pred005", default=DEFAULT_PRED005_PATH)
    parser.add_argument("--clim005", default=DEFAULT_CLIM005_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics-csv", default=DEFAULT_METRICS_CSV)
    parser.add_argument("--importance-csv", default=DEFAULT_IMPORTANCE_CSV)
    parser.add_argument("--fit-nc", default=DEFAULT_FIT_NC)
    parser.add_argument("--target-var", default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    
    parser.add_argument("--clim-start-year", type=int, default=CLIM_START_YEAR, help="Start year for climatology baseline and yearly predictor aggregation")
    parser.add_argument("--train-months", default=TRAIN_MONTHS_STR, help="Comma-separated calendar months to train (default warm season May–Sep)")
    parser.add_argument("--clim-end-year", type=int, default=CLIM_END_YEAR, help="End year for climatology baseline and yearly predictor aggregation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        help="Train only one calendar month (for multi-GPU parallel launch).",
    )
    args = parser.parse_args()

    global TARGET_VAR_NAME
    if args.target_var is not None:
        TARGET_VAR_NAME = args.target_var

    ensure_dir(args.model_dir)
    ensure_parent_dir(args.metrics_csv)
    ensure_parent_dir(args.importance_csv)
    ensure_parent_dir(args.fit_nc)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log(f"Using device: {device}")

    x_maps, predictor_names, feature_means, feature_stds, pred_mask, pred_lat, pred_lon = load_clim_predictor_maps(
        args.pred005,
        clim_start_year=args.clim_start_year,
        clim_end_year=args.clim_end_year,
    )
    ds_tgt, month_name, tgt_lat_name, tgt_lon_name, target_var = load_target(args.clim005)

    tgt_lat = ds_tgt[tgt_lat_name].values
    tgt_lon = ds_tgt[tgt_lon_name].values

    if pred_lat.size != tgt_lat.size or pred_lon.size != tgt_lon.size:
        raise ValueError("Predictor grid shape does not match target grid shape.")
    if not np.allclose(pred_lat, tgt_lat):
        raise ValueError("Predictor latitude coordinates do not match target latitude coordinates.")
    if not np.allclose(pred_lon, tgt_lon):
        raise ValueError("Predictor longitude coordinates do not match target longitude coordinates.")

    if len(predictor_names) == 0:
        raise ValueError("No predictor variables left after exclusion.")

    log(f"Target variable: {target_var}")
    log(f"Predictor variables ({len(predictor_names)}): {', '.join(predictor_names)}")
    log(f"Yearly predictors aggregated over {args.clim_start_year}-{args.clim_end_year}")

    months = ds_tgt[month_name].values
    nlat = ds_tgt.sizes[tgt_lat_name]
    nlon = ds_tgt.sizes[tgt_lon_name]

    fit_cube = np.full((len(months), nlat, nlon), np.nan, dtype=np.float32)
    metrics_rows = []
    importance_rows = []

    y_da = ds_tgt[target_var].transpose(month_name, tgt_lat_name, tgt_lon_name).astype(np.float32)

    train_months = resolve_train_months([int(m) for m in months], args.train_months)
    if args.month is not None:
        if args.month not in train_months:
            raise ValueError(
                f"--month {args.month} is not in training months {train_months}."
            )
        train_months = [args.month]
    log(f"Training months: {train_months}")

    for im, month_value in enumerate(months):
        month_int = int(month_value)
        if month_int not in train_months:
            log(f"Skipping month {month_int:02d} (not in --train-months)")
            continue
        log(f"Training month {month_int:02d}")

        y2d = y_da.sel({month_name: month_value}).values.astype(np.float32)
        land_mask = np.isfinite(y2d) & pred_mask
        y_mean = float(np.mean(y2d[land_mask]))
        y_std = float(np.std(y2d[land_mask]))
        if y_std < 1e-6:
            y_std = 1.0
        y_std_np = np.full_like(y2d, 0.0, dtype=np.float32)
        y_std_np[land_mask] = ((y2d[land_mask] - y_mean) / y_std).astype(np.float32)
        x_np = x_maps
        feat_means, feat_stds = feature_means, feature_stds

        train_mask_np, test_mask_np = build_train_test_masks(
            land_mask=land_mask,
            test_fraction=args.test_size,
            random_state=args.random_state + month_int,
        )

        model, pred_std_np, best_state = train_single_month(
            month_int=month_int,
            x_np=x_np,
            y_np=y_std_np,
            train_mask_np=train_mask_np,
            test_mask_np=test_mask_np,
            feature_names=predictor_names,
            args=args,
            device=device,
        )

        pred_np = pred_std_np * y_std + y_mean
        pred_np[~land_mask] = np.nan
        fit_cube[im, :, :] = pred_np.astype(np.float32)

        train_true = y2d[train_mask_np]
        train_pred = pred_np[train_mask_np]
        test_true = y2d[test_mask_np]
        test_pred = pred_np[test_mask_np]

        train_mae = mae_np(train_true, train_pred)
        train_rmse = rmse_np(train_true, train_pred)
        train_r2 = r2_np(train_true, train_pred)

        if test_true.size > 0:
            test_mae = mae_np(test_true, test_pred)
            test_rmse = rmse_np(test_true, test_pred)
            test_r2 = r2_np(test_true, test_pred)
        else:
            test_mae = float("nan")
            test_rmse = float("nan")
            test_r2 = float("nan")

        model_path = os.path.join(args.model_dir, f"unet001_month_{month_int:02d}.pt")
        payload = {
            "month": month_int,
            "feature_names": predictor_names,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "clim_start_year": int(args.clim_start_year),
            "clim_end_year": int(args.clim_end_year),
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "clim_start_year": int(args.clim_start_year),
            "clim_end_year": int(args.clim_end_year),
            "feature_means": feat_means,
            "feature_stds": feat_stds,
            "target_mean": y_mean,
            "target_std": y_std,
            "target_variable": target_var,
            "predictor_file": args.pred005,
            "target_file": args.clim005,
            "random_state": args.random_state,
            "base_channels": args.base_channels,
            "arch_name": "basic_unet",
            "model_state_dict": model.state_dict(),
        }
        torch.save(payload, model_path)

        metrics_rows.append(
            {
                "month": month_int,
                "n_land_total": int(land_mask.sum()),
                "n_train": int(train_mask_np.sum()),
                "n_test": int(test_mask_np.sum()),
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "best_epoch": int(best_state["best_epoch"]),
                "best_score": float(best_state["best_score"]),
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "base_channels": args.base_channels,
                "model_path": model_path,
            }
        )

        proxy_rows = first_layer_weight_proxy(model, predictor_names)
        for feat_name, proxy_score in proxy_rows:
            importance_rows.append(
                {
                    "month": month_int,
                    "feature": feat_name,
                    "importance_proxy": proxy_score,
                }
            )

        log(
            f"Month {month_int:02d} done | "
            f"train_rmse={train_rmse:.4f}, test_rmse={test_rmse if np.isfinite(test_rmse) else float('nan'):.4f}"
        )

    if args.month is not None:
        save_csv(metrics_rows, f"{args.metrics_csv}.month{args.month:02d}.csv")
        save_csv(importance_rows, f"{args.importance_csv}.month{args.month:02d}.csv")
        month_row = int(np.where(np.asarray(months) == args.month)[0][0])
        ds_fit_part = xr.Dataset(
            {
                "clim_tmin_unet001_fit": (("month", "lat", "lon"), fit_cube[month_row:month_row + 1].astype(np.float32)),
            },
            coords={
                "month": np.asarray([args.month], dtype=np.int32),
                "lat": ds_tgt[tgt_lat_name].values,
                "lon": ds_tgt[tgt_lon_name].values,
            },
        )
        fit_part_path = f"{args.fit_nc}.month{args.month:02d}.nc"
        ds_fit_part.to_netcdf(
            fit_part_path,
            encoding={"clim_tmin_unet001_fit": {"zlib": True, "complevel": 4, "dtype": "float32"}},
        )
        ds_tgt.close()
        log(f"Month {args.month:02d} done (parallel mode).")
        return

    save_csv(metrics_rows, args.metrics_csv)
    save_csv(importance_rows, args.importance_csv)

    ds_fit = xr.Dataset(
        {
            "clim_tmin_unet001_fit": (
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
            "title": "UNet-001 fitted monthly climatology on 0.05-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictor_file": args.pred005,
            "target_file": args.clim005,
            "target_variable": target_var,
            "feature_count": len(predictor_names),
            "feature_names": ",".join(predictor_names),
        },
    )

    ds_fit["clim_tmin_unet001_fit"].attrs = {
        "long_name": "UNet-001 fitted monthly climatology on training grid",
        "units": ds_tgt[target_var].attrs.get("units", ""),
    }
    ds_fit["month"].attrs = {"long_name": "calendar month", "units": "1"}
    ds_fit["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_fit["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    ds_fit = filter_monthly_dataset_to_output_months(ds_fit)

    log(f"Saving fitted monthly field: {args.fit_nc}")
    ds_fit.to_netcdf(
        args.fit_nc,
        encoding={"clim_tmin_unet001_fit": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    )

    ds_tgt.close()
    log(f"Metrics saved: {args.metrics_csv}")
    log(f"Feature proxy saved: {args.importance_csv}")
    log("Done.")


if __name__ == "__main__":
    main()