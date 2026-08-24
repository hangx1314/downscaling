#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_predict_clim001_unet.py

Predict 0.01-degree monthly climatology using trained UNet-001 monthly models.

Inputs:
- predictors_001_static.nc
- unet001_month_01.pt ... unet001_month_12.pt

Output:
- tmax_001_clim_unet.nc
"""

import os
import argparse
from datetime import datetime

import os
# Pin one GPU before torch initializes CUDA (avoids invalid device index on multi-GPU nodes).
if os.environ.get("ZCN_UNET_PREDICT_SINGLE_GPU", "1") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import xarray as xr
import torch
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
from split_config import CLIM_START_YEAR, CLIM_END_YEAR, TRAIN_MONTHS_STR
from clim_month_tools import (
    resolve_train_months,
    find_first_model_path,
    load_source_clim_on_grid,
    subset_monthly_cube_to_output_months,
    output_months_attr,
)
from predictor_tools import load_clim_predictor_maps, build_feature_matrix_from_maps

import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Default paths
# =========================================================
DEFAULT_PRED001_PATH = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_static/predictors_001_static.nc"
DEFAULT_MODEL_DIR = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/05unet/02exp/unet001/models"
DEFAULT_SOURCE_CLIM = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/05unet/02exp/unet001/interim/tmax_005_clim_monthly.nc"
DEFAULT_OUT_PATH = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/05unet/02exp/unet001/interim/tmax_001_clim_unet.nc"

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



def get_model_path(model_dir: str, month_int: int) -> str:
    return os.path.join(model_dir, f"unet001_month_{month_int:02d}.pt")


def build_input_tensor_from_payload(vars_proc: dict, payload: dict):
    feature_names = list(payload["feature_names"])
    feat_means = payload["feature_means"]
    feat_stds = payload["feature_stds"]

    missing = [name for name in feature_names if name not in vars_proc]
    if len(missing) > 0:
        raise KeyError("Missing predictor variables required by model: " + ", ".join(missing))

    x_list = []
    for name, mean_val, std_val in zip(feature_names, feat_means, feat_stds):
        arr = vars_proc[name].astype(np.float32)
        if std_val is None or abs(float(std_val)) < 1e-6:
            std_val = 1.0
        arr_std = np.where(np.isfinite(arr), (arr - float(mean_val)) / float(std_val), 0.0).astype(np.float32)
        x_list.append(arr_std)

    x = np.stack(x_list, axis=0).astype(np.float32)
    return x, feature_names



def make_blend_weight(height: int, width: int, edge: int = 16) -> np.ndarray:
    if edge <= 0:
        return np.ones((height, width), dtype=np.float32)

    wy = np.ones(height, dtype=np.float32)
    wx = np.ones(width, dtype=np.float32)

    ey = min(edge, max(1, height // 2))
    ex = min(edge, max(1, width // 2))

    ramp_y = np.linspace(1.0 / ey, 1.0, ey, dtype=np.float32)
    ramp_x = np.linspace(1.0 / ex, 1.0, ex, dtype=np.float32)

    wy[:ey] = ramp_y
    wy[-ey:] = ramp_y[::-1]
    wx[:ex] = ramp_x
    wx[-ex:] = ramp_x[::-1]

    return np.outer(wy, wx).astype(np.float32)


def predict_full_or_tiled(
    model: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    tile_size: int = 96,
    overlap: int = 24,
) -> np.ndarray:
    _, height, width = x_np.shape

    if max(height, width) <= tile_size:
        x = torch.from_numpy(x_np[None, ...]).to(device)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()[0, 0]
        return pred.astype(np.float32)

    stride = max(1, tile_size - overlap)
    pred_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    ys = list(range(0, max(1, height - tile_size + 1), stride))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if ys[-1] != height - tile_size:
        ys.append(max(0, height - tile_size))
    if xs[-1] != width - tile_size:
        xs.append(max(0, width - tile_size))

    for y0 in ys:
        for x0 in xs:
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)

            x_tile = x_np[:, y0:y1, x0:x1]
            blend = make_blend_weight(y1 - y0, x1 - x0, edge=max(8, overlap // 2))

            x_tensor = torch.from_numpy(x_tile[None, ...]).to(device)
            with torch.no_grad():
                pred_tile = model(x_tensor).detach().cpu().numpy()[0, 0].astype(np.float32)

            pred_sum[y0:y1, x0:x1] += pred_tile * blend
            weight_sum[y0:y1, x0:x1] += blend

    out = pred_sum / np.clip(weight_sum, 1e-6, None)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Predict 0.01-degree monthly climatology using trained UNet-001 models.")
    parser.add_argument("--pred001", default=DEFAULT_PRED001_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-clim", default=DEFAULT_SOURCE_CLIM, help="Source monthly climatology for non-trained months (bilinear to target grid)")
    parser.add_argument("--train-months", default=TRAIN_MONTHS_STR, help="Months with trained ML models (others use --source-clim)")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--pred-lat", default=None)
    parser.add_argument("--pred-lon", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tile-size", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=24)
    args = parser.parse_args()

    ensure_parent_dir(args.out)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    log(f"Using device: {device}")

    train_months = resolve_train_months(range(1, 13), args.train_months)
    first_model_path = find_first_model_path(args.model_dir, "unet001", train_months)
    first_payload = torch.load(first_model_path, map_location="cpu")
    feature_names_ref = list(first_payload["feature_names"])
    x_maps, _, _, _, pred_mask_arr, lat_proc, lon_proc = load_clim_predictor_maps(
        args.pred001,
        feature_names=feature_names_ref,
        stats=(first_payload["feature_means"], first_payload["feature_stds"]),
        clim_start_year=int(first_payload.get("clim_start_year", CLIM_START_YEAR)),
        clim_end_year=int(first_payload.get("clim_end_year", CLIM_END_YEAR)),
    )
    lat_raw = lat_proc.copy()
    lon_raw = lon_proc.copy()
    lat_ascending = bool(np.all(np.diff(lat_proc) > 0))
    lon_ascending = bool(np.all(np.diff(lon_proc) > 0))
    nlat = lat_proc.size
    nlon = lon_proc.size
    land_mask = pred_mask_arr
    log(f"Using feature names: {', '.join(feature_names_ref)}")

    month_values = np.arange(1, 13, dtype=np.int32)
    clim_cube_proc = np.full((12, nlat, nlon), np.nan, dtype=np.float32)


    for im, month_int in enumerate(month_values):
        model_path = get_model_path(args.model_dir, month_int)
        if not os.path.exists(model_path):
            log(f"Month {month_int:02d}: model missing ({model_path}); will use source clim fallback")
            continue

        log(f"Loading model for month {month_int:02d}: {model_path}")
        payload = torch.load(model_path, map_location="cpu")

        if "model_state_dict" not in payload or "feature_names" not in payload:
            raise KeyError(f"Invalid model payload: {model_path}")

        feature_names = list(payload["feature_names"])

        if feature_names != feature_names_ref:
            raise ValueError(
                f"Feature names mismatch in month {month_int:02d} model.\n"
                f"Reference: {feature_names_ref}\n"
                f"Current:   {feature_names}"
            )

        x_np = x_maps

        model = U_Net(in_ch=x_np.shape[0], out_ch=1).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        pred_std = predict_full_or_tiled(
            model=model,
            x_np=x_np,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        target_mean = float(payload["target_mean"])
        target_std = float(payload["target_std"])
        pred = pred_std * target_std + target_mean
        pred[~land_mask] = np.nan

        clim_cube_proc[im, :, :] = pred.astype(np.float32)
        log(f"Predicted month {month_int:02d}")

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
            "clim_tmax_unet": (("month", "lat", "lon"), clim_out.astype(np.float32))
        },
        coords={
            "month": output_month_values,
            "lat": lat_raw,
            "lon": lon_raw,
        },
        attrs={
            "title": "UNet-001 predicted monthly climatology on 0.01-degree grid",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictor_file": args.pred001,
            "model_dir": args.model_dir,
            "feature_names": ",".join(feature_names_ref) if feature_names_ref is not None else "",
            "source_grid": "0.01_degree",
            "prediction_type": "monthly_climatology",
            "output_calendar_months": output_months_attr(train_months),
        },
    )

    ds_out["clim_tmax_unet"].attrs = {
        "long_name": "UNet-001 predicted monthly climatology of Tmax on 0.01-degree grid",
        "units": "",
    }
    ds_out["month"].attrs = {"long_name": "calendar month", "units": "1"}
    ds_out["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds_out["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    encoding = {"clim_tmax_unet": {"zlib": True, "complevel": 4, "dtype": "float32"}}

    log(f"Saving output: {args.out}")
    ds_out.to_netcdf(args.out, encoding=encoding)
    log("Done.")


if __name__ == "__main__":
    main()