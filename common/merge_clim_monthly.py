#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge per-month UNet climatology train metrics, importance CSV, or fit NetCDF parts."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr


def merge_metrics(pattern: str, out_csv: str) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No metrics parts match: {pattern}")
    rows = [pd.read_csv(path) for path in files]
    df = pd.concat(rows, ignore_index=True).sort_values("month")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[MERGE] metrics -> {out_csv} ({len(df)} rows)")


def merge_csv(pattern: str, out_csv: str, sort_cols=None) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV parts match: {pattern}")
    rows = [pd.read_csv(path) for path in files]
    df = pd.concat(rows, ignore_index=True)
    if sort_cols:
        df = df.sort_values(sort_cols)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[MERGE] csv -> {out_csv} ({len(df)} rows)")


def merge_netcdf(part_glob: str, out_nc: str, var_name: str | None = None) -> None:
    files = sorted(glob.glob(part_glob))
    if not files:
        raise FileNotFoundError(f"No NetCDF parts match: {part_glob}")
    parts = [xr.open_dataset(path) for path in files]
    try:
        merged = xr.concat(parts, dim="month")
        if "month" in merged.coords:
            merged = merged.sortby("month")
        if var_name is None:
            var_name = next(iter(merged.data_vars))
        os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
        encoding = {var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
        merged.to_netcdf(out_nc, encoding=encoding)
        print(f"[MERGE] netcdf -> {out_nc} ({len(files)} parts, var={var_name})")
    finally:
        for ds in parts:
            ds.close()
    for path in files:
        os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge monthly UNet partial outputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_m = sub.add_parser("metrics")
    p_m.add_argument("--pattern", required=True)
    p_m.add_argument("--out", required=True)

    p_c = sub.add_parser("csv")
    p_c.add_argument("--pattern", required=True)
    p_c.add_argument("--out", required=True)
    p_c.add_argument("--sort-by", default="month,feature")

    p_n = sub.add_parser("netcdf")
    p_n.add_argument("--pattern", required=True)
    p_n.add_argument("--out", required=True)
    p_n.add_argument("--var", default=None)

    args = parser.parse_args()
    if args.cmd == "metrics":
        merge_metrics(args.pattern, args.out)
    elif args.cmd == "csv":
        sort_cols = [c.strip() for c in args.sort_by.split(",") if c.strip()]
        merge_csv(args.pattern, args.out, sort_cols=sort_cols)
    else:
        merge_netcdf(args.pattern, args.out, var_name=args.var)


if __name__ == "__main__":
    main()
