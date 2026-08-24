#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_eval_temporal_holdout.py

Evaluate final downscaled daily field on independent temporal hold-out (2016-2025).
Spatial CV / spatial split are used only for model selection during training.
"""

import argparse
import os
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

from split_config import EVAL_END_YEAR, EVAL_START_YEAR
from eval_temporal_holdout import compute_holdout_metrics

DEFAULT_PRED = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb001/outputs/tmax_001_final.nc"
DEFAULT_REF = "/public/home/ggao001/users/xhang/Projects/CN_YANHAI_DOWN/01data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc"
DEFAULT_OUT = "/public/home/ggao001/users/xhang/Projects/zcn/tmax/04xgb/02exp/xgb001/metrics/tmax_001_holdout_2016_2025.csv"


def main():
    parser = argparse.ArgumentParser(description="Temporal hold-out evaluation on final downscaled product.")
    parser.add_argument("--pred", default=DEFAULT_PRED)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--out-csv", default=DEFAULT_OUT)
    parser.add_argument("--eval-start-year", type=int, default=EVAL_START_YEAR)
    parser.add_argument("--eval-end-year", type=int, default=EVAL_END_YEAR)
    args = parser.parse_args()
    compute_holdout_metrics(
        pred_path=args.pred,
        ref_path=args.ref,
        out_csv=args.out_csv,
        eval_start_year=args.eval_start_year,
        eval_end_year=args.eval_end_year,
    )


if __name__ == "__main__":
    main()
