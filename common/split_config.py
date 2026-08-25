#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal split defaults for zcn downscaling pipelines."""

# Train-period monthly climatology baseline (also used to aggregate yearly predictors).
CLIM_START_YEAR = 1961
CLIM_END_YEAR = 2015

# Temporal hold-out period for model comparison and technical validation.
EVAL_START_YEAR = 2016
EVAL_END_YEAR = 2025

# Available year dimension in coastal_static predictors.
PREDICTOR_YEAR_START = 1985
PREDICTOR_YEAR_END = 2025

# Monthly climatology ML: train/predict with models for May–Sep only.
TRAIN_MONTHS = (5, 6, 7, 8, 9)
TRAIN_MONTHS_STR = "5,6,7,8,9"
