#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill NaN inside land/coastal mask; keep outside mask as NaN."""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage
except Exception:
    ndimage = None


def fill_nan_inside_mask_2d(arr: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Fill NaN cells inside *mask* from nearest valid neighbor; mask exterior stays NaN."""
    out = np.array(arr, dtype=np.float32, copy=True)
    need = mask & (~np.isfinite(out))
    if not np.any(need):
        out[~mask] = np.nan
        return out
    valid = mask & np.isfinite(out)
    if not np.any(valid):
        out[need] = float(fallback)
        out[~mask] = np.nan
        return out
    if ndimage is None:
        out[need] = float(np.nanmean(out[valid]))
        out[~mask] = np.nan
        return out
    _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    out[need] = out[idx[0][need], idx[1][need]]
    still = mask & (~np.isfinite(out))
    if np.any(still):
        out[still] = float(np.nanmean(out[valid]))
    out[~mask] = np.nan
    return out.astype(np.float32)


def fill_nan_inside_mask_chunk(chunk: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    out = np.empty_like(chunk, dtype=np.float32)
    for i in range(chunk.shape[0]):
        out[i] = fill_nan_inside_mask_2d(chunk[i], mask, fallback=fallback)
    return out


def finalize_masked_field_2d(arr: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    return fill_nan_inside_mask_2d(arr, mask, fallback=fallback)


def finalize_masked_field_chunk(chunk: np.ndarray, mask: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    return fill_nan_inside_mask_chunk(chunk, mask, fallback=fallback)
