# U-Net Downscaling of Daily Maximum and Minimum Temperature over Coastal China


This directory contains a two-stage spatial downscaling workflow for daily maximum temperature (Tmax) and daily minimum temperature (Tmin) over coastal China. The workflow first downscales daily temperature from 0.25° to 0.05°, then from 0.05° to 0.01°. U-Net models predict the monthly climatology, daily anomalies are interpolated to the target grid, and residual and scale-consistency corrections ensure that the high-resolution results remain consistent with the coarser source data after aggregation.


## 1. Workflow overview

Tmax and Tmin use the same processing chain:

```text
0.25° daily temperature
  ├─ Calculate the 1961–2015 monthly climatology and daily anomalies
  ├─ Predict the 0.05° monthly climatology with U-Net
  ├─ Interpolate 0.25° daily anomalies to 0.05°
  ├─ Monthly climatology + daily anomalies → initial 0.05° field
  ├─ Aggregate to 0.25°, calculate residuals, and interpolate residuals
  └─ Apply residual and strict scale-consistency corrections → final 0.05° field
        │
        └─ Use the final 0.05° field as input and repeat the workflow
           → final 0.01° field
```

Default temporal settings are defined in `common/split_config.py`:

- Climatology and training baseline: 1961–2015;
- Independent temporal holdout period: 2016–2025;
- Default U-Net training months: May through September;
- Configured predictor-year range: 1985–2025.


## 2. Directory structure

```text
├─ common/                         # Shared configuration and utilities
│  ├─ split_config.py              # Time ranges and modeled months
│  ├─ clim_month_tools.py          # Month filtering and parsing utilities
│  ├─ predictor_tools.py           # Predictor loading and standardization
│  ├─ mask_loader.py               # Coastal-mask loading and grid alignment
│  ├─ mask_fill.py                 # Nearest-neighbor filling inside masks
│  └─ merge_clim_monthly.py        # Merge independently trained month outputs
├─ tmax/
│  ├─ step01_unet005/              # Tmax: 0.25° → 0.05°
│  └─ step02_unet001/              # Tmax: 0.05° → 0.01°
├─ tmin/
│  ├─ step01_unet005/              # Tmin: 0.25° → 0.05°
│  └─ step02_unet001/              # Tmin: 0.05° → 0.01°
├─ requirements.txt
└─ README.md
```

## 3. Environment setup

Python 3.9 or later is recommended. Install the dependencies in an isolated virtual environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The primary dependencies are NumPy, pandas, xarray, netCDF4, PyTorch, and scikit-learn. `common/mask_fill.py` can use SciPy for efficient nearest-neighbor filling, so installing it is recommended:

```bash
python -m pip install scipy
```

For GPU training, install the PyTorch build that matches the system's CUDA version.

## 4. Required input data

At minimum, the workflow requires:

1. Coastal daily Tmax and/or Tmin NetCDF data at 0.25° resolution covering the required years;
2. Predictor NetCDF files on the 0.25°, 0.05°, and 0.01° grids;
3. Coastal or land-mask NetCDF files on the 0.05° and 0.01° grids;
4. Sufficient disk space for daily NetCDF intermediates, models, metrics, and final products.

Input data should have time, latitude, and longitude dimensions. Common coordinate names such as `time`, `lat`/`latitude`/`y`, and `lon`/`longitude`/`x` are detected automatically. If a target variable cannot be detected, specify it with the relevant `--var`, `--target-var`, `--clim-var`, `--anom-var`, `--raw-var`, or `--res-var` option.

A predictor file may contain:

- Static variables with dimensions `(lat, lon)`;
- Optional yearly variables with dimensions `(year, lat, lon)`;
- An optional `land_mask`; if absent, the full grid is treated as valid.


