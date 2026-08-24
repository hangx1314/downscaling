# Coastal China Daily Temperature Downscaling

This repository contains code for downscaling daily maximum temperature (Tmax) and minimum temperature (Tmin) over coastal China from 0.25° to 0.05° and then to 0.01°. It supports machine-learning downscaling, conventional spatial interpolation, residual correction, scale-consistency correction, and warm-season validation.

> This is a code-only package. Input data, masks, trained models, and outputs are not included. Most scripts contain default paths from the original Linux environment (`/public/home/...`); use command-line arguments to provide paths for your own system.

## 1. Methods

The repository includes the following downscaling approaches:

- **Random Forest (RF):** monthly models trained with repeated spatial block cross-validation;
- **XGBoost (XGB):** monthly boosted-tree models with repeated spatial block cross-validation;
- **U-Net:** convolutional models for spatially structured monthly climatology prediction;
- **Bilinear interpolation, inverse distance weighting (IDW), and local ordinary kriging:** direct 0.25°→0.01° interpolation baselines for Tmax and Tmin.

The RF, XGB, and U-Net pipelines use the same general sequence: predict the high-resolution monthly climatology, interpolate daily anomalies, reconstruct the daily field, apply residual correction, and enforce consistency with the coarser parent grid.

## 2. Directory structure

```text
00code/
├─ common/          # Shared configuration and data-processing utilities
├─ interpolation/   # Bilinear, IDW, and kriging baselines
├─ validation/      # Shared validation metric formulas
├─ tmax/
│  ├─ rf/           # RF pipelines for Tmax
│  ├─ xgb/          # XGBoost pipelines for Tmax
│  └─ unet/         # U-Net pipelines for Tmax
├─ tmin/
│  ├─ rf/           # RF pipelines for Tmin
│  ├─ xgb/          # XGBoost pipelines for Tmin
│  └─ unet/         # U-Net pipelines for Tmin
└─ requirements.txt
```

Within each machine-learning method, `step02_*005` performs 0.25°→0.05° downscaling and `step03_*001` performs 0.05°→0.01° downscaling. Run scripts inside each stage in numeric order.

## 3. Data and temporal configuration

Required inputs include 0.25° daily Tmax/Tmin NetCDF data, predictor fields on the required grids, and coastal masks for the target grids. Predictor and target coordinates must be spatially aligned.

The default settings in `common/split_config.py` are:

- 1961–2015: climatology and model-training baseline;
- 2016–2025: **temporal holdout period for model comparison and technical validation**;
- May–September: default modeled and output months.

Changing the final output months requires updating `TRAIN_MONTHS` in `common/split_config.py` and regenerating downstream products.

## 4. Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install scipy
```

SciPy is used by the interpolation and nearest-neighbor filling code but is not currently listed in `requirements.txt`. For GPU-based U-Net training, install a PyTorch build compatible with the local CUDA version.

## 5. Running the workflows

For RF, XGB, or U-Net, select `tmax` or `tmin`, complete the `step02_*005` stage, and then run `step03_*001`. Typical numbered steps are:

```text
Create climatology/anomalies → train monthly models → predict climatology
→ interpolate anomalies → build the initial daily field
→ calculate/interpolate residuals → apply residual correction
→ enforce parent-grid consistency → evaluate
```

Use `--help` to inspect the parameters of any script and pass explicit data and output paths. For example:

```bash
python tmax/rf/step02_rf005/01_train_rf005_monthly.py --help
python tmax/xgb/step02_xgb005/01_train_xgb005_monthly.py --help
python tmax/unet/step02_unet005/01_train_unet005_monthly.py --help
```

The interpolation baselines can run one or all supported methods:

```bash
python interpolation/interpolate_tmax_025_to_001_methods.py \
  --input /path/to/tmax_025.nc \
  --mask /path/to/coastal001mask.nc \
  --methods all \
  --out-bilinear /path/to/tmax_001_bilinear.nc \
  --out-idw /path/to/tmax_001_idw.nc \
  --out-kriging /path/to/tmax_001_kriging.nc
```

Use the corresponding Tmin script for minimum temperature.

## 6. Validation and implementation notes

`validation/metrics_core.py` provides shared formulas for Bias, MAE, RMSE, R², correlation, standard-deviation ratio, pattern similarity score, percentile bias, heat-event occurrence and intensity statistics, diurnal temperature range, station-level evaluation, and parent-grid aggregation checks. These routines support comparison among RF, XGB, U-Net, and interpolation baselines during the 2016–2025 technical validation period.

The current code tree imports several project-specific modules that are not included in `common/`: `interp_chunk_tools.py`, `build_raw_chunk_tools.py`, `correct_chunk_tools.py`, and `eval_temporal_holdout.py`. Restore these files from the complete project before running affected pipeline and evaluation scripts. A successful syntax check does not confirm that these runtime modules or input datasets are available.
