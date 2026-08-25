# Daily Temperature Downscaling

This repository contains code for downscaling daily maximum temperature (Tmax) and minimum temperature (Tmin) over coastal China from 0.25° to 0.05° and then to 0.01°. It supports machine-learning downscaling, conventional spatial interpolation, residual correction, scale-consistency correction, and warm-season validation.


## 1. Methods

- **Random Forest (RF):** monthly models trained with repeated spatial block cross-validation
- **XGBoost (XGB):** monthly boosted-tree models with repeated spatial block cross-validation
- **U-Net:** convolutional models for spatially structured monthly climatology prediction
- **Bilinear interpolation, inverse distance weighting (IDW), and local ordinary kriging:** direct 0.25°→0.01° interpolation baselines

RF, XGB, and U-Net follow the same sequence: predict high-resolution monthly climatology, interpolate daily anomalies, reconstruct the daily field, apply residual correction, and enforce consistency with the coarser parent grid.

## 2. Directory structure

```text
code/
├─ common/          # Shared configuration and data-processing utilities
├─ interpolation/   # Bilinear, IDW, and kriging baselines
├─ validation/      # Shared validation metric formulas
├─ tests/           # Unit tests
├─ tmax/            # RF, XGBoost, and U-Net pipelines for Tmax
├─ tmin/            # RF, XGBoost, and U-Net pipelines for Tmin
├─ requirements.txt
├─ LICENSE
└─ CITATION.cff
```

Within each machine-learning method, `step02_*005` performs 0.25°→0.05° downscaling and `step03_*001` performs 0.05°→0.01° downscaling. Run scripts inside each stage in numeric order.

## 3. Data and temporal configuration

Required inputs include 0.25° daily Tmax/Tmin NetCDF data, predictor fields, and coastal masks. Predictor and target coordinates must be spatially aligned.

Place input files under `./data`, for example:

```text
./data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc
./data/coastal_daily_025/CN05.1_Tmin_1961_2025_daily_025x025_coastal.nc
./data/coastal_static/predictors_025_static.nc
./data/coastal_static/predictors_005_static.nc
./data/coastal_static/predictors_001_static.nc
./data/coastal_masks/coastal005mask.nc
./data/coastal_masks/coastal001mask.nc
```

Defaults in `common/split_config.py`:

- 1961–2015: climatology and model-training baseline
- 2016–2025: temporal holdout period for model comparison and technical validation
- May–September: default modeled and output months

RF, XGBoost, and U-Net all use the same 22 predictors listed in `common/predictor_names.yaml`. `land_mask` is used only as a spatial mask, not as a model input. Checkpoints trained with the old 23-channel layout (`land_mask` as an input) or with a different RF/XGBoost feature set cannot be loaded; retrain with this whitelist.

## 4. Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU-based U-Net training, install a PyTorch build compatible with the local CUDA version.

```bash
python -m unittest tests/test_metrics_sdr.py
```

## 5. Running the workflows

### 5.1 Machine-learning pipelines

Example: Tmax U-Net, 0.25°→0.05°. Repeat the same numeric order for RF (`tmax/rf/step02_rf005`), XGBoost (`tmax/xgb/step02_xgb005`), Tmin, and the 0.05°→0.01° `step03_*001` stages.

```bash
python tmax/unet/step02_unet005/00_make_clim_anom_025.py \
  --input ./data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc \
  --clim-out ./outputs/tmax/unet/unet005/interim/tmax_025_clim_monthly.nc \
  --anom-out ./outputs/tmax/unet/unet005/interim/tmax_025_anom_daily.nc

python tmax/unet/step02_unet005/01_train_unet005_monthly.py \
  --pred025 ./data/coastal_static/predictors_025_static.nc \
  --clim025 ./outputs/tmax/unet/unet005/interim/tmax_025_clim_monthly.nc \
  --model-dir ./outputs/tmax/unet/unet005/models \
  --epochs 1000 --patience 80 --base-channels 64 --random-state 42

python tmax/unet/step02_unet005/02_predict_clim005_unet.py \
  --pred005 ./data/coastal_static/predictors_005_static.nc \
  --model-dir ./outputs/tmax/unet/unet005/models \
  --out ./outputs/tmax/unet/unet005/interim/tmax_005_clim_unet.nc

python tmax/unet/step02_unet005/03_interp_anom025_to_005.py
python tmax/unet/step02_unet005/04_build_tmax005_raw.py
python tmax/unet/step02_unet005/05_compute_residual025.py
python tmax/unet/step02_unet005/06_interp_residual025_to_005.py
python tmax/unet/step02_unet005/07_correct_tmax005.py
python tmax/unet/step02_unet005/08_consistency_correct_005.py
python tmax/unet/step02_unet005/18_eval_temporal_holdout.py
```

RF and XGBoost use the same step numbers under `tmax/rf/step02_rf005` and `tmax/xgb/step02_xgb005`. Use `--help` on any script to inspect arguments.

### 5.2 Interpolation baselines

The published interpolation products use these settings (Tmax and Tmin share them):

- IDW: `k=12`, power `p=2.0`
- Local ordinary kriging: `k=12`, spherical covariance with fixed `range=1.0` and `nugget=1e-6`
- Distances are Euclidean in latitude/longitude degrees

```bash
python interpolation/interpolate_tmax_025_to_001_methods.py \
  --input ./data/coastal_daily_025/CN05.1_Tmax_1961_2025_daily_025x025_coastal.nc \
  --mask ./data/coastal_masks/coastal001mask.nc \
  --methods all \
  --idw-k 12 --idw-power 2.0 \
  --kriging-k 12 --kriging-range 1.0 --kriging-nugget 1.0e-6 \
  --out-bilinear ./outputs/interpolation/tmax/01bilinear/tmax_001_bilinear.nc \
  --out-idw ./outputs/interpolation/tmax/03IDW/tmax_001_idw.nc \
  --out-kriging ./outputs/interpolation/tmax/02kriging/tmax_001_kriging.nc
```

Use `interpolation/interpolate_tmin_025_to_001_methods.py` for Tmin.

## 6. Validation notes

`validation/metrics_core.py` provides Bias, MAE, RMSE, R², correlation, standard-deviation ratio (SDR), Perkins skill score (PSS), percentile bias, heat-event occurrence and intensity statistics, diurnal temperature range, station-level evaluation, and parent-grid aggregation checks.

The 2016–2025 window is a temporal holdout for technical comparison. CN05.1 and station observations are not independent samples of one another.
