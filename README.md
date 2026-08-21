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

## 5. Execution order

Tmax and Tmin can be processed independently. For either variable, complete the 0.25°→0.05° stage before starting the 0.05°→0.01° stage.


## 6. Important options and resource considerations

- `--device {auto,cpu,cuda}`: training or prediction device;
- `--epochs`, `--patience`, `--lr`, `--weight-decay`: optimization and early-stopping settings;
- `--base-channels`: base U-Net channel count; this strongly affects GPU memory usage;
- `--test-size`, `--random-state`: internal spatial split settings used during training;
- `--tile-size`, `--overlap`: tile dimensions and overlap for prediction on large grids;
- `--time-chunk`: time-block size for daily interpolation, aggregation, correction, or writing;
- `--train-months`: comma-separated modeled months, such as `5,6,7,8,9`;
- `--clim-start-year`, `--clim-end-year`: climatology baseline period.

Daily 0.01° datasets can be very large. Validate the complete workflow first with a short time range, a small spatial subset, and a small `--time-chunk`. Before a full run, confirm that all inputs use compatible time axes, coordinate orientation, units, and missing-data masks.

## 7. Known missing modules in this code-only package

Scripts that depend on these modules currently fail at startup with `ModuleNotFoundError`. This affects anomaly and residual interpolation, initial-field construction, some correction steps, and temporal-holdout evaluation. Copy the missing modules from the complete project into `common/` before running the full workflow. Installing `requirements.txt` alone will not resolve these imports.

## 8. Troubleshooting

### `FileNotFoundError`

The embedded default data paths are not portable. Pass explicit input, output, model, and mask paths to every script, and confirm that each upstream output exists before starting the next step.

### `ModuleNotFoundError: ..._chunk_tools` or `eval_temporal_holdout`

The code-only package lacks the internal modules listed in Section 9. Restore them from the complete project; do not install unrelated packages with similar names from PyPI.

### Predictor and target grids do not match

The training scripts require identical latitude and longitude arrays and grid shapes. Inspect the coordinates with xarray and sort, crop, or regrid the data during preprocessing when necessary.

### CUDA out-of-memory errors

Reduce `--base-channels`, reduce `--tile-size` during prediction, or use `--device cpu`. For memory errors during daily processing, reduce `--time-chunk`.

### Output contains only warm-season months

The default configuration restricts monthly and daily intermediates to May–September. To change the final output months, edit `TRAIN_MONTHS` in `common/split_config.py`. Passing `--train-months` only to selected training or prediction scripts is insufficient because shared filters are used throughout the workflow. Regenerate all downstream products after changing the configuration.

### Minimal checks

```bash
python -m compileall .
python tmax/step01_unet005/00_make_clim_anom_025.py --help
python common/merge_clim_monthly.py --help
```

A successful syntax check only confirms that Python can parse the files. It does not confirm that the required data, internal modules, or runtime environment are complete.

