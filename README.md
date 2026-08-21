# U-Net Downscaling of Daily Maximum and Minimum Temperature over Coastal China


This directory contains a two-stage spatial downscaling workflow for daily maximum temperature (Tmax) and daily minimum temperature (Tmin) over coastal China. The workflow first downscales daily temperature from 0.25° to 0.05°, then from 0.05° to 0.01°. U-Net models predict the monthly climatology, daily anomalies are interpolated to the target grid, and residual and scale-consistency corrections ensure that the high-resolution results remain consistent with the coarser source data after aggregation.


## development

Requires Python 3 and the packages in `requirements.txt`. The cluster conda env `cn` is already set up for this pipeline. From this directory:

```bash
pip install -r requirements.txt
```

Layout:

```
  common/                 # shared helpers (predictors, month filters, merge)
  tmax/
    step02_unet005/       # 0.25 → 0.05 stage
    step03_unet001/       # 0.05 → 0.01 stage
  tmin/
    step02_unet005/
    step03_unet001/
  requirements.txt
```

Training scripts import helpers from `common/` via `sys.path`. Large NetCDF files and model weights are gitignored; keep them outside this repo.

### run

Paths for experiment outputs still default to the cluster `zcn/.../02exp` tree. Override with env vars:

| Variable | Meaning | Default |
|---|---|---|
| `ZCN` | parent project root for data/exp | `/public/home/ggao001/users/xhang/Projects/zcn` |
| `PYTHON_BIN` | Python interpreter | conda `cn` python |
| `CLIM_MONTHS` | months to train in parallel | `5 6 7 8 9` |
| `NPROC` | parallel GPU workers | `#GPUs` |

Scripts locate the repo via `SCRIPT_DIR` / `REPO_ROOT` (no hard-coded code path).

## data access

Daily downscaled fields are written as NetCDF. There are two resolutions per variable.
