# UNet warm-season downscaling (tmax / tmin)
Warm-season U-Net pipeline for coastal China temperature downscaling(0.25° → 0.05° → 0.01°). 

## Layout

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

## Setup

```bash
pip install -r requirements.txt
```

## Run

Paths for experiment outputs still default to the cluster `zcn/.../02exp` tree.
Override with env vars:

| Variable | Meaning | Default |
|---|---|---|
| `CLIM_MONTHS` | months to train in parallel | `5 6 7 8 9` |
| `NPROC` | parallel GPU workers | `#GPUs` |

## Notes

- Large NetCDF / model weights are gitignored; keep them outside this repo.
- Training scripts import helpers from `common/` via `sys.path`.
