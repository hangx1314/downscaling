# UNet warm-season downscaling (tmax / tmin)
Warm-season U-Net pipeline for coastal China temperature downscaling(0.25° → 0.05° → 0.01°). 

## Layout

```
code/
  common/                 # shared helpers (predictors, month filters, merge)
  tmax/
    step02_unet005/       # 0.25 → 0.05 stage
    step03_unet001/       # 0.05 → 0.01 stage
    run_pipeline.sh
    run_resume_*.sh
  tmin/
    step02_unet005/
    step03_unet001/
    run_pipeline.sh
    run_resume_*.sh
  requirements.txt
```
