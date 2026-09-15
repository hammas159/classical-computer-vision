### Methods at beta 1.4 (6 images)

| Method | PSNR (dB) | SSIM | RMS contrast | Transmission MAE | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Dark channel prior | 18.936 | 0.8356 | 0.1471 | 0.1113 | 30.944 |
| DCP + guided refine | 19.496 | 0.8724 | 0.1559 | 0.1059 | 39.803 |
| CLAHE (contrast only) | 12.926 | 0.724 | 0.1811 | n/a | 0.876 |
| Multi-scale Retinex | 8.94 | 0.6383 | 0.154 | n/a | 426.15 |
| Gamma curve (control) | 15.703 | 0.7988 | 0.169 | n/a | 10.601 |
| True transmission (oracle) | 50.799 | 0.995 | 0.1928 | n/a | 10.179 |

### PSNR vs haze density, with the oracle ceiling

| Beta | min t | Dark channel prior | DCP + guided refine | CLAHE (contrast only) | Multi-scale Retinex | Gamma curve (control) | True transmission (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.6703 | 19.964 | 20.731 | 16.91 | 10.799 | 21.934 | 55.74 |
| 0.8 | 0.4493 | 20.518 | 21.319 | 15.096 | 10.006 | 20.35 | 53.25 |
| 1.2 | 0.3012 | 19.639 | 20.286 | 13.566 | 9.263 | 17.009 | 51.851 |
| 1.6 | 0.2019 | 18.101 | 18.589 | 12.295 | 8.678 | 14.637 | 49.736 |
| 2.2 | 0.1108 | 15.915 | 16.225 | 10.799 | 8.019 | 12.429 | 46.146 |
| 3 | 0.0498 | 13.569 | 13.725 | 9.463 | 7.422 | 10.771 | 40.749 |

### A better airlight makes the OUTPUT worse

| Airlight estimator | Airlight error | Transmission MAE | Saturated | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|---:|
| Brightest candidate (default) | 0.0631 | 0.1059 | 1 | 19.496 | 0.8724 |
| Median of candidates | 0.0508 | 0.0967 | 0 | 18.495 | 0.8619 |
| 90th percentile per channel | 0.0452 | 0.1002 | 0 | 18.73 | 0.865 |
| Brightest, excluding saturated | 0.0607 | 0.1054 | 0 | 19.464 | 0.8721 |
