### Methods at beta 1.4 (6 images)

| Method | PSNR (dB) | SSIM | RMS contrast | Transmission MAE | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Dark channel prior | 21.438 | 0.9264 | 0.1813 | 0.0725 | 45.778 |
| DCP + guided refine | 22.489 | 0.96 | 0.1857 | 0.0638 | 59.01 |
| CLAHE (contrast only) | 15.003 | 0.8392 | 0.2067 | n/a | 1.551 |
| Multi-scale Retinex | 11.763 | 0.7586 | 0.1759 | n/a | 651.108 |
| Gamma curve (control) | 17.986 | 0.8694 | 0.1937 | n/a | 16.089 |
| True transmission (oracle) | 50.764 | 0.9969 | 0.1972 | n/a | 15.375 |

### PSNR vs haze density, with the oracle ceiling

| Beta | min t | Dark channel prior | DCP + guided refine | CLAHE (contrast only) | Multi-scale Retinex | Gamma curve (control) | True transmission (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.6703 | 18.769 | 19.844 | 17.524 | 13.773 | 21.09 | 55.721 |
| 0.8 | 0.4493 | 20.507 | 21.725 | 16.789 | 13.205 | 22.145 | 53.233 |
| 1.2 | 0.3012 | 21.345 | 22.48 | 15.624 | 12.256 | 19.356 | 51.84 |
| 1.6 | 0.2019 | 21.157 | 22.091 | 14.405 | 11.308 | 16.824 | 49.733 |
| 2.2 | 0.1108 | 18.814 | 19.254 | 12.812 | 10.207 | 14.34 | 46.129 |
| 3 | 0.0498 | 15.589 | 15.696 | 11.253 | 9.214 | 12.426 | 40.697 |

### A better airlight makes the OUTPUT worse

| Airlight estimator | Airlight error | Transmission MAE | Saturated | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|---:|
| Brightest candidate (default) | 0.0445 | 0.0638 | 0 | 22.489 | 0.96 |
| Median of candidates | 0.034 | 0.062 | 0 | 22.429 | 0.9606 |
| 90th percentile per channel | 0.0383 | 0.0626 | 0 | 22.424 | 0.9604 |
| Brightest, excluding saturated | 0.0445 | 0.0638 | 0 | 22.489 | 0.96 |
