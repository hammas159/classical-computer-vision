### Every method

| Method | PSNR (dB) | PSNR on edges | Edge penalty | Colour fringing | SSIM | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 22.877 | 20.127 | 2.75 | 13.895 | 0.7621 | 1.085 |
| Bilinear (own) | 26.63 | 24.396 | 2.234 | 8.897 | 0.8481 | 3.326 |
| Bilinear (OpenCV) | 27.18 | 24.485 | 2.695 | 8.732 | 0.8504 | 0.069 |
| Malvar (cross-channel) | 32.039 | 29.972 | 2.067 | 5.359 | 0.958 | 11.074 |
| VNG (gradient) | 32.632 | 30.042 | 2.59 | 4.961 | 0.9597 | 0.909 |
| Edge-aware | 27.114 | 24.425 | 2.689 | 9.48 | 0.8469 | 0.086 |

### The green density argument

| Method | Red | Green | Blue | Green advantage |
|---|---:|---:|---:|---:|
| Nearest neighbour | 22.38 | 23.816 | 22.576 | 1.338 |
| Bilinear (own) | 25.85 | 29.553 | 25.552 | 3.852 |
| Bilinear (OpenCV) | 26.254 | 29.802 | 26.338 | 3.506 |
| Malvar (cross-channel) | 30.983 | 34.583 | 31.376 | 3.403 |
| VNG (gradient) | 31.842 | 35.605 | 31.545 | 3.912 |
| Edge-aware | 26.254 | 29.507 | 26.338 | 3.211 |

### The four Bayer phases

| pattern | psnr_db |
|---|---:|
| RGGB | 32.039 |
| BGGR | 32.014 |
| GRBG | 32.028 |
| GBRG | 32.017 |

### The ISP, one stage at a time

| configuration | psnr_db |
|---|---:|
| Full ISP | 12.339 |
| No white balance | 12.65 |
| No denoise | 12.285 |
| No gamma | 24.036 |
| Demosaic only | 29.994 |

### Four scenes down the rows

| Sr | Scene | Nearest neighbour | Bilinear (own) | Bilinear (OpenCV) | Malvar (cross-channel) | VNG (gradient) | Edge-aware |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | sea stacks · sat. edges 0.0% | 20.83 | 24.25 | 24.83 | 30.05 | 30.56 | 24.70 |
| 2 | two in headscarves · sat. edges 2.6% | 24.21 | 30.14 | 30.19 | 36.94 | 36.96 | 30.39 |
| 3 | lynx on birch · sat. edges 16.6% | 22.12 | 26.60 | 26.73 | 32.26 | 32.30 | 26.61 |
| 4 | flounder on gravel · sat. edges 71.7% | 20.51 | 24.66 | 24.87 | 30.21 | 30.69 | 24.68 |
