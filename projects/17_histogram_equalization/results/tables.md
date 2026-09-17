### Every method, full-reference and no-reference

| Method | PSNR (dB) | SSIM | Entropy (bits) | RMS contrast | Noise sigma | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Do nothing (control) | 15.349 | 0.6356 | 5.615 | 0.0648 | 3.25 | 0.028 |
| Global HE | 13.826 | 0.5049 | 5.598 | 0.2871 | 20.66 | 0.456 |
| AHE (unclipped) | 10.983 | 0.3629 | 7.953 | 0.2671 | 36.884 | 0.619 |
| CLAHE (clip 2.0) | 15.939 | 0.734 | 6.625 | 0.1095 | 8.106 | 0.608 |
| Histogram matching | 13.772 | 0.5054 | 5.551 | 0.3075 | 21.803 | 3.801 |
| Gamma 0.6 | 10.581 | 0.5266 | 5.325 | 0.0516 | 2.639 | 8.06 |
| Match the TRUE histogram (oracle) | 24.527 | 0.8335 | 5.596 | 0.1836 | 8.771 | 2.13 |

### CLAHE clip limit swept

| Clip limit | PSNR (dB) | SSIM | Entropy (bits) | Noise sigma |
|---|---:|---:|---:|---:|
| 0.5 | 15.527 | 0.6852 | 5.918 | 4.313 |
| 1 | 15.782 | 0.7256 | 6.206 | 5.601 |
| 2 | 15.939 | 0.734 | 6.625 | 8.106 |
| 3 | 15.764 | 0.6946 | 6.898 | 10.469 |
| 5 | 14.952 | 0.596 | 7.239 | 14.573 |
| 10 | 13.16 | 0.4463 | 7.639 | 22.726 |
| 20 | 11.642 | 0.3751 | 7.874 | 31.53 |
| 40 | 10.983 | 0.3629 | 7.953 | 36.884 |

### Four scenes down the rows

| Sr | Scene | Degraded | Do nothing (control) | Global HE | AHE (unclipped) | CLAHE (clip 2.0) | Histogram matching | Gamma 0.6 | Match the TRUE histogram (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | moonlit pines · tone 25% | 15.3 dB | 15.3 dB | 9.4 dB | 8.6 dB | 14.3 dB | 9.3 dB | 9.7 dB | 22.2 dB |
| 2 | mare and foal · tone 76% | 14.9 dB | 14.9 dB | 11.1 dB | 9.9 dB | 15.4 dB | 11.4 dB | 9.7 dB | 26.2 dB |
| 3 | covered wagons · tone 84% | 16.4 dB | 16.4 dB | 18.8 dB | 11.7 dB | 16.5 dB | 17.9 dB | 12.2 dB | 23.1 dB |
| 4 | child on water · tone 98% | 13.0 dB | 13.0 dB | 15.1 dB | 10.9 dB | 15.3 dB | 16.9 dB | 9.5 dB | 30.4 dB |
