### Six damages, all at 28 dB

| Degradation | Strength | PSNR | MSE | SSIM | MS-SSIM | GMSD | VIF (approx) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian noise | 10.515 | 27.998 | 0.00159 | 0.69227 | 0.88013 | 0.23294 | 0.92836 |
| Blur | 1.05 | 28.005 | 0.00158 | 0.86232 | 0.94758 | 0.16807 | 0.56531 |
| JPEG | 78.375 | 27.959 | 0.0016 | 0.82506 | 0.9117 | 0.21716 | 0.70733 |
| Contrast loss | 0.1444 | 28.02 | 0.00158 | 0.9547 | 0.95884 | 0.01095 | 0.83542 |
| Sub-pixel shift | 1 | 24.26 | 0.00426 | 0.76877 | 0.90455 | 0.17973 | 0.54158 |
| Salt & pepper | 0.0047 | 28.011 | 0.00158 | 0.90611 | 0.95307 | 0.12028 | 0.93915 |

### How much each metric disagrees with PSNR

| Metric | Kendall tau vs PSNR |
|---|---:|
| MSE | 1 |
| PSNR | 1 |
| SSIM | 0.7333 |
| MS-SSIM | 0.7333 |
| GMSD | 0.6 |
| VIF (approx) | 0.4667 |

### Four scenes down the rows

| Sr | Scene | Gaussian noise | Blur | JPEG | Contrast loss | Sub-pixel shift | Salt & pepper |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wolf dark wood · brightness 33 | 0.512 | 0.912 | 0.805 | 0.872 | 0.901 | 0.917 |
| 2 | elephant pair · brightness 98 | 0.649 | 0.702 | 0.712 | 0.975 | 0.650 | 0.871 |
| 3 | iceberg cloud · brightness 117 | 0.627 | 0.867 | 0.835 | 0.965 | 0.824 | 0.886 |
| 4 | woman on steps · brightness 212 | 0.696 | 0.881 | 0.767 | 0.995 | 0.785 | 0.931 |
