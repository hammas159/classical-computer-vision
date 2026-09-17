### Sharpening a CLEAN image — nothing to restore

| Method | PSNR (dB) | SSIM | Acutance | Overshoot | RMS contrast | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Laplacian (add, +centre) | 20.19 | 0.6437 | 0.58564 | 0.4582 | 0.1994 | 3.641 |
| Laplacian (sub, -centre) | 20.19 | 0.6437 | 0.58564 | 0.4582 | 0.1994 | 3.411 |
| Laplacian (WRONG sign) | 18.896 | 0.1494 | 0.30204 | 0.223 | 0.1628 | 3.453 |
| Unsharp mask | 28.883 | 0.9056 | 0.40415 | 0.1882 | 0.1651 | 4.297 |
| High-boost | 13.688 | 0.7662 | 0.28677 | 0.8972 | 0.0965 | 4.282 |
| Do nothing (control) | inf | 1 | 0.24452 | 0 | 0.1465 | 0.015 |

### Sharpening after a sigma 1.5 blur

| Method | PSNR (dB) | PSNR matched (dB) | SSIM | Acutance |
|---|---:|---:|---:|---:|
| Laplacian (add, +centre) | 29.849 | 29.85 | 0.8122 | 0.15231 |
| Laplacian (sub, -centre) | 29.849 | 29.85 | 0.8122 | 0.15231 |
| Laplacian (WRONG sign) | 26.537 | 26.533 | 0.6311 | 0.08965 |
| Unsharp mask | 29.557 | 29.554 | 0.8031 | 0.14392 |
| High-boost | 13.471 | 30.083 | 0.5773 | 0.08906 |
| Do nothing (control) | 28.395 | 28.391 | 0.7493 | 0.11378 |
| Wiener deconvolution (oracle) | 31.902 | 31.899 | 0.8833 | 0.22081 |

### Four scenes down the rows

| Sr | Scene | Laplacian (add, +centre) | Laplacian (sub, -centre) | Laplacian (WRONG sign) | Unsharp mask | High-boost | Do nothing (control) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | elk in water · clean · detail 11 | 22.5 dB | 22.5 dB | 21.7 dB | 31.1 dB | 25.3 dB (rescaled; raw 13.6 dB) | inf dB |
| 2 | carved stone · clean · detail 45 | 15.2 dB (rescaled; raw 14.7 dB) | 15.2 dB (rescaled; raw 14.7 dB) | 12.3 dB | 23.0 dB | 18.2 dB (rescaled; raw 13.7 dB) | inf dB |
| 3 | lionesses · blurred · sigma 1.5 | 33.2 dB | 33.2 dB | 30.0 dB | 33.0 dB | 33.5 dB (rescaled; raw 13.8 dB) | 31.9 dB |
| 4 | rhino on gravel · blurred · detail 23, sigma 1.5 | 27.3 dB | 27.3 dB | 24.2 dB | 26.9 dB | 27.5 dB (rescaled; raw 10.5 dB) | 25.9 dB |
