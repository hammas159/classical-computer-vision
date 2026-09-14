### Methods at gamma 3.0 (3 images)

| Method | PSNR (dB) | SSIM | PSNR matched | SSIM matched | Entropy (bits) | Noise sigma | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gamma 1/2.2 | 19.778 | 0.6415 | 21.731 | 0.6146 | 7.45 | 8.295 | 10.065 |
| Histogram equalisation | 17.348 | 0.596 | 18.621 | 0.6282 | 6.558 | 10.657 | 0.571 |
| CLAHE | 15.773 | 0.5717 | 16.845 | 0.5324 | 7.531 | 9.726 | 0.784 |
| Single-scale Retinex | 8.392 | 0.4593 | 14.571 | 0.4785 | 6.915 | 20.809 | 95.61 |
| Multi-scale Retinex | 8.058 | 0.4563 | 14.451 | 0.4773 | 6.843 | 21.593 | 365.102 |
| MSRCR | 11.591 | 0.516 | 15.447 | 0.5614 | 7.32 | 15.142 | 399.659 |
| LIME | 15.317 | 0.4453 | 14.936 | 0.4459 | 7.528 | 11.053 | 18.766 |
| Inverse gamma (oracle) | 22.127 | 0.64 | 22.142 | 0.6405 | 7.482 | 10.663 | 9.883 |

### PSNR vs darkness, with the oracle ceiling

| Gamma | Levels left | Gamma 1/2.2 | Histogram equalisation | CLAHE | Single-scale Retinex | Multi-scale Retinex | MSRCR | LIME | Inverse gamma (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 218 | 19.113 | 19.436 | 18.42 | 7.912 | 7.605 | 10.532 | 16.05 | 33.32 |
| 2 | 192 | 26.18 | 18.767 | 17.936 | 7.399 | 7.13 | 10.923 | 16.633 | 28.941 |
| 2.5 | 173 | 24.278 | 18.063 | 17.003 | 7.943 | 7.641 | 11.302 | 16.15 | 25.135 |
| 3 | 158 | 19.778 | 17.348 | 15.773 | 8.392 | 8.058 | 11.591 | 15.317 | 22.127 |
| 4 | 136 | 15.209 | 16.096 | 13.439 | 8.995 | 8.633 | 11.896 | 13.744 | 17.947 |
| 5 | 120 | 12.914 | 15.064 | 11.759 | 9.248 | 8.893 | 11.928 | 12.569 | 15.284 |

### Noise amplification at gamma 3.0

| Method | Noise after | Amplification |
|---|---:|---:|
| Gamma 1/2.2 | 8.295 | 2.09 |
| Histogram equalisation | 10.657 | 2.77 |
| CLAHE | 9.726 | 2.49 |
| Single-scale Retinex | 20.809 | 5.15 |
| Multi-scale Retinex | 21.593 | 5.35 |
| MSRCR | 15.142 | 3.93 |
| LIME | 11.053 | 2.87 |
| Inverse gamma (oracle) | 10.663 | 2.67 |
