### Methods at gamma 3.0 (6 images)

| Method | PSNR (dB) | SSIM | PSNR matched | SSIM matched | Entropy (bits) | Noise sigma | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gamma 1/2.2 (fixed) | 19.609 | 0.5789 | 20.961 | 0.5447 | 7.198 | 8.491 | 11.468 |
| Gamma (auto-estimated) | 17.327 | 0.533 | 19.692 | 0.5477 | 7.36 | 15.57 | 40.855 |
| Histogram equalisation | 15.33 | 0.499 | 18.03 | 0.5417 | 6.211 | 15.076 | 0.668 |
| CLAHE | 15.578 | 0.5101 | 17.276 | 0.4554 | 7.099 | 8.939 | 0.997 |
| Single-scale Retinex | 7.967 | 0.4079 | 14.708 | 0.4388 | 6.816 | 24.373 | 114.254 |
| Multi-scale Retinex | 7.713 | 0.4052 | 14.62 | 0.4367 | 6.737 | 25.298 | 428.72 |
| MSRCR | 12.161 | 0.453 | 15.92 | 0.5077 | 7.212 | 16.08 | 455.017 |
| LIME | 14.799 | 0.3953 | 14.302 | 0.4101 | 7.215 | 13.228 | 21.86 |
| Inverse gamma (oracle) | 22.314 | 0.5731 | 22.245 | 0.5718 | 7.263 | 11.353 | 11.403 |

### PSNR vs darkness, with the oracle ceiling

| Gamma | Levels left | Gamma 1/2.2 (fixed) | Gamma (auto-estimated) | Histogram equalisation | CLAHE | Single-scale Retinex | Multi-scale Retinex | MSRCR | LIME | Inverse gamma (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 218 | 19.016 | 22.899 | 16.499 | 19.343 | 8.602 | 8.311 | 10.991 | 15.404 | 33.541 |
| 2 | 192 | 26.46 | 21.158 | 16.161 | 18.315 | 8.407 | 8.085 | 11.552 | 15.98 | 29.363 |
| 2.5 | 173 | 24.225 | 19.2 | 15.761 | 16.881 | 8.579 | 8.21 | 11.942 | 15.577 | 25.476 |
| 3 | 158 | 19.609 | 17.327 | 15.33 | 15.578 | 7.967 | 7.713 | 12.161 | 14.799 | 22.314 |
| 4 | 136 | 15.255 | 14.511 | 14.509 | 13.51 | 8.671 | 8.406 | 12.241 | 13.336 | 17.977 |
| 5 | 120 | 13.2 | 12.765 | 13.778 | 12.101 | 9.011 | 8.741 | 12.115 | 12.303 | 15.314 |

### Noise amplification at gamma 3.0

| Method | Noise after | Amplification |
|---|---:|---:|
| Gamma 1/2.2 (fixed) | 8.491 | 2.72 |
| Gamma (auto-estimated) | 15.57 | 5.28 |
| Histogram equalisation | 15.076 | 5.16 |
| CLAHE | 8.939 | 2.73 |
| Single-scale Retinex | 24.373 | 8.14 |
| Multi-scale Retinex | 25.298 | 8.45 |
| MSRCR | 16.08 | 5.26 |
| LIME | 13.228 | 4.37 |
| Inverse gamma (oracle) | 11.353 | 3.69 |

### Auto-gamma: recovering an exponent it was never told

| True gamma | Estimated (mean) | min | max | Error |
|---|---:|---:|---:|---:|
| 1.5 | 1.766 | 0.78 | 2.697 | 17.7 |
| 2 | 2.36 | 1.04 | 3.732 | 18 |
| 2.5 | 3.01 | 1.301 | 5.057 | 20.4 |
| 3 | 3.764 | 1.563 | 6.857 | 25.5 |
| 4 | 5.705 | 2.095 | 12.574 | 42.6 |
| 5 | 7.946 | 2.643 | 20 | 58.9 |

### Where the mid-grey assumption holds, and where it fails

| Image | True mean brightness | Gap from target | Mean gamma error % | Worst % |
|---|---:|---:|---:|---:|
| astronaut | 0.4494 | -0.0006 | -6.4 | 11.5 |
| coffee | 0.3867 | -0.0633 | 29.1 | 30 |
| chelsea | 0.4522 | 0.0022 | 4.9 | 13.9 |
| rocket | 0.256 | -0.194 | 169.5 | 300 |
| retina | 0.3518 | -0.0982 | 55 | 70.4 |
| immunohistochemistry | 0.6287 | 0.1787 | -47.7 | 48 |
