### Methods at gamma 3.0 (6 images)

| Method | PSNR (dB) | SSIM | PSNR matched | SSIM matched | Entropy (bits) | Noise sigma | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gamma 1/2.2 (fixed) | 19.287 | 0.6511 | 21.243 | 0.604 | 7.147 | 8.337 | 17.42 |
| Gamma (auto-estimated) | 20.638 | 0.6337 | 21.963 | 0.6326 | 7.201 | 12.559 | 46.196 |
| Histogram equalisation | 15.484 | 0.5409 | 17.555 | 0.5735 | 6.296 | 15.32 | 1.593 |
| CLAHE | 14.806 | 0.5617 | 17.073 | 0.5172 | 7.345 | 10.561 | 1.483 |
| Single-scale Retinex | 7.85 | 0.4669 | 15.221 | 0.5054 | 6.451 | 20.394 | 173.703 |
| Multi-scale Retinex | 7.656 | 0.4648 | 15.169 | 0.5054 | 6.394 | 21.182 | 679.728 |
| MSRCR | 13.525 | 0.4976 | 17.189 | 0.5617 | 7.236 | 15.164 | 696.458 |
| LIME | 15.327 | 0.4197 | 14.972 | 0.4207 | 7.353 | 14.434 | 32.649 |
| Inverse gamma (oracle) | 23.17 | 0.6504 | 23.018 | 0.6473 | 7.181 | 10.427 | 17.22 |

### PSNR vs darkness, with the oracle ceiling

| Gamma | Levels left | Gamma 1/2.2 (fixed) | Gamma (auto-estimated) | Histogram equalisation | CLAHE | Single-scale Retinex | Multi-scale Retinex | MSRCR | LIME | Inverse gamma (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 218 | 18.62 | 26.18 | 16.382 | 18.461 | 10.024 | 9.75 | 13.063 | 13.931 | 34.451 |
| 2 | 192 | 27.354 | 24.874 | 16.193 | 17.881 | 7.942 | 7.654 | 13.269 | 15.169 | 30.646 |
| 2.5 | 173 | 24.493 | 22.793 | 15.876 | 16.388 | 7.299 | 7.105 | 13.439 | 15.542 | 26.514 |
| 3 | 158 | 19.287 | 20.638 | 15.484 | 14.806 | 7.85 | 7.656 | 13.525 | 15.327 | 23.17 |
| 4 | 136 | 14.671 | 16.94 | 14.629 | 12.358 | 8.662 | 8.461 | 13.427 | 14.029 | 18.456 |
| 5 | 120 | 12.489 | 14.352 | 13.777 | 10.848 | 9.092 | 8.891 | 13.136 | 12.677 | 15.537 |

### Noise amplification at gamma 3.0

| Method | Noise after | Amplification |
|---|---:|---:|
| Gamma 1/2.2 (fixed) | 8.337 | 2.31 |
| Gamma (auto-estimated) | 12.559 | 3.8 |
| Histogram equalisation | 15.32 | 4.48 |
| CLAHE | 10.561 | 2.73 |
| Single-scale Retinex | 20.394 | 6.38 |
| Multi-scale Retinex | 21.182 | 6.64 |
| MSRCR | 15.164 | 4.3 |
| LIME | 14.434 | 3.74 |
| Inverse gamma (oracle) | 10.427 | 3 |

### Auto-gamma: recovering an exponent it was never told

| True gamma | Estimated (mean) | min | max | Error |
|---|---:|---:|---:|---:|
| 1.5 | 1.54 | 0.933 | 2.46 | 2.7 |
| 2 | 2.08 | 1.242 | 3.417 | 4 |
| 2.5 | 2.665 | 1.549 | 4.582 | 6.6 |
| 3 | 3.302 | 1.85 | 5.941 | 10.1 |
| 4 | 4.636 | 2.422 | 8.505 | 15.9 |
| 5 | 6.052 | 2.938 | 10.517 | 21 |

### Where the mid-grey assumption holds, and where it fails

| Image | True mean brightness | Gap from target | Mean gamma error % | Worst % |
|---|---:|---:|---:|---:|
| red_door | 0.3015 | -0.1485 | 90.8 | 110.3 |
| sailboat_race | 0.469 | 0.019 | -0.4 | 9.3 |
| apple_desk | 0.449 | -0.001 | 7.5 | 17.8 |
| baboon | 0.4955 | 0.0455 | -10.7 | 13.9 |
| office_block | 0.5493 | 0.0993 | -39.1 | 41.2 |
| red_barn | 0.422 | -0.028 | 19.6 | 36 |
