### Motion blur, noise sigma 3

| Method | Knows the kernel | PSNR (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|---:|
| Inverse filter | True | 6.183 | 0.0178 | 37.718 |
| Wiener | True | 23.408 | 0.6609 | 38.425 |
| Richardson-Lucy | True | 25.991 | 0.7133 | 191.673 |
| Regularised LS | True | 20.718 | 0.5257 | 52.484 |
| Unsharp (no kernel) | False | 22.67 | 0.5164 | 1.518 |
| Do nothing (control) | False | 23.157 | 0.5665 | 0.058 |

### The same methods on a defocus blur

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Inverse filter | 5.786 | 0.0122 |
| Wiener | 22.488 | 0.5548 |
| Richardson-Lucy | 24.372 | 0.6159 |
| Regularised LS | 19.403 | 0.433 |
| Unsharp (no kernel) | 22.156 | 0.4128 |
| Do nothing (control) | 22.457 | 0.4825 |

### Richardson-Lucy iterations swept

| Iterations | PSNR (dB) | SSIM |
|---|---:|---:|
| 1 | 22.821 | 0.5663 |
| 3 | 23.922 | 0.6284 |
| 5 | 24.367 | 0.6541 |
| 10 | 25 | 0.6863 |
| 20 | 25.665 | 0.7101 |
| 30 | 25.991 | 0.7133 |
| 50 | 26.156 | 0.6978 |
| 80 | 25.877 | 0.6615 |
| 120 | 25.209 | 0.6161 |
| 200 | 23.862 | 0.549 |

### Wiener noise-to-signal ratio swept

| NSR | PSNR (dB) | SSIM |
|---|---:|---:|
| 0.0001 | 12.16 | 0.1492 |
| 0.001 | 17.589 | 0.3381 |
| 0.005 | 21.728 | 0.5214 |
| 0.01 | 23.056 | 0.5959 |
| 0.05 | 23.408 | 0.6609 |
| 0.1 | 21.868 | 0.6422 |
| 0.3 | 17.38 | 0.576 |

### Blind angle estimation

| True angle | Mean abs error |
|---|---:|
| 0 | 1.17 |
| 15 | 5.67 |
| 30 | 10.83 |
| 45 | 11.75 |
| 60 | 9.58 |
| 90 | 1.75 |
| 120 | 10.25 |
| 135 | 11.25 |
| 160 | 8.17 |

### Four scenes down the rows

| Sr | Scene | Blurred | Inverse filter | Wiener | Richardson-Lucy | Regularised LS | Unsharp (no kernel) | Do nothing (control) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | paraglider peak · detail 65 | 32.6 dB | 7.1 dB | 29.8 dB | 33.0 dB | 28.5 dB | 30.6 dB | 32.6 dB |
| 2 | three owlets · detail 227 | 23.3 dB | 6.3 dB | 25.2 dB | 26.6 dB | 26.4 dB | 22.9 dB | 23.3 dB |
| 3 | castle gatehouse · detail 365 | 21.1 dB | 5.1 dB | 20.8 dB | 24.0 dB | 15.4 dB | 20.8 dB | 21.1 dB |
| 4 | man laying paving · detail 409 | 21.4 dB | 5.9 dB | 21.9 dB | 24.9 dB | 18.1 dB | 21.2 dB | 21.4 dB |
