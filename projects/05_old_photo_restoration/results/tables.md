### Inpainting at 3 px, true mask (6 images)

| Method | PSNR whole (dB) | PSNR on damage (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|---:|
| Telea (fast marching) | 38.11 | 26.555 | 0.9826 | 16.359 |
| Navier-Stokes | 38.405 | 26.85 | 0.9838 | 16.441 |
| Iterative masked mean | 37.328 | 25.773 | 0.9808 | 84.564 |
| Harmonic diffusion | 38.21 | 26.655 | 0.983 | 401.898 |

### PSNR on damaged pixels vs scratch width

| Width (px) | Damage fraction | Telea (fast marching) | Navier-Stokes | Iterative masked mean | Harmonic diffusion |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0182 | 28.097 | 29.178 | 27.202 | 28.389 |
| 3 | 0.0707 | 26.555 | 26.85 | 25.773 | 26.655 |
| 5 | 0.0944 | 25.332 | 25.461 | 24.813 | 25.306 |
| 9 | 0.1375 | 23.825 | 23.756 | 23.647 | 20.515 |
| 15 | 0.1973 | 22.525 | 22.476 | 22.545 | 15.132 |
| 25 | 0.2869 | 20.862 | 20.763 | 20.884 | 10.524 |
| 40 | 0.3964 | 19.038 | 19.131 | 19.201 | 8.231 |

### Damage detection, and restoring with the detected mask

| Detector | Mask IoU | Precision | Recall | Flagged | PSNR on damage (dB) | PSNR whole (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Intensity threshold | 0.4199 | 0.602 | 0.5711 | 0.066 | 9.882 | 20.768 |
| Top-hat + black-hat | 0.3797 | 0.4182 | 0.7722 | 0.1378 | 12.706 | 21.925 |
| Median residual (one scale) | 0.4185 | 0.4664 | 0.776 | 0.122 | 12.865 | 21.772 |
| Median residual (multi-scale) | 0.3486 | 0.3595 | 0.8866 | 0.1867 | 19.368 | 22.756 |

### Median-residual detector vs its window size (3 px damage)

| Median window (px) | Damage (px) | Mask IoU | Recall | Precision |
|---|---:|---:|---:|---:|
| 5 | 3 | 0.0807 | 0.1995 | 0.2252 |
| 7 | 3 | 0.0915 | 0.1727 | 0.2294 |
| 9 | 3 | 0.1695 | 0.2876 | 0.3101 |
| 11 | 3 | 0.3498 | 0.5922 | 0.4491 |
| 15 | 3 | 0.3932 | 0.6996 | 0.4627 |
| 21 | 3 | 0.4185 | 0.776 | 0.4664 |
| 31 | 3 | 0.426 | 0.8327 | 0.4572 |

### Fade correction (4 colour images)

| Method | PSNR (dB) | SSIM | Cast error (deg) | Chroma | RMS contrast | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| None (control) | 18.189 | 0.731 | 7.718 | 18.933 | 0.0983 | 0.025 |
| Gray-world balance | 16.476 | 0.7167 | 14.282 | 5.625 | 0.0945 | 5.396 |
| CLAHE on L only | 18.844 | 0.7258 | 7.75 | 18.944 | 0.1471 | 1.749 |
| Per-channel stretch | 20.916 | 0.6994 | 6.573 | 23.025 | 0.2271 | 10.354 |
| Stretch + CLAHE | 18.072 | 0.5622 | 5.88 | 22.977 | 0.2329 | 11.844 |
| Stretch + saturate | 21.245 | 0.6601 | 7.028 | 26.213 | 0.2278 | 18.214 |

### Both degradations together, and the order they are undone in

| Stage | PSNR whole (dB) | PSNR on damage (dB) | SSIM | Cast error (deg) |
|---|---:|---:|---:|---:|
| Damaged + faded (input) | 14.26 | 5.328 | 0.6179 | 8.207 |
| Inpaint only | 18.139 | 17.733 | 0.7184 | 7.72 |
| Fade correct only | 13.315 | 4.921 | 0.6099 | 6.843 |
| Inpaint then fade correct | 20.817 | 18.336 | 0.6543 | 7.096 |
| Fade correct then inpaint | 16.16 | 15.725 | 0.7111 | 7.818 |
