### Low-pass filters at cutoff 40

| Filter | PSNR (dB) | SSIM | Ringing score | Time (ms) |
|---|---:|---:|---:|---:|
| Ideal | 23.573 | 0.6075 | 0.2733 | 22.851 |
| Butterworth (n=2) | 24.378 | 0.6826 | 0.3057 | 22.617 |
| Butterworth (n=8) | 23.78 | 0.6343 | 0.2833 | 22.721 |
| Gaussian | 24.995 | 0.7229 | 0.3276 | 23.384 |

### Ringing on a step edge at cutoff 20

| Filter | Oscillations | Overshoot |
|---|---:|---:|
| Ideal | 4 | 0.09078 |
| Butterworth (n=2) | 0 | 0.03355 |
| Butterworth (n=8) | 4 | 0.08406 |
| Gaussian | 0 | 0 |

### Periodic interference removed

| Method | PSNR (dB) |
|---|---:|
| Noisy input | 15.517 |
| Median filter (spatial) | 17.248 |
| Notch, blind peaks | 30.135 |
| Notch, true peaks (oracle) | 30.135 |

### Uneven illumination corrected

| Method | PSNR (dB) | PSNR matched (dB) |
|---|---:|---:|
| Uneven input | 14.613 | 14.613 |
| CLAHE (spatial) | 17.707 | 17.707 |
| Homomorphic | 9.46 | 15.839 |

### Four scenes down the rows

| Sr | Scene | periodic noise added | Median filter (spatial) | Notch, blind peaks | Notch, true peaks (oracle) |
|---|---:|---:|---:|---:|---:|
| 1 | child red jumper · detail 197 | 15.5 dB | 18.7 dB | 31.5 dB | 31.5 dB |
| 2 | fjord harbour · detail 256 | 15.2 dB | 16.6 dB | 34.1 dB | 34.1 dB |
| 3 | lizard on gravel · detail 360 | 15.3 dB | 16.8 dB | 30.7 dB | 30.7 dB |
| 4 | tower and spire · detail 457 | 15.2 dB | 17.3 dB | 34.9 dB | 34.9 dB |
