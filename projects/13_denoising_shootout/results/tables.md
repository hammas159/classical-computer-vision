### Gaussian sigma=25 at DEFAULT parameters

| Filter | PSNR (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|
| Box | 27.733 | 0.7077 | 1.249 |
| Gaussian | 28.27 | 0.7386 | 0.316 |
| Median | 27.685 | 0.6778 | 1.974 |
| Bilateral | 26.604 | 0.5479 | 5.813 |
| Non-local means | 27.517 | 0.7034 | 578.306 |
| Wiener (adaptive) | 28.559 | 0.7149 | 21.471 |
| Do nothing (control) | 20.476 | 0.2959 | 0.039 |

### EVERY FILTER x EVERY NOISE, each at its own tuned parameter (PSNR dB)

| Noise | Box | Gaussian | Median | Bilateral | Non-local means | Wiener (adaptive) | Do nothing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian sigma=25 | 27.733 | 28.48 | 27.685 | 29.792 | 28.618 | 28.559 | 20.476 |
| Salt & pepper 6% | 26.565 | 26.892 | 33.688 | 24.981 | 24.81 | 25.118 | 17.421 |
| Poisson lambda=30 | 27.197 | 27.674 | 26.504 | 28.553 | 28.86 | 27.221 | 18.626 |

### What a default parameter costs, on Gaussian sigma=25

| Filter | Default (dB) | Tuned (dB) | Gain (dB) |
|---|---:|---:|---:|
| Box | 27.733 | 27.733 | 0 |
| Gaussian | 28.27 | 28.48 | 0.21 |
| Median | 27.685 | 27.685 | 0 |
| Bilateral | 26.604 | 29.792 | 3.189 |
| Non-local means | 27.517 | 28.618 | 1.101 |
| Wiener (adaptive) | 28.559 | 28.559 | 0 |

### HELD-OUT transfer: tuned on 3 images, scored on 3 others

| Filter | Fitted on 3 images | Train (dB) | Held-out (dB) | Held-out at default | Transfer gain (dB) |
|---|---:|---:|---:|---:|---:|
| Box | ksize=3 | 26.995 | 28.214 | 29.022 | -0.808 |
| Gaussian | sigma=1.2 | 27.415 | 29.545 | 29.562 | -0.017 |
| Median | ksize=5 | 26.712 | 28.658 | 28.658 | 0 |
| Bilateral | sigma_color=120.0 | 28.753 | 30.832 | 26.699 | 4.133 |
| Non-local means | h=18.0 | 27.07 | 30.166 | 29.965 | 0.201 |
| Wiener (adaptive) | ksize=5 | 27.993 | 29.124 | 29.124 | 0 |

### PSNR vs Gaussian noise level

| Sigma | Noisy input | Box | Gaussian | Median | Bilateral | Non-local means | Wiener (adaptive) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 34.2362 | 29.56 | 30.156 | 30.778 | 36.03 | 32.767 | 34.197 |
| 10 | 28.2718 | 29.234 | 29.821 | 30.083 | 34.851 | 32.405 | 32.603 |
| 20 | 22.3557 | 28.278 | 28.834 | 28.462 | 29.674 | 29.848 | 29.742 |
| 35 | 17.6869 | 26.603 | 27.097 | 26.246 | 21.567 | 20.965 | 26.514 |
| 50 | 14.852 | 24.938 | 25.369 | 24.355 | 16.817 | 15.464 | 24.123 |
