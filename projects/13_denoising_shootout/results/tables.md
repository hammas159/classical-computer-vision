### Gaussian sigma=25 at DEFAULT parameters

| Filter | PSNR (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|
| Box | 26.538 | 0.6253 | 0.502 |
| Gaussian | 27.181 | 0.6745 | 0.229 |
| Median | 26.294 | 0.5943 | 0.951 |
| Bilateral | 26.055 | 0.591 | 2.54 |
| Non-local means | 25.518 | 0.6291 | 495.942 |
| Wiener (adaptive) | 27.417 | 0.6635 | 9.49 |
| Do nothing (control) | 20.32 | 0.3583 | 0.016 |

### EVERY FILTER x EVERY NOISE, each at its own tuned parameter (PSNR dB)

| Noise | Box | Gaussian | Median | Bilateral | Non-local means | Wiener (adaptive) | Do nothing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian sigma=25 | 26.538 | 27.399 | 26.294 | 28.533 | 26.643 | 27.417 | 20.32 |
| Salt & pepper 6% | 25.502 | 25.944 | 30.673 | 24.324 | 24.07 | 24.583 | 17.527 |
| Poisson lambda=30 | 26.237 | 26.923 | 25.642 | 27.492 | 26.585 | 26.23 | 19.142 |

### What a default parameter costs, on Gaussian sigma=25

| Filter | Default (dB) | Tuned (dB) | Gain (dB) |
|---|---:|---:|---:|
| Box | 26.538 | 26.538 | 0 |
| Gaussian | 27.181 | 27.399 | 0.218 |
| Median | 26.294 | 26.294 | 0 |
| Bilateral | 26.055 | 28.533 | 2.479 |
| Non-local means | 25.518 | 26.643 | 1.126 |
| Wiener (adaptive) | 27.417 | 27.417 | 0 |

### HELD-OUT transfer: tuned on 3 images, scored on 3 others

| Filter | Fitted on 3 images | Train (dB) | Held-out (dB) | Held-out at default | Transfer gain (dB) |
|---|---:|---:|---:|---:|---:|
| Box | ksize=5 | 27.661 | 25.425 | 25.425 | 0 |
| Gaussian | sigma=1.2 | 28.448 | 26.342 | 26.058 | 0.284 |
| Median | ksize=5 | 27.369 | 25.216 | 25.216 | 0 |
| Bilateral | sigma_color=120.0 | 29.337 | 27.723 | 25.806 | 1.916 |
| Non-local means | h=18.0 | 27.372 | 25.922 | 25.955 | -0.032 |
| Wiener (adaptive) | ksize=5 | 28.321 | 26.518 | 26.518 | 0 |

### PSNR vs Gaussian noise level

| Sigma | Noisy input | Box | Gaussian | Median | Bilateral | Non-local means | Wiener (adaptive) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 34.1688 | 27.63 | 28.335 | 28.27 | 33.555 | 29.401 | 31.244 |
| 10 | 28.1735 | 27.469 | 28.165 | 27.897 | 32.877 | 29.446 | 30.288 |
| 20 | 22.2105 | 26.904 | 27.567 | 26.866 | 28.879 | 27.953 | 28.319 |
| 35 | 17.5353 | 25.726 | 26.324 | 25.144 | 21.216 | 19.347 | 25.821 |
| 50 | 14.7425 | 24.443 | 24.971 | 23.518 | 16.606 | 15.045 | 23.87 |
