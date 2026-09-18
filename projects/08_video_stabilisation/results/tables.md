### How well each estimator recovers the known motion

| Estimator | Translation error (px) | Rotation error (deg) | ms per frame |
|---|---:|---:|---:|
| No motion (control) | 1.9856 | 0.1214 | 0 |
| Features + LK + RANSAC | 0.0353 | 0.0032 | 10.49 |
| Phase correlation | 0.1177 | 0.1214 | 11.1 |
| ECC (direct) | 0.1586 | 0.013 | 67.65 |
| Block matching | 0.0924 | 0.0163 | 5.52 |

### The same estimators on footage that never moved

| Estimator | Invented motion (px/frame) | Worst drift (px) |
|---|---:|---:|
| No motion (control) | 0 | 0 |
| Features + LK + RANSAC | 0.0232 | 2.263 |
| Phase correlation | 0.0787 | 6.779 |
| ECC (direct) | 0.144 | 25.291 |
| Block matching | 0.0251 | 2.581 |

### Every estimator against every smoother

| Estimator | Smoother | Residual jitter | Input jitter | Reduction | Crop |
|---|---:|---:|---:|---:|---:|
| No motion (control) | Fix the camera (control) | 1.766 | 1.766 | 1 | 0 |
| No motion (control) | Moving average (r=15) | 1.766 | 1.766 | 1 | 0 |
| No motion (control) | Gaussian (sigma=8) | 1.766 | 1.766 | 1 | 0 |
| No motion (control) | Kalman (causal) | 1.766 | 1.766 | 1 | 0 |
| Features + LK + RANSAC | Fix the camera (control) | 0.0323 | 1.766 | 54.71 | 0.0728 |
| Features + LK + RANSAC | Moving average (r=15) | 0.1013 | 1.766 | 17.43 | 0.0483 |
| Features + LK + RANSAC | Gaussian (sigma=8) | 0.0683 | 1.766 | 25.87 | 0.0392 |
| Features + LK + RANSAC | Kalman (causal) | 0.707 | 1.766 | 2.5 | 0.0233 |
| Phase correlation | Fix the camera (control) | 0.132 | 1.766 | 13.38 | 0.0572 |
| Phase correlation | Moving average (r=15) | 0.1667 | 1.766 | 10.59 | 0.0367 |
| Phase correlation | Gaussian (sigma=8) | 0.1446 | 1.766 | 12.21 | 0.0298 |
| Phase correlation | Kalman (causal) | 0.7138 | 1.766 | 2.47 | 0.0192 |
| ECC (direct) | Fix the camera (control) | 0.0434 | 1.766 | 40.68 | 0.0808 |
| ECC (direct) | Moving average (r=15) | 0.1074 | 1.766 | 16.45 | 0.0492 |
| ECC (direct) | Gaussian (sigma=8) | 0.0744 | 1.766 | 23.73 | 0.0392 |
| ECC (direct) | Kalman (causal) | 0.7086 | 1.766 | 2.49 | 0.0234 |
| Block matching | Fix the camera (control) | 0.1223 | 1.766 | 14.43 | 0.0728 |
| Block matching | Moving average (r=15) | 0.1589 | 1.766 | 11.11 | 0.0482 |
| Block matching | Gaussian (sigma=8) | 0.137 | 1.766 | 12.89 | 0.039 |
| Block matching | Kalman (causal) | 0.7127 | 1.766 | 2.48 | 0.0232 |

### Stability bought with field of view

| Sigma | Residual jitter | Crop |
|---|---:|---:|
| 2 | 0.3739 | 0.0119 |
| 4 | 0.165 | 0.0214 |
| 8 | 0.0646 | 0.0356 |
| 16 | 0.0332 | 0.0486 |
| 32 | 0.0295 | 0.0585 |

### How much shake each estimator survives

| Translation step (px) | No motion (control) | Features + LK + RANSAC | Phase correlation | ECC (direct) | Block matching |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.4804 | 0.028 | 0.1214 | 0.1882 | 0.1451 |
| 1 | 0.9607 | 0.0272 | 0.1188 | 0.1889 | 0.1142 |
| 2 | 1.9214 | 0.0343 | 0.1166 | 0.1932 | 0.0969 |
| 4 | 3.8428 | 0.0806 | 0.1961 | 0.2232 | 0.1152 |
| 8 | 7.6856 | 0.288 | 0.57 | 0.3899 | 0.4329 |
| 16 | 15.3713 | 1.138 | 1.9368 | 2.1577 | 4.1234 |

### Four frames down the rows

| Sr | Frame | True offset | Phase correlation + Gaussian (sigma=8) | Features + LK + RANSAC + Gaussian (sigma=8) | Features + LK + RANSAC + Kalman (causal) |
|---|---:|---:|---:|---:|---:|
| 1 | 12 | (-5.1, -3.8) | 0.27 | 0.04 | 5.98 |
| 2 | 24 | (+0.2, +4.3) | 0.41 | 0.04 | 7.00 |
| 3 | 36 | (-16.5, +4.3) | 0.74 | 0.11 | 3.24 |
| 4 | 48 | (-19.7, +14.6) | 0.07 | 0.07 | 3.43 |
