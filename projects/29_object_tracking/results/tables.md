### Every tracker over twelve runs

| Tracker | Mean IoU | Centre error (px) | Survival rate | Frames survived |
|---|---:|---:|---:|---:|
| Static box (control) | 0.248 | 92.86 | 0.3289 | 16.4 |
| Template matching | 0.6079 | 27.22 | 0.8206 | 41 |
| Mean shift | 0.2541 | 79.89 | 0.3724 | 18.6 |
| CamShift | 0.1476 | 110.05 | 0.0451 | 2.2 |
| Optical flow (LK) | 0.5516 | 22.47 | 0.7072 | 35.3 |
| LK + Kalman | 0.5155 | 24.62 | 0.4856 | 24.2 |
| MOSSE (from scratch) | 0.5671 | 32.09 | 0.8272 | 41.3 |

### Accuracy and persistence separated

| Tracker | IoU while alive | Survival rate |
|---|---:|---:|
| Static box (control) | 0.5893 | 0.3289 |
| Template matching | 0.7173 | 0.8206 |
| Mean shift | 0.3244 | 0.3724 |
| CamShift | 0.2167 | 0.0451 |
| Optical flow (LK) | 0.639 | 0.7072 |
| LK + Kalman | 0.5978 | 0.4856 |
| MOSSE (from scratch) | 0.6714 | 0.8272 |

### How far each truth chain gets

| Start | Frames tracked | Of | Max area jump | Merge events |
|---|---:|---:|---:|---:|
| 50 | 50 | 50 | 1.41 | 0 |
| 140 | 44 | 50 | 2.22 | 2 |
| 170 | 50 | 50 | 1.26 | 0 |
| 210 | 50 | 50 | 1.27 | 0 |
| 240 | 50 | 50 | 1.26 | 0 |
| 410 | 50 | 50 | 1.27 | 0 |
| 490 | 50 | 50 | 1.41 | 0 |
| 550 | 50 | 50 | 2.9 | 1 |
| 580 | 50 | 50 | 2.9 | 1 |
| 610 | 50 | 50 | 1.25 | 0 |
| 640 | 50 | 50 | 1.44 | 0 |
| 740 | 50 | 50 | 3.06 | 2 |

### What the colour trackers were given

| Start | Inside | Outside | Ratio |
|---|---:|---:|---:|
| 50 | 0.12 | 0.01 | 13.1 |
| 140 | 0.56 | 0.23 | 2.41 |
| 170 | 0.38 | 0.17 | 2.24 |
| 210 | 0.08 | 0.01 | 6 |
| 240 | 0.32 | 0.1 | 3.17 |
| 410 | 0.25 | 0.16 | 1.56 |
| 490 | 0.25 | 0.16 | 1.53 |
| 550 | 0.62 | 0.41 | 1.5 |
| 580 | 0.71 | 0.4 | 1.77 |
| 610 | 0.61 | 0.43 | 1.41 |
| 640 | 0.53 | 0.43 | 1.21 |
| 740 | 0.15 | 0.01 | 15.14 |

### Four runs down the rows

| Sr | Run | Truth frames | Static box (control) | Template matching | Mean shift | CamShift | Optical flow (LK) | LK + Kalman | MOSSE (from scratch) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | frames 50–100 | 50 | 0.072 | 0.648 | 0.496 | 0.376 | 0.353 | 0.326 | 0.566 |
| 2 | frames 170–220 | 50 | 0.065 | 0.683 | 0.580 | 0.105 | 0.538 | 0.464 | 0.725 |
| 3 | frames 210–260 | 50 | 0.124 | 0.747 | 0.313 | 0.245 | 0.504 | 0.488 | 0.724 |
| 4 | frames 240–290 | 50 | 0.070 | 0.789 | 0.564 | 0.098 | 0.668 | 0.601 | 0.765 |
