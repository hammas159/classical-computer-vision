### Drawn silhouettes against real people

| People | Detections | Mean margin | Max margin | Above 0.5 |
|---|---:|---:|---:|---:|
| Drawn silhouettes | 13 | 0.51 | 1.236 | 5 |
| Real pedestrians | 52 | 1.591 | 4.444 | 40 |

### Real footage, frame by frame

| Frame | HOG detections | Moving regions | HOG on a moving region | Moving regions detected | Mean margin |
|---|---:|---:|---:|---:|---:|
| 50 | 5 | 2 | 2 | 2 | 1.961 |
| 110 | 1 | 1 | 0 | 0 | 1.352 |
| 170 | 3 | 3 | 2 | 2 | 0.955 |
| 230 | 4 | 4 | 3 | 3 | 2.858 |
| 290 | 3 | 1 | 0 | 0 | 1.66 |
| 350 | 4 | 3 | 4 | 3 | 1.819 |
| 410 | 2 | 0 | 0 | 0 | 1.017 |
| 470 | 4 | 4 | 3 | 3 | 2.224 |
| 530 | 3 | 4 | 1 | 1 | 3.619 |
| 600 | 5 | 5 | 3 | 3 | 2.194 |
| 670 | 3 | 4 | 2 | 2 | 1.948 |
| 740 | 5 | 6 | 2 | 2 | 1.227 |

### The threshold trade on real footage

| Hit threshold | Detections | On a moving region | Moving regions covered |
|---|---:|---:|---:|
| -0.5 | 52 | 0.4808 | 0.6486 |
| 0 | 42 | 0.5238 | 0.5676 |
| 0.3 | 36 | 0.5556 | 0.5135 |
| 0.6 | 26 | 0.6154 | 0.4324 |
| 1 | 19 | 0.6316 | 0.3243 |
| 1.5 | 11 | 0.6364 | 0.1892 |

### The pyramid step

| Scale | False positives | Time (ms) |
|---|---:|---:|
| 1.01 | 11 | 58.6 |
| 1.03 | 13 | 34 |
| 1.05 | 7 | 32.6 |
| 1.1 | 3 | 15.9 |
| 1.2 | 0 | 15.2 |
| 1.4 | 0 | 14.6 |

### Four frames down the rows

| Sr | Frame | Moving regions | HOG at -0.5 | HOG at +0.0 | HOG at +0.6 |
|---|---:|---:|---:|---:|---:|
| 1 | 230 | 4 | 5 | 4 | 3 |
| 2 | 470 | 4 | 4 | 4 | 2 |
| 3 | 600 | 5 | 6 | 5 | 4 |
| 4 | 740 | 6 | 5 | 5 | 4 |
