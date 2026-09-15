### Counting and measuring 24 coins

| Method | Count | Error | Smallest (mm) | Largest (mm) | Diameter CV | Implausible | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Otsu + components | 18 | -6 | 6.76 | 24.25 | 0.4305 | 13 | 1.316 |
| Adaptive + components | 24 | 0 | 4.77 | 24.25 | 0.2501 | 2 | 1.767 |
| Watershed (global seed) | 23 | -1 | 7.74 | 24.25 | 0.1991 | 1 | 10.334 |
| Watershed (local maxima) | 24 | 0 | 5.75 | 24.25 | 0.2425 | 2 | 10.757 |
| Hough circles | 24 | 0 | 14.06 | 24.25 | 0.1758 | 0 | 2.729 |

### ABLATION: mask flattening x seeding rule (coins counted)

| Seeding rule | Plain Otsu | Top-hat then Otsu |
|---|---:|---:|
| Global fraction of dist.max() (the tutorial) | 1 | 23 |
| Local maxima of the distance transform | 24 | 24 |

### The tutorial's seed ratio, swept

| fg_ratio | Global seed count | Error | Local maxima count |
|---|---:|---:|---:|
| 0.3 | 25 | 1 | 24 |
| 0.4 | 24 | 0 | 24 |
| 0.5 | 23 | -1 | 24 |
| 0.55 | 23 | -1 | 24 |
| 0.6 | 22 | -2 | 24 |
| 0.7 | 16 | -8 | 24 |
| 0.8 | 7 | -17 | 24 |

### Calibration error propagates 1:1

| Reference error (%) | Assumed reference (mm) | Mean measured (mm) | Measured error (%) |
|---|---:|---:|---:|
| -10 | 21.82 | 15.742 | -10 |
| -5 | 23.04 | 16.617 | -5 |
| -2 | 23.77 | 17.142 | -2 |
| 0 | 24.25 | 17.492 | 0 |
| 2 | 24.73 | 17.841 | 2 |
| 5 | 25.46 | 18.366 | 5 |
| 10 | 26.68 | 19.241 | 10 |

### Every coin, measured (watershed, local maxima)

| Rank | Area (px) | Diameter (px) | Diameter (mm) |
|---|---:|---:|---:|
| 1 | 2420 | 55.51 | 24.25 |
| 2 | 2327 | 54.43 | 23.78 |
| 3 | 2220 | 53.17 | 23.23 |
| 4 | 1911 | 49.33 | 21.55 |
| 5 | 1806 | 47.95 | 20.95 |
| 6 | 1688 | 46.36 | 20.25 |
| 7 | 1639 | 45.68 | 19.96 |
| 8 | 1504 | 43.76 | 19.12 |
| 9 | 1426 | 42.61 | 18.62 |
| 10 | 1408 | 42.34 | 18.5 |
| 11 | 1379 | 41.9 | 18.31 |
| 12 | 1344 | 41.37 | 18.07 |
| 13 | 1211 | 39.27 | 17.15 |
| 14 | 1170 | 38.6 | 16.86 |
| 15 | 1136 | 38.03 | 16.61 |
| 16 | 1110 | 37.59 | 16.42 |
| 17 | 1076 | 37.01 | 16.17 |
| 18 | 1045 | 36.48 | 15.94 |
| 19 | 1030 | 36.21 | 15.82 |
| 20 | 1013 | 35.91 | 15.69 |
| 21 | 1009 | 35.84 | 15.66 |
| 22 | 739 | 30.67 | 13.4 |
| 23 | 247 | 17.73 | 7.75 |
| 24 | 136 | 13.16 | 5.75 |
