### Counting and measuring 24 coins

| Method | Count | Error | Smallest (mm) | Largest (mm) | Diameter CV | Implausible | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Otsu + components | 18 | -6 | 6.76 | 24.25 | 0.4305 | 13 | 1.239 |
| Adaptive + components | 24 | 0 | 4.77 | 24.25 | 0.2501 | 2 | 1.82 |
| Watershed (global seed) | 24 | 0 | 7.2 | 24.25 | 0.2018 | 1 | 96.67 |
| Watershed (local maxima) | 24 | 0 | 6.38 | 24.25 | 0.208 | 1 | 99.155 |
| Hough circles | 24 | 0 | 14.06 | 24.25 | 0.1758 | 0 | 5.14 |

### ABLATION: mask flattening x seeding rule (coins counted)

| Seeding rule | Plain Otsu | Top-hat then Otsu |
|---|---:|---:|
| Global fraction of dist.max() (the tutorial) | 1 | 24 |
| Local maxima of the distance transform | 24 | 24 |

### The tutorial's seed ratio, swept

| fg_ratio | Global seed count | Error | Local maxima count |
|---|---:|---:|---:|
| 0.3 | 24 | 0 | 24 |
| 0.4 | 24 | 0 | 24 |
| 0.5 | 24 | 0 | 24 |
| 0.55 | 24 | 0 | 24 |
| 0.6 | 19 | -5 | 24 |
| 0.7 | 8 | -16 | 24 |
| 0.8 | 4 | -20 | 24 |

### Calibration error propagates 1:1

| Reference error (%) | Assumed reference (mm) | Mean measured (mm) | Measured error (%) |
|---|---:|---:|---:|
| -10 | 21.82 | 14.984 | -10 |
| -5 | 23.04 | 15.817 | -5 |
| -2 | 23.77 | 16.316 | -2 |
| 0 | 24.25 | 16.649 | 0 |
| 2 | 24.73 | 16.982 | 2 |
| 5 | 25.46 | 17.482 | 5 |
| 10 | 26.68 | 18.314 | 10 |

### Every coin, measured (watershed, local maxima)

| Rank | Area (px) | Diameter (px) | Diameter (mm) |
|---|---:|---:|---:|
| 1 | 3015 | 61.96 | 24.25 |
| 2 | 2464 | 56.01 | 21.92 |
| 3 | 2343 | 54.62 | 21.38 |
| 4 | 2200 | 52.93 | 20.71 |
| 5 | 1910 | 49.31 | 19.3 |
| 6 | 1798 | 47.85 | 18.73 |
| 7 | 1687 | 46.35 | 18.14 |
| 8 | 1639 | 45.68 | 17.88 |
| 9 | 1520 | 43.99 | 17.22 |
| 10 | 1499 | 43.69 | 17.1 |
| 11 | 1425 | 42.6 | 16.67 |
| 12 | 1409 | 42.36 | 16.58 |
| 13 | 1379 | 41.9 | 16.4 |
| 14 | 1340 | 41.31 | 16.17 |
| 15 | 1213 | 39.3 | 15.38 |
| 16 | 1136 | 38.03 | 14.89 |
| 17 | 1118 | 37.73 | 14.77 |
| 18 | 1110 | 37.59 | 14.71 |
| 19 | 1076 | 37.01 | 14.49 |
| 20 | 1045 | 36.48 | 14.28 |
| 21 | 1024 | 36.11 | 14.13 |
| 22 | 1013 | 35.91 | 14.06 |
| 23 | 1012 | 35.9 | 14.05 |
| 24 | 209 | 16.31 | 6.38 |
