### Every measure and every control

| Method | PSNR (dB) | Agreement with truth | Control? |
|---|---:|---:|---:|
| Variance of Laplacian | 43.645 | 0.9724 | False |
| Tenengrad | 43.347 | 0.9541 | False |
| Modified Laplacian | 43.652 | 0.9763 | False |
| Local variance | 42.325 | 0.9307 | False |
| Wavelet detail | 42.209 | 0.9536 | False |
| Random pick (control) | 26.711 | 0.2002 | True |
| First frame (control) | 24.17 | 0.0248 | True |
| Oracle (truth) | 43.686 | 1 | True |

### The pooling window

| Window | PSNR (dB) | Agreement with truth |
|---|---:|---:|
| 1 | 42.992 | 0.6581 |
| 3 | 46.127 | 0.9223 |
| 5 | 46.612 | 0.959 |
| 9 | 46.738 | 0.972 |
| 15 | 46.743 | 0.9731 |
| 25 | 46.693 | 0.9675 |
| 41 | 46.416 | 0.9482 |

### Where the measures disagree

| Photograph | Disagreement | Flat share | Disagreement in flat | Disagreement in detailed |
|---|---:|---:|---:|---:|
| whitewashed_bell_tower | 0.1566 | 0.0122 | 0.5782 | 0.1514 |
| mossy_boulders_in_a_valley | 0.1394 | 0.0843 | 0.5529 | 0.1014 |
| golfer_by_the_sea | 0.0539 | 0 | 0 | 0.0539 |
| wall_across_the_hills | 0.06 | 0 | 0 | 0.06 |
| partridge_on_gravel | 0.0456 | 0 | 0 | 0.0456 |
| two_jackals | 0.0496 | 0 | 0 | 0.0496 |
| race_cars_on_a_bend | 0.1095 | 0.0021 | 0.2277 | 0.1092 |
| picnic_in_the_snow | 0.1411 | 0.0513 | 0.571 | 0.1178 |
| alpine_cottage_with_flowers | 0.0912 | 0.0109 | 0.6342 | 0.0853 |
| lily_pond_and_pagoda | 0.0499 | 0.0009 | 0.7727 | 0.0493 |
| waterfall_under_a_bridge | 0.0274 | 0 | 0 | 0.0274 |
| mountain_lake_and_scree | 0.1041 | 0.0015 | 0.7457 | 0.1031 |

### How many frames the stack has

| Frames | Oracle (dB) | Variance of Laplacian (dB) | Gap (dB) |
|---|---:|---:|---:|
| 3 | 36.701 | 36.692 | 0.009 |
| 5 | 46.784 | 46.738 | 0.047 |
| 9 | 51.394 | 51.266 | 0.127 |
| 17 | 51.456 | 51.29 | 0.166 |

### How much defocus a measure needs

| Max sigma | Variance of Laplacian | Tenengrad | Modified Laplacian | Local variance | Wavelet detail |
|---|---:|---:|---:|---:|---:|
| 1 | 0.5454 | 0.5173 | 0.4473 | 0.46 | 0.4528 |
| 2 | 0.9206 | 0.909 | 0.8689 | 0.8641 | 0.855 |
| 4 | 0.972 | 0.9555 | 0.9746 | 0.9311 | 0.9568 |
| 8 | 0.9724 | 0.9433 | 0.973 | 0.894 | 0.9447 |

### Four photographs down the rows

| Sr | Photograph | Detail | Flat share | Random (dB) | Oracle (dB) | Variance of Laplacian | Tenengrad | Modified Laplacian | Local variance | Wavelet detail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | whitewashed bell tower | 103 | 1.2% | 31.31 | 48.61 | 48.52 | 47.35 | 48.52 | 44.99 | 42.63 |
| 2 | partridge on gravel | 225 | 0.0% | 29.13 | 48.13 | 48.12 | 48.10 | 48.12 | 47.54 | 48.07 |
| 3 | alpine cottage with flowers | 383 | 1.1% | 24.27 | 41.58 | 41.55 | 41.31 | 41.55 | 40.50 | 39.15 |
| 4 | mountain lake and scree | 617 | 0.1% | 20.01 | 35.99 | 35.97 | 35.93 | 35.98 | 35.75 | 35.93 |
