### On undegraded photographs

| Method | Success rate | Mean error (px) | Time (ms) |
|---|---:|---:|---:|
| SSD (SQDIFF) | 1 | 0 | 5.471 |
| SSD normalised | 1 | 0 | 5.854 |
| Cross-correlation | 0.1111 | 160.013 | 2.001 |
| NCC (CCORR_NORMED) | 1 | 0 | 2.666 |
| ZNCC (CCOEFF_NORMED) | 1 | 0 | 3.907 |

### Brightness scale

| Gain | SSD (SQDIFF) | SSD normalised | Cross-correlation | NCC (CCORR_NORMED) | ZNCC (CCOEFF_NORMED) |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.125 | 1 | 1 |
| 0.5 | 0.4583 | 0.2083 | 0.125 | 1 | 1 |
| 0.3 | 0.2083 | 0 | 0.125 | 1 | 1 |
| 0.1 | 0.125 | 0 | 0.125 | 1 | 1 |
| 2 | 0.3333 | 0.4583 | 0.0833 | 0.9583 | 0.9583 |
| 3 | 0.0417 | 0.125 | 0 | 0.875 | 0.9583 |

### Brightness offset

| Offset | SSD (SQDIFF) | SSD normalised | Cross-correlation | NCC (CCORR_NORMED) | ZNCC (CCOEFF_NORMED) |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0.125 | 1 | 1 |
| -0.1 | 0.8333 | 0.7917 | 0.125 | 1 | 1 |
| -0.2 | 0.6667 | 0.4583 | 0.125 | 0.875 | 1 |
| -0.3 | 0.5 | 0.2083 | 0.125 | 0.6667 | 1 |
| -0.4 | 0.25 | 0.1667 | 0.1667 | 0.4583 | 1 |
| 0.2 | 0.5 | 0.6667 | 0.125 | 1 | 1 |
| 0.4 | 0.0833 | 0.1667 | 0.125 | 0.9583 | 0.9583 |

### Response sharpness

| Method | Peak / mean |
|---|---:|
| SSD (SQDIFF) | 1.6824 |
| SSD normalised | 2.8511 |
| Cross-correlation | 1.7026 |
| NCC (CCORR_NORMED) | 1.2158 |
| ZNCC (CCOEFF_NORMED) | 433.943 |

### Single scale against a pyramid search

| True scale | Single scale | Pyramid | Scale error | Single (ms) | Pyramid (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 | 4.24 | 90.9 |
| 0.9 | 0.6667 | 1 | 0.0021 | 4.07 | 86.83 |
| 0.8 | 0.2083 | 1 | 0 | 4.27 | 92.85 |
| 1.1 | 0.5 | 1 | 0 | 4.2 | 89.45 |
| 1.25 | 0.0833 | 1 | 0 | 4.63 | 99.69 |

### Four scenes down the rows

| Sr | Scene | SSD (SQDIFF) | SSD normalised | Cross-correlation | NCC (CCORR_NORMED) | ZNCC (CCOEFF_NORMED) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | hippo in green water · entropy 5.3 | 225 | 342 | 225 | 99 | 0 |
| 2 | elephants and grooms · entropy 7.4 | 0 | 111 | 111 | 305 | 0 |
| 3 | two at a railing · entropy 7.6 | 239 | 239 | 239 | 0 | 0 |
| 4 | monks at a noticeboard · entropy 7.9 | 154 | 144 | 144 | 0 | 0 |
