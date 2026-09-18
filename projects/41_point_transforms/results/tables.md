### What each table costs, before any image

| Transform | Levels surviving | Levels lost | Invertible | Longest run | Largest gap |
|---|---:|---:|---:|---:|---:|
| Identity (control) | 256 | 0 | True | 1 | 1 |
| Negative | 256 | 0 | True | 1 | 1 |
| Gamma 0.5 (brighten) | 192 | 64 | False | 2 | 16 |
| Gamma 2.2 (darken) | 184 | 72 | False | 15 | 3 |
| Log | 125 | 131 | False | 6 | 32 |
| Inverse log | 125 | 131 | False | 24 | 6 |
| Piecewise linear | 176 | 80 | False | 3 | 2 |
| Contrast stretch | 191 | 65 | False | 36 | 2 |
| Posterise (6 levels) | 6 | 250 | False | 51 | 51 |
| Threshold 127 | 2 | 254 | False | 128 | 255 |

### A lookup table against the same maths

| Transform | Bit-identical | LUT (ms) | Arithmetic (ms) | Speedup |
|---|---:|---:|---:|---:|
| Identity (control) | True | 0.0475 | 0.0992 | 2.1 |
| Negative | True | 0.0476 | 0.7328 | 15.4 |
| Gamma 0.5 (brighten) | True | 0.0486 | 2.0941 | 43 |
| Gamma 2.2 (darken) | True | 0.0486 | 2.0218 | 41.6 |

### On twelve photographs

| Transform | Levels surviving | Entropy (bits) | RMS contrast | PSNR vs original |
|---|---:|---:|---:|---:|
| Identity (control) | 256 | 7.1734 | 0.1973 | inf |
| Negative | 256 | 7.1734 | 0.1973 | 6.276 |
| Gamma 0.5 (brighten) | 192 | 6.7585 | 0.1581 | 13.931 |
| Gamma 2.2 (darken) | 184 | 6.3792 | 0.1911 | 13.313 |
| Log | 125 | 6.0405 | 0.1012 | 8.108 |
| Inverse log | 125 | 5.0505 | 0.1362 | 9.332 |
| Piecewise linear | 176 | 6.6401 | 0.2621 | 20.571 |
| Contrast stretch | 191 | 6.5194 | 0.2503 | 23.259 |
| Posterise (6 levels) | 6 | 1.8479 | 0.206 | 24.632 |
| Threshold 127 | 2 | 0.7857 | 0.4183 | 10.318 |

### The gamma sweep

| Gamma | Levels | Longest run | Largest gap | Entropy (bits) |
|---|---:|---:|---:|---:|
| 0.3 | 149 | 4 | 48 | 6.3611 |
| 0.5 | 192 | 2 | 16 | 6.7585 |
| 0.7 | 223 | 2 | 5 | 6.9756 |
| 1 | 256 | 1 | 1 | 7.1734 |
| 1.5 | 218 | 5 | 2 | 6.8828 |
| 2.2 | 184 | 15 | 3 | 6.3792 |
| 3 | 158 | 32 | 3 | 5.7842 |

### Bit planes

| Top planes kept | PSNR (dB) | Entropy (bits) |
|---|---:|---:|
| 1 | 11.232 | 0.7857 |
| 2 | 16.889 | 1.5524 |
| 3 | 22.811 | 2.4182 |
| 4 | 29.176 | 3.3092 |
| 5 | 35.755 | 4.243 |
| 6 | 42.615 | 5.2107 |
| 7 | 51.196 | 6.18 |
| 8 | inf | 7.1734 |

### Four scenes down the rows

| Sr | Scene | Gamma 0.5 (brighten) | Gamma 2.2 (darken) | Contrast stretch | Posterise (6 levels) |
|---|---:|---:|---:|---:|---:|
| 1 | young monks crowding · mean 74 | 6.417 | 5.655 | 6.846 | 2.968 |
| 2 | clapboard houses · mean 93 | 7.080 | 6.359 | 7.223 | 3.687 |
| 3 | chipmunk on granite · mean 119 | 7.043 | 7.124 | 7.790 | 2.916 |
| 4 | aircrew on tarmac · mean 189 | 6.431 | 7.601 | 5.925 | 3.851 |
