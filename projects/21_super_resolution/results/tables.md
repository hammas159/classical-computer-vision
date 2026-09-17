### Every method at 4x

| Method | PSNR (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|
| Nearest | 22.45 | 0.5398 | 0.199 |
| Bilinear | 22.738 | 0.5505 | 0.23 |
| Bicubic | 22.913 | 0.5676 | 0.848 |
| Lanczos-4 | 22.937 | 0.5711 | 2.051 |
| Edge-directed | 22.913 | 0.5676 | 13.435 |
| Back-projection | 24.56 | 0.6435 | 117.073 |
| Band-limited reference | 24.863 | 0.6568 | 0.744 |

### Scale factor swept

| Scale | Nearest | Bilinear | Bicubic | Lanczos-4 | Edge-directed | Back-projection | Spread | Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 25.679 | 25.771 | 26.263 | 26.337 | 26.308 | 28.29 | 2.611 | -0.166 |
| 3 | 23.594 | 23.898 | 24.114 | 24.145 | 24.117 | 25.872 | 2.278 | 0.16 |
| 4 | 22.45 | 22.738 | 22.913 | 22.937 | 22.913 | 24.56 | 2.11 | 0.303 |
| 6 | 21.203 | 21.481 | 21.588 | 21.603 | 21.588 | 23.007 | 1.804 | 0.43 |
| 8 | 20.387 | 20.657 | 20.733 | 20.746 | 20.733 | 21.979 | 1.592 | 0.565 |

### High-frequency energy recovered

| Method | High-freq energy |
|---|---:|
| Original (truth) | 0.55121 |
| Nearest | 0.34429 |
| Bilinear | 0.1605 |
| Bicubic | 0.14661 |
| Lanczos-4 | 0.14216 |
| Edge-directed | 0.14668 |
| Back-projection | 0.1365 |

### Four scenes down the rows

| Sr | Scene | Nearest | Bilinear | Bicubic | Lanczos-4 | Edge-directed | Back-projection | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | hawk in scrub · high-freq 39.0% | 29.8 dB | 30.4 dB | 30.8 dB | 30.8 dB | 30.8 dB | 34.6 dB | 34.2 dB |
| 2 | rider and herd · high-freq 48.1% | 20.4 dB | 20.7 dB | 20.9 dB | 21.0 dB | 20.9 dB | 23.5 dB | 23.8 dB |
| 3 | longtail boats · high-freq 54.7% | 23.5 dB | 23.8 dB | 23.9 dB | 24.0 dB | 23.9 dB | 25.8 dB | 26.3 dB |
| 4 | parasols willows · high-freq 60.0% | 20.8 dB | 21.0 dB | 21.1 dB | 21.1 dB | 21.1 dB | 22.3 dB | 22.6 dB |
