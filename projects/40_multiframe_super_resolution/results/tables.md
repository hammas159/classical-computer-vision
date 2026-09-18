### At x3 from 8 frames

| Method | PSNR (dB) | SSIM | Gain over one frame (dB) | Time (ms) |
|---|---:|---:|---:|---:|
| Single frame (bicubic) | 24.488 | 0.6366 | 0 | 0.39 |
| Naive average | 23.97 | 0.5926 | -0.518 | 16.98 |
| Shift-and-add | 25.47 | 0.6838 | 0.982 | 50.09 |
| Iterative back-projection | 26.743 | 0.7634 | 2.255 | 849.52 |

### Against the number of frames

| Frames | Single frame (bicubic) | Naive average | Shift-and-add | Iterative back-projection |
|---|---:|---:|---:|---:|
| 1 | 24.488 | 24.488 | 24.64 | 25.17 |
| 2 | 24.488 | 23.906 | 24.813 | 25.9 |
| 4 | 24.488 | 23.838 | 25.065 | 26.367 |
| 8 | 24.488 | 23.97 | 25.47 | 26.743 |
| 16 | 24.488 | 23.814 | 25.851 | 27.083 |
| 32 | 24.488 | 23.928 | 26.052 | 27.283 |

### Sub-pixel against whole-pixel offsets

| Offsets | Single frame (bicubic) | Naive average | Shift-and-add | Iterative back-projection |
|---|---:|---:|---:|---:|
| Sub-pixel offsets | 24.488 | 23.97 | 25.47 | 26.743 |
| Integer-pixel offsets | 24.488 | 22.101 | 24.652 | 25.937 |

### Registration accuracy

| Method | Mean error (px) | Median (px) |
|---|---:|---:|
| Phase correlation | 0.4095 | 0.407 |
| ECC | 0.0646 | 0.0637 |

### Tolerance to registration error

| Injected error (px) | Single frame (bicubic) | Naive average | Shift-and-add | Iterative back-projection |
|---|---:|---:|---:|---:|
| 0 | 24.488 | 23.97 | 25.47 | 26.743 |
| 0.1 | 24.488 | 23.97 | 25.489 | 26.741 |
| 0.25 | 24.488 | 23.97 | 25.455 | 26.74 |
| 0.5 | 24.488 | 23.97 | 25.313 | 26.578 |
| 1 | 24.488 | 23.97 | 25.058 | 25.999 |
| 2 | 24.488 | 23.97 | 24.041 | 24.08 |

### Four scenes down the rows

| Sr | Scene | Single frame (bicubic) | Naive average | Shift-and-add | Iterative back-projection |
|---|---:|---:|---:|---:|---:|
| 1 | swallow tailed gulls · detail 675 | 26.49 | 25.39 | 27.69 | 29.42 |
| 2 | three schoolchildren · detail 1493 | 23.51 | 22.31 | 24.69 | 26.35 |
| 3 | skier on a slope · detail 2111 | 22.86 | 21.84 | 23.56 | 24.97 |
| 4 | mayan stone carving · detail 6057 | 18.19 | 17.12 | 18.94 | 20.45 |
