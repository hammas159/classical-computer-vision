### Page-boundary detection (30 scenes)

| Method | Found a quad | Usable (≤10 px) | Mean corner err (px) | Median (px) | p90 (px) | Area IoU | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canny + contour | 100% | 100% | 2.676 | 2.689 | 2.817 | 0.9821 | 2.353 |
| Otsu + contour | 100% | 100% | 1.068 | 1.041 | 1.305 | 0.9934 | 1.634 |
| Morph gradient | 100% | 100% | 1.326 | 1.328 | 1.465 | 0.9911 | 1.944 |
| Saturation (HSV) | 100% | 100% | 0.898 | 0.87 | 1.19 | 0.994 | 2.652 |
| Hough lines | 100% | 73% | 30.288 | 1.379 | 106.063 | 0.8785 | 11.829 |
| minAreaRect (baseline) | 100% | 10% | 20.15 | 22.158 | 28.136 | 0.8932 | 1.425 |

### Binarisation (30 scenes, illumination ratio 0.62)

| Method | Text IoU (mean) | Text IoU (median) | Time (ms) |
|---|---:|---:|---:|
| Otsu (global) | 0.9045 | 0.9048 | 0.121 |
| Adaptive mean | 0.8009 | 0.7999 | 0.325 |
| Adaptive Gaussian | 0.872 | 0.872 | 1.064 |
| Sauvola | 0.8483 | 0.8479 | 18.341 |

### Binarisation vs illumination (10 scenes per level, text IoU)

| Page illum. ratio | Otsu (global) | Adaptive mean | Adaptive Gaussian | Sauvola | Best global (oracle) |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9018 | 0.7925 | 0.8646 | 0.839 | 0.9317 |
| 0.8496 | 0.9044 | 0.7995 | 0.8712 | 0.8469 | 0.931 |
| 0.7342 | 0.9046 | 0.8057 | 0.8769 | 0.8522 | 0.9284 |
| 0.638 | 0.9022 | 0.8119 | 0.8827 | 0.8561 | 0.9241 |
| 0.5468 | 0.8959 | 0.8189 | 0.8893 | 0.8594 | 0.917 |
| 0.4712 | 0.8595 | 0.8262 | 0.8949 | 0.8621 | 0.9092 |
| 0.4232 | 0.773 | 0.8314 | 0.8984 | 0.8636 | 0.9004 |
| 0.3984 | 0.7127 | 0.8344 | 0.9001 | 0.8644 | 0.8938 |
| 0.3858 | 0.6775 | 0.836 | 0.9009 | 0.8648 | 0.8896 |

### Aspect-ratio recovery (30 scenes, % error)

| Method | True w/h | Recovered w/h | Mean error | Worst error |
|---|---:|---:|---:|---:|
| Edge lengths (tutorial method) | 0.7143 | 0.7329 | 6.62 | 17.5 |
| Perspective (closed form) | 0.7143 | 0.7138 | 0.07 | 0.07 |
