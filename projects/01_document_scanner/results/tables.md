### Page-boundary detection (30 scenes)

| Method | Found a quad | Usable (≤10 px) | Mean corner err (px) | Median (px) | p90 (px) | Area IoU | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canny + contour | 100% | 100% | 2.703 | 2.701 | 2.844 | 0.982 | 2.019 |
| Otsu + contour | 100% | 100% | 1.112 | 1.055 | 1.361 | 0.9931 | 1.476 |
| Morph gradient | 100% | 100% | 1.302 | 1.268 | 1.499 | 0.9915 | 1.832 |
| Saturation (HSV) | 100% | 100% | 1.02 | 0.993 | 1.422 | 0.9932 | 2.422 |
| Hough lines | 100% | 63% | 46.644 | 1.622 | 159.837 | 0.8142 | 7.08 |
| minAreaRect (baseline) | 100% | 17% | 19.486 | 21.79 | 27.029 | 0.8996 | 1.387 |

### Binarisation (30 scenes, illumination ratio 0.62)

| Method | Text IoU (mean) | Text IoU (median) | Time (ms) |
|---|---:|---:|---:|
| Otsu (global) | 0.9977 | 0.9989 | 0.134 |
| Adaptive mean | 0.8111 | 0.8126 | 0.282 |
| Adaptive Gaussian | 0.9408 | 0.9428 | 0.909 |
| Sauvola | 0.9067 | 0.9059 | 12.255 |

### Binarisation vs illumination (10 scenes per level, text IoU)

| Page illum. ratio | Otsu (global) | Adaptive mean | Adaptive Gaussian | Sauvola | Best global (oracle) |
|---|---:|---:|---:|---:|---:|
| 1 | 0.999 | 0.793 | 0.9323 | 0.8701 | 0.9995 |
| 0.853 | 0.9972 | 0.8056 | 0.9366 | 0.8989 | 0.9989 |
| 0.7395 | 0.9949 | 0.8181 | 0.9406 | 0.9153 | 0.9978 |
| 0.6444 | 0.9893 | 0.8317 | 0.9443 | 0.9253 | 0.995 |
| 0.5539 | 0.9793 | 0.8485 | 0.9482 | 0.9325 | 0.989 |
| 0.4785 | 0.9022 | 0.865 | 0.9518 | 0.9368 | 0.9817 |
| 0.4305 | 0.7772 | 0.8759 | 0.9543 | 0.9391 | 0.9742 |
| 0.4057 | 0.4588 | 0.8817 | 0.956 | 0.9401 | 0.9679 |
| 0.3931 | 0.4296 | 0.8847 | 0.9569 | 0.9404 | 0.9638 |

### Aspect-ratio recovery (30 scenes, % error)

| Method | True w/h | Recovered w/h | Mean error | Worst error |
|---|---:|---:|---:|---:|
| Edge lengths (tutorial method) | 0.7143 | 0.738 | 8.01 | 22.68 |
| Perspective (closed form) | 0.7143 | 0.7138 | 0.07 | 0.07 |
