### Page-boundary detection (12 scenes)

| Method | Found a quad | Usable (≤10 px) | Mean corner err (px) | Median (px) | p90 (px) | Area IoU | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canny + contour | 100% | 100% | 2.647 | 2.667 | 2.773 | 0.9827 | 2.294 |
| Otsu + contour | 100% | 100% | 1.068 | 1.041 | 1.224 | 0.9935 | 1.535 |
| Morph gradient | 100% | 100% | 1.269 | 1.283 | 1.373 | 0.9917 | 2.203 |
| Saturation (HSV) | 100% | 100% | 0.852 | 0.851 | 1.067 | 0.9939 | 2.369 |
| Hough lines | 100% | 58% | 35.177 | 1.575 | 96.356 | 0.8602 | 10.223 |
| minAreaRect (baseline) | 100% | 0% | 20.55 | 21.11 | 25.231 | 0.8909 | 1.397 |

### Binarisation (12 scenes, illumination ratio 0.62)

| Method | Text IoU (mean) | Text IoU (median) | Time (ms) |
|---|---:|---:|---:|
| Otsu (global) | 0.9052 | 0.9056 | 0.143 |
| Adaptive mean | 0.8017 | 0.8006 | 0.278 |
| Adaptive Gaussian | 0.8732 | 0.873 | 0.915 |
| Sauvola | 0.849 | 0.8485 | 11.649 |

### Binarisation vs illumination (4 scenes per level, text IoU)

| Page illum. ratio | Otsu (global) | Adaptive mean | Adaptive Gaussian | Sauvola | Best global (oracle) |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9012 | 0.7914 | 0.8638 | 0.8377 | 0.9313 |
| 0.8526 | 0.9048 | 0.7985 | 0.8707 | 0.8457 | 0.9306 |
| 0.7387 | 0.9042 | 0.8049 | 0.8766 | 0.8511 | 0.9282 |
| 0.6432 | 0.9015 | 0.8112 | 0.8824 | 0.8553 | 0.9243 |
| 0.5524 | 0.8959 | 0.8182 | 0.8889 | 0.8585 | 0.9168 |
| 0.4767 | 0.8597 | 0.8255 | 0.8944 | 0.8612 | 0.9084 |
| 0.4286 | 0.7876 | 0.831 | 0.8981 | 0.863 | 0.9004 |
| 0.4038 | 0.7391 | 0.834 | 0.9001 | 0.8636 | 0.8941 |
| 0.3911 | 0.701 | 0.8355 | 0.9007 | 0.8641 | 0.8898 |

### Aspect-ratio recovery (12 scenes, % error)

| Method | True w/h | Recovered w/h | Mean error | Worst error |
|---|---:|---:|---:|---:|
| Edge lengths (tutorial method) | 0.7143 | 0.7383 | 6.25 | 14.72 |
| Perspective (closed form) | 0.7143 | 0.7138 | 0.07 | 0.07 |
