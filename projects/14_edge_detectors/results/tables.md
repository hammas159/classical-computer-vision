### Clean scene — six of seven score 1.000

| Operator | F1 | Precision | Recall | Pratt FOM | Best threshold | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Roberts | 0.9998 | 0.9995 | 1 | 0.9381 | 0.06 | 2.048 |
| Prewitt | 0.9999 | 0.9998 | 1 | 0.9546 | 0.1867 | 2.339 |
| Sobel | 1 | 1 | 1 | 0.962 | 0.3333 | 2.084 |
| Scharr | 1 | 1 | 1 | 0.962 | 0.24 | 2.288 |
| LoG | 0.9009 | 0.8196 | 1 | 0.7967 | 0.22 | 1.806 |
| Canny (fixed 50/150) | 1 | 1 | 1 | 0.9816 | n/a | 0.239 |
| Canny (auto median) | 1 | 1 | 1 | 0.9816 | n/a | 1.508 |

### At sigma 30, where they come apart

| Operator | F1 | Precision | Recall | Pratt FOM | Best threshold | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Roberts | 0.9183 | 0.9595 | 0.8806 | 0.6837 | 0.4667 | 2.151 |
| Prewitt | 0.9847 | 0.9849 | 0.9845 | 0.947 | 0.4067 | 2.557 |
| Sobel | 0.9837 | 0.9917 | 0.9758 | 0.9564 | 0.4133 | 2.35 |
| Scharr | 0.9804 | 0.9901 | 0.9709 | 0.9558 | 0.42 | 2.129 |
| LoG | 0.8606 | 0.8421 | 0.8811 | 0.7779 | 0.34 | 1.92 |
| Canny (fixed 50/150) | 1 | 1 | 1 | 0.9722 | n/a | 0.296 |
| Canny (auto median) | 0.1344 | 0.072 | 1 | 0.0935 | n/a | 2.35 |

### F1 against noise

| Noise sigma | Roberts | Prewitt | Sobel | Scharr | LoG | Canny (fixed 50/150) | Canny (auto median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9998 | 0.9999 | 1 | 1 | 0.9009 | 1 | 1 |
| 5 | 0.9998 | 0.9999 | 1 | 1 | 0.8966 | 1 | 1 |
| 15 | 0.997 | 0.9999 | 0.9999 | 0.9999 | 0.8835 | 1 | 0.9039 |
| 30 | 0.9215 | 0.9852 | 0.9848 | 0.9813 | 0.8627 | 1 | 0.1359 |
| 50 | 0.4336 | 0.8654 | 0.865 | 0.8343 | 0.7695 | 0.5302 | 0.0675 |

### The same detections, scored at five tolerances

| Tolerance (px) | Roberts | Prewitt | Sobel | Scharr | LoG | Canny (fixed 50/150) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.3457 | 0.7651 | 0.7784 | 0.7783 | 0.2458 | 0.8013 |
| 1 | 0.9986 | 0.9994 | 0.9994 | 0.9994 | 0.6207 | 1 |
| 2 | 0.9998 | 0.9999 | 1 | 1 | 0.9008 | 1 |
| 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| 5 | 1 | 1 | 1 | 1 | 1 | 1 |

### Four scenes down the rows

| Sr | Scene | Roberts | Prewitt | Sobel | Scharr | LoG | Canny (fixed 50/150) | Canny (auto median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | clean · sigma 0 | F1 1.000 | F1 1.000 | F1 1.000 | F1 1.000 | F1 0.901 | F1 1.000 | F1 1.000 |
| 2 | moderate noise, other shapes · sigma 15 | F1 0.996 | F1 1.000 | F1 1.000 | F1 1.000 | F1 0.882 | F1 1.000 | F1 0.906 |
| 3 | heavy noise, other shapes · sigma 30 | F1 0.917 | F1 0.982 | F1 0.984 | F1 0.980 | F1 0.861 | F1 1.000 | F1 0.136 |
| 4 | severe noise · sigma 50 | F1 0.433 | F1 0.875 | F1 0.882 | F1 0.846 | F1 0.775 | F1 0.518 | F1 0.068 |
