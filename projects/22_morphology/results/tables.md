### The algebraic identities morphology must obey

| Element | Opening idempotent | Opening anti-extensive | Closing extensive | Erode/dilate dual |
|---|---:|---:|---:|---:|
| rect | True | True | True | True |
| ellipse | True | True | True | True |
| cross | True | True | True | True |

### What survives a 9 px erosion, by element and structure kind

| Element | Axis-aligned | Diagonal | Curved | Thin |
|---|---:|---:|---:|---:|
| rect | 0.8897 | 0.0003 | 0.8592 | 0 |
| ellipse | 0.8897 | 0.0003 | 0.8867 | 0 |
| cross | 0.8897 | 0.1084 | 0.8996 | 0 |

### Cleaning binarised photographs

| Size | Operation | IoU |
|---|---:|---:|
| 3 | Do nothing (control) | 0.9338 |
| 3 | Median filter | 0.9167 |
| 3 | Close | 0.899 |
| 3 | Open | 0.892 |
| 3 | Dilate | 0.7513 |
| 3 | Erode | 0.6942 |
| 3 | Gradient | 0.2229 |
| 3 | Top-hat | 0.0733 |
| 3 | Black-hat | 0.0258 |
| 5 | Do nothing (control) | 0.9338 |
| 5 | Median filter | 0.8824 |
| 5 | Close | 0.8137 |
| 5 | Open | 0.7687 |
| 5 | Dilate | 0.6067 |
| 5 | Erode | 0.4004 |
| 5 | Gradient | 0.3562 |
| 5 | Top-hat | 0.1931 |
| 5 | Black-hat | 0.0245 |
| 7 | Do nothing (control) | 0.9338 |
| 7 | Median filter | 0.8632 |
| 7 | Close | 0.735 |
| 7 | Open | 0.6444 |
| 7 | Dilate | 0.5445 |
| 7 | Gradient | 0.4191 |
| 7 | Top-hat | 0.313 |
| 7 | Erode | 0.2208 |
| 7 | Black-hat | 0.0223 |

### Skeletonisation

| Method | Components | Thinness | Time (ms) |
|---|---:|---:|---:|
| Morphological (open-subtract) | 61 | 0.0429 | 14.64 |
| Zhang-Suen thinning | 9 | 0.0332 | 6986.18 |

### Four scenes down the rows

| Sr | Scene | Do nothing (control) | Median filter | Erode | Dilate | Open | Close | Gradient | Top-hat | Black-hat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bomber overcast · edges 2% | 0.968 | 0.997 | 0.849 | 0.981 | 0.967 | 0.995 | 0.148 | 0.003 | 0.030 |
| 2 | climber on dome · edges 14% | 0.943 | 0.977 | 0.824 | 0.840 | 0.945 | 0.958 | 0.148 | 0.024 | 0.028 |
| 3 | monk at table · edges 19% | 0.944 | 0.938 | 0.712 | 0.785 | 0.916 | 0.916 | 0.226 | 0.051 | 0.025 |
| 4 | diver sea fans · edges 30% | 0.938 | 0.886 | 0.637 | 0.740 | 0.877 | 0.873 | 0.268 | 0.087 | 0.025 |
