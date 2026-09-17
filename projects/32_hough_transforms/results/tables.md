### Lines on the generated scene, with clutter

| Method | Recall | Precision | Angle error (deg) | Rho error (px) | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Standard Hough | 1 | 0.335 | 0.225 | 0.753 | 1.758 |
| Probabilistic Hough | 1 | 0.2137 | 0.1233 | 0.667 | 4.031 |

### Circles on the generated scene

| Method | recall | precision | centre_error_px | radius_error_px | median_ms |
|---|---:|---:|---:|---:|---:|
| Hough circles (gradient) | 1 | 0.754 | 1.837 | 0.661 | 5.329 |

### Lines on photographs, against human boundaries

| Method | Precision | % of frame drawn |
|---|---:|---:|
| Canny edges (control) | 0.1937 | 17.781 |
| Standard Hough | 0.112 | 55.288 |
| Probabilistic Hough | 0.1167 | 27.528 |

### Four scenes down the rows

| Sr | Scene | Canny edges (control) | Standard Hough | Probabilistic Hough |
|---|---:|---:|---:|---:|
| 1 | camel at sunset · edges 2% | 0.240 | 0.028 | 0.026 |
| 2 | held sunfish · edges 12% | 0.251 | 0.112 | 0.111 |
| 3 | two women street · edges 19% | 0.364 | 0.231 | 0.234 |
| 4 | ocelot on rock · edges 36% | 0.078 | 0.088 | 0.093 |
