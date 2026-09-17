### Every method at 4 px

| Method | EPE (px) | EPE / displacement | Time (ms) |
|---|---:|---:|---:|
| Lucas-Kanade (dense) | 3.9321 | 0.983 | 9.66 |
| LK pyramid (3 levels) | 0.1474 | 0.0369 | 21.835 |
| Horn-Schunck | 4.1233 | 1.0308 | 1084.39 |
| Farneback | 0.6266 | 0.1566 | 38.523 |
| DIS | 0.0145 | 0.0036 | 14.207 |
| Predict zero (control) | 4.4721 | 1.118 | 0.205 |

### Endpoint error against displacement

| Displacement (px) | Lucas-Kanade (dense) | LK pyramid (3 levels) | Horn-Schunck | Farneback | DIS | Predict zero (control) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.1123 | 0.1138 | 0.2562 | 0.106 | 0.0689 | 0.559 |
| 1 | 0.2207 | 0.1311 | 0.5514 | 0.1658 | 0.0798 | 1.118 |
| 2 | 1.5235 | 0.121 | 1.7555 | 0.2879 | 0.0457 | 2.2361 |
| 4 | 3.9321 | 0.1474 | 4.1233 | 0.6266 | 0.0145 | 4.4721 |
| 8 | 8.4961 | 0.2928 | 8.6586 | 1.3806 | 0.0265 | 8.9443 |
| 16 | 17.6175 | 2.4619 | 17.6654 | 3.0178 | 0.049 | 17.8885 |
| 32 | 35.5801 | 25.6853 | 35.6181 | 11.9112 | 0.2384 | 35.7771 |

### Horn-Schunck convergence

| Iterations | EPE (px) | Time (ms) |
|---|---:|---:|
| 30 | 1.0639 | 29.7 |
| 100 | 0.9751 | 83.3 |
| 300 | 0.8138 | 238.4 |
| 1000 | 0.5514 | 782.1 |
| 3000 | 0.3105 | 2382.3 |

### Four scenes down the rows

| Sr | Scene | Lucas-Kanade (dense) | LK pyramid (3 levels) | Horn-Schunck | Farneback | DIS |
|---|---:|---:|---:|---:|---:|---:|
| 1 | man striped shirt · texture 61 | 3.117 px | 0.765 px | 3.324 px | 1.667 px | 0.307 px |
| 2 | bighorn rock · texture 69 | 3.320 px | 0.247 px | 3.392 px | 0.226 px | 0.208 px |
| 3 | stone archway · texture 73 | 3.579 px | 0.428 px | 3.720 px | 0.777 px | 0.200 px |
| 4 | carved mask thatch · texture 85 | 3.672 px | 0.250 px | 3.730 px | 0.331 px | 0.227 px |
