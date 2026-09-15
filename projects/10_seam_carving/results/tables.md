### Energy functions at 20% reduction (4 images)

| Energy | Subject kept | Aspect retained | Energy kept | Time (ms) |
|---|---:|---:|---:|---:|
| Gradient \|dx\|+\|dy\| | 0.9158 | 0.9344 | 0.9488 | 1364.2 |
| Sobel magnitude | 0.9148 | 0.9341 | 0.949 | 1308.3 |
| Laplacian | 0.9199 | 0.9375 | 0.9567 | 1159.7 |
| Local std (entropy-like) | 0.9187 | 0.9328 | 0.9503 | 1313.8 |
| Plain rescale (control) | 0.8002 | 0.8002 | 0.7938 | 0.4 |

### PER IMAGE at 20% — the average hides a sign change

| Image | Carved: ROI kept | Rescale: ROI kept | Advantage | Carved: energy | Rescale: energy |
|---|---:|---:|---:|---:|---:|
| coffee | 0.7896 | 0.8 | -0.0104 | 0.9342 | 0.7795 |
| rocket | 0.9245 | 0.8008 | 0.1237 | 0.974 | 0.7469 |
| chelsea | 1 | 0.8 | 0.2 | 0.9315 | 0.8086 |
| astronaut | 0.9492 | 0.8 | 0.1492 | 0.9554 | 0.8401 |

### Seam carving vs plain rescale, by reduction

| Reduction | Carved: subject | Rescale: subject | Carved: aspect | Rescale: aspect | Carved: energy | Rescale: energy |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.9854 | 0.9499 | 0.9917 | 0.9499 | 0.9897 | 0.9115 |
| 0.1 | 0.966 | 0.9 | 0.976 | 0.9 | 0.9778 | 0.8724 |
| 0.2 | 0.9158 | 0.8002 | 0.9344 | 0.8002 | 0.9488 | 0.7938 |
| 0.3 | 0.864 | 0.6988 | 0.8891 | 0.6988 | 0.9078 | 0.7113 |
| 0.45 | 0.7444 | 0.5493 | 0.7916 | 0.5493 | 0.8279 | 0.5859 |
| 0.6 | 0.5995 | 0.4006 | 0.66 | 0.4006 | 0.7165 | 0.4579 |
| 0.7 | 0.4831 | 0.2996 | 0.5509 | 0.2996 | 0.617 | 0.3715 |

### What the advantage costs (last row: difference, and the TIME RATIO)

| Method | Subject kept | Aspect retained | Energy kept | Time (ms) |
|---|---:|---:|---:|---:|
| Seam carving | 0.9158 | 0.9344 | 0.9488 | 1194.39 |
| Plain rescale | 0.8002 | 0.8002 | 0.7938 | 0.42 |
| Difference | 0.1156 | 0.1342 | 0.155 | 2844 |
