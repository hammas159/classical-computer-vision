### Energy functions at 20% reduction (6 images)

| Energy | Subject kept | Aspect retained | Energy kept | Time (ms) |
|---|---:|---:|---:|---:|
| Gradient \|dx\|+\|dy\| | 0.9642 | 0.9835 | 0.8978 | 842.7 |
| Sobel magnitude | 0.9594 | 0.9831 | 0.8981 | 738.3 |
| Laplacian | 0.9158 | 0.9453 | 0.907 | 682 |
| Local std (entropy-like) | 0.9671 | 0.9861 | 0.896 | 766.7 |
| Plain rescale (control) | 0.7999 | 0.7999 | 0.7689 | 0.4 |

### PER IMAGE at 20% — the average hides a sign change

| Image | Carved: ROI kept | Rescale: ROI kept | Advantage | Carved: energy | Rescale: energy |
|---|---:|---:|---:|---:|---:|
| giraffe | 0.9721 | 0.7969 | 0.1753 | 0.9082 | 0.7779 |
| rocky_coast | 0.9444 | 0.8047 | 0.1397 | 0.8888 | 0.7559 |
| stone_arch | 0.9245 | 0.8021 | 0.1224 | 0.8961 | 0.7502 |
| squirrel_rock | 0.9779 | 0.8021 | 0.1758 | 0.8797 | 0.7794 |
| harbour_boat | 0.9943 | 0.7969 | 0.1975 | 0.8971 | 0.7635 |
| gallery_visitors | 0.972 | 0.7969 | 0.1751 | 0.9167 | 0.7865 |

### Seam carving vs plain rescale, by reduction

| Reduction | Carved: subject | Rescale: subject | Carved: aspect | Rescale: aspect | Carved: energy | Rescale: energy |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.9973 | 0.947 | 0.9991 | 0.947 | 0.9767 | 0.8993 |
| 0.1 | 0.9902 | 0.8976 | 0.9965 | 0.8976 | 0.9525 | 0.8563 |
| 0.2 | 0.9642 | 0.7999 | 0.9835 | 0.7999 | 0.8978 | 0.7689 |
| 0.3 | 0.9182 | 0.6979 | 0.9544 | 0.6979 | 0.8341 | 0.6806 |
| 0.45 | 0.797 | 0.5486 | 0.855 | 0.5486 | 0.7231 | 0.5477 |
| 0.6 | 0.6204 | 0.3993 | 0.6853 | 0.3993 | 0.5925 | 0.4127 |
| 0.7 | 0.4947 | 0.299 | 0.5651 | 0.299 | 0.4909 | 0.3209 |

### What the advantage costs (last row: difference, and the TIME RATIO)

| Method | Subject kept | Aspect retained | Energy kept | Time (ms) |
|---|---:|---:|---:|---:|
| Seam carving | 0.9642 | 0.9835 | 0.8978 | 729.02 |
| Plain rescale | 0.7999 | 0.7999 | 0.7689 | 0.42 |
| Difference | 0.1643 | 0.1836 | 0.1289 | 1736 |
