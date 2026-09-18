### On twelve photographs

| Conversion | Levels from BT.601 | Contrast retained | Worst colour-edge recall | Time (ms) |
|---|---:|---:|---:|---:|
| Average (R+G+B)/3 | 5.95 | 0.9982 | 0.9996 | 3.811 |
| BT.601 (OpenCV default) | 0 | 0.9972 | 0.9994 | 1.498 |
| BT.709 (HDTV) | 2.182 | 0.9966 | 0.9991 | 1.564 |
| Linear-light BT.709 | 3.092 | 0.9963 | 0.9989 | 12.358 |
| Value (max channel) | 15.86 | 0.9875 | 0.9978 | 6.22 |
| Contrast-preserving | 5.529 | 0.9983 | 0.9998 | 207.293 |

### Every weighting's blind plane (grey levels of separation)

| Scene built against | Average (R+G+B)/3 | BT.601 (OpenCV default) | BT.709 (HDTV) | Linear-light BT.709 | Value (max channel) | Contrast-preserving |
|---|---:|---:|---:|---:|---:|---:|
| Average (R+G+B)/3 | 0.461 | 33.177 | 21.198 | 22.354 | 37.857 | 193.207 |
| BT.601 (OpenCV default) | 20.214 | 0.116 | 18.702 | 9.488 | 37.857 | 158.581 |
| BT.709 (HDTV) | 12.977 | 14.17 | 0.388 | 7.678 | 37.857 | 205.901 |

### Contrast kept at each photograph's worst colour edge

| Photograph | Chroma | 1st pct kept | Worst pixel |
|---|---:|---:|---:|
| penguin_on_pebbles | 5.8 | 0.9776 | 0.9613 |
| teotihuacan_pyramids | 17.1 | 0.97 | 0.9412 |
| helicopter_and_pilot | 14.7 | 0.9855 | 0.9725 |
| leopard_along_branch | 22.7 | 0.967 | 0.9504 |
| milking_the_cow | 18.2 | 0.9778 | 0.9556 |
| black_bear_wading | 29.7 | 0.9767 | 0.957 |
| ducks_in_reeds | 26.9 | 0.9772 | 0.9244 |
| fox_and_daisies | 30.9 | 0.9678 | 0.8874 |
| beached_boats | 24.2 | 0.9506 | 0.8826 |
| damselfly_on_leaf | 64.5 | 0.7877 | 0.4495 |
| runners_on_track | 44.2 | 0.9207 | 0.8535 |
| roller_coaster_loop | 80.3 | 0.9225 | 0.8537 |

### How isoluminant is too isoluminant

| Luma offset | Separation | Canny recall | Otsu IoU |
|---|---:|---:|---:|
| 0 | 0.116 | 0 | 0.219 |
| 1 | 1.217 | 0 | 0.3604 |
| 2 | 1.754 | 0 | 0.6971 |
| 4 | 4.443 | 0 | 0.989 |
| 8 | 8.158 | 0 | 1 |
| 12 | 12.294 | 0 | 1 |
| 16 | 16.062 | 0 | 1 |
| 20 | 20.369 | 0 | 1 |
| 24 | 24.115 | 0.6639 | 1 |
| 28 | 28.188 | 1 | 1 |
| 32 | 31.974 | 1 | 1 |

### Four photographs down the rows

| Sr | Photograph | Average (R+G+B)/3 | BT.601 (OpenCV default) | BT.709 (HDTV) | Linear-light BT.709 | Value (max channel) | Contrast-preserving |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | helicopter and pilot · chroma 15 | 2.60 | 0.00 | 0.74 | 0.65 | 6.89 | 2.23 |
| 2 | leopard along branch · chroma 23 | 5.43 | 0.00 | 1.08 | 1.54 | 6.86 | 7.01 |
| 3 | fox and daisies · chroma 31 | 6.81 | 0.00 | 1.13 | 1.77 | 11.21 | 5.28 |
| 4 | roller coaster loop · chroma 80 | 8.02 | 0.00 | 5.62 | 12.36 | 51.51 | 12.71 |
