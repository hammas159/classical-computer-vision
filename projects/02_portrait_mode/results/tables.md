### Subject matting (12 scenes, GrabCut pinned to seed 0)

| Method | Subject found | IoU | Dice | Body recall | Fine detail recall | Background FPR | Boundary F1 | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Face rect (baseline) | 100% | 0.3881 | 0.5591 | 0.5825 | 0.3908 | 0.0892 | 0.0292 | 19.838 |
| Face ellipse prior | 100% | 0.3457 | 0.5137 | 0.4814 | 0.2174 | 0.0501 | 0.0235 | 19.595 |
| Haar + GrabCut | 100% | 0.4765 | 0.6454 | 0.5214 | 0.325 | 0 | 0.3814 | 311.782 |
| GrabCut (centre rect) | 100% | 0.7117 | 0.8316 | 0.6978 | 0.7604 | 0.0001 | 0.6868 | 543.097 |
| Skin colour (YCrCb) | 100% | 0.1909 | 0.3206 | 0.1667 | 0.4106 | 0.038 | 0.2016 | 1.257 |
| Watershed + markers | 100% | 0.3488 | 0.5172 | 0.5144 | 0.2734 | 0.0728 | 0.1432 | 25.602 |

### GrabCut seed stability (24 seeds on each of 5 photographs)

| Scene | Seeds | Seed agreement | IoU mean | IoU std | Worst | Best | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| footballer · person, crowd behind | 24 | 0.8655 | 0.89 | 0.1267 | 0.5953 | 0.9655 | 0.3701 |
| girl · person, soft background | 24 | 0.9333 | n/a | n/a | n/a | n/a | n/a |
| dog · animal, head on | 24 | 0.9988 | n/a | n/a | n/a | n/a | n/a |
| butterfly · insect, busy background | 24 | 0.9963 | n/a | n/a | n/a | n/a | n/a |
| coffee cup · object, table top | 24 | 0.9776 | n/a | n/a | n/a | n/a | n/a |

### Bokeh kernels (radius 15 px)

| Kernel | Radius (px) | Peak / mean | Rim energy |
|---|---:|---:|---:|
| Gaussian | 15 | 2.9419 | 0.2023 |
| Box | 15 | 1 | 0.3205 |
| Disc (circular aperture) | 15 | 1 | 0.4344 |
| Hexagon (6-blade) | 15 | 1 | 0.3317 |

### Compositing (12 scenes, true matte, disc r=15)

| Strategy | Halo error (0-255) | Whole-background error | Time (ms) |
|---|---:|---:|---:|
| Naive (blur all, paste back) | 12.545 | 1.661 | 8.572 |
| Masked (normalised convolution) | 2.036 | 0.273 | 25.63 |
