### Subject matting (12 scenes, GrabCut pinned to seed 0)

| Method | Subject found | IoU | Dice | Body recall | Fine detail recall | Background FPR | Boundary F1 | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Face rect (baseline) | 100% | 0.3881 | 0.5591 | 0.5825 | 0.3908 | 0.0892 | 0.0292 | 18.168 |
| Face ellipse prior | 100% | 0.3457 | 0.5137 | 0.4814 | 0.2174 | 0.0501 | 0.0235 | 18.949 |
| Haar + GrabCut | 100% | 0.4765 | 0.6454 | 0.5214 | 0.325 | 0 | 0.3814 | 288.431 |
| GrabCut (centre rect) | 100% | 0.7117 | 0.8316 | 0.6978 | 0.7604 | 0.0001 | 0.6868 | 496.146 |
| Skin colour (YCrCb) | 100% | 0.1909 | 0.3206 | 0.1667 | 0.4106 | 0.038 | 0.2016 | 1.242 |
| Watershed + markers | 100% | 0.3488 | 0.5172 | 0.5144 | 0.2734 | 0.0728 | 0.1432 | 23.657 |

### GrabCut seed stability (24 seeds on each identical image)

| Scene | Seeds | IoU mean | IoU std | Worst | Best | Spread |
|---|---:|---:|---:|---:|---:|---:|
| 0 (coffee) | 24 | 0.4761 | 0.0005 | 0.4752 | 0.477 | 0.0019 |
| 1 (rocket) | 24 | 0.4761 | 0.0005 | 0.4752 | 0.477 | 0.0019 |
| 2 (grass) | 24 | 0.4761 | 0.0005 | 0.4752 | 0.477 | 0.0019 |
| 3 (brick) | 24 | 0.4761 | 0.0005 | 0.4752 | 0.477 | 0.0019 |

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
| Naive (blur all, paste back) | 12.545 | 1.661 | 8.101 |
| Masked (normalised convolution) | 2.036 | 0.273 | 23.77 |
