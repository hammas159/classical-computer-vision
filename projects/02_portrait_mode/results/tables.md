### Subject matting (12 scenes, GrabCut pinned to seed 0)

| Method | Subject found | IoU | Dice | Body recall | Hair recall | Background FPR | Boundary F1 | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Face rect (baseline) | 100% | 0.5654 | 0.7224 | 0.9422 | 0.97 | 0.3441 | 0.0236 | 34.56 |
| Face ellipse prior | 100% | 0.928 | 0.9627 | 0.9664 | 0.0624 | 0.0091 | 0.1659 | 39.224 |
| Haar + GrabCut | 100% | 0.8721 | 0.929 | 0.9343 | 0.4823 | 0.0368 | 0.5849 | 1092.21 |
| GrabCut (centre rect) | 100% | 0.7985 | 0.8716 | 0.8683 | 0.5285 | 0.0454 | 0.6362 | 1156.48 |
| Skin colour (YCrCb) | 100% | 0.1735 | 0.2914 | 0.2227 | 0.947 | 0.3288 | 0.25 | 3.041 |
| Watershed + markers | 100% | 0.7155 | 0.833 | 0.7876 | 0.2362 | 0.0452 | 0.3728 | 49.524 |

### GrabCut seed stability (24 seeds on each identical image)

| Scene | Seeds | IoU mean | IoU std | Worst | Best | Spread |
|---|---:|---:|---:|---:|---:|---:|
| 0 (coffee) | 24 | 0.6634 | 0.1162 | 0.1515 | 0.9039 | 0.7524 |
| 1 (rocket) | 24 | 0.8905 | 0.0177 | 0.8491 | 0.9064 | 0.0574 |
| 2 (grass) | 24 | 0.9148 | 0.0054 | 0.8926 | 0.92 | 0.0273 |
| 3 (brick) | 24 | 0.941 | 0.0008 | 0.9402 | 0.9421 | 0.0019 |

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
| Naive (blur all, paste back) | 9.48 | 1.636 | 15.81 |
| Masked (normalised convolution) | 1.576 | 0.271 | 45.523 |
