### Foreground shape crossed with illumination

| Foreground | Illumination | Fixed 127 (control) | Otsu | Triangle | Multi-Otsu (3 class) | Adaptive mean | Adaptive Gaussian | Niblack | Sauvola |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solid | 1 | 1 | 1 | 1 | 1 | 0.447 | 0.2647 | 0.1318 | 0.4428 |
| solid | 0.8 | 1 | 1 | 0.8485 | 1 | 0.4372 | 0.2551 | 0.1661 | 0.4387 |
| solid | 0.6 | 0.842 | 1 | 0.8065 | 1 | 0.4261 | 0.2462 | 0.1405 | 0.4346 |
| solid | 0.45 | 0.319 | 0.9499 | 0.8065 | 1 | 0.4162 | 0.236 | 0.1329 | 0.4322 |
| solid | 0.3 | 0.2406 | 0.2983 | 0.8065 | 0.8929 | 0.4028 | 0.2249 | 0.1252 | 0.4303 |
| solid | 0.2 | 0.2082 | 0.2799 | 0.8065 | 0.4089 | 0.3911 | 0.2162 | 0.1208 | 0.4274 |
| solid | 0.1 | 0.1866 | 0.269 | 0.6836 | 0.3313 | 0.376 | 0.205 | 0.1162 | 0.4264 |
| thin | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.2406 | 1 |
| thin | 0.8 | 1 | 1 | 1 | 1 | 1 | 1 | 0.3127 | 1 |
| thin | 0.6 | 0.8608 | 1 | 1 | 1 | 1 | 1 | 0.2916 | 1 |
| thin | 0.45 | 0.3296 | 0.9549 | 1 | 1 | 1 | 1 | 0.2765 | 1 |
| thin | 0.3 | 0.2243 | 0.2907 | 1 | 0.8151 | 1 | 0.9994 | 0.2629 | 1 |
| thin | 0.2 | 0.1985 | 0.2562 | 0.2562 | 0.5487 | 1 | 0.9974 | 0.2532 | 1 |
| thin | 0.1 | 0.1815 | 0.2403 | 0.1717 | 0.3914 | 1 | 0.9828 | 0.2459 | 1 |

### Methods on the default scene

| Method | IoU | Dice | Time (ms) |
|---|---:|---:|---:|
| Fixed 127 (control) | 1 | 1 | 0.245 |
| Otsu | 1 | 1 | 0.22 |
| Triangle | 1 | 1 | 0.19 |
| Multi-Otsu (3 class) | 1 | 1 | 1.277 |
| Adaptive mean | 0.3813 | 0.5517 | 0.324 |
| Adaptive Gaussian | 0.2217 | 0.3627 | 1.442 |
| Niblack | 0.3771 | 0.5475 | 15.363 |
| Sauvola | 0.3779 | 0.5482 | 17.287 |
| Best global (oracle) | 1 | 1 | n/a |

### Four scenes down the rows

| Sr | Scene | Fixed 127 (control) | Otsu | Triangle | Multi-Otsu (3 class) | Adaptive mean | Adaptive Gaussian | Niblack | Sauvola | Best global (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | solid shapes · even light | 1.000 | 1.000 | 1.000 | 1.000 | 0.447 | 0.265 | 0.132 | 0.443 | 1.000 |
| 2 | solid shapes · strong gradient | 0.241 | 0.298 | 0.806 | 0.893 | 0.403 | 0.225 | 0.125 | 0.430 | 1.000 |
| 3 | thin strokes · even light | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.241 | 1.000 | 1.000 |
| 4 | thin strokes · severe gradient | 0.189 | 0.249 | 0.346 | 0.443 | 1.000 | 0.992 | 0.248 | 1.000 | 0.741 |
