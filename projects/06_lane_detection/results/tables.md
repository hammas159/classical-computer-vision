### Two numbers, one per failure mode

| Detector | Plausible | VP spread (px) | Lane width spread | Yaw error (% width) | Gap to ROI control (% width) |
|---|---:|---:|---:|---:|---:|
| Fixed guess (control) | 14/14 | 0.0 | 0.0000 | 2.637 | 3.51 |
| Bright pixels in the ROI (control) | 14/14 | 21.7 | 0.0554 | 1.203 | 0.00 |
| Canny + Hough | 14/14 | 7.6 | 0.0244 | 0.170 | 1.75 |
| HLS colour + Canny + Hough | 13/14 | 5.2 | 0.0152 | 0.129 | 0.72 |
| Lab colour + Canny + Hough | 14/14 | 7.0 | 0.0183 | 0.152 | 0.88 |
| HLS colour + Hough | 13/14 | 6.1 | 0.0151 | 0.294 | 0.46 |
| Sobel-x + Hough | 13/14 | 8.9 | 0.0170 | 0.116 | 0.54 |

### How far the ROI control sits from the full pipeline, per camera

| Camera | Frames | Median gap | Worst gap | Its pixels that are lane-coloured |
|---|---:|---:|---:|---:|
| 1280x720 | 8 | 4.71% | 21.36% | 66% |
| 960x540 | 6 | 0.32% | 1.10% | 81% |

### Removing one step at a time

| Variant | Plausible | VP spread (% width) | Lane width spread |
|---|---:|---:|---:|
| Full pipeline | 13/14 | 0.66 | 0.019 |
| without the colour mask | 14/14 | 0.81 | 0.032 |
| without the edge detector | 13/14 | 0.68 | 0.019 |
| without the slope filter | 14/14 | 0.85 | 0.111 |
| without the region of interest | 12/14 | 5.03 | 0.059 |

### The region of interest written in pixels

| Camera | Frames | ROI covers (fractions) | ROI covers (pixels) | Plausible (fractions) | Plausible (pixels) | VP spread (fractions) | VP spread (pixels) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1280x720 | 8 | 19.1% | 10.8% | 7/8 | 6/8 | 9.0 px | 16.0 px |
| 960x540 | 6 | 19.2% | 19.0% | 6/6 | 6/6 | 1.4 px | 1.4 px |

### Per photograph

| Photograph | Camera | Paint contrast | Shadow | Plausible | Spread between pipelines |
|---|---:|---:|---:|---:|---:|
| solidWhiteCurve | 960x540 | 141 | 0.0% | 5/5 | 0.38% |
| solidWhiteRight | 960x540 | 143 | 0.0% | 5/5 | 0.44% |
| solidYellowCurve | 960x540 | 96 | 0.0% | 5/5 | 0.68% |
| solidYellowCurve2 | 960x540 | 111 | 0.0% | 5/5 | 0.57% |
| solidYellowLeft | 960x540 | 105 | 0.0% | 5/5 | 0.47% |
| straight_lines1 | 1280x720 | 105 | 0.1% | 5/5 | 0.56% |
| straight_lines2 | 1280x720 | 170 | 0.1% | 5/5 | 0.53% |
| test1 | 1280x720 | 38 | 9.0% | 4/5 | 3.75% |
| test2 | 1280x720 | 82 | 0.0% | 3/5 | 2.22% |
| test3 | 1280x720 | 115 | 0.9% | 5/5 | 0.62% |
| test4 | 1280x720 | 111 | 13.8% | 5/5 | 2.90% |
| test5 | 1280x720 | 23 | 26.9% | 5/5 | 2.04% |
| test6 | 1280x720 | 108 | 0.7% | 5/5 | 1.05% |
| whiteCarLaneSwitch | 960x540 | 110 | 0.0% | 5/5 | 0.88% |
