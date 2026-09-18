### Segmenters against a matte that is exact by construction

| Segmenter | Mean IoU | Median IoU | Recovered (IoU>=0.5) | Feature gap | Fingers |
|---|---:|---:|---:|---:|---:|
| Oracle mask (control) | 1.000 | 1.000 | 27/27 | 0.000 | 3/5 |
| Whole frame (control) | 0.103 | 0.085 | 0/27 | 0.758 | 0/5 |
| Centre ellipse (control) | 0.441 | 0.419 | 9/27 | 0.489 | 0/5 |
| Nothing (control) | 0.000 | 0.000 | 0/27 | 1.222 | 0/5 |
| YCrCb skin | 0.477 | 0.416 | 13/27 | 0.340 | 1/5 |
| HSV skin | 0.554 | 0.523 | 15/27 | 0.192 | 3/5 |
| Lab skin | 0.747 | 0.865 | 23/27 | 0.060 | 2/5 |
| YCrCb and HSV | 0.586 | 0.537 | 16/27 | 0.212 | 2/5 |
| Adaptive Cr | 0.104 | 0.090 | 0/27 | 0.757 | 0/5 |
| Otsu on grey | 0.135 | 0.096 | 0/27 | 0.461 | 0/5 |
| GrabCut from a box | 0.665 | 0.670 | 20/27 | 0.186 | 2/5 |

### Finger counting on a perfect mask

| Rule | Parameter | Correct |
|---|---:|---:|
| convexity defects + 1 | 0.01 | 1/27 |
| convexity defects + 1 | 0.02 | 0/27 |
| convexity defects + 1 | 0.04 | 1/27 |
| convexity defects + 1 | 0.08 | 1/27 |
| palm-circle crossings | 1.6 | 2/27 |
| palm-circle crossings | 1.8 | 3/27 |
| palm-circle crossings | 2 | 2/27 |
| palm-circle crossings | 2.2 | 1/27 |
| palm-circle crossings | 2.6 | 1/27 |
| palm-circle crossings | 3 | 0/27 |

### Every background, for Lab skin

| Background | Skin-like | Mean IoU | Median IoU | Total failures |
|---|---:|---:|---:|---:|
| dolphins in open water | 0.0% | 0.915 | 0.939 | 0/27 |
| a painted mural and a city street | 1.7% | 0.907 | 0.925 | 0/27 |
| a cormorant on a branch | 6.9% | 0.915 | 0.944 | 0/27 |
| a black dog on grass | 9.4% | 0.363 | 0.000 | 15/27 |
| a polo player on a horse | 16.7% | 0.777 | 0.802 | 0/27 |
| the steps of a Mayan pyramid | 24.3% | 0.915 | 0.944 | 0/27 |
| a bronze human figure on a rock | 33.4% | 0.914 | 0.940 | 0/27 |
| a woman on a beach — real skin in the background | 44.1% | 0.670 | 0.672 | 0/27 |
| people unloading a plane on snow | 49.8% | 0.501 | 0.701 | 9/27 |
| orange koi carp under water | 63.0% | 0.718 | 0.765 | 1/27 |
| a guard beside a sentry box | 75.1% | 0.504 | 0.503 | 1/27 |
| a man sweeping beside a sunlit sandy wall | 98.6% | 0.084 | 0.081 | 5/27 |

### How exact the inherited matte is

| Gesture | Halo |
|---|---:|
| 4 | 11.30% |
| F | 6.96% |
| H | 4.61% |
| U | 2.02% |
| C | 1.41% |
| A | 1.29% |
