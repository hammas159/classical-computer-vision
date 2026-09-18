### Every detector, both arms

| Detector | Detection rate | IoU | False alarm share | Control? |
|---|---:|---:|---:|---:|
| Flag nothing (control) | 0 | 0 | 0 | True |
| Flag everything (control) | 1 | 0.0107 | 1 | True |
| Median residual | 0.3125 | 0.048 | 0.14265 | False |
| Gaussian residual | 0.1667 | 0.0112 | 0.05356 | False |
| Morphological top/black-hat | 0.125 | 0.0061 | 0.07187 | False |
| Local standard deviation | 0.5 | 0.1169 | 0.04392 | False |
| Fourier notch | 0.1667 | 0.0091 | 0.0663 | False |
| Spectral residual | 0.3333 | 0.0365 | 0.03697 | False |

### Which detector finds which defect

| Defect | Median residual | Gaussian residual | Morphological top/black-hat | Local standard deviation | Fourier notch | Spectral residual |
|---|---:|---:|---:|---:|---:|---:|
| scratch | 0.8333 | 0.6667 | 0.3333 | 0.4167 | 0.5833 | 0.9167 |
| blob | 0.25 | 0 | 0.0833 | 0.3333 | 0.0833 | 0 |
| hole | 0.0833 | 0 | 0.0833 | 0.6667 | 0 | 0.4167 |
| smear | 0.0833 | 0 | 0 | 0.5833 | 0 | 0 |

### The surface decides

| Surface | Uniformity | Noise ceiling | Best detection | Lowest false alarm | Worst false alarm |
|---|---:|---:|---:|---:|---:|
| surface_fine_weave | 1.75 | 4.45 | 0.25 | 0 | 0.02358 |
| surface_brick_paving | 2.07 | 4.45 | 0.75 | 0 | 0.75649 |
| surface_sand_ripple | 3.67 | 11.86 | 0.75 | 0.00407 | 0.04108 |
| surface_water_ripples | 4.37 | 5.93 | 0.75 | 0.00087 | 0.08876 |
| surface_field_mosaic | 4.95 | 2.97 | 0.5 | 0.00158 | 0.37214 |
| surface_pebbled_render | 8.66 | 8.9 | 0.5 | 0.00219 | 0.01286 |
| surface_brick_wall | 10.02 | 5.93 | 0.5 | 0.01819 | 0.2649 |
| surface_coarse_cloth | 10.04 | 10.38 | 1 | 0 | 0.02891 |
| surface_dry_grass | 13.72 | 37.06 | 0.75 | 0 | 0.00683 |
| surface_roof_slates | 14.03 | 5.93 | 1 | 0 | 0.68266 |
| surface_straw_thatch | 19.3 | 37.06 | 0.75 | 0 | 0.02605 |
| surface_knitted_fabric | 20.24 | 5.93 | 0.5 | 0.00193 | 0.10913 |

### What filling is worth

| Detector | IoU raw | IoU filled | Gain |
|---|---:|---:|---:|
| Median residual | 0.0495 | 0.0431 | -0.0064 |
| Gaussian residual | 0.0168 | 0.0154 | -0.0015 |
| Morphological top/black-hat | 0.0056 | 0.0065 | 0.0009 |
| Local standard deviation | 0.0382 | 0.0404 | 0.0023 |
| Fourier notch | 0.0115 | 0.0118 | 0.0003 |
| Spectral residual | 0.0327 | 0.0321 | -0.0006 |

### How large a defect has to be

| Severity | Median residual | Gaussian residual | Morphological top/black-hat | Local standard deviation | Fourier notch | Spectral residual |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0938 | 0.0625 | 0.0938 | 0.2812 | 0.0938 | 0.0938 |
| 2 | 0.125 | 0.0625 | 0.0938 | 0.2812 | 0.0938 | 0.0938 |
| 4 | 0.1562 | 0.0625 | 0.0938 | 0.375 | 0.125 | 0.1875 |
| 8 | 0.3125 | 0.25 | 0.1875 | 0.4375 | 0.25 | 0.3125 |
| 16 | 0.3125 | 0.25 | 0.3125 | 0.5938 | 0.2812 | 0.3438 |

### Found or missed, on one surface

| Sr | Defect | Median residual | Gaussian residual | Morphological top/black-hat | Local standard deviation | Fourier notch | Spectral residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | scratch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | blob | — | — | — | ✅ | — | — |
| 3 | hole | — | — | — | ✅ | — | — |
| 4 | smear | — | — | — | ✅ | — | — |
