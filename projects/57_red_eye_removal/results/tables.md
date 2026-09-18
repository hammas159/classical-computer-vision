### Detectors on six real portraits

| Detector | IoU | Pupil recall | FP area / pupil area | ms |
|---|---:|---:|---:|---:|
| Colour only (control) | 0.0225 | 0.7277 | 64.155 | 1.67 |
| Colour + shape | 0.1687 | 0.6918 | 14.589 | 1.84 |
| Face-constrained | 0.5979 | 0.6918 | 0.295 | 17.3 |
| Eye-constrained | 0.3899 | 0.4847 | 0.259 | 12.68 |

### Generated against real

| Detector | Generated IoU | Real IoU | Generated FP | Real FP |
|---|---:|---:|---:|---:|
| Colour only (control) | 0.06 | 0.0225 | 15.754 | 64.155 |
| Colour + shape | 0.0675 | 0.1687 | 14.417 | 14.589 |
| Face-constrained | 0.9842 | 0.5979 | 0 | 0.295 |
| Eye-constrained | 0.9842 | 0.3899 | 0 | 0.259 |

### Six photographs with no red-eye at all

| Photograph | Pupil-like blobs | Colour only (control) | Colour + shape | Face-constrained | Eye-constrained |
|---|---:|---:|---:|---:|---:|
| orange_lichen_on_rock | 43 | 6007 | 1714 | 0 | 0 |
| children_carrying_pots | 59 | 34871 | 3259 | 98 | 0 |
| feather_duster_worms | 39 | 33328 | 2984 | 0 | 0 |
| runners_in_the_stadium | 38 | 5003 | 1951 | 0 | 0 |
| scattered_sweets | 12 | 3613 | 2645 | 0 | 47 |
| red_brick_house | 19 | 3472 | 775 | 0 | 0 |

### Corrections on real skin, true mask

| Photograph | Zero red channel | Mean of G and B | Desaturate (feathered) | Did nothing |
|---|---:|---:|---:|---:|
| two_women_in_headdress | 13.06 | 20.56 | 20.94 | 12.21 |
| girl_with_tulips | 16.26 | 22.89 | 18.72 | 11.79 |
| woman_in_red_scarf | 16.08 | 21.76 | 20.25 | 12.24 |
| girl_in_pink_shirt | 10.28 | 22.92 | 21.14 | 12.28 |
| two_firefighters | 9.55 | 16.77 | 21.51 | 13.79 |
| woman_with_curly_hair | 10.27 | 17.78 | 19.44 | 12.94 |

### The whole pipeline on real photographs

| Detector | Correction | Pupil PSNR (dB) |
|---|---:|---:|
| Colour only (control) | Mean of G and B | 20.084 |
| Colour + shape | Mean of G and B | 20.208 |
| Face-constrained | Mean of G and B | 20.208 |
| Eye-constrained | Mean of G and B | 17.816 |
| Did nothing (control) | none | 12.542 |

### Four photographs down the rows

| Sr | Photograph | Pupils | Pupil-like blobs | Colour only (control) | Colour + shape | Face-constrained | Eye-constrained |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | two women in headdress | 4 | 10 | 0.015 | 0.058 | 0.799 | 0.352 |
| 2 | girl with tulips | 2 | 8 | 0.006 | 0.012 | 0.755 | 0.378 |
| 3 | woman in red scarf | 2 | 2 | 0.018 | 0.428 | 0.784 | 0.784 |
| 4 | woman with curly hair | 2 | 2 | 0.078 | 0.084 | 0.429 | 0.225 |
