### Matched intensities

| Method | Mean error (px) | Success rate | Time (ms) |
|---|---:|---:|---:|
| Phase correlation | 0.0028 | 1 | 3.9 |
| Phase corr. (no window) | 0.0125 | 1 | 3.43 |
| ECC | 0.0117 | 1 | 18.68 |
| Mutual information | 0 | 1 | 1661.52 |

### Four intensity relationships

| Intensities | Phase correlation | Phase corr. (no window) | ECC | Mutual information |
|---|---:|---:|---:|---:|
| same | 0.0028 | 0.0125 | 0.0117 | 0 |
| gamma | 0.0422 | 0.0381 | 0.0344 | 0 |
| inverted | 151.241 | 0.0612 | nan | 0 |
| synthetic_mri | 19.5018 | 38.9063 | 32.7036 | 0 |

### The Hanning window on genuine crop pairs

| Image | With Hanning (px) | Without (px) |
|---|---:|---:|
| elk_in_long_grass | 0.7015 | 0.7001 |
| yacht_and_bridge | 0.7018 | 0.6076 |
| sled_dogs_on_ice | 0.703 | 0.6754 |
| husky_puppies | 0.7072 | 0.6937 |
| giraffe_head_on | 0.7024 | 0.6368 |
| long_jetty | 0.7036 | 0.7037 |
| biwa_player | 0.6932 | 0.6876 |
| marmot_on_rock | 0.7058 | 0.6945 |
| cannon_on_cobbles | 0.7172 | 0.6629 |
| layered_sandstone | 0.71 | 0.6917 |
| stone_guardian | 0.7075 | 0.6943 |
| snake_on_needles | 0.6961 | 0.7114 |

### The Hanning window on inverted intensities

| Image | With Hanning (px) | Without (px) |
|---|---:|---:|
| elk_in_long_grass | 124.044 | 0.1235 |
| yacht_and_bridge | 211.049 | 0.0122 |
| sled_dogs_on_ice | 30.4457 | 0.0531 |
| husky_puppies | 173.892 | 0.0523 |
| giraffe_head_on | 208.435 | 0.0853 |
| long_jetty | 635.424 | 0.0464 |
| biwa_player | 156.081 | 0.0837 |
| marmot_on_rock | 0.0084 | 0.0798 |
| cannon_on_cobbles | 123.957 | 0.0452 |
| layered_sandstone | 3.8972 | 0.0523 |
| stone_guardian | 147.644 | 0.0531 |
| snake_on_needles | 0.015 | 0.0476 |

### Rotation

| Rotation (deg) | ECC angle error (deg) | Phase corr. error (px) |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 0.0004 | 0.6176 |
| 3 | 0.0007 | 3.9762 |
| 7 | 0.0003 | 21.8504 |
| 15 | 0.0003 | 29.4118 |

### Four scenes down the rows

| Sr | Scene | Phase correlation | Phase corr. (no window) | ECC | Mutual information |
|---|---:|---:|---:|---:|---:|
| 1 | yacht and bridge · texture 11.8 | 211.049 | 0.012 | n/a | 0.000 |
| 2 | long jetty · texture 17.6 | 635.424 | 0.046 | n/a | 0.000 |
| 3 | biwa player · texture 18.0 | 156.081 | 0.084 | n/a | 0.000 |
| 4 | stone guardian · texture 27.1 | 147.644 | 0.053 | n/a | 0.000 |
