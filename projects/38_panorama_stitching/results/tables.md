### Every blender on all twelve

| Blender | PSNR (dB) | Seam step | Seam visibility | Control? |
|---|---:|---:|---:|---:|
| Keep the first frame (control) | 23.853 | 29.92 | 1.257 | True |
| Overwrite (control) | 21.608 | 20.5 | 1.477 | True |
| Average | 23.194 | 22.25 | 1.289 | False |
| Feather (41 px) | 22.198 | 19.79 | 1.125 | False |
| Multi-band (5 levels) | 22.259 | 19.5 | 1.132 | False |

### Registrability: the selection axis

| Photograph | Matches | Inliers | Inlier rate | Per Mpx |
|---|---:|---:|---:|---:|
| brick_wall_courses | 6 | 5 | 0.8333 | 19.1 |
| roof_shingles | 1040 | 972 | 0.9346 | 2373 |
| coastal_city_from_the_air | 1977 | 1973 | 0.998 | 4816.9 |
| irrigated_fields_from_the_air | 806 | 789 | 0.9789 | 1926.3 |
| tugboat_under_the_bridge | 318 | 311 | 0.978 | 2014.2 |
| coarse_woven_fabric | 2460 | 2457 | 0.9988 | 5998.5 |
| elephant_crossing_a_road | 397 | 392 | 0.9874 | 2538.8 |
| cyclists_on_a_track | 560 | 552 | 0.9857 | 3575.1 |
| tank_in_scrub | 1057 | 1050 | 0.9934 | 4005.4 |
| street_grid_from_the_air | 1778 | 1776 | 0.9989 | 6774.9 |
| fibrous_matting | 2199 | 2197 | 0.9991 | 8380.9 |
| hillside_town_from_the_air | 743 | 737 | 0.9919 | 11245.7 |

### The exposure difference is the whole story

| Exposure | Keep the first frame (control) | Overwrite (control) | Average | Feather (41 px) | Multi-band (5 levels) |
|---|---:|---:|---:|---:|---:|
| 1 | 17.92 | 17.67 | 17.5 | 17.58 | 17.75 |
| 1.05 | 18.92 | 18.46 | 18.04 | 17.92 | 17.88 |
| 1.1 | 22.08 | 19.54 | 19.17 | 18.67 | 18.33 |
| 1.2 | 29.92 | 20.5 | 22.25 | 19.79 | 19.5 |
| 1.4 | 49.67 | 21.58 | 29.38 | 21.17 | 20.75 |

### Pyramid levels

| Bands | Seam step |
|---|---:|
| 2 | 15 |
| 3 | 15 |
| 5 | 14.67 |
| 7 | 15.5 |

### Feather ramp width

| Width | Seam step |
|---|---:|
| 5 | 15.17 |
| 21 | 14.5 |
| 41 | 15 |
| 81 | 16 |
| 161 | 18.17 |

### Four photographs down the rows

| Sr | Photograph | Inliers/Mpx | Keep the first frame (control) | Overwrite (control) | Average | Feather (41 px) | Multi-band (5 levels) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | brick wall courses | 19 | 21 | 7 | 11 | 6 | 6 |
| 2 | tugboat under the bridge | 2014 | 26 | 13 | 15 | 9 | 9 |
| 3 | cyclists on a track | 3575 | 28 | 24 | 24 | 24 | 24 |
| 4 | hillside town from the air | 11246 | 43 | 42 | 42 | 42 | 41 |
