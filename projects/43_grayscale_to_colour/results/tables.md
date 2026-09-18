### Every method on all twelve

| Method | Chroma error | RGB PSNR (dB) | Colourfulness | Oracle? |
|---|---:|---:|---:|---:|
| Do nothing (grey, control) | 18.675 | 22.24 | 0.28 | False |
| Global mean chroma (control) | 10.119 | 26.469 | 12.99 | True |
| Pseudo-colour (viridis) | 45.288 | 13.281 | 92.58 | False |
| Welsh transfer (reference) | 25.274 | 18.636 | 37.99 | False |
| Levin scribbles (40) | 5.674 | 30.676 | 30.66 | True |
| Luminance lookup (oracle) | 7.49 | 29.291 | 27.15 | True |

### The selection axis against the oracle's residual

| Photograph | Ambiguity | Oracle chroma error |
|---|---:|---:|
| seal_on_grey_ice | 1.996 | 1.66 |
| farmland_from_the_air | 4.097 | 3.569 |
| wine_bottles_in_a_rack | 4.782 | 4.08 |
| llama_at_a_stone_wall | 5.685 | 4.977 |
| tree_against_tropical_sky | 6.57 | 5.412 |
| otter_on_a_log | 7.146 | 6.298 |
| chicks_in_a_nest | 8.229 | 7.342 |
| lizard_on_pebbles | 8.979 | 7.556 |
| bears_at_the_water | 11.186 | 9.604 |
| glacier_cave_mouth | 12.46 | 11.074 |
| soldier_on_the_grass | 13.831 | 12.633 |
| woman_in_a_red_top | 19.555 | 15.677 |

### What PSNR is actually measuring

| Photograph | PSNR, luminance only (dB) | PSNR, chroma only (dB) | Chroma error, luminance only | Chroma error, chroma only |
|---|---:|---:|---:|---:|
| seal_on_grey_ice | 31.235 | 16.626 | 6.377 | 0.03 |
| farmland_from_the_air | 15.8 | 19.623 | 44.605 | 0.041 |
| wine_bottles_in_a_rack | 30.484 | 11.547 | 6.304 | 0.047 |
| llama_at_a_stone_wall | 27.22 | 11.379 | 8.978 | 0.044 |
| tree_against_tropical_sky | 14.944 | 11.042 | 23.048 | 0.953 |
| otter_on_a_log | 23.675 | 16.246 | 16.217 | 0.146 |
| chicks_in_a_nest | 19.6 | 18.334 | 26.493 | 0.095 |
| lizard_on_pebbles | 21.275 | 13.874 | 17.638 | 0.028 |
| bears_at_the_water | 22.055 | 17.193 | 15.541 | 0.044 |
| glacier_cave_mouth | 22.378 | 14.374 | 16.104 | 0.086 |
| soldier_on_the_grass | 20.637 | 15.413 | 21.414 | 0.04 |
| woman_in_a_red_top | 17.574 | 11.824 | 21.383 | 0.783 |

### How many scribbles Levin's method needs

| Scribbles | Chroma error |
|---|---:|
| 1 | 13.703 |
| 4 | 9.346 |
| 16 | 7.016 |
| 40 | 5.674 |
| 100 | 4.454 |
| 250 | 3.162 |

### Welsh transfer with every other photograph as reference

| Photograph | Rule reference | With rule reference | Best reference | Best | Worst | Spread | Grey control |
|---|---:|---:|---:|---:|---:|---:|---:|
| seal_on_grey_ice | farmland_from_the_air | 44.059 | wine_bottles_in_a_rack | 10.385 | 44.059 | 33.674 | 6.377 |
| farmland_from_the_air | wine_bottles_in_a_rack | 42.118 | tree_against_tropical_sky | 42.053 | 71.042 | 28.99 | 44.605 |
| wine_bottles_in_a_rack | llama_at_a_stone_wall | 12.953 | seal_on_grey_ice | 10.427 | 39.724 | 29.297 | 6.304 |
| llama_at_a_stone_wall | tree_against_tropical_sky | 27.063 | otter_on_a_log | 11.425 | 48.14 | 36.714 | 8.978 |
| tree_against_tropical_sky | otter_on_a_log | 35.235 | bears_at_the_water | 19.019 | 45.557 | 26.539 | 23.048 |
| otter_on_a_log | chicks_in_a_nest | 13.279 | llama_at_a_stone_wall | 10.187 | 60.132 | 49.946 | 16.217 |
| chicks_in_a_nest | lizard_on_pebbles | 19.581 | otter_on_a_log | 13.731 | 70.469 | 56.738 | 26.493 |
| lizard_on_pebbles | bears_at_the_water | 17.063 | otter_on_a_log | 10.699 | 57.933 | 47.233 | 17.638 |
| bears_at_the_water | glacier_cave_mouth | 16.418 | seal_on_grey_ice | 16.083 | 54.29 | 38.207 | 15.541 |
| glacier_cave_mouth | soldier_on_the_grass | 27.082 | wine_bottles_in_a_rack | 14.912 | 48.662 | 33.75 | 16.104 |
| soldier_on_the_grass | woman_in_a_red_top | 25.082 | otter_on_a_log | 16.263 | 63.137 | 46.874 | 21.414 |
| woman_in_a_red_top | seal_on_grey_ice | 23.352 | llama_at_a_stone_wall | 18.251 | 59.732 | 41.482 | 21.383 |

### Four photographs down the rows

| Sr | Photograph | Ambiguity | Do nothing (grey) | Global mean chroma (control) | Pseudo-colour (viridis) | Welsh transfer (reference) | Levin scribbles (40) | Luminance lookup (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | seal on grey ice | 2.0 | 6.4 | 5.3 | 28.2 | 44.1 | 2.1 | 1.7 |
| 2 | llama at a stone wall | 5.7 | 9.0 | 6.2 | 48.5 | 27.1 | 3.8 | 5.0 |
| 3 | lizard on pebbles | 9.0 | 17.6 | 8.8 | 46.0 | 17.1 | 8.1 | 7.6 |
| 4 | woman in a red top | 19.6 | 21.4 | 18.7 | 57.0 | 23.4 | 11.2 | 15.7 |
