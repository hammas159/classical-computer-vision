### Generated clutter against real photographs

| Background | Gradient + morphology | Local variance | QRCodeDetector | Decoded |
|---|---:|---:|---:|---:|
| Generated clutter | 1 | 1 | 1 | 1 |
| Photographs | 0.0208 | 1 | 1 | 1 |

### Per photograph

| Photograph | Barcode-like % | Gradient + morphology | Local variance | QRCodeDetector | Decoded |
|---|---:|---:|---:|---:|---:|
| bomber_over_cloud | 9.4 | 0.25 | 1 | 1 | 1 |
| turquoise_lake | 18.9 | 0 | 1 | 1 | 1 |
| polar_bears_playing | 22.2 | 0 | 1 | 1 | 1 |
| zebra_herd | 26.1 | 0 | 1 | 1 | 1 |
| canoe_on_the_lake | 29.7 | 0 | 1 | 1 | 1 |
| cougar_among_birches | 30.4 | 0 | 1 | 1 | 1 |
| trocadero_statue | 33 | 0 | 1 | 1 | 1 |
| skiff_in_weed | 33.9 | 0 | 1 | 1 | 1 |
| saguaro_blossom | 41.2 | 0 | 1 | 1 | 1 |
| coiled_rope | 43.4 | 0 | 1 | 1 | 1 |
| giraffes_drinking | 50 | 0 | 1 | 1 | 1 |
| buffalo_in_the_river | 76.7 | 0 | 1 | 1 | 1 |

### Blur

| Blur sigma | Found | Decoded | Gap |
|---|---:|---:|---:|
| 0 | 1 | 1 | 0 |
| 1 | 1 | 1 | 0 |
| 2 | 1 | 1 | 0 |
| 3.5 | 1 | 1 | 0 |
| 5 | 1 | 0 | 1 |
| 8 | 1 | 0 | 1 |

### Resolution

| Scale | Px per module | Found | Decoded |
|---|---:|---:|---:|
| 1 | 9.6 | 1 | 1 |
| 0.7 | 6.72 | 1 | 0 |
| 0.5 | 4.8 | 1 | 0 |
| 0.35 | 3.36 | 1 | 0 |
| 0.25 | 2.4 | 0.75 | 0 |

### Four backgrounds down the rows

| Sr | Background | Gradient + morphology | Local variance | QRCodeDetector | Decoded |
|---|---:|---:|---:|---:|---:|
| 1 | turquoise lake · barcode-like 19% | 0.071 | 0.924 | 0.701 | yes |
| 2 | canoe on the lake · barcode-like 30% | 0.273 | 0.882 | 0.701 | yes |
| 3 | coiled rope · barcode-like 43% | 0.250 | 0.938 | 0.701 | yes |
| 4 | buffalo in the river · barcode-like 77% | 0.250 | 0.804 | 0.701 | yes |
