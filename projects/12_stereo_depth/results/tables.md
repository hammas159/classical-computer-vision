### Generated pairs (12 scenes)

| Method | Density | MAE (px) | Bad 1px (all) | Bad 2px (answered) | Bad 2px (all) | Bad 4px (all) | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Block matching (BM) | 0.7742 | 0.3535 | 0.2388 | 0.0132 | 0.236 | 0.2326 | 4.78 |
| Semi-global (SGBM) | 0.8279 | 0.492 | 0.1916 | 0.0204 | 0.189 | 0.1858 | 35.99 |
| Semi-global, 8-path (HH) | 0.8274 | 0.3683 | 0.1863 | 0.0132 | 0.1835 | 0.1802 | 52.553 |
| Naive SAD (no filtering) | 0.8394 | 1.0284 | 0.2062 | 0.0496 | 0.2021 | 0.1955 | 77.542 |
| Constant disparity (control) | 1 | 11.2887 | 0.9981 | 0.997 | 0.997 | 0.8754 | 0.041 |

### The real Middlebury Aloe pair

| Method | Density | MAE (px) | Bad 1px (all) | Bad 2px (answered) | Bad 2px (all) | Bad 4px (all) | Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Block matching (BM) | 0.7105 | 1.0307 | 0.3461 | 0.046 | 0.3222 | 0.3133 | 13.15 |
| Semi-global (SGBM) | 0.8114 | 1.5878 | 0.268 | 0.0717 | 0.2468 | 0.2361 | 83.326 |
| Semi-global, 8-path (HH) | 0.8071 | 1.5089 | 0.2749 | 0.0666 | 0.2466 | 0.2367 | 128.427 |
| Naive SAD (no filtering) | 0.8835 | 3.4374 | 0.2903 | 0.1691 | 0.2659 | 0.2413 | 224.774 |
| Constant disparity (control) | 1 | 19.8647 | 1 | 1 | 1 | 1 | 0.247 |

### Bad-2px rate by region, on Aloe

| Method | discontinuity | textureless | well-textured |
|---|---:|---:|---:|
| Block matching (BM) | 0.4189 | 0.1611 | 0.2382 |
| Semi-global (SGBM) | 0.3033 | 0.1403 | 0.1998 |
| Semi-global, 8-path (HH) | 0.3005 | 0.1405 | 0.2025 |
| Naive SAD (no filtering) | 0.3295 | 0.1911 | 0.2056 |
| Constant disparity (control) | 1 | 1 | 1 |

### Four pairs down the rows

| Sr | Scene | Block matching (BM) | Semi-global (SGBM) | Semi-global, 8-path (HH) | Naive SAD (no filtering) | Constant disparity (control) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | windows and flowers · repeating shutters | 16.2% bad | 13.4% bad | 12.9% bad | 14.5% bad | 99.8% bad |
| 2 | tiger on rocks · stripes and rubble | 20.8% bad | 15.0% bad | 14.8% bad | 16.8% bad | 99.6% bad |
| 3 | temple dragon · ornament against towers | 22.9% bad | 17.3% bad | 16.9% bad | 19.4% bad | 99.7% bad |
| 4 | wolf in leaf litter · scattered fine detail | 23.2% bad | 18.3% bad | 17.8% bad | 19.8% bad | 99.6% bad |
