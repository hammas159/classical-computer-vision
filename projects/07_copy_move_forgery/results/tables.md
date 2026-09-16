### Exact 96px copy-move (6 images)

| Method | Mask IoU | Precision | Recall | Pixel accuracy | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Block matching | 0.9925 | 0.9974 | 0.9951 | 0.9995 | 224.148 |
| SIFT + similarity verify | 0.6767 | 0.7531 | 0.8854 | 0.9232 | 54.847 |
| ORB + similarity verify | 0.5155 | 0.6305 | 0.8293 | 0.8257 | 82.883 |
| SIFT + translation verify | 0.7944 | 0.9083 | 0.8675 | 0.98 | 45.028 |
| SIFT blobs (no verify) | 0.4441 | 0.644 | 0.6648 | 0.9214 | 40.055 |
| Predict nothing (control) | 0 | 0 | 0 | 0.9181 | 0.01 |

### Mask IoU vs paste rotation

| Rotation (deg) | Block matching | SIFT + similarity verify | ORB + similarity verify | SIFT + translation verify | SIFT blobs (no verify) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9925 | 0.6767 | 0.5155 | 0.7944 | 0.4441 |
| 2 | 0 | 0.6633 | 0.5162 | 0.2973 | 0.3857 |
| 5 | 0 | 0.6477 | 0.4559 | 0.1315 | 0.3797 |
| 15 | 0 | 0.5916 | 0.4062 | 0.101 | 0.3318 |
| 30 | 0 | 0.6023 | 0.4217 | 0.0246 | 0.3364 |
| 45 | 0 | 0.536 | 0.3917 | 0.019 | 0.3527 |
| 90 | 0 | 0.4986 | 0.4583 | 0 | 0.4329 |

### Mask IoU vs paste scale

| Scale | Block matching | SIFT + similarity verify | ORB + similarity verify | SIFT + translation verify | SIFT blobs (no verify) |
|---|---:|---:|---:|---:|---:|
| 0.8 | 0 | 0.6815 | 0.4582 | 0.0946 | 0.3847 |
| 0.9 | 0 | 0.6902 | 0.4368 | 0.2187 | 0.3693 |
| 0.95 | 0 | 0.67 | 0.4632 | 0.3261 | 0.3747 |
| 1 | 0.9925 | 0.6767 | 0.5155 | 0.7944 | 0.4441 |
| 1.05 | 0 | 0.5319 | 0.4162 | 0.3312 | 0.3549 |
| 1.2 | 0 | 0.5184 | 0.371 | 0.1828 | 0.3231 |
| 1.5 | 0 | 0.2392 | 0.1467 | 0 | 0.215 |

### Mask IoU vs forgery size

| Size (px) | Area fraction | Block matching | SIFT + similarity verify | ORB + similarity verify | SIFT + translation verify | SIFT blobs (no verify) |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 0.0022 | 0 | 0 | 0.0004 | 0 | 0.0498 |
| 32 | 0.0039 | 0 | 0 | 0.0003 | 0 | 0.0266 |
| 48 | 0.0088 | 0.9985 | 0.2904 | 0.073 | 0.4333 | 0.2937 |
| 64 | 0.0156 | 0.9909 | 0.3708 | 0.1758 | 0.5321 | 0.2751 |
| 96 | 0.0352 | 0.9925 | 0.6767 | 0.5155 | 0.7944 | 0.4441 |
| 128 | 0.0625 | 0.9943 | 0.7559 | 0.6194 | 0.8433 | 0.5111 |
| 160 | 0.0977 | 0.975 | 0.7697 | 0.7206 | 0.8541 | 0.5549 |

### FALSE ALARMS: fraction of pixels flagged on UNTAMPERED images

| Method | Mean flagged | Worst image | Images accused |
|---|---:|---:|---:|
| Block matching | 0 | 0 | 0/6 |
| SIFT + similarity verify | 0.06322 | 0.34164 | 2/6 |
| ORB + similarity verify | 0.14949 | 0.44629 | 3/6 |
| SIFT + translation verify | 0.00817 | 0.04903 | 1/6 |
| SIFT blobs (no verify) | 0.06566 | 0.26363 | 6/6 |
| Predict nothing (control) | 0 | 0 | 0/6 |
