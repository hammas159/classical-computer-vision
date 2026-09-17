### Descriptors

| Descriptor | Matches | Inlier precision | Reprojection (px) | Time (ms) |
|---|---:|---:|---:|---:|
| SIFT | 596 | 0.9854 | 0.222 | 27.31 |
| ORB | 969 | 0.9633 | 0.927 | 5.71 |
| AKAZE | 369 | 0.9823 | 0.363 | 15.42 |

### Match filters

| Filter | Matches kept | Inlier precision | Inlier recall |
|---|---:|---:|---:|
| All nearest neighbours | 989 | 0.6347 | 1 |
| Cross-check | 664 | 0.9113 | 0.9774 |
| Ratio test 0.75 | 596 | 0.9854 | 0.9486 |

### The ratio test swept

| Ratio | Matches kept | Inlier precision |
|---|---:|---:|
| 0.5 | 515 | 0.999 |
| 0.6 | 558 | 0.9969 |
| 0.7 | 584 | 0.9917 |
| 0.75 | 596 | 0.9854 |
| 0.8 | 610 | 0.9719 |
| 0.9 | 679 | 0.8921 |
| 1 | 989 | 0.6347 |

### Estimators against the outlier fraction

| Outliers | Iterations | Least squares (control) | RANSAC | LMEDS | MAGSAC++ |
|---|---:|---:|---:|---:|---:|
| 0 | 0.2 | 1.53 | 0.222 | 0.22 | 0.223 |
| 0.2 | 8.7 | 26.471 | 0.224 | 0.222 | 0.227 |
| 0.4 | 33.2 | 88.005 | 0.224 | 0.222 | 0.227 |
| 0.5 | 71.4 | 122.583 | 0.219 | 2.004 | 0.225 |
| 0.6 | 177.6 | 194.47 | 0.218 | 190.862 | 0.224 |
| 0.7 | 566.2 | 186.002 | 0.23 | 224.154 | 0.242 |
| 0.8 | 2875.9 | 256.445 | 0.229 | 365.15 | 0.245 |
| 0.9 | 46049.4 | 385.639 | 13.362 | 341.584 | 98.458 |

### Four scenes down the rows

| Sr | Scene | Least squares (control) | RANSAC | LMEDS | MAGSAC++ |
|---|---:|---:|---:|---:|---:|
| 1 | potted bonsai · 6.94 bits | 128.74 px | 0.16 px | 0.86 px | 0.17 px |
| 2 | roadrunner rocks · 7.16 bits | 122.66 px | 0.15 px | 0.60 px | 0.16 px |
| 3 | leopard in tree · 7.51 bits | 297.49 px | 0.22 px | 1.30 px | 0.23 px |
| 4 | sparkler family · 7.68 bits | 100.61 px | 0.30 px | 4.34 px | 0.32 px |
