### At a 0.25 brightness offset

| Method | Seam visibility | Gradient fidelity | Pixels moved | Time (ms) |
|---|---:|---:|---:|---:|
| Copy-paste (control) | 2.8046 | 0.8266 | 0 | 0.37 |
| Alpha feather | 1.0544 | 0.9299 | 4.45 | 3.25 |
| Poisson (OpenCV) | 1.1403 | 0.7822 | 51.04 | 4.31 |
| Poisson (mixed gradients) | 0.85 | 0.4694 | 56.52 | 4.17 |
| Poisson (Jacobi, from scratch) | 1.1337 | 0.8792 | 22.62 | 162.26 |

### Against the brightness difference

| Offset | Copy-paste (control) | Alpha feather | Poisson (OpenCV) | Poisson (mixed gradients) | Poisson (Jacobi, from scratch) |
|---|---:|---:|---:|---:|---:|
| 0 | 2.5703 | 1.0353 | 1.0752 | 0.8032 | 1.0866 |
| 0.1 | 2.2686 | 1.0169 | 1.0882 | 0.8141 | 1.11 |
| 0.25 | 2.8046 | 1.0544 | 1.1403 | 0.85 | 1.1337 |
| 0.5 | 5.9604 | 1.3417 | 1.3813 | 0.9295 | 1.2501 |

### Against the region size

| Region size | Copy-paste (control) | Alpha feather | Poisson (OpenCV) | Poisson (mixed gradients) | Poisson (Jacobi, from scratch) |
|---|---:|---:|---:|---:|---:|
| 60 | 2.95 | 0.9155 | 1.0338 | 0.866 | 0.997 |
| 100 | 2.7578 | 1.0039 | 1.0847 | 0.8241 | 1.0924 |
| 150 | 2.9961 | 1.0869 | 1.2081 | 0.862 | 1.1675 |
| 200 | 3.196 | 1.0522 | 1.0864 | 0.7408 | 1.0589 |

### Jacobi convergence

| Iterations | Seam | Time (ms) |
|---|---:|---:|
| 10 | 1.2252 | 7.57 |
| 50 | 1.1443 | 23.88 |
| 100 | 1.1342 | 44.1 |
| 200 | 1.1315 | 82.69 |
| 400 | 1.1337 | 155.47 |
| 800 | 1.1393 | 317.2 |

### Four pairs down the rows

| Sr | Pair | Copy-paste (control) | Alpha feather | Poisson (OpenCV) | Poisson (mixed gradients) | Poisson (Jacobi, from scratch) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | bird in a meadow + snowshoes on snow | 1.790 | 0.760 | 0.543 | 0.533 | 0.623 |
| 2 | tent on the ice + bobcat and daisies | 1.626 | 0.700 | 0.885 | 0.823 | 0.842 |
| 3 | wallaby in scrub + spear fisher | 4.519 | 1.179 | 1.286 | 0.866 | 1.207 |
| 4 | stone viaduct + skier mid air | 4.758 | 1.148 | 1.091 | 0.836 | 1.022 |
