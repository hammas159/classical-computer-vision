### The premise

| Signal | Top 10% energy share |
|---|---:|
| packhorse_bridge | 0.6878 |
| black_panther | 0.7876 |
| lone_tree_on_a_hill | 0.804 |
| church_spire | 0.8315 |
| warthogs_drinking | 0.8546 |
| wallaby_and_joey | 0.8746 |
| warbler_at_the_nest | 0.8907 |
| four_children_on_a_wall | 0.9047 |
| cormorants_nesting | 0.9215 |
| woman_among_roses | 0.9232 |
| taj_mahal_reflected | 0.9468 |
| hawk_on_a_branch | 0.9894 |
| white noise | 0.4388 |

### At sigma 25

| Method | PSNR (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|
| Do nothing (control) | 20.572 | 0.4235 | 0.035 |
| Gaussian (spatial) | 26.136 | 0.6695 | 2.056 |
| Bilateral (spatial) | 27.671 | 0.732 | 4.788 |
| Non-local means (spatial) | 25.624 | 0.682 | 603.924 |
| Wavelet VisuShrink hard | 23.74 | 0.5799 | 17.111 |
| Wavelet VisuShrink soft | 22.765 | 0.5223 | 19.067 |
| Wavelet BayesShrink hard | 23.533 | 0.5413 | 17.375 |
| Wavelet BayesShrink soft | 25.575 | 0.6576 | 19.689 |

### Across the noise range

| Sigma | Noisy input | Do nothing (control) | Gaussian (spatial) | Bilateral (spatial) | Non-local means (spatial) | Wavelet VisuShrink hard | Wavelet VisuShrink soft | Wavelet BayesShrink hard | Wavelet BayesShrink soft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 34.241 | 34.241 | 33.61 | 36.611 | 33.855 | 29.351 | 26.783 | 34.109 | 34.7 |
| 10 | 28.314 | 28.314 | 30.11 | 33.15 | 30.589 | 26.889 | 24.993 | 28.942 | 30.379 |
| 20 | 22.434 | 22.434 | 27.07 | 29.068 | 26.8 | 24.464 | 23.271 | 24.616 | 26.647 |
| 35 | 17.828 | 17.828 | 24.725 | 25.549 | 24.081 | 22.712 | 22.032 | 22.102 | 24.039 |
| 50 | 15.053 | 15.053 | 23.156 | 23.239 | 22.571 | 21.622 | 21.201 | 20.789 | 22.513 |

### Noise estimation

| True sigma | Estimated | relative_error |
|---|---:|---:|
| 5 | 5.992 | 0.1984 |
| 10 | 8.772 | -0.1228 |
| 20 | 14.517 | -0.2742 |
| 35 | 22.856 | -0.347 |
| 50 | 30.64 | -0.3872 |

### Four scenes down the rows

| Sr | Scene | Gaussian (spatial) | Bilateral (spatial) | Wavelet VisuShrink soft | Wavelet BayesShrink soft |
|---|---:|---:|---:|---:|---:|
| 1 | black panther · sparsity 0.788 | 24.54 | 26.08 | 20.39 | 23.82 |
| 2 | warthogs drinking · sparsity 0.855 | 26.44 | 27.41 | 22.57 | 25.29 |
| 3 | cormorants nesting · sparsity 0.921 | 25.91 | 28.16 | 22.51 | 25.55 |
| 4 | taj mahal reflected · sparsity 0.947 | 25.52 | 28.15 | 22.19 | 25.63 |
