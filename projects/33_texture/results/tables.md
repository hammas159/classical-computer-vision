### Clean accuracy at the operating point

| Descriptor | Accuracy | Separability | Dimensions | Time (ms) |
|---|---:|---:|---:|---:|
| GLCM (Haralick) | 0.953 ± 0.025 | 6.0572 | 72 | 1.241 |
| LBP (uniform) | 0.701 ± 0.034 | 1.6392 | 26 | 0.639 |
| Gabor bank | 0.910 ± 0.021 | 5.8339 | 48 | 2.675 |
| Laws energy | 0.951 ± 0.025 | 17.5252 | 30 | 0.994 |
| Raw histogram (control) | 0.764 ± 0.040 | 2.2561 | 32 | 0.112 |

### Accuracy under each degradation

| Descriptor | Clean | Relit | Gamma | Noise | Rotated 45 | Rotated 90 |
|---|---:|---:|---:|---:|---:|---:|
| GLCM (Haralick) | 0.953 ± 0.025 | 0.086 ± 0.006 | 0.682 ± 0.018 | 0.163 ± 0.012 | 0.499 ± 0.043 | 0.615 ± 0.018 |
| LBP (uniform) | 0.701 ± 0.034 | 0.640 ± 0.042 | 0.574 ± 0.029 | 0.086 ± 0.003 | 0.543 ± 0.037 | 0.703 ± 0.035 |
| Gabor bank | 0.910 ± 0.021 | 0.090 ± 0.014 | 0.296 ± 0.035 | 0.611 ± 0.035 | 0.553 ± 0.021 | 0.596 ± 0.051 |
| Laws energy | 0.951 ± 0.025 | 0.083 ± 0.000 | 0.676 ± 0.045 | 0.110 ± 0.034 | 0.521 ± 0.031 | 0.951 ± 0.025 |
| Raw histogram (control) | 0.764 ± 0.040 | 0.085 ± 0.012 | 0.125 ± 0.027 | 0.210 ± 0.019 | 0.728 ± 0.038 | 0.764 ± 0.040 |

### Accuracy against patch size

| Patch | GLCM (Haralick) | LBP (uniform) | Gabor bank | Laws energy | Raw histogram (control) |
|---|---:|---:|---:|---:|---:|
| 24 | 0.8972 | 0.557 | 0.8361 | 0.925 | 0.7347 |
| 32 | 0.9528 | 0.7014 | 0.9097 | 0.9514 | 0.7639 |
| 48 | 0.9958 | 0.9 | 0.9639 | 0.9917 | 0.8375 |
| 64 | 0.9972 | 0.9847 | 0.9833 | 0.9945 | 0.9 |
| 96 | 1 | 1 | 0.9972 | 1 | 0.9431 |

### Four textures down the rows

| Sr | Texture | GLCM (Haralick) | LBP (uniform) | Gabor bank | Laws energy | Raw histogram (control) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | packed cobbles | stipple plaster | tree bark ridged | thatch fibres | crushed gravel | correct |
| 2 | thatch fibres | dry straw | sand ripples | correct | crushed gravel | packed cobbles |
| 3 | coarse stucco | tree bark ridged | correct | thatch fibres | dry straw | packed cobbles |
| 4 | perforated metal | sand ripples | correct | tree bark ridged | crushed gravel | coarse stucco |
