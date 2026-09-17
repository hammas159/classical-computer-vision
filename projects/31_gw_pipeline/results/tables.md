### Each stage removed in turn

| Configuration | Acutance | Dark detail | RMS contrast | SSIM |
|---|---:|---:|---:|---:|
| Full pipeline | 0.34271 | 0.50856 | 0.1731 | 0.7716 |
| Without c_sharpened | 0.27787 | 0.35361 | 0.1664 | 0.78 |
| Without e_smoothed_sobel | 0.3451 | 0.51886 | 0.1738 | 0.754 |
| Without f_mask | 0.6194 | 1.11516 | 0.2137 | 0.4382 |
| Without g_sum | 0.25415 | 0.35753 | 0.1522 | 0.8005 |
| Without h_power_law | 0.44245 | 0.56758 | 0.2121 | 0.9304 |

### The pipeline against one-line alternatives

| Method | Acutance | Dark detail | RMS contrast | SSIM |
|---|---:|---:|---:|---:|
| G&W 8-stage pipeline | 0.34271 | 0.50856 | 0.1731 | 0.7716 |
| CLAHE only | 0.56977 | 0.7489 | 0.2303 | 0.7471 |
| Unsharp mask only | 0.50281 | 0.78757 | 0.203 | 0.9068 |
| Gamma 0.5 only | 0.25415 | 0.35753 | 0.1522 | 0.8005 |
| Original (control) | 0.30058 | 0.34799 | 0.1741 | 1 |

### The Laplacian sign trap, measured

| Configuration | Acutance | SSIM |
|---|---:|---:|
| Original (no sharpening) | 0.30058 | 1 |
| Correct sign (add +8 centre) | 1.06663 | 0.4568 |
| Wrong sign (add -8 centre) | 0.91565 | -0.0548 |

### Four scenes down the rows

| Sr | Scene | G&W 8-stage pipeline | CLAHE only | Unsharp mask only | Gamma 0.5 only |
|---|---:|---:|---:|---:|---:|
| 1 | red canoes · brightness 65 | 0.242 | 0.776 | 0.935 | 0.192 |
| 2 | model red black · brightness 80 | 0.671 | 0.817 | 1.122 | 0.460 |
| 3 | glass tower tulips · brightness 96 | 0.526 | 0.847 | 0.929 | 0.347 |
| 4 | geisha street · brightness 107 | 0.796 | 0.921 | 1.191 | 0.578 |
