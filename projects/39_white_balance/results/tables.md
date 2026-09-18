### Under a tungsten cast

| Method | Angular error (deg) | PSNR (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|---:|
| Do nothing (control) | 15.949 | 18.863 | 0.9203 | 0.005 |
| Grey-world | 6.977 | 23.939 | 0.9403 | 2.525 |
| White-patch (99th pct) | 6.212 | 26.895 | 0.9699 | 4.165 |
| White-patch (true max) | 7.884 | 25.358 | 0.9709 | 2.734 |
| Shades-of-grey (p=6) | 4.426 | 27.588 | 0.9707 | 12.908 |
| Grey-edge (p=6) | 5.062 | 28.539 | 0.9764 | 21.455 |

### Four casts

| Cast | Do nothing (control) | Grey-world | White-patch (99th pct) | White-patch (true max) | Shades-of-grey (p=6) | Grey-edge (p=6) |
|---|---:|---:|---:|---:|---:|---:|
| tungsten (warm) | 15.949 | 6.977 | 6.212 | 7.884 | 4.426 | 5.062 |
| daylight (neutral) | 0.936 | 8.973 | 4.703 | 1.343 | 4.956 | 2.712 |
| shade (cool) | 12.471 | 10.366 | 7.58 | 7.03 | 6.272 | 4.614 |
| strong green | 14.771 | 7.543 | 7.308 | 7.616 | 5.137 | 5.137 |

### Each method's designed failure

| Scene | Do nothing (control) | Grey-world | White-patch (99th pct) | White-patch (true max) | Shades-of-grey (p=6) | Grey-edge (p=6) |
|---|---:|---:|---:|---:|---:|---:|
| Dominant colour, neutral light (breaks grey-world) | 0 | 23.034 | 4.349 | 2.649 | 11.18 | 3.201 |
| Clipped highlight, warm light (breaks white-patch) | 15.949 | 6.119 | 7.971 | 15.949 | 5.607 | 5.342 |

### Grey-world against how much of the frame is one colour

| Dominant fraction | Do nothing (control) | Grey-world | White-patch (99th pct) | White-patch (true max) | Shades-of-grey (p=6) | Grey-edge (p=6) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.169 | 6.038 | 2.934 | 2.942 | 6.144 |
| 0.2 | 0 | 7.356 | 6.038 | 2.934 | 6.588 | 4.259 |
| 0.4 | 0 | 13.936 | 6.038 | 2.934 | 8.989 | 4.724 |
| 0.6 | 0 | 20.281 | 6.038 | 2.934 | 10.791 | 5.012 |
| 0.8 | 0 | 26.211 | 6.038 | 2.934 | 12.273 | 4.512 |

### The Minkowski exponent

| p | Shades-of-grey | Grey-edge |
|---|---:|---:|
| 1 | 6.975 | 3.929 |
| 2 | 5.669 | 3.541 |
| 4 | 4.665 | 4.194 |
| 6 | 4.426 | 5.062 |
| 10 | 4.371 | 6.233 |
| 20 | 4.897 | 7.242 |
| 50 | 6.132 | 7.795 |

### Noise

| Sigma | Do nothing (control) | Grey-world | White-patch (99th pct) | White-patch (true max) | Shades-of-grey (p=6) | Grey-edge (p=6) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 15.949 | 6.977 | 6.212 | 7.884 | 4.426 | 5.062 |
| 5 | 15.949 | 6.964 | 6.347 | 8.379 | 4.462 | 5.184 |
| 15 | 15.949 | 6.816 | 6.816 | 10.612 | 4.49 | 5.317 |
| 30 | 15.949 | 6.375 | 8.024 | 14.734 | 4.738 | 5.805 |

### Four scenes down the rows

| Sr | Scene | Do nothing (control) | Grey-world | White-patch (99th pct) | White-patch (true max) | Shades-of-grey (p=6) | Grey-edge (p=6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | beach baseball · 3.1 deg from grey | 15.95 | 1.99 | 4.80 | 8.58 | 2.01 | 2.92 |
| 2 | desert dune ripples · 4.1 deg from grey | 15.95 | 2.98 | 1.39 | 2.37 | 2.36 | 5.14 |
| 3 | leopard in dry grass · 10.8 deg from grey | 15.95 | 7.31 | 5.89 | 7.90 | 6.23 | 1.49 |
| 4 | red chrysanthemums · 29.2 deg from grey | 15.95 | 21.68 | 8.48 | 7.92 | 9.33 | 5.85 |
