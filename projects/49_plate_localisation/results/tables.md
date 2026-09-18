### Three metrics, three winners

| Locator | IoU >= 0.5 | Coverage >= 0.95 | Characters recoverable | Median IoU | Median coverage |
|---|---:|---:|---:|---:|---:|
| Whole frame (control) | 0/14 | 14/14 | 0/14 | 0.01 | 1.00 |
| Fixed box (control) | 0/14 | 1/14 | 1/14 | 0.00 | 0.00 |
| Nothing (control) | 0/14 | 0/14 | 0/14 | 0.00 | 0.00 |
| Sobel-x + morphology | 8/14 | 5/14 | 8/14 | 0.53 | 0.82 |
| Top-hat + Otsu | 8/14 | 6/14 | 7/14 | 0.54 | 0.89 |
| MSER text lines | 4/14 | 1/14 | 5/14 | 0.35 | 0.46 |
| Contour + aspect | 0/14 | 0/14 | 0/14 | 0.00 | 0.00 |
| Haar plate cascade | 5/14 | 9/14 | 4/14 | 0.45 | 1.00 |
| Haar plate, 16 stages | 1/14 | 0/14 | 0/14 | 0.00 | 0.00 |

### What a pixel of localisation error costs

| Shift | IoU sideways | IoU vertically | Coverage vertically |
|---|---:|---:|---:|
| 2 px | 0.97 | 0.90 | 0.95 |
| 4 px | 0.95 | 0.81 | 0.89 |
| 8 px | 0.90 | 0.66 | 0.79 |
| 16 px | 0.82 | 0.44 | 0.57 |

### The readability proxy against the typed text

| Photograph | Plate text | Characters | Blobs found | Exact |
|---|---:|---:|---:|---:|
| eu1 | M5XSX | 5 | 5 | yes |
| eu2 | GWAGEN | 6 | 5 | no |
| eu3 | FWE50 | 5 | 5 | yes |
| eu4 | BIMMIAN | 7 | 7 | yes |
| eu5 | OYO9FEU | 7 | 7 | yes |
| eu6 | WOBVWMK4 | 8 | 8 | yes |
| eu7 | VW4X4WP | 7 | 7 | yes |
| eu8 | WSQ3021 | 7 | 2 | no |
| eu9 | W053011 | 7 | 2 | no |
| eu10 | WA56660 | 7 | 8 | no |
| eu11 | BS47040 | 7 | 7 | yes |
| AZJ6991 | AZJ6991 | 7 | 7 | yes |
| AYO9034 | AYO9034 | 7 | 7 | yes |
| FZB9581 | FZB9581 | 7 | 7 | yes |

### Per photograph, by plate size

| Photograph | Plate | Share of frame | Aspect | Found (IoU) | Found (coverage) |
|---|---:|---:|---:|---:|---:|
| eu10 | 78x18 | 0.27% | 4.3 | 3/6 | 3/6 |
| eu5 | 183x42 | 0.84% | 4.4 | 0/6 | 3/6 |
| eu7 | 343x79 | 0.86% | 4.3 | 1/6 | 1/6 |
| AYO9034 | 162x52 | 0.91% | 3.1 | 3/6 | 2/6 |
| AZJ6991 | 236x76 | 1.09% | 3.1 | 3/6 | 0/6 |
| eu3 | 91x21 | 1.11% | 4.3 | 1/6 | 1/6 |
| eu1 | 203x46 | 1.25% | 4.4 | 1/6 | 2/6 |
| FZB9581 | 258x83 | 1.31% | 3.1 | 2/6 | 0/6 |
| eu6 | 96x22 | 1.39% | 4.4 | 1/6 | 2/6 |
| eu11 | 122x28 | 1.47% | 4.4 | 1/6 | 2/6 |
| eu9 | 131x30 | 1.54% | 4.4 | 0/6 | 3/6 |
| eu8 | 303x69 | 2.67% | 4.4 | 4/6 | 0/6 |
| eu2 | 139x32 | 3.07% | 4.3 | 3/6 | 1/6 |
| eu4 | 505x116 | 18.31% | 4.4 | 3/6 | 1/6 |
