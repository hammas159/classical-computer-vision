### Methods on 6 scenes of 12 stops

| Method | PSNR (dB) | PSNR matched (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|---:|
| Mertens fusion | 20.922 | 23.948 | 0.8477 | 21.65 |
| Debevec + Reinhard | 14.695 | 14.828 | 0.6342 | 1838.41 |
| Debevec + Drago | 11.51 | 22.283 | 0.6873 | 6.92 |
| Debevec + Mantiuk | 9.887 | 17.625 | 0.5195 | 34.494 |
| Mean of frames (control) | 23.456 | 24.169 | 0.9282 | 9.309 |
| Middle exposure only (control) | 20.897 | 26.2 | 0.8901 | 0.002 |

### Four exposures down the rows

| Sr | Scene | Mertens fusion | Debevec + Reinhard | Debevec + Drago | Debevec + Mantiuk | Mean of frames (control) | Middle exposure only (control) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | lake and shrine · bright sky, dark foreground | SSIM 0.873 | SSIM 0.735 | SSIM 0.817 | SSIM 0.623 | SSIM 0.943 | SSIM 0.896 |
| 2 | woman and child · close faces, flat light | SSIM 0.835 | SSIM 0.562 | SSIM 0.647 | SSIM 0.464 | SSIM 0.932 | SSIM 0.897 |
| 3 | giraffe · lit animal, flat background | SSIM 0.874 | SSIM 0.676 | SSIM 0.731 | SSIM 0.586 | SSIM 0.931 | SSIM 0.887 |
| 4 | harbour and boat · sunlit town, shaded hull | SSIM 0.870 | SSIM 0.619 | SSIM 0.594 | SSIM 0.479 | SSIM 0.931 | SSIM 0.892 |

### How many frames pay

| Frames | Unrecoverable | Mertens fusion | Debevec + Reinhard | Debevec + Drago | Debevec + Mantiuk |
|---|---:|---:|---:|---:|---:|
| 2 | 0.068197 | 24.103 | 18.583 | 18.398 | 16.871 |
| 3 | 0.043937 | 22.386 | 17.476 | 14.666 | 13.674 |
| 5 | 0.005007 | 20.922 | 14.695 | 11.51 | 9.887 |

### What the bracket never recorded

| Bracket spread (± stops) | Frames | Blown everywhere | Crushed everywhere | Unrecoverable |
|---|---:|---:|---:|---:|
| 0 | 1 | 0.174925 | 0.023982 | 0.198906 |
| 1 | 3 | 0.066729 | 0.009686 | 0.076415 |
| 2 | 3 | 0.037123 | 0.006814 | 0.043937 |
| 3 | 3 | 0.016448 | 0.004444 | 0.020892 |
| 4 | 3 | 0.004623 | 0.00297 | 0.007593 |
