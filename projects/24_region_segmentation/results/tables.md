### Every method: boundary F against human consensus

| Method | Boundary F | Precision | Recall | Regions | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Mean-shift | 0.4331 | 0.3728 | 0.6244 | 185 | 1041.18 |
| Region growing | 0.4051 | 0.4103 | 0.4677 | 7 | 1.87 |
| Watershed + markers | 0.3512 | 0.4181 | 0.3342 | 7 | 4.69 |
| GrabCut (seeded) | 0.3358 | 0.5291 | 0.2732 | 7 | 1169.03 |
| SLIC superpixels | 0.3115 | 0.2046 | 0.743 | 171 | 140.81 |
| Watershed (no markers) | 0.219 | 0.1258 | 0.9993 | 7126 | 6.59 |
| Grid tiles (control) | 0.1663 | 0.1225 | 0.292 | 175 | 0.99 |

### The same methods on IoU, which rewards over-segmentation

| Method | IoU (oracle labelling) | Regions | Underseg. error |
|---|---:|---:|---:|
| SLIC superpixels | 0.9295 | 171 | 0.0408 |
| Grid tiles (control) | 0.914 | 175 | 0.0499 |
| Watershed (no markers) | 0.6876 | 7126 | 0.005 |
| Region growing | 0.4081 | 7 | 0.3119 |
| Mean-shift | 0.3699 | 185 | 0.0259 |
| GrabCut (seeded) | 0.3384 | 7 | 0.0197 |
| Watershed + markers | 0.286 | 7 | 0.0071 |

### How much the humans agree with each other

| Image | Annotators | Best pair | Mean pair |
|---|---:|---:|---:|
| wolf_on_snowline | 5 | 0.9975 | 0.9556 |
| tiger_in_shade | 5 | 0.8826 | 0.8304 |
| train_on_viaduct | 5 | 0.9295 | 0.8457 |
| caterpillar_on_stem | 5 | 0.928 | 0.8681 |
| gunner_reenactor | 5 | 0.917 | 0.8744 |
| morel_mushrooms | 7 | 0.9345 | 0.8799 |
| woman_and_child | 5 | 0.9409 | 0.9046 |
| fox_cubs | 5 | 0.7902 | 0.7081 |
| memorial_arch | 6 | 0.8657 | 0.7714 |
| florence_duomo | 5 | 0.7932 | 0.7193 |
| two_beefeaters | 6 | 0.8807 | 0.772 |
| man_yellow_barrels | 7 | 0.992 | 0.9639 |

### Four scenes down the rows

| Sr | Scene | Humans (F) | Grid tiles (control) | Watershed (no markers) | Watershed + markers | Region growing | Mean-shift | SLIC superpixels | GrabCut (seeded) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wolf on snowline | 0.998 | 0.053 | 0.070 | 0.482 | 0.812 | 0.871 | 0.142 | 0.634 |
| 2 | woman and child | 0.941 | 0.197 | 0.252 | 0.441 | 0.328 | 0.513 | 0.385 | 0.376 |
| 3 | two beefeaters | 0.881 | 0.150 | 0.206 | 0.550 | 0.670 | 0.633 | 0.306 | 0.369 |
| 4 | man yellow barrels | 0.992 | 0.267 | 0.379 | 0.450 | 0.454 | 0.511 | 0.551 | 0.221 |
