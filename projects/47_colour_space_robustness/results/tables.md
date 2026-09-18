### On human-traced regions in photographs

| Space | Tolerance | Undegraded | Brightness x0.6 | Brightness x1.3 | Warm cast | Cool cast | Gamma 2.0 | Worst case | Mean loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RGB | 60 | 0.6463 | 0.219 | 0.3587 | 0.4476 | 0.5389 | 0.2432 | 0.219 | 0.2848 |
| HSV | 60 | 0.4872 | 0.4873 | 0.493 | 0.3088 | 0.5017 | 0.2915 | 0.2915 | 0.0707 |
| Lab | 25 | 0.7839 | 0.6781 | 0.6257 | 0.579 | 0.5704 | 0.643 | 0.5704 | 0.1647 |
| YCrCb | 25 | 0.7901 | 0.5299 | 0.5937 | 0.4861 | 0.4528 | 0.5834 | 0.4528 | 0.2609 |
| Normalised RGB | 25 | 0.5952 | 0.5949 | 0.5778 | 0.3689 | 0.34 | 0.2017 | 0.2017 | 0.1785 |

### Choosing each space's tolerance

| Tolerance | RGB | HSV | Lab | YCrCb | Normalised RGB |
|---|---:|---:|---:|---:|---:|
| 15 | 0.1962 | 0.3375 | 0.7613 | 0.7479 | 0.4991 |
| 25 | 0.3781 | 0.4431 | 0.7839 | 0.7901 | 0.5952 |
| 35 | 0.5145 | 0.4435 | 0.653 | 0.6873 | 0.5839 |
| 45 | 0.6039 | 0.4572 | 0.5035 | 0.5814 | 0.5232 |
| 60 | 0.6463 | 0.4872 | 0.2352 | 0.3391 | 0.4873 |
| 80 | 0.6131 | 0.4234 | 0.1781 | 0.205 | 0.4124 |

### On the synthetic colour chart

| space | Brightness scale worst | Brightness scale mean | Colour cast worst | Colour cast mean | Gamma worst | Gamma mean |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 0 | 0.2 | 0 | 0.4 | 0 | 0.2 |
| HSV | 0.8727 | 0.8727 | 0 | 0.4691 | 0 | 0.3331 |
| Lab | 0 | 0.7722 | 0 | 0.6785 | 0.8202 | 0.9353 |
| YCrCb | 0 | 0.5634 | 0 | 0.5262 | 0.8755 | 0.9334 |
| Normalised RGB | 0.9995 | 0.9995 | 0 | 0.5849 | 0 | 0.1999 |

### White balance as the actual fix

| cast_level | RGB raw | RGB balanced | HSV raw | HSV balanced | Lab raw | Lab balanced | YCrCb raw | YCrCb balanced | Normalised RGB raw | Normalised RGB balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.3333 | 0.8727 | 0.5962 | 0.9705 | 1 | 0.9329 | 0.6667 | 0.9995 | 0.6667 |
| 0.1 | 1 | 0.6667 | 0.7375 | 0.9296 | 0.82 | 1 | 0.8766 | 0.6667 | 0.9995 | 0.6667 |
| 0.2 | 0 | 0.6667 | 0.7352 | 0.9744 | 0.82 | 1 | 0.8213 | 0.6667 | 0.9254 | 0.6667 |
| 0.35 | 0 | 0.3333 | 0 | 0.6662 | 0.7822 | 1 | 0 | 0.6667 | 0 | 0.6667 |
| 0.5 | 0 | 0 | 0 | 0.6411 | 0 | 0.6667 | 0 | 0.6667 | 0 | 0.6662 |

### Four scenes down the rows

| Sr | Scene | RGB | HSV | Lab | YCrCb | Normalised RGB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | anteater at sunset · chroma 59 | 0.695 | 0.119 | 0.668 | 0.751 | 0.151 |
| 2 | kabuki pair · chroma 55 | 0.739 | 0.228 | 0.719 | 0.813 | 0.447 |
| 3 | red sports car · chroma 46 | 0.132 | 0.288 | 0.576 | 0.359 | 0.229 |
| 4 | green field worker · chroma 40 | 0.297 | 0.565 | 0.596 | 0.519 | 0.355 |
