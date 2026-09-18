### Invariance on generated shapes

| Descriptor | Rotation | Scale | Translation |
|---|---:|---:|---:|
| Hu moments (log) | 0.15019 | 0.23703 | 0 |
| Hu moments (textbook log) | 5.05861 | 3.3268 | 0 |
| Fourier descriptors | 0.22276 | 1.63923 | 0 |
| Chain code histogram | 0.48128 | 0.00931 | 0 |
| Simple geometry | 0.09346 | 0.00669 | 0 |

### Invariance on human-traced silhouettes

| Descriptor | Rotation | Scale | Translation |
|---|---:|---:|---:|
| Hu moments (log) | 0.00103 | 0.00486 | 0 |
| Hu moments (textbook log) | 0.00099 | 0.00481 | 0 |
| Fourier descriptors | 0.05646 | 0.0279 | 0 |
| Chain code histogram | 0.54455 | 0.07451 | 0 |
| Simple geometry | 0.24621 | 0.01032 | 0 |

### How far apart two people are on the same object

| Descriptor | Mean change | Max | Pairs |
|---|---:|---:|---:|
| Hu moments (log) | 0.75313 | 2.21879 | 44 |
| Hu moments (textbook log) | 0.75253 | 2.20246 | 44 |
| Fourier descriptors | 0.57813 | 6.74613 | 44 |
| Chain code histogram | 0.17302 | 0.50615 | 44 |
| Simple geometry | 0.10817 | 0.31384 | 44 |

### Recognising an object from another person's tracing

| Descriptor | Upright | Turned 30 deg |
|---|---:|---:|
| Hu moments (log) | 0.5455 | 0.5227 |
| Hu moments (textbook log) | 0.5455 | 0.5227 |
| Fourier descriptors | 0.3864 | 0.3409 |
| Chain code histogram | 0.75 | 0.1136 |
| Simple geometry | 0.5 | 0.2273 |

### Reflection: only the 7th Hu moment notices

| Object | Mirror symmetry | h7 flips | Fourier change | Geometry change |
|---|---:|---:|---:|---:|
| greek_amphora | 0.9628 | True | 0 | 0 |
| collie_standing | 0.5474 | True | 0.00205 | 0 |
| flatfish_on_sand | 0.6553 | True | 0.00257 | 0 |
| man_in_a_fez | 0.8961 | True | 0.00737 | 0 |
| made_up_face | 0.5622 | True | 0.00204 | 0 |
| roman_amphitheatre | 0.5304 | True | 0.00053 | 0 |
| green_mountain_ridge | 0.7629 | True | 0.00742 | 0 |
| basket_of_grain | 0.8935 | False | 0.00264 | 0 |
| buttressed_trunk | 0.5293 | True | 0.00077 | 0 |
| brain_coral | 0.7516 | True | 0.00969 | 0 |
| lizard_on_a_leaf | 0.5205 | True | 0.00117 | 0 |
| spotted_fish_head | 0.6281 | True | 0.00328 | 0 |

### Four objects down the rows

| Sr | Object | Hu moments (log) | Hu moments (textbook log) | Fourier descriptors | Chain code histogram | Simple geometry |
|---|---:|---:|---:|---:|---:|---:|
| 1 | greek amphora | correct | correct | flatfish on sand | collie standing | man in a fez |
| 2 | collie standing | green mountain ridge | green mountain ridge | man in a fez | basket of grain | correct |
| 3 | man in a fez | spotted fish head | spotted fish head | flatfish on sand | collie standing | correct |
| 4 | made up face | greek amphora | greek amphora | correct | collie standing | man in a fez |
