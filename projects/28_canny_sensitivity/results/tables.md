### Best settings against human boundaries

| Sigma | Low | Ratio | F | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 2 | 50 | 2 | 0.4367 | 0.3489 | 0.7048 |
| 2 | 50 | 3 | 0.4256 | 0.3978 | 0.5177 |
| 1.4 | 100 | 2 | 0.4194 | 0.3784 | 0.5335 |
| 1 | 100 | 2 | 0.41 | 0.319 | 0.7121 |
| 1.4 | 50 | 3 | 0.409 | 0.3117 | 0.7531 |
| 2 | 25 | 3 | 0.4016 | 0.2987 | 0.8236 |
| 1.4 | 100 | 3 | 0.4004 | 0.4206 | 0.4262 |
| 1 | 100 | 3 | 0.395 | 0.3292 | 0.5869 |
| 1.4 | 50 | 2 | 0.385 | 0.2814 | 0.8435 |
| 2 | 100 | 2 | 0.3695 | 0.4746 | 0.341 |

### Which parameter matters, on the synthetic scene

| Parameter | Variance explained | Best value | F1 range |
|---|---:|---:|---:|
| sigma | 0.3004 | 2 | 0.705 |
| low | 0.0212 | 50 | 0.1996 |
| ratio | 0.0003 | 2 | 0.0199 |

### Four scenes down the rows

| Sr | Scene | textbook (s1.4, 50, 3.0) | best on shapes (s1.0, 50, 3.0) | best on photos (s2.0, 50, 2.0) | no smoothing (s0, 50, 3.0) |
|---|---:|---:|---:|---:|---:|
| 1 | acacia and herd · edges 2% | 0.970 | 0.972 | 0.955 | 0.846 |
| 2 | alpine church · edges 14% | 0.507 | 0.468 | 0.531 | 0.391 |
| 3 | deer and fawn · edges 21% | 0.623 | 0.486 | 0.719 | 0.307 |
| 4 | bears on hillside · edges 37% | 0.238 | 0.146 | 0.310 | 0.135 |
