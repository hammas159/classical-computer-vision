### Six decisions on one perfect mask

| Rule | Alerts | Found | Missed | Latency (s) | False alerts | False per minute |
|---|---:|---:|---:|---:|---:|---:|
| Any motion in the zone | 244 | 7/7 | 0 | 0.00 | 12 | 9.1 |
| + minimum area | 229 | 7/7 | 0 | 0.00 | 0 | 0.0 |
| + persistence 3 | 213 | 7/7 | 0 | 0.20 | 0 | 0.0 |
| + persistence 8 | 173 | 7/7 | 0 | 0.70 | 0 | 0.0 |
| + cooldown 50 | 8 | 7/7 | 0 | 0.70 | 0 | 0.0 |
| + cooldown 100 | 5 | 4/7 | 3 | 0.70 | 0 | 0.0 |
| Always alert (control) | 795 | 7/7 | 0 | 0.00 | 563 | 424.9 |
| Never alert (control) | 0 | 0/7 | 7 | — | 0 | 0.0 |
| Random, rate-matched (control) | 8 | 3/7 | 4 | 1.30 | 4 | 3.0 |

### The cooldown sweep

| Cooldown (s) | Alerts | Found | Missed | Latency (s) |
|---|---:|---:|---:|---:|
| 0.0 | 173 | 7/7 | 0 | 0.70 |
| 1.0 | 20 | 7/7 | 0 | 0.70 |
| 2.0 | 13 | 7/7 | 0 | 0.70 |
| 3.0 | 8 | 7/7 | 0 | 0.70 |
| 5.0 | 8 | 7/7 | 0 | 0.70 |
| 7.5 | 6 | 5/7 | 2 | 0.70 |
| 10.0 | 5 | 4/7 | 3 | 0.70 |
| 15.0 | 4 | 3/7 | 4 | 0.70 |

### The same rules on a causal mask

| Rule | Alerts | Found | False alerts |
|---|---:|---:|---:|
| Any motion in the zone | 241 | 7/7 | 9 |
| + minimum area | 229 | 7/7 | 0 |
| + persistence 3 | 213 | 7/7 | 0 |
| + persistence 8 | 173 | 7/7 | 0 |
| + cooldown 50 | 8 | 7/7 | 0 |
| + cooldown 100 | 5 | 4/7 | 0 |

### Planting, and what was refused

| Frames | Seconds | Overlaps real activity | Planted |
|---|---:|---:|---:|
| 120-150 | 3.1 | 0 | yes |
| 230-258 | 2.9 | 0 | yes |
| 300-325 | 2.6 | 0 | yes |
| 400-430 | 3.1 | 0 | yes |
| 470-505 | 3.6 | 4 | REFUSED |
| 545-575 | 3.1 | 0 | yes |
| 650-672 | 2.3 | 0 | yes |
| 700-730 | 3.1 | 0 | yes |
