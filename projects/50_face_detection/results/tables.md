### The training window explains what a cascade cannot see

| Cascade | Training window | Boxes at 1x | at 1.5x | at 2x | at 3x | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| Haar default | 24x24 | 97 | 115 | 123 | 149 | 1.54x |
| Haar alt | 20x20 | 92 | 98 | 100 | 103 | 1.12x |
| Haar alt2 | 20x20 | 95 | 98 | 101 | 106 | 1.12x |
| Haar alt_tree | 20x20 | 50 | 62 | 62 | 62 | 1.24x |
| LBP frontal | 24x24 | 90 | 97 | 100 | 106 | 1.18x |
| LBP improved | 45x45 | 3 | 51 | 80 | 86 | 28.67x |

### Rotation survival, against a truth that is the rotation

| Cascade | 0° | 5° | 10° | 15° | 20° | 30° | 45° |
|---|---:|---:|---:|---:|---:|---:|---:|
| Haar default | 1.00 | 0.98 | 0.88 | 0.79 | 0.48 | 0.03 | 0.01 |
| Haar alt | 1.00 | 0.97 | 0.87 | 0.63 | 0.37 | 0.01 | 0.00 |
| Haar alt2 | 1.00 | 0.97 | 0.91 | 0.66 | 0.41 | 0.01 | 0.00 |
| Haar alt_tree | 1.00 | 0.80 | 0.44 | 0.14 | 0.00 | 0.00 | 0.00 |
| LBP frontal | 1.00 | 0.91 | 0.87 | 0.57 | 0.31 | 0.00 | 0.00 |
| LBP improved | 1.00 | 1.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |

### Both arms at minNeighbors 5

| Method | Boxes on faces | False alarms | Per megapixel |
|---|---:|---:|---:|
| Haar default | 97 | 1 | 0.6 |
| Haar alt | 92 | 0 | 0.0 |
| Haar alt2 | 95 | 0 | 0.0 |
| Haar alt_tree | 50 | 0 | 0.0 |
| LBP frontal | 90 | 0 | 0.0 |
| LBP improved | 3 | 0 | 0.0 |
| Nothing (control) | 0 | 0 | 0.0 |
| One centre box (control) | 11 | 11 | 6.5 |
| Every box (control) | 6276 | 6996 | 4119.1 |

### Agreement between the six cascades

| Photograph | Distinct boxes | Found by 3+ | Found by all 6 |
|---|---:|---:|---:|
| addams-family | 7 | 6 | 0 |
| audrybt1 | 1 | 1 | 0 |
| bttf301 | 6 | 6 | 1 |
| churchill-downs | 0 | 0 | 0 |
| class57 | 61 | 55 | 0 |
| er | 9 | 6 | 0 |
| karen-and-rob | 2 | 2 | 0 |
| larroquette | 7 | 6 | 0 |
| mona-lisa | 1 | 1 | 1 |
| rehg-thanksgiving-1994 | 7 | 6 | 0 |
| waynesworld2 | 3 | 1 | 0 |
