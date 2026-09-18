### Every method against the empty-scene oracle

| Method | IoU | Precision | Recall | F1 | Pixel accuracy | ms |
|---|---:|---:|---:|---:|---:|---:|
| All background (control) | 0 | 0 | 0 | 0 | 0.9618 | 0.02 |
| All foreground (control) | 0.0382 | 0.0382 | 1 | 0.0732 | 0.0382 | 0.02 |
| Frame difference | 0.4368 | 0.6472 | 0.5862 | 0.5994 | 0.9702 | 4.04 |
| Running average | 0.1854 | 0.1924 | 0.8909 | 0.3117 | 0.8571 | 51.66 |
| Median background | 0.3969 | 0.4181 | 0.9086 | 0.5585 | 0.9434 | 193.49 |
| MOG2 | 0.399 | 0.929 | 0.4117 | 0.5601 | 0.9753 | 215.92 |
| KNN | 0.4359 | 0.9357 | 0.4494 | 0.5988 | 0.9768 | 261.52 |

### Sustained change against instantaneous change

| Method | Sustained IoU | Instantaneous IoU | Difference |
|---|---:|---:|---:|
| All background (control) | 0 | 0 | 0 |
| All foreground (control) | 0.0382 | 0.0319 | 0.0063 |
| Frame difference | 0.4368 | 0.3514 | 0.0854 |
| Running average | 0.1854 | 0.13 | 0.0554 |
| Median background | 0.3969 | 0.1803 | 0.2166 |
| MOG2 | 0.399 | 0.1067 | 0.2923 |
| KNN | 0.4385 | 0.0968 | 0.3417 |

### Is the oracle finding people?

| Frame | Oracle blobs | HOG detections | Blobs confirmed by HOG | HOG found by oracle |
|---|---:|---:|---:|---:|
| 80 | 4 | 2 | 2 | 2 |
| 140 | 17 | 2 | 2 | 2 |
| 200 | 6 | 2 | 2 | 2 |
| 260 | 7 | 3 | 3 | 3 |
| 320 | 4 | 3 | 2 | 3 |
| 380 | 5 | 2 | 1 | 2 |
| 440 | 3 | 3 | 3 | 3 |
| 500 | 4 | 3 | 1 | 1 |
| 560 | 5 | 3 | 3 | 3 |
| 620 | 6 | 3 | 2 | 3 |
| 690 | 5 | 3 | 3 | 3 |
| 760 | 6 | 4 | 4 | 4 |

### How much of each pasted rectangle actually changed

| Frame | Donor | Rectangle px | Changed px | Changed share |
|---|---:|---:|---:|---:|
| 80 | 300 | 33000 | 8402 | 0.2546 |
| 140 | 420 | 26000 | 8121 | 0.3123 |
| 200 | 520 | 38400 | 17051 | 0.444 |
| 260 | 640 | 34500 | 10510 | 0.3046 |
| 320 | 120 | 36000 | 9108 | 0.253 |
| 380 | 700 | 35000 | 12301 | 0.3515 |
| 440 | 180 | 35700 | 20541 | 0.5754 |
| 500 | 260 | 38400 | 11531 | 0.3003 |
| 560 | 340 | 33000 | 17321 | 0.5249 |
| 620 | 60 | 36000 | 20476 | 0.5688 |
| 690 | 480 | 37500 | 16972 | 0.4526 |
| 760 | 560 | 39100 | 17126 | 0.438 |

### Interior against boundary

| Method | Interior recall | Edge recall | Edge − interior |
|---|---:|---:|---:|
| Frame difference | 0.9985 | 0.582 | -0.4166 |
| Running average | 0.9633 | 0.6207 | -0.3426 |
| Median background | 0.8926 | 0.4265 | -0.4661 |
| MOG2 | 0.3741 | 0.0712 | -0.3029 |
| KNN | 0.359 | 0.0616 | -0.2974 |

### The median background with more history

| History | Precision | Recall | IoU |
|---|---:|---:|---:|
| 5 | 0.5584 | 0.7397 | 0.4525 |
| 15 | 0.4667 | 0.7794 | 0.4041 |
| 30 | 0.4128 | 0.8065 | 0.3623 |
| 60 | 0.3539 | 0.8535 | 0.3208 |
| 120 | 0.3173 | 0.9519 | 0.3117 |
| 240 | 0.2533 | 0.9786 | 0.2521 |

### What the shadow flag costs

| Method | Mask share, shadows excluded | Mask share, shadows included | IoU, shadows excluded | IoU, shadows included |
|---|---:|---:|---:|---:|
| MOG2 | 0.01712 | 0.02876 | 0.0747 | 0.1082 |
| KNN | 0.01799 | 0.02513 | 0.0691 | 0.0785 |

### Four frames down the rows

| Sr | Frame | Foreground % | Frame difference | Running average | Median background | MOG2 | KNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 140 | 8.0 | 0.237 | 0.147 | 0.225 | 0.186 | 0.201 |
| 2 | 560 | 4.5 | 0.287 | 0.207 | 0.484 | 0.251 | 0.316 |
| 3 | 620 | 5.2 | 0.417 | 0.195 | 0.528 | 0.464 | 0.499 |
| 4 | 690 | 3.9 | 0.524 | 0.155 | 0.426 | 0.501 | 0.544 |
