### Rate-distortion, against OpenCV's libjpeg

| Quality | bpp (ours) | PSNR (ours) | SSIM (ours) | Blockiness | bpp (OpenCV) | PSNR (OpenCV) |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 0.2266 | 22.839 | 0.5682 | 4.944 | 0.26 | 23.194 |
| 10 | 0.3373 | 25.133 | 0.685 | 3.009 | 0.3852 | 25.67 |
| 20 | 0.5001 | 26.932 | 0.7818 | 2.125 | 0.6087 | 27.828 |
| 30 | 0.6264 | 27.884 | 0.8239 | 1.84 | 0.7975 | 29.011 |
| 50 | 0.8202 | 28.964 | 0.8633 | 1.611 | 1.0953 | 30.464 |
| 70 | 1.0989 | 30.057 | 0.8991 | 1.298 | 1.5561 | 32.083 |
| 85 | 1.798 | 31.327 | 0.9468 | 0.925 | 2.7675 | 34.312 |
| 95 | 2.3697 | 33.901 | 0.9881 | 1.028 | 4.1424 | 43.542 |

### Each stage turned off

| Configuration | PSNR (dB) | SSIM | bpp | Blockiness |
|---|---:|---:|---:|---:|
| Full codec | 28.964 | 0.8633 | 0.8202 | 1.611 |
| No chroma subsampling | 28.994 | 0.869 | 1.0728 | 1.532 |
| No colour transform (RGB) | 26.82 | 0.7516 | 1.0952 | 1.327 |
| No quantisation | 34.943 | 0.9942 | 2.9104 | 1.064 |
| No DCT (quantise pixels) | 19.632 | 0.5066 | 1.8896 | 1.313 |
| No DCT, no quantisation | 46.791 | 0.9969 | 6.1342 | 0.952 |

### Which coefficients survive quantisation

| Quality | Non-zero coefficients | DC survives | Top frequency survives |
|---|---:|---:|---:|
| 10 | 0.0524 | 0.9113 | 0 |
| 30 | 0.1286 | 0.9715 | 0 |
| 50 | 0.1798 | 0.9856 | 0 |
| 75 | 0.2641 | 0.9902 | 0.0005 |
| 95 | 0.5309 | 0.9983 | 0.0767 |

### Four scenes down the rows

| Sr | Scene | Q5 | Q20 | Q50 | Q95 |
|---|---:|---:|---:|---:|---:|
| 1 | sprinter start · texture 55 | 24.8 dB | 30.2 dB | 32.6 dB | 37.0 dB |
| 2 | borobudur stupas · texture 69 | 21.3 dB | 25.1 dB | 27.2 dB | 32.2 dB |
| 3 | mare foal meadow two · texture 75 | 21.0 dB | 24.8 dB | 27.2 dB | 33.2 dB |
| 4 | tulip beds · texture 78 | 21.7 dB | 25.7 dB | 28.0 dB | 35.5 dB |
