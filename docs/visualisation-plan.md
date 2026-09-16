# One signature visualisation per project

## Why this exists

Every shipped project ended with the same three panels: a pixel-value
histogram, a grid of raw pixel numbers, and a methods × metrics matrix. Three
panels repeated across fifty-eight projects is the same problem as fifty-eight
apps in one stock theme — the reader cannot tell from a figure which project
they are looking at, and more importantly, a generic panel cannot carry a
specific finding.

So each project gets a **signature visualisation**: the one picture that only
makes sense for *its* question, and that its finding lives inside.

Two layers, then:

| Layer | What | Applies to |
|---|---|---|
| **Comparison figure** | `Sr \| Input \| Method 1 … N`, four samples down the rows, score in every cell | all 58, at the top of the README |
| **Signature visualisation** | listed below, unique to the project | all 58 |
| **Methods × metrics matrix** | kept, because it is what lets a reader compare *across* the repo | all 58 |

The histogram and the raw-pixel grid are dropped unless a project has a
specific reason to keep them — project 15 genuinely needs a histogram, because
the thing being compared *is* where each method cuts it.

## The list

| # | Project | Methods compared | Signature visualisation |
|---:|---|---|---|
| 1 | Document scanner | Canny / Otsu / morph-gradient / saturation / Hough / minAreaRect, + 4 binarisers | Illumination sweep: binariser IoU vs light ratio, with the best-possible-global-cut oracle drawn above it |
| 2 | Portrait mode | 6 mattes, 4 apertures, 2 compositors | Halo error profile vs distance from the matte edge; bokeh kernel as a 3-D surface |
| 3 | Low-light | gamma, HE, CLAHE, SSR/MSR/MSRCR, LIME | Tone-level survival curve — how many of 256 levels remain, computed with **no image at all** |
| 4 | Dehazing | DCP, + guided refine, CLAHE, Retinex, gamma | Signed transmission error vs beta, crossing zero exactly where the gain peaks |
| 5 | Old photo restoration | Telea, Navier–Stokes, masked mean, harmonic diffusion | Damage mask as a three-colour confusion overlay: hit, false alarm, miss |
| 6 | Lane detection | colour mask + Canny + Hough + ROI | Lane offset traced over every frame — stability, not one good frame |
| 7 | Copy-move forgery | block matching, SIFT/ORB self-match, RANSAC | Match-vector field: arrows from source to paste, with a rotation-robustness curve |
| 8 | Video stabilisation | feature trajectories + trajectory smoothing | Camera path x / y / theta per frame, raw against smoothed, plus crop loss |
| 9 | Coin counting | Otsu, top-hat, distance transform, watershed, Hough | **Diameter histogram with denomination bands** — what turns counting into identifying |
| 10 | Seam carving | 4 energy functions + DP + rescale control | Every removed seam drawn on the original as a path bundle |
| 11 | HDR exposure fusion | Debevec, Mertens, Reinhard | Per-pixel map of *which bracket* supplied each pixel |
| 12 | Stereo to depth | BM vs SGBM | Cost-volume slice along one epipolar line — where the match is ambiguous |
| 13 | Denoising shootout | box, Gaussian, median, bilateral, NLM, Wiener | The **residual** each filter removed — noise, or structure? |
| 14 | Edge detectors | Roberts, Prewitt, Sobel, Scharr, LoG, Canny | Precision–recall curve per operator, each at its own best threshold |
| 15 | Thresholding family | Otsu, triangle, multi-Otsu, adaptive, Niblack, Sauvola | One histogram with all six cuts drawn on it |
| 16 | Sharpening | Laplacian (both signs), unsharp, high-boost | MTF — frequency response per kernel — plus the overshoot profile across an edge |
| 17 | Histogram equalisation | HE, AHE, CLAHE, matching, gamma | The transfer curve itself, input level to output level. The curve *is* the method |
| 18 | Optical flow | LK, pyramidal LK, Horn–Schunck, Farnebäck, DIS | Endpoint error vs true displacement — the exact pixel count where LK dies |
| 19 | Keypoint detectors | Harris, Shi-Tomasi, FAST, SIFT, ORB, AKAZE, BRISK | Repeatability vs rotation, scale and viewpoint — three curves |
| 20 | Deblurring | inverse, Wiener, Richardson–Lucy, regularised | RL iteration curve: the optimum, then the divergence past it |
| 21 | Super-resolution | nearest, bilinear, bicubic, Lanczos, back-projection | Edge profile across a step, one line per interpolator |
| 22 | Morphology | erosion to top-hat, skeletonisation, hit-or-miss | Structuring element × operation grid; skeleton with branch points marked |
| 23 | FFT filtering | ideal, Butterworth, Gaussian, notch, homomorphic | The filter mask in frequency space, plus ringing counted as error sign flips along a scanline |
| 24 | Region segmentation | watershed ± markers, region growing, mean-shift, SLIC, GrabCut | Region count vs IoU scatter — they move in opposite directions |
| 25 | Matching + RANSAC | ratio test, RANSAC, LMEDS, MAGSAC++ | Inlier/outlier match lines, and breakdown vs outlier fraction against the closed-form prediction |
| 26 | Do quality metrics agree? | MSE, PSNR, SSIM, MS-SSIM, GMSD, VIF | Rank-correlation heatmap between metrics on a PSNR-equalised set |
| 27 | JPEG from scratch | DCT, quantisation tables, zig-zag, RLE | Rate–distortion curve, plus the quantisation table as a heatmap |
| 28 | Canny sensitivity | sigma × low × ratio grid | Parameter-grid heatmap and a variance decomposition: which knob actually matters |
| 29 | Tracking | Kalman, mean-shift, CAMShift, KCF, CSRT, MOSSE | IoU over time per tracker — who drifts, and when |
| 30 | Background subtraction | frame diff, running average, MOG, MOG2, KNN | F1 per frame as the model warms up, plus the learned background plate |
| 31 | G&W 8-stage pipeline | Laplacian + Sobel + smoothing + power-law | Ablation waterfall — what each stage actually contributes |
| 32 | Hough transforms | lines, probabilistic lines, circles | The accumulator array itself, and runtime vs parameter dimensionality |
| 33 | Texture | GLCM, LBP, Gabor, Laws | Invariance matrix: descriptor × transform (rotate, scale, relight) |
| 34 | RGB to grayscale | BT.601, BT.709, linear-light, value, contrast-preserving | Colours that collapse to the same grey — the iso-luminance confusion |
| 35 | Camera calibration | reprojection error vs number of views | Distortion as a vector field, and error vs number of views |
| 36 | Shape descriptors | Hu moments, Fourier descriptors, chain codes | Descriptor value under rotation and scale — it should be a flat line; show whether it is |
| 37 | Template matching | SSD, NCC, ZNCC, multi-scale | The score surface per function — peak sharpness and false peaks |
| 38 | Panorama stitching | homography, cylindrical warp, multi-band blending | The blend seam, and the error profile crossing it |
| 39 | White balance | grey-world, white-patch, shades-of-grey, grey-edge | Chromaticity plane: estimated illuminant against true, as an angle |
| 40 | Multi-frame SR | shift-and-add, iterative back-projection | PSNR vs frame count — breaking project 21's plateau |
| 41 | Point transforms | log, power-law, piecewise-linear, bit-plane | Transfer curves, plus the bit-plane decomposition strip |
| 42 | Image registration | phase correlation, ECC, mutual information | MI surface over shift, and a checkerboard overlay of the aligned pair |
| 43 | Grayscale to colour | Levin, Welsh transfer, pseudo-colour | a–b chroma plane: predicted against true |
| 44 | Poisson blending | paste, feather, Poisson, mixed gradients | Gradient-domain residual map |
| 45 | Wavelet denoising | soft/hard, VisuShrink, BayesShrink | Coefficient magnitude distribution with the threshold drawn on it |
| 46 | Epipolar geometry | 8-point vs RANSAC | Epipolar lines drawn on both images, and Sampson error |
| 47 | Colour space robustness | RGB, HSV, Lab, YCrCb, normalised RGB | Per-channel response to a lighting change — where "HSV is robust" stops being true |
| 48 | Chroma key | colour keying, spill suppression, matting | Alpha histogram and the spill map |
| 49 | Plate localisation | edge + morphology + contour filtering | Candidate-rejection funnel: how many survive each stage |
| 50 | Face recognition | Eigenfaces, Fisherfaces, LBPH | The basis images themselves, and accuracy vs number of components |
| 51 | Demosaicing | nearest, bilinear, Malvar, VNG, edge-aware | Error vs distance from the nearest edge — what whole-image PSNR hides |
| 52 | Focus stacking | focus measures over a focal stack | Focus measure vs slice index per pixel — the peak *is* the depth |
| 53 | Barcode / QR | gradient + morphology, variance, finder patterns | Two curves diverging: still located, against still decodable |
| 54 | Defect detection | template + morphology + blob analysis | Defect size vs detectability |
| 55 | Security alert | background subtraction + blob tracking | Alert timeline against ground-truth events |
| 56 | Hand gesture | skin colour + contours + convexity defects | Convexity defects drawn on the hand contour, plus a gesture confusion matrix |
| 57 | Pedestrian detection | HOG + linear SVM | DET curve — miss rate vs false positives per image — and the HOG descriptor rendered |
| 58 | Red-eye removal | colour, + shape, + face, + eye constraints | Constraint ablation funnel: false positives falling as each constraint is added |

## Three things to be clear about

**Most of these plotters do not exist yet.** `shared/figures.py` currently has
`grid`, `lines`, `histogram`, `value_matrix`, `comparison_matrix` and
`gallery`. A cost-volume slice, an MTF curve, a DET curve, a chromaticity
scatter and a rejection funnel are new plotting code, not restyling.

**The signature visualisation is where a finding lives or dies.** For the
thirty-two projects that have never been run, nobody knows yet what the curve
will show. The visualisation gets built, run, and then reports what it actually
says — including "nothing surprising here" where that is the truth. A project
with no honest finding says so.

**The methods × metrics matrix stays.** It is the one panel that is *supposed*
to be the same everywhere, because it is what lets a reader compare project 14's
best edge detector against project 28's without relearning a chart.
