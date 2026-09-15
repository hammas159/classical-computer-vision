# Classical Computer Vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV 4.14](https://img.shields.io/badge/OpenCV-4.14-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![No deep learning](https://img.shields.io/badge/deep%20learning-none-success)](#the-rules)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-lightgrey)](#the-rules)
[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen)](#running-the-tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Measured comparisons of classical computer vision algorithms.** Every project
takes one problem, runs 3–6 classical methods against it, and reports a
comparative table, a side-by-side figure, pixel-level metrics, wall-clock timing
and **one stated finding with a number in it**.

No neural networks. No training. No GPU. No dataset downloads — ground truth is
generated, so it is exact rather than annotated.

**Jump to:** [Why](#why-this-exists) · [The rules](#the-rules) ·
[Projects](#projects) · [Findings so far](#findings-so-far) ·
[Quick start](#quick-start) · [The shared layer](#the-shared-layer) ·
[Tests](#running-the-tests)

---

## Why this exists

Classical computer vision is usually taught as a list of function calls: *here is
Canny, here is Otsu, here is watershed.* What is almost never given is the
number — how much better, how much slower, and **at what point does the standard
advice stop being true?**

Every project here exists to answer a question of that shape, and several of the
answers contradict what the textbooks say.

---

## The rules

Applied to every project, without exception:

1. **No deep learning, no training, no fine-tuning.** A small number of projects
   use pre-trained but *non-deep* components that ship inside OpenCV — a Haar
   cascade, HOG+SVM, LBPH. They are trained by someone else, never by us, and
   each project says so in its own README rather than letting a reader discover it.
2. **One dataset · 3–6 methods · a comparative table · a side-by-side figure ·
   pixel metrics · one finding with a number.**
3. **Always report wall-clock time.** The whole argument for classical methods is
   "no training, no GPU, milliseconds". A comparison without a time column throws
   away the main result.
4. **Never write a number the code did not produce.** Every figure in every README
   comes out of that project's `run.py` and is mirrored in `results/results.json`.
5. **Ground truth is generated, not annotated.** Darken an image and you know the
   true brightness; paste a region and you know exactly which pixels were forged.

---

## Projects

**Status is stated honestly, because "it runs" and "it was measured" are different
claims:**

| | Meaning |
|:--:|---|
| ✅ | **Shipped.** Measured, figures generated, UI built and screenshotted, tests passing, findings written from real output |
| 🟡 | **Code written, executes, not yet measured.** Runs without error on a smoke test, but no figures, no UI, no verified numbers |
| ⚪ | **Code written, not yet executed.** Imports cleanly; nothing beyond that is claimed |

| # | Project | Methods compared | Headline finding | Status |
|---:|---|---|---|:--:|
| [01](projects/01_document_scanner/) | **Document scanner** | 6 page detectors · 4 binarisers · 2 aspect estimators | Otsu's failure on shadowed pages is **not** because a global threshold is impossible — the best global cut scores **0.964** where Otsu scores **0.430** | ✅ |
| [02](projects/02_portrait_mode/) | **Portrait mode** | 6 matting methods · 4 aperture shapes · 2 compositors | Ranking by IoU crowns a method that recovers **6.2%** of hair; ranking by boundary F1 crowns one that recovers **52.9%**. Also: `grabCut` is **non-deterministic** — 24 seeds on one image span IoU **0.15–0.90** | ✅ |
| [03](projects/03_low_light_enhancement/) | **Low-light enhancement** | 8 methods: fixed/auto gamma, HE, CLAHE, SSR/MSR/MSRCR, LIME | The ceiling is **not** the algorithms: at gamma 3 only **158 of 256** tone levels survive, so even an exact inverse reaches **22.31 dB**. And an adaptive method beats a fixed constant by **+5.46 dB** where its assumption holds, loses by **−7.70 dB** where it does not — averaging to a number that describes neither | ✅ |
| [04](projects/04_dehazing/) | Dehazing | dark channel prior, guided refine, CLAHE, Retinex | scored on **transmission recovery**, not just output contrast | 🟡 |
| [05](projects/05_old_photo_restoration/) | Old photo restoration | Telea, Navier–Stokes, median, harmonic diffusion | swept over **damage width** — the axis that separates the methods | 🟡 |
| [07](projects/07_copy_move_forgery/) | Copy-move forgery | block matching, SIFT, ORB self-match | swept over rotation **and** scale | 🟡 |
| [09](projects/09_coin_counting/) | Coin counting & measurement | Otsu, watershed, Hough circles, adaptive | counting is objectively right or wrong; then calibrated to mm | 🟡 |
| [10](projects/10_seam_carving/) | Seam carving | 4 energy functions + plain rescale control | object preservation and line-bend distortion, both scoreable | 🟡 |
| [13](projects/13_denoising_shootout/) | Denoising shootout | box, Gaussian, median, bilateral, NLM, Wiener | 6 filters × 3 noise types, **each tuned on its own grid** | 🟡 |
| [14](projects/14_edge_detectors/) | Edge detectors | Roberts, Prewitt, Sobel, Scharr, LoG, Canny | every operator at its **own** best threshold | 🟡 |
| [15](projects/15_thresholding_family/) | Thresholding family | Otsu, triangle, multi-Otsu, adaptive, Niblack, Sauvola | illumination, class imbalance and noise varied **separately** | 🟡 |
| [16](projects/16_sharpening/) | Sharpening | Laplacian (both signs), unsharp, high-boost | tests "sharpening adds contrast, not information" | 🟡 |
| [17](projects/17_histogram_equalization/) | Histogram equalisation | HE, AHE, CLAHE, matching, gamma | full-reference vs no-reference metrics disagreeing | ⚪ |
| [18](projects/18_optical_flow/) | Optical flow | LK, pyramidal LK, Horn–Schunck, Farnebäck, DIS | quantifies "LK fails past 1–2 px" | ⚪ |
| [19](projects/19_keypoint_detectors/) | Keypoint detectors | Harris, Shi-Tomasi, FAST, SIFT, ORB, AKAZE, BRISK | repeatability under **known** homographies | ⚪ |
| [20](projects/20_deblurring/) | Deblurring | inverse, Wiener, Richardson–Lucy, regularised | the Richardson–Lucy **iteration optimum** | ⚪ |
| [21](projects/21_super_resolution/) | Single-image super-resolution | nearest, bilinear, bicubic, Lanczos, back-projection | the interpolation **plateau** | ⚪ |
| [22](projects/22_morphology/) | Morphology | erosion…top-hat, skeletons, hit-or-miss | structuring element vs operation | ⚪ |
| [23](projects/23_fft_filtering/) | FFT filtering | ideal, Butterworth, Gaussian, notch, homomorphic | ringing measured via error sign changes | ⚪ |
| [24](projects/24_region_segmentation/) | Region segmentation | watershed ±markers, region growing, mean-shift, SLIC, GrabCut | region count and accuracy move **opposite** ways | ⚪ |
| [25](projects/25_matching_ransac/) | Matching + RANSAC | ratio test, RANSAC, LMEDS, MAGSAC++ | measured breakdown vs the closed-form prediction | ⚪ |
| [26](projects/26_quality_metrics/) | **Do quality metrics agree?** | MSE, PSNR, SSIM, MS-SSIM, GMSD, VIF | equalise PSNR, then ask the other metrics | ⚪ |
| [27](projects/27_jpeg_from_scratch/) | JPEG from scratch | DCT, quantisation tables, zig-zag, RLE | the rate–distortion curve **is** the result | ⚪ |
| [28](projects/28_canny_sensitivity/) | Canny parameter sensitivity | σ × low × ratio grid | variance decomposition: which knob matters | ⚪ |
| [31](projects/31_gw_pipeline/) | G&W 8-stage pipeline | Laplacian + Sobel + smoothing + power-law | ablation: which stages earn their place | ⚪ |
| [32](projects/32_hough_transforms/) | Hough transforms | lines, probabilistic lines, circles | cost scales with parameter count | ⚪ |
| [33](projects/33_texture/) | Texture | GLCM, LBP, Gabor, Laws | which **invariance** each one actually has | ⚪ |
| [34](projects/34_rgb_to_grayscale/) | RGB → grayscale | BT.601, BT.709, linear-light, value, contrast-preserving | when the one-line choice actually matters | ⚪ |
| [36](projects/36_shape_descriptors/) | Shape descriptors | Hu moments, Fourier descriptors, chain codes | invariance claims verified **directly** | ⚪ |
| [37](projects/37_template_matching/) | Template matching | SSD, NCC, ZNCC, multi-scale | what each scoring function is blind to | ⚪ |
| [39](projects/39_white_balance/) | White balance | grey-world, white-patch, shades-of-grey, grey-edge | **angular error** in degrees, the standard metric | ⚪ |
| [40](projects/40_multiframe_super_resolution/) | Multi-frame super-resolution | shift-and-add, iterative back-projection | breaks project 21's plateau — the honest contrast | ⚪ |
| [41](projects/41_point_transforms/) | Point transforms | log, power-law, piecewise-linear, bit-plane | LUT vs arithmetic must be **bit-identical** | ⚪ |
| [42](projects/42_image_registration/) | Image registration | phase correlation, ECC, mutual information | only MI survives a modality change | ⚪ |
| [44](projects/44_poisson_blending/) | Poisson blending | copy-paste, feather, Poisson, mixed gradients | the metric choice **inverts** the conclusion | ⚪ |
| [45](projects/45_wavelet_denoising/) | Wavelet denoising | soft/hard, VisuShrink, BayesShrink | the sparsity premise, tested directly | ⚪ |
| [47](projects/47_colour_space_robustness/) | Colour space robustness | RGB, HSV, Lab, YCrCb, normalised RGB | "HSV is lighting robust" is half true | ⚪ |
| [51](projects/51_demosaicing/) | Demosaicing / camera ISP | nearest, bilinear, Malvar, VNG, edge-aware | error concentrates on **edges**, hidden by whole-image PSNR | ⚪ |
| [53](projects/53_barcode_qr/) | Barcode / QR detection | gradient+morphology, variance, QR finder | localisation and **decoding** degrade at different rates | ⚪ |
| [57](projects/57_pedestrian_detection/) | Pedestrian detection | HOG + linear SVM | the pyramid step matters more than the SVM threshold | ⚪ |
| [58](projects/58_red_eye_removal/) | Red-eye removal | colour, +shape, +face, +eye constraints | geometry is what removes the false positives | ⚪ |

**41 projects spanning 14 algorithm families, none of which need a download.**

🚨 **Do not cite a number from a 🟡 or ⚪ project.** Those modules have not been
run end to end, so any figure they would produce is unverified. Only ✅ rows have
numbers that came out of an actual execution, and those numbers are reproducible
with `python run.py` inside the project.

---

## Findings so far

### 01 · Document scanner

* **The tutorial method is not the best one.** Canny + contour — the approach in
  essentially every "build a document scanner" article — lands at **2.70 px**
  mean corner error. Plain Otsu on brightness gets **1.11 px** and is 24% faster.
  Thresholding HSV *saturation* does best at **1.02 px**, because saturation is
  `(max−min)/max` and is therefore invariant to the lighting gradient.
* **A mean can hide a catastrophe.** Hough-line fitting has a *median* error of
  1.62 px — better than Canny — and a *mean* of 46.6 px, because it locks onto a
  desk edge in 37% of scenes. It returned a quadrilateral 100% of the time and was
  right 63% of the time.
* **The textbook explanation for Otsu's failure is wrong at realistic ratios.**
  An oracle — the best global threshold by exhaustive search — scores **0.964**
  where Otsu scores **0.430**. The separation was available; Otsu's between-class
  variance criterion picked the wrong cut, because the shadow splits the *paper*
  more strongly than paper splits from ink.
* **Page aspect ratio is recoverable to 0.07%** in closed form from four corners,
  against **8.01%** (worst case 22.68%) for the edge-length heuristic everyone uses.

---

## Quick start

```bash
git clone https://github.com/hammas159/classical-computer-vision.git
cd classical-computer-vision

python -m venv .venv && .venv/Scripts/activate       # Windows
# python3 -m venv .venv && source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
```

Then run any project:

```bash
cd projects/01_document_scanner
python run.py                 # reproduces every number and figure
streamlit run ui/app.py       # interactive demo, upload your own image
```

Total install is about **60 MB** — OpenCV, scikit-image and matplotlib. There are
no model weights and no datasets to fetch.

> **OpenCV is pinned below 5.0 on purpose.** OpenCV 5 removed the bundled Haar
> cascade XML files from `cv2/data/`, which projects 02 and 57 load at runtime.
> See the comment in [`pyproject.toml`](pyproject.toml).

---

## The shared layer

Written once, imported by every project. This is what makes 41 projects tractable
rather than 41 copies of the same boilerplate.

| Module | What it provides |
|---|---|
| [`shared/io.py`](shared/io.py) | loading, saving, dtype conversion, and **one BGR/RGB convention** enforced everywhere |
| [`shared/synth.py`](shared/synth.py) | every ground-truth generator: noise, blur kernels, haze, low light, known homographies and flow fields, copy-move forgery, damage masks, and a camera-accurate document scene |
| [`shared/metrics.py`](shared/metrics.py) | PSNR, SSIM, IoU, Dice, edge P/R/F1 **with a pixel tolerance**, Pratt's FOM, endpoint error, repeatability, reprojection error |
| [`shared/figures.py`](shared/figures.py) | comparison grids, before/after pairs, error heatmaps, sweep line plots, **pixel-value distributions, confusion matrices, numeric pixel matrices, and methods-x-metrics comparison matrices** |
| [`shared/ui.py`](shared/ui.py) | the same distributions and matrices as **live** components for the apps — returns matplotlib figures and pandas Stylers, and deliberately does not import Streamlit so the shared layer stays testable headless |
| [`shared/bench.py`](shared/bench.py) | timing harness — warm-up discarded, median of N runs |
| [`shared/report.py`](shared/report.py) | markdown tables, `results.json` with version provenance, a UTF-8-safe console |
| [`tools/screenshot.py`](tools/screenshot.py) | headless screenshots of the Streamlit apps, driven over the DevTools Protocol |

### Every project shows its results four ways

A table alone hides mechanism, so each project also renders:

* **a distribution** — the pixel populations a method actually has to separate,
  with the threshold it chose drawn on top;
* **a pixel matrix** — a small patch of the image printed as raw numbers,
  because at some point the argument *is* the numbers;
* **a confusion matrix** — counts and per-class recall, so a high headline
  accuracy built on a majority class is visible rather than implied;
* **a comparison matrix** — every method against every metric, each column
  scaled on its own and coloured by rank so one catastrophic outlier cannot
  flatten the scale. Ties share a shade, so the colouring never invents an
  ordering the numbers do not support.

All four are live in the Streamlit apps as well as static figures in the
READMEs, and the comparison matrix downloads as CSV.

Three conventions are enforced by the shared layer because getting them wrong
produces a *plausible wrong answer* rather than an error:

* **Images are RGB uint8 everywhere.** OpenCV loads BGR and matplotlib expects
  RGB; mixing them makes every figure blue and raises no exception.
* **`cv2` uses `(x, y)`, numpy uses `[row, col]`.** Every helper documents which.
* **uint8 arithmetic wraps.** `numpy +` turns 250 + 10 into 4; `cv2.add`
  saturates. All arithmetic goes through float and is clipped on the way back.

---

## Running the tests

```bash
python -m pytest -q          # 57 tests
```

The tests assert *numerical* behaviour, not just that the code runs — PSNR of
identical images is infinite, IoU of two half-overlapping masks is exactly 1/3,
a one-pixel edge offset scores badly at zero tolerance and well at two. Each
project's central finding is also asserted as a regression test, so a change that
quietly breaks a published result fails the build.

---

## License

MIT — see [LICENSE](LICENSE).
