# Classical Computer Vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV 4.14](https://img.shields.io/badge/OpenCV-4.14-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![No deep learning](https://img.shields.io/badge/deep%20learning-none-success)](#the-rules)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-lightgrey)](#the-rules)
[![Tests](https://img.shields.io/badge/tests-483%20passing-brightgreen)](#running-the-tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Measured comparisons of classical computer vision algorithms.** Every project
takes one problem, runs 3–6 classical methods against it, and reports a
comparative table, a side-by-side figure, pixel-level metrics, wall-clock timing
and **one stated finding with a number in it**.

No neural networks. No training. No GPU. Wherever a degradation can be applied
on purpose — darken an image, add haze, paste a region — the ground truth is
**generated**, so it is exact.

Where it cannot be, there are two other cases and they are kept apart on
purpose. Some projects have **human annotations**: BSDS500 ships five to seven
people's segmentations per image, so projects 24, 28 and 32 score against what
someone actually drew *and* report the **human ceiling** — one annotator against
the others' consensus, which lands near F 0.90 and not 1.0. The rest have
**nothing**, and those photographs are shown and never scored, because quoting an
accuracy against an image with no answer key would be inventing a number.

**259 photographs, twelve per project, and no image appears in two projects.**
Each pool is chosen on a *measured* axis — detail density, texture, tone range,
edge density, entropy, colourfulness, brightness — so that the pool spans
whatever the project actually tests, rather than being four pictures someone
liked. `tools/check_image_reuse.py` enforces the no-reuse rule by perceptual
hash, because filenames cannot: every image is renamed to something descriptive
on the way in.

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
   Real photographs are used as *inputs* to that process — project 04 hazes twelve
   Kodak scenes and scores against the originals — but a number is never quoted
   against an image whose answer nobody recorded.
6. **Four different subjects per comparison, chosen by the code.** Each project
   tries 10–12 candidates, drops any that fail a stated quality gate, and keeps
   the best survivor of each *family* so a figure cannot fill up with four
   variations on one thing. The rejects are printed in the README. **No photograph
   is reused across projects**, because a reader who sees the same cup four times
   learns nothing about the fourth method.

---

## Projects

**Status is stated honestly, because "it runs" and "it was measured" are different
claims:**

| | Meaning |
|:--:|---|
| ✅ | **Shipped.** Measured, figures generated, tests passing, findings written from real output |
| ❌ | **Not started.** No code. Every one of these needs data that is not yet in hand |

**All 58 are listed below in order, including the ones not yet built.** Leaving
the unbuilt ones out would make the plan look smaller than it is.

| # | Project | Methods compared | Headline finding | Status |
|---:|---|---|---|:--:|
| [01](projects/01_document_scanner/) | [**Document scanner**](projects/01_document_scanner/) | 6 page detectors · 4 binarisers · 2 aspect estimators | Otsu's failure on shadowed pages is **not** because a global threshold is impossible — the best global cut scores **0.890** where Otsu scores **0.678**. Also: the detector that wins on the benchmark (`Otsu + contour`, 1.07 px) **fails on a real photograph** | ✅ |
| [02](projects/02_portrait_mode/) | [**Portrait mode**](projects/02_portrait_mode/) | 6 matting methods · 4 aperture shapes · 2 compositors | Runs on **one real photograph of one real person**. Naive compositing looks fine and is **6.2× worse** (12.5 vs 2.04) in the ring outside the subject. A Gaussian blur renders highlights at peak/mean **2.94** where a real aperture is **1.00** | ✅ |
| [03](projects/03_low_light_enhancement/) | [**Low-light enhancement**](projects/03_low_light_enhancement/) | 8 methods: fixed/auto gamma, HE, CLAHE, SSR/MSR/MSRCR, LIME | The ceiling is **not** the algorithms: at gamma 3 only **158 of 256** tone levels survive, so even an exact inverse reaches **22.31 dB**. And an adaptive method beats a fixed constant by **+5.46 dB** where its assumption holds, loses by **−7.70 dB** where it does not — averaging to a number that describes neither | ✅ |
| [04](projects/04_dehazing/) | [**Dehazing**](projects/04_dehazing/) | dark channel prior, guided refine, CLAHE, Retinex, gamma | A **more accurate** airlight and transmission map produce a **worse** image — 19.50 dB falls to 18.50 dB when the airlight error is cut from 0.063 to 0.051; the two errors cancel. And CLAHE wins the contrast column (0.181 vs 0.156) while losing by **6.6 dB** | ✅ |
| [05](projects/05_old_photo_restoration/) | [**Old photo restoration**](projects/05_old_photo_restoration/) | Telea, Navier–Stokes, masked mean, harmonic diffusion, top-hat/black-hat, median residual, per-channel stretch | Choosing the best inpainting method is worth **1.1 dB**; knowing *where the damage is* is worth **14.0 dB**. Ranking detectors by IoU gets it **backwards** — the best-IoU detector restores to 9.88 dB, a worse-IoU one to 12.87 dB. And gray-world drives the no-reference cast to 0.04° while landing **further from the truth** (14.28°) than the faded input (7.72°) | ✅ |
| [06](projects/06_lane_detection/) | [**Lane detection**](projects/06_lane_detection/) | colour/Lab masks, Canny, Sobel-x, Hough · ROI-only and fixed-guess controls | **The region of interest is the algorithm.** Keep only the trapezoid and fit lines to its brightest pixels — that control lands a median **0.32%** of frame width from the full five-step pipeline on one camera. Removing the ROI costs **23×** more than removing any other step; removing Canny costs **0.016%**. And vanishing-point consistency, the natural annotation-free score, ranks a **constant guess first at 0.0 px** — only a recorded camera yaw exposes it | ✅ |
| [07](projects/07_copy_move_forgery/) | [**Copy-move forgery**](projects/07_copy_move_forgery/) | block matching, SIFT/ORB self-match, RANSAC similarity, dense verification | The best method on an exact copy is the worst at every other setting: block matching scores **0.9925 IoU** unrotated and **0.0000** at 2°. The decisive choice is the verifier's *hypothesis*, not the descriptor — identical SIFT matches score 0.794 vs 0.677 at 0° and 0.000 vs 0.499 at 90°. And rotation-robustness is paid for in false accusations: **6.3% of an untampered photo flagged**, vs 0.0% for block matching | ✅ |
| [08](projects/08_video_stabilisation/) | [**Video stabilisation**](projects/08_video_stabilisation/) | features+LK, phase correlation, ECC, block matching · 3 smoothers · 2 controls | **The estimator is not the bottleneck.** The best recovers the camera path to **0.035 px/frame**, 62× finer than the shake; swapping estimators moves the result **0.076**, swapping the *smoother* **0.639** — 8.4× more. Pointed at footage that never moved, ECC invents **25.3 px of drift**. And σ 2→32 is 13× steadier for **5× more crop** | ✅ |
| [09](projects/09_coin_counting/) | [**Coin counting & measurement**](projects/09_coin_counting/) | Otsu, top-hat illumination flattening, distance transform, local-maxima watershed, Hough circles | The OpenCV tutorial's seed rule (a fraction of the **global** distance maximum) counts **1 coin of 24** when a lighting artefact merges the mask; local-maxima seeding counts **24 of 24** on the same broken mask. And three methods count 24 while only **one** measures them plausibly — watershed's smallest basin implies a **5.75 mm** coin next to a 24.25 mm reference | ✅ |
| [10](projects/10_seam_carving/) | [**Seam carving**](projects/10_seam_carving/) | dynamic programming, 4 energy functions, integral-image ROI, plain-rescale control | **11.6 points** more of the high-energy region retained than `cv2.resize`, for **2,844×** the compute. The four energy functions span **0.5 points** — the one-line choice write-ups argue about is **23× smaller** than the choice they skip. And it wins its own objective (retained energy) on 4 images of 4 while winning region retention on only 3 | ✅ |
| [11](projects/11_hdr_exposure_fusion/) | [**HDR exposure fusion**](projects/11_hdr_exposure_fusion/) | Debevec + Reinhard/Drago/Mantiuk, Mertens fusion, naive mean, single-exposure controls | **Every real fusion method loses to averaging the frames** — the naive mean scores **0.928 SSIM** against Mertens' 0.848 and Debevec+Reinhard's 0.634, and runs in 9 ms against 1.8 s. And taking *one* photograph and doing nothing beats all three Debevec pipelines: 0.890 SSIM in 0.002 ms | ✅ |
| [12](projects/12_stereo_depth/) | [**Stereo → depth**](projects/12_stereo_depth/) | naive SAD, StereoBM, StereoSGBM, disparity → depth, occlusion handling | **Block matching is at once the most accurate method here and the worst** — 1.3% of the pixels it answers are wrong, and 23.6% if its refusals count as misses. Nothing changed but the convention for scoring a blank. Stripping BM's uniqueness, left-right and speckle checks costs **3.8× the error rate**, so most of the quality is not the matching cost | ✅ |
| [13](projects/13_denoising_shootout/) | [**Denoising shootout**](projects/13_denoising_shootout/) | box, Gaussian, median, bilateral, non-local means, adaptive Wiener, do-nothing control | Three noise models, **three different winners** — and the worst filter rotates too. Median wins impulse noise by **6.80 dB** and is the worst on Gaussian. Tuning transfers for **one filter of six** (bilateral +4.13 dB on held-out images; box **−0.81**). And below **sigma 10** every filter scores worse than leaving the image alone | ✅ |
| [14](projects/14_edge_detectors/) | [**Edge detectors**](projects/14_edge_detectors/) | Roberts, Prewitt, Sobel, Scharr, LoG, Canny | Normalising a gradient by its own maximum turned float rounding into a **constant 0.119 response on a flat image** — found by a test, not by looking. The operators separate on *thin* structure and nowhere else | ✅ |
| [15](projects/15_thresholding_family/) | [**Thresholding family**](projects/15_thresholding_family/) | fixed, Otsu, triangle, multi-Otsu, 2 adaptive, Niblack, Sauvola, + exhaustive-search oracle | "Use adaptive thresholding when the light is uneven" is **half a sentence**. On the *same* bad lighting Sauvola beats Otsu by 0.75 IoU on thin strokes and by 0.13 on solid shapes — it depends on the **shape of the foreground**, not the light. The oracle separates "Otsu picked the wrong cut" (1.000 available) from "no cut exists" (0.741) | ✅ |
| [16](projects/16_sharpening/) | [**Sharpening**](projects/16_sharpening/) | Laplacian (both signs + a deliberate sign error), unsharp, high-boost, + Wiener oracle | **Acutance cannot tell a sharpener from a sign error** — the wrong-signed Laplacian *raises* it (0.302 vs 0.245) while SSIM collapses to 0.149. High-boost goes from last to first, **+16.61 dB**, purely by matching brightness. And what a sharpener recovers collapses 12× with the blur while what *was* recoverable falls only 3.5× | ✅ |
| [17](projects/17_histogram_equalization/) | [**Histogram equalisation**](projects/17_histogram_equalization/) | HE, unclipped AHE, CLAHE, histogram matching, gamma, + match-the-true-histogram oracle | **Entropy is blind to a 9 dB improvement.** The oracle gains +9.18 dB while entropy moves −0.019 bits — and entropy's favourite method is the *worst* in the table. On **5 of 11** photographs nothing beats doing nothing, and CLAHE's clip limit has an optimum entropy points away from | ✅ |
| [18](projects/18_optical_flow/) | [**Optical flow**](projects/18_optical_flow/) | dense LK, pyramidal LK, Horn–Schunck, Farnebäck, DIS, + predict-zero control | "LK fails for large motion" — **the number is 1 pixel**, and each pyramid level roughly doubles it (1→4→8→16→32). Two harness bugs got there first: a negated truth made *every* method score worse than predicting zero, and an unnormalised Sobel made LK recover exactly ⅛ of the motion | ✅ |
| [19](projects/19_keypoint_detectors/) | [**Keypoint detectors**](projects/19_keypoint_detectors/) | Harris, Shi-Tomasi, FAST, SIFT, ORB, AKAZE, BRISK | **Harris (1988) is the most repeatable detector**, and ORB is both faster *and* more repeatable than SIFT — the folklore is wrong about detection. But Harris covers **33% of the frame** against SIFT's 81%, so the column that wins it the ranking is the one that makes it a poor choice. Right-angle rotations test nothing | ✅ |
| [20](projects/20_deblurring/) | [**Deblurring**](projects/20_deblurring/) | inverse, Wiener, Richardson–Lucy, regularised LS, unsharp control, + blind angle estimation | Handed the **true kernel**, the exact inverse scores **17 dB below doing nothing**. Only Richardson–Lucy clearly beats the control, and it diverges past an optimum PSNR and SSIM disagree about. The blind angle estimator was wrong twice over, the two errors cancelling at exactly 90° | ✅ |
| [21](projects/21_super_resolution/) | [**Single-image super-resolution**](projects/21_super_resolution/) | nearest, bilinear, bicubic, Lanczos, edge-directed, back-projection, + band-limited reference | The entire nearest-to-Lanczos argument is worth **0.49 dB**; modelling the degradation is worth **+1.62**. And the choice of *downsampler* moves the score by **2.12 dB** — more than the spread between all six methods. Nearest scores the highest "high-frequency energy" and is the worst method | ✅ |
| [22](projects/22_morphology/) | [**Morphology**](projects/22_morphology/) | erosion → black-hat, 3 structuring elements, skeletons, hit-or-miss | On a binarised photograph **no operation at any kernel size beats doing nothing** — a real binarisation has genuine single-pixel structure. The synthetic scene says the opposite, which is why both are reported. A cross keeps **360×** more diagonal structure than a rectangle | ✅ |
| [23](projects/23_fft_filtering/) | [**FFT filtering**](projects/23_fft_filtering/) | ideal/Butterworth/Gaussian low-pass, notch rejection, homomorphic | The one thing the spatial domain cannot do: a **blind** notch removes periodic interference for **+14.62 dB** where a median filter manages +1.73 — and it matches its own oracle to **0.0 px**. Ringing had to be measured on a step edge, because on photographs the obvious metric ranks Gaussian as the *worst* ringer | ✅ |
| [24](projects/24_region_segmentation/) | [**Region segmentation**](projects/24_region_segmentation/) | watershed ±markers, region growing, mean-shift, SLIC, GrabCut, + a grid of rectangles | **A grid of rectangles that never looked at the image beats five of six real methods on IoU.** Scored against *human* boundaries instead, the grid comes last where it belongs and the best method reaches **less than half** the human ceiling (0.433 against 0.904) | ✅ |
| [25](projects/25_matching_ransac/) | [**Matching + RANSAC**](projects/25_matching_ransac/) | SIFT/ORB/AKAZE, ratio test, cross-check, least squares, RANSAC, LMEDS, MAGSAC++ | **LMEDS breaks at exactly its theoretical 50%** — 0.222 px at 40%, 190.9 px at 60%. RANSAC holds to 80% and costs **46,000 iterations** at 90%. And SIFT wins here on descriptor accuracy (0.222 px vs ORB's 0.927) having *lost* to ORB on detection in project 19 | ✅ |
| [26](projects/26_quality_metrics/) | [**Do quality metrics agree?**](projects/26_quality_metrics/) | MSE, PSNR, SSIM, MS-SSIM, GMSD, VIF — six damages bisected to one PSNR | At an identical **28 dB**, SSIM spans 0.692 to 0.955. A one-pixel shift **cannot be made mild enough to reach 28 dB** while being nearly invisible. VIF was scoring the damage as an improvement, climbing to **40.7** as contrast was destroyed | ✅ |
| [27](projects/27_jpeg_from_scratch/) | [**JPEG from scratch**](projects/27_jpeg_from_scratch/) | DCT, standard tables, 4:2:0, zig-zag, RLE, entropy estimate — every stage switchable | Quantising **pixels** instead of DCT coefficients costs **9.33 dB at 2.3× the bitrate** — the transform is what makes the quantisation affordable. Chroma subsampling costs **0.03 dB** for 24% of the bits. Below quality 75 the top-frequency coefficient survives in **no block of any image** | ✅ |
| [28](projects/28_canny_sensitivity/) | [**Canny parameter sensitivity**](projects/28_canny_sensitivity/) | σ × low × ratio grid, on shapes **and** on human-annotated photographs | **Twelve settings score a perfect 1.000 on the synthetic scene** — a saturated benchmark cannot choose between them, and picking the wrong one costs **19%** on photographs. σ explains 30% of the variance; the high:low ratio every tutorial discusses explains **0.03%** | ✅ |
| [29](projects/29_object_tracking/) | [**Object tracking**](projects/29_object_tracking/) | template, mean-shift, CamShift, LK, LK+Kalman, **MOSSE from scratch** | **Three metrics, three winners** — template on mean IoU (0.608), MOSSE on survival (0.827), LK on centre error (22.5 px). A box nailed to frame 1 **beats CamShift** (0.248 vs 0.148). Adding a Kalman filter to the flow tracker *costs* 0.22 of survival: smoothing a confident error carries it further | ✅ |
| [30](projects/30_background_subtraction/) | [**Background subtraction**](projects/30_background_subtraction/) | frame diff, running average, median, MOG2, KNN + 2 controls | **The truth threshold picks the winner.** At 1.5× the measured noise floor the median background wins (0.450) and KNN is 4th; at 4× KNN wins (0.604) and the median is last. Pixel accuracy is **beaten by doing nothing** (0.962). And a *better* background model gives a worse mask: recall 0.74→0.98 while IoU falls 0.45→0.25 | ✅ |
| [31](projects/31_gw_pipeline/) | [**The G&W 8-stage pipeline**](projects/31_gw_pipeline/) | Laplacian, Sobel, smoothing, mask, sum, power-law — ablated one stage at a time | **A single unsharp mask beats all eight stages on every column**, including SSIM. Stage (e) contributes **nothing** (+0.010 when removed), and the mask stage *restrains* rather than adds. Both Laplacian signs raise acutance; the wrong one scores SSIM **−0.055** | ✅ |
| [32](projects/32_hough_transforms/) | [**Hough transforms**](projects/32_hough_transforms/) | standard and probabilistic lines, circles, on shapes **and** human-annotated photographs | Recall stays at **1.000 out to noise σ45** — voting is the strongest robustness result here. On photographs it is **less precise than the Canny it votes on** (0.117 vs 0.194) and marks **55% of the frame** as line | ✅ |
| [33](projects/33_texture/) | [**Texture descriptors**](projects/33_texture/) | GLCM, LBP, Gabor, Laws + raw-histogram control | **A descriptor is an invariance, not a quality.** LBP is the *worst* on clean patches (0.701, below a plain histogram) and the only one above chance after a relight (0.640 vs 0.083 = chance). Laws loses **nothing** to a 90° turn; GLCM computes four angles and still drops to 0.615. At 45° the do-nothing control beats all four | ✅ |
| [34](projects/34_rgb_to_grayscale/) | [**RGB → grayscale**](projects/34_rgb_to_grayscale/) | BT.601, BT.709, linear-light, value, contrast-preserving | **Every fixed weighting has a blind plane, including the correct one.** A scene on BT.709's plane gives it **0.39 grey levels** where BT.601 sees 14.17. The whole 601-vs-709 argument is worth **2.18 levels**; contrast-preserving costs **109×** the time for 0.0011 of edge recall. Across 55,596 real colour edges the catastrophe is essentially absent | ✅ |
| [35](projects/35_camera_calibration/) | [**Camera calibration & distortion**](projects/35_camera_calibration/) | 7 distortion models · view-count sweep · stereo · straightness check | **The reported RMS gets better as the calibration gets worse, twice over.** Three views report **0.205 px** (the best here) and score **0.557** on unseen ones; ten report 0.412 and score 0.311. The 14-coefficient model has the lowest error and puts the principal point **67 px** out. Meanwhile the stereo baseline is stable to **0.38%** | ✅ |
| [36](projects/36_shape_descriptors/) | [**Shape descriptors**](projects/36_shape_descriptors/) | Hu moments, Fourier descriptors, chain codes | **The log-Hu recipe in every tutorial is broken at zero** — `np.sign(0)` is 0, so an exactly-zero moment logs to 0 not −12; flooring takes rotation error **5.059 → 0.150**. Against 12 human-traced silhouettes, Hu's rotation invariance is 0.001 where **two people tracing the same object differ by 0.753** | ✅ |
| [37](projects/37_template_matching/) | [**Template matching**](projects/37_template_matching/) | SSD, NCC, ZNCC, multi-scale | **Raw cross-correlation localises 1 template in 9** on undegraded photographs — it finds the brightest patch, not the matching one. Only ZNCC survives an offset, and it takes a **negative** one to show it. ZNCC and NCC localise identically on a clean scene and differ **357×** in peak-to-mean: that is confidence, not accuracy | ✅ |
| [38](projects/38_panorama_stitching/) | [**Panorama stitching**](projects/38_panorama_stitching/) | 5 blenders incl. multi-band · exposure sweep · one real pair | **PSNR ranks the blenders backwards.** The winner is the control that does **not stitch** (23.85 dB) and it comes **last of five** on the seam (29.9 grey levels against multi-band's 19.5) — a panorama cannot be scored by fidelity to one of its own inputs. And at matched exposure every method lands within **0.42** grey levels: a blender comparison without an exposure difference measures nothing | ✅ |
| [39](projects/39_white_balance/) | [**White balance**](projects/39_white_balance/) | grey-world, white-patch, shades-of-grey, grey-edge | **One clipped highlight reduces white-patch to exactly the do-nothing control** — 15.949° both. Grey-world hallucinates **23.0°** out of a neutrally-lit scene where doing nothing scores 0.000. On an almost-uncast image **all five methods make it worse**. The usual Minkowski p=6 costs grey-edge **43% more error** than p=2 | ✅ |
| [40](projects/40_multiframe_super_resolution/) | [**Multi-frame super-resolution**](projects/40_multiframe_super_resolution/) | shift-and-add, iterative back-projection, n=1 control | **Frames break project 21's plateau: +2.26 dB against 0.49 for the whole nearest-to-Lanczos argument** — but **+0.68 dB of it is available from one frame**, so the n=1 control is what separates deblurring from fusion. Naive averaging is **worse than a single frame**. Bug: a negated ECC translation reported it 11× worse when it is **6.3× better** | ✅ |
| [41](projects/41_point_transforms/) | [**Point transforms**](projects/41_point_transforms/) | log, power-law, piecewise-linear, bit-plane | **A LUT and the arithmetic it replaces are bit-identical only if they round the same way** — truncating instead of rounding disagreed on **136 of 256 levels**, every one by exactly one, with no visible symptom. Fixed, identical at **43×** the speed. The textbook claim that brightening bands *because it duplicates levels* is backwards | ✅ |
| [42](projects/42_image_registration/) | [**Image registration**](projects/42_image_registration/) | phase correlation, ECC, mutual information | **Mutual information is exact (0.000 px) on every intensity relationship** including a non-monotonic remap, and costs **426×**. ECC returns *nothing* on inverted intensities. **The Hanning-window advice is backwards**: worth −0.024 px on the case it is for, and on inverted images it *causes* the failure — 9 of 12 go 30–635 px out with it, all 12 under 0.13 px without | ✅ |
| [43](projects/43_grayscale_to_colour/) | [**Grayscale → colour**](projects/43_grayscale_to_colour/) | Levin scribbles, Welsh transfer, pseudo-colour, luminance lookup + 2 controls | **The ceiling was knowable before anything ran**: the selection axis predicts an oracle's residual at **r = 0.995**. Two of four colourisers score **worse than returning the grey image**. Welsh transfer's score is the *reference*, not the method — the spread across eleven references is **39.1** where the method is worth 6.6 | ✅ |
| [44](projects/44_poisson_blending/) | [**Poisson blending**](projects/44_poisson_blending/) | copy-paste, feather, Poisson, mixed gradients | **The method that works moves the pasted pixels most** — Poisson changes the region by 51 grey levels, copy-paste by 0, and copy-paste has the worst seam. Any fidelity-to-source metric ranks them exactly backwards. Bug: **the from-scratch solver was a complete no-op**, seeding its boundary from the source so `f = source` is a fixed point | ✅ |
| [45](projects/45_wavelet_denoising/) | [**Wavelet denoising**](projects/45_wavelet_denoising/) | soft/hard × VisuShrink/BayesShrink + spatial competitors | **The sparsity premise holds and the method still loses** — signal carries 0.69–0.99 of its detail energy in the top 10% of coefficients against noise's 0.44, and a **bilateral filter beats every wavelet variant by 2.1 dB** for a quarter of the time. At σ 5 five of seven methods are worse than doing nothing. Soft vs hard flips with the threshold rule | ✅ |
| [46](projects/46_epipolar_geometry/) | [**Epipolar geometry**](projects/46_epipolar_geometry/) | 8-point raw/normalised, 7-point, LMedS, RANSAC + 3 controls | **The best estimate of F here has one parameter.** A control that assumes a rectified rig and fits a single vertical offset by a median scores **0.74 px** against RANSAC's 1.46. RANSAC reports an **81% inlier rate on a coplanar configuration that cannot determine F**. And the inlier rate *improves* as the real error doubles | ✅ |
| [47](projects/47_colour_space_robustness/) | [**Colour space robustness**](projects/47_colour_space_robustness/) | RGB, HSV, Lab, YCrCb, normalised RGB | **"HSV is lighting robust" is exactly half true.** HSV is *exactly* brightness-invariant (0.487 → 0.487) and the **least accurate** space undegraded (Lab 0.784, YCrCb 0.790) — and a warm cast costs it **37%**. At cast level 0.5 **every space scores 0.000**, and a white balance restores four of five: a colour space is not a substitute for correcting the illuminant | ✅ |
| 48 | **Chroma key / green screen** | colour keying, spill suppression, matting | *needs green-screen footage* | ❌ |
| [49](projects/49_plate_localisation/) | [**Licence-plate localisation**](projects/49_plate_localisation/) | Sobel+morphology, top-hat, MSER, contour+aspect, 2 Haar cascades · whole-frame and fixed-box controls | The repository's **only human annotation** — a drawn box *and* the typed plate text. **Two defensible metrics, two opposite orders**: `IoU >= 0.5` puts Sobel first at **8/14** and OpenCV's plate cascade at 5; coverage of the plate reverses it to **9** and 5. The metric introduced to fix IoU is the worse one — the character count, the only measure using the text, sides with IoU (**8** vs 4), and coverage is won outright by returning the whole photograph (**14/14**, IoU 0.01). Plate size (**67×** range) does not predict difficulty (r=+0.29), and the winner is the same at every threshold from 0.3 to 0.7 | ✅ |
| [50](projects/50_face_detection/) | [**Face detection**](projects/50_face_detection/) | 6 OpenCV cascades (4 Haar, 2 LBP) · nothing / one-centre-box / every-box controls | **Retitled from *face recognition*** — see below. A cascade cannot see below its own **training window**: `lbpcascade_frontalface_improved` is 45×45 and finds **3 faces** where its 24×24 predecessor finds **90**; upscaling recovers it **28.7×** against a next best of 1.54×, and `minSize` 12 vs 24 is **byte-identical for all six** because minSize is a floor, not a resampling. At **20°** the best cascade keeps **48%** of its own detections, at 30° **3%** — and that score alone ranks a fixed centre box first at **1.00**. Six cascades agree unanimously on **2 of 104** boxes | ✅ |
| [51](projects/51_demosaicing/) | [**Demosaicing / camera ISP**](projects/51_demosaicing/) | nearest, bilinear, Malvar, VNG, edge-aware | **Every method is 2.1–2.8 dB worse on edge pixels than over the whole frame** — whole-image PSNR averages the failure away. Cross-channel interpolation is the whole gain (**+4.86 dB**). OpenCV's edge-aware flag produces genuinely different pixels and scores **identically to its own bilinear on edges**, the pixels it is named for | ✅ |
| [52](projects/52_focus_stacking/) | [**Focus stacking / depth from focus**](projects/52_focus_stacking/) | 5 focus measures × 7 pooling windows + 3 controls | **The pooling window matters 2.3× more than the focus measure** (3.34 dB against 1.44), and the best measure is **0.034 dB** from an oracle handed the answer. On flat regions no measure *can* be right: agreement 0.982 → 0.496, and the five disagree on 58% of those pixels. Detail predicts the ceiling at **r = −0.969** | ✅ |
| [53](projects/53_barcode_qr/) | [**Barcode / QR detection**](projects/53_barcode_qr/) | gradient+morphology, variance, QR finder | **The synthetic background was hiding a useless localiser** — 1.000 on generated clutter, **0.021** on twelve real photographs, a 48× difference from changing nothing but the background. And **found is not decoded**: at 6.7 px per module the code is located 1.000 and decoded 0.000 | ✅ |
| [54](projects/54_defect_detection/) | [**Industrial defect detection**](projects/54_defect_detection/) | 6 residual/texture/spectral detectors · 4 defect kinds · clean-surface arm | **The detectors are complementary, not competing.** A **smear** — a local loss of texture at unchanged brightness — is found by the local standard deviation (0.58) and by **4 of the other 5 exactly never**. And on twelve surfaces with **nothing wrong with them** they mark **3.7% to 14.3%**; on brick paving one marks **76%**. Pixel accuracy is unusable: a defect covers 1.07%, so flagging nothing is right 98.9% of the time | ✅ |
| 55 | **Motion-triggered security alert** | background subtraction + blob tracking | *needs a surveillance clip* | ❌ |
| [56](projects/56_hand_gesture/) | [**Hand gesture recognition**](projects/56_hand_gesture/) | 7 segmenters (YCrCb/HSV/Lab skin, adaptive, Otsu, GrabCut) · oracle-mask, ellipse, whole-frame and nothing controls | **Settle the second stage first: it does not work.** Given a *perfect* mask, two classical finger rules over ten settings manage **3 of 5**. Then the first stage: over **324 composites** the spread across backgrounds is **0.831 IoU** against **0.425** across hands — the background moves the answer **2.0×** further than which hand it is, and how skin-coloured it is predicts the score at **r = −0.70** before anything runs. An ellipse drawn without reading the image scores 0.441 and beats **2 of the 7** | ✅ |
| [57](projects/57_pedestrian_detection/) | [**Pedestrian detection**](projects/57_pedestrian_detection/) | HOG + linear SVM, real footage vs drawn silhouettes | **Drawn silhouettes are not a benchmark.** HOG's SVM margin is **0.51** on drawings against **1.59** on real people, and recall on the composited scenes never exceeds 1 in 12 at any setting — so the project was rebuilt on real footage, checked against motion evidence that shares no information with it | ✅ |
| [58](projects/58_red_eye_removal/) | [**Red-eye removal**](projects/58_red_eye_removal/) | colour, +shape, +face, +eye constraints + empty-truth control | **Pupil PSNR rates the naive detector within 0.124 dB of the best while it marks 217× more of the frame.** On six photographs with *no red-eye at all*, colour-only marks 86,294 px and the face constraint 98. And zeroing the red channel scores **below doing nothing** on half the portraits | ✅ |

**58 projects spanning 14 algorithm families. 56 built, 2 not started.**

**The two that remain:**

| # | Project | What is actually missing |
|---:|---|---|
| **48** | Chroma key / green screen | a photograph shot against a real green screen. The subject could be composited onto green synthetically — that is how matting benchmarks are built — but then the spill model would be this repository's own, and spill is half the problem |
| **55** | Motion-triggered security alert | a surveillance clip with something to trigger on. `vtest.avi` is already carrying projects 29, 30 and 57, and a fourth project on the same 795 frames would be measuring the clip rather than the method |

Two projects changed shape when the data was finally found, and both say so in
their own README rather than quietly:

* **50** was listed as *face recognition* (Eigenfaces, Fisherfaces, LBPH). The
  eleven photographs that could be obtained are **group** photographs, which give
  many faces and no repeated identity, and classical recognition needs several
  images **per person**. It was built as face **detection** instead — six
  cascades, two truth arms, three controls. The recognition project is not done
  and is not claimed.
* **06** is built on fourteen dashcam **stills** from two cameras rather than a
  road clip, which turned out to be the better experiment: the two resolutions
  are what expose a region of interest written in pixels.
* **56** was listed as needing webcam footage. It does not: skin segmentation and
  convexity defects are per-frame methods. It is built on 27 real hand
  photographs composited onto 12 real backgrounds, which gives an exactly known
  mask. What it cannot measure is anything temporal, and it says so.

Where data was obtainable the project was built; where it was not, the row stays
❌ rather than being filled with something generated. Projects 08, 52 and 57 each
say in their own README exactly which part of their input is photographed and
which part is constructed, and why that is the right way round for the question
being asked.

**Every number in this table came out of an actual execution.** There is no
intermediate status any more: a project is either run end to end — figures
generated, tests passing, findings written from real output — or it has no code at
all. Each ✅ row is reproducible with `python run.py` inside the project.

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
python run.py                          # reproduces every number and figure
python infer.py path/to/your/image.jpg # run it on your own image
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
| [`shared/bench.py`](shared/bench.py) | timing harness — warm-up discarded, median of N runs |
| [`shared/report.py`](shared/report.py) | markdown tables, `results.json` with version provenance, a UTF-8-safe console |
| [`tools/verify.py`](tools/verify.py) | the pre-push checklist — every figure regenerates, no number is stale, no UI has crept back in |

### What a reader actually sees

**There is no app to launch.** This repo briefly had a Streamlit dashboard per
project and they were deleted: a screenshot of somebody else's control panel is
not a result, and it put a launch step between the reader and the comparison.

Every project opens with the same figure instead — **the input down the left,
every method across the columns, four different subjects down the rows, and the
score printed inside each cell.** Nothing to run, nothing to install, visible in
the README.

Under it sits **one signature visualisation chosen for that project alone** —
see [`docs/visualisation-plan.md`](docs/visualisation-plan.md). Project 03 gets
the tone-level survival curve, 09 a diameter histogram with denomination bands,
57 a DET curve. Three generic panels repeated fifty-eight times cannot carry
fifty-eight different findings.

And **one panel that is deliberately the same everywhere**: the methods ×
metrics matrix, each column scaled on its own and coloured by rank, so a reader
can compare project 14's best edge detector against project 28's without
relearning a chart. Ties share a shade, so the colouring never invents an
ordering the numbers do not support.

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
