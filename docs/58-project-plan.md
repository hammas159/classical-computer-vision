# 58 classical computer vision projects — priority order

## 📌 THESE TWO FILES GO TOGETHER — READ BOTH

| Order | File | What it holds |
|:---:|---|---|
| **1st** | `chat classical cv.md` *(not in this repo — written on the machine it was planned on)* | The decisions and **why**, machine state, which metric for which task, synthetic ground-truth recipes, gotchas that waste hours, `shared/` design, repo layout |
| **2nd** | `docs/58-project-plan.md` | **← you are here.** The 58 projects, the standing brief, the 14 families, the download split |
| **3rd** | [`docs/visualisation-plan.md`](visualisation-plan.md) | The signature visualisation chosen for each of the 58 |

**If you are reading this file, open the others as well.** This file says *what* to
build; the first says *why*, *how*, and *what will go wrong*.

> This list lived only on a desktop until 2026-09-16, which is why the project
> numbering in `projects/` had unexplained gaps at 6, 8, 11, 12 and thirteen
> other numbers — the names could not be recovered from the repo alone. It is
> vendored here so that cannot happen again.

---

Written 2026-09-14. **No deep learning, no training, no fine-tuning** in any of these.
Target repo: `classical-cv-lab` (the empty `computer-vision-lab` folder was deleted).

Every project = **one dataset · 3–6 classical methods · comparative table · side-by-side
figure · pixel metrics · one stated finding with a number**.

**Ranks 1–12 are the applied/real-world set.** Ranks 13 onward mix foundational family
surveys with the remaining applied work.

---

## The standing brief

This is the instruction every one of the 58 is built against. It supersedes the
looser "one dataset, 3–6 methods" line above wherever the two disagree.

> Work through all 58 in order: 1, 2, 3, 4, 5, 6, 7 … 58. Complete one project
> fully, push it to GitHub, then move to the next automatically. Do not stop to
> ask.
>
> For every one of the 58:
>
> 1. **UI** — each project gets its own distinct look. Different palette, accent
>    colour, heading font and corner radius. No two of the 58 may look alike.
>    Projects that have no UI yet need one built.
>
> 2. **Images** — download 10 to 12 candidates, run all of them, keep the best 4
>    for the results figure. Drop any sample that fails the project's quality
>    gate and replace it rather than shipping a broken one. Print the accept /
>    reject decision for every candidate in the README.
>
> 3. **Variety** — the 4 kept samples must be genuinely different KINDS of
>    subject: plant, animal, landscape, boy, girl, horse, couple, family photo,
>    object, building, and so on. Never four variations of one thing.
>
> 4. **No image is reused across projects.** 58 projects, roughly 580 distinct
>    images. Keep a registry so this is enforced, not hoped for.
>
> 5. **Video projects** (6, 8, 29, 30, 48, 55) — download 10 videos, keep the
>    best 4.
>
> 6. **Results at the top** of the README, right after a short intro. The format
>    is a comparison table made of pictures:
>    `Sr | Input | Method 1 | Method 2 | …`, one row per sample, numbered, with
>    the score printed inside each cell. If a project has only one method it is
>    still `Sr | Input | Result`.
>
> 7. **No new methods.** The algorithms are fine. Only the presentation, the
>    images and the UI change.
>
> 8. **Screenshots must differ from each other.** Measure before shipping — if
>    two captures agree over more than 35% of their rows, keep one.
>
> 9. **Never write a number the code did not produce.** If a project has no
>    honest finding, say so in the README rather than inventing one. If a number
>    moves because the images changed, report that it moved.
>
> 10. **No `Co-Authored-By` or AI footer** on any commit.
>
> Push after each project. Keep going until all 58 are done.

Each project also gets its **own signature visualisation** rather than the same
three panels repeated fifty-eight times — see
[`visualisation-plan.md`](visualisation-plan.md) for the one chosen for each.

How the brief is enforced rather than remembered:

| Rule | Enforced by |
|---|---|
| distinct UI per project | `shared/theme.py` — a palette per number, plus a generated `.streamlit/config.toml` |
| screenshots must differ | `tools/shoot.py` refuses two captures agreeing over 35% of their rows |
| no broken samples | each `run.py` gates candidates and prints the accept/reject log |
| no reused images | `assets/real/README.md` records every image and which project uses it |

---

## How to build one project, start to finish

Everything below is operational knowledge that was learned the expensive way.
Read it before starting a project rather than rediscovering it.

### 0 · Where images can actually come from

**Only GitHub raw is reachable from this machine.** `upload.wikimedia.org`
returns 400 and `commons.wikimedia.org` does not resolve. Do not waste a
download budget finding this out again.

| Source | URL pattern | What is there |
|---|---|---|
| Kodak PhotoCD suite | `raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/NN.png` | 24 varied 768×512 photographs, `01`–`24`. **All 24 are now used** (04 took twelve, 05 took five) |
| OpenCV samples | `raw.githubusercontent.com/opencv/opencv/4.x/samples/data/<name>` | ~90 images. `aloeL/aloeR/aloeGT` = stereo **with ground truth**; `leuvenA/leuvenB` = an exposure pair; `left01`–`left14`/`right01`–`right14` = checkerboards for calibration |
| OpenCV extra testdata | `raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/...` | faces under `cv/face/`, plus much more |
| Ultralytics | `raw.githubusercontent.com/ultralytics/yolov5/master/data/images/<name>` | `zidane.jpg`, `bus.jpg` — people |
| scikit-image | bundled, no download | 16 samples via `shared.io.sample` |

List any GitHub folder before downloading:
`curl -sS "https://api.github.com/repos/OWNER/REPO/contents/PATH" | grep '"download_url"'`

Avoid `lena.jpg` — widely deprecated as a test image, and a portfolio is exactly
the wrong place to use it.

### 1 · Pick the candidates

Ten to twelve, each tagged with a **family** describing what *kind* of subject
it is. The family tags are what stop the figure filling with four portraits.
Make them fine-grained on whatever axis the project cares about — project 05
splits people into boy / girl / child / woman / man / couple / group, because a
reader asking "will this fix my photo" is asking about a specific person.

Install them with `cv2.imwrite(..., [cv2.IMWRITE_JPEG_QUALITY, 92])`, cap the
long edge at 768 px, register them in `shared.io.REAL_PHOTOS` with a one-line
description, and record provenance in `assets/real/README.md`.

### 2 · Gate, then select

In `run.py`: run every candidate, compute the project's own quality measure,
**drop anything below a named threshold**, then keep the best survivor of each
family and take the top four. Print one line per candidate — the accept/reject
log goes in the README verbatim, because it is evidence the four were chosen by
the code and not by hand.

```
gallery candidate old_street        keep — 12.7 dB hazy, +11.2 dB best  [street]
gallery candidate warplane          keep — 17.8 dB hazy,  +3.3 dB best  [sky]
```

### 3 · Build the comparison figure

`shared.figures.gallery(columns, rows, out, cell_notes=..., suptitle=...)`

`columns` is `["input"] + method_names`; `rows` is `[(sample_label, [images])]`;
`cell_notes` puts the score inside each cell. It numbers rows `Sr 1…4` and
letterboxes **per row**, so stages of differing aspect still line up.

Emit the same numbers as a markdown table from `run.py` so the README's table is
generated, never transcribed.

### 4 · Build the signature visualisation

See [`visualisation-plan.md`](visualisation-plan.md). One per project, chosen to
carry that project's finding. Most of these plotters do not exist in
`shared/figures.py` yet and are real work.

### 5 · Theme and screenshot the UI

`shared/theme.py` owns the look. In the app, immediately after
`st.set_page_config`:

```python
PALETTE = theme.apply(NN)     # NN = project number
```

and regenerate `.streamlit/config.toml` from the same palette — CSS alone leaves
BaseWeb widgets (selectbox, radio, checkbox) in the stock dark theme, which puts
a navy dropdown on a cream page.

Then:

```bash
python tools/shoot.py 04_dehazing 01_pipeline "02_transmission:Transmission map"
```

It picks a free port, launches from **inside** the project folder (Streamlit
resolves `.streamlit/config.toml` against the working directory), photographs
each tab, tears the server down, and **fails if two captures agree over 35% of
their rows**.

### 6 · README order

```
# title + badges
short intro
## Results          <- comparison figure, generated table, accept/reject log
> the finding, in one sentence
Jump to …
## What it does
## Screenshots
## Full results tables
## Run it yourself / Inference / How it works
## Problems hit, and how they were solved
## Limitations / Tests / Keywords / References
```

### 7 · Verify, then commit

```bash
cd projects/NN_name && python run.py
cd ../.. && .venv/Scripts/python.exe -m pytest projects/NN_name/tests -q
```

Look at the generated `docs/images/samples.png` before committing. Four rows,
four genuinely different subjects, every cell a usable result.

Commit messages state what was found and what it cost. **No `Co-Authored-By`,
no AI footer** — it adds a second name to GitHub's contributor list.

### Traps already paid for

* **A stale Streamlit server answers.** It prints `Port N is not available`, the
  new server exits, and the *old* one keeps serving — so the screenshot silently
  shows the previous build. `tools/shoot.py` exists because of this.
* **Bash heredocs mangle `\n` inside Python string literals**, producing real
  newlines and an unterminated-string SyntaxError. Use the `Edit` tool for code.
* **`st.set_page_config` on one line** has no trailing comma, so inserting an
  argument after it breaks the file. Four apps broke this way.
* **Whole-image PSNR hides everything** when the change covers a few percent of
  pixels. Score the region that changed.
* **A method can win its own objective and lose the real one.** Keep a
  do-nothing control column in every comparison; several projects turn out worse
  than doing nothing in part of their range.

### Known outstanding

* `projects/05_old_photo_restoration` has one failing test,
  `test_restore_detects_a_mask_when_none_is_given` — it asserts saturation rises
  after restoration, which is not universally true. Pre-existing, not caused by
  the format work.
* Eight photographs in `assets/real/` have **no recorded provenance**
  (`girl`, `dog`, `coffee_cup`, `woman_field`, `leopard`, `man_camera`, `hiker`,
  `man_skyline`). Stated in `assets/real/README.md` rather than guessed at.

---

## Legend

**Data** — `none` means no download: either synthetic (you create the degradation, so
ground truth is exact) or it ships inside `scikit-image` / OpenCV.

**Family** — which of the 14 algorithm families it belongs to (see the end of this file).
Two projects in the same family overlap; two in different families do not.

---

## THE LIST

| # | Project | Methods compared | Data | Family |
|---:|---|---|---|:---:|
| **1** | **Document scanner** | edges → contour → perspective transform → binarise | none | 8,10 |
| **2** | **Portrait mode / background blur** | Haar + GrabCut + bokeh kernels | none | 12 |
| **3** | **Low-light enhancement** | single/multi-scale Retinex, LIME, gamma, HE | none | 1 |
| **4** | **Dehazing** | dark channel prior, CLAHE baseline, Retinex | none | 1 |
| **5** | **Old photo restoration** | Telea, Navier–Stokes, exemplar-based inpainting | none | 12 |
| **6** | **Lane detection** | colour mask + Canny + Hough + ROI | road clip | 8 |
| **7** | **Copy-move forgery detection** | block matching, keypoint-based forensics | none | 9 |
| **8** | **Video stabilisation** | feature trajectories + trajectory smoothing | handheld clip | 11 |
| **9** | **Coin counting & measurement** | watershed + morphology + calibration to mm | ships free | 6,7 |
| **10** | **Seam carving** | energy maps + dynamic programming | none | 12 |
| **11** | **HDR exposure fusion** | Debevec, Mertens, Reinhard tone mapping | brackets | 14 |
| **12** | **Stereo → depth → 3D point cloud** | BM vs SGBM + interactive 3D render | stereo pair | 10 |
| 13 | Denoising shootout | box, Gaussian, median, bilateral, NLM, Wiener | none | 2 |
| 14 | Edge detectors | Sobel, Prewitt, Roberts, Scharr, LoG, Canny | none | 2,8 |
| 15 | Thresholding family | global, Otsu, multi-Otsu, adaptive, Niblack, Sauvola | none | 6 |
| 16 | Sharpening | Laplacian, unsharp mask, high-boost | none | 2 |
| 17 | Histogram equalization family | HE, AHE, CLAHE, matching, gamma | none | 1 |
| 18 | Optical flow | LK, pyramidal LK, Farnebäck, Horn–Schunck, DIS | none | 11 |
| 19 | Keypoint detectors | Harris, Shi-Tomasi, FAST, SIFT, ORB, AKAZE, BRISK | none | 9 |
| 20 | Deblurring | inverse, Wiener, Richardson–Lucy, regularised, blind | none | 4 |
| 21 | Single-image super-resolution | nearest, bilinear, bicubic, Lanczos, NEDI, back-projection | none | 5 |
| 22 | Morphology | erosion…top-hat, skeletonisation, hit-or-miss | none | 7 |
| 23 | FFT filtering | ideal/Butterworth/Gaussian, notch, homomorphic | none | 3 |
| 24 | Region segmentation | watershed, region growing, mean-shift, SLIC, GrabCut | ships free | 6,12 |
| 25 | Matching + RANSAC homography | ratio test, RANSAC vs LMEDS | none | 10 |
| 26 | **Do quality metrics agree?** | MSE, PSNR, SSIM, MS-SSIM, VIF ranked against each other | none | meta |
| 27 | JPEG from scratch | DCT, quantisation tables, quality vs PSNR/SSIM | none | 3 |
| 28 | Canny parameter sensitivity | σ × low × high threshold grid | none | 2 |
| 29 | Tracking | Kalman, mean-shift, CAMShift, KCF, CSRT, MOSSE | short clip | 11 |
| 30 | Background subtraction | frame diff, running average, MOG, MOG2, KNN | short clip | 11 |
| 31 | G&W 8-stage enhancement pipeline | Laplacian + Sobel + smoothing + power-law, stage by stage | none | 2 |
| 32 | Hough transforms | lines, circles, generalised | none | 8 |
| 33 | Texture | GLCM, LBP, Gabor, Laws' energy | ships free | 9 |
| 34 | RGB → grayscale | BT.601, BT.709, average, luminosity, contrast-preserving | none | 1 |
| 35 | Camera calibration & distortion | reprojection error vs number of views | checkerboard | 10 |
| 36 | Shape descriptors | Hu moments, Fourier descriptors, chain codes | none | 8 |
| 37 | Template matching | SSD, NCC, ZNCC, multi-scale | none | 9 |
| 38 | Panorama stitching | homography, cylindrical warp, multi-band blending | 2–3 photos | 10 |
| 39 | White balance | grey-world, white-patch, grey-edge, manual | none | 1 |
| 40 | Multi-frame super-resolution | shift-and-add, iterative back-projection | none | 14 |
| 41 | Point transforms & contrast stretching | log, power-law, piecewise-linear, bit-plane | none | 1 |
| 42 | Image registration | phase correlation, ECC, mutual information | none | 10 |
| 43 | Grayscale → colour | Levin scribble-optimization, Welsh transfer, pseudo-colour | reference img | 12 |
| 44 | Poisson / seamless blending | gradient-domain compositing | none | 12 |
| 45 | Wavelet vs spatial denoising | soft/hard thresholding, BayesShrink | none | 3 |
| 46 | Epipolar geometry | fundamental/essential matrix, 8-point vs RANSAC | stereo pair | 10 |
| 47 | Colour space robustness | RGB/HSV/Lab/YCbCr under illumination change | none | 1 |
| 48 | Chroma key / green screen | colour keying, spill suppression, matting | green footage | 6 |
| 49 | License plate localisation | edge + morphology + contour filtering | plate images | 7,8 |
| 50 | Face recognition | Eigenfaces (PCA), Fisherfaces (LDA), LBPH | face dataset | 13 |
| 51 | Demosaicing / camera ISP | Bayer → bilinear/Malvar → white balance → denoise | none | 1,5 |
| 52 | Focus stacking / depth from focus | focus measures, depth map from a focal stack | focal stack | 14 |
| 53 | Barcode / QR detection | gradient + morphology + decode | none | 7 |
| 54 | Industrial defect detection | template + morphology + blob analysis | defect imagery | 7 |
| 55 | Motion-triggered security alert | background subtraction + blob tracking | surveillance clip | 11 |
| 56 | Hand gesture recognition | skin colour + contours + convexity defects | webcam | 13 |
| 57 | Pedestrian detection | HOG + SVM | none | 13 |
| 58 | Red-eye removal | colour thresholding + morphology | none | 1 |

---

## The 14 algorithm families

The 58 projects sit on **14 distinct families**, roughly 120 algorithms total. Two
projects sharing a family number overlap; covering all 14 once means no family is missing.

| # | Family | What it computes |
|---:|---|---|
| 1 | Point / intensity ops | per-pixel mapping |
| 2 | Local convolution | smoothing, sharpening, gradients |
| 3 | Frequency transforms | FFT, DCT, wavelet |
| 4 | Deconvolution | inverse problems |
| 5 | Interpolation / resampling | scale change |
| 6 | Thresholding & clustering | split pixels into groups |
| 7 | Morphology | set operations on shape |
| 8 | Contour & shape analysis | boundaries, moments |
| 9 | Local features & descriptors | keypoints, matching |
| 10 | Geometric estimation | homography, F-matrix, calibration |
| 11 | Temporal / motion | flow, subtraction, tracking |
| 12 | Energy minimisation | graph cuts, DP, gradient-domain |
| 13 | Statistical recognition | PCA, LDA, LBP, HOG+SVM |
| 14 | Multi-image fusion | combine exposures / frames |

---

## SPLIT BY WHAT YOU NEED TO DOWNLOAD

Counted from the table above, not estimated. **41 of 58 need nothing at all** — 38 marked
`none` plus 3 whose images ship inside `scikit-image`.

### A · No download needed — 41 projects, priority order

Buildable the moment the ~60 MB of libraries are installed. Nothing else. These are
unblocked regardless of what else is downloading.

| Rank | Project | Why no data is needed |
|---:|---|---|
| 1 | Document scanner | photograph any page yourself |
| 2 | Portrait mode / background blur | Haar cascade ships inside OpenCV |
| 3 | Low-light enhancement | darken an image → you know the true brightness |
| 4 | Dehazing | add synthetic haze → you know the true clear image |
| 5 | Old photo restoration | draw the damage → you know which pixels to repair |
| 7 | Copy-move forgery detection | you paste the region → exact forgery mask |
| 9 | Coin counting & measurement | `skimage.data.coins` ships free |
| 10 | Seam carving | any image works |
| 13 | Denoising shootout | you set the noise σ → exact ground truth |
| 14 | Edge detectors | synthetic shapes → exact edge map |
| 15 | Thresholding family | synthetic regions → exact mask |
| 16 | Sharpening | any image |
| 17 | Histogram equalization family | any image |
| 18 | Optical flow | you warp by a known field → exact flow |
| 19 | Keypoint detectors | known homography → exact correspondence |
| 20 | Deblurring | you apply the kernel → exact ground truth |
| 21 | Single-image super-resolution | downsample then restore → original is the truth |
| 22 | Morphology | synthetic shapes |
| 23 | FFT filtering | inject known periodic noise |
| 24 | Region segmentation | `skimage.data` ships free |
| 25 | Matching + RANSAC homography | known transform → exact reprojection error |
| 26 | Do quality metrics agree? | any images |
| 27 | JPEG from scratch | any image |
| 28 | Canny parameter sensitivity | synthetic edges |
| 31 | G&W 8-stage enhancement pipeline | any image |
| 32 | Hough transforms | synthetic lines and circles |
| 33 | Texture | `skimage.data` textures ship free |
| 34 | RGB → grayscale | any colour image |
| 36 | Shape descriptors | synthetic shapes |
| 37 | Template matching | crop a patch from the image itself |
| 39 | White balance | apply a known colour cast |
| 40 | Multi-frame super-resolution | generate sub-pixel shifted frames |
| 41 | Point transforms & contrast stretching | any image |
| 42 | Image registration | apply a known shift/rotation |
| 44 | Poisson / seamless blending | any two images |
| 45 | Wavelet vs spatial denoising | known noise σ |
| 47 | Colour space robustness | simulate illumination change |
| 51 | Demosaicing / camera ISP | mosaic an RGB image → original is the truth |
| 53 | Barcode / QR detection | generate the codes yourself |
| 57 | Pedestrian detection | HOG+SVM ships inside OpenCV |
| 58 | Red-eye removal | paint synthetic red-eye |

### B · Needs data — 17 projects

Most of these are **not downloads** — they are things you can shoot with a phone in a
few minutes. Only six require fetching someone else's dataset.

**B1 · Shoot it yourself with a phone — 12 projects**

| Rank | Project | What to capture |
|---:|---|---|
| 6 | Lane detection | a short clip driving or walking a marked road |
| 8 | Video stabilisation | 15 seconds of deliberately shaky handheld video |
| 11 | HDR exposure fusion | same scene at 3 exposures (lock focus, change EV) |
| 29 | Tracking | any clip with one moving object |
| 30 | Background subtraction | fixed camera, something walks through |
| 35 | Camera calibration & distortion | print a checkerboard, photograph it ~15 times |
| 38 | Panorama stitching | 2–3 overlapping photos, pivot on the spot |
| 43 | Grayscale → colour | any colour photo as the reference |
| 48 | Chroma key / green screen | a green cloth and a phone |
| 52 | Focus stacking | same scene, refocused near → far |
| 55 | Motion-triggered security alert | fixed camera clip (can reuse rank 30's) |
| 56 | Hand gesture recognition | webcam |

**B2 · Must download — 5 projects**

| Rank | Project | Dataset | Rough size |
|---:|---|---|---:|
| 12 | Stereo → depth → 3D point cloud | Middlebury stereo pair | small |
| 46 | Epipolar geometry | same stereo pair as rank 12 | reuses 12 |
| 49 | License plate localisation | plate images | moderate |
| 50 | Face recognition | AT&T / Yale faces | ~10 MB |
| 54 | Industrial defect detection | MVTec-style defect imagery | large |

**So the true blocking set is four datasets** — ranks 12 and 46 share one stereo pair.

**41 + 12 + 5 = 58.**

### What this means for sequencing

Build **A** first — 41 projects, all unblocked, nothing competing with the model
download. Capture **B1**'s footage in one 20-minute session with a phone whenever
convenient. Leave **B2** until the network is free.

---

## If building fewer than 58

**The recommended 24** — full coverage, zero redundancy:

- **Ranks 1–12** (all applied) — but drop any whose data you don't have
- **Ranks 13–24** (foundational surveys) — these cover every one of the 14 families
- Optionally **26** (quality metrics) as the meta piece

Everything from rank 25 down is a *second* project inside a family already covered —
real work, but diminishing returns per hour.

---

## Honest notes

🚨 **Overlap is real.** Canny appears in at least six projects, GrabCut in three,
morphology in five. That is fine only if each project asks a **different question**.
Four projects that read as "I ran Canny again" are worse than one that reads as a study.

🚨 **Three use pre-trained but non-deep components** — rank 2 (Haar cascade), 50 (LBPH),
57 (HOG+SVM). All ship inside OpenCV: no download, no training by you. Say so in the
README rather than let a reader discover it.

🚨 **Synthetic beats real for most of these.** Darken an image and you know the true
brightness; add haze and you know the true clear image; paste a region and you know
exactly which pixels were forged. Exact ground truth, zero download — the same pattern
that makes `swebench-localization` and `code-eval-harness` credible.

⚠️ **Enhancement cluster risk.** Ranks 3, 4, 16, 20, 21 all look like "make the image
better". Each needs a distinct question or they read as one project split five ways:
- 3 — can you recover what darkness hid?
- 4 — can you remove a known scattering model?
- 16 — does sharpening add information, or only contrast?
- 20 — can you undo a *known* blur?
- 21 — can you invent pixels that were never captured?

---

## Install needed

```
opencv-python-headless   ~40 MB
scikit-image             ~13 MB
matplotlib                ~8 MB
```

Roughly **60 MB total**. Nothing else — no model weights, no training data for the top
of the list.

## Related files

- **`chat classical cv.md`** — the companion to this file. Decisions and reasoning,
  machine state, metrics reference, synthetic ground-truth recipes, the gotchas list,
  `shared/` design, repo layout, and the open questions. **Read it with this one.**
- `SWE PROJECTS - queue.md` — all other pending work, priority ordered
- `GENERATIVE PROJECTS - 5 domains.md` — the deep-learning vision roadmap (separate repo)
- `LLM MODELS - what runs on 16GB.md`
- `VOICE ASSISTANT - models and feasibility.md`
