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

| # | Project | Methods compared | Headline finding | Status |
|---:|---|---|---|:--:|
| [01](projects/01_document_scanner/) | **Document scanner** | 6 page detectors · 4 binarisers · 2 aspect estimators | Otsu's failure on shadowed pages is **not** because a global threshold is impossible — the best global cut scores 0.964 where Otsu scores 0.430 | ✅ |
| 02 | Portrait mode / background blur | Haar + GrabCut + bokeh kernels | — | 🔜 |
| 03 | Low-light enhancement | Retinex (SSR/MSR), LIME, gamma, HE | — | 🔜 |
| 04 | Dehazing | dark channel prior, CLAHE, Retinex | — | 🔜 |
| 05 | Old photo restoration | Telea, Navier–Stokes, exemplar inpainting | — | 🔜 |

Planned coverage is 41 projects spanning 14 algorithm families, all of which need
no download.

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
