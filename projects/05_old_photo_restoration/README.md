# 05 · Old photo restoration — classical computer vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV 4.14](https://img.shields.io/badge/OpenCV-4.14-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![No deep learning](https://img.shields.io/badge/deep%20learning-none-success)](#)
[![Runs on CPU](https://img.shields.io/badge/hardware-CPU%20only-lightgrey)](#)

Two unrelated jobs share the name "restoration". **Inpainting** replaces pixels
that are *missing* — scratches, tears, emulsion loss. **Fade correction** fixes
pixels that are *present but wrong* — a print that has gone flat, warm and
desaturated. Neither method touches the other's problem, and this project scores
them separately so that cannot be hidden.

**No neural network, no training, no GPU, no dataset download.**

> **The finding, in one sentence.** Choosing the best of four inpainting methods
> is worth **1.1 dB**. Knowing *where the damage is* is worth **14.0 dB**. The
> literature ranks inpainters; the decibels are almost entirely in the step those
> comparisons skip.

> **The second finding.** Ranking damage detectors by mask IoU gets the answer
> **wrong**. The intensity threshold has the best IoU (0.4199) and restores to
> 9.88 dB; the median-residual detector has a slightly *worse* IoU (0.4185) and
> restores to **12.87 dB**. Precision and recall do not cost the same here — a
> missed scratch stays in the picture, while a falsely flagged healthy pixel is
> replaced by an average of its healthy neighbours, which is approximately itself.

> **The metric trap.** Gray-world white balance drives the *no-reference* colour
> cast to **0.04°** — a perfect score — while moving the colour balance
> **further from the truth** than the faded print it started from (14.28° of
> error against the input's 7.72°). It did not correct the colour. It removed it.

**Jump to:** [What it does](#what-it-does) · [Screenshots](#screenshots) ·
[UI → results](#how-the-ui-connects-to-the-results) · [Results](#results) ·
[Run it](#run-it-yourself) · [Inference](#inference-try-it-on-your-own-photo) ·
[How it works](#how-it-works) · [Problems solved](#problems-hit-and-how-they-were-solved) ·
[Limitations](#limitations) · [Keywords](#keywords)

---

## What it does

```mermaid
flowchart LR
    A[Clean photo<br/>known exactly] --> B[Fade it<br/>dye loss, yellowing, flat tone]
    B --> C[Scratch it<br/>exact damage mask retained]
    C --> D[3 damage detectors<br/>find the mask for themselves]
    C -.the TRUE mask.-> F
    D --> E[4 inpainting methods]
    E --> F[Score: PSNR on the<br/>DAMAGED PIXELS ONLY]
    C --> G[6 fade corrections]
    G --> H[Score: cast error vs the<br/>original, and LAB chroma]
    F --> I{Where did the<br/>decibels actually go?}
    H --> I

    style C fill:#fef3c7,stroke:#d97706
    style F fill:#fee2e2,stroke:#dc2626
```

Because the damage is generated, **three** things are known that a real archive
scan cannot give you:

| Known | Lets us ask |
|---|---|
| the undamaged photograph | how close is the output? (PSNR, SSIM) |
| the exact damaged-pixel mask | **how much did detection cost?** (the ceiling) |
| the original's colour balance | did it *correct* the cast, or just neutralise it? |

That second row is the project. Every restoration demo shows a before and an
after; almost none shows what the same pipeline would have produced if it had
been handed a perfect mask, which is the only way to tell whether the method or
the detector is the limiting factor.

---

## Screenshots

The interactive app — `streamlit run ui/app.py` — takes an image, ages it by a
controllable amount, and runs both halves of the restoration with live
measurement.

### The main view: five stages, one row

![The restoration UI](results/ui_main.png)

Left to right: the truth, the aged print, what the detector found, what
inpainting fixed, and what fade correction fixed. Note panel 4 — **after
inpainting the picture is still yellow**, because inpainting only ever touches
the pixels in the mask. The sentence under the metrics is the project in one
line: *5.79 dB of this result is lost to detection, not to the inpainting
method.*

### The experiment that separates the methods

![Damage width sweep](results/ui_width_sweep.png)

Every method here is interpolation across a gap, so the only real question is how
wide a gap it survives. At 1 px they are within 2 dB of each other. Harmonic
diffusion — the textbook Laplace solution — falls off a cliff past 9 px, because
the solution to Laplace's equation over a wide hole is a smooth surface with no
texture at all: plausible, and wrong. The 20-line masked-mean baseline is the
green line that *stops losing* at the right-hand end.

### Detector comparison — where IoU and PSNR disagree

![Detector matrix](results/ui_detector_matrix.png)

Three detectors, scored on mask quality *and* on what that mask is worth
downstream. The IoU column and the restored-PSNR column do not rank the same way,
and precision/recall is why.

### Method comparison

![Method matrix](results/ui_method_matrix.png)

Two PSNR columns, deliberately. Whole-image PSNR barely moves between methods
because 93% of the photograph was never damaged — the damage-only column is the
one that is actually about inpainting.

### The scratch, read as numbers

![Pixel matrix](results/ui_pixel_matrix.png)

A 12×12 patch straddling a scratch. The damaged column is a block of values with
no relationship to their neighbours; every method in this project is guessing
what belongs there, and this is the guess.

### Tone distribution

![Tone distribution](results/ui_tone.png)

A faded print occupies a narrow band in the middle of the range. The per-channel
stretch is nothing more sophisticated than putting the two ends back where they
belong — which is exactly why it beats CLAHE, a *local* method applied to a
*global* problem.

### Every method, and every fade correction, on one image

![All inpainting methods](results/ui_all_methods.png)

![All fade corrections](results/ui_all_fades.png)

---

## How the UI connects to the results

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        I1[Pick a sample image]
        I2[Scratch width 1-40 px]
        I3[Blotch count]
        I4[Fade the print?]
        I5[OR upload your own scan]
    end

    subgraph PIPE["Pipeline — the app runs the same code as run.py"]
        P1[synth.fade_photo]
        P2[synth.add_scratches<br/>keeps the exact mask]
        P3[DETECTORS - find the damage]
        P4[METHODS - fill it]
        P5[FADE_METHODS - fix the tone]
    end

    subgraph OUT["Displayed results"]
        O1[5-panel strip:<br/>truth, damaged, mask, inpainted, restored]
        O2[Metrics: time, flagged %,<br/>chroma, contrast, PSNR]
        O3[The ceiling sentence:<br/>what the TRUE mask would have scored]
        O4[Width sweep, live, on YOUR image]
        O5[Method x metric matrix + CSV]
        O6[Detector x metric matrix + CSV]
        O7[Pixel grid and histogram]
    end

    I1 & I2 & I3 & I4 --> P1 --> P2 --> P3 --> P4 --> P5
    I5 --> P3
    P5 --> O1 & O2
    P2 -.true mask.-> O3
    P4 --> O4 & O5
    P3 --> O6
    P5 --> O7

    style O3 fill:#fee2e2,stroke:#dc2626
    style P2 fill:#fef3c7,stroke:#d97706
```

Uploading your own scan disables O3, O4 and the reference columns of O5/O6 — and
the app says so rather than showing a number it cannot compute. There is no
original to compare a real archive print against, and inventing one is how
restoration demos end up reporting scores for images that have no ground truth.

---

## Results

All numbers from `python run.py`, written to
[`results/results.json`](results/results.json) and
[`results/tables.md`](results/tables.md). Six images, 3 px damage unless stated.

### Inpainting, handed the true mask

| Method | PSNR whole (dB) | PSNR on damage (dB) | SSIM | Time (ms) |
|---|---:|---:|---:|---:|
| Telea (fast marching) | 38.110 | 26.555 | 0.9826 | **14.7** |
| **Navier-Stokes** | **38.405** | **26.850** | **0.9838** | 14.7 |
| Iterative masked mean | 37.328 | 25.773 | 0.9808 | 79.8 |
| Harmonic diffusion | 38.210 | 26.655 | 0.9830 | 392.7 |

The damaged input scores 17.36 dB whole-image and **6.03 dB on the damaged
pixels**. That gap is why the damage-only column exists: doing nothing at all
already earns a whole-image number that sounds like a working method.

**The spread across four methods is 1.1 dB.** The 20-line baseline reaches 96% of
the best. At this damage width the choice of method is very nearly irrelevant.

### …and width is what actually matters

| Width (px) | Damage fraction | Telea | Navier-Stokes | Masked mean | Harmonic diffusion |
|---|---:|---:|---:|---:|---:|
| 1 | 0.018 | 28.097 | **29.178** | 27.202 | 28.389 |
| 3 | 0.071 | 26.555 | **26.850** | 25.773 | 26.655 |
| 5 | 0.094 | 25.332 | **25.461** | 24.813 | 25.306 |
| 9 | 0.138 | **23.825** | 23.756 | 23.647 | 20.515 |
| 15 | 0.197 | 22.525 | 22.476 | **22.545** | 15.132 |
| 25 | 0.287 | 20.862 | 20.763 | **20.884** | 10.524 |
| 40 | 0.396 | 19.038 | 19.131 | **19.201** | **8.231** |

Harmonic diffusion loses **20.2 dB** across this sweep; Telea loses 9.1 dB. And
the winner at the wide end is the naive masked mean — not because it got better,
but because once the gap is wide enough that nothing can reconstruct the texture,
the method that promises least loses least.

### Detection — the step that actually costs the decibels

| Detector | Mask IoU | Precision | Recall | Flagged | PSNR on damage (dB) |
|---|---:|---:|---:|---:|---:|
| Intensity threshold | **0.4199** | **0.602** | 0.571 | 0.066 | 9.882 |
| Top-hat + black-hat | 0.3797 | 0.418 | 0.772 | 0.138 | 12.706 |
| **Median residual** | 0.4185 | 0.466 | **0.776** | 0.122 | **12.865** |
| *(true mask — the ceiling)* | *1.000* | *1.000* | *1.000* | *0.071* | ***26.850*** |

**The detection tax is 14.0 dB.** The method spread is 1.1 dB. The two numbers
differ by more than an order of magnitude, and every one of the four inpainting
methods sits on the wrong side of that comparison.

The IoU column and the PSNR column rank differently because **precision and
recall do not cost the same**:

* A **missed** scratch is never handed to the inpainter. No method can recover it.
* A **falsely flagged** healthy pixel is replaced by an average of its healthy
  neighbours — which is approximately itself.

So a detector should be tuned for recall and allowed to over-flag, and IoU, which
weights the two errors equally, is the wrong objective for this job.

### The detector's blind spot, measured

| Median window (px) | Mask IoU | Recall | Precision |
|---|---:|---:|---:|
| 5 | 0.081 | **0.200** | 0.225 |
| 7 | 0.092 | 0.173 | 0.229 |
| 9 | 0.170 | 0.288 | 0.310 |
| 11 | 0.350 | 0.592 | 0.449 |
| 15 | 0.393 | 0.700 | 0.463 |
| **21** *(default)* | **0.419** | **0.776** | 0.466 |
| 31 | 0.426 | 0.833 | 0.457 |

A median filter rejects a **minority** of outliers. Once a 3 px scratch fills
half of a 5 px window the median *becomes* the scratch, the residual the detector
looks for goes to zero, and it reports a clean image. Recall does not degrade
gracefully — it falls off a cliff between 11 px and 7 px. The default of 21 was
chosen from this table, not from a paper.

### Fade correction — the other half

| Method | PSNR (dB) | SSIM | Cast error (deg) | Chroma | RMS contrast |
|---|---:|---:|---:|---:|---:|
| None (control) | 18.189 | **0.731** | 7.718 | 18.933 | 0.0983 |
| Gray-world balance | 16.476 | 0.717 | **14.282** | 5.625 | 0.0945 |
| CLAHE on L only | 18.844 | 0.726 | 7.750 | 18.944 | 0.1471 |
| Per-channel stretch | 20.916 | 0.699 | 6.573 | 23.025 | 0.2271 |
| Stretch + CLAHE | 18.072 | 0.562 | **5.880** | 22.977 | **0.2329** |
| **Stretch + saturate** | **21.245** | 0.660 | 7.028 | **26.213** | 0.2278 |

*The originals' mean chroma is **26.214**. The recommended method lands on
**26.213**.* That is the whole justification for the one free parameter in this
project — the saturation factor is `26.214 / 23.025 = 1.139`, rounded to 1.15,
rather than turned up until it looked nice.

**Gray-world is the cautionary tale.** It takes the no-reference cast measure to
0.04° by forcing the mean pixel to grey, and in doing so lands **14.28° from the
photograph's real colour balance** — nearly double the 7.72° it started with. It
also strips the chroma from 18.9 to 5.6. A metric that cannot see the truth
rewarded a method for destroying the thing it was supposed to restore.

Note also that **no method wins every column**, and that `Stretch + CLAHE` has the
lowest cast error while having the *worst* SSIM in the table. CLAHE was measured
and dropped from the recommendation: it costs 1.4 dB and 0.04 SSIM to buy 0.7° of
cast.

### Both halves together, and the order

| Stage | PSNR whole (dB) | PSNR on damage (dB) | Cast error (deg) |
|---|---:|---:|---:|
| Damaged + faded (input) | 14.260 | 5.328 | 8.207 |
| Inpaint only | 18.139 | 17.733 | 7.720 |
| Fade correct only | 13.315 | 4.921 | 6.843 |
| **Inpaint then fade correct** | **20.817** | **18.336** | 7.096 |
| Fade correct then inpaint | 16.160 | 15.725 | 7.818 |

Two things fall out of this table:

1. **Neither half fixes the other's problem.** Inpainting alone leaves the cast
   at 7.72°; fade correction alone leaves the damage at 4.92 dB — *worse* than the
   input's 5.33 dB, because stretching the tone stretched the scratches too.
2. **Order is worth 4.7 dB.** Correcting the fade first hands the inpainter a
   higher-contrast scratch to remove. Inpaint first.

---

## Run it yourself

```bash
git clone https://github.com/hammas159/classical-computer-vision.git
cd classical-computer-vision/projects/05_old_photo_restoration
```

```bash
pip install -r ../../requirements.txt

python run.py                  # regenerate every number and figure (~3 min)
python run.py --thickness 15   # rerun the whole thing at a different damage width
streamlit run ui/app.py        # the interactive app
pytest ../..                   # 29 tests for this project, 193 for the repo
```

`run.py` rewrites `results/results.json` and `results/tables.md`. **Every number
in this README is copied from those files rather than typed**, so a changed
result changes the document.

---

## Inference: try it on your own photo

You can restore a real scan at any time, with no ground truth required:

```bash
python infer.py my_grandmothers_photo.jpg
python infer.py scan.png --out restored.png --save-mask
python infer.py scan.png --mask painted_mask.png      # you marked the damage
python infer.py scan.png --detector "Top-hat + black-hat"
python infer.py scan.png --all-methods                # compare all four
python infer.py scan.png --no-fade                    # inpaint only
```

Typical output:

```
input   : scan.jpg  1400x1050
cast    : 9.14 deg from neutral   chroma: 14.2   contrast: 0.0871
mask    : Median residual — flagged 4.83% of pixels
inpaint : Telea (fast marching)   41.2 ms
fade    : Stretch + saturate
cast    : 9.14 -> 4.02 deg
chroma  : 14.2 -> 24.6
contrast: 0.0871 -> 0.2104
wrote   : restored.png
```

**No PSNR is reported and none should be** — your photograph has no undamaged
original to compare against. Everything printed (cast, chroma, contrast, flagged
fraction) needs no reference.

`infer.py` also warns you when the numbers say something actionable: if the
detector flagged over 25% of the image it is responding to texture rather than
damage; if it flagged under 0.5% the damage is probably wider than the 21 px
median window and no detector setting will find it. In both cases painting a mask
and passing `--mask` removes the variable that this project measured as being
worth 14 dB.

The UI's "Upload your own damaged scan" mode does the same thing interactively,
and greys out every metric that would need a ground truth.

---

## How it works

### The degradation model

Fading is applied in the order time applies it, and the split matters:

```
1. dye loss         f = gray + s·(f − gray)      s = 0.72   ← mixes CHANNELS
2. unequal fading   f = f · [0.90, 0.82, 0.62]              ← diagonal
3. density collapse f = f · 0.62                            ← diagonal
4. paper yellows    f = f + 0.22·[1.00, 0.94, 0.76]         ← diagonal
```

Steps 2–4 are diagonal — independent per channel — so a per-channel percentile
stretch inverts them. **Step 1 is not.** Desaturation is a rank-reducing mix
across channels, and no per-channel curve can undo it, which is exactly why the
recommended correction needs a second, chromatic term. The measured consequence:
per-channel stretch alone recovers chroma to 23.0 against the original's 26.2 and
stops there.

Cyan dye is the least stable of the three layers, then magenta, then yellow —
hence `[0.90, 0.82, 0.62]` and hence the warm drift of every surviving colour
print.

### The four inpainting methods

| Method | Idea | Fails when |
|---|---|---|
| Telea | fast marching inward from the boundary; distance- and gradient-weighted average | wide holes — "smooth" becomes "blurred" |
| Navier-Stokes | continue isophotes across the gap, borrowing incompressible-flow maths | no edge genuinely continues across |
| Iterative masked mean | peel one ring at a time, averaging **known pixels only** | never wins, never collapses |
| Harmonic diffusion | Laplace equation by Jacobi iteration | wide holes — the solution is a smooth surface with *no texture* |

### Scoring on the damaged pixels only

```python
sel = mask > 0
mse = mean((pred[sel] − truth[sel])²)
```

Whole-image PSNR on a photograph with 7% damage is dominated by the 93% that was
never touched. The damaged input scores 17.36 dB whole-image and 6.03 dB on the
damage — and the second number is the one that moves when a method works.

---

## Problems hit, and how they were solved

Every entry below is a real defect in this project's own code, with the symptom
that exposed it and the measurement that confirmed the fix.

### 1 · The simple baseline averaged the scratch into its own replacement — twice

**Symptom.** `Iterative median` scored **3.867 dB** on the damaged pixels.
Leaving the damage completely untouched scores 6.03 dB, so the "restoration" was
*worse than doing nothing* — and it had a plausible-looking implementation and a
passing shape test.

**Cause, version 1** — `src/restoration.py`, filling every damaged pixel at once:

```python
blurred = cv2.medianBlur(out, ksize)
fill = remaining.astype(bool)
out[fill] = blurred[fill]        # the hole's CENTRE is filled from its own damage
remaining = cv2.erode(remaining, np.ones((3, 3), np.uint8))
```

**Cause, version 2** — peeling rings, which fixed the ordering and not the
arithmetic. Still 6.44 dB, because `cv2.medianBlur` has no idea which pixels are
damaged: a 3 px scratch through a 7×7 window occupies 21 of 49 samples, and a
median with 43% of its input at one extreme is dragged most of the way there.

**Fix** — [`src/restoration.py:109-112`](src/restoration.py#L109-L112) —
normalized convolution. Divide by the number of **known** pixels in the window,
not by the window size:

```python
weight = box(known)                      # how many known pixels in the window
num = box(out * known[..., None])
fill = num / np.maximum(weight, 1.0)[..., None]
```

**Result: 6.44 dB → 25.77 dB**, and the baseline went from embarrassing to 96% of
the best method — which is what made "the method barely matters" a defensible
claim instead of a cherry-picked one. Pinned by
`test_the_masked_mean_must_not_average_in_the_damage`, which reconstructs the
broken version and asserts a 10 dB gap.

### 2 · The damage generator handed the detectors the answer

**Symptom.** A plain intensity threshold recovered the damage mask at **0.68
IoU** and beat the shape-aware morphological detector (0.47). That ordering is
the opposite of what morphology is for, and I nearly wrote it up as a finding.

**Cause** — `shared/synth.py`, one line:

```python
damaged[mask > 0] = 255        # every damaged pixel is EXACTLY 255
```

The rule `pixel >= 250` was therefore a near-perfect mask oracle. The whole
detection experiment was measuring the generator.

**Fix** — [`shared/synth.py:337-362`](../../shared/synth.py#L337-L362). Damage is
now a per-stroke brightness, jittered per pixel, and one stroke in five is a
*dark* crease rather than a bright emulsion scratch:

```python
level = float(rng.integers(20, 70) if rng.random() < 0.2 else rng.integers(195, 256))
...
damaged[sel] = level[sel, None].astype(np.uint8)
```

**Result:** the cheat's IoU fell from 0.68 to **0.14**, the three detectors
separated on recall instead of on an artefact, and the precision/recall asymmetry
— the project's second finding — became visible. Pinned by
`test_damage_is_not_detectable_by_brightness_alone` in the project suite and
`test_scratch_damage_is_not_one_constant_value` in the shared suite.

### 3 · The median-residual detector found 20% of the damage and reported nothing wrong

**Symptom.** A new detector, written to be the good one, scored **0.022 IoU** —
recall 3.3%. No exception, no warning; it returned a nearly empty mask and the
pipeline restored nothing.

**Cause.** `cv2.absdiff(g, cv2.medianBlur(g, 5))` with **3 px damage in a 5 px
window**. The scratch occupied 15 of 25 samples, so the median *was* the scratch
and the residual was zero. The detector was blindest exactly where the damage was
densest.

**Fix** — [`src/restoration.py:186`](src/restoration.py#L186), with the sweep
that chose it rather than a guess:

```python
MEDIAN_RESIDUAL_KSIZE = 21
```

**Result: recall 0.199 → 0.776.** More usefully, the failure turned into
[`sweep_detector_window`](src/restoration.py) and the results table above, because
"this filter only rejects a minority of outliers" is the same constraint the
whole project turns on, and a cliff is more convincing than a sentence.

### 4 · The fade model made the question unanswerable

**Symptom.** Every fade correction scored 12–18 dB with SSIM falling as PSNR
rose. Nothing could recover the print, and no method was clearly right.

**Cause.** The first `fade_photo` converted the image to greyscale and re-tinted
it:

```python
gray = f @ np.array([0.299, 0.587, 0.114], np.float32)
tone = np.stack([gray * 1.07, gray * 0.94, gray * 0.74], axis=-1)
f = (1.0 - sepia) * f + sepia * tone      # sepia = 0.55
```

At 55% strength this discards most of the chroma *irreversibly*. The experiment
was measuring how well six methods could invert something no per-pixel method can
invert, which is a question with only one answer.

**Fix** — [`shared/synth.py:368-404`](../../shared/synth.py#L368-L404). The model
was rebuilt around what actually happens to a print, with the invertible and
non-invertible parts **separated on purpose**: one desaturation term that no
diagonal method can undo, and three diagonal terms that a per-channel stretch
undoes exactly.

```python
FADE_GAIN = np.array([0.90, 0.82, 0.62], np.float32)   # cyan dies first
...
f = gray + saturation * (f - gray)          # 1. dye loss
```

**Result:** the input went from 12.42 dB to 18.19 dB (recoverable rather than
destroyed), the methods separated by 4.8 dB instead of 2, and the model now makes
a *prediction* — a stretch must plateau below the original's chroma — that
`test_fading_is_reversible_only_up_to_its_non_diagonal_part` checks.

### 5 · The saturation factor was set by eye and overshot

**Symptom.** With `factor = 1.45`, restored chroma reached **32.5** against the
originals' **26.2**. It looked vivid. It was 24% too vivid, and PSNR was buying
that at the cost of SSIM.

**Fix** — [`src/restoration.py:276`](src/restoration.py#L276), derived rather
than chosen:

```python
# 26.214 / 23.025 = 1.139, rounded to 1.15 -> restored chroma lands on 26.21
SATURATION_FACTOR = 1.15
```

**Result:** 26.213 against a target of 26.214. Pinned by
`test_the_saturation_factor_lands_on_the_originals_chroma`, which asserts the
match to 5% rather than asserting the constant — so the test still means
something if the test images change.

### 6 · Renaming a method broke the pipeline at runtime, not at import

**Symptom.** `run.py` completed six experiments and every table, then died at the
last figure:

```
File "run.py", line 152, in main
    detected, inpainted, restored = rs.restore(both)
KeyError: 'Median residual (MAD)'
```

**Cause.** `DETECTORS` was re-keyed from `"Median residual (MAD)"` to
`"Median residual"`, but the default argument of `restore()` still held the old
string. A dict lookup on a default argument fails only when that code path runs —
here, four minutes in.

**Fix** — [`src/restoration.py:633`](src/restoration.py#L633):

```python
detector: str = "Median residual",
```

**Result:** and a test that makes the class of bug loud —
`test_unknown_names_raise_rather_than_silently_defaulting` asserts every dispatch
path raises `KeyError` on a bad name rather than quietly falling back to a
default, because a silent fallback here would have produced numbers for the wrong
method.

### 7 · A test asserted something mathematically impossible

**Symptom.**

```
assert 15.31571773747149 > 15.31571789523846
```

I had asserted that damage-only PSNR *improves more* than whole-image PSNR when a
method works.

**Cause.** It cannot. An inpainter that leaves undamaged pixels alone changes the
same total squared error in both sums, so the two dB *gaps* are identical to
seven decimal places by construction. Fifteen digits of agreement is a proof, not
a flake.

**Fix** — the point was always about the *level*, not the improvement, so the test
now asserts what is actually true and actually surprising: the untouched damaged
input scores **> 15 dB** whole-image while scoring **< 8 dB** on the pixels that
changed. Same argument, correct statement, and the docstring records why the
first version was wrong so nobody re-adds it.

### 8 · `comparison_matrix(key=...)`

**Symptom.** `TypeError: comparison_matrix() got an unexpected keyword argument 'key'`.

**Cause.** The shared helper's parameter is `row_key`; `key` is what I guessed
from memory.

**Fix** — [`run.py:232`](run.py#L232), `row_key="detector"`. Trivial, and logged
here because it is the honest majority of debugging time: a parameter name I did
not look up.

---

## Limitations

* **The damage is synthetic.** Real scratches have soft, anti-aliased edges and
  partial transparency; these have hard boundaries so the mask can be exact. That
  trade buys the 14 dB detection-tax measurement and costs realism at the
  boundary — a real detector faces a harder problem than the one measured here.
* **`add_damage_and_fade` fades first, then scratches.** That is the physical
  order (dye loss over decades, damage inflicted on the surviving print), but it
  means the scratch values are *not* faded, which makes them very slightly easier
  to find than a scratch that has itself aged.
* **No texture synthesis.** Every method here interpolates. Past ~15 px the
  honest answer is that the information is gone, and exemplar-based inpainting
  (Criminisi et al.) — which copies texture from elsewhere in the image — is the
  next thing to try. It would not raise PSNR; it would raise *plausibility*, and
  measuring that needs project 26's machinery, not this one's.
* **Global fade only.** A real album print fades unevenly — worse at the edges,
  worse where light reached it. A single global stretch cannot follow that.
* **The stretch percentile has no single right value.** It is a robustness dial,
  and the two halves of the project want it set in opposite directions. On a
  cleanly faded print, `0.5/99.5` scores **22.38 dB** against `1/99`'s 21.25,
  because a tighter percentile uses more of the print's real range. Run
  end-to-end, where the detector missed ~22% of the damage and those pixels are
  now setting the black and white points, `2/98` scores **18.05 dB** against
  `1/99`'s 17.47. The shipped default is the compromise, and the measured table
  is in the `correct_channel_stretch` docstring rather than hidden in a constant.
* **The colour metrics assume the original's balance was correct.** For a
  photograph that was badly white-balanced in 1974, "restore it to what it was"
  and "make it look right" are different targets, and this project optimises the
  first.

---

## Keywords

image inpainting, old photo restoration, scratch removal, photo repair, damage
detection, Telea inpainting, fast marching method, Navier-Stokes inpainting,
harmonic inpainting, Laplace equation, normalized convolution, masked
convolution, morphological top-hat, black-hat transform, median residual
detection, MAD threshold, Otsu thresholding, colour cast correction, gray-world
white balance, per-channel histogram stretch, CLAHE, LAB colour space, chroma
restoration, dye fading, sepia, photo colourisation alternatives, PSNR, SSIM,
IoU, Dice, precision recall asymmetry, ground truth mask, oracle ceiling,
classical computer vision, OpenCV, Python, no deep learning, CPU only, Streamlit,
image restoration without neural networks

---

## See also

* [`PROJECT.md`](PROJECT.md) — the complete workflow: build order, every decision
  and what it cost.
* [Project 04 · Dehazing](../04_dehazing) — the other project in this repo built
  around an oracle ceiling.
* [Project 26 · Quality metrics](../26_quality_metrics) — takes the
  PSNR-versus-SSIM disagreement seen here as its own subject.
* [Repo index](../../README.md)
