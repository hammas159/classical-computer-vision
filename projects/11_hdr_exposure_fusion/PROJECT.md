# Project 11 — HDR exposure fusion: complete workflow

The [README](README.md) states the findings. This states how they were reached,
in what order, and what each decision cost.

---

## 1 · Why this question, and not "which fusion method looks best"

Every HDR write-up ends with a picture and the word *better*. Better than what?
The comparison has no reference, because a real bracketed set does not come with
one — nobody recorded the radiance of the scene.

So the honest version of the question needs a scene whose answer is known, and
that forces the whole design:

> **Given a bracket of a scene, how close can a method get to what it would
> produce with the exact radiance?**

Everything below follows from needing that last clause to be computable.

---

## 2 · Three generators, two of them wrong

This is the part worth reading. The first two designs both ran, produced
figures, and measured nothing.

### Attempt 1 — bracket the photograph

Linearise an 8-bit photo, scale by exposure, clip, encode. Reference: the
photograph.

**Result:** the `Middle exposure only` control scored **39.8 dB**; the best
fusion method scored 20.8.

**Why:** an 8-bit photograph, linearised and re-exposed, *fits back into 8 bits*.
The middle frame clipped 1.5% of the image. There was no range to recover, so
the trivial control was simply right and the comparison was measuring nothing.

### Attempt 2 — narrow the sensor instead

Keep the photograph as the scene, and make each frame hold only 4 stops.

**Result:** the middle frame clipped **0.4%**. Worse than before.

**Why:** arithmetic. Dynamic range is saturation over noise floor. Lowering the
white level moves the window; it does not narrow it. Narrowing it means raising
the noise floor — `read_noise = 255 / 2**stops` — and even at 3 stops of range
the middle frame only lost 3.3%, because the *scene* still fitted.

The lesson generalises past this project: **you cannot make a high-dynamic-range
test out of a low-dynamic-range image without adding range somewhere.**

### Attempt 3 — widen the scene, and fix the reference

Range is added where a real scene has it: in the illumination. The photograph
supplies reflectance — real detail — and a smooth low-frequency field spanning
12 stops supplies the lighting.

That still left the reference wrong. Scored against the original photograph,
every method landed near **0.5 SSIM** and the comparison figure showed four
columns with the same bright and dark blotches. The photograph has no
illumination field, so the reference was asking every method to *remove the
lighting* — an intrinsic-image problem none of them attempts.

The fix is this repo's existing pattern: an **oracle**. A fixed global Reinhard
curve applied to the *true* radiance. Every method is then asked how close it
got to perfect information, with the tone curve held constant so the comparison
is about recovered radiance rather than anyone's taste in tone curves.

| Attempt | Middle-exposure control | Best fusion | Verdict |
|---|---:|---:|---|
| 1 · bracket the photo | 39.8 dB | 20.8 dB | no range to recover |
| 2 · narrow the sensor | — | — | range unchanged; 0.4% clipped |
| 3 · widen scene, oracle reference | 20.9 dB | 23.5 dB | answerable |

---

## 3 · Why the oracle's global curve is declared

The oracle uses `L / (1 + L)` — a global operator. Mertens blends locally. So
the reference structurally favours global behaviour, and the naive mean of the
frames is about as global as it gets.

That could have been left unsaid; the result would still be "the mean wins" and
would still be reproducible. It is stated in the README instead, because the
size of the effect matters to how the result should be read:

* it plausibly explains Mertens losing to the mean by 0.08 SSIM;
* it does **not** explain Debevec + Mantiuk losing by 0.41.

A caveat that explains part of a result is worth more than one that is used to
dismiss all of it.

---

## 4 · What each stage costs

Six scenes, five exposures:

| Stage | Time |
|---|---:|
| Middle exposure (control) | 0.002 ms |
| Mean of frames (control) | 9.3 ms |
| Mertens fusion | 19.3 ms |
| Debevec + Drago | 7.1 ms* |
| Debevec + Mantiuk | 34.6 ms* |
| Debevec + Reinhard | **1,808 ms** |

\* The three Debevec rows share one radiance map, cached after the first. Without
the cache each pays the full ~1.8 s calibration and the timing column reports
`CalibrateDebevec` rather than the tone mapper being compared — which is the
thing the column exists to measure.

---

## 5 · Findings, and how confident each is

| Finding | Evidence | Confidence |
|---|---|---|
| A single exposure of a 12-stop scene loses 29% of it | arithmetic on the clipping masks, no method involved | **certain** |
| The naive mean beats every real fusion method | 6 scenes, SSIM 0.928 vs 0.848 | **strong, with a stated metric bias** |
| One exposure beats all three Debevec pipelines | 6 scenes, every one | **strong** |
| A wider bracket makes every method worse | 6 scenes; reverses for Mertens on 2 of them | **average, not universal — said so** |
| Raw and matched PSNR pick different winners | in the table | **certain** |

The fourth row is the one to be careful with, and the test that pins it says so
in its docstring: on a two-scene subset Mertens improves with more frames. A
smaller sample would have asserted the opposite.

---

## 6 · What is deliberately absent

* **Alignment.** `cv2.createAlignMTB` exists and is unused, because nothing is
  misaligned here. With no misalignment to correct it would measure nothing, and
  including it would imply the project had tested the commonest real failure.
* **Ghost removal.** Same reason — nothing moves between frames.
* **A one-frame bracket.** `BRACKET_SIZES` starts at 2. Debevec recovers the
  camera response from how a pixel changes *across* exposures; with one there is
  nothing to fit, and OpenCV's Mantiuk operator fails outright on the degenerate
  radiance map. The one-frame case is the `Middle exposure only` control.
