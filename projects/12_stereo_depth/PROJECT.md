# Project 12 — Stereo to depth: complete workflow

The [README](README.md) states the findings. This states how they were reached,
in what order, and what each decision cost.

---

## 1 · Why this question

`cv2.StereoBM_create(...).compute(left, right)` is four lines and produces a
disparity map. Every tutorial shows the map and stops.

The map has holes in it. How the holes are counted decides which method wins,
and no tutorial counts them at all — so the question worth asking is not *which
matcher is best* but:

> **What does a disparity map get wrong, where, and what does filling the gaps
> cost?**

---

## 2 · Two kinds of ground truth, and why both

| Source | Truth | What it can answer |
|---|---|---|
| Middlebury *Aloe* | measured with structured light | what the numbers look like on a real photograph |
| generated pairs | exact by construction | *how does this change with texture, repetition, depth layers* |

One real pair is not four samples and cannot be varied — you cannot ask "what
happens with less texture" of a photograph that has the texture it has. Twelve
generated pairs can be varied on exactly that axis.

But generated pairs flatter themselves, and the Aloe pair is here to prove it:
every matcher scores **6.3 points worse** on the real pair, with **4× the mean
error**. Reporting only the generated numbers would overstate every method by
roughly a quarter. That check is the reason both are in the project.

### The forward warp had to model occlusion properly

A right view is made by shifting each pixel left by its disparity. Two source
pixels can land on the same target column, and which one wins is not arbitrary:

```python
# nearest surface wins the pixel, which is what occlusion physically is
order = np.argsort(td)          # ascending disparity
right[y, tx[order]] = img[y, sx[order]]
```

Sorting the other way puts the *far* surface in front of the near one, which
produces a right view where objects are transparent — and every matcher is then
scored against a scene that could not exist.

Pixels the right camera cannot see at all are inpainted so the view looks like a
photograph rather than a comb, and are **excluded from `valid`**. Scoring a
matcher where only one camera can see measures nothing, and including those
pixels is a common way to make stereo look worse than it is.

---

## 3 · The decision the project turns on: how to count a refusal

A matcher can return "no answer" where the matching cost has no clear minimum.
Every scoring choice here is downstream of what to do with those pixels.

```
bad2_answered   of the pixels it ANSWERED, how many are wrong   -> rewards silence
bad2_all        of the pixels the truth covers, how many are     -> punishes honesty
                wrong OR unanswered
```

Block matching answers 77% of the frame with **1.32%** of those wrong, and is
the most accurate method in the table. Count refusals as misses and it is last
at **23.6%**. Nothing about the method differs between those sentences.

Reporting one number would have been a choice about which method to favour,
disguised as a measurement. Both are reported, and
`test_density_and_accuracy_pick_different_winners` fails if a future change
makes them agree without anyone noticing.

---

## 4 · What the naive baseline was for

`match_sad_naive` is the same sum-of-absolute-differences cost `StereoBM` uses,
written out with nothing around it — no uniqueness ratio, no left-right check,
no speckle filter, no sub-pixel interpolation.

| | Bad 2px (answered) | MAE |
|---|---:|---:|
| `StereoBM` | 1.32% | 0.354 px |
| the same cost, unwrapped | 4.96% | 1.028 px |

**3.8× the error rate and 2.9× the mean error.** Almost all of the quality
attributed to "block matching" is the machinery around the cost function rather
than the cost function. That is worth knowing before writing a better cost.

---

## 5 · A finding that contradicts the textbook, and why it is not a bug

Textureless regions come out as the **easiest** of the three, at 14–19% against
20–24% in well-textured areas. The received wisdom is the opposite.

Both are true, about different things. The textbook claim is about finding a
**unique match** — and in a blank region there genuinely is none. But these
methods are not scored on unique matches; they are scored on the final
disparity, and SGBM's smoothness penalty supplies one by assuming neighbours
agree. In these scenes the low-texture regions are large flat surfaces at nearly
constant depth, so that assumption is *correct*, and the answer is right for a
reason that has nothing to do with matching.

On a textureless surface that is **not** flat — a curved white wall — this would
reverse, and the smoothness penalty would be confidently wrong. The project does
not contain that case, and the README says so rather than generalising.

---

## 6 · Errors hit

**A one-frame search range that silently truncates.** If the scene contains
disparities beyond `MAX_DISPARITY`, near objects are not reported as
out-of-range — they are reported as a *wrong disparity*, and the matcher is
charged for something the search could never have found.
`test_disparity_stays_inside_the_search_range` pins the generated scenes inside
it, and `infer.py` warns when a real pair's maximum lands at the edge:

```
disparity : 0.0 to 63.0 px (search range 0 to 64)
  -> the maximum is at the edge of the search range. Anything nearer than
     this is being truncated, not measured.
```

**Casting NaN to uint8.** The colouriser clipped the disparity and cast it, and
the unanswered pixels are NaN. `np.clip` passes NaN through, and casting NaN to
an integer is undefined — numpy warns instead of raising, so the holes came out
as whatever bit pattern the platform produced rather than as black. Replaced
before the cast, not after.

---

## 7 · Findings, and how confident each is

| Finding | Evidence | Confidence |
|---|---|---|
| Density and accuracy pick different winners | every scene, both ground truths | **certain** |
| Error concentrates at depth discontinuities | every matcher, 10–18 points | **strong** |
| The wrapper matters more than the cost | 3.8× error rate | **strong** |
| The real pair is harder than generated ones | 6.3 points, 4× MAE | **strong, one real pair** |
| Textureless is the easiest region | both ground truths | **true here, and explained — not general** |

The last row is the one to be careful with, and section 5 says why.
