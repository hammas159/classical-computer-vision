"""Thresholding family: six ways to pick the cut, and when each one breaks.

The question
------------
Project 01 found, as a side effect, that Otsu fails on a shadowed page well
before a global threshold becomes impossible. This project takes that seriously
as the main question:

> **For each method, what property of the image decides whether it works?**

Not "which is best" — that has no answer. Otsu is *optimal* on a clean bimodal
histogram and useless on a gradient. Sauvola is unbeatable under a gradient and
noticeably worse than Otsu on a clean scan. The useful output is the boundary
between those regimes, measured.

Three image properties are varied independently, which is what makes the answer
attributable:

* **illumination gradient** — the classic global-vs-local dividing line
* **foreground fraction** — Otsu maximises between-class variance, which is
  biased when one class is tiny
* **noise** — local methods estimate statistics from small windows, so they
  should degrade faster than global ones

An **oracle** — the best global threshold found by exhaustive search — separates
"no global cut exists" from "this method chose the wrong global cut".
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import dice, iou

EPS = 1e-6


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def thresh_fixed(gray: np.ndarray, value: int = 127) -> np.ndarray:
    """A hard-coded cut. The control: it knows nothing about the image at all."""
    return (gray > value).astype(np.uint8) * 255


def thresh_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu: maximise between-class variance over all 255 possible cuts.

    Provably optimal when the histogram really is two Gaussians of similar size.
    Both of those conditions fail often, and each failure has a different
    signature — a gradient smears one class, an imbalanced foreground biases the
    cut toward the majority.
    """
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def thresh_triangle(gray: np.ndarray) -> np.ndarray:
    """Triangle method: geometric, for a histogram with one dominant peak.

    Draws a line from the histogram peak to the far end and takes the point of
    maximum perpendicular distance. Designed for exactly the case Otsu handles
    worst — a small foreground against a large background.
    """
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return out


def thresh_multiotsu(gray: np.ndarray, classes: int = 3) -> np.ndarray:
    """Multi-Otsu with three classes, merged back to two.

    Splitting into three and then grouping the darkest class as foreground copes
    with a mid-grey band that binary Otsu has to assign wholesale to one side.
    """
    from skimage.filters import threshold_multiotsu

    try:
        cuts = threshold_multiotsu(gray, classes=classes)
    except ValueError:  # degenerate histogram, e.g. a constant image
        return thresh_otsu(gray)
    return (gray > cuts[0]).astype(np.uint8) * 255


def thresh_adaptive_mean(gray: np.ndarray, block: int = 31, c: int = 10) -> np.ndarray:
    """Local mean minus a constant. The cheapest local method."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c
    )


def thresh_adaptive_gaussian(gray: np.ndarray, block: int = 31, c: int = 10) -> np.ndarray:
    """Local Gaussian-weighted mean minus a constant."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c
    )


def thresh_niblack(gray: np.ndarray, window: int = 31, k: float = -0.2) -> np.ndarray:
    """Niblack: ``mean + k * std`` in a local window.

    Adapts to local *contrast* as well as brightness. Its known weakness is flat
    background regions: where the local std is pure noise, the rule produces
    speckle everywhere, which is precisely what Sauvola was designed to fix.
    """
    from skimage.filters import threshold_niblack

    t = threshold_niblack(gray, window_size=window, k=k)
    return (gray > t).astype(np.uint8) * 255


def thresh_sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2) -> np.ndarray:
    """Sauvola: ``mean * (1 + k * (std/R - 1))``, tuned for document images.

    The dynamic-range term ``std/R`` is the fix for Niblack: in a flat region the
    local std is small, the term pushes the threshold *down* toward the mean, and
    the background stays background instead of turning to speckle.
    """
    from skimage.filters import threshold_sauvola

    t = threshold_sauvola(gray, window_size=window, k=k)
    return (gray > t).astype(np.uint8) * 255


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Fixed 127 (control)": thresh_fixed,
    "Otsu": thresh_otsu,
    "Triangle": thresh_triangle,
    "Multi-Otsu (3 class)": thresh_multiotsu,
    "Adaptive mean": thresh_adaptive_mean,
    "Adaptive Gaussian": thresh_adaptive_gaussian,
    "Niblack": thresh_niblack,
    "Sauvola": thresh_sauvola,
}

ORACLE_NAME = "Best global (oracle)"


def thresh_best_global(gray: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, int]:
    """**Oracle**: the best single global cut, by exhaustive search.

    Needs the ground truth, so it is not a method. It answers the question every
    global-threshold failure raises and almost no comparison asks: was a global
    threshold available at all?
    """
    best_iou, best_t, best_above = -1.0, 0, True
    for t in range(1, 255):
        # BOTH polarities. "Foreground is above the cut" is a convention, not a
        # property of thresholding -- cv2.THRESH_BINARY and THRESH_BINARY_INV
        # are the same operation. This scene is dark-on-light, so searching only
        # `gray > t` cannot express the right answer at any threshold, and the
        # oracle settled on a degenerate cut scoring **0.122 IoU** -- below every
        # method it exists to place a ceiling over. A ceiling that sits under the
        # thing it bounds is worse than no ceiling: it reads as "no global
        # threshold was available" when in fact a perfect one was.
        for above in (True, False):
            mask = ((gray > t) if above else (gray < t)).astype(np.uint8) * 255
            score = iou(mask, truth)
            if score > best_iou:
                best_iou, best_t, best_above = score, t, above
    mask = ((gray > best_t) if best_above else (gray < best_t)).astype(np.uint8) * 255
    return mask, best_t


# --------------------------------------------------------------------------- #
# the test scene — three properties varied independently
# --------------------------------------------------------------------------- #


#: Stroke width, in pixels, of the "thin" scene. Deliberately well under the
#: 31 px window the local methods use — that relationship is the whole reason
#: the two scene kinds rank the methods differently.
THIN_STROKE_PX = 3


def _draw_thin_strokes(truth: np.ndarray, rng, size: int, fg_fraction: float) -> None:
    """Text-like strokes: thin, scattered, and mostly boundary.

    Local thresholding compares a pixel to its own neighbourhood, so it needs a
    neighbourhood that contains some background. A stroke narrower than the
    window always has some; the interior of a filled circle never does.
    """
    target = fg_fraction * size * size
    guard = 0
    while float((truth > 0).sum()) < target and guard < 4000:
        guard += 1
        x = int(rng.integers(8, size - 8))
        y = int(rng.integers(8, size - 8))
        length = int(rng.integers(size // 20, size // 5))
        if rng.random() < 0.5:
            cv2.line(truth, (x, y), (min(x + length, size - 1), y), 255, THIN_STROKE_PX)
        else:
            cv2.line(truth, (x, y), (x, min(y + length, size - 1)), 255, THIN_STROKE_PX)


def synthetic_scene(
    size: int = 512,
    fg_fraction: float = 0.3,
    illum_min: float = 1.0,
    noise_sigma: float = 0.0,
    fg_level: int = 60,
    bg_level: int = 200,
    kind: str = "solid",
    seed: int = 0,
):
    """Foreground of known area on a known background, with known degradations.

    Returns ``(image, truth_mask)``. The foreground fraction is *constructed*
    rather than measured, so each method's bias against class imbalance can be
    read straight off the sweep.

    ``kind`` selects what the foreground is made of, and it decides the whole
    result:

    ``"solid"``
        filled circles. A 31 px window placed inside one contains no background
        at all, so a local threshold has nothing to compare against and hollows
        the shape out — which is why the adaptive family scores 0.2-0.4 here at
        *every* illumination level, including the ones they exist for.
    ``"thin"``
        text-like strokes 3 px wide. Every window containing a stroke also
        contains background, which is the condition local thresholding is built
        on.

    The project had only ``"solid"`` at first, and concluded that adaptive
    thresholding simply loses. That conclusion was about the shape of the
    foreground and not about illumination at all.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), bg_level, np.float32)
    truth = np.zeros((size, size), np.uint8)

    if kind == "thin":
        _draw_thin_strokes(truth, rng, size, fg_fraction)
    elif kind == "solid":
        target = fg_fraction * size * size
        placed = 0.0
        guard = 0
        while placed < target and guard < 400:
            guard += 1
            r = int(rng.integers(size // 22, size // 7))
            cx = int(rng.integers(r, size - r))
            cy = int(rng.integers(r, size - r))
            before = int((truth > 0).sum())
            cv2.circle(truth, (cx, cy), r, 255, -1)
            placed += int((truth > 0).sum()) - before
    else:
        raise ValueError(f"unknown scene kind {kind!r}; choose 'solid' or 'thin'")

    img[truth > 0] = fg_level

    if illum_min < 1.0:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        ramp = 1.0 - (xx / size) * 0.75 - (yy / size) * 0.25
        img *= illum_min + (1.0 - illum_min) * ramp

    out = to_uint8(np.clip(img, 0, 255) / 255.0)
    if noise_sigma > 0:
        from shared import synth

        out = synth.gaussian_noise(out, sigma=noise_sigma, seed=seed)
    return out, truth


#: Foreground reflectance over background reflectance. A perfect global cut can
#: exist only while the illumination ratio across the image exceeds this.
CONTRAST_RATIO = 60.0 / 200.0


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

ILLUM_LEVELS = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1)
FG_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
NOISE_LEVELS = (0.0, 5.0, 12.0, 25.0, 40.0)


def evaluate_methods(
    illum_min: float = 1.0, fg_fraction: float = 0.3, noise_sigma: float = 0.0,
    seeds=(0, 1, 2), runs: int = 3,
):
    """Score every method on one scene configuration, plus the oracle."""
    acc = {n: {"iou": [], "dice": [], "ms": []} for n in list(METHODS) + [ORACLE_NAME]}
    oracle_ts = []

    for seed in seeds:
        img, truth = synthetic_scene(
            fg_fraction=fg_fraction, illum_min=illum_min, noise_sigma=noise_sigma, seed=seed
        )
        gray = to_gray(img)
        for name, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(gray), runs=runs, warmup=1)
            # foreground is DARK, so the mask is the inverse of the threshold
            pred = 255 - out
            acc[name]["iou"].append(iou(pred, truth))
            acc[name]["dice"].append(dice(pred, truth))
            acc[name]["ms"].append(timing.median_ms)

        best, t = thresh_best_global(gray, 255 - truth)
        oracle_ts.append(t)
        acc[ORACLE_NAME]["iou"].append(iou(255 - best, truth))
        acc[ORACLE_NAME]["dice"].append(dice(255 - best, truth))
        acc[ORACLE_NAME]["ms"].append(float("nan"))

    rows = []
    for name, a in acc.items():
        rows.append(
            {
                "method": name,
                "iou": round(float(np.mean(a["iou"])), 4),
                "dice": round(float(np.mean(a["dice"])), 4),
                "median_ms": (
                    None if np.isnan(np.nanmedian(a["ms"])) else round(float(np.nanmedian(a["ms"])), 3)
                ),
            }
        )
    return rows, {"oracle_threshold_mean": round(float(np.mean(oracle_ts)), 1)}


def sweep_illumination(levels=ILLUM_LEVELS, seeds=(0, 1)):
    """The global-vs-local dividing line, with the theoretical limit marked."""
    rows = []
    for level in levels:
        scored, _ = evaluate_methods(illum_min=level, seeds=seeds, runs=1)
        row: dict[str, float] = {"illum_min": level}
        for r in scored:
            row[r["method"]] = r["iou"]
        rows.append(row)
    return rows


def sweep_foreground_fraction(fractions=FG_FRACTIONS, seeds=(0, 1)):
    """Class imbalance. Otsu's criterion is biased when one class is small;
    the Triangle method was designed for that case and should overtake it."""
    rows = []
    for frac in fractions:
        scored, _ = evaluate_methods(fg_fraction=frac, seeds=seeds, runs=1)
        row: dict[str, float] = {"fg_fraction": frac}
        for r in scored:
            row[r["method"]] = r["iou"]
        rows.append(row)
    return rows


def sweep_noise(levels=NOISE_LEVELS, seeds=(0, 1)):
    """Local methods estimate statistics from small windows, so they should
    degrade faster under noise than a global histogram method does."""
    rows = []
    for sigma in levels:
        scored, _ = evaluate_methods(noise_sigma=sigma, seeds=seeds, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["method"]] = r["iou"]
        rows.append(row)
    return rows


def binarise(img: np.ndarray, method: str = "Otsu") -> np.ndarray:
    return METHODS[method](to_gray(img))
