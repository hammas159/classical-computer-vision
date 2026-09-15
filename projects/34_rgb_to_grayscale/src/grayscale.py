"""RGB to grayscale: five conversions, and when the choice actually matters.

The question
------------
Converting to grayscale is one line of code that nobody thinks about. It is also
a **projection from 3-D to 1-D**, which means information is destroyed, and which
weights you use decides *what* is destroyed.

> **The claim under test:** for most images the conversion is irrelevant — the
> five methods agree to within a couple of grey levels. But for images with
> **isoluminant colours** — different hues at the same luminance — the standard
> conversions collapse a visible boundary to nothing, and a contrast-preserving
> method keeps it. The question is not which is best but how often it matters.

That is testable without opinion. Generate a target that is clearly visible in
colour and *exactly* isoluminant under BT.601, then measure the contrast each
conversion retains. A method that scores zero has destroyed a boundary a human
can see.

The downstream question matters more than the conversion itself: edge detectors,
thresholding and keypoints all run on grayscale, so a conversion that loses a
boundary loses it for every method that follows.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_uint8

EPS = 1e-9

#: BT.601 luma weights — the old standard, still what cv2.COLOR_RGB2GRAY uses.
#: Derived from the phosphors of CRT televisions, which is why they persist for
#: historical rather than perceptual reasons.
BT601 = (0.299, 0.587, 0.114)

#: BT.709 — the HDTV standard, and the correct choice for modern sRGB content.
#: Green is weighted higher and red lower, reflecting measured sensitivity rather
#: than 1950s hardware.
BT709 = (0.2126, 0.7152, 0.0722)


# --------------------------------------------------------------------------- #
# the conversions
# --------------------------------------------------------------------------- #


def gray_average(img: np.ndarray) -> np.ndarray:
    """(R + G + B) / 3. Wrong, and instructive about why.

    Treats the channels as interchangeable. The eye is roughly six times more
    sensitive to green than to blue, so this over-weights blue badly — blue
    objects come out far too bright.
    """
    return to_uint8(to_float(img).mean(axis=2))


def gray_bt601(img: np.ndarray) -> np.ndarray:
    """The weights OpenCV's ``COLOR_RGB2GRAY`` uses."""
    return to_uint8(to_float(img) @ np.array(BT601, np.float32))


def gray_bt709(img: np.ndarray) -> np.ndarray:
    """The HDTV weights, correct for sRGB content."""
    return to_uint8(to_float(img) @ np.array(BT709, np.float32))


def gray_luminosity_linear(img: np.ndarray) -> np.ndarray:
    """BT.709 applied in **linear light**, which is the physically correct way.

    sRGB values are gamma-encoded. Averaging them directly averages encoded
    numbers, not light. Linearising first, mixing, then re-encoding is what a
    colour scientist would call correct — and it produces visibly different
    midtones from the usual shortcut.
    """
    f = to_float(img)
    linear = np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)
    y = linear @ np.array(BT709, np.float32)
    encoded = np.where(y <= 0.0031308, y * 12.92, 1.055 * np.power(np.maximum(y, 0), 1 / 2.4) - 0.055)
    return to_uint8(encoded)


def gray_value(img: np.ndarray) -> np.ndarray:
    """max(R, G, B) — the V of HSV.

    Not a luminance at all. It is scale-invariant per pixel, which makes it
    useful for shadow-robust work and useless as a brightness measure.
    """
    return to_float(img).max(axis=2).__mul__(255).astype(np.uint8)


def gray_decolorise(img: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Contrast-preserving decolorisation: choose weights **per image**.

    Instead of fixed weights, pick the channel mixture that best preserves the
    colour differences actually present. Implemented as a search over the weight
    simplex, maximising the correlation between the grayscale gradient magnitude
    and the full colour gradient magnitude.

    This is the only method here that can survive an isoluminant boundary,
    because it is the only one allowed to notice that the boundary exists.
    """
    f = to_float(img)
    colour_grad = _colour_gradient(f)

    best_w, best_score = np.array(BT709, np.float32), -np.inf
    for wr in np.linspace(0, 1, 11):
        for wg in np.linspace(0, 1 - wr, int(11 * (1 - wr)) + 1):
            wb = 1.0 - wr - wg
            gray = f @ np.array([wr, wg, wb], np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
            mag = np.sqrt(gx * gx + gy * gy)
            score = float(np.corrcoef(mag.ravel(), colour_grad.ravel())[0, 1])
            if np.isfinite(score) and score > best_score:
                best_score, best_w = score, np.array([wr, wg, wb], np.float32)
    return to_uint8(f @ best_w)


def _colour_gradient(f: np.ndarray) -> np.ndarray:
    """Gradient magnitude in colour space — the contrast that *should* survive.

    Summed over channels, so a boundary visible in any channel counts. This is
    the reference the conversions are scored against.
    """
    total = np.zeros(f.shape[:2], np.float32)
    for c in range(f.shape[2]):
        gx = cv2.Sobel(f[..., c], cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(f[..., c], cv2.CV_32F, 0, 1, 3)
        total += gx * gx + gy * gy
    return np.sqrt(total)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Average (R+G+B)/3": gray_average,
    "BT.601 (OpenCV default)": gray_bt601,
    "BT.709 (HDTV)": gray_bt709,
    "Linear-light BT.709": gray_luminosity_linear,
    "Value (max channel)": gray_value,
    "Contrast-preserving": gray_decolorise,
}


# --------------------------------------------------------------------------- #
# the adversarial scene
# --------------------------------------------------------------------------- #


def isoluminant_scene(size: int = 384, weights=BT601, seed: int = 0):
    """A shape that is obvious in colour and **exactly invisible** under ``weights``.

    The background and foreground colours are chosen to have identical luma under
    the given weights while differing strongly in hue. Any conversion using those
    weights must return a flat image; a contrast-preserving one must not.

    Returns ``(image, mask)``. The mask is the shape that *should* be detectable.
    """
    rng = np.random.default_rng(seed)
    wr, wg, wb = weights

    # pick a background, then solve for a foreground with the same luma
    bg = np.array([0.85, 0.20, 0.20], np.float32)
    target_luma = float(bg @ np.array(weights, np.float32))
    # fix green and blue, solve for the red that matches the luma
    g, b = 0.60, 0.35
    r = (target_luma - wg * g - wb * b) / max(wr, EPS)
    fg = np.clip(np.array([r, g, b], np.float32), 0, 1)

    img = np.zeros((size, size, 3), np.float32)
    img[:] = bg
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 4, 255, -1)
    cv2.rectangle(mask, (30, 30), (110, 110), 255, -1)
    img[mask > 0] = fg

    noise = rng.normal(0, 0.004, img.shape).astype(np.float32)
    return to_uint8(img + noise), mask


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def contrast_retained(gray: np.ndarray, colour: np.ndarray) -> float:
    """Correlation between the grayscale gradient and the colour gradient.

    1.0 means every colour boundary survived the projection; 0 means the
    conversion is blind to the image's structure.
    """
    g = to_float(gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    ref = _colour_gradient(to_float(colour))
    if mag.std() < EPS or ref.std() < EPS:
        return 0.0
    return float(np.corrcoef(mag.ravel(), ref.ravel())[0, 1])


def region_separation(gray: np.ndarray, mask: np.ndarray) -> float:
    """Mean grey difference between the masked shape and the background, 0-255.

    The direct measure for the isoluminant scene: how many grey levels separate
    a region a human sees instantly. Zero means the conversion destroyed it.
    """
    g = gray.astype(np.float64)
    inside, outside = g[mask > 0], g[mask == 0]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(abs(inside.mean() - outside.mean()))


IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "retina")


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_on_photos(images=IMAGES, runs: int = 5):
    """How much do the conversions differ on ordinary photographs?

    Expected answer: barely. Reporting the mean absolute difference from the
    OpenCV default in grey levels makes "barely" a number, and sets up the
    contrast with the isoluminant case.
    """
    from shared import io

    rows = []
    reference_name = "BT.601 (OpenCV default)"
    for name, fn in METHODS.items():
        diffs, retained, ms = [], [], []
        for image in images:
            img = io.sample(image)
            out, timing = timeit(lambda f=fn, a=img: f(a), runs=runs, warmup=1)
            ref = METHODS[reference_name](img)
            diffs.append(float(np.mean(np.abs(out.astype(np.float64) - ref.astype(np.float64)))))
            retained.append(contrast_retained(out, img))
            ms.append(timing.median_ms)
        rows.append(
            {
                "method": name,
                "mean_diff_vs_bt601": round(float(np.mean(diffs)), 3),
                "contrast_retained": round(float(np.mean(retained)), 4),
                "median_ms": round(float(np.median(ms)), 3),
            }
        )
    return rows


def evaluate_isoluminant(seeds=(0, 1, 2)):
    """The adversarial case. A conversion that scores 0 has destroyed the shape."""
    rows = []
    for name, fn in METHODS.items():
        separations, retained = [], []
        for seed in seeds:
            img, mask = isoluminant_scene(seed=seed)
            gray = fn(img)
            separations.append(region_separation(gray, mask))
            retained.append(contrast_retained(gray, img))
        rows.append(
            {
                "method": name,
                "region_separation": round(float(np.mean(separations)), 3),
                "contrast_retained": round(float(np.mean(retained)), 4),
            }
        )
    return rows


def downstream_effect(seeds=(0, 1, 2)):
    """What the conversion costs the methods that run after it.

    Edge detection and thresholding both consume grayscale. If a conversion
    flattens a boundary, every downstream method inherits the blindness — which
    is the reason this one-line choice is worth measuring at all.
    """
    from shared.metrics import iou

    rows = []
    for name, fn in METHODS.items():
        edge_recall, thresh_iou = [], []
        for seed in seeds:
            img, mask = isoluminant_scene(seed=seed)
            gray = fn(img)

            truth_edges = cv2.Canny(mask, 50, 150)
            found = cv2.Canny(gray, 50, 150)
            dist = cv2.distanceTransform((found == 0).astype(np.uint8), cv2.DIST_L2, 3)
            edge_recall.append(
                float((dist[truth_edges > 0] <= 2).mean()) if truth_edges.any() else 0.0
            )

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_iou.append(max(iou(binary, mask), iou(255 - binary, mask)))

        rows.append(
            {
                "method": name,
                "canny_edge_recall": round(float(np.mean(edge_recall)), 4),
                "otsu_iou": round(float(np.mean(thresh_iou)), 4),
            }
        )
    return rows


def weight_sensitivity(images=IMAGES, steps: int = 9):
    """How much does moving the weights actually change the output?

    Sweeps the green weight from 0 to 1 with red and blue sharing the rest, and
    reports the mean grey shift. It puts a bound on how much the BT.601-versus-709
    argument can possibly be worth on real photographs.
    """
    from shared import io

    rows = []
    for wg in np.linspace(0.0, 1.0, steps):
        rest = (1.0 - wg) / 2.0
        w = np.array([rest, wg, rest], np.float32)
        diffs = []
        for image in images:
            img = io.sample(image)
            gray = to_uint8(to_float(img) @ w)
            ref = gray_bt601(img)
            diffs.append(float(np.mean(np.abs(gray.astype(float) - ref.astype(float)))))
        rows.append(
            {
                "green_weight": round(float(wg), 3),
                "mean_diff_vs_bt601": round(float(np.mean(diffs)), 3),
            }
        )
    return rows


def convert(img: np.ndarray, method: str = "BT.601 (OpenCV default)") -> np.ndarray:
    return METHODS[method](img)
