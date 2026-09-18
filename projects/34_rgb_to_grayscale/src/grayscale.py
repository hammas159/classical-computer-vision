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


def matched_luma_colour(bg: np.ndarray, target_luma: float, weights,
                        grid: int = 48) -> np.ndarray:
    """The most different in-gamut colour with the requested luma.

    Solving one channel algebraically is the obvious way to do this and it is
    wrong: for BT.709 the required red comes out **negative**, gets clipped to
    zero, and the pair is no longer isoluminant at all — which made BT.709 score
    29.7 grey levels against a scene that was supposed to be invisible to it.
    Clipping a colour silently changes its luma.

    Searching the cube instead keeps every candidate in gamut by construction,
    and picks the one furthest from the background so the boundary is as obvious
    to the eye as the constraint allows.
    """
    w = np.array(weights, np.float32)
    axis = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    candidates = np.stack([r, g, b], axis=-1).reshape(-1, 3)

    luma = candidates @ w
    tolerance = 0.5 / 255.0
    close = np.abs(luma - target_luma) <= tolerance
    if not close.any():  # widen until something is in gamut at this luma
        close = np.abs(luma - target_luma) <= float(np.abs(luma - target_luma).min()) + EPS
    usable = candidates[close]

    distance = np.linalg.norm(usable - bg[None, :], axis=1)
    return usable[int(np.argmax(distance))].astype(np.float32)


def isoluminant_scene(size: int = 384, weights=BT601, seed: int = 0,
                      luma_offset: float = 0.0):
    """A shape that is obvious in colour and **exactly invisible** under ``weights``.

    The background and foreground colours are chosen to have identical luma under
    the given weights while differing strongly in hue. Any conversion using those
    weights must return a flat image; a contrast-preserving one must not.

    ``luma_offset`` (in 0-255 grey levels) deliberately breaks the isoluminance
    by that much. It is what turns a party trick into a measurement: at offset 0
    the failure is true *by construction* and proves nothing, but sweeping the
    offset says how close to isoluminant a real boundary would have to be before
    the conversion starts costing anything.

    Returns ``(image, mask)``. The mask is the shape that *should* be detectable.
    """
    rng = np.random.default_rng(seed)

    bg = np.array([0.85, 0.20, 0.20], np.float32)
    target_luma = float(bg @ np.array(weights, np.float32)) + luma_offset / 255.0
    fg = matched_luma_colour(bg, target_luma, weights)

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


#: A colour edge is "strong" if its colour-gradient magnitude is in the top
#: 100 - `EDGE_PERCENTILE` percent of the image.
EDGE_PERCENTILE = 97.0

#: How far from a Canny edge a colour-edge pixel may be and still count as found.
#: The same 2 px tolerance the BSDS boundary protocol uses elsewhere in this repo.
EDGE_TOLERANCE = 2


def colour_edge_recall(gray: np.ndarray, colour: np.ndarray) -> float:
    """Of the image's strongest **colour** edges, how many survive into Canny.

    `contrast_retained` is a correlation over every pixel, and on a photograph
    every method here scores 0.996 or better — it cannot tell them apart,
    because almost all of a photograph is ordinary luminance structure that
    survives any projection. The pixels where the choice matters are rare, and a
    whole-image mean drowns them.

    This asks the question the way a downstream stage would: run Canny on the
    grayscale, and check whether each strong colour edge is within 2 px of
    something it found. It is the real-photograph counterpart of the synthetic
    isoluminant test, and unlike that test it is not true by construction.
    """
    ref = _colour_gradient(to_float(colour))
    strong = ref >= float(np.percentile(ref, EDGE_PERCENTILE))
    if not strong.any():
        return 1.0

    found = cv2.Canny(gray, 50, 150)
    if not found.any():
        return 0.0
    distance = cv2.distanceTransform((found == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float((distance[strong] <= EDGE_TOLERANCE).mean())


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


#: Twelve photographs selected by `tools/select_images.py --axis colour`, which
#: measures mean chroma. Colour is the axis this project is about: a conversion
#: that discards it can only be wrong where there is some to discard, so the pool
#: has to span the range rather than sit at one end of it.
#:
#: The spread is 9.7 to 112.4 — a penguin on grey pebbles at one end, a yellow
#: roller coaster against a deep blue sky at the other. The near-monochrome
#: photograph is the control: on it, every method here must agree, and if one
#: does not the difference is not about colour.
IMAGES = (
    "penguin_on_pebbles",     # colour   9.7  — nearly monochrome already
    "teotihuacan_pyramids",   #         22.2
    "helicopter_and_pilot",   #         26.6
    "leopard_along_branch",   #         29.7
    "milking_the_cow",        #         32.4
    "black_bear_wading",      #         35.4
    "ducks_in_reeds",         #         38.2
    "fox_and_daisies",        #         42.3
    "beached_boats",          #         46.9
    "damselfly_on_leaf",      #         54.5  — blue-green on green, near isoluminant
    "runners_on_track",       #         64.3
    "roller_coaster_loop",    #        112.4  — the most colourful in the pool
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs, RGB.

    Named so that `run.py`, the tests and `infer.py` all read the same pixels.
    """
    from shared import io

    return io.real_photo(name)


def chroma(img: np.ndarray) -> float:
    """Mean distance from the grey axis, in 0-255 units.

    The axis the pool was selected on, recomputed here so the README's numbers
    come from the project rather than from the selection tool.
    """
    f = img.astype(np.float32)
    grey = f.mean(axis=2, keepdims=True)
    return float(np.sqrt(((f - grey) ** 2).sum(axis=2)).mean())


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_on_photos(images=IMAGES, runs: int = 5):
    """How much do the conversions differ on ordinary photographs?

    Expected answer: barely. Reporting the mean absolute difference from the
    OpenCV default in grey levels makes "barely" a number, and sets up the
    contrast with the isoluminant case.
    """
    rows = []
    reference_name = "BT.601 (OpenCV default)"
    for name, fn in METHODS.items():
        diffs, retained, lost, ms = [], [], [], []
        for image in images:
            img = load_scene(image)
            out, timing = timeit(lambda f=fn, a=img: f(a), runs=runs, warmup=1)
            ref = METHODS[reference_name](img)
            diffs.append(float(np.mean(np.abs(out.astype(np.float64) - ref.astype(np.float64)))))
            retained.append(contrast_retained(out, img))
            lost.append(colour_edge_recall(out, img))
            ms.append(timing.median_ms)
        rows.append(
            {
                "method": name,
                "mean_diff_vs_bt601": round(float(np.mean(diffs)), 3),
                "contrast_retained": round(float(np.mean(retained)), 4),
                "colour_edge_recall": round(float(np.mean(lost)), 4),
                "worst_image_recall": round(float(np.min(lost)), 4),
                "median_ms": round(float(np.median(ms)), 3),
            }
        )
    return rows


def per_image_colour_edge_recall(images=IMAGES):
    """Colour-edge recall per photograph — where the conversion choice bites.

    Reported per image rather than averaged, because the whole question is
    whether this matters *sometimes* rather than on average. An average over
    twelve photographs of which one is adversarial reads as "it never matters".
    """
    rows = []
    for image in images:
        img = load_scene(image)
        row: dict[str, float | str] = {"image": image, "chroma": round(chroma(img), 1)}
        for name, fn in METHODS.items():
            row[name] = round(colour_edge_recall(fn(img), img), 4)
        rows.append(row)
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


#: Grey levels of luma difference to break the isoluminance by. The interesting
#: range is small: Canny's default low threshold is 50 on a Sobel magnitude, so
#: a step edge needs only a few levels to be found.
LUMA_OFFSETS = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)

#: The three conversions that are genuinely a fixed linear weighting of R, G, B,
#: and so have an isoluminant plane that can be solved for. `Value` and
#: `Contrast-preserving` are not linear maps and `Linear-light` applies its
#: weights after a gamma decode, so no scene can be constructed against them
#: this way — they appear as columns in the matrix below but not as rows.
LINEAR_WEIGHTINGS = {
    "Average (R+G+B)/3": (1 / 3, 1 / 3, 1 / 3),
    "BT.601 (OpenCV default)": BT601,
    "BT.709 (HDTV)": BT709,
}


def blind_spot_matrix(seeds=(0, 1, 2)):
    """Build a scene invisible to each linear conversion, and score all six.

    This is the project's central claim made falsifiable. Every fixed set of
    weights is a projection onto a line, so every one of them has a whole plane
    of colours it maps to a single grey — the choice of weights moves the blind
    plane, it does not remove it. The diagonal of this matrix is where each
    conversion meets its own blind plane; the off-diagonal says whether anyone
    else can see what it missed.

    Rows are the conversion the scene was *built against*, columns are the
    conversion used to *read* it, cells are region separation in grey levels.
    """
    rows = []
    for built_for, weights in LINEAR_WEIGHTINGS.items():
        row: dict[str, float | str] = {"built_against": built_for}
        for reader, fn in METHODS.items():
            scores = []
            for seed in seeds:
                img, mask = isoluminant_scene(weights=weights, seed=seed)
                scores.append(region_separation(fn(img), mask))
            row[reader] = round(float(np.mean(scores)), 3)
        rows.append(row)
    return rows


def sweep_isoluminance(offsets=LUMA_OFFSETS, seeds=(0, 1, 2)):
    """How close to isoluminant does a boundary have to be before it is lost?

    This is the question the synthetic scene exists to answer. At offset 0 the
    BT.601 failure is true by construction and says nothing about photographs;
    the sweep turns it into a threshold that can be compared against what real
    images actually contain.
    """
    from shared.metrics import iou

    rows = []
    for offset in offsets:
        row: dict[str, float] = {"luma_offset": offset}
        for name, fn in METHODS.items():
            separations, recalls, ious = [], [], []
            for seed in seeds:
                img, mask = isoluminant_scene(seed=seed, luma_offset=offset)
                gray = fn(img)
                separations.append(region_separation(gray, mask))

                truth = cv2.Canny(mask, 50, 150)
                found = cv2.Canny(gray, 50, 150)
                if found.any() and truth.any():
                    dist = cv2.distanceTransform((found == 0).astype(np.uint8),
                                                 cv2.DIST_L2, 3)
                    recalls.append(float((dist[truth > 0] <= EDGE_TOLERANCE).mean()))
                else:
                    recalls.append(0.0)

                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ious.append(max(iou(binary, mask), iou(255 - binary, mask)))

            row[name] = round(float(np.mean(recalls)), 4)
            row[f"{name}__separation"] = round(float(np.mean(separations)), 3)
            row[f"{name}__otsu_iou"] = round(float(np.mean(ious)), 4)
        rows.append(row)
    return rows


def luma_retention(images=IMAGES, weights=BT601, percentile: float = 1.0):
    """How much of each photograph's strongest colour contrast survives the projection.

    For every pixel in the top 3% of colour gradient, the ratio of its grayscale
    gradient to its colour gradient (scaled so an achromatic edge gives 1.0).
    The low percentile is the point: the *worst* colour edge in the picture is
    what an isoluminance argument is about, and an average over 150,000 pixels
    cannot see it.
    """
    w = np.array(weights, np.float32)
    rows = []
    for image in images:
        img = load_scene(image)
        ref = _colour_gradient(to_float(img))
        strong = ref >= float(np.percentile(ref, EDGE_PERCENTILE))
        g = to_float(img) @ w
        mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3) ** 2
                      + cv2.Sobel(g, cv2.CV_32F, 0, 1, 3) ** 2)
        ratio = mag[strong] / np.maximum(ref[strong], EPS) * np.sqrt(3.0)
        rows.append({
            "image": image,
            "chroma": round(chroma(img), 1),
            "strong_edges": int(strong.sum()),
            "worst_retained": round(float(np.percentile(ratio, percentile)), 4),
            "median_retained": round(float(np.median(ratio)), 4),
            "minimum_retained": round(float(ratio.min()), 4),
        })
    return rows


def weight_sensitivity(images=IMAGES, steps: int = 9):
    """How much does moving the weights actually change the output?

    Sweeps the green weight from 0 to 1 with red and blue sharing the rest, and
    reports the mean grey shift. It puts a bound on how much the BT.601-versus-709
    argument can possibly be worth on real photographs.
    """
    rows = []
    for wg in np.linspace(0.0, 1.0, steps):
        rest = (1.0 - wg) / 2.0
        w = np.array([rest, wg, rest], np.float32)
        diffs = []
        for image in images:
            img = load_scene(image)
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
