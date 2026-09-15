"""Morphology: set operations on shape, and what the structuring element decides.

The question
------------
Erosion, dilation, opening, closing, gradient, top-hat, black-hat, skeleton,
hit-or-miss. Every tutorial lists them. Almost none measure anything.

> **The claim under test:** the *structuring element* matters more than the
> operation. A cross, a square and a disc of the same nominal size produce
> measurably different results, and the difference is largest exactly where
> people care — on diagonal and curved structures.

Morphology is one of the few areas where exact ground truth is trivially
available: these are set operations with algebraic identities that must hold.
Those identities are checked as measurements rather than assumed:

* opening is **idempotent**: ``open(open(A)) == open(A)``
* opening is **anti-extensive**: ``open(A) ⊆ A``; closing is extensive
* erosion and dilation are **dual**: ``erode(A) == complement(dilate(complement(A)))``

A violation means a bug, not a finding — which makes them the cheapest possible
correctness net for the whole project.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import dice, iou

EPS = 1e-6


# --------------------------------------------------------------------------- #
# structuring elements
# --------------------------------------------------------------------------- #


def element(shape: str, size: int) -> np.ndarray:
    """Build a structuring element by name.

    The three shapes are not interchangeable. A square (``RECT``) reaches
    ``size/2 * sqrt(2)`` along a diagonal but only ``size/2`` along an axis, so
    it is *anisotropic* — it erodes diagonal structures harder than horizontal
    ones. A disc (``ELLIPSE``) is the isotropic choice. A cross touches only four
    neighbours and preserves thin diagonal lines that either of the others
    destroy.
    """
    kinds = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    if shape not in kinds:
        raise ValueError(f"unknown structuring element {shape!r}")
    return cv2.getStructuringElement(kinds[shape], (size, size))


ELEMENTS = ("rect", "ellipse", "cross")


# --------------------------------------------------------------------------- #
# the operations
# --------------------------------------------------------------------------- #


def op_erode(img, se):
    """Shrink bright regions. Removes anything the element cannot fit inside."""
    return cv2.erode(img, se)


def op_dilate(img, se):
    """Grow bright regions. The dual of erosion."""
    return cv2.dilate(img, se)


def op_open(img, se):
    """Erode then dilate: removes small bright specks, keeps large shapes' size.

    The size restoration is the point — a plain erosion also removes specks but
    shrinks everything else too.
    """
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, se)


def op_close(img, se):
    """Dilate then erode: fills small dark holes without growing the object."""
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)


def op_gradient(img, se):
    """dilate - erode: a boundary detector with no derivatives at all.

    Responds to any intensity step regardless of sign or orientation, and needs
    no threshold pair, unlike Canny.
    """
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)


def op_tophat(img, se):
    """image - open: bright features *smaller than* the structuring element.

    This is the size-selective filter. It is how you find thin scratches on a
    bright print, or text on an unevenly lit page, without any thresholding.
    """
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)


def op_blackhat(img, se):
    """close - image: dark features smaller than the element."""
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)


OPERATIONS: dict[str, Callable] = {
    "Erode": op_erode,
    "Dilate": op_dilate,
    "Open": op_open,
    "Close": op_close,
    "Gradient": op_gradient,
    "Top-hat": op_tophat,
    "Black-hat": op_blackhat,
}


# --------------------------------------------------------------------------- #
# skeletonisation
# --------------------------------------------------------------------------- #


def skeleton_morphological(binary: np.ndarray, se_shape: str = "cross", size: int = 3):
    """Lantuejoul's morphological skeleton: repeated open-subtract.

    Simple and provably reconstructible, but it does **not** guarantee
    connectivity — the skeleton can come out in disconnected fragments, which is
    exactly why thinning algorithms exist. Measuring that disconnection is more
    useful than warning about it.
    """
    img = (binary > 0).astype(np.uint8) * 255
    se = element(se_shape, size)
    skel = np.zeros_like(img)
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = cv2.erode(img, se)
    return skel


def skeleton_thinning(binary: np.ndarray):
    """Zhang-Suen thinning: iteratively delete boundary pixels that are safe.

    "Safe" means deleting the pixel cannot break local connectivity, which is
    checked with two conditions on the 8-neighbourhood. That connectivity
    guarantee is the whole difference from the morphological skeleton.
    """
    img = (binary > 0).astype(np.uint8).copy()

    def _neighbours(y, x, im):
        return [
            im[y - 1, x], im[y - 1, x + 1], im[y, x + 1], im[y + 1, x + 1],
            im[y + 1, x], im[y + 1, x - 1], im[y, x - 1], im[y - 1, x - 1],
        ]

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            marked = []
            ys, xs = np.nonzero(img[1:-1, 1:-1])
            for y, x in zip(ys + 1, xs + 1):
                n = _neighbours(y, x, img)
                b = sum(n)
                if not (2 <= b <= 6):
                    continue
                transitions = sum((n[i] == 0 and n[(i + 1) % 8] == 1) for i in range(8))
                if transitions != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = n
                if step == 0:
                    if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                        continue
                else:
                    if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                        continue
                marked.append((y, x))
            if marked:
                changed = True
                for y, x in marked:
                    img[y, x] = 0
    return img * 255


SKELETONS: dict[str, Callable] = {
    "Morphological (open-subtract)": lambda b: skeleton_morphological(b),
    "Zhang-Suen thinning": skeleton_thinning,
}


def hit_or_miss(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Hit-or-miss: find an exact local configuration of foreground and background.

    The only morphological operator that is a true *pattern* detector — it
    requires certain pixels set AND certain pixels clear, which is how corners
    and line ends are found without any learning.
    """
    return cv2.morphologyEx((binary > 0).astype(np.uint8), cv2.MORPH_HITMISS, kernel)


#: Kernels for the four convex corners of a rectangle. 1 = must be foreground,
#: -1 = must be background, 0 = don't care.
CORNER_KERNELS = {
    "top-left": np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]], np.int8),
    "top-right": np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 0]], np.int8),
    "bottom-left": np.array([[0, 1, 0], [1, 1, -1], [0, -1, -1]], np.int8),
    "bottom-right": np.array([[0, 1, 0], [-1, 1, 1], [-1, -1, 0]], np.int8),
}


# --------------------------------------------------------------------------- #
# the test scene
# --------------------------------------------------------------------------- #


def binary_scene(size: int = 512, noise_density: float = 0.0, seed: int = 0):
    """Shapes with known geometry: axis-aligned, diagonal, curved and thin.

    Returns ``(binary, parts)`` where ``parts`` holds each structure separately,
    so an operation's effect on *diagonal* structures can be measured apart from
    its effect on axis-aligned ones. That separation is what makes the
    structuring-element claim testable.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    parts: dict[str, np.ndarray] = {}

    axis = np.zeros_like(img)
    cv2.rectangle(axis, (40, 40), (180, 180), 255, -1)
    parts["axis_aligned"] = axis

    diagonal = np.zeros_like(img)
    cv2.line(diagonal, (220, 40), (400, 220), 255, 5)
    cv2.line(diagonal, (400, 40), (220, 220), 255, 5)
    parts["diagonal"] = diagonal

    curved = np.zeros_like(img)
    cv2.circle(curved, (130, 380), 70, 255, -1)
    parts["curved"] = curved

    thin = np.zeros_like(img)
    for i in range(6):
        cv2.line(thin, (280 + i * 22, 300), (280 + i * 22, 470), 255, 1 + i % 3)
    parts["thin"] = thin

    for p in parts.values():
        img = cv2.bitwise_or(img, p)

    if noise_density > 0:
        r = rng.random(img.shape)
        img[r < noise_density / 2] = 255
        img[r > 1.0 - noise_density / 2] = 0
    return img, parts


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

SIZES = (3, 5, 7, 9, 13, 21)


def verify_algebraic_identities(size: int = 5, seeds=(0, 1, 2)) -> list[dict]:
    """Check the laws morphology must obey. A failure here is a bug, not a result."""
    rows = []
    for shape in ELEMENTS:
        se = element(shape, size)
        idempotent, anti_extensive, extensive, dual = [], [], [], []
        for seed in seeds:
            img, _ = binary_scene(seed=seed, noise_density=0.02)
            opened = op_open(img, se)
            idempotent.append(bool(np.array_equal(op_open(opened, se), opened)))
            anti_extensive.append(bool(np.all((opened > 0) <= (img > 0))))
            closed = op_close(img, se)
            extensive.append(bool(np.all((img > 0) <= (closed > 0))))
            eroded = op_erode(img, se)
            dual_of = cv2.bitwise_not(op_dilate(cv2.bitwise_not(img), se))
            dual.append(bool(np.array_equal(eroded > 0, dual_of > 0)))
        rows.append(
            {
                "element": shape,
                "opening_idempotent": all(idempotent),
                "opening_anti_extensive": all(anti_extensive),
                "closing_extensive": all(extensive),
                "erode_dilate_dual": all(dual),
            }
        )
    return rows


def compare_elements(operation: str = "Erode", size: int = 9, seeds=(0, 1)) -> list[dict]:
    """How much of the result does the structuring element decide?

    Reports, per element, how much of each *kind* of structure survives — so an
    element's anisotropy shows up as a difference between the axis-aligned and
    diagonal columns rather than as a single blended number.
    """
    fn = OPERATIONS[operation]
    rows = []
    for shape in ELEMENTS:
        se = element(shape, size)
        survival = {k: [] for k in ("axis_aligned", "diagonal", "curved", "thin")}
        for seed in seeds:
            img, parts = binary_scene(seed=seed)
            out = fn(img, se)
            for key, part in parts.items():
                before = float((part > 0).sum())
                after = float(((out > 0) & (part > 0)).sum())
                survival[key].append(after / max(before, 1.0))
        rows.append(
            {
                "element": shape,
                **{k: round(float(np.mean(v)), 4) for k, v in survival.items()},
            }
        )
    return rows


def sweep_size(operation: str = "Open", sizes=SIZES, shape: str = "ellipse", seeds=(0, 1)):
    """Effect of element size on how much of the image survives."""
    fn = OPERATIONS[operation]
    rows = []
    for size in sizes:
        se = element(shape, size)
        kept = []
        for seed in seeds:
            img, _ = binary_scene(seed=seed)
            out = fn(img, se)
            kept.append(float((out > 0).sum()) / max(float((img > 0).sum()), 1.0))
        rows.append({"size": size, "fraction_kept": round(float(np.mean(kept)), 4)})
    return rows


def evaluate_denoising(density: float = 0.05, sizes=(3, 5, 7), seeds=(0, 1)):
    """Opening as a speckle remover, scored against the clean binary truth.

    A rare case where a morphological operation has an unambiguous right answer:
    the noise-free scene is known, so IoU after opening is a real score.
    """
    rows = []
    for shape in ELEMENTS:
        for size in sizes:
            se = element(shape, size)
            scores, dices = [], []
            for seed in seeds:
                clean, _ = binary_scene(seed=seed, noise_density=0.0)
                noisy, _ = binary_scene(seed=seed, noise_density=density)
                out = op_open(noisy, se)
                scores.append(iou(out, clean))
                dices.append(dice(out, clean))
            rows.append(
                {
                    "element": shape,
                    "size": size,
                    "iou": round(float(np.mean(scores)), 4),
                    "dice": round(float(np.mean(dices)), 4),
                }
            )
    return rows


def evaluate_skeletons(seeds=(0, 1), runs: int = 1):
    """Skeleton quality: connectivity and thinness, both measurable."""
    rows = []
    for name, fn in SKELETONS.items():
        comps, thinness, ms = [], [], []
        for seed in seeds:
            img, _ = binary_scene(seed=seed)
            skel, timing = timeit(lambda f=fn, a=img: f(a), runs=runs, warmup=0)
            n, _ = cv2.connectedComponents((skel > 0).astype(np.uint8))
            comps.append(n - 1)
            thinness.append(float((skel > 0).sum()) / max(float((img > 0).sum()), 1.0))
            ms.append(timing.median_ms)
        rows.append(
            {
                "method": name,
                "components": round(float(np.mean(comps)), 2),
                "thinness": round(float(np.mean(thinness)), 4),
                "median_ms": round(float(np.median(ms)), 2),
            }
        )
    return rows


def evaluate_hit_or_miss(seeds=(0,)):
    """Corner detection by exact pattern match, scored against the known count."""
    rows = []
    for seed in seeds:
        img, parts = binary_scene(seed=seed)
        rect = parts["axis_aligned"]
        for name, kernel in CORNER_KERNELS.items():
            hits = hit_or_miss(rect, kernel)
            rows.append(
                {
                    "corner": name,
                    "detections": int((hits > 0).sum()),
                    "expected": 1,
                }
            )
    return rows


def apply(img: np.ndarray, operation: str = "Open", shape: str = "ellipse", size: int = 5):
    return OPERATIONS[operation](to_gray(img), element(shape, size))
