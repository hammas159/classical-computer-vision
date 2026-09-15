"""Region segmentation: five ways to group pixels, and what each one assumes.

The question
------------
Watershed, region growing, mean-shift, SLIC and GrabCut all "segment". They are
not variations on one idea — each optimises a different thing, and the right
question is not which wins but which assumption matches your image.

> **The claim under test:** watershed **over-segments** without markers, and the
> amount is measurable. Feeding it distance-transform markers should collapse the
> region count by an order of magnitude while *raising* accuracy — so region
> count and accuracy move in opposite directions, which is the clearest possible
> demonstration that "more regions" is not "better segmentation".

A second, separable axis: **superpixels are not a segmentation.** SLIC produces
hundreds of regions that respect boundaries but mean nothing individually. Scored
as a segmentation it looks terrible; scored as a *boundary-preserving
over-segmentation* — how well its edges align with true edges — it is excellent.
Both numbers are reported, because quoting only the first is how superpixels get
unfairly dismissed.
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
# the methods — each returns an int32 label image, 0 = unlabelled
# --------------------------------------------------------------------------- #


def seg_watershed_unmarked(img: np.ndarray) -> np.ndarray:
    """Watershed flooded from *every* local minimum. The over-segmentation demo.

    With no markers, each local minimum of the gradient becomes its own basin.
    Noise creates minima, so the region count tracks the noise rather than the
    scene.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    markers = np.zeros(gray.shape, np.int32)
    # one marker per local minimum: erode the gradient and find where it is unchanged
    local_min = (grad == cv2.erode(grad, np.ones((3, 3), np.uint8))).astype(np.uint8)
    n, labels = cv2.connectedComponents(local_min)
    markers = labels.astype(np.int32)
    cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    return np.where(markers > 0, markers, 0).astype(np.int32)


def seg_watershed_markers(img: np.ndarray, fg_ratio: float = 0.5) -> np.ndarray:
    """Watershed seeded from a distance transform — the version that works.

    The distance transform peaks at the centre of each object even when objects
    touch, so thresholding it gives one seed per object rather than one per
    local minimum.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    return np.where(markers > 1, markers - 1, 0).astype(np.int32)


def seg_region_growing(img: np.ndarray, tolerance: int = 12, step: int = 40) -> np.ndarray:
    """Seeded region growing via flood fill on a grid of seeds.

    The oldest idea here and still instructive: grow from a seed while
    neighbouring pixels are within ``tolerance``. It is entirely local, so it
    leaks through any weak boundary — and the leak is sudden, not gradual, which
    makes ``tolerance`` unusually hard to set.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    h, w = gray.shape
    labels = np.zeros((h, w), np.int32)
    current = 0
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if labels[y, x] != 0:
                continue
            current += 1
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            ff_mask[1:-1, 1:-1] = (labels != 0).astype(np.uint8)
            cv2.floodFill(
                gray.copy(), ff_mask, (x, y), 255,
                loDiff=tolerance, upDiff=tolerance,
                flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
            )
            grown = (ff_mask[1:-1, 1:-1] == 255) & (labels == 0)
            labels[grown] = current
    return labels


def seg_mean_shift(img: np.ndarray, sp: int = 20, sr: int = 40) -> np.ndarray:
    """Mean-shift in joint spatial-colour space, then connected components.

    Makes no assumption about how many regions there are — it finds modes of the
    colour density. That is its strength and its cost: no ``k``, but two
    bandwidth parameters and a slow iteration per pixel.
    """
    filtered = cv2.pyrMeanShiftFiltering(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        sp, sr,
    )
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels = cv2.connectedComponents(binary)
    return labels.astype(np.int32)


def seg_slic(img: np.ndarray, n_segments: int = 250, compactness: float = 10.0) -> np.ndarray:
    """SLIC superpixels: k-means in 5-D (L, a, b, x, y).

    Deliberately produces far more regions than there are objects. It is an
    over-segmentation *primitive*, not a segmentation — the intended use is as
    input to something else, which is why it needs its own metric.
    """
    from skimage.segmentation import slic

    return slic(
        img, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=-1
    ).astype(np.int32)


def seg_grabcut(img: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """GrabCut from a centred rectangle.

    🚨 Seeded explicitly. OpenCV's GrabCut initialises its colour mixtures with
    k-means from the **global** RNG, so unseeded runs on the same image differ —
    measured in project 02 at up to 0.75 IoU apart. Any unseeded number here
    would be a draw from a distribution, not a measurement.
    """
    cv2.setRNGSeed(rng_seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    cv2.grabCut(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask, rect,
        np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
        5, cv2.GC_INIT_WITH_RECT,
    )
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0)
    n, labels = cv2.connectedComponents(fg.astype(np.uint8))
    return labels.astype(np.int32)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Watershed (no markers)": seg_watershed_unmarked,
    "Watershed + markers": seg_watershed_markers,
    "Region growing": seg_region_growing,
    "Mean-shift": seg_mean_shift,
    "SLIC superpixels": seg_slic,
    "GrabCut (seeded)": seg_grabcut,
}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def labels_to_foreground(labels: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Collapse a label image to the foreground guess that best matches truth.

    Needed because these methods return incompatible things: watershed returns
    objects, SLIC returns tiles, GrabCut returns one blob. Assigning each label
    to foreground when a majority of its pixels are foreground puts them all on
    one comparable footing — and is deliberately generous, so a bad score cannot
    be blamed on the conversion.
    """
    out = np.zeros(labels.shape, np.uint8)
    t = truth > 0
    for label in np.unique(labels):
        if label <= 0:
            continue
        region = labels == label
        if region.sum() and (t & region).sum() / region.sum() > 0.5:
            out[region] = 255
    return out


def boundary_recall(labels: np.ndarray, truth: np.ndarray, tolerance: int = 2) -> float:
    """Fraction of true boundary pixels lying within ``tolerance`` of a region edge.

    The correct metric for an over-segmentation: SLIC is not trying to find the
    object, it is trying not to *cross* the object's edge. This measures exactly
    that and nothing else.
    """
    truth_edges = cv2.Canny((truth > 0).astype(np.uint8) * 255, 50, 150)
    if truth_edges.sum() == 0:
        return 1.0
    seg_edges = (cv2.Laplacian(labels.astype(np.float32), cv2.CV_32F) != 0).astype(np.uint8)
    dist = cv2.distanceTransform((seg_edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist[truth_edges > 0] <= tolerance).mean())


def undersegmentation_error(labels: np.ndarray, truth: np.ndarray) -> float:
    """How much each region spills across the true boundary, normalised by area.

    The companion to boundary recall: a method can achieve perfect boundary
    recall by producing one region per pixel, and this penalises exactly that.
    """
    t = truth > 0
    total = float(t.size)
    err = 0.0
    for label in np.unique(labels):
        if label <= 0:
            continue
        region = labels == label
        inside = float((region & t).sum())
        outside = float((region & ~t).sum())
        err += min(inside, outside)
    return err / max(total, 1.0)


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #


def scene(name: str = "coins", noise_sigma: float = 0.0, seed: int = 0):
    """A bundled image plus a foreground truth mask.

    ``coins`` is the touching-objects case; ``horse`` ships with an exact
    silhouette, which is the only true annotation available without downloading
    anything.
    """
    from shared import io, synth

    if name == "horse":
        img = io.sample("horse")
        truth = (to_gray(img) < 128).astype(np.uint8) * 255
    else:
        img = io.sample(name)
        gray = to_gray(img)
        _, truth = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        truth = cv2.morphologyEx(truth, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return img, truth


IMAGES = ("coins", "horse", "cell")
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0)
SLIC_COUNTS = (50, 100, 250, 600, 1200)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_methods(images=IMAGES, noise_sigma: float = 0.0, runs: int = 1):
    """Score every method on both framings: as a segmentation and as a boundary map."""
    acc = {n: {"iou": [], "dice": [], "regions": [], "brec": [], "userr": [], "ms": []}
           for n in METHODS}

    for i, name in enumerate(images):
        img, truth = scene(name, noise_sigma=noise_sigma, seed=i)
        for method, fn in METHODS.items():
            labels, timing = timeit(lambda f=fn: f(img), runs=runs, warmup=0)
            fg = labels_to_foreground(labels, truth)
            acc[method]["iou"].append(iou(fg, truth))
            acc[method]["dice"].append(dice(fg, truth))
            acc[method]["regions"].append(int(len(np.unique(labels)) - 1))
            acc[method]["brec"].append(boundary_recall(labels, truth))
            acc[method]["userr"].append(undersegmentation_error(labels, truth))
            acc[method]["ms"].append(timing.median_ms)

    return [
        {
            "method": m,
            "iou": round(float(np.mean(a["iou"])), 4),
            "dice": round(float(np.mean(a["dice"])), 4),
            "regions": int(np.mean(a["regions"])),
            "boundary_recall": round(float(np.mean(a["brec"])), 4),
            "underseg_error": round(float(np.mean(a["userr"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for m, a in acc.items()
    ]


def sweep_watershed_markers(ratios=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), images=IMAGES):
    """The marker threshold: region count and accuracy moving in opposite directions."""
    rows = []
    for r in ratios:
        counts, ious = [], []
        for i, name in enumerate(images):
            img, truth = scene(name, seed=i)
            labels = seg_watershed_markers(img, fg_ratio=r)
            counts.append(int(len(np.unique(labels)) - 1))
            ious.append(iou(labels_to_foreground(labels, truth), truth))
        rows.append(
            {
                "fg_ratio": r,
                "regions": int(np.mean(counts)),
                "iou": round(float(np.mean(ious)), 4),
            }
        )
    return rows


def sweep_slic_count(counts=SLIC_COUNTS, images=IMAGES):
    """Superpixel count against boundary recall and under-segmentation error.

    The two metrics pull in opposite directions, which is the whole trade-off of
    an over-segmentation and is invisible if you only quote IoU.
    """
    rows = []
    for n in counts:
        brec, userr, regions = [], [], []
        for i, name in enumerate(images):
            img, truth = scene(name, seed=i)
            labels = seg_slic(img, n_segments=n)
            brec.append(boundary_recall(labels, truth))
            userr.append(undersegmentation_error(labels, truth))
            regions.append(int(len(np.unique(labels)) - 1))
        rows.append(
            {
                "requested": n,
                "actual_regions": int(np.mean(regions)),
                "boundary_recall": round(float(np.mean(brec)), 4),
                "underseg_error": round(float(np.mean(userr)), 4),
            }
        )
    return rows


def sweep_noise(levels=NOISE_LEVELS, images=IMAGES):
    """Noise creates local minima, so unmarked watershed should explode first."""
    rows = []
    for sigma in levels:
        scored = evaluate_methods(images=images, noise_sigma=sigma, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["method"]] = r["regions"]
        rows.append(row)
    return rows


def segment(img: np.ndarray, method: str = "Watershed + markers") -> np.ndarray:
    return METHODS[method](img)


def colourise(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random colour per label, for the figures."""
    rng = np.random.default_rng(seed)
    out = np.zeros((*labels.shape, 3), np.uint8)
    for label in np.unique(labels):
        if label <= 0:
            continue
        out[labels == label] = rng.integers(40, 255, 3)
    return out
