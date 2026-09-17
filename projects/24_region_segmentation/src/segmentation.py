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


#: Region count the grid control uses. Matched to what SLIC returns at its
#: default settings so the comparison is like-for-like.
GRID_TILES = 180


def seg_grid(img: np.ndarray, n_tiles: int = GRID_TILES) -> np.ndarray:
    """**Control**: cut the image into rectangular tiles. Looks at nothing.

    This is not a segmentation method and it is in the table on purpose.

    `labels_to_foreground` assigns each region to foreground or background by
    which it overlaps more -- using the truth. That is deliberately generous, so
    that a low score cannot be blamed on the conversion, and the docstring says
    so. What it does not say is the consequence: **the more regions a method
    returns, the more the oracle assignment can do for it**, and at one region
    per pixel the score is 1.0 regardless of the method.

    A grid of rectangles that has never looked at the image scores **0.88 mean
    IoU against SLIC's 0.90**, and beats SLIC outright on two of six
    photographs. Any row of that table above the grid's is measuring region
    count, not segmentation.
    """
    h, w = img.shape[:2]
    side = max(1, int(round(np.sqrt(n_tiles * w / h))))
    rows = max(1, int(round(n_tiles / side)))
    yy = np.arange(h)[:, None] * rows // h
    xx = np.arange(w)[None, :] * side // w
    return (yy * side + xx + 1).astype(np.int32)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Grid tiles (control)": seg_grid,
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


#: Twelve photographs with **human** segmentations — five to seven annotators
#: each, from BSDS500. Selected by `tools/select_images.py --axis colour`,
#: because every method here groups pixels by appearance and colourfulness is
#: what decides whether appearance and object agree.
IMAGES = (
    "wolf_on_snowline",     # colour   9.3 — two flat regions and one small subject
    "tiger_in_shade",       # colour  21.7
    "train_on_viaduct",     # colour  26.1
    "caterpillar_on_stem",  # colour  29.7 — subject and background share a hue
    "gunner_reenactor",     # colour  32.7
    "morel_mushrooms",      # colour  36.2
    "woman_and_child",      # colour  38.9
    "fox_cubs",             # colour  43.0
    "memorial_arch",        # colour  47.2
    "florence_duomo",       # colour  54.1
    "two_beefeaters",       # colour  63.1
    "man_yellow_barrels",   # colour 124.3 — the most saturated frame in the pool
)

ORACLE_NAME = "Another human (ceiling)"


def load_scene(name: str) -> np.ndarray:
    from shared import io

    return io.real_photo(name)


def scene(name: str = "tiger_in_shade", noise_sigma: float = 0.0, seed: int = 0,
          annotator: int = 0):
    """A photograph and a **human** foreground annotation.

    Earlier versions of this used Otsu's binarisation as the target, which is
    what the rest of this repository does when no annotation exists. For a
    segmentation project that is close to circular: Otsu is itself a
    segmentation method, so every method was being scored on how well it
    reproduces one particular competitor.

    BSDS500 ships real human labellings, and they are different in kind. A tiger
    and its shadow are one region to a person and two to any clustering; a
    tiger's stripes are one region to a person and thirty to any clustering. The
    gap between the two is not noise in the annotation, it is the thing
    segmentation is hard for.
    """
    from shared import bsds, synth

    img = load_scene(name)
    truth = bsds.dominant_foreground(name, annotator=annotator)
    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return img, truth


def labels_to_boundaries(labels: np.ndarray) -> np.ndarray:
    """Where a label image changes value — a method's boundary map.

    Needed because the methods return labellings and the human annotations are
    boundaries. Converting the method rather than the annotation is the right
    direction: a labelling determines its boundaries exactly, where a boundary
    map does not determine a labelling.
    """
    a = labels.astype(np.int32)
    edge = np.zeros(a.shape, bool)
    edge[:, :-1] |= a[:, :-1] != a[:, 1:]
    edge[:-1, :] |= a[:-1, :] != a[1:, :]
    return edge


def evaluate_boundaries(images=None, min_annotators: int = 2, runs: int = 1):
    """Boundary precision, recall and F against what at least two humans agreed on.

    **This is the project's headline metric, and the IoU table is kept only to
    show why.** A human segmentation is a labelling, not a figure/ground split;
    forcing it into a binary mask produced targets that corresponded to nothing
    anyone drew, and a grid of rectangles then beat five real methods on the
    result. Boundary F is the metric BSDS was built for and the one every
    published number on it uses.
    """
    from shared import bsds

    images = images if images is not None else IMAGES
    acc = {n: {"p": [], "r": [], "f": [], "regions": [], "ms": []} for n in METHODS}

    for name in images:
        img = load_scene(name)
        target = bsds.consensus_boundaries(name, min_annotators)
        for method, fn in METHODS.items():
            labels, timing = timeit(lambda f=fn: f(img), runs=runs, warmup=0)
            score = bsds.boundary_f_measure(labels_to_boundaries(labels), target)
            acc[method]["p"].append(score["precision"])
            acc[method]["r"].append(score["recall"])
            acc[method]["f"].append(score["f"])
            acc[method]["regions"].append(int(len(np.unique(labels)) - 1))
            acc[method]["ms"].append(timing.median_ms)

    return [
        {
            "method": m,
            "precision": round(float(np.mean(a["p"])), 4),
            "recall": round(float(np.mean(a["r"])), 4),
            "f": round(float(np.mean(a["f"])), 4),
            "regions": int(np.mean(a["regions"])),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for m, a in acc.items()
    ]


def human_boundary_ceiling(name: str, min_annotators: int = 2) -> dict:
    """How well one annotator reproduces what the others agreed on.

    The ceiling, measured rather than assumed, and on the metric the table
    actually uses. Published BSDS human agreement is about 0.79 F; the numbers
    here are in that range, which is the check that the measurement is right.
    """
    from shared import bsds

    ann = bsds.load_annotations(name)
    target = bsds.consensus_boundaries(name, min_annotators)
    scores = [bsds.boundary_f_measure(a["boundaries"], target)["f"] for a in ann]
    return {
        "annotators": len(ann),
        "best": round(float(np.max(scores)), 4),
        "mean": round(float(np.mean(scores)), 4),
    }


def human_ceiling(name: str) -> dict:
    """How well one annotator reproduces another — the ceiling, measured not assumed.

    Scores annotator 0's foreground against every other annotator's and returns
    the best and the mean. No algorithm has any business scoring above the best,
    and a method that does is being flattered by the particular annotator it was
    scored against.

    This is the one place in the repository where the ceiling is a fact about
    people rather than a construction. It is also, consistently, **not close to
    1.0** -- which is the finding.
    """
    from shared import bsds

    ann = bsds.load_annotations(name)
    if len(ann) < 2:
        return {"annotators": len(ann), "best": float("nan"), "mean": float("nan")}

    reference = bsds.dominant_foreground(name, annotator=0) > 0
    scores = []
    for i in range(1, len(ann)):
        other = bsds.dominant_foreground(name, annotator=i) > 0
        scores.append(iou(other, reference))
    return {
        "annotators": len(ann),
        "best": round(float(np.max(scores)), 4),
        "mean": round(float(np.mean(scores)), 4),
    }


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
