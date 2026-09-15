"""Coin counting and measurement: segment touching objects, then calibrate to mm.

The question
------------
`skimage.data.coins` is the classic touching-objects image, and counting them is
the classic watershed demo. The demo usually stops at "it found the coins".

> **Two harder questions:** how many coins does each method *actually* get right,
> and once you have them, can you measure their diameter in millimetres?

Counting is objectively scoreable — a coin count is right or wrong, no metric
choice required. Measurement needs a calibration: one known reference object in
the frame sets the pixels-per-mm scale, and every other coin is then measured
against it. That converts "looks segmented" into a number with a unit.

Segmentation of *touching* objects is the real difficulty. A simple threshold
merges neighbouring coins into one blob; separating them needs either markers
(watershed) or a shape prior (Hough circles).
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray

EPS = 1e-6

#: `skimage.data.coins` contains this many coins. Counted by hand from the
#: image once, and used as the ground truth throughout — the dataset ships no
#: annotation, so this is the one hand-made number in the project and it is
#: flagged as such rather than buried.
TRUE_COIN_COUNT = 24


# --------------------------------------------------------------------------- #
# segmentation methods
# --------------------------------------------------------------------------- #


def _clean_binary(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    """Remove speckle and fill pinholes before any counting is attempted."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))


def segment_otsu_components(img: np.ndarray) -> np.ndarray:
    """Otsu threshold then connected components — the naive baseline.

    Expected to *undercount*, because two touching coins form one connected
    component. The size of that undercount is a measurement of how much of the
    problem is thresholding and how much is separation.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _clean_binary(mask)
    n, labels = cv2.connectedComponents(mask)
    return labels


def segment_watershed(img: np.ndarray, fg_ratio: float = 0.55) -> np.ndarray:
    """Distance-transform watershed — the standard answer for touching objects.

    The distance transform peaks at each coin's centre even when the coins touch,
    so thresholding it yields one seed per coin. Watershed then floods outward
    from those seeds and puts the boundary in the pinch between them.

    ``fg_ratio`` sets how aggressively the seeds are eroded. Too low and two
    touching coins share a seed (undercount); too high and a single coin splits
    into several (overcount). It is the one knob that matters.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _clean_binary(mask)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(to_gray(img), cv2.COLOR_GRAY2BGR), markers)

    labels = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    labels[markers == -1] = 0  # watershed marks boundaries with -1
    return labels


def segment_hough_circles(img: np.ndarray, dp: float = 1.2, min_dist: int = 30) -> np.ndarray:
    """Hough circle transform — a *shape prior* rather than a region method.

    Coins are circles, so fitting circles uses information no region-based method
    has. It should therefore separate touching coins effortlessly and fail
    entirely on anything that is not round — which is the trade-off worth
    stating rather than hiding.
    """
    gray = cv2.GaussianBlur(to_gray(img), (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=120, param2=30, minRadius=12, maxRadius=45,
    )
    labels = np.zeros(gray.shape, np.int32)
    if circles is None:
        return labels
    for i, (x, y, r) in enumerate(np.round(circles[0]).astype(int), start=1):
        cv2.circle(labels, (x, y), r, int(i), -1)
    return labels


def segment_adaptive_components(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold then components — robust to uneven lighting, not to touching."""
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -8
    )
    mask = _clean_binary(mask, open_size=5, close_size=9)
    n, labels = cv2.connectedComponents(mask)
    return labels


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Otsu + components": segment_otsu_components,
    "Watershed (distance)": segment_watershed,
    "Hough circles": segment_hough_circles,
    "Adaptive + components": segment_adaptive_components,
}


# --------------------------------------------------------------------------- #
# counting and measurement
# --------------------------------------------------------------------------- #


def region_properties(labels: np.ndarray, min_area: int = 250) -> list[dict]:
    """Area, centroid and equivalent diameter for every labelled region.

    ``min_area`` discards fragments. Without it, watershed's boundary slivers and
    morphological speckle are counted as coins and the count is meaningless.
    """
    props = []
    for label in range(1, int(labels.max()) + 1):
        blob = (labels == label).astype(np.uint8)
        area = int(blob.sum())
        if area < min_area:
            continue
        m = cv2.moments(blob, binaryImage=True)
        if m["m00"] <= 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # equivalent diameter: the diameter of a circle of the same area, which
        # is far more stable than a bounding box for a roughly round object
        props.append(
            {
                "label": label,
                "area_px": area,
                "centroid": (float(cx), float(cy)),
                "diameter_px": float(2.0 * np.sqrt(area / np.pi)),
            }
        )
    return props


def count_coins(labels: np.ndarray, min_area: int = 250) -> int:
    return len(region_properties(labels, min_area))


def calibrate_mm_per_px(props: list[dict], reference_diameter_mm: float) -> float:
    """Pixels-to-millimetres from one known object.

    The **largest** detected object is taken as the reference. This is the step
    that turns a segmentation into a measurement, and it is also where the error
    compounds: every diameter in the output inherits the reference's error, so a
    5% mistake on the reference is a 5% mistake on all of them.
    """
    if not props:
        return 0.0
    largest = max(props, key=lambda p: p["area_px"])
    return reference_diameter_mm / max(largest["diameter_px"], EPS)


def measure_mm(props: list[dict], mm_per_px: float) -> list[float]:
    return [p["diameter_px"] * mm_per_px for p in props]


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Diameter in mm assumed for the largest coin, used only to fix the scale.
REFERENCE_DIAMETER_MM = 24.25


def evaluate_methods(runs: int = 3) -> list[dict]:
    """Count coins with every method and compare against the hand-counted truth."""
    from shared import io

    img = io.sample("coins")
    rows = []
    for name, fn in METHODS.items():
        labels, timing = timeit(lambda f=fn: f(img), runs=runs, warmup=1)
        props = region_properties(labels)
        diameters = [p["diameter_px"] for p in props]
        mm_per_px = calibrate_mm_per_px(props, REFERENCE_DIAMETER_MM)
        mm = measure_mm(props, mm_per_px)
        rows.append(
            {
                "method": name,
                "count": len(props),
                "true_count": TRUE_COIN_COUNT,
                "count_error": len(props) - TRUE_COIN_COUNT,
                "abs_count_error": abs(len(props) - TRUE_COIN_COUNT),
                "mean_diameter_px": round(float(np.mean(diameters)), 2) if diameters else None,
                "mean_diameter_mm": round(float(np.mean(mm)), 2) if mm else None,
                "diameter_spread_mm": round(float(np.std(mm)), 2) if mm else None,
                "median_ms": round(float(timing.median_ms), 3),
            }
        )
    return rows


def sweep_watershed_seed(ratios=(0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8)) -> list[dict]:
    """The one knob that decides watershed's count.

    Below some ratio touching coins share a seed and the count collapses; above
    it single coins fragment and the count explodes. Locating that window is more
    useful than quoting one tuned number.
    """
    from shared import io

    img = io.sample("coins")
    rows = []
    for r in ratios:
        labels = segment_watershed(img, fg_ratio=r)
        n = count_coins(labels)
        rows.append(
            {
                "fg_ratio": r,
                "count": n,
                "error": n - TRUE_COIN_COUNT,
            }
        )
    return rows


def analyse(img: np.ndarray, method: str = "Watershed (distance)"):
    """Full pipeline for the UI: labels, per-coin properties and mm measurements."""
    labels = METHODS[method](img)
    props = region_properties(labels)
    mm_per_px = calibrate_mm_per_px(props, REFERENCE_DIAMETER_MM)
    for p in props:
        p["diameter_mm"] = p["diameter_px"] * mm_per_px
    return labels, props, mm_per_px
