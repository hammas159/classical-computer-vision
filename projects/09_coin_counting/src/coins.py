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
from shared.io import to_float, to_gray
from shared import synth

EPS = 1e-6

#: `skimage.data.coins` contains this many coins. Counted by hand from the
#: image once, and used as the ground truth throughout — the dataset ships no
#: annotation, so this is the one hand-made number in the project and it is
#: flagged as such rather than buried.
TRUE_COIN_COUNT = 24


# --------------------------------------------------------------------------- #
# segmentation methods
# --------------------------------------------------------------------------- #


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every enclosed hole by redrawing the outer contours solid.

    A coin is solid. Any hole inside its outline is a failure of the threshold,
    not a feature of the object — and left alone it puts a spurious local maximum
    in the distance transform, which becomes a spurious seed, which becomes a
    spurious coin.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def _clean_binary(mask: np.ndarray, open_size: int = 3, close_size: int = 11) -> np.ndarray:
    """Remove speckle, rejoin fragments, and fill pinholes.

    The closing is larger than it looks like it needs to be. Flattening the
    illumination costs contrast inside the darker coins, so their masks come back
    broken into two or three pieces — and a fragment is counted as a coin unless
    it is rejoined here. Closing then filling recovers 3 coins on this image.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    )
    return _fill_holes(mask)


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


#: Diameter of the largest expected coin, in pixels. The top-hat kernel must be
#: bigger than this or it removes the coins along with the background.
BACKGROUND_KERNEL = 61


def background_kernel_for(gray: np.ndarray, floor: int = BACKGROUND_KERNEL) -> int:
    """Pick a top-hat kernel guaranteed to be larger than the biggest object.

    :func:`_flatten_illumination` only works when the structuring element is
    **bigger than any coin**; below that, the opening fails to erase the coin and
    the top-hat subtracts the coin's own interior, leaving a ring. The
    requirement was in the docstring and nowhere else, and the constant 61 was
    sized for `skimage.data.coins`.

    The moment this project was pointed at scenes with 64 px coins, flattening
    silently deleted five of twenty — on a scene where the coins were not even
    touching — and every downstream method reported 15. Nothing raised; the
    count was simply wrong.

    So the size is now measured rather than assumed. A provisional un-flattened
    Otsu gives a distance transform whose maximum is the radius of the largest
    blob; 2.5x that, made odd, clears the largest coin with room to spare and
    still removes a gradient spanning the frame.
    """
    _, provisional = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    radius = float(cv2.distanceTransform(provisional, cv2.DIST_L2, 5).max())
    kernel = int(2 * np.ceil(2.5 * radius) + 1)
    return max(floor, kernel)


def _flatten_illumination(gray: np.ndarray, kernel: int | None = None) -> np.ndarray:
    """Remove a slowly-varying background with a white top-hat.

    `skimage.data.coins` is lit unevenly — the background at the top of the frame
    is **brighter than Otsu's global threshold**, so a plain Otsu classifies a
    band of empty table as foreground and merges it with the entire top row of
    coins. One component of 13,433 px where there should have been six coins.

    A white top-hat is `image − opening(image)`. Opening with a structuring
    element larger than any coin erases the coins and leaves the illumination, so
    subtracting it leaves the coins on a flat background. This is the same
    operator project 05 uses to find scratches, for the same reason: it selects
    by *size*, and the thing being removed here is larger than everything being
    kept.
    """
    if kernel is None:
        kernel = background_kernel_for(gray)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


def _foreground_mask(img: np.ndarray, flatten: bool = True) -> np.ndarray:
    """Illumination-flattened Otsu, cleaned. Shared by every region method."""
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    if flatten:
        gray = _flatten_illumination(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean_binary(mask)


def segment_watershed_global_seed(
    img: np.ndarray, fg_ratio: float = 0.55, flatten: bool = True
) -> np.ndarray:
    """Watershed seeded by a **global** fraction of the distance transform.

    This is the version in the OpenCV tutorial, kept here because it is what
    almost every watershed demo copies and because it is wrong on this image in
    an instructive way.

    ``cv2.threshold(dist, fg_ratio * dist.max(), ...)`` compares every pixel's
    distance-to-background against **one** number derived from the single deepest
    point in the whole image. That is only a sensible seed rule if every object is
    about the same size *and* already separated. Here the coins touch, so the
    merged blob has a deep interior, and a threshold set from it erases the local
    peak of every coin except the largest — **24 coins become 1, at every ratio
    from 0.5 upward.**
    """
    mask = _foreground_mask(img, flatten=flatten)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    return _flood_from_seeds(img, mask, sure_fg.astype(np.uint8))


def segment_watershed(
    img: np.ndarray, min_distance: int = 12, min_radius_px: float = 10.0, flatten: bool = True
):
    """Distance-transform watershed seeded by **local** maxima — the working version.

    The distance transform peaks at each coin's centre even when the coins touch,
    so the seeds have to be found *per coin*: a pixel is a seed if it is the
    maximum of its own neighbourhood.

    Both knobs are quantities you can estimate by looking at the image, which is
    the point of them:

    ``min_distance``
        Radius of the neighbourhood a peak must dominate, in pixels. Too small
        and one coin's noisy distance ridge produces several peaks (overcount);
        too large and two touching coins share one (undercount).
    ``min_radius_px``
        The distance transform's value *is* the distance to the nearest
        background pixel, so at a coin's centre it is that coin's radius. A floor
        on it therefore says "ignore peaks that could not be a coin" — in the
        same units as the answer.

    Deliberately **not** a fraction of ``dist.max()``. That is the rule
    :func:`segment_watershed_global_seed` uses, and it is the bug this function
    exists to avoid: any threshold derived from the single deepest point in the
    image is a threshold set by the largest merged blob.
    """
    mask = _foreground_mask(img, flatten=flatten)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    k = 2 * int(min_distance) + 1
    dilated = cv2.dilate(dist, np.ones((k, k), np.float32))
    peaks = ((dist >= dilated - 1e-6) & (dist >= min_radius_px)).astype(np.uint8) * 255
    peaks = cv2.dilate(peaks, np.ones((3, 3), np.uint8))  # widen 1-px peaks into seeds
    return _flood_from_seeds(img, mask, peaks)


def _flood_from_seeds(img: np.ndarray, mask: np.ndarray, sure_fg: np.ndarray) -> np.ndarray:
    """Run OpenCV's watershed from a given seed image and return clean labels."""
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(to_gray(img), cv2.COLOR_GRAY2BGR), markers)

    labels = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    labels[markers == -1] = 0  # watershed marks boundaries with -1
    # the flood fills the background region too; drop anything outside the mask
    labels[mask == 0] = 0
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
    "Adaptive + components": segment_adaptive_components,
    "Watershed (global seed)": segment_watershed_global_seed,
    "Watershed (local maxima)": segment_watershed,
    "Hough circles": segment_hough_circles,
}


# --------------------------------------------------------------------------- #
# counting and measurement
# --------------------------------------------------------------------------- #


#: Smallest radius, in pixels, that a region has to be consistent with to count
#: as a coin. Expressed as a radius rather than an area because that is the
#: quantity you can read off the image, and it is the same number the watershed
#: seeding uses.
MIN_COIN_RADIUS_PX = 6.0
MIN_COIN_AREA_PX = int(np.pi * MIN_COIN_RADIUS_PX**2)  # 113


def region_properties(labels: np.ndarray, min_area: int = MIN_COIN_AREA_PX) -> list[dict]:
    """Area, centroid and equivalent diameter for every labelled region.

    ``min_area`` discards fragments. Without it, morphological speckle is counted
    as a coin and the count is meaningless.

    **Set it too high and it discards coins.** The previous default of 250 px was
    picked to be "obviously small", and it threw away two of the 24 basins that
    correctly-seeded watershed produces — turning a perfect 24/24 into 22/24 after
    the segmentation had already got the answer right. The floor is now derived
    from :data:`MIN_COIN_RADIUS_PX`, so it means something in the units of the
    problem instead of being a round number.
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


def count_coins(labels: np.ndarray, min_area: int = MIN_COIN_AREA_PX) -> int:
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


#: No coin in this image is smaller than about half the largest. A measured
#: diameter below this fraction of the reference is not a small coin, it is a
#: broken region — so it is counted and reported rather than averaged in.
PLAUSIBLE_MIN_FRACTION = 0.45


def evaluate_methods(runs: int = 3) -> list[dict]:
    """Count coins with every method, and separately ask whether it *measured* them.

    Two columns that a segmentation demo collapses into one:

    ``abs_count_error``
        Did it find the right number of objects?
    ``implausible``
        How many of those objects have a diameter no coin could have? A region
        squeezed to 136 px by a watershed boundary still counts as one coin — the
        count is right — while implying a 5.75 mm coin next to a 24.25 mm
        reference. Getting the count right says nothing about whether the
        measurement is usable, and this is the column that shows it.
    """
    from shared import io

    img = io.sample("coins")
    rows = []
    for name, fn in METHODS.items():
        labels, timing = timeit(lambda f=fn: f(img), runs=runs, warmup=1)
        props = region_properties(labels)
        diameters = [p["diameter_px"] for p in props]
        mm_per_px = calibrate_mm_per_px(props, REFERENCE_DIAMETER_MM)
        mm = measure_mm(props, mm_per_px)
        floor = PLAUSIBLE_MIN_FRACTION * REFERENCE_DIAMETER_MM
        rows.append(
            {
                "method": name,
                "count": len(props),
                "true_count": TRUE_COIN_COUNT,
                "count_error": len(props) - TRUE_COIN_COUNT,
                "abs_count_error": abs(len(props) - TRUE_COIN_COUNT),
                "mean_diameter_px": round(float(np.mean(diameters)), 2) if diameters else None,
                "mean_diameter_mm": round(float(np.mean(mm)), 2) if mm else None,
                "min_diameter_mm": round(float(np.min(mm)), 2) if mm else None,
                "max_diameter_mm": round(float(np.max(mm)), 2) if mm else None,
                "diameter_cv": round(float(np.std(mm) / np.mean(mm)), 4) if mm else None,
                "implausible": int(sum(1 for d in mm if d < floor)),
                "median_ms": round(float(timing.median_ms), 3),
            }
        )
    return rows


def sweep_watershed_seed(ratios=(0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8)) -> list[dict]:
    """The tutorial's one knob, swept — and the knob that replaces it.

    ``fg_ratio`` thresholds the distance transform at a fraction of its **global**
    maximum. Sweeping it shows there is no value that works: the count is either
    collapsed or unstable, because the quantity being thresholded is set by the
    largest blob in the image rather than by the object being seeded.

    The local-maxima column is in the same table to make the comparison direct —
    it has no ``fg_ratio`` at all, so its count is flat by construction, and that
    flatness *is* the result.
    """
    from shared import io

    img = io.sample("coins")
    local = count_coins(segment_watershed(img))
    rows = []
    for r in ratios:
        n = count_coins(segment_watershed_global_seed(img, fg_ratio=r))
        rows.append(
            {
                "fg_ratio": r,
                "global_seed_count": n,
                "global_seed_error": n - TRUE_COIN_COUNT,
                "local_maxima_count": local,
                "local_maxima_error": local - TRUE_COIN_COUNT,
            }
        )
    return rows


def ablate_mask_and_seeding() -> list[dict]:
    """Two independent decisions, all four combinations.

    The project's central result. Flattening the illumination fixes the *mask*;
    local-maxima seeding fixes the *seeds*. They were found as two separate bugs,
    and crossing them shows they are not equally important:

    * With a broken mask, the tutorial's global-fraction rule counts **1** coin.
      Local-maxima seeding counts **24**.
    * Fixing the mask rescues the tutorial rule only to 23.

    So the headline is not "the lighting was bad". It is that one seeding rule
    survives a bad mask and the other does not, and the usual demo uses the one
    that does not.
    """
    from shared import io

    img = io.sample("coins")
    rows = []
    for seed_name, fn in (
        ("Global fraction of dist.max() (the tutorial)", segment_watershed_global_seed),
        ("Local maxima of the distance transform", segment_watershed),
    ):
        plain = count_coins(fn(img, flatten=False))
        flat = count_coins(fn(img, flatten=True))
        rows.append(
            {
                "seeding": seed_name,
                "plain_otsu": plain,
                "tophat_otsu": flat,
                "plain_error": plain - TRUE_COIN_COUNT,
                "tophat_error": flat - TRUE_COIN_COUNT,
            }
        )
    return rows


def calibration_sensitivity(errors=(-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)) -> list[dict]:
    """What a wrong reference costs every other measurement.

    Calibration from one known object is the step that turns a segmentation into
    a measurement with a unit — and it is a single multiplication, so the
    reference's error propagates to **every** diameter, undiminished and
    unsignalled. Nothing downstream can detect it: the numbers stay perfectly
    self-consistent, they are just all wrong by the same factor.

    This is worth a table rather than a sentence because a reader looking at
    "17.49 mm" has no way to see that it rests entirely on one assumed number.
    """
    from shared import io

    img = io.sample("coins")
    props = region_properties(segment_watershed(io.sample("coins")))
    truth_scale = calibrate_mm_per_px(props, REFERENCE_DIAMETER_MM)
    truth = measure_mm(props, truth_scale)

    rows = []
    for pct in errors:
        assumed = REFERENCE_DIAMETER_MM * (1.0 + pct / 100.0)
        mm = measure_mm(props, calibrate_mm_per_px(props, assumed))
        rows.append(
            {
                "reference_error_pct": pct,
                "assumed_reference_mm": round(assumed, 2),
                "mean_diameter_mm": round(float(np.mean(mm)), 3),
                "measured_error_pct": round(
                    100.0 * (float(np.mean(mm)) - float(np.mean(truth))) / float(np.mean(truth)), 3
                ),
                "smallest_mm": round(float(np.min(mm)), 2),
                "largest_mm": round(float(np.max(mm)), 2),
            }
        )
    return rows


def diameter_distribution(method: str = "Watershed (local maxima)") -> list[dict]:
    """Per-coin diameters, sorted — the output a measuring tool actually produces."""
    from shared import io

    labels = METHODS[method](io.sample("coins"))
    props = region_properties(labels)
    mm_per_px = calibrate_mm_per_px(props, REFERENCE_DIAMETER_MM)
    rows = [
        {
            "rank": i,
            "area_px": p["area_px"],
            "diameter_px": round(p["diameter_px"], 2),
            "diameter_mm": round(p["diameter_px"] * mm_per_px, 2),
        }
        for i, p in enumerate(
            sorted(props, key=lambda q: q["diameter_px"], reverse=True), start=1
        )
    ]
    return rows


def analyse(img: np.ndarray, method: str = "Watershed (local maxima)"):
    """Full pipeline for the UI: labels, per-coin properties and mm measurements."""
    labels = METHODS[method](img)
    props = region_properties(labels)
    mm_per_px = calibrate_mm_per_px(props, REFERENCE_DIAMETER_MM)
    for p in props:
        p["diameter_mm"] = p["diameter_px"] * mm_per_px
    return labels, props, mm_per_px


# --------------------------------------------------------------------------- #
# denomination identification
# --------------------------------------------------------------------------- #
#
# Counting is the easy half. "How much money is on the table" needs each coin
# named, and a diameter is the only thing a single overhead photograph offers.
#
# Two things make that a real question rather than a lookup:
#
#   * the scale has to come from somewhere, and
#   * two Indian coins have the SAME published diameter.
#
# Both are measured below rather than asserted.

#: Published diameters, in millimetres, of the Indian coins this project reads.
#: Imported from the scene generator so the classifier and the truth cannot
#: drift apart — if a diameter is wrong it is wrong in both places and the
#: accuracy stays honest.
COIN_TABLE_MM = dict(synth.RUPEE_COINS_MM)

#: Denominations a diameter can actually separate. 10 and 20 are both 27.00 mm,
#: so a size-based reader is asked to name one of them and can only ever be
#: right by luck. `20` is excluded from the *classifier's* vocabulary, not from
#: the scenes: the confusion it causes is reported instead of hidden.
IDENTIFIABLE = (1, 2, 5, 10)


def identify_by_diameter(diameter_mm: float, table: dict[int, float] | None = None) -> int:
    """Name the coin whose published diameter is nearest. No tie-breaking.

    Deliberately the simplest possible rule, because the interesting question is
    not which classifier to use — with one feature and four classes there is
    nothing to choose — but how much of the error is *measurement* and how much
    is the coinage being genuinely ambiguous.
    """
    table = table or {d: COIN_TABLE_MM[d] for d in IDENTIFIABLE}
    return min(table, key=lambda d: abs(table[d] - diameter_mm))


def calibrate_from_largest(props: list[dict], largest_mm: float) -> float:
    """mm per pixel, assuming the biggest region in the frame is ``largest_mm``.

    This is what a user can actually do without a ruler in the shot: name the
    largest coin they know is present. It fails in a specific and checkable way
    — if the largest coin is *absent*, every diameter is wrong by the ratio of
    the assumed size to the true one, and nothing downstream can tell.
    """
    biggest = max((p["diameter_px"] for p in props), default=0.0)
    return largest_mm / max(biggest, EPS)


def read_scene(
    labels: np.ndarray,
    mm_per_px: float,
    min_area: int = MIN_COIN_AREA_PX,
) -> list[dict]:
    """Turn a label image into a list of named, measured coins."""
    out = []
    for prop in region_properties(labels, min_area=min_area):
        d_mm = prop["diameter_px"] * mm_per_px
        out.append(
            {
                "centre_xy": prop["centroid"],
                "diameter_px": prop["diameter_px"],
                "diameter_mm": d_mm,
                "denomination": identify_by_diameter(d_mm),
            }
        )
    return out


def match_to_truth(read: list[dict], truth: list[dict], tol_px: float = 24.0):
    """Pair each detected coin with the true coin nearest its centre.

    Greedy nearest-centre rather than Hungarian: the coins are further apart
    than the error in locating them, so the assignment is not ambiguous, and a
    greedy pass makes the failure mode obvious — an unmatched truth coin is a
    miss, an unmatched detection is a false positive.
    """
    remaining = list(range(len(truth)))
    pairs, spurious = [], 0
    for det in read:
        best, best_d = None, tol_px
        for j in remaining:
            tx, ty = truth[j]["centre_xy"]
            d = float(np.hypot(det["centre_xy"][0] - tx, det["centre_xy"][1] - ty))
            if d < best_d:
                best, best_d = j, d
        if best is None:
            spurious += 1
            continue
        remaining.remove(best)
        pairs.append((det, truth[best]))
    return pairs, spurious, len(remaining)


def evaluate_identification(scenes, method: str | None = None):
    """Score naming the coins, under an exact scale and a self-derived one.

    ``scenes`` is a list of ``(image, truth_coins)``. Returns one row per
    calibration strategy plus a confusion matrix, because the headline accuracy
    hides which pairs it confuses and that is the whole result.
    """
    method = method or IDENT_METHOD
    rows, confusion = [], {}
    for calib in ("exact scale", "largest coin assumed 27 mm"):
        total = correct = value_true = value_read = 0
        missed = spurious_total = 0
        for img, truth, mm_per_px in scenes:
            labels = METHODS[method](img)
            props = region_properties(labels)
            if not props:
                missed += len(truth)
                continue
            scale = (
                mm_per_px
                if calib == "exact scale"
                else calibrate_from_largest(props, max(COIN_TABLE_MM.values()))
            )
            read = read_scene(labels, scale)
            pairs, spurious, unmatched = match_to_truth(read, truth)
            spurious_total += spurious
            missed += unmatched
            for det, tru in pairs:
                total += 1
                value_read += det["denomination"]
                value_true += tru["denomination"]
                correct += int(det["denomination"] == tru["denomination"])
                # Only the exact-scale pass contributes to the confusion
                # matrix. Accumulating both passes into one dict counted every
                # coin twice and made the off-diagonal totals larger than the
                # number of errors in the accuracy column beside it, which is
                # the kind of inconsistency a reader is entitled to trust is
                # absent. The other calibration's damage is reported as its own
                # accuracy row instead.
                if calib == "exact scale":
                    key = (tru["denomination"], det["denomination"])
                    confusion[key] = confusion.get(key, 0) + 1
        rows.append(
            {
                "calibration": calib,
                "coins_matched": total,
                "identified": correct,
                "accuracy": round(correct / max(total, 1), 4),
                "missed": missed,
                "spurious": spurious_total,
                "true_value": value_true,
                "read_value": value_read,
                "value_error_pct": round(
                    100.0 * (value_read - value_true) / max(value_true, 1), 2
                ),
            }
        )
    return rows, confusion


# --------------------------------------------------------------------------- #
# drawing, so a reader can see what was segmented and what it was called
# --------------------------------------------------------------------------- #

#: The segmentation the identification results are computed on. Named once so
#: the figure, the tables and `infer.py` cannot drift apart.
IDENT_METHOD = "Hough circles"

#: One colour per denomination, in RGB. Chosen to stay legible over brass and
#: steel on a dark table, which rules out yellow and pale grey.
DENOM_COLOURS = {
    1: (0, 190, 255),
    2: (0, 235, 120),
    5: (255, 105, 180),
    10: (255, 80, 60),
    20: (200, 120, 255),
}


def overlay_regions(img: np.ndarray, labels: np.ndarray, min_area: int = MIN_COIN_AREA_PX):
    """Outline every counted region and number it.

    Drawn on a darkened copy so the outlines read against bright metal. The
    numbering matters more than it looks: a count in a caption is a claim, and an
    outline with a number beside it is the same claim in a form a reader can
    check by eye.
    """
    out = (to_float(img) * 0.55 * 255).astype(np.uint8).copy()
    for i, prop in enumerate(region_properties(labels, min_area=min_area), start=1):
        cx, cy = prop["centroid"]
        r = int(round(prop["diameter_px"] / 2))
        cv2.circle(out, (int(round(cx)), int(round(cy))), r, (60, 255, 90), 2)
        cv2.putText(
            out, str(i), (int(cx) - 8, int(cy) + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return out


def overlay_denominations(img: np.ndarray, read: list[dict]):
    """Outline each coin in its denomination's colour and print the value."""
    out = (to_float(img) * 0.55 * 255).astype(np.uint8).copy()
    for coin in read:
        cx, cy = coin["centre_xy"]
        r = int(round(coin["diameter_px"] / 2))
        colour = DENOM_COLOURS.get(coin["denomination"], (255, 255, 255))
        cv2.circle(out, (int(round(cx)), int(round(cy))), r, colour, 2)
        text = str(coin["denomination"])
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            out, text, (int(cx) - tw // 2, int(cy) + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            out, text, (int(cx) - tw // 2, int(cy) + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
        )
    total = sum(c["denomination"] for c in read)
    cv2.putText(out, f"Rs {total}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, f"Rs {total}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out
