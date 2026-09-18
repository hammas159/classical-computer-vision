"""Background subtraction: five models, two controls, and a truth mask that is earned.

The question
------------
A static camera sees a scene that mostly does not change. Subtract the part that
does not change and what is left is the part that does. Every step of that
sentence hides a decision.

> **The claim under test:** the adaptive mixture models (MOG2, KNN) beat the
> simple ones (frame differencing, running average, a median background) on real
> footage. **It depends entirely on what kind of change you mean**, and by more
> than the gap between any two methods: going from an instantaneous change to a
> sustained one, KNN gains 0.339 IoU and frame differencing gains 0.085. On one
> arm differencing wins by 3.6x; on the other they tie.

Where the ground truth comes from
---------------------------------
This clip has no annotation, and none is invented. Two arms, deliberately
different, because a single one would have produced a ranking that is an
artefact of how the test was built.

**The oracle arm** (the main one). A pedestrian covers any given pixel for a few
seconds out of eighty, so the **per-pixel median of the whole clip is the empty
plaza** -- and visibly is: there is not one person in it. Truth for a frame is
where it differs from that empty scene. The oracle sees all 795 frames including
the future; every method sees 60 past frames, which is what makes the oracle
truth rather than a seventh competitor. It is checked against OpenCV's HOG
pedestrian detector, which shares no information with it, in `oracle_versus_hog`.

**The composite arm.** A rectangle of frame *B* pasted into frame *A*: both
halves photographed, the position known exactly. Truth is **not the rectangle** --
it is the rectangle intersected with where the pixels actually differ, because
an unchanged patch of grass inside the pasted box is not an observable change and
marking it foreground would score every method on detecting nothing. On this clip
the rectangles turn out to be only 25-58% real change.

The two controls
----------------
`all_background` calls every pixel background and `all_foreground` calls every
pixel foreground. They are in the results table because **pixel accuracy is a
broken metric here** -- foreground is under 4% of the frame, so doing nothing
scores 0.962 and beats four of the five real methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import ensure_rgb, to_gray
from shared.metrics import iou

EPS = 1e-9

#: The same 768x576 plaza clip projects 29, 55 and 57 use. A static camera is
#: the whole premise: background subtraction has nothing to subtract without one.
VIDEO = Path.home() / ".cache" / "classical-cv-images" / "assets" / "video" / "vtest.avi"

#: The twelve target frames. Different from project 57's twelve, and far enough
#: apart that no two show the same arrangement of people.
#:
#: They are **not twelve distinct images** in the sense the rest of this
#: repository means it, and pretending otherwise would be dishonest: they share
#: a background, and a perceptual hash puts them 2 to 6 bits apart. That is what
#: a static-camera surveillance clip *is*. The variety in this project comes from
#: twelve different composites with twelve different truth masks, not from twelve
#: different scenes.
FRAMES = (80, 140, 200, 260, 320, 380, 440, 500, 560, 620, 690, 760)

#: How many frames of history every adaptive model is given before the target
#: frame. The same number for all of them, because otherwise the comparison
#: measures how much history each one was handed.
HISTORY = 60


def video_available() -> bool:
    return VIDEO.exists()


def _require_video() -> None:
    if not video_available():
        raise FileNotFoundError(
            f"{VIDEO} is missing. Run `python tools/fetch_assets.py --set video`."
        )


def frame_count() -> int:
    _require_video()
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def load_frames(start: int, stop: int) -> list[np.ndarray]:
    """Frames ``[start, stop)`` in order, RGB.

    Read sequentially rather than by seeking to each index: seeking in a
    compressed stream lands on the nearest keyframe, which silently changes
    which frames a "history of 60" actually contains.
    """
    _require_video()
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))
        out = []
        for _ in range(max(0, int(stop) - max(0, int(start)))):
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
    finally:
        cap.release()


def load_frame(index: int) -> np.ndarray:
    frames = load_frames(index, index + 1)
    if not frames:
        raise RuntimeError(f"could not read frame {index} of {VIDEO}")
    return frames[0]


# --------------------------------------------------------------------------- #
# the noise floor, measured
# --------------------------------------------------------------------------- #

#: A patch of the frame that nobody walks through: the grass in the lower left.
#: Used only to measure how much a pixel's value moves when nothing happens.
STATIC_PATCH = (20, 430, 120, 100)  # x, y, w, h


def measure_noise_floor(frames: int = 120, patch=STATIC_PATCH) -> float:
    """How much a background pixel changes between consecutive frames.

    The threshold every simple method needs has to come from somewhere. Choosing
    it by eye tunes the method to the answer; this measures it once, from a
    region of the clip that nothing walks through, and every method that needs a
    threshold is given the same number.

    Returns the 99th percentile of the absolute inter-frame difference, so a
    method using it as a threshold rejects 99% of pure sensor noise.
    """
    x, y, w, h = patch
    seq = [to_gray(f)[y:y + h, x:x + w].astype(np.float32)
           for f in load_frames(0, frames)]
    diffs = np.abs(np.diff(np.stack(seq), axis=0))
    return float(np.percentile(diffs, 99))


#: Cached so twelve composites do not re-read a hundred frames twelve times.
_NOISE: float | None = None


def noise_floor() -> float:
    global _NOISE
    if _NOISE is None:
        _NOISE = measure_noise_floor()
    return _NOISE


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def _clean(mask: np.ndarray, open_k: int = 3, close_k: int = 9) -> np.ndarray:
    """Identical morphology for every method, so the comparison is of models.

    Any of these would look better with post-processing tuned to it, and that is
    exactly why they all get the same.
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (close_k, close_k)))


def all_background(history: list[np.ndarray], frame: np.ndarray,
                   threshold: float | None = None) -> np.ndarray:
    """Call every pixel background. The control that breaks pixel accuracy."""
    return np.zeros(frame.shape[:2], np.uint8)


def all_foreground(history: list[np.ndarray], frame: np.ndarray,
                   threshold: float | None = None) -> np.ndarray:
    """Call every pixel foreground. The control at the other end."""
    return np.full(frame.shape[:2], 255, np.uint8)


def frame_difference(history: list[np.ndarray], frame: np.ndarray,
                     threshold: float | None = None) -> np.ndarray:
    """|this frame - the previous one|, thresholded. No model at all.

    Its failure is structural rather than a matter of tuning: a uniformly
    coloured object moving slowly differs from itself, so only its leading and
    trailing edges appear. The interior is invisible however the threshold is
    set.
    """
    threshold = noise_floor() if threshold is None else threshold
    previous = to_gray(history[-1]).astype(np.float32)
    current = to_gray(frame).astype(np.float32)
    return _clean(((np.abs(current - previous) > threshold) * 255).astype(np.uint8))


def running_average(history: list[np.ndarray], frame: np.ndarray,
                    threshold: float | None = None, alpha: float = 0.05) -> np.ndarray:
    """An exponentially decaying background, thresholded.

    One number of memory per pixel. Cheap, and it smears: an object that stops
    is absorbed into the background over roughly ``1/alpha`` frames, and leaves a
    ghost behind it for as long again.
    """
    threshold = noise_floor() if threshold is None else threshold
    background = to_gray(history[0]).astype(np.float32)
    for f in history[1:]:
        cv2.accumulateWeighted(to_gray(f).astype(np.float32), background, alpha)
    current = to_gray(frame).astype(np.float32)
    return _clean(((np.abs(current - background) > threshold) * 255).astype(np.uint8))


def median_background(history: list[np.ndarray], frame: np.ndarray,
                      threshold: float | None = None, samples: int = 25) -> np.ndarray:
    """The per-pixel median of the history. The strongest of the simple models.

    A median is unmoved by an object that covers a pixel for less than half the
    history, which is exactly the right property for a crossing pedestrian and
    exactly the wrong one for somebody who stands still.
    """
    threshold = noise_floor() if threshold is None else threshold
    step = max(1, len(history) // samples)
    stack = np.stack([to_gray(f) for f in history[::step]]).astype(np.float32)
    background = np.median(stack, axis=0)
    current = to_gray(frame).astype(np.float32)
    return _clean(((np.abs(current - background) > threshold) * 255).astype(np.uint8))


def _opencv_subtractor(make, history: list[np.ndarray], frame: np.ndarray,
                       shadows_as_background: bool = True) -> np.ndarray:
    sub = make()
    for f in history:
        sub.apply(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    raw = sub.apply(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), learningRate=0)
    binary = (raw == 255) if shadows_as_background else (raw > 0)
    return _clean((binary * 255).astype(np.uint8))


def mog2(history: list[np.ndarray], frame: np.ndarray,
         threshold: float | None = None) -> np.ndarray:
    """A Gaussian mixture per pixel, with the number of components adapting.

    `detectShadows` marks shadow pixels 127 rather than 255; here they are
    treated as background, which is what the shadow flag is for. The cost of
    that choice is measured in `shadow_handling`.
    """
    return _opencv_subtractor(
        lambda: cv2.createBackgroundSubtractorMOG2(
            history=HISTORY, varThreshold=16, detectShadows=True),
        history, frame)


def knn(history: list[np.ndarray], frame: np.ndarray,
        threshold: float | None = None) -> np.ndarray:
    """K-nearest-neighbours on the recent samples of each pixel."""
    return _opencv_subtractor(
        lambda: cv2.createBackgroundSubtractorKNN(
            history=HISTORY, dist2Threshold=400.0, detectShadows=True),
        history, frame)


METHODS: dict[str, Callable] = {
    "All background (control)": all_background,
    "All foreground (control)": all_foreground,
    "Frame difference": frame_difference,
    "Running average": running_average,
    "Median background": median_background,
    "MOG2": mog2,
    "KNN": knn,
}

#: The methods that are actually models, for tables where a control would only
#: take up a row.
REAL_METHODS = tuple(m for m in METHODS if "control" not in m)


# --------------------------------------------------------------------------- #
# the composites, and the truth that goes with them
# --------------------------------------------------------------------------- #

#: (target frame, donor frame, rectangle) for each of the twelve tests. The
#: rectangles walk across the frame so that no two composites put the change in
#: the same place, and they are sized like the things that actually move here.
SWAPS: tuple[tuple[int, int, tuple[int, int, int, int]], ...] = (
    (80, 300, (60, 180, 150, 220)),
    (140, 420, (230, 200, 130, 200)),
    (200, 520, (430, 170, 160, 240)),
    (260, 640, (590, 190, 150, 230)),
    (320, 120, (120, 230, 180, 200)),
    (380, 700, (330, 160, 140, 250)),
    (440, 180, (500, 220, 170, 210)),
    (500, 260, (40, 150, 160, 240)),
    (560, 340, (250, 180, 150, 220)),
    (620, 60, (420, 210, 180, 200)),
    (690, 480, (560, 160, 150, 250)),
    (760, 560, (150, 200, 170, 230)),
)


def swap_composite(target: int, donor: int, rect: tuple[int, int, int, int]):
    """Paste a rectangle of the donor frame into the target frame.

    Returns ``(composite, truth, original)``. Both frames are photographs of the
    same place at different moments, so the pasted region carries real people,
    real shadows and real sensor noise rather than anything drawn.

    ``truth`` is the rectangle **intersected with where the pixels actually
    changed** by more than the measured noise floor. Where the two frames agree —
    unchanged grass, unchanged tarmac — nothing observable happened, and marking
    it foreground would score every method on its ability to detect nothing.
    """
    original = load_frame(target)
    donor_frame = load_frame(donor)
    x, y, w, h = rect
    h = min(h, original.shape[0] - y)
    w = min(w, original.shape[1] - x)

    composite = original.copy()
    composite[y:y + h, x:x + w] = donor_frame[y:y + h, x:x + w]

    changed = np.abs(to_gray(composite).astype(np.float32)
                     - to_gray(original).astype(np.float32)) > noise_floor()
    truth = np.zeros(original.shape[:2], np.uint8)
    box = np.zeros(original.shape[:2], bool)
    box[y:y + h, x:x + w] = True
    truth[box & changed] = 255
    return composite, truth, original


def truth_coverage() -> list[dict]:
    """What fraction of each pasted rectangle actually changed.

    Printed rather than assumed, because it is the number that justifies not
    using the rectangle itself as the mask: on this clip the rectangles are only
    a third to a half real change, and the rest is scenery that happens to match.
    """
    rows = []
    for target, donor, rect in SWAPS:
        _, truth, _ = swap_composite(target, donor, rect)
        x, y, w, h = rect
        rows.append({
            "frame": target,
            "donor": donor,
            "rectangle_px": int(w * h),
            "changed_px": int((truth > 0).sum()),
            "changed_share": round(float((truth > 0).sum()) / max(w * h, 1), 4),
            "foreground_share_of_frame": round(
                float((truth > 0).mean()), 5),
        })
    return rows


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = pred > 0
    t = truth > 0
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    tn = float((~p & ~t).sum())
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    return {
        "iou": iou(pred, truth),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, EPS),
        "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, EPS),
    }


def evaluate(swaps=SWAPS, history: int = HISTORY, runs: int = 1) -> list[dict]:
    """Every method on every composite, against the earned truth mask."""
    acc = {name: {"iou": [], "precision": [], "recall": [], "f1": [],
                  "pixel_accuracy": [], "ms": []} for name in METHODS}

    for target, donor, rect in swaps:
        past = load_frames(max(0, target - history), target)
        composite, truth, _ = swap_composite(target, donor, rect)
        for name, fn in METHODS.items():
            pred, timing = timeit(lambda f=fn: f(past, composite), runs=runs, warmup=0)
            for key, value in score(pred, truth).items():
                acc[name][key].append(value)
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": name,
            "iou": round(float(np.mean(a["iou"])), 4),
            "precision": round(float(np.mean(a["precision"])), 4),
            "recall": round(float(np.mean(a["recall"])), 4),
            "f1": round(float(np.mean(a["f1"])), 4),
            "pixel_accuracy": round(float(np.mean(a["pixel_accuracy"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for name, a in acc.items()
    ]


def per_composite(swaps=SWAPS, history: int = HISTORY) -> list[dict]:
    """The same measurement one row per composite, because the mean hides two."""
    rows = []
    for target, donor, rect in swaps:
        past = load_frames(max(0, target - history), target)
        composite, truth, _ = swap_composite(target, donor, rect)
        row = {"frame": target, "donor": donor,
               "foreground_px": int((truth > 0).sum())}
        for name in REAL_METHODS:
            row[name] = round(iou(METHODS[name](past, composite), truth), 4)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# the oracle: what the plaza looks like with nobody in it
# --------------------------------------------------------------------------- #

#: Every third frame of the whole clip. A pedestrian covers any given pixel for
#: a few seconds out of eighty, so the per-pixel median over the whole clip is
#: the empty plaza -- and it is, visibly: there is not one person in it.
ORACLE_STRIDE = 3

#: The oracle marks a pixel foreground when it differs from the empty plaza by
#: more than this multiple of the measured noise floor. Two rather than one,
#: because the oracle is a claim about truth and should be the conservative one.
ORACLE_MULTIPLE = 2.0

_ORACLE: np.ndarray | None = None


def empty_scene(stride: int = ORACLE_STRIDE) -> np.ndarray:
    """The per-pixel median of the entire clip: the plaza with nobody in it.

    **This is information no method is given.** Every method here sees 60 past
    frames; the oracle sees all 795, including the future. That asymmetry is what
    makes it usable as truth rather than as a seventh competitor.

    What it is not: an annotation. Two honest caveats, both stated here rather
    than discovered later.

    1. It is the same *family* of estimator as the `median_background` method, so
       this truth flatters that method more than the others. `oracle_versus_hog`
       checks the oracle against an appearance-based detector that shares no
       information with it, precisely because that objection is a real one.
    2. It calls the fluttering barrier tape foreground, because the tape really
       does move. That is not noise and it is not removed.
    """
    global _ORACLE
    if _ORACLE is None:
        frames = load_frames(0, frame_count())
        _ORACLE = np.median(np.stack(frames[::stride]), axis=0).astype(np.uint8)
    return _ORACLE


def oracle_mask(index: int, multiple: float = ORACLE_MULTIPLE) -> np.ndarray:
    """What actually moved in one frame, from the empty-scene median.

    Cleaned with a 3x3 opening and nothing else. A heavier cleanup would make the
    masks prettier and would quietly delete the thin moving things -- the tape,
    an arm -- that are the hard part.
    """
    frame = load_frame(index)
    diff = np.abs(to_gray(frame).astype(np.float32)
                  - to_gray(empty_scene()).astype(np.float32))
    mask = ((diff > multiple * noise_floor()) * 255).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def oracle_versus_hog(frames=None, iou_threshold: float = 0.25) -> list[dict]:
    """Does the oracle find people, or only find change?

    Checked against OpenCV's HOG pedestrian detector, which works from gradient
    orientations in a single frame and knows nothing about time. The oracle works
    from time and knows nothing about what a person looks like. They share no
    information, so agreement is evidence rather than circularity.

    Reported, not asserted: the oracle also marks the barrier tape, which HOG
    will never confirm, and HOG finds people who are standing still, which the
    oracle will never confirm.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    rows = []
    for index in (frames or FRAMES):
        frame = load_frame(index)
        boxes, _ = hog.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                        winStride=(8, 8), padding=(16, 16), scale=1.05)
        tight = []
        for (x, y, w, h) in np.asarray(boxes, np.int32).reshape(-1, 4):
            dw, dh = int(w * 0.15), int(h * 0.05)
            tight.append((x + dw, y + dh, max(1, w - 2 * dw), max(1, h - 2 * dh)))

        mask = oracle_mask(index)
        n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        blobs = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n)
                 if stats[i, 4] >= 400]

        def overlap(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            return inter / max(aw * ah + bw * bh - inter, 1)

        confirmed = sum(1 for b in blobs
                        if max((overlap(b, d) for d in tight), default=0.0)
                        >= iou_threshold)
        found = sum(1 for d in tight
                    if max((overlap(b, d) for b in blobs), default=0.0)
                    >= iou_threshold)
        rows.append({
            "frame": int(index),
            "oracle_blobs": len(blobs),
            "hog_detections": len(tight),
            "oracle_blobs_confirmed_by_hog": confirmed,
            "hog_detections_found_by_oracle": found,
        })
    return rows


def evaluate_against_oracle(frames=None, history: int = HISTORY,
                            runs: int = 1) -> list[dict]:
    """Every method on unmodified frames, against the empty-scene oracle.

    The main table. Unlike the composites, this asks what the methods are
    actually for: sustained motion through a scene, with the history containing
    the object's own past.
    """
    frames = frames or FRAMES
    acc = {name: {"iou": [], "precision": [], "recall": [], "f1": [],
                  "pixel_accuracy": [], "ms": []} for name in METHODS}

    for index in frames:
        past = load_frames(max(0, index - history), index)
        frame = load_frame(index)
        truth = oracle_mask(index)
        for name, fn in METHODS.items():
            pred, timing = timeit(lambda f=fn: f(past, frame), runs=runs, warmup=0)
            for key, value in score(pred, truth).items():
                acc[name][key].append(value)
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": name,
            "iou": round(float(np.mean(a["iou"])), 4),
            "precision": round(float(np.mean(a["precision"])), 4),
            "recall": round(float(np.mean(a["recall"])), 4),
            "f1": round(float(np.mean(a["f1"])), 4),
            "pixel_accuracy": round(float(np.mean(a["pixel_accuracy"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for name, a in acc.items()
    ]


def per_frame_against_oracle(frames=None, history: int = HISTORY) -> list[dict]:
    rows = []
    for index in (frames or FRAMES):
        past = load_frames(max(0, index - history), index)
        frame = load_frame(index)
        truth = oracle_mask(index)
        row = {"frame": int(index),
               "foreground_share": round(float((truth > 0).mean()), 4)}
        for name in REAL_METHODS:
            row[name] = round(iou(METHODS[name](past, frame), truth), 4)
        rows.append(row)
    return rows


def sustained_versus_instantaneous(frames=None, swaps=None,
                                   history: int = HISTORY) -> list[dict]:
    """The two arms side by side. They do not agree, and that is the result.

    A region-swap composite is a change that exists in exactly one frame. An
    unmodified frame of the clip is a change that has been building for as long
    as the person has been walking. Memoryless differencing is the right tool for
    the first and the wrong one for the second, and no single ranking of these
    methods survives both.
    """
    sustained = {r["method"]: r["iou"] for r in
                 evaluate_against_oracle(frames=frames, history=history)}
    instantaneous = {r["method"]: r["iou"] for r in
                     evaluate(swaps=swaps or SWAPS, history=history)}
    return [
        {
            "method": name,
            "sustained_iou": sustained[name],
            "instantaneous_iou": instantaneous[name],
            "difference": round(sustained[name] - instantaneous[name], 4),
        }
        for name in METHODS
    ]


# --------------------------------------------------------------------------- #
# what the composites cannot measure
# --------------------------------------------------------------------------- #


def interior_versus_edge(swaps=SWAPS, history: int = HISTORY,
                         erode: int = 7) -> list[dict]:
    """Recall split into the interior of the change and its boundary.

    Frame differencing's failure is not a low score, it is a *shape*: it reports
    edges and misses interiors. A single recall number cannot say that, and two
    can.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
    acc = {name: {"interior": [], "edge": []} for name in REAL_METHODS}

    for target, donor, rect in swaps:
        past = load_frames(max(0, target - history), target)
        composite, truth, _ = swap_composite(target, donor, rect)
        interior = cv2.erode(truth, kernel) > 0
        edge = (truth > 0) & ~interior
        if not interior.any() or not edge.any():
            continue
        for name in REAL_METHODS:
            pred = METHODS[name](past, composite) > 0
            acc[name]["interior"].append(float(pred[interior].mean()))
            acc[name]["edge"].append(float(pred[edge].mean()))

    return [
        {
            "method": name,
            "interior_recall": round(float(np.mean(a["interior"])), 4),
            "edge_recall": round(float(np.mean(a["edge"])), 4),
            "edge_minus_interior": round(
                float(np.mean(a["edge"]) - np.mean(a["interior"])), 4),
        }
        for name, a in acc.items()
    ]


def stopped_object_test(target: int = 500, history: int = HISTORY,
                        rect=(300, 180, 140, 230), hold=(5, 20, 40, 80, 160)):
    """What happens to an object that stops moving.

    A rectangle of a donor frame is held in place for a growing number of frames
    before the test frame. A model with short memory absorbs it into the
    background; a model with long memory keeps reporting it. Neither is wrong —
    the answer depends on whether a parked car is foreground — but the *rate* is
    a property of the model and is measurable.
    """
    donor = load_frame(max(0, target - 200))
    x, y, w, h = rect
    rows = []
    for held in hold:
        past = []
        for i, f in enumerate(load_frames(max(0, target - history), target)):
            g = f.copy()
            if i >= len(range(max(0, target - history), target)) - held:
                g[y:y + h, x:x + w] = donor[y:y + h, x:x + w]
            past.append(g)
        composite, truth, _ = swap_composite(target, max(0, target - 200), rect)
        row = {"frames_held_still": held}
        for name in REAL_METHODS:
            pred = METHODS[name](past, composite)
            row[name] = round(float((pred[truth > 0] > 0).mean()), 4)
        rows.append(row)
    return rows


def shadow_handling(swaps=SWAPS[:6], history: int = HISTORY) -> list[dict]:
    """How much of each mask MOG2 and KNN would gain by calling shadows foreground.

    The shadow flag is the one place these two models encode something about the
    world rather than about pixel statistics, and turning it off is a one-line
    change with a measurable cost.
    """
    rows = []
    for name, make in (("MOG2", lambda: cv2.createBackgroundSubtractorMOG2(
                            history=HISTORY, varThreshold=16, detectShadows=True)),
                       ("KNN", lambda: cv2.createBackgroundSubtractorKNN(
                            history=HISTORY, dist2Threshold=400.0, detectShadows=True))):
        without, with_, ious_without, ious_with = [], [], [], []
        for target, donor, rect in swaps:
            past = load_frames(max(0, target - history), target)
            composite, truth, _ = swap_composite(target, donor, rect)
            a = _opencv_subtractor(make, past, composite, shadows_as_background=True)
            b = _opencv_subtractor(make, past, composite, shadows_as_background=False)
            without.append(float((a > 0).mean()))
            with_.append(float((b > 0).mean()))
            ious_without.append(iou(a, truth))
            ious_with.append(iou(b, truth))
        rows.append({
            "method": name,
            "mask_share_shadows_excluded": round(float(np.mean(without)), 5),
            "mask_share_shadows_included": round(float(np.mean(with_)), 5),
            "iou_shadows_excluded": round(float(np.mean(ious_without)), 4),
            "iou_shadows_included": round(float(np.mean(ious_with)), 4),
        })
    return rows


def sweep_history(lengths=(5, 15, 30, 60, 120, 240), swaps=SWAPS[:6]) -> list[dict]:
    """How much history each model needs before it is worth anything."""
    rows = []
    for length in lengths:
        row: dict[str, float | int] = {"history": length}
        for name in REAL_METHODS:
            ious = []
            for target, donor, rect in swaps:
                past = load_frames(max(0, target - length), target)
                if len(past) < 2:
                    continue
                composite, truth, _ = swap_composite(target, donor, rect)
                ious.append(iou(METHODS[name](past, composite), truth))
            row[name] = round(float(np.mean(ious)), 4) if ious else 0.0
        rows.append(row)
    return rows


def sweep_history_against_oracle(lengths=(5, 15, 30, 60, 120, 240),
                                 frames=None) -> list[dict]:
    """How much history each model needs, measured on unmodified frames.

    The sweep that matters. Run against the composites instead, every model looks
    worse the more history it has -- which is true of a one-frame anomaly and
    says nothing about background subtraction.
    """
    frames = frames or FRAMES[:6]
    rows = []
    for length in lengths:
        row: dict[str, float | int] = {"history": length}
        for name in REAL_METHODS:
            ious = []
            for index in frames:
                past = load_frames(max(0, index - length), index)
                if len(past) < 2:
                    continue
                ious.append(iou(METHODS[name](past, load_frame(index)),
                                oracle_mask(index)))
            row[name] = round(float(np.mean(ious)), 4) if ious else 0.0
        rows.append(row)
    return rows


def history_precision_recall(method: str = "Median background",
                            lengths=(5, 15, 30, 60, 120, 240),
                            frames=None) -> list[dict]:
    """Why a better background model can give a worse mask.

    The median background's IoU *falls* as it is given more history, which reads
    like a broken implementation until precision and recall are separated. With
    240 frames it recovers 0.979 of the real foreground -- close to the oracle
    itself -- and its precision has collapsed to 0.253, because the threshold is
    a fixed number of grey levels and a longer window also exposes every slow
    illumination drift and compression artefact.

    A single IoU cannot say "the model got better and the threshold did not".
    """
    frames = frames or FRAMES[:6]
    rows = []
    for length in lengths:
        acc = {"precision": [], "recall": [], "iou": []}
        for index in frames:
            past = load_frames(max(0, index - length), index)
            if len(past) < 2:
                continue
            s = score(METHODS[method](past, load_frame(index)), oracle_mask(index))
            for key in acc:
                acc[key].append(s[key])
        rows.append({
            "history": length,
            "precision": round(float(np.mean(acc["precision"])), 4),
            "recall": round(float(np.mean(acc["recall"])), 4),
            "iou": round(float(np.mean(acc["iou"])), 4),
        })
    return rows


def oracle_threshold_sensitivity(multiples=(1.5, 2.0, 3.0, 4.0),
                                 frames=None, history: int = HISTORY) -> list[dict]:
    """Does the ranking survive a different oracle threshold?

    It has to be asked, because the oracle's threshold is a number of grey levels
    and **the clip is lossily compressed**. Frame 140 is the proof: its oracle
    mask contains a field of 8x8 blocks over the flat tarmac, where the codec
    allocated fewer bits than usual, and its foreground share is 8.0% against
    3.9-5.2% for the other eleven. That contamination is real and it is not
    removed here -- it is measured, and the ranking is shown to survive it.
    """
    frames = frames or FRAMES
    rows = []
    for multiple in multiples:
        row: dict[str, float] = {"oracle_multiple": multiple}
        shares = []
        for name in REAL_METHODS:
            ious = []
            for index in frames:
                past = load_frames(max(0, index - history), index)
                truth = oracle_mask(index, multiple=multiple)
                shares.append(float((truth > 0).mean()))
                ious.append(iou(METHODS[name](past, load_frame(index)), truth))
            row[name] = round(float(np.mean(ious)), 4)
        row["mean_foreground_share"] = round(float(np.mean(shares)), 4)
        rows.append(row)
    return rows


def oracle_frame_outliers(frames=None) -> list[dict]:
    """Per-frame oracle foreground, and how much of it is in person-sized blobs.

    A frame whose foreground is mostly scattered small components is a frame
    where the truth is compression noise rather than people.
    """
    rows = []
    for index in (frames or FRAMES):
        mask = oracle_mask(index)
        n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        total = float((mask > 0).sum())
        big = float(sum(stats[i, 4] for i in range(1, n) if stats[i, 4] >= 400))
        rows.append({
            "frame": int(index),
            "foreground_share": round(float((mask > 0).mean()), 4),
            "components": int(n - 1),
            "share_in_blobs_over_400px": round(big / max(total, 1.0), 4),
        })
    return rows


def sweep_threshold(values=None, swaps=SWAPS[:6], history: int = HISTORY) -> list[dict]:
    """The threshold the three simple methods share, swept around the measured one.

    Each is given the same value at each step, so the sweep cannot be read as one
    method being tuned better than another.
    """
    floor = noise_floor()
    values = values or tuple(round(floor * m, 2) for m in (0.25, 0.5, 1.0, 2.0, 4.0))
    simple = ("Frame difference", "Running average", "Median background")
    rows = []
    for t in values:
        row: dict[str, float] = {"threshold": t, "as_multiple_of_measured_floor":
                                 round(t / max(floor, EPS), 2)}
        for name in simple:
            ious = []
            for target, donor, rect in swaps:
                past = load_frames(max(0, target - history), target)
                composite, truth, _ = swap_composite(target, donor, rect)
                ious.append(iou(METHODS[name](past, composite, threshold=t), truth))
            row[name] = round(float(np.mean(ious)), 4)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# the signature visualisation
# --------------------------------------------------------------------------- #


def pixel_history(x: int, y: int, start: int = 0, length: int = 300) -> dict:
    """One pixel's value over time, with what each simple model calls background.

    This is the project's own picture. Every method here is a claim about what
    the flat part of this curve is, and the spikes are people crossing that
    pixel. A median sits on the flat part; a running average is dragged toward
    every spike and takes tens of frames to come back.
    """
    frames = load_frames(start, start + length)
    values = np.array([float(to_gray(f)[y, x]) for f in frames], np.float32)

    running = np.empty_like(values)
    background = values[0]
    for i, v in enumerate(values):
        background = 0.05 * v + 0.95 * background
        running[i] = background

    window = 25
    median = np.array([
        float(np.median(values[max(0, i - window):i + 1])) for i in range(len(values))
    ], np.float32)

    return {
        "x": x, "y": y, "start": start,
        "values": values,
        "running_average": running,
        "median": median,
        "noise_floor": noise_floor(),
    }
