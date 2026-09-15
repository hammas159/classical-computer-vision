"""Pedestrian detection with HOG + SVM: the last pre-deep-learning detector.

The question
------------
HOG + linear SVM (Dalal & Triggs, 2005) was the state of the art for pedestrian
detection until deep learning replaced it. OpenCV ships the trained weights, so
it runs anywhere with no download.

> **The claim under test:** the detector's weakness is not accuracy at a single
> threshold, it is the **precision-recall trade-off** — and the parameter that
> matters most is not the SVM threshold but ``scale``, the pyramid step, which
> controls how many candidate windows exist at all.

One pre-trained component is used and it is stated, not hidden: OpenCV's default
people detector. It is a **linear SVM**, trained by someone else on INRIA Person,
and it is not a neural network. Nothing here is trained.

Because the scenes are composited here, every pedestrian's true box is known
exactly, so precision, recall and IoU are measurable — which is what turns this
from a demo into a study.

The HOG descriptor itself is also computed directly, so the thing being fed to
the SVM can be shown rather than described.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8

EPS = 1e-9

#: The detector window Dalal & Triggs used, and what the bundled weights expect.
#: Feeding a different window size silently produces nonsense, because the SVM's
#: weight vector has a fixed length tied to this geometry.
WINDOW = (64, 128)
BLOCK = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL = (8, 8)
N_BINS = 9


_DETECTOR = None


def detector() -> cv2.HOGDescriptor:
    """The bundled HOG + linear SVM people detector, loaded once.

    Loading costs a few milliseconds; doing it inside a timed function would put
    setup into the benchmark and make every timing meaningless.
    """
    global _DETECTOR
    if _DETECTOR is None:
        hog = cv2.HOGDescriptor(WINDOW, BLOCK, BLOCK_STRIDE, CELL, N_BINS)
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _DETECTOR = hog
    return _DETECTOR


# --------------------------------------------------------------------------- #
# the descriptor, computed directly
# --------------------------------------------------------------------------- #


def hog_cells(gray: np.ndarray, cell: int = 8, bins: int = 9):
    """Orientation histograms per cell — the HOG descriptor before normalisation.

    Gradients are binned into **unsigned** orientations (0-180, not 0-360), so a
    dark-to-light edge and a light-to-dark edge land in the same bin. That is
    deliberate: a person's outline against a light or a dark background should
    produce the same descriptor.

    Returns an (rows, cols, bins) array.
    """
    g = to_float(gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)

    h, w = g.shape
    rows, cols = h // cell, w // cell
    out = np.zeros((rows, cols, bins), np.float32)
    bin_width = 180.0 / bins

    for r in range(rows):
        for c in range(cols):
            m = mag[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell].ravel()
            a = ang[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell].ravel()
            # linear interpolation between neighbouring bins, which is what stops
            # the descriptor jumping when an edge rotates slightly
            idx = a / bin_width
            lo = np.floor(idx).astype(int) % bins
            hi = (lo + 1) % bins
            frac = idx - np.floor(idx)
            np.add.at(out[r, c], lo, m * (1 - frac))
            np.add.at(out[r, c], hi, m * frac)
    return out


def hog_visualisation(gray: np.ndarray, cell: int = 8, bins: int = 9, scale: float = 4.0):
    """Render the cell histograms as oriented line segments, for the figures."""
    cells = hog_cells(gray, cell, bins)
    rows, cols, _ = cells.shape
    canvas = np.zeros((rows * cell, cols * cell), np.float32)
    peak = max(float(cells.max()), EPS)

    for r in range(rows):
        for c in range(cols):
            cy, cx = r * cell + cell // 2, c * cell + cell // 2
            for b in range(bins):
                strength = cells[r, c, b] / peak
                if strength < 0.05:
                    continue
                angle = np.deg2rad(b * 180.0 / bins)
                dx = np.cos(angle) * cell * 0.5 * strength
                dy = np.sin(angle) * cell * 0.5 * strength
                cv2.line(
                    canvas,
                    (int(cx - dx), int(cy - dy)),
                    (int(cx + dx), int(cy + dy)),
                    float(strength),
                    1,
                )
    return to_uint8(np.clip(canvas * scale, 0, 1))


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def detect(img: np.ndarray, win_stride=(8, 8), padding=(16, 16), scale: float = 1.05,
           hit_threshold: float = 0.0, nms_threshold: float = 0.45):
    """Run the sliding-window detector across an image pyramid.

    ``scale`` is the pyramid step and it is the parameter that dominates
    everything: 1.01 evaluates hundreds of scales and is very slow; 1.2 evaluates
    a handful and misses people whose size falls between levels.

    Returns ``(boxes, scores)`` with boxes as (x, y, w, h).
    """
    rects, weights = detector().detectMultiScale(
        to_gray(img), winStride=win_stride, padding=padding,
        scale=scale, hitThreshold=hit_threshold,
    )
    if len(rects) == 0:
        return np.zeros((0, 4), np.int32), np.zeros(0, np.float32)

    scores = np.array([float(w) for w in np.asarray(weights).ravel()], np.float32)
    boxes = np.asarray(rects, np.int32)

    keep = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), score_threshold=float(hit_threshold),
        nms_threshold=float(nms_threshold),
    )
    if len(keep) == 0:
        return np.zeros((0, 4), np.int32), np.zeros(0, np.float32)
    keep = np.asarray(keep).ravel()
    return boxes[keep], scores[keep]


# --------------------------------------------------------------------------- #
# synthetic pedestrians
# --------------------------------------------------------------------------- #


def draw_pedestrian(canvas: np.ndarray, x: int, y: int, height: int, colour: int = 40):
    """Draw a simple upright human silhouette.

    HOG responds to the *shape of the outline*, not to texture or colour, so a
    correctly proportioned silhouette is enough to trigger the real detector.
    The proportions matter: roughly 1:2 width to height with a head, torso and
    legs, which is what the trained weights encode.
    """
    w = max(6, int(height * 0.38))
    head_r = max(3, int(height * 0.11))
    cx = x + w // 2

    cv2.circle(canvas, (cx, y + head_r), head_r, colour, -1)
    torso_top = y + head_r * 2
    torso_bottom = y + int(height * 0.60)
    cv2.rectangle(canvas, (cx - w // 3, torso_top), (cx + w // 3, torso_bottom), colour, -1)

    leg_w = max(2, w // 5)
    cv2.rectangle(canvas, (cx - w // 3, torso_bottom), (cx - w // 3 + leg_w, y + height), colour, -1)
    cv2.rectangle(canvas, (cx + w // 3 - leg_w, torso_bottom), (cx + w // 3, y + height), colour, -1)
    cv2.rectangle(canvas, (cx - w // 2, torso_top), (cx - w // 3, torso_bottom - 4), colour, -1)
    cv2.rectangle(canvas, (cx + w // 3, torso_top), (cx + w // 2, torso_bottom - 4), colour, -1)

    return (x, y, w, height)


def make_scene(n_people: int = 3, size: int = 480, height_range=(140, 260),
               clutter: int = 12, noise_sigma: float = 0.0, seed: int = 0):
    """A scene with known pedestrian boxes on a cluttered background.

    Returns ``(image, boxes)``. The clutter is vertical rectangles, chosen
    deliberately: they have the same dominant gradient orientation as a standing
    person and are the classic HOG false positive.
    """
    from shared import synth

    rng = np.random.default_rng(seed)
    img = np.full((size, size), 200, np.uint8)
    img = cv2.GaussianBlur(
        to_uint8(to_float(img) + rng.normal(0, 0.04, img.shape)), (0, 0), 2.0
    )

    for _ in range(clutter):
        x = int(rng.integers(0, size - 40))
        y = int(rng.integers(0, size - 120))
        h = int(rng.integers(60, 160))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(12, 34)), y + h),
                      int(rng.integers(60, 170)), -1)

    boxes = []
    for _ in range(n_people):
        h = int(rng.integers(*height_range))
        x = int(rng.integers(20, max(21, size - 80)))
        y = int(rng.integers(10, max(11, size - h - 10)))
        boxes.append(draw_pedestrian(img, x, y, h))

    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return img, np.asarray(boxes, np.int32)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, EPS))


def match_detections(predicted, truth, iou_threshold: float = 0.5):
    """Greedy matching by IoU. Returns (tp, fp, fn, matched IoUs).

    Each true box may be matched at most once, so two overlapping detections of
    the same person count as one true positive and one **false** positive — which
    is why non-maximum suppression matters and why omitting it inflates recall.
    """
    used = set()
    tp, ious = 0, []
    for p in predicted:
        best, best_iou = None, 0.0
        for i, t in enumerate(truth):
            if i in used:
                continue
            score = box_iou(p, t)
            if score > best_iou:
                best, best_iou = i, score
        if best is not None and best_iou >= iou_threshold:
            used.add(best)
            tp += 1
            ious.append(best_iou)
    fp = len(predicted) - tp
    fn = len(truth) - tp
    return tp, fp, fn, ious


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

SCALES = (1.01, 1.03, 1.05, 1.1, 1.2, 1.4)
THRESHOLDS = (-1.0, -0.5, 0.0, 0.3, 0.6, 1.0)
HEIGHT_RANGES = ((80, 110), (110, 150), (150, 220), (220, 320))
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0)
OCCLUSIONS = (0.0, 0.1, 0.25, 0.4)


def evaluate(scenes: int = 8, scale: float = 1.05, hit_threshold: float = 0.0,
             noise_sigma: float = 0.0, height_range=(140, 260), runs: int = 1):
    """Precision, recall and cost at one operating point."""
    tp = fp = fn = 0
    ious, ms = [], []
    for seed in range(scenes):
        img, truth = make_scene(
            noise_sigma=noise_sigma, height_range=height_range, seed=seed
        )
        (boxes, _), timing = timeit(
            lambda: detect(img, scale=scale, hit_threshold=hit_threshold), runs=runs, warmup=0
        )
        a, b, c, matched = match_detections(boxes, truth)
        tp += a
        fp += b
        fn += c
        ious.extend(matched)
        ms.append(timing.median_ms)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, EPS), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "median_ms": round(float(np.median(ms)), 1),
    }


def sweep_threshold(scenes: int = 8, thresholds=THRESHOLDS, scale: float = 1.05):
    """The precision-recall curve, which is the only honest summary of a detector.

    A single accuracy number for a detector is meaningless without stating the
    threshold it was measured at — this makes the whole trade-off visible.
    """
    rows = []
    for t in thresholds:
        r = evaluate(scenes=scenes, scale=scale, hit_threshold=t)
        rows.append({"hit_threshold": t, **r})
    return rows


def sweep_scale(scenes: int = 8, scales=SCALES, hit_threshold: float = 0.0):
    """The pyramid step: the parameter that actually decides the result.

    Smaller steps mean more scales searched, so higher recall and much more time.
    The accuracy-versus-milliseconds trade here should be far steeper than the
    one from the SVM threshold.
    """
    rows = []
    for s in scales:
        r = evaluate(scenes=scenes, scale=s, hit_threshold=hit_threshold)
        rows.append({"scale": s, **r})
    return rows


def sweep_person_size(scenes: int = 8, ranges=HEIGHT_RANGES):
    """The detector window is 64x128, so people much smaller than that cannot be
    resolved. This finds the practical minimum height."""
    rows = []
    for lo, hi in ranges:
        r = evaluate(scenes=scenes, height_range=(lo, hi))
        rows.append({"height_range": f"{lo}-{hi}", **r})
    return rows


def sweep_noise(scenes: int = 8, levels=NOISE_LEVELS):
    """HOG normalises gradients per block, so it should be fairly noise tolerant."""
    rows = []
    for sigma in levels:
        r = evaluate(scenes=scenes, noise_sigma=sigma)
        rows.append({"noise_sigma": sigma, **r})
    return rows


def occlusion_test(scenes: int = 8, fractions=OCCLUSIONS):
    """HOG is a *holistic* template: it scores one whole window at a time.

    So partial occlusion should hurt it badly — unlike a part-based model, it
    cannot recognise a person from the visible half. Blocking a fraction of each
    person's box measures exactly how badly.
    """
    rows = []
    for frac in fractions:
        tp = fp = fn = 0
        for seed in range(scenes):
            img, truth = make_scene(seed=seed)
            if frac > 0:
                for (x, y, w, h) in truth:
                    cut = int(h * frac)
                    cv2.rectangle(img, (x, y + h - cut), (x + w, y + h), 210, -1)
            boxes, _ = detect(img)
            a, b, c, _ = match_detections(boxes, truth)
            tp += a
            fp += b
            fn += c
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "occluded_fraction": frac,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    return rows


def nms_effect(scenes: int = 8, thresholds=(0.1, 0.3, 0.45, 0.7, 1.0)):
    """Non-maximum suppression: the difference between one detection and five.

    Without it, a sliding-window detector fires on every window overlapping a
    person. Recall looks excellent and precision collapses, which is exactly the
    wrong trade to report.
    """
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for seed in range(scenes):
            img, truth = make_scene(seed=seed)
            boxes, _ = detect(img, nms_threshold=t)
            a, b, c, _ = match_detections(boxes, truth)
            tp += a
            fp += b
            fn += c
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "nms_threshold": t,
                "detections": tp + fp,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    return rows


def descriptor_size():
    """How many numbers the SVM actually sees, worked out from the geometry.

    A 64x128 window with 8x8 cells, 2x2 blocks at stride 8 and 9 bins gives
    7 x 15 x 4 x 9 = 3780 features. Stating it makes concrete how small this
    model is next to anything learned end to end.
    """
    win_w, win_h = WINDOW
    bw, bh = BLOCK
    sx, sy = BLOCK_STRIDE
    cw, ch = CELL
    blocks_x = (win_w - bw) // sx + 1
    blocks_y = (win_h - bh) // sy + 1
    cells_per_block = (bw // cw) * (bh // ch)
    return {
        "window": f"{win_w}x{win_h}",
        "blocks_x": blocks_x,
        "blocks_y": blocks_y,
        "cells_per_block": cells_per_block,
        "bins": N_BINS,
        "total_features": blocks_x * blocks_y * cells_per_block * N_BINS,
        "svm_parameters": blocks_x * blocks_y * cells_per_block * N_BINS + 1,
    }


def draw_boxes(img: np.ndarray, boxes, colour=(0, 200, 0), thickness: int = 2):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), colour, thickness)
    return out
