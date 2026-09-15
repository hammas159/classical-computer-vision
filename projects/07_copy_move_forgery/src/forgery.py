"""Copy-move forgery detection: which pixels were pasted?

The question
------------
A region of an image is copied and pasted elsewhere to hide or duplicate
something. The forged region is *identical* to a real region of the same photo,
so it has the same noise, the same lighting and the same compression history —
which is precisely why it defeats most tampering detectors.

The generator pastes the patch itself, so the forgery mask is exact and detection
becomes a scoreable segmentation problem rather than an eyeball test.

> **The question that makes this more than a demo:** copy-move detection is easy
> when the pasted patch is an exact copy. Real forgeries rotate and scale it. How
> much rotation does each method survive?

Two families are compared:

* **Block matching** — slide a window over the image, describe each block, sort
  the descriptors, and look for near-duplicates at a consistent offset. Exact and
  slow; blind to rotation by construction.
* **Keypoint matching** — detect SIFT/ORB keypoints and match the image against
  *itself*, excluding trivial self-matches. Sparse and fast, and rotation
  invariant because the descriptors are.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray
from shared.metrics import dice, iou

EPS = 1e-6


# --------------------------------------------------------------------------- #
# block matching
# --------------------------------------------------------------------------- #


def _block_descriptors(gray: np.ndarray, block: int, stride: int):
    """Mean-and-moment descriptor per block, plus each block's position."""
    h, w = gray.shape
    positions, feats = [], []
    f = to_float(gray)
    for y in range(0, h - block + 1, stride):
        for x in range(0, w - block + 1, stride):
            patch = f[y : y + block, x : x + block]
            # a cheap, rotation-*sensitive* descriptor: quadrant means plus spread.
            # Quadrant means are what let it localise a duplicate precisely, and
            # also what make it fail the moment the copy is rotated.
            half = block // 2
            q = (
                patch[:half, :half].mean(),
                patch[:half, half:].mean(),
                patch[half:, :half].mean(),
                patch[half:, half:].mean(),
            )
            feats.append((*q, patch.mean(), patch.std()))
            positions.append((x, y))
    return np.asarray(feats, np.float32), np.asarray(positions, np.int32)


def detect_block_matching(
    img: np.ndarray,
    block: int = 16,
    stride: int = 8,
    tol: float = 0.02,
    min_offset: int = 24,
    min_votes: int = 4,
) -> np.ndarray:
    """Lexicographic block matching with offset voting.

    Sorting the descriptors puts near-identical blocks next to each other, so
    duplicates can be found in one linear pass instead of comparing every pair.

    **Offset voting is what removes the false positives.** Any smooth region —
    sky, a wall — produces thousands of similar blocks. A genuine copy-move
    produces many matching pairs that all share the *same* displacement vector,
    and requiring a minimum number of votes per offset discards the rest.
    """
    gray = to_gray(img)
    feats, pos = _block_descriptors(gray, block, stride)
    if len(feats) < 2:
        return np.zeros(gray.shape, np.uint8)

    order = np.lexsort(feats.T[::-1])
    feats, pos = feats[order], pos[order]

    offsets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(len(feats) - 1):
        for j in (i + 1, i + 2):  # neighbours in sorted order only
            if j >= len(feats):
                break
            if np.abs(feats[i] - feats[j]).max() > tol:
                continue
            dx, dy = int(pos[j][0] - pos[i][0]), int(pos[j][1] - pos[i][1])
            if abs(dx) + abs(dy) < min_offset:
                continue
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            offsets.setdefault((dx, dy), []).append((i, j))

    mask = np.zeros(gray.shape, np.uint8)
    for (dx, dy), pairs in offsets.items():
        if len(pairs) < min_votes:
            continue
        for i, j in pairs:
            for (x, y) in (pos[i], pos[j]):
                mask[y : y + block, x : x + block] = 255
    return mask


# --------------------------------------------------------------------------- #
# keypoint matching
# --------------------------------------------------------------------------- #


def _keypoint_forgery(
    img: np.ndarray, detector, ratio: float = 0.6, min_distance: int = 30, radius: int = 14
) -> np.ndarray:
    """Match an image against itself and mark consistent duplicate regions."""
    gray = to_gray(img)
    kp, desc = detector.detectAndCompute(gray, None)
    mask = np.zeros(gray.shape, np.uint8)
    if desc is None or len(kp) < 4:
        return mask

    desc = desc.astype(np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # k=3: the nearest neighbour of a keypoint is always itself, so the first
    # match is discarded and the ratio test is applied to the next two
    matches = matcher.knnMatch(desc, desc, k=3)

    for m in matches:
        if len(m) < 3:
            continue
        _self, first, second = m
        if first.distance > ratio * second.distance:
            continue
        p1 = np.array(kp[first.queryIdx].pt)
        p2 = np.array(kp[first.trainIdx].pt)
        if np.linalg.norm(p1 - p2) < min_distance:
            continue  # neighbouring keypoints on the same object, not a copy
        for p in (p1, p2):
            cv2.circle(mask, (int(p[0]), int(p[1])), radius, 255, -1)

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def detect_sift(img: np.ndarray) -> np.ndarray:
    """SIFT keypoints matched against themselves.

    SIFT descriptors are scale and rotation invariant, so this should survive a
    rotated paste that block matching cannot. Free since OpenCV 4.4 — the patent
    expired and it needs no contrib build.
    """
    return _keypoint_forgery(img, cv2.SIFT.create(nfeatures=3000))


def detect_orb(img: np.ndarray) -> np.ndarray:
    """ORB keypoints — binary descriptors, far faster than SIFT.

    ORB is rotation invariant but not scale invariant, which predicts a specific
    pattern of results: it should hold up under rotation and fall over under
    scaling.
    """
    orb = cv2.ORB.create(nfeatures=3000)
    gray = to_gray(img)
    kp, desc = orb.detectAndCompute(gray, None)
    mask = np.zeros(gray.shape, np.uint8)
    if desc is None or len(kp) < 4:
        return mask
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(desc, desc, k=3)
    for m in matches:
        if len(m) < 3:
            continue
        _self, first, second = m
        if first.distance > 0.75 * max(second.distance, 1):
            continue
        p1 = np.array(kp[first.queryIdx].pt)
        p2 = np.array(kp[first.trainIdx].pt)
        if np.linalg.norm(p1 - p2) < 30:
            continue
        for p in (p1, p2):
            cv2.circle(mask, (int(p[0]), int(p[1])), 14, 255, -1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def detect_zero_baseline(img: np.ndarray) -> np.ndarray:
    """Predict "nothing was forged" — the control every detector must beat.

    On an image where the forgery covers a few percent of pixels, this scores a
    high pixel accuracy and an IoU of zero. It is in the table to make the
    difference between those two numbers impossible to ignore.
    """
    return np.zeros(img.shape[:2], np.uint8)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Block matching": detect_block_matching,
    "SIFT self-match": detect_sift,
    "ORB self-match": detect_orb,
    "Predict nothing (control)": detect_zero_baseline,
}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "grass", "brick")
ANGLE_LEVELS = (0.0, 5.0, 15.0, 30.0, 90.0)
SCALE_LEVELS = (1.0, 0.9, 0.8, 1.2, 1.5)


def _score(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p, t = pred > 0, truth > 0
    tp = float((p & t).sum())
    return {
        "iou": iou(pred, truth),
        "dice": dice(pred, truth),
        "precision": tp / max(float(p.sum()), 1.0),
        "recall": tp / max(float(t.sum()), 1.0),
        "pixel_accuracy": float((p == t).mean()),
    }


def evaluate_methods(angle: float = 0.0, scale: float = 1.0, images=IMAGES, runs: int = 1):
    """Score every detector on a forgery with the given rotation and scale."""
    from shared import io, synth

    acc = {n: {k: [] for k in ("iou", "dice", "precision", "recall", "pixel_accuracy", "ms")}
           for n in METHODS}

    for i, name in enumerate(images):
        clean = io.sample(name)
        f = synth.copy_move_forgery(clean, size=96, angle_deg=angle, scale=scale, seed=i)
        for method, fn in METHODS.items():
            pred, timing = timeit(lambda g=fn: g(f.image), runs=runs, warmup=0)
            for k, v in _score(pred, f.mask).items():
                acc[method][k].append(v)
            acc[method]["ms"].append(timing.median_ms)

    return [
        {
            "method": m,
            "iou": round(float(np.mean(a["iou"])), 4),
            "dice": round(float(np.mean(a["dice"])), 4),
            "precision": round(float(np.mean(a["precision"])), 4),
            "recall": round(float(np.mean(a["recall"])), 4),
            "pixel_accuracy": round(float(np.mean(a["pixel_accuracy"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for m, a in acc.items()
    ]


def sweep_rotation(images=IMAGES, angles=ANGLE_LEVELS):
    """How much rotation does each method survive? The project's central sweep."""
    rows = []
    for angle in angles:
        scored = evaluate_methods(angle=angle, images=images)
        row: dict[str, float] = {"angle_deg": angle}
        for r in scored:
            row[r["method"]] = r["iou"]
        rows.append(row)
    return rows


def sweep_scale(images=IMAGES, scales=SCALE_LEVELS):
    """And how much rescaling? ORB is rotation- but not scale-invariant."""
    rows = []
    for s in scales:
        scored = evaluate_methods(scale=s, images=images)
        row: dict[str, float] = {"scale": s}
        for r in scored:
            row[r["method"]] = r["iou"]
        rows.append(row)
    return rows


def detect(img: np.ndarray, method: str = "SIFT self-match") -> np.ndarray:
    """Run one named detector, for the UI and for inference."""
    return METHODS[method](img)
