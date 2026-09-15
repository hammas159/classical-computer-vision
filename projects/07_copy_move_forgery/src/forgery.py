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
> much rotation does each method survive, and what does surviving it cost?

Two families are compared:

* **Block matching** — slide a window over the image, describe each block, sort
  the descriptors, and look for near-duplicates at a consistent offset. Exact and
  slow; blind to rotation by construction.
* **Keypoint matching** — detect SIFT/ORB keypoints and match the image against
  *itself*, excluding trivial self-matches. Sparse and fast, and rotation
  invariant because the descriptors are.

Every detector here is really two decisions, and the project separates them:

1. **What to match** — blocks, SIFT keypoints, ORB keypoints.
2. **What hypothesis to verify** — nothing (paint blobs on the matches), a pure
   translation, or a full similarity transform.

The second decision turns out to matter more than the first. `SIFT +
translation verify` and `SIFT + similarity verify` share every keypoint and every
match and differ only in step 2, and they score 0.794 and 0.000 respectively at 90
degrees — in opposite directions from their scores on an unrotated copy.

Two things this module scores that a copy-move demo usually does not
--------------------------------------------------------------------
* **Both copies** (:func:`evaluate_methods` uses ``mask_both``). After the paste
  the source and the destination are identical, so no pixel-based method can say
  which is the forgery. Scoring against the pasted region alone caps precision at
  about 0.5 and measures that ambiguity rather than the detector.
* **Untampered images** (:func:`evaluate_false_alarms`). A detector that marks
  part of every photograph is not a detector. This is where the rotation-robust
  methods stop looking free.
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
    """Mean-and-moment descriptor per block, plus each block's position.

    Six numbers per block: four quadrant means, the block mean, and the block
    standard deviation. Deliberately rotation-*sensitive* — the quadrant means
    are what localise a duplicate precisely, and also what make the method fail
    the moment the copy is rotated. That failure is the point of having it here.

    Every one of the six is a box filter, so they are computed for **every**
    pixel position at once rather than in a Python loop over blocks. That is what
    makes ``stride=1`` affordable, and `stride=1` is not optional: see
    :func:`detect_block_matching`.
    """
    f = to_float(gray)
    half = block // 2
    anchored = dict(anchor=(0, 0), borderType=cv2.BORDER_ISOLATED)

    mean_half = cv2.boxFilter(f, cv2.CV_32F, (half, half), **anchored)
    mean_full = cv2.boxFilter(f, cv2.CV_32F, (block, block), **anchored)
    sq_full = cv2.boxFilter(f * f, cv2.CV_32F, (block, block), **anchored)
    std_full = np.sqrt(np.maximum(sq_full - mean_full * mean_full, 0.0))

    h, w = gray.shape
    ys = np.arange(0, h - block + 1, stride)
    xs = np.arange(0, w - block + 1, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    feats = np.stack(
        [
            mean_half[yy, xx],                  # top-left quadrant
            mean_half[yy, xx + half],           # top-right
            mean_half[yy + half, xx],           # bottom-left
            mean_half[yy + half, xx + half],    # bottom-right
            mean_full[yy, xx],
            std_full[yy, xx],
        ],
        axis=-1,
    ).reshape(-1, 6).astype(np.float32)
    positions = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.int32)
    return feats, positions


def detect_block_matching(
    img: np.ndarray,
    block: int = 16,
    stride: int = 1,
    tol: float = 0.002,
    min_offset: int = 24,
    min_votes: int = 400,
    neighbours: int = 4,
) -> np.ndarray:
    """Lexicographic block matching with offset voting.

    Sorting the descriptors puts near-identical blocks next to each other, so
    duplicates can be found in one linear pass over the sorted array instead of
    comparing every pair against every other.

    **Offset voting is what removes the false positives.** Any smooth region —
    sky, a wall — produces thousands of similar blocks. A genuine copy-move
    produces many matching pairs that all share the *same* displacement vector,
    and requiring ``min_votes`` pairs per offset discards the rest.

    **``stride`` must be 1.** This is the single thing that decides whether the
    method works at all, and it is not a speed/quality dial. A block at
    ``(x, y)`` in the source region has its duplicate at ``(x + dx, y + dy)``,
    and if the stride does not divide both ``dx`` and ``dy`` then the copy is
    simply never sampled — the two blocks are offset by a sub-stride phase and
    describe different pixels. With ``stride=8`` and a paste offset that is
    arbitrary, the chance both components align is 1 in 64, and the method
    reported **0.068 IoU on an exact copy** for that reason alone. See the README.
    """
    gray = to_gray(img)
    feats, pos = _block_descriptors(gray, block, stride)
    if len(feats) < 2:
        return np.zeros(gray.shape, np.uint8)

    order = np.lexsort(feats.T[::-1])
    feats, pos = feats[order], pos[order]

    offsets: dict[tuple[int, int], list[int]] = {}
    for step in range(1, neighbours + 1):
        a, b = feats[:-step], feats[step:]
        close = np.abs(a - b).max(axis=1) <= tol
        pa, pb = pos[:-step][close], pos[step:][close]
        d = pb - pa
        keep = np.abs(d).sum(axis=1) >= min_offset
        pa, pb, d = pa[keep], pb[keep], d[keep]
        # fold (dx, dy) and (-dx, -dy) together: the pair is symmetric
        flip = (d[:, 0] < 0) | ((d[:, 0] == 0) & (d[:, 1] < 0))
        d[flip] *= -1
        for k in range(len(d)):
            offsets.setdefault((int(d[k, 0]), int(d[k, 1])), []).append(
                (int(pa[k, 0]), int(pa[k, 1]), int(pb[k, 0]), int(pb[k, 1]))
            )

    mask = np.zeros(gray.shape, np.uint8)
    for pairs in offsets.values():
        if len(pairs) < min_votes:
            continue
        for ax, ay, bx, by in pairs:
            mask[ay : ay + block, ax : ax + block] = 255
            mask[by : by + block, bx : bx + block] = 255
    return mask


# --------------------------------------------------------------------------- #
# keypoint matching
# --------------------------------------------------------------------------- #


def self_matches(img: np.ndarray, detector, norm: int, ratio: float, min_distance: int):
    """Match a keypoint set against itself, returning surviving ``(p1, p2)`` pairs.

    Three filters, each removing a different kind of nonsense:

    * **k=3, discard the first.** A descriptor's nearest neighbour in its own set
      is always itself, at distance zero. Matching k=2 and keeping the best match
      would return every keypoint paired with itself.
    * **Lowe's ratio test** on the *second* and *third* neighbours. A real
      duplicate is distinctly closer than the next candidate; a repeated texture
      is not.
    * **A minimum separation.** Two keypoints 5 px apart on the same corner are
      not a copy-move, they are one feature detected twice.
    """
    gray = to_gray(img)
    kp, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kp) < 4:
        return []

    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(desc, desc, k=3)

    pairs = []
    for m in matches:
        if len(m) < 3:
            continue
        _self, first, second = m
        if first.distance > ratio * max(second.distance, 1e-6):
            continue
        p1 = np.array(kp[first.queryIdx].pt)
        p2 = np.array(kp[first.trainIdx].pt)
        if np.linalg.norm(p1 - p2) < min_distance:
            continue
        pairs.append((p1, p2))
    return pairs


def verify_by_offset(
    img: np.ndarray,
    pairs,
    tol: float = 6.0,
    min_votes: int = 4,
    patch: int = 9,
    threshold: float = 0.015,
    min_area_frac: float = 0.004,
) -> np.ndarray:
    """Turn sparse matches into a dense region mask, by *checking* the offset.

    The keypoints say "something around here was duplicated, displaced by about
    (dx, dy)". That is a hypothesis about the whole image, and it is cheap to
    test everywhere at once: shift the image by the offset, and mark every pixel
    whose neighbourhood still matches after the shift.

    This is what turns the detector from "circles where keypoints happened to
    fire" into an actual segmentation. Blob-painting a fixed radius around each
    matched keypoint reached **0.18 IoU** even on an exact copy, because the
    keypoints land on corners and the duplicated *region* is mostly not corners.
    Dense verification reads the region boundary off the image itself.

    Each surviving offset is voted for; offsets with fewer than ``min_votes``
    supporting keypoint pairs are not tested, because testing an arbitrary shift
    on a self-similar image finds self-similarity.
    """
    gray = to_float(to_gray(img))
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    if not pairs:
        return mask

    # Vote for offsets, folding (dx, dy) and (-dx, -dy) together: the pair is
    # symmetric and there is nothing in the image to say which end is the copy.
    #
    # The bucket is only used to GROUP; the offset actually tested is the median
    # of the members. Testing the bucket centre instead misaligns the comparison
    # by up to tol/2 px, and a 3 px misalignment makes a textured region disagree
    # with itself — which is how this verifier first scored 0.18 IoU on an exact
    # copy while its own arithmetic was correct.
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p1, p2 in pairs:
        dx, dy = int(round(p2[0] - p1[0])), int(round(p2[1] - p1[1]))
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        key = (int(round(dx / tol)), int(round(dy / tol)))
        buckets.setdefault(key, []).append((dx, dy))

    kernel = np.ones((patch, patch), np.float32) / (patch * patch)
    for members in buckets.values():
        if len(members) < min_votes:
            continue
        arr = np.asarray(members)
        dx, dy = int(np.median(arr[:, 0])), int(np.median(arr[:, 1]))
        if abs(dx) + abs(dy) < 16:
            continue

        # shift by (dx, dy) and compare; only the overlap region is meaningful
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(gray, m, (w, h), borderValue=np.nan)
        diff = np.abs(gray - shifted)
        valid = np.isfinite(diff)
        diff = np.where(valid, diff, 1.0)
        local = cv2.filter2D(diff, -1, kernel)
        hit = ((local < threshold) & valid).astype(np.uint8)

        # a duplicated region is contiguous; scattered agreement is coincidence
        hit = cv2.morphologyEx(hit, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hit, 8)
        min_area = int(min_area_frac * h * w)
        keep = np.zeros_like(hit)
        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == label] = 1

        mask |= keep * 255
        # the matching region's partner is the same region shifted back
        back = cv2.warpAffine(keep * 255, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h))
        mask |= (back > 127).astype(np.uint8) * 255

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


# SIFT's default contrastThreshold of 0.04 is tuned for matching two photographs,
# where a few hundred strong keypoints is plenty. Copy-move needs keypoints
# *inside one 96x96 region*, and at the default the whole duplicated patch yielded
# 7 of them — only 3 survived the ratio test, below the 4 votes the verifier
# needs, so the detector returned an empty mask and reported no forgery.
#
#   contrastThreshold   keypoints   in the patch   usable pairs
#   0.04 (default)           1272              7              3   <- silent failure
#   0.02                     1719             15              8
#   0.01                     2023             29             22   <- used here
#   0.005                    2273             40             25
#
# 0.01 rather than 0.005 because the last step buys 3 pairs for 250 keypoints.
SIFT_CONTRAST_THRESHOLD = 0.01


def make_sift():
    """SIFT configured for copy-move, not for photo matching. See the table above."""
    return cv2.SIFT.create(
        nfeatures=0, contrastThreshold=SIFT_CONTRAST_THRESHOLD, edgeThreshold=16
    )


def verify_by_transform(
    img: np.ndarray,
    pairs,
    ransac_thresh: float = 3.0,
    min_inliers: int = 4,
    patch: int = 9,
    threshold: float = 0.022,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.30,
    attempts: int = 4,
) -> np.ndarray:
    """Dense verification under a **similarity** transform, not just a shift.

    :func:`verify_by_offset` tests the hypothesis "this region was translated by
    (dx, dy)". That is the right hypothesis only for an unrotated paste. Once the
    copy is rotated, the two regions are related by a rotation *and* a
    translation, a shift-and-compare finds nothing, and the method scores
    **0.061 IoU at 15 degrees** — despite its own keypoint matches still being
    correct, which is the frustrating part.

    So the transform is estimated instead of assumed: RANSAC over the matched
    keypoint pairs fits a partial affine (translation, rotation, uniform scale —
    four degrees of freedom, which is exactly what a copy-move paste is), the
    image is warped by it, and the comparison proceeds as before.

    This is the step that makes "SIFT is rotation invariant" true of the
    *detector* rather than only of its descriptors.
    """
    gray = to_float(to_gray(img))
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    if len(pairs) < min_inliers:
        return mask

    kernel = np.ones((patch, patch), np.float32) / (patch * patch)
    min_area = int(min_area_frac * h * w)
    max_area = int(max_area_frac * h * w)
    remaining = list(pairs)

    # More than one hypothesis, because the biggest consensus is not always the
    # forgery. On `grass`, 818 self-matches from the repeating texture outvoted
    # the real paste and RANSAC returned a 0.01-degree rotation -- a perfectly
    # good description of grass, and not the answer. Fitting, testing, then
    # discarding those inliers and refitting finds the real transform underneath.
    for _ in range(attempts):
        if len(remaining) < min_inliers:
            break
        src = np.float32([p[0] for p in remaining]).reshape(-1, 1, 2)
        dst = np.float32([p[1] for p in remaining]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=5000
        )
        if matrix is None or inliers is None or int(inliers.sum()) < min_inliers:
            break

        flat = inliers.ravel().astype(bool)
        remaining = [p for p, is_in in zip(remaining, flat) if not is_in]

        warped = cv2.warpAffine(gray, matrix, (w, h), borderValue=float("nan"))
        diff = np.abs(gray - warped)
        valid = np.isfinite(diff)
        local = cv2.filter2D(np.where(valid, diff, 1.0), -1, kernel)
        hit = ((local < threshold) & valid).astype(np.uint8)

        hit = cv2.morphologyEx(hit, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hit, 8)
        keep = np.zeros_like(hit)
        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            # too small is noise; too large is the whole image agreeing with a
            # near-identity transform, which describes a flat sky, not a forgery
            if min_area <= area <= max_area:
                keep[labels == label] = 1
        if not keep.any():
            continue

        mask |= keep * 255
        # the partner region is this one carried back through the same transform
        inverse = cv2.invertAffineTransform(matrix)
        back = cv2.warpAffine(keep * 255, inverse, (w, h))
        mask |= (back > 127).astype(np.uint8) * 255

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def detect_sift(img: np.ndarray) -> np.ndarray:
    """SIFT keypoints matched against themselves, then verified under a similarity.

    SIFT descriptors are scale and rotation invariant, and because the
    verification step fits a transform rather than assuming a shift, the whole
    detector is too. Free since OpenCV 4.4 — the patent expired and it needs no
    contrib build.
    """
    return verify_by_transform(img, self_matches(img, make_sift(), cv2.NORM_L2, 0.6, 30))


def detect_sift_translation_only(img: np.ndarray) -> np.ndarray:
    """The same SIFT matches, verified by a pure **translation**.

    In the table to isolate one decision. It matches `detect_sift` almost exactly
    on an unrotated paste and collapses the moment the copy is turned — so the
    difference between the two rows is entirely the choice of hypothesis in the
    verifier, not the choice of descriptor.
    """
    return verify_by_offset(img, self_matches(img, make_sift(), cv2.NORM_L2, 0.6, 30))


def detect_orb(img: np.ndarray) -> np.ndarray:
    """ORB keypoints — binary descriptors, far faster than SIFT.

    ORB is rotation invariant but **not** scale invariant, which predicts a
    specific pattern of results: it should hold up under rotation and fall over
    under scaling. That prediction is what `sweep_scale` tests.

    ``fastThreshold=5`` for the same reason SIFT's contrast threshold is lowered:
    the default is tuned for finding a few strong corners across a whole scene,
    not many inside one small region.
    """
    orb = cv2.ORB.create(nfeatures=6000, fastThreshold=5)
    return verify_by_transform(img, self_matches(img, orb, cv2.NORM_HAMMING, 0.75, 30))


def detect_sift_keypoints_only(img: np.ndarray, radius: int = 14) -> np.ndarray:
    """The same SIFT matches, painted as fixed-radius blobs — no dense verification.

    Kept as a method rather than deleted, because it is the version this project
    started with and the gap between the two rows is the largest single
    improvement in the table. Keypoints land on corners; a duplicated *region* is
    mostly not corners, so painting circles around the matches can only ever
    approximate the region's shape — and the radius that does it best is a
    property of the patch size, which a real detector does not know.
    """
    pairs = self_matches(img, make_sift(), cv2.NORM_L2, 0.6, 30)
    mask = np.zeros(img.shape[:2], np.uint8)
    for p1, p2 in pairs:
        for p in (p1, p2):
            cv2.circle(mask, (int(p[0]), int(p[1])), radius, 255, -1)
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
    "SIFT + similarity verify": detect_sift,
    "ORB + similarity verify": detect_orb,
    "SIFT + translation verify": detect_sift_translation_only,
    "SIFT blobs (no verify)": detect_sift_keypoints_only,
    "Predict nothing (control)": detect_zero_baseline,
}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "grass", "brick")
ANGLE_LEVELS = (0.0, 2.0, 5.0, 15.0, 30.0, 45.0, 90.0)
SCALE_LEVELS = (0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5)
SIZE_LEVELS = (24, 32, 48, 64, 96, 128, 160)


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


def evaluate_methods(
    angle: float = 0.0, scale: float = 1.0, size: int = 96, images=IMAGES, runs: int = 1
):
    """Score every detector on a forgery with the given rotation, scale and size.

    Scored against ``mask_both`` — the pasted region **and** the region it came
    from. See :class:`shared.synth.Forgery`: after the paste the two are
    identical, so no pixel-based method can say which is the copy, and scoring
    against the pasted region alone caps precision near 0.5 for every method.
    """
    from shared import io, synth

    acc = {n: {k: [] for k in ("iou", "dice", "precision", "recall", "pixel_accuracy", "ms")}
           for n in METHODS}

    for i, name in enumerate(images):
        clean = io.sample(name)
        f = synth.copy_move_forgery(clean, size=size, angle_deg=angle, scale=scale, seed=i)
        for method, fn in METHODS.items():
            pred, timing = timeit(lambda g=fn: g(f.image), runs=runs, warmup=0)
            for k, v in _score(pred, f.mask_both).items():
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


def sweep_size(images=IMAGES, sizes=SIZE_LEVELS):
    """How small a forgery is still findable?

    Both families have a hard floor built into them, and neither floor is a
    tuning accident:

    * Block matching needs ``min_votes`` blocks sharing one offset, and a
      ``s x s`` paste only contains ``(s - block + 1)**2`` of them. Below
      ``s = 35`` there are fewer than 400 blocks in the entire region.
    * The dense verifier discards connected components below ``min_area_frac``
      of the image, because a region that small is indistinguishable from
      coincidental agreement.
    """
    rows = []
    for s in sizes:
        scored = evaluate_methods(size=s, images=images)
        row: dict[str, float] = {
            "size_px": s,
            "area_fraction": round(s * s / (512.0 * 512.0), 4),
        }
        for r in scored:
            row[r["method"]] = r["iou"]
        rows.append(row)
    return rows


def evaluate_false_alarms(images=IMAGES):
    """Run every detector on **untampered** images and count what it flags.

    The single most important number for a forgery detector and the one demos
    never show. A tool that marks part of every photograph you feed it is not a
    detector, it is a random accusation generator — and on an untampered image
    IoU, precision and recall are all undefined (the truth is empty), so the only
    honest measure is the fraction of pixels flagged, which should be zero.

    Note the images here are the *same* photographs used everywhere else in this
    project, untouched. `grass` and `brick` are in the set deliberately: a
    repeating texture is a duplicated region in every sense except intent.
    """
    from shared import io

    rows = []
    for method, fn in METHODS.items():
        flagged, images_flagged = [], 0
        for name in images:
            pred = fn(io.sample(name)) > 0
            flagged.append(float(pred.mean()))
            images_flagged += int(pred.any())
        rows.append(
            {
                "method": method,
                "mean_flagged": round(float(np.mean(flagged)), 5),
                "max_flagged": round(float(np.max(flagged)), 5),
                "images_with_any_flag": f"{images_flagged}/{len(images)}",
            }
        )
    return rows


def detect(img: np.ndarray, method: str = "SIFT + similarity verify") -> np.ndarray:
    """Run one named detector, for the UI and for inference."""
    return METHODS[method](img)


def describe_match(img: np.ndarray) -> dict:
    """What transform relates the two copies? Reported by `infer.py`.

    On a real image there is no ground truth, so the useful output is not a score
    but the *hypothesis*: how many keypoint pairs agreed, and what rotation and
    scale the agreement implies. A fit on 300 pairs at 14.9 degrees is evidence;
    a fit on 5 pairs at 0.03 degrees is a flat sky.
    """
    pairs = self_matches(img, make_sift(), cv2.NORM_L2, 0.6, 30)
    out = {"pairs": len(pairs), "inliers": 0, "angle_deg": None, "scale": None}
    if len(pairs) < 4:
        return out
    src = np.float32([p[0] for p in pairs]).reshape(-1, 1, 2)
    dst = np.float32([p[1] for p in pairs]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000
    )
    if matrix is None or inliers is None:
        return out
    out["inliers"] = int(inliers.sum())
    out["scale"] = round(float(np.hypot(matrix[0, 0], matrix[0, 1])), 4)
    out["angle_deg"] = round(float(np.degrees(np.arctan2(matrix[0, 1], matrix[0, 0]))), 2)
    return out
