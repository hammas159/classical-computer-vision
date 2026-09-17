"""Keypoint detectors: repeatability under a known transform, and the real cost.

The question
------------
"ORB is faster than SIFT but less accurate" is repeated everywhere. The speed
ratio is quoted occasionally; the accuracy cost almost never is.

> **The claim under test:** quantify both. How many times faster is ORB, and how
> much repeatability does that buy you — under rotation, under scale, and under
> noise, separately?

Repeatability is only meaningful when the transform between the two images is
**known**, which is why this uses a synthetic homography rather than two photos.
A keypoint is "repeated" if, after projecting it through the true homography, a
detection in the second image lies within a few pixels.

Separating the three degradations matters because the detectors differ in which
invariances they actually have:

* SIFT and AKAZE are scale *and* rotation invariant;
* ORB and BRISK are rotation invariant but **not** scale invariant;
* Harris, Shi-Tomasi and FAST are neither — they are corner *detectors*, with no
  descriptor and no invariance beyond translation.

So the results should not be a single ranking. They should be a different
ranking per degradation, and that is the finding.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import repeatability

EPS = 1e-6


# --------------------------------------------------------------------------- #
# detectors — each returns an (N, 2) array of (x, y) keypoints
# --------------------------------------------------------------------------- #


def kp_harris(gray: np.ndarray, max_points: int = 1000, k: float = 0.04) -> np.ndarray:
    """Harris corner response, thresholded and non-max suppressed.

    Detects points where the local autocorrelation changes in *both* directions.
    It has no scale selection at all: the window size is fixed, so a corner that
    is sharp at one zoom level is invisible at another.
    """
    r = cv2.cornerHarris(np.float32(gray), blockSize=3, ksize=3, k=k)
    r = cv2.dilate(r, None)
    ys, xs = np.where(r > 0.01 * r.max())
    pts = np.stack([xs, ys], axis=-1).astype(np.float32)
    if len(pts) > max_points:
        scores = r[ys, xs]
        pts = pts[np.argsort(-scores)[:max_points]]
    return pts


def kp_shi_tomasi(gray: np.ndarray, max_points: int = 1000) -> np.ndarray:
    """Shi-Tomasi: Harris with ``min(lambda1, lambda2)`` instead of the determinant form.

    A small change with a real consequence: the minimum eigenvalue is a more
    direct measure of "corner-ness", which is why this is the default for
    tracking ("good features to track").
    """
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=0.01, minDistance=5)
    return np.zeros((0, 2), np.float32) if corners is None else corners.reshape(-1, 2)


def kp_fast(gray: np.ndarray, threshold: int = 25) -> np.ndarray:
    """FAST: a segment test on a 16-pixel circle. Extremely cheap, no descriptor.

    Decides cornerness by comparing intensities around a ring, with no
    derivatives at all. That is why it is fast, and why it has no invariance.
    """
    det = cv2.FastFeatureDetector.create(threshold=threshold)
    return np.array([k.pt for k in det.detect(gray, None)], np.float32).reshape(-1, 2)


def _from_detector(det, gray: np.ndarray) -> np.ndarray:
    kp = det.detect(gray, None)
    return np.array([k.pt for k in kp], np.float32).reshape(-1, 2)


def kp_sift(gray: np.ndarray, n: int = 1000) -> np.ndarray:
    """SIFT: difference-of-Gaussian scale space with explicit scale selection.

    Free since OpenCV 4.4 — the patent expired, so no contrib build is needed.
    The scale-space search is exactly what costs the time and buys the scale
    invariance.
    """
    return _from_detector(cv2.SIFT.create(nfeatures=n), gray)


def kp_orb(gray: np.ndarray, n: int = 1000) -> np.ndarray:
    """ORB: FAST keypoints with an orientation, plus a binary descriptor.

    Rotation invariance comes from an intensity-centroid orientation. Scale
    invariance is approximated with an image pyramid, which is coarser than
    SIFT's continuous scale selection — so it should degrade under scaling in a
    way SIFT does not.
    """
    return _from_detector(cv2.ORB.create(nfeatures=n), gray)


def kp_akaze(gray: np.ndarray) -> np.ndarray:
    """AKAZE: nonlinear diffusion scale space.

    Builds its scale space with edge-preserving diffusion instead of Gaussian
    blur, so features stay localised at coarse scales rather than drifting.
    """
    return _from_detector(cv2.AKAZE.create(), gray)


def kp_brisk(gray: np.ndarray) -> np.ndarray:
    """BRISK: scale-space FAST with a concentric sampling-pattern descriptor."""
    return _from_detector(cv2.BRISK.create(), gray)


DETECTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Harris": kp_harris,
    "Shi-Tomasi": kp_shi_tomasi,
    "FAST": kp_fast,
    "SIFT": kp_sift,
    "ORB": kp_orb,
    "AKAZE": kp_akaze,
    "BRISK": kp_brisk,
}


# --------------------------------------------------------------------------- #
# known transforms
# --------------------------------------------------------------------------- #


def homography_rotation(shape: tuple[int, int], degrees: float) -> np.ndarray:
    h, w = shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return np.vstack([m, [0, 0, 1]]).astype(np.float64)


def homography_scale(shape: tuple[int, int], factor: float) -> np.ndarray:
    h, w = shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, factor)
    return np.vstack([m, [0, 0, 1]]).astype(np.float64)


def apply_homography(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **edge density** — the percentage of pixels
#: Canny calls an edge at its own Otsu-derived thresholds. Every detector here
#: looks for distinctive local structure, so a pool that did not vary along that
#: axis would measure the same thing twelve times. Selected by
#: `tools/select_images.py --axis edges`; the range is 0.9% to 37.2%.
IMAGES = (
    "eagle_flat_sky",        # edges  0.9 — an eagle on plain blue, nothing to find
    "regatta_spinnakers",    # edges  8.4
    "scuba_diver_fish",      # edges 11.3
    "portrait_yellow",       # edges 13.1 — a backdrop of repeated identical dots
    "blue_footed_boobies",   # edges 15.0
    "polar_bear_rail",       # edges 17.1
    "geologist_rocks",       # edges 18.6
    "man_fur_hat",           # edges 20.2
    "glass_pyramid",         # edges 22.7 — dense man-made corners
    "bobcat_rock",           # edges 25.2
    "snake_coiled",          # edges 29.5
    "monitor_lizard_grass",  # edges 37.2 — the busiest frame in the pool
)


#: Grid the coverage measure divides the frame into.
SPREAD_GRID = 8


def spatial_coverage(pts: np.ndarray, shape: tuple[int, int]) -> float:
    """Percentage of an 8x8 grid of cells holding at least one keypoint.

    Reported beside repeatability because the two disagree, and the disagreement
    changes the headline. Harris is the most repeatable detector in the table and
    covers **a third of the frame**; every other detector covers 58-81%.

    Part of why it is so repeatable is that it concentrates on the few strongest
    corners, and the strongest corners are the most stable ones. That is a real
    property, and it is also exactly what you do not want when estimating a
    homography: a thousand keypoints inside one patch of gravel constrain a
    global transform no better than a handful spanning the picture.

    A repeatability table alone would have reported the concentration as a
    virtue.
    """
    if len(pts) == 0:
        return 0.0
    h, w = shape[:2]
    xs = np.clip((pts[:, 0] / w * SPREAD_GRID).astype(int), 0, SPREAD_GRID - 1)
    ys = np.clip((pts[:, 1] / h * SPREAD_GRID).astype(int), 0, SPREAD_GRID - 1)
    return float(len(set(zip(xs.tolist(), ys.tolist())))) / (SPREAD_GRID ** 2) * 100


def load_scene(name: str) -> np.ndarray:
    """Load one of this project's photographs by name."""
    from shared import io

    return io.real_photo(name)


ROTATIONS = (0.0, 10.0, 30.0, 60.0, 90.0, 180.0)
SCALES = (1.0, 0.9, 0.75, 0.6, 1.25, 1.6)
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0)
MATCH_THRESHOLD = 3.0


def evaluate_detectors(
    transform: str = "rotation", amount: float = 30.0, noise_sigma: float = 0.0,
    images=IMAGES, runs: int = 3,
):
    """Repeatability and cost for every detector under one known transform."""
    from shared import io, synth

    acc = {n: {"rep": [], "count": [], "ms": [], "cov": []} for n in DETECTORS}

    for i, name in enumerate(images):
        img = load_scene(name)
        gray = to_gray(img)
        if transform == "rotation":
            H = homography_rotation(gray.shape, amount)
        elif transform == "scale":
            H = homography_scale(gray.shape, amount)
        elif transform == "identity":
            H = np.eye(3)
        else:
            raise ValueError(f"unknown transform {transform!r}")

        warped = apply_homography(img, H)
        if noise_sigma > 0:
            warped = synth.gaussian_noise(warped, sigma=noise_sigma, seed=i)
        warped_gray = to_gray(warped)

        for det_name, fn in DETECTORS.items():
            a, timing = timeit(lambda f=fn: f(gray), runs=runs, warmup=1)
            b = fn(warped_gray)
            # `shape` excludes keypoints the transform carried out of frame --
            # see `shared.metrics.repeatability`. Without it a 45-degree
            # rotation scores every detector down by the third of the image the
            # crop removed, which reads as a failure of rotation invariance
            # rather than as the crop it actually is.
            acc[det_name]["rep"].append(
                repeatability(a, b, H, MATCH_THRESHOLD, shape=warped_gray.shape))
            acc[det_name]["count"].append(len(a))
            acc[det_name]["cov"].append(spatial_coverage(a, gray.shape))
            acc[det_name]["ms"].append(timing.median_ms)

    rows = [
        {
            "detector": n,
            "repeatability": round(float(np.mean(a["rep"])), 4),
            "keypoints": int(np.mean(a["count"])),
            "coverage_pct": round(float(np.mean(a["cov"])), 1),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for n, a in acc.items()
    ]
    fastest = min(r["median_ms"] for r in rows)
    for r in rows:
        r["slowdown_vs_fastest"] = round(r["median_ms"] / max(fastest, EPS), 1)
    return rows


def sweep_rotation(images=IMAGES, angles=ROTATIONS):
    """Rotation invariance. ORB and BRISK should hold; Harris and FAST should not."""
    rows = []
    for angle in angles:
        scored = evaluate_detectors("rotation", angle, images=images, runs=1)
        row: dict[str, float] = {"rotation_deg": angle}
        for r in scored:
            row[r["detector"]] = r["repeatability"]
        rows.append(row)
    return rows


def sweep_scale(images=IMAGES, scales=SCALES):
    """Scale invariance — the axis that should separate SIFT/AKAZE from ORB."""
    rows = []
    for s in scales:
        scored = evaluate_detectors("scale", s, images=images, runs=1)
        row: dict[str, float] = {"scale": s}
        for r in scored:
            row[r["detector"]] = r["repeatability"]
        rows.append(row)
    return rows


def sweep_noise(images=IMAGES, levels=NOISE_LEVELS):
    """Noise robustness, with the transform held at identity so only noise varies."""
    rows = []
    for sigma in levels:
        scored = evaluate_detectors("identity", 1.0, noise_sigma=sigma, images=images, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["detector"]] = r["repeatability"]
        rows.append(row)
    return rows


KEYPOINT_BUDGETS = (100, 250, 500, 1000)


def sweep_keypoint_budget(images=IMAGES, budgets=KEYPOINT_BUDGETS,
                          detector: str = "Harris", degrees: float = 30.0,
                          seed: int = 0):
    """Control for the obvious objection to the repeatability table.

    Harris returns 1000 keypoints and AKAZE returns 248. More keypoints means
    more chances for one to land within the 3 px match radius, so the natural
    suspicion is that Harris wins on density rather than on quality.

    This answers it by randomly discarding Harris keypoints down to each budget
    and re-scoring. Randomly, so the subsample is not quietly reselected for the
    good ones. If density were driving the result, repeatability would fall with
    the budget.
    """
    from shared.io import to_gray

    rng = np.random.default_rng(seed)
    fn = DETECTORS[detector]
    rows = []
    for n in budgets:
        scores = []
        for name in images:
            img = load_scene(name)
            gray = to_gray(img)
            H = homography_rotation(gray.shape, degrees)
            warped_gray = to_gray(apply_homography(img, H))
            a, b = fn(gray), fn(warped_gray)
            if len(a) > n:
                a = a[rng.choice(len(a), n, replace=False)]
            scores.append(repeatability(a, b, H, MATCH_THRESHOLD, shape=warped_gray.shape))
        rows.append({"budget": n, "repeatability": round(float(np.mean(scores)), 4)})
    return rows


def detect(img: np.ndarray, detector: str = "SIFT") -> np.ndarray:
    return DETECTORS[detector](to_gray(img))


def draw_keypoints(img: np.ndarray, pts: np.ndarray, colour=(0, 200, 0), radius: int = 3):
    out = img.copy()
    for x, y in np.asarray(pts, np.float32).reshape(-1, 2):
        cv2.circle(out, (int(x), int(y)), radius, colour, 1)
    return out
