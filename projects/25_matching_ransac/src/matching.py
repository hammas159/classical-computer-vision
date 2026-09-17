"""Feature matching and RANSAC homography: how many outliers can you survive?

The question
------------
Matching descriptors produces wrong matches. RANSAC removes them. The interesting
number is not "RANSAC works" but:

> **At what outlier fraction does each robust estimator break, and how does the
> required iteration count grow?**

RANSAC's iteration count has a closed form:

    N = log(1 - p) / log(1 - (1 - e)^s)

with ``p`` the desired confidence, ``e`` the outlier fraction and ``s = 4`` the
minimal sample for a homography. That is a *prediction*, and the measured
breakdown point can be checked against it — which turns a tutorial into an
experiment.

The homography is generated here, so the true transform is known exactly and
every match can be labelled inlier or outlier before any estimator runs. That
makes precision and recall of the inlier set measurable, not just the final
reprojection error.

🚨 A homography needs **≥ 4 point correspondences**, and collinear configurations
fail **silently** — returning a matrix that is numerically valid and
geometrically meaningless. That degeneracy is tested rather than trusted.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import reprojection_error

EPS = 1e-9


# --------------------------------------------------------------------------- #
# detection and description
# --------------------------------------------------------------------------- #


def describe_sift(gray: np.ndarray, n: int = 2000):
    det = cv2.SIFT.create(nfeatures=n)
    kp, desc = det.detectAndCompute(gray, None)
    return kp, desc, cv2.NORM_L2


def describe_orb(gray: np.ndarray, n: int = 2000):
    det = cv2.ORB.create(nfeatures=n)
    kp, desc = det.detectAndCompute(gray, None)
    return kp, desc, cv2.NORM_HAMMING


def describe_akaze(gray: np.ndarray, n: int = 2000):
    det = cv2.AKAZE.create()
    kp, desc = det.detectAndCompute(gray, None)
    return kp, desc, cv2.NORM_HAMMING


DESCRIPTORS: dict[str, Callable] = {
    "SIFT": describe_sift,
    "ORB": describe_orb,
    "AKAZE": describe_akaze,
}


# --------------------------------------------------------------------------- #
# match filtering
# --------------------------------------------------------------------------- #


def match_all(desc_a, desc_b, norm) -> list:
    """Every descriptor's single nearest neighbour. No filtering at all."""
    return cv2.BFMatcher(norm, crossCheck=False).match(desc_a, desc_b)


def match_cross_check(desc_a, desc_b, norm) -> list:
    """Keep a match only if each is the other's nearest neighbour.

    A symmetry constraint rather than a distance one, so it needs no threshold —
    which is exactly why it is worth comparing against the ratio test.
    """
    return cv2.BFMatcher(norm, crossCheck=True).match(desc_a, desc_b)


def match_ratio(desc_a, desc_b, norm, ratio: float = 0.75) -> list:
    """Lowe's ratio test: keep a match if the best is clearly better than the second.

    The insight is that a *correct* match is distinctive — its nearest neighbour
    is much closer than its second. An incorrect match sits in a crowd of equally
    plausible candidates. So the ratio, not the absolute distance, is what
    separates them.
    """
    knn = cv2.BFMatcher(norm, crossCheck=False).knnMatch(desc_a, desc_b, k=2)
    return [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]


FILTERS: dict[str, Callable] = {
    "All nearest neighbours": match_all,
    "Cross-check": match_cross_check,
    "Ratio test 0.75": match_ratio,
}


# --------------------------------------------------------------------------- #
# robust estimation
# --------------------------------------------------------------------------- #


def estimate_least_squares(src: np.ndarray, dst: np.ndarray):
    """Plain least squares over *all* matches. The control.

    Least squares minimises squared error, so a single gross outlier can dominate
    the fit. Its breakdown point is zero: one bad match is enough to ruin it,
    which is the entire reason robust estimators exist.
    """
    if len(src) < 4:
        return None, None
    H, _ = cv2.findHomography(src, dst, 0)
    return H, np.ones(len(src), bool)


def estimate_ransac(src, dst, thresh: float = 3.0, max_iters: int = 2000):
    """RANSAC: sample 4, fit, count inliers, keep the best consensus."""
    if len(src) < 4:
        return None, None
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh, maxIters=max_iters)
    return H, (mask.ravel().astype(bool) if mask is not None else None)


def estimate_lmeds(src, dst):
    """Least median of squares: minimise the *median* residual.

    Needs no threshold, which is its selling point — but it assumes fewer than
    50% outliers by construction, because past that the median residual is itself
    an outlier. That hard ceiling should show up in the sweep as a cliff at 0.5.
    """
    if len(src) < 4:
        return None, None
    H, mask = cv2.findHomography(src, dst, cv2.LMEDS)
    return H, (mask.ravel().astype(bool) if mask is not None else None)


def estimate_magsac(src, dst, thresh: float = 3.0):
    """MAGSAC++: marginalises over the threshold instead of fixing one."""
    if len(src) < 4:
        return None, None
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, thresh)
    return H, (mask.ravel().astype(bool) if mask is not None else None)


ESTIMATORS: dict[str, Callable] = {
    "Least squares (control)": lambda s, d: estimate_least_squares(s, d),
    "RANSAC": lambda s, d: estimate_ransac(s, d),
    "LMEDS": lambda s, d: estimate_lmeds(s, d),
    "MAGSAC++": lambda s, d: estimate_magsac(s, d),
}


def ransac_iterations_needed(outlier_fraction: float, confidence: float = 0.99, sample: int = 4):
    """The closed form for how many RANSAC samples are required.

    Reported alongside the measured breakdown so the theory is checked, not
    quoted. At 70% outliers a homography needs roughly 600 iterations; at 90% it
    needs about 46,000, which is why high-outlier matching is a practical problem
    and not just a statistical one.
    """
    e = min(max(outlier_fraction, 0.0), 0.999)
    inlier_prob = (1.0 - e) ** sample
    if inlier_prob <= EPS:
        return float("inf")
    return float(np.log(1.0 - confidence) / np.log(max(1.0 - inlier_prob, EPS)))


# --------------------------------------------------------------------------- #
# the scene: a known homography with controllable outliers
# --------------------------------------------------------------------------- #


def make_pair(image: str = "leopard_in_tree", jitter: float = 0.08, seed: int = 0):
    """Two views of one image related by a **known** homography."""
    from shared import synth

    first = load_scene(image)
    H = synth.random_homography(first.shape, jitter=jitter, seed=seed)
    return first, synth.warp_by_homography(first, H), H


def label_matches(kp_a, kp_b, matches, H, threshold: float = 3.0):
    """Split matches into true inliers and outliers using the known homography.

    This is what the whole project rests on: with ``H`` known, every match can be
    labelled *before* any estimator runs, so an estimator's inlier set can be
    scored for precision and recall rather than judged by its own self-report.
    """
    if not matches:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, bool)
    src = np.float32([kp_a[m.queryIdx].pt for m in matches])
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches])
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    truth = np.linalg.norm(projected - dst, axis=1) <= threshold
    return src, dst, truth


def inject_outliers(src, dst, truth, fraction: float, shape, seed: int = 0):
    """Replace a fraction of correspondences with random wrong ones.

    Controlling the outlier fraction directly is what makes the breakdown point
    measurable instead of incidental to whichever image pair was chosen.
    """
    rng = np.random.default_rng(seed)
    src, dst, truth = src.copy(), dst.copy(), truth.copy()
    n = len(src)
    if n == 0:
        return src, dst, truth
    n_bad = int(round(fraction * n))
    idx = rng.choice(n, size=min(n_bad, n), replace=False)
    h, w = shape[:2]
    dst[idx] = rng.uniform([0, 0], [w, h], size=(len(idx), 2)).astype(np.float32)
    truth[idx] = False
    return src, dst, truth


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **entropy** — bits per pixel in the luminance
#: histogram, a direct measure of how much there is in the frame for a
#: descriptor to describe. A dancer on a black stage and a wall of dried fish
#: are opposite problems for matching, and a pool that did not span that would
#: run one experiment twelve times. Selected by
#: `tools/select_images.py --axis entropy`; the range is 4.34 to 7.91 bits.
IMAGES = (
    "archer_dancer",        # entropy 4.34 — most of the frame is black stage
    "sea_shell_coral",      # entropy 6.53
    "potted_bonsai",        # entropy 6.91
    "porcupine_on_branch",  # entropy 7.02
    "roadrunner_rocks",     # entropy 7.15
    "gilded_stupa",         # entropy 7.24 — repeated architecture
    "headland_lighthouse",  # entropy 7.32
    "station_platform",     # entropy 7.41
    "leopard_in_tree",      # entropy 7.49
    "snowboarder_pines",    # entropy 7.59
    "sparkler_family",      # entropy 7.68
    "man_drying_fish",      # entropy 7.91 — the densest frame in the pool
)


def load_scene(name: str):
    from shared import io

    return io.real_photo(name)
OUTLIER_FRACTIONS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RATIOS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)


def evaluate_filters(descriptor: str = "SIFT", images=IMAGES):
    """How many matches each filter keeps, and how many of them are correct."""
    rows = []
    for fname, ffn in FILTERS.items():
        kept, precision, recall = [], [], []
        for i, image in enumerate(images):
            a, b, H = make_pair(image, seed=i)
            kp_a, desc_a, norm = DESCRIPTORS[descriptor](to_gray(a))
            kp_b, desc_b, _ = DESCRIPTORS[descriptor](to_gray(b))
            if desc_a is None or desc_b is None:
                continue
            matches = ffn(desc_a, desc_b, norm)
            _, _, truth = label_matches(kp_a, kp_b, matches, H)
            all_matches = match_all(desc_a, desc_b, norm)
            _, _, all_truth = label_matches(kp_a, kp_b, all_matches, H)
            kept.append(len(matches))
            precision.append(float(truth.mean()) if len(truth) else 0.0)
            recall.append(
                float(truth.sum()) / max(float(all_truth.sum()), 1.0) if len(all_truth) else 0.0
            )
        rows.append(
            {
                "filter": fname,
                "matches_kept": int(np.mean(kept)) if kept else 0,
                "inlier_precision": round(float(np.mean(precision)), 4) if precision else 0.0,
                "inlier_recall": round(float(np.mean(recall)), 4) if recall else 0.0,
            }
        )
    return rows


def sweep_ratio(descriptor: str = "SIFT", ratios=RATIOS, images=IMAGES):
    """Lowe's ratio: the precision/recall trade-off, as a curve.

    0.75 is the value everyone uses because the paper used it. This puts numbers
    on what it costs either side.
    """
    rows = []
    for r in ratios:
        kept, precision = [], []
        for i, image in enumerate(images):
            a, b, H = make_pair(image, seed=i)
            kp_a, desc_a, norm = DESCRIPTORS[descriptor](to_gray(a))
            kp_b, desc_b, _ = DESCRIPTORS[descriptor](to_gray(b))
            if desc_a is None or desc_b is None:
                continue
            matches = match_ratio(desc_a, desc_b, norm, ratio=r)
            _, _, truth = label_matches(kp_a, kp_b, matches, H)
            kept.append(len(matches))
            precision.append(float(truth.mean()) if len(truth) else 0.0)
        rows.append(
            {
                "ratio": r,
                "matches_kept": int(np.mean(kept)) if kept else 0,
                "inlier_precision": round(float(np.mean(precision)), 4) if precision else 0.0,
            }
        )
    return rows


def estimate_and_draw(a, b, H, estimator: str = "RANSAC", descriptor: str = "SIFT",
                      outlier_fraction: float = 0.5, seed: int = 0,
                      max_lines: int = 60):
    """One pair, one estimator: the reprojection error and a picture of the inliers.

    Returns ``(error_px, inlier_count, image)``. The error is measured against
    the **true** homography using the true correspondences, not against the
    estimator's own chosen inliers -- an estimator that keeps four points and
    fits them perfectly would otherwise report zero.
    """
    kp_a, desc_a, norm = DESCRIPTORS[descriptor](to_gray(a))
    kp_b, desc_b, _ = DESCRIPTORS[descriptor](to_gray(b))
    matches = match_ratio(desc_a, desc_b, norm)
    src, dst, truth = label_matches(kp_a, kp_b, matches, H)
    src, dst, truth = inject_outliers(src, dst, truth, outlier_fraction, a.shape, seed=seed)

    Hest, mask = ESTIMATORS[estimator](src, dst)
    good_src, good_dst = src[truth], dst[truth]
    err = (float("nan") if Hest is None or len(good_src) < 4
           else reprojection_error(Hest, good_src, good_dst))

    keep = mask if mask is not None else np.ones(len(src), bool)
    canvas = a.copy()
    idx = np.flatnonzero(keep)[:max_lines]
    for j in idx:
        x0, y0 = src[j]
        x1, y1 = dst[j]
        # green where the estimator is right, red where it kept an outlier --
        # the picture then shows the mistake rather than only the score
        colour = (0, 200, 0) if truth[j] else (220, 0, 0)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), colour, 1, cv2.LINE_AA)
    return err, int(keep.sum()), canvas


def sweep_outliers(
    descriptor: str = "SIFT", fractions=OUTLIER_FRACTIONS, images=IMAGES, runs: int = 1
):
    """The central experiment: where does each estimator break?

    Reprojection error is measured against the **true** homography using the true
    inlier correspondences, so an estimator cannot flatter itself by reporting on
    its own chosen inliers.
    """
    rows = []
    for frac in fractions:
        per = {n: [] for n in ESTIMATORS}
        times = {n: [] for n in ESTIMATORS}
        for i, image in enumerate(images):
            a, b, H = make_pair(image, seed=i)
            kp_a, desc_a, norm = DESCRIPTORS[descriptor](to_gray(a))
            kp_b, desc_b, _ = DESCRIPTORS[descriptor](to_gray(b))
            if desc_a is None or desc_b is None:
                continue
            matches = match_ratio(desc_a, desc_b, norm)
            src, dst, truth = label_matches(kp_a, kp_b, matches, H)
            if len(src) < 8:
                continue
            src, dst, truth = inject_outliers(src, dst, truth, frac, a.shape, seed=i)
            good_src, good_dst = src[truth], dst[truth]

            for ename, efn in ESTIMATORS.items():
                (Hest, _), timing = timeit(lambda f=efn: f(src, dst), runs=runs, warmup=0)
                times[ename].append(timing.median_ms)
                if Hest is None or len(good_src) < 4:
                    per[ename].append(float("nan"))
                    continue
                per[ename].append(reprojection_error(Hest, good_src, good_dst))

        row: dict[str, float] = {
            "outlier_fraction": frac,
            "iterations_needed": round(ransac_iterations_needed(frac), 1),
        }
        for ename, vals in per.items():
            row[ename] = round(float(np.nanmean(vals)), 3) if vals else None
        rows.append(row)
    return rows


def evaluate_descriptors(images=IMAGES, runs: int = 1):
    """Descriptor choice: match quality and cost, end to end."""
    rows = []
    for dname, dfn in DESCRIPTORS.items():
        errs, counts, precision, ms = [], [], [], []
        for i, image in enumerate(images):
            a, b, H = make_pair(image, seed=i)
            (kp_a, desc_a, norm), timing = timeit(
                lambda f=dfn, g=to_gray(a): f(g), runs=runs, warmup=0
            )
            kp_b, desc_b, _ = dfn(to_gray(b))
            if desc_a is None or desc_b is None:
                continue
            matches = match_ratio(desc_a, desc_b, norm)
            src, dst, truth = label_matches(kp_a, kp_b, matches, H)
            counts.append(len(matches))
            precision.append(float(truth.mean()) if len(truth) else 0.0)
            ms.append(timing.median_ms)
            Hest, _ = estimate_ransac(src, dst)
            if Hest is not None and truth.sum() >= 4:
                errs.append(reprojection_error(Hest, src[truth], dst[truth]))
        rows.append(
            {
                "descriptor": dname,
                "matches": int(np.mean(counts)) if counts else 0,
                "inlier_precision": round(float(np.mean(precision)), 4) if precision else 0.0,
                "reprojection_px": round(float(np.mean(errs)), 3) if errs else None,
                "median_ms": round(float(np.median(ms)), 2) if ms else None,
            }
        )
    return rows


def test_degeneracy():
    """Collinear points produce a homography that is valid and meaningless.

    Not a warning in a comment: four collinear correspondences are fed in and the
    result is reported, because "fails silently" is only convincing when you can
    see it return a matrix.
    """
    collinear_src = np.float32([[0, 0], [10, 10], [20, 20], [30, 30]])
    collinear_dst = np.float32([[5, 5], [15, 15], [25, 25], [35, 35]])
    H, _ = cv2.findHomography(collinear_src, collinear_dst, 0)
    square_src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    return {
        "collinear_returned_matrix": H is not None,
        "collinear_determinant": None if H is None else float(np.linalg.det(H)),
        "collinear_maps_square_sanely": (
            None
            if H is None
            else bool(
                np.isfinite(
                    cv2.perspectiveTransform(square_src.reshape(-1, 1, 2), H)
                ).all()
            )
        ),
    }


def match_and_estimate(a: np.ndarray, b: np.ndarray, descriptor: str = "SIFT",
                       filt: str = "Ratio test 0.75", estimator: str = "RANSAC"):
    """End-to-end pipeline, for the UI and for inference."""
    kp_a, desc_a, norm = DESCRIPTORS[descriptor](to_gray(a))
    kp_b, desc_b, _ = DESCRIPTORS[descriptor](to_gray(b))
    if desc_a is None or desc_b is None:
        return None, [], None
    matches = FILTERS[filt](desc_a, desc_b, norm)
    if len(matches) < 4:
        return None, matches, None
    src = np.float32([kp_a[m.queryIdx].pt for m in matches])
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches])
    H, mask = ESTIMATORS[estimator](src, dst)
    return H, matches, mask
