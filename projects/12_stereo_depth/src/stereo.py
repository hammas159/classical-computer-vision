"""Stereo to depth: matching two views, and what "accuracy" hides.

The question
------------
Two rectified views of a scene differ by a horizontal shift per pixel — the
disparity — which is inversely proportional to depth. Finding it is the oldest
problem in the field and `cv2.StereoBM` is four lines.

> **What does a disparity map get wrong, where, and what does filling the gaps
> cost?**

Two things separate a useful comparison from a picture of a depth map:

* **Density and accuracy trade against each other.** A matcher can decline to
  answer where it is unsure. Scoring only the pixels it answered rewards
  silence, and scoring every pixel punishes honesty. Both columns are reported,
  because a method that answers 55% of the frame at 4% error and one that
  answers 100% at 12% are not comparable on one number.
* **Error concentrates.** Bad pixels are not scattered; they sit in occlusions,
  in textureless regions and on disparity discontinuities. A single percentage
  averages over all three and describes none of them.

Ground truth
------------
Both kinds. The Middlebury *Aloe* pair ships with measured disparity, and
generated pairs (:func:`shared.synth.stereo_pair`) have disparity that is exact
by construction and can be varied — which a photograph cannot.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray

EPS = 1e-6

#: Disparities are searched in [0, MAX_DISPARITY). Must be a multiple of 16 for
#: OpenCV's matchers, and wide enough for the nearest object in the scene — too
#: small and near objects are silently truncated rather than reported as wrong.
MAX_DISPARITY = 64
#: A pixel counts as wrong if it misses by more than this many disparities.
#: 2.0 px is the Middlebury convention; 1.0 and 4.0 are also reported, because
#: which threshold you pick changes the ranking.
BAD_PIXEL_THRESHOLD = 2.0


def _to_float_disparity(raw: np.ndarray) -> np.ndarray:
    """OpenCV returns fixed-point disparity scaled by 16, with negatives invalid."""
    d = raw.astype(np.float32) / 16.0
    d[d < 0] = np.nan
    return d


def match_bm(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Block matching — sum of absolute differences over a window, per row.

    The fastest thing that works, and the one that leaves holes: it refuses to
    answer wherever the matching cost has no clear minimum, which is most of a
    blank wall.
    """
    matcher = cv2.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=15)
    matcher.setUniquenessRatio(10)
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(32)
    return _to_float_disparity(matcher.compute(to_gray(left), to_gray(right)))


def _sgbm(mode: int, block: int = 5) -> "cv2.StereoSGBM":
    channels = 3
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=MAX_DISPARITY,
        blockSize=block,
        # the smoothness penalties from the OpenCV sample, which are themselves
        # the paper's suggestion scaled by the block area
        P1=8 * channels * block * block,
        P2=32 * channels * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=mode,
    )


def match_sgbm(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Semi-global matching: block costs plus a smoothness penalty along 5 paths.

    Hirschmüller's approximation to a global optimisation — aggregate the cost
    along several 1-D paths instead of solving the 2-D problem. The penalty is
    what fills the textureless regions block matching abandons, and it fills
    them by assuming the answer is similar to the neighbours'. When that is true
    it is nearly free; when it is not, it is confidently wrong.
    """
    return _to_float_disparity(_sgbm(cv2.STEREO_SGBM_MODE_SGBM).compute(left, right))


def match_sgbm_hh(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """SGBM with the full 8-path aggregation — slower, and the usual default."""
    return _to_float_disparity(_sgbm(cv2.STEREO_SGBM_MODE_HH).compute(left, right))


def match_sad_naive(left: np.ndarray, right: np.ndarray, block: int = 9) -> np.ndarray:
    """A plain winner-take-all SAD search, written out.

    No uniqueness test, no left-right check, no speckle filter, no sub-pixel
    interpolation. It is here to show how much of `StereoBM`'s quality is the
    matching cost and how much is everything OpenCV wraps around it — which
    turns out to be most of it.
    """
    gl = to_gray(left).astype(np.int16)
    gr = to_gray(right).astype(np.int16)
    h, w = gl.shape
    k = block // 2
    kernel = np.ones((block, block), np.float32)

    best = np.full((h, w), np.inf, np.float32)
    argbest = np.zeros((h, w), np.float32)
    for d in range(MAX_DISPARITY):
        shifted = np.roll(gr, d, axis=1)
        shifted[:, :d] = gr[:, :1]
        cost = cv2.filter2D(np.abs(gl - shifted).astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REFLECT)
        better = cost < best
        best[better] = cost[better]
        argbest[better] = d
    argbest[:, :MAX_DISPARITY] = np.nan  # no search range at the left edge
    argbest[:k, :] = np.nan
    argbest[-k:, :] = np.nan
    return argbest


def match_constant(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Predict one disparity everywhere — the control.

    Here because "percentage of pixels within 2 disparities" sounds like a hard
    bar and is not: a scene whose depth is mostly one plane can score well from
    a constant. If a method cannot beat this, its disparity map is decoration.
    """
    return np.full(left.shape[:2], MAX_DISPARITY / 4.0, np.float32)


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Block matching (BM)": match_bm,
    "Semi-global (SGBM)": match_sgbm,
    "Semi-global, 8-path (HH)": match_sgbm_hh,
    "Naive SAD (no filtering)": match_sad_naive,
    "Constant disparity (control)": match_constant,
}

CONTROLS = ("Constant disparity (control)",)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score_disparity(
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    thresholds=(1.0, 2.0, 4.0),
) -> dict:
    """Bad-pixel rates and density, over the pixels the truth actually covers.

    Two rates are reported for each threshold, and the difference between them
    is the whole density question:

    ``bad_answered``
        of the pixels the method *answered*, how many are wrong. Rewards
        declining to answer.
    ``bad_all``
        of the pixels the truth covers, how many are wrong **or unanswered**.
        Treats a refusal as a miss.

    Neither is the honest number on its own.
    """
    valid = valid & np.isfinite(truth)
    answered = valid & np.isfinite(pred)
    n_valid = int(valid.sum())
    n_answered = int(answered.sum())
    if n_valid == 0:
        return {}

    err = np.abs(pred[answered] - truth[answered])
    out = {
        "density": round(n_answered / n_valid, 4),
        "mae_px": round(float(err.mean()), 4) if n_answered else float("nan"),
    }
    for t in thresholds:
        bad = int((err > t).sum())
        out[f"bad{t:g}_answered"] = round(bad / max(n_answered, 1), 4)
        out[f"bad{t:g}_all"] = round((bad + n_valid - n_answered) / n_valid, 4)
    return out


def error_regions(truth: np.ndarray, valid: np.ndarray, left: np.ndarray,
                  edge_width: int = 5, texture_percentile: float = 25.0):
    """Split the frame into the three places stereo actually fails.

    Returns boolean masks for ``discontinuity`` (near a depth edge),
    ``textureless`` (low local gradient) and ``rest``. A single bad-pixel
    percentage averages over all three and describes none of them, which is the
    second thing this project measures.
    """
    d = np.nan_to_num(truth, nan=0.0).astype(np.float32)
    grad = cv2.Laplacian(d, cv2.CV_32F, ksize=3)
    disc = cv2.dilate((np.abs(grad) > 2.0).astype(np.uint8),
                      np.ones((edge_width, edge_width), np.uint8)) > 0

    g = to_gray(left).astype(np.float32)
    texture = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F)
    texture = cv2.blur(np.abs(texture), (9, 9))
    flat = texture < np.percentile(texture[valid], texture_percentile)

    disc &= valid
    flat &= valid & ~disc
    rest = valid & ~disc & ~flat
    return {"discontinuity": disc, "textureless": flat, "well-textured": rest}
