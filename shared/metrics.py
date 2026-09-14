"""Quantitative metrics, one per task type.

Choosing the wrong metric is the fastest way to make a comparison meaningless,
so each function below documents *what it is for* and *how it lies*:

===========================  ==========================================
Task                         Metric
===========================  ==========================================
denoise / deblur / SR        :func:`psnr` **and** :func:`ssim` (both)
enhancement, no reference    :func:`entropy`, :func:`rms_contrast`
segmentation                 :func:`iou`, :func:`dice`
edges                        :func:`edge_prf` (with tolerance), :func:`pratt_fom`
optical flow                 :func:`endpoint_error`
keypoints                    :func:`repeatability`
homography / geometry        :func:`reprojection_error`, :func:`corner_error`
===========================  ==========================================

PSNR and SSIM are both reported because they *disagree*, and which one you quote
changes which method "wins". That disagreement is itself a project in the list.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .io import to_float, to_gray

# --------------------------------------------------------------------------- #
# full-reference image quality
# --------------------------------------------------------------------------- #


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error in 0-1 units."""
    fa, fb = to_float(a), to_float(b)
    return float(np.mean((fa - fb) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. Higher is better.

    PSNR is a per-pixel error measure: it has no idea whether the error lands on
    a flat wall or on someone's face. It rewards blur, because blurring reduces
    squared error while destroying detail. Never quote it alone.
    """
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / m))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity in [-1, 1]. Higher is better.

    Compares local luminance, contrast and structure rather than raw difference,
    so it tracks perceived quality better than PSNR — and it penalises blur that
    PSNR forgives.
    """
    fa, fb = to_float(a), to_float(b)
    if fa.ndim == 3:
        return float(structural_similarity(fa, fb, channel_axis=-1, data_range=1.0))
    return float(structural_similarity(fa, fb, data_range=1.0))


# --------------------------------------------------------------------------- #
# no-reference quality (for enhancement, where no "true" bright image exists)
# --------------------------------------------------------------------------- #


def entropy(img: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram, in bits.

    A proxy for how much information an enhanced image carries. A stretched
    histogram that merely clips highlights will *lose* entropy, which is exactly
    the failure mode it is here to catch.
    """
    g = to_gray(img)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def rms_contrast(img: np.ndarray) -> float:
    """Root-mean-square contrast: the std-dev of the grayscale image in 0-1 units."""
    return float(np.std(to_float(to_gray(img))))


def mean_brightness(img: np.ndarray) -> float:
    """Mean grayscale level in 0-1 units."""
    return float(np.mean(to_float(to_gray(img))))


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Estimate the additive noise standard deviation, in 0-255 units.

    Uses the Immerkaer (1996) fast estimator: convolve with a kernel that is
    zero-response to any locally linear intensity ramp, so edges and gradients
    contribute nothing and only noise survives. The scaling factor converts the
    mean absolute response back to a Gaussian sigma.

    This matters for enhancement work: brightening a dark photo also multiplies
    whatever noise was in the shadows, and a method that "wins" on brightness
    while tripling the noise has not actually improved the image.
    """
    g = to_gray(img).astype(np.float64)
    if min(g.shape) < 3:
        return 0.0
    laplacian_like = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    response = cv2.filter2D(g, -1, laplacian_like, borderType=cv2.BORDER_REFLECT)
    # discard a 1-px frame, where the border mode biases the response
    response = response[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(response)) / 6.0)


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool) if mask.dtype == bool else mask > 127


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union (Jaccard) in [0, 1].

    IoU punishes thin structures brutally: a one-pixel-wide ground-truth line
    offset by one pixel scores 0.0 despite being visually perfect. For hair,
    wires or blood vessels, read it alongside :func:`dice`.
    """
    p, t = _as_bool(pred), _as_bool(truth)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0  # both empty: a correct prediction of "nothing here"
    return float(np.logical_and(p, t).sum() / union)


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice / F1 coefficient in [0, 1]. Always >= IoU; kinder to small regions."""
    p, t = _as_bool(pred), _as_bool(truth)
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, t).sum() / total)


def pixel_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of pixels labelled correctly.

    Included because it is the metric people reach for first and it is almost
    always the wrong one: on an image that is 95% background, predicting
    "background everywhere" scores 0.95.
    """
    return float((_as_bool(pred) == _as_bool(truth)).mean())


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def edge_prf(
    pred: np.ndarray, truth: np.ndarray, tolerance: int = 2
) -> dict[str, float]:
    """Precision / recall / F1 for an edge map, with a **pixel tolerance**.

    The tolerance is mandatory, not a convenience. Edge detectors legitimately
    place a boundary one pixel to either side of the nominal truth; scoring at
    exactly zero tolerance measures sub-pixel luck, not detection quality, and
    gives every detector a near-zero F1.

    A predicted edge counts as correct if a true edge lies within ``tolerance``
    pixels of it, and vice versa — computed with a distance transform.
    """
    p, t = _as_bool(pred), _as_bool(truth)
    if p.sum() == 0 and t.sum() == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if p.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if t.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # distance from every pixel to the nearest *true* edge, and vice versa
    dist_to_truth = cv2.distanceTransform((~t).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~p).astype(np.uint8), cv2.DIST_L2, 3)

    precision = float((dist_to_truth[p] <= tolerance).mean())
    recall = float((dist_to_pred[t] <= tolerance).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": float(f1)}


def pratt_fom(pred: np.ndarray, truth: np.ndarray, alpha: float = 1.0 / 9.0) -> float:
    """Pratt's Figure of Merit in [0, 1]. Higher is better.

    Unlike a plain F1 it penalises *displaced* edges smoothly rather than as a
    hard miss, which is why it remains the standard single number for edge
    detector comparison.
    """
    p, t = _as_bool(pred), _as_bool(truth)
    n_p, n_t = int(p.sum()), int(t.sum())
    if n_p == 0 or n_t == 0:
        return 0.0
    dist = cv2.distanceTransform((~t).astype(np.uint8), cv2.DIST_L2, 3)
    d = dist[p]
    return float(np.sum(1.0 / (1.0 + alpha * d**2)) / max(n_p, n_t))


# --------------------------------------------------------------------------- #
# motion
# --------------------------------------------------------------------------- #


def endpoint_error(pred_flow: np.ndarray, true_flow: np.ndarray) -> float:
    """Mean endpoint error in **pixels** — the standard optical-flow metric."""
    d = pred_flow.astype(np.float64) - true_flow.astype(np.float64)
    return float(np.mean(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def reprojection_error(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    """Mean reprojection error in pixels after mapping ``src_pts`` through ``H``."""
    src = np.asarray(src_pts, np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(src, H.astype(np.float64)).reshape(-1, 2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))


def corner_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean distance in pixels between two ordered sets of corners.

    Used by the document scanner: the four detected page corners against the four
    the scene generator actually placed.
    """
    p = np.asarray(pred, np.float64).reshape(-1, 2)
    t = np.asarray(truth, np.float64).reshape(-1, 2)
    if p.shape != t.shape:
        return float("inf")
    return float(np.mean(np.linalg.norm(p - t, axis=1)))


def repeatability(
    kp_a: np.ndarray, kp_b: np.ndarray, H: np.ndarray, threshold: float = 3.0
) -> float:
    """Fraction of keypoints in image A that survive into image B under known ``H``.

    Only meaningful when the transform between the images is *known* — which is
    why the keypoint projects use a synthetic homography rather than two photos.
    """
    if len(kp_a) == 0 or len(kp_b) == 0:
        return 0.0
    a = np.asarray(kp_a, np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(a, H.astype(np.float64)).reshape(-1, 2)
    b = np.asarray(kp_b, np.float32).reshape(-1, 2)
    d = np.linalg.norm(proj[:, None, :] - b[None, :, :], axis=2)
    return float((d.min(axis=1) <= threshold).mean())


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #


def full_reference_report(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """PSNR + SSIM + MSE in one call, for the many restoration-style projects."""
    return {"psnr_db": psnr(pred, truth), "ssim": ssim(pred, truth), "mse": mse(pred, truth)}


def no_reference_report(img: np.ndarray) -> dict[str, float]:
    """Entropy + RMS contrast + mean brightness, for enhancement without a reference."""
    return {
        "entropy_bits": entropy(img),
        "rms_contrast": rms_contrast(img),
        "mean_brightness": mean_brightness(img),
    }
