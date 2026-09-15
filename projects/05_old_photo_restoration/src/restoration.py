"""Old photo restoration: inpainting scratches, and what "restored" can mean.

The question
------------
A damaged print has scratches, blotches and fading. Inpainting fills the damaged
pixels from their surroundings. The obvious question — "which inpainting method
is best?" — hides a more useful one:

> **How much does the answer depend on the damage being thin?**

Inpainting is interpolation. Across a 3 px scratch the surrounding pixels
genuinely constrain the answer, and any method does well. Across a 40 px blotch
they do not, and every method invents something. Sweeping the damage width finds
where each method stops working, which a single-width comparison cannot.

A second, separable question the same scene answers: **does knowing the mask
matter more than the method?** Real restoration has to *detect* the damage first,
so the pipeline is scored both with the true mask and with a detected one.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import dice, iou, psnr, ssim

EPS = 1e-6


# --------------------------------------------------------------------------- #
# inpainting methods
# --------------------------------------------------------------------------- #


def inpaint_telea(img: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Telea (2004): fast marching, filling inward from the damage boundary.

    Each unknown pixel is a weighted average of known neighbours, weighted by
    distance and by the local gradient direction. Fast and smooth — which is both
    its strength on thin scratches and its weakness on wide holes, where "smooth"
    means "blurred".
    """
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def inpaint_navier_stokes(img: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Navier-Stokes (Bertalmio 2001): propagate isophotes into the hole.

    Treats image intensity as a stream function and continues level lines across
    the gap, borrowing the mathematics of incompressible flow. It preserves edge
    *continuity* better than Telea, at more cost.
    """
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)


def inpaint_median(img: np.ndarray, mask: np.ndarray, ksize: int = 7, iters: int = 6) -> np.ndarray:
    """Iterative median fill — a deliberately simple baseline.

    Repeatedly replaces damaged pixels with the median of their known
    neighbourhood, dilating inward each pass. No gradient information, no
    isophotes: it is here to show how much of the result comes from the
    sophisticated part of a sophisticated method.
    """
    out = img.copy()
    remaining = (mask > 0).astype(np.uint8)
    for _ in range(iters):
        if not remaining.any():
            break
        blurred = cv2.medianBlur(out, ksize)
        fill = remaining.astype(bool)
        out[fill] = blurred[fill]
        remaining = cv2.erode(remaining, np.ones((3, 3), np.uint8))
    return out


def inpaint_diffusion(img: np.ndarray, mask: np.ndarray, iters: int = 120) -> np.ndarray:
    """Isotropic diffusion: repeatedly blur, keeping known pixels pinned.

    This is the Laplace equation solved by Jacobi iteration — the textbook
    "harmonic inpainting". It always converges to something smooth, so it is the
    clearest demonstration that a plausible-looking fill is not the same as a
    correct one.
    """
    out = img.astype(np.float32).copy()
    hole = (mask > 0)[..., None] if img.ndim == 3 else (mask > 0)
    known = img.astype(np.float32)
    for _ in range(iters):
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
        out = np.where(hole, blurred, known)
    return out.astype(np.uint8)


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Telea (fast marching)": inpaint_telea,
    "Navier-Stokes": inpaint_navier_stokes,
    "Iterative median": inpaint_median,
    "Harmonic diffusion": inpaint_diffusion,
}


# --------------------------------------------------------------------------- #
# damage detection — because a real pipeline is not given the mask
# --------------------------------------------------------------------------- #


def detect_damage_threshold(img: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Find bright scratches by intensity alone.

    The generator draws damage as white strokes, so a high percentile finds them
    — and also finds every genuine highlight in the photograph, which is exactly
    the failure worth measuring.
    """
    g = to_gray(img)
    thresh = np.percentile(g, percentile)
    mask = (g >= max(thresh, 250)).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def detect_damage_tophat(img: np.ndarray, size: int = 9) -> np.ndarray:
    """Morphological top-hat: bright structures *thinner than* the kernel.

    This is the shape-aware version. A white top-hat keeps only bright features
    that fit inside the structuring element, so a thin scratch survives and a
    large bright region — a sky, a shirt — does not.
    """
    g = to_gray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


DETECTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Intensity threshold": detect_damage_threshold,
    "Morphological top-hat": detect_damage_tophat,
}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "camera", "moon")
WIDTH_LEVELS = (1, 3, 5, 9, 15, 25, 40)


def _damage_only(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """PSNR computed **only over the damaged pixels**.

    Whole-image PSNR is dominated by the undamaged majority: on a photo where the
    damage covers 3% of pixels, doing nothing at all already scores well. Scoring
    the repaired region alone is the only way to see which method repaired it.
    """
    sel = mask > 0
    if not sel.any():
        return float("inf")
    a = pred[sel].astype(np.float64) / 255.0
    b = truth[sel].astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def evaluate_methods(thickness: int = 3, images=IMAGES, runs: int = 3):
    """Score every inpainting method with the **true** damage mask."""
    from shared import io, synth

    acc = {n: {"psnr": [], "ssim": [], "dmg": [], "ms": []} for n in METHODS}
    damaged_stats = {"psnr": [], "damage_fraction": []}

    for i, name in enumerate(images):
        clean = io.sample(name)
        damaged, mask = synth.add_scratches(clean, thickness=thickness, seed=i)
        damaged_stats["psnr"].append(psnr(damaged, clean))
        damaged_stats["damage_fraction"].append(float((mask > 0).mean()))

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(damaged, mask), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["dmg"].append(_damage_only(out, clean, mask))
            acc[method]["ms"].append(timing.median_ms)

    rows = []
    for method, a in acc.items():
        rows.append(
            {
                "method": method,
                "psnr_db": round(float(np.mean(a["psnr"])), 3),
                "ssim": round(float(np.mean(a["ssim"])), 4),
                "damage_psnr_db": round(float(np.mean(a["dmg"])), 3),
                "median_ms": round(float(np.median(a["ms"])), 3),
            }
        )
    return rows, {k: round(float(np.mean(v)), 4) for k, v in damaged_stats.items()}


def sweep_thickness(images=IMAGES, levels=WIDTH_LEVELS):
    """Trace every method as the damage gets wider.

    This is the experiment that separates the methods. At 1 px they are
    indistinguishable; the interesting question is where each one gives up.
    """
    from shared import io, synth

    rows = []
    for t in levels:
        row: dict[str, float | int] = {"thickness_px": t}
        per = {n: [] for n in METHODS}
        fracs = []
        for i, name in enumerate(images):
            clean = io.sample(name)
            damaged, mask = synth.add_scratches(clean, thickness=t, seed=i)
            fracs.append(float((mask > 0).mean()))
            for method, fn in METHODS.items():
                per[method].append(_damage_only(fn(damaged, mask), clean, mask))
        row["damage_fraction"] = round(float(np.mean(fracs)), 4)
        for method, vals in per.items():
            row[method] = round(float(np.mean(vals)), 3)
        rows.append(row)
    return rows


def evaluate_detectors(thickness: int = 3, images=IMAGES):
    """Score damage *detection*, and the cost of using a detected mask.

    A restoration pipeline that is handed the mask is solving an easier problem
    than one that has to find it. The gap between the two is reported rather than
    quietly ignored.
    """
    from shared import io, synth

    rows = []
    for det_name, det in DETECTORS.items():
        ious, dices, psnrs = [], [], []
        for i, name in enumerate(images):
            clean = io.sample(name)
            damaged, true_mask = synth.add_scratches(clean, thickness=thickness, seed=i)
            pred_mask = det(damaged)
            ious.append(iou(pred_mask, true_mask))
            dices.append(dice(pred_mask, true_mask))
            restored = inpaint_telea(damaged, pred_mask)
            psnrs.append(_damage_only(restored, clean, true_mask))
        rows.append(
            {
                "detector": det_name,
                "mask_iou": round(float(np.mean(ious)), 4),
                "mask_dice": round(float(np.mean(dices)), 4),
                "damage_psnr_db": round(float(np.mean(psnrs)), 3),
            }
        )
    return rows


def restore(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "Telea (fast marching)",
    detector: str = "Morphological top-hat",
):
    """End-to-end restoration, for the UI and for inference.

    If no mask is supplied the damage is detected first, which is the realistic
    case. Returns ``(mask_used, restored)``.
    """
    used = mask if mask is not None else DETECTORS[detector](img)
    return used, METHODS[method](img, used)
