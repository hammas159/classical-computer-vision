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

And a third, because "restored" is two different jobs
-----------------------------------------------------
A surviving print is rarely only scratched. It has also gone flat, warm and
desaturated, and **no inpainting method fixes any of that** — inpainting only
touches pixels you tell it are missing. So this module has two halves that share
one test image and never share a method:

* :data:`METHODS` / :data:`DETECTORS` — missing pixels. Scored with PSNR
  restricted to the damaged pixels, because whole-image PSNR is dominated by the
  97% of the photograph that was never damaged.
* :data:`FADE_METHODS` — present but wrong pixels. Scored with colour-cast error
  against the original and with LAB chroma, because PSNR alone cannot tell
  "corrected the cast" from "removed the colour".

Keeping them apart is the point. A single "restoration score" would let a method
that is good at one and useless at the other look adequate at both.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import dice, iou, psnr, rms_contrast, ssim

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


def inpaint_masked_mean(img: np.ndarray, mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Iterative masked average — a deliberately simple baseline.

    Two ideas, both necessary:

    * **Peel one ring at a time.** Each pass fills only the outer boundary of
      what is still unknown, then shrinks the unknown region and repeats. Filling
      the whole hole at once means the centre is averaged from pixels that are
      themselves still damaged.
    * **Average over known pixels only.** This is normalized convolution: sum the
      known values in the window, divide by *how many* were known, rather than by
      the window size. A plain blur divides by the window size and so silently
      mixes the scratch back in — see "Errors hit" in the README, where the
      unmasked version scored 6.4 dB against this one's 21.6.

    No gradient information, no isophotes: it is here to show how much of the
    result comes from the sophisticated part of a sophisticated method.
    """
    out = img.astype(np.float32)
    remaining = (mask > 0).astype(np.uint8)
    erode_k = np.ones((3, 3), np.uint8)
    known = (mask == 0).astype(np.float32)
    box = lambda a: cv2.boxFilter(a, -1, (ksize, ksize), normalize=False)  # noqa: E731

    for _ in range(64):
        if not remaining.any():
            break
        eroded = cv2.erode(remaining, erode_k)
        ring = ((remaining > 0) & (eroded == 0)).astype(bool)
        if not ring.any():
            break
        weight = box(known)                      # how many known pixels in the window
        if out.ndim == 3:
            num = box(out * known[..., None])
            fill = num / np.maximum(weight, 1.0)[..., None]
        else:
            fill = box(out * known) / np.maximum(weight, 1.0)
        usable = ring & (weight > 0)             # an all-unknown window has no answer
        out[usable] = fill[usable]
        known[usable] = 1.0                      # filled pixels become known for the next ring
        remaining = eroded
    return np.clip(out, 0, 255).astype(np.uint8)


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
    "Iterative masked mean": inpaint_masked_mean,
    "Harmonic diffusion": inpaint_diffusion,
}


# --------------------------------------------------------------------------- #
# damage detection — because a real pipeline is not given the mask
# --------------------------------------------------------------------------- #


def detect_damage_threshold(img: np.ndarray, percentile: float = 97.0) -> np.ndarray:
    """Find damage by intensity alone — the naive baseline.

    "Scratches are the bright bits" is the first idea anyone has, and it is in
    the table to be beaten. It cannot see a dark crease at all, and it cannot
    tell a scratch from a genuine highlight: on `astronaut` the white spacesuit
    is brighter than most of the damage.
    """
    g = to_gray(img)
    mask = (g >= np.percentile(g, percentile)).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def detect_damage_tophat(img: np.ndarray, size: int = 9) -> np.ndarray:
    """Morphological top-hat **and** black-hat: thin structures of either polarity.

    The shape-aware version. A top-hat keeps bright features that fit inside the
    structuring element and a black-hat keeps dark ones, so a thin scratch or a
    thin crease survives while a large bright region — a sky, a spacesuit — does
    not. Size is the discriminator, not brightness.
    """
    g = to_gray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    both = cv2.max(tophat, blackhat)
    _, mask = cv2.threshold(both, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


# A median filter only rejects a *minority* of outliers. If the damage fills more
# than half the window, the median becomes the damage and the residual is zero —
# the detector goes blind precisely where the damage is worst. 21 px was measured
# (see `sweep_detector_window`); the naive 5 px found 3% of the damage.
MEDIAN_RESIDUAL_KSIZE = 21


def detect_damage_median_residual(img: np.ndarray, ksize: int = MEDIAN_RESIDUAL_KSIZE) -> np.ndarray:
    """Flag pixels that disagree with their own median neighbourhood.

    A median filter is immune to a minority of outliers, so ``|img − median(img)|``
    is large exactly where a pixel is *not explained by* its surroundings — which
    is a workable definition of damage. Otsu then splits the residual into
    "explained" and "not", so the threshold adapts per image instead of being a
    constant tuned on one photo.

    **The window must be wider than the damage.** That is not a tuning detail, it
    is the same constraint the whole project turns on: every one of these methods
    assumes the damage is a local minority, and stops working when it is not.
    """
    g = to_gray(img)
    residual = cv2.absdiff(g, cv2.medianBlur(g, ksize))
    _, mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


DETECTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Intensity threshold": detect_damage_threshold,
    "Top-hat + black-hat": detect_damage_tophat,
    "Median residual": detect_damage_median_residual,
}


# --------------------------------------------------------------------------- #
# fade correction — the other half of "restored"
# --------------------------------------------------------------------------- #
#
# Inpainting fixes pixels that are *missing*. It does nothing at all for a print
# that has simply gone flat and yellow, which is what most surviving photographs
# actually look like. These are scored separately because no method here does
# both, and a project that reports one number for "restoration" hides that.


def correct_none(img: np.ndarray) -> np.ndarray:
    """The control: hand the faded print straight back."""
    return img.copy()


def correct_gray_world(img: np.ndarray) -> np.ndarray:
    """Gray-world white balance: assume the scene averages to neutral.

    Scale each channel so its mean matches the overall mean. Cheap, and the
    assumption fails loudly on any image that is genuinely dominated by one
    colour — which is why a sunset "corrected" this way turns grey.
    """
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    return np.clip(f * (means.mean() / (means + EPS)), 0, 255).astype(np.uint8)


def correct_channel_stretch(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Stretch each channel independently between its own percentiles.

    This fixes the cast *and* the flatness in one pass: the yellowing lifted the
    blue channel's black point and the fading compressed every channel's range,
    and a per-channel percentile stretch undoes both. Percentiles rather than
    min/max, because one dust speck at 255 would otherwise set the white point
    for the whole image.

    **The percentile is a robustness dial, and the two halves of this project
    want it set differently.** Measured on the four colour images:

    ======  =====================  ==============================
    low/hi  faded only (PSNR dB)   faded + residual damage (PSNR)
    ======  =====================  ==============================
    0.5/99.5              22.380                          --
    1/99                  21.245                      17.468
    2/98                  19.908                      18.053
    3/97                  18.891                          --
    ======  =====================  ==============================

    A tight percentile uses more of the print's true range, and is therefore
    better when the input is clean. A wide one ignores outliers, and is therefore
    better when the detector missed some damage and those pixels are now setting
    the black and white points. No single value wins both columns; 1/99 is the
    compromise, chosen because the recommended pipeline inpaints *first* and so
    usually presents this function with the cleaner of the two cases.
    """
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[..., c].astype(np.float32)
        lo, hi = np.percentile(ch, [low, high])
        out[..., c] = np.clip((ch - lo) * (255.0 / max(hi - lo, EPS)), 0, 255).astype(np.uint8)
    return out


def correct_clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on L only, leaving a and b alone.

    Restores local contrast without touching chroma — which is the point: it is
    the method that provably *cannot* fix a colour cast, and it is in the table
    so the cast-versus-contrast distinction is visible instead of asserted.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Measured, not chosen by eye. After the per-channel stretch the mean LAB chroma
# of the six test images is 23.02 against the originals' 26.21; 26.21/23.02 = 1.139.
# Rounded to 1.15 the restored chroma lands on 26.21 — the same number, which is
# the only defensible stopping point for a saturation slider.
SATURATION_FACTOR = 1.15


def boost_saturation(img: np.ndarray, factor: float = SATURATION_FACTOR) -> np.ndarray:
    """Scale chroma about the neutral axis in LAB, leaving lightness alone.

    Dye loss pulled every pixel toward its own luminance. That is a mix *across*
    channels, so no per-channel curve can undo it — which is why this step is
    separate from the stretch rather than folded into it.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1:] = np.clip((lab[..., 1:] - 128.0) * factor + 128.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def correct_stretch_then_clahe(img: np.ndarray) -> np.ndarray:
    """Per-channel stretch for the cast and range, then mild CLAHE for local contrast."""
    return correct_clahe_lab(correct_channel_stretch(img), clip=1.5)


def correct_full(img: np.ndarray) -> np.ndarray:
    """Stretch (undoes the diagonal part), then re-saturate (undoes the rest).

    One term for each thing the fading did, in the reverse of the order it did
    it, and **nothing else**. The obvious third step — a pass of CLAHE for local
    contrast — was measured and dropped: it costs 1.4 dB and 0.04 SSIM to buy
    0.7 degrees of colour-cast error. It is kept in the table as
    "Stretch + CLAHE" so that trade is visible rather than asserted.

    This is the project's recommended fade correction.
    """
    return boost_saturation(correct_channel_stretch(img))


FADE_METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "None (control)": correct_none,
    "Gray-world balance": correct_gray_world,
    "CLAHE on L only": correct_clahe_lab,
    "Per-channel stretch": correct_channel_stretch,
    "Stretch + CLAHE": correct_stretch_then_clahe,
    "Stretch + saturate": correct_full,
}


def colour_cast(img: np.ndarray) -> float:
    """Angular distance, in degrees, between the image's mean RGB and neutral grey.

    A **no-reference** measure, which is the only kind available on a real archive
    print. Zero means the average pixel is grey. Note that a correctly restored
    photograph does *not* score zero — a picture of a forest legitimately averages
    green — so this number is only interpretable next to
    :func:`colour_cast_error`, which compares against the original instead.
    """
    mean = img.reshape(-1, 3).mean(axis=0)
    grey = np.ones(3, np.float64)
    cos = float(mean @ grey / (np.linalg.norm(mean) * np.linalg.norm(grey) + EPS))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def colour_cast_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Angle in degrees between the mean RGB of ``pred`` and of ``truth``.

    The full-reference version: how far the restored colour balance sits from the
    colour balance the photograph actually had. This is what
    "did it fix the cast" should mean, and it is the column that shows gray-world
    driving :func:`colour_cast` to nearly zero while getting the balance *wrong*.
    """
    a = pred.reshape(-1, 3).mean(axis=0)
    b = truth.reshape(-1, 3).mean(axis=0)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def saturation_of(img: np.ndarray) -> float:
    """Mean LAB chroma — how colourful the image is, on the 0-128 LAB scale."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.mean(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "camera", "moon")
# `camera` and `moon` are greyscale plates. Fading and white balance are colour
# questions, so scoring a colour cast on them would average in two zeros.
COLOUR_IMAGES = ("astronaut", "coffee", "chelsea", "rocket")
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

    Precision and recall are reported separately, not folded into IoU, because
    **their costs are not symmetric here**. A missed scratch stays in the picture.
    A falsely flagged pixel is inpainted from its own neighbours, which are
    healthy, so it is replaced by approximately itself. Over-detection is nearly
    free; under-detection is not. Any ranking by IoU alone hides that, and the
    `whole_psnr_db` column is what makes it visible.
    """
    from shared import io, synth

    rows = []
    for det_name, det in DETECTORS.items():
        ious, dices, precs, recs, dmg, whole, fracs = [], [], [], [], [], [], []
        for i, name in enumerate(images):
            clean = io.sample(name)
            damaged, true_mask = synth.add_scratches(clean, thickness=thickness, seed=i)
            pred_mask = det(damaged)
            p, t = pred_mask > 0, true_mask > 0
            hit = float((p & t).sum())
            ious.append(iou(pred_mask, true_mask))
            dices.append(dice(pred_mask, true_mask))
            precs.append(hit / max(float(p.sum()), 1.0))
            recs.append(hit / max(float(t.sum()), 1.0))
            fracs.append(float(p.mean()))
            restored = inpaint_telea(damaged, pred_mask)
            dmg.append(_damage_only(restored, clean, true_mask))
            whole.append(psnr(restored, clean))
        rows.append(
            {
                "detector": det_name,
                "mask_iou": round(float(np.mean(ious)), 4),
                "mask_dice": round(float(np.mean(dices)), 4),
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "flagged_fraction": round(float(np.mean(fracs)), 4),
                "damage_psnr_db": round(float(np.mean(dmg)), 3),
                "whole_psnr_db": round(float(np.mean(whole)), 3),
            }
        )
    return rows


def sweep_detector_window(images=IMAGES, thickness: int = 3, windows=(5, 7, 9, 11, 15, 21, 31)):
    """How much of the damage the median-residual detector finds, by window size.

    The single most instructive curve in the project. A median filter rejects a
    *minority* of outliers; once the scratch fills half the window the median
    becomes the scratch, the residual goes to zero, and the detector reports a
    clean image. Recall collapses not gradually but as a cliff.
    """
    from shared import io, synth

    rows = []
    for k in windows:
        ious, recs, precs = [], [], []
        for i, name in enumerate(images):
            clean = io.sample(name)
            damaged, true_mask = synth.add_scratches(clean, thickness=thickness, seed=i)
            p = detect_damage_median_residual(damaged, ksize=k) > 0
            t = true_mask > 0
            hit = float((p & t).sum())
            ious.append(iou(p, t))
            recs.append(hit / max(float(t.sum()), 1.0))
            precs.append(hit / max(float(p.sum()), 1.0))
        rows.append(
            {
                "window_px": k,
                "damage_px": thickness,
                "mask_iou": round(float(np.mean(ious)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "precision": round(float(np.mean(precs)), 4),
            }
        )
    return rows


def evaluate_fade(images=COLOUR_IMAGES, runs: int = 3):
    """Score fade correction against the true original.

    The faded input is produced by a known transform, so unlike a real archive
    print there is something to score against. Both a full-reference metric
    (PSNR/SSIM vs the original) and a no-reference one (colour cast, contrast)
    are reported, because on a genuine old photo only the second is available.
    """
    from shared import io, synth

    keys = ("psnr", "ssim", "cast_err", "chroma", "contrast", "ms")
    acc = {n: {k: [] for k in keys} for n in FADE_METHODS}
    faded_stats = {"psnr": [], "cast_err": [], "chroma": [], "contrast": []}
    truth_stats = {"chroma": [], "contrast": []}

    for i, name in enumerate(images):
        clean = io.sample(name)
        faded = synth.fade_photo(clean, seed=i)
        faded_stats["psnr"].append(psnr(faded, clean))
        faded_stats["cast_err"].append(colour_cast_error(faded, clean))
        faded_stats["chroma"].append(saturation_of(faded))
        faded_stats["contrast"].append(rms_contrast(faded))
        truth_stats["chroma"].append(saturation_of(clean))
        truth_stats["contrast"].append(rms_contrast(clean))

        for method, fn in FADE_METHODS.items():
            out, timing = timeit(lambda f=fn: f(faded), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["cast_err"].append(colour_cast_error(out, clean))
            acc[method]["chroma"].append(saturation_of(out))
            acc[method]["contrast"].append(rms_contrast(out))
            acc[method]["ms"].append(timing.median_ms)

    rows = [
        {
            "method": method,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "cast_error_deg": round(float(np.mean(a["cast_err"])), 3),
            "chroma": round(float(np.mean(a["chroma"])), 3),
            "rms_contrast": round(float(np.mean(a["contrast"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for method, a in acc.items()
    ]
    stats = {k: round(float(np.mean(v)), 4) for k, v in faded_stats.items()}
    stats |= {f"original_{k}": round(float(np.mean(v)), 4) for k, v in truth_stats.items()}
    return rows, stats


def evaluate_pipeline(thickness: int = 3, images=COLOUR_IMAGES):
    """Score the two halves separately and together.

    The point of this table is that the two degradations are independent. Fixing
    the scratches on a faded print leaves it faded; correcting the fade on a
    scratched print leaves the scratches — and does so while *stretching them*,
    which is worth seeing as a number rather than being told.
    """
    from shared import io, synth

    stages = {
        "Damaged + faded (input)": lambda img, m: img,
        "Inpaint only": lambda img, m: inpaint_telea(img, m),
        "Fade correct only": lambda img, m: correct_full(img),
        "Inpaint then fade correct": lambda img, m: correct_full(inpaint_telea(img, m)),
        "Fade correct then inpaint": lambda img, m: inpaint_telea(correct_full(img), m),
    }
    acc = {n: {"psnr": [], "ssim": [], "cast_err": [], "dmg": []} for n in stages}

    for i, name in enumerate(images):
        clean = io.sample(name)
        damaged, mask = add_damage_and_fade(clean, thickness=thickness, seed=i)
        for stage, fn in stages.items():
            out = fn(damaged, mask)
            acc[stage]["psnr"].append(psnr(out, clean))
            acc[stage]["ssim"].append(ssim(out, clean))
            acc[stage]["cast_err"].append(colour_cast_error(out, clean))
            acc[stage]["dmg"].append(_damage_only(out, clean, mask))

    return [
        {
            "stage": stage,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "damage_psnr_db": round(float(np.mean(a["dmg"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "cast_error_deg": round(float(np.mean(a["cast_err"])), 3),
        }
        for stage, a in acc.items()
    ]


def add_damage_and_fade(img: np.ndarray, thickness: int = 3, seed: int | None = 0):
    """Both degradations at once, in the order time applies them.

    The photograph fades first — dye loss happens over decades in an album — and
    the scratches are inflicted on the already-faded print. Doing it the other
    way round would fade the scratches too, which no physical process does.
    """
    from shared import synth

    faded = synth.fade_photo(img, seed=seed)
    return synth.add_scratches(faded, thickness=thickness, seed=seed)


def restore(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "Telea (fast marching)",
    detector: str = "Median residual",
    fade: str = "Stretch + saturate",
):
    """End-to-end restoration, for the UI and for inference.

    If no mask is supplied the damage is detected first, which is the realistic
    case. Inpainting runs **before** fade correction, because correcting first
    stretches the damage along with everything else and then asks the inpainter
    to remove a higher-contrast scratch — see `evaluate_pipeline`.

    Returns ``(mask_used, inpainted, restored)``.
    """
    used = mask if mask is not None else DETECTORS[detector](img)
    inpainted = METHODS[method](img, used)
    return used, inpainted, FADE_METHODS[fade](inpainted)
