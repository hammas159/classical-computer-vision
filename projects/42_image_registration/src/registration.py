"""Image registration: phase correlation, ECC, and mutual information.

The question
------------
Registration aligns two images. The three classical methods rest on completely
different assumptions, and each one's assumption predicts exactly where it fails.

> **The claim under test:** phase correlation and ECC both assume the two images
> have the *same* intensities (or a linear relationship). Mutual information does
> not — it only assumes a statistical *dependence*. So when the two images come
> from different modalities, or one is inverted, the first two should collapse
> and MI should not.

That is the multi-modal case that makes MI standard in medical imaging, and it is
testable here by simply inverting one image: same structure, opposite intensities,
zero linear correlation.

The transform is applied here, so the true alignment is known exactly and error
is reported in **pixels** and **degrees** rather than as a success rate.

🚨 Phase correlation requires a **Hanning window**. Without it the image border
behaves as a huge step edge whose spectrum swamps the correlation peak — the
single most common reason an implementation silently returns nonsense.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8

EPS = 1e-9


# --------------------------------------------------------------------------- #
# the methods — each returns an estimated (dx, dy) or a full warp matrix
# --------------------------------------------------------------------------- #


def register_phase_correlation(reference: np.ndarray, moving: np.ndarray):
    """Translation from the phase of the cross-power spectrum.

    A shift in space is a linear phase ramp in frequency, so the inverse
    transform of the normalised cross-power spectrum is a delta at the shift.
    Because only *phase* is used, it is immune to overall brightness scaling —
    but not to intensity inversion, which flips the phase by pi everywhere.

    Translation only. It cannot express rotation or scale at all.
    """
    a = to_float(to_gray(reference))
    b = to_float(to_gray(moving))
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a * window, b * window)
    return (float(dx), float(dy)), float(response)


def register_phase_correlation_unwindowed(reference: np.ndarray, moving: np.ndarray):
    """The same thing without the Hanning window — included to be beaten.

    Demonstrates the border-effect failure as a measurement rather than a
    footnote.
    """
    a = to_float(to_gray(reference))
    b = to_float(to_gray(moving))
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return (float(dx), float(dy)), float(response)


def register_ecc(reference: np.ndarray, moving: np.ndarray,
                 motion=cv2.MOTION_TRANSLATION, iterations: int = 200):
    """Enhanced Correlation Coefficient: maximise a normalised correlation.

    Iterative and gradient-based, so it needs a reasonable starting guess and
    converges to sub-pixel precision. The correlation coefficient is invariant to
    brightness and contrast changes, which makes it robust to exposure
    differences — but it is still a *linear* similarity, so an inverted image
    gives it a perfect *negative* correlation that it treats as a terrible match.

    Unlike phase correlation it can estimate rotation and affine warps.
    """
    a = to_float(to_gray(reference))
    b = to_float(to_gray(moving))
    warp = np.eye(2, 3, dtype=np.float32)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            a, b, warp, motion,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-7), None, 5,
        )
    except cv2.error:
        return None, 0.0
    return warp, float(cc)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Mutual information between two images, in bits.

    ``MI(A, B) = H(A) + H(B) - H(A, B)``, computed from the joint histogram. It
    measures how much knowing one image tells you about the other **with no
    assumption about the form of the relationship** — inverted, non-linear or
    cross-modality all count equally.

    That generality is the reason it is the standard for multi-modal medical
    registration, and the reason it is slow: there is no closed form, so
    alignment means searching.
    """
    ga = to_gray(a).ravel()
    gb = to_gray(b).ravel()
    joint, _, _ = np.histogram2d(ga, gb, bins=bins, range=[[0, 256], [0, 256]])
    joint = joint / max(joint.sum(), 1.0)

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    h_joint = -np.sum(joint[nz] * np.log2(joint[nz]))
    h_x = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    h_y = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    return float(h_x + h_y - h_joint)


def register_mutual_information(reference: np.ndarray, moving: np.ndarray,
                                search: int = 20, coarse_step: int = 4):
    """Exhaustive search maximising mutual information, coarse then fine.

    MI has no gradient in closed form, so alignment is a search. Two passes keep
    it tractable: a coarse grid, then a fine one around the winner. This is why MI
    registration is orders of magnitude slower than phase correlation — and the
    timing column should show exactly that.
    """
    best, best_mi = (0, 0), -np.inf
    h, w = to_gray(reference).shape

    for step, centre, radius in ((coarse_step, (0, 0), search), (1, None, coarse_step)):
        cx, cy = centre if centre is not None else best
        for dy in range(cy - radius, cy + radius + 1, step):
            for dx in range(cx - radius, cx + radius + 1, step):
                m = np.float32([[1, 0, -dx], [0, 1, -dy]])
                shifted = cv2.warpAffine(moving, m, (w, h), borderMode=cv2.BORDER_REFLECT)
                mi = mutual_information(reference, shifted)
                if mi > best_mi:
                    best_mi, best = mi, (dx, dy)
    return (float(best[0]), float(best[1])), float(best_mi)


METHODS: dict[str, Callable] = {
    "Phase correlation": register_phase_correlation,
    "Phase corr. (no window)": register_phase_correlation_unwindowed,
    "ECC": lambda r, m: register_ecc(r, m),
    "Mutual information": register_mutual_information,
}

#: Methods that return a (dx, dy) pair rather than a warp matrix.
RETURNS_SHIFT = {"Phase correlation", "Phase corr. (no window)", "Mutual information"}


def extract_shift(method: str, result) -> tuple[float, float]:
    """Normalise the two return shapes into one (dx, dy) for comparison."""
    if result is None:
        return (float("nan"), float("nan"))
    if method in RETURNS_SHIFT:
        return result
    warp = result
    if warp is None:
        return (float("nan"), float("nan"))
    return (float(-warp[0, 2]), float(-warp[1, 2]))


# --------------------------------------------------------------------------- #
# the test cases
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "camera", "brick", "chelsea")
SHIFTS = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0)
ROTATIONS = (0.0, 1.0, 3.0, 7.0, 15.0)
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0, 50.0)


def make_pair(
    image: str = "astronaut", dx: float = 5.0, dy: float = 3.0, rotation: float = 0.0,
    modality: str = "same", noise_sigma: float = 0.0, seed: int = 0,
):
    """Build a reference and a transformed moving image with known parameters.

    ``modality`` controls the intensity relationship, which is the axis that
    separates the methods:

    * ``same`` — identical intensities
    * ``inverted`` — 255 minus, so structure is identical and linear correlation
      is perfectly negative
    * ``gamma`` — a non-linear but monotonic remap
    * ``synthetic_mri`` — a non-monotonic remap, the closest stand-in here for a
      genuinely different imaging modality
    """
    from shared import io, synth

    reference = io.sample(image)
    h, w = reference.shape[:2]

    m = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    moving = cv2.warpAffine(reference, m, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT)

    if modality == "inverted":
        moving = 255 - moving
    elif modality == "gamma":
        moving = to_uint8(np.power(to_float(moving), 2.5))
    elif modality == "synthetic_mri":
        # a non-monotonic mapping: mid-greys become bright, extremes become dark.
        # No linear or even monotonic relationship survives, so only MI can cope.
        f = to_float(to_gray(moving))
        moving = to_uint8(np.abs(np.sin(f * np.pi * 1.5)))
        moving = cv2.cvtColor(moving, cv2.COLOR_GRAY2RGB)

    if noise_sigma > 0:
        moving = synth.gaussian_noise(moving, sigma=noise_sigma, seed=seed)

    return reference, moving, (dx, dy)


def shift_error(estimated, truth) -> float:
    if not np.all(np.isfinite(estimated)):
        return float("nan")
    return float(np.hypot(estimated[0] - truth[0], estimated[1] - truth[1]))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

SUCCESS_PX = 1.0


def evaluate_methods(images=IMAGES, dx: float = 7.0, dy: float = 4.0,
                     modality: str = "same", noise_sigma: float = 0.0, runs: int = 1):
    """Registration error and cost for every method under one condition."""
    acc = {n: {"err": [], "hit": [], "ms": []} for n in METHODS}
    for i, image in enumerate(images):
        ref, moving, truth = make_pair(
            image, dx=dx, dy=dy, modality=modality, noise_sigma=noise_sigma, seed=i
        )
        for name, fn in METHODS.items():
            result, timing = timeit(lambda f=fn: f(ref, moving)[0], runs=runs, warmup=0)
            est = extract_shift(name, result)
            err = shift_error(est, truth)
            acc[name]["err"].append(err)
            acc[name]["hit"].append(bool(np.isfinite(err) and err <= SUCCESS_PX))
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": n,
            "mean_error_px": round(float(np.nanmean(a["err"])), 4),
            "median_error_px": round(float(np.nanmedian(a["err"])), 4),
            "success_rate": round(float(np.mean(a["hit"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for n, a in acc.items()
    ]


def sweep_shift(images=IMAGES, shifts=SHIFTS):
    """How large a displacement does each method handle?

    Phase correlation is global and should be flat across the whole range. ECC is
    a local optimiser and should fail once the displacement exceeds its capture
    radius — a different failure shape entirely.
    """
    rows = []
    for s in shifts:
        scored = evaluate_methods(images=images, dx=s, dy=s * 0.6)
        row: dict[str, float] = {"shift_px": s}
        for r in scored:
            row[r["method"]] = r["mean_error_px"]
        rows.append(row)
    return rows


def sweep_modality(images=IMAGES, modalities=("same", "gamma", "inverted", "synthetic_mri")):
    """The central experiment: what happens when intensities stop matching.

    The prediction is specific — phase correlation and ECC should hold for
    ``gamma`` (still monotonic) and fail for ``inverted`` and
    ``synthetic_mri``, while mutual information should hold throughout.
    """
    rows = []
    for modality in modalities:
        scored = evaluate_methods(images=images, modality=modality)
        row: dict[str, float | str] = {"modality": modality}
        for r in scored:
            row[r["method"]] = r["mean_error_px"]
        rows.append(row)
    return rows


def sweep_noise(images=IMAGES, levels=NOISE_LEVELS):
    rows = []
    for sigma in levels:
        scored = evaluate_methods(images=images, noise_sigma=sigma)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["method"]] = r["mean_error_px"]
        rows.append(row)
    return rows


def windowing_effect(images=IMAGES, shifts=SHIFTS):
    """Quantify the Hanning window, rather than repeating the warning.

    Both variants on identical inputs. If the windowed version is dramatically
    better, the border effect is real and the advice has a number attached.
    """
    rows = []
    for s in shifts:
        with_win, without = [], []
        for i, image in enumerate(images):
            ref, moving, truth = make_pair(image, dx=s, dy=s * 0.6, seed=i)
            with_win.append(shift_error(register_phase_correlation(ref, moving)[0], truth))
            without.append(
                shift_error(register_phase_correlation_unwindowed(ref, moving)[0], truth)
            )
        rows.append(
            {
                "shift_px": s,
                "with_hanning_px": round(float(np.nanmean(with_win)), 4),
                "without_hanning_px": round(float(np.nanmean(without)), 4),
            }
        )
    return rows


def rotation_capability(images=IMAGES, rotations=ROTATIONS):
    """Only ECC can express rotation. The others are measured failing at it.

    Phase correlation estimates a translation; asked to align a rotated image it
    returns the best translation, which is not an alignment. Reporting that as
    "error" is fair only if it is also stated that the method was never able to
    represent the transform — so both facts are in the table.
    """
    rows = []
    for rot in rotations:
        ecc_err, phase_err = [], []
        for i, image in enumerate(images):
            ref, moving, truth = make_pair(image, dx=0.0, dy=0.0, rotation=rot, seed=i)
            warp, _ = register_ecc(ref, moving, motion=cv2.MOTION_EUCLIDEAN)
            if warp is not None:
                estimated_angle = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
                ecc_err.append(abs(abs(estimated_angle) - rot))
            phase_err.append(shift_error(register_phase_correlation(ref, moving)[0], truth))
        rows.append(
            {
                "rotation_deg": rot,
                "ecc_angle_error_deg": round(float(np.nanmean(ecc_err)), 4) if ecc_err else None,
                "phase_translation_error_px": round(float(np.nanmean(phase_err)), 4),
                "phase_can_represent_rotation": False,
            }
        )
    return rows


def mi_landscape(image: str = "astronaut", modality: str = "inverted", search: int = 15):
    """The mutual-information surface around the true alignment, for the figures.

    Worth plotting once: MI has a clear peak at the correct shift even when the
    two images have opposite intensities, which is the whole argument for the
    method in one picture.
    """
    ref, moving, truth = make_pair(image, dx=6.0, dy=4.0, modality=modality)
    h, w = to_gray(ref).shape
    grid = np.zeros((2 * search + 1, 2 * search + 1), np.float32)
    for iy, dy in enumerate(range(-search, search + 1)):
        for ix, dx in enumerate(range(-search, search + 1)):
            m = np.float32([[1, 0, -dx], [0, 1, -dy]])
            shifted = cv2.warpAffine(moving, m, (w, h), borderMode=cv2.BORDER_REFLECT)
            grid[iy, ix] = mutual_information(ref, shifted)
    return grid, truth


def align(reference: np.ndarray, moving: np.ndarray, method: str = "Phase correlation"):
    """Estimate and apply the alignment, for the UI."""
    result = METHODS[method](reference, moving)[0]
    dx, dy = extract_shift(method, result)
    if not np.all(np.isfinite([dx, dy])):
        return None, (float("nan"), float("nan"))
    h, w = reference.shape[:2]
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(moving, m, (w, h), borderMode=cv2.BORDER_REFLECT), (dx, dy)
