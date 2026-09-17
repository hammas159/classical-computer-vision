"""Frequency-domain filtering: ideal, Butterworth, Gaussian, notch, homomorphic.

The question
------------
Filtering in the frequency domain is where the sharp-cutoff intuition goes wrong
in a way you can see and measure.

> **The claim under test:** an *ideal* low-pass filter — a hard circular cutoff,
> the most obvious thing to write — produces **ringing**, because a sharp edge in
> frequency is a sinc in space, and a sinc oscillates forever. Butterworth and
> Gaussian filters trade a softer cutoff for no ringing, and the ringing can be
> measured rather than pointed at.

Periodic noise is the case where frequency filtering is not merely an alternative
to spatial filtering but the *only* practical approach: a sinusoidal interference
pattern is a couple of bright points in the spectrum and an intractable mess in
the image. The noise is injected here at a known frequency, so notch filtering
has an exact target and the removal is scoreable.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim

EPS = 1e-8


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #


def fft(img: np.ndarray) -> np.ndarray:
    """Centred 2-D FFT of a grayscale image.

    ``fftshift`` moves DC to the middle. Without it every filter mask has to be
    written with the origin at the corners and wrapped, which is where most
    hand-written frequency filters go wrong.
    """
    return np.fft.fftshift(np.fft.fft2(to_float(to_gray(img))))


def ifft(spectrum: np.ndarray) -> np.ndarray:
    return to_uint8(np.real(np.fft.ifft2(np.fft.ifftshift(spectrum))))


def spectrum_image(spectrum: np.ndarray) -> np.ndarray:
    """Log magnitude, normalised for display.

    The log is not cosmetic: DC is typically millions of times larger than any
    other component, so a linear magnitude image is one white pixel on black.
    """
    mag = np.log1p(np.abs(spectrum))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)


# --------------------------------------------------------------------------- #
# filter masks
# --------------------------------------------------------------------------- #


def mask_ideal(shape, cutoff: float, high: bool = False) -> np.ndarray:
    """Hard circular cutoff. The source of the ringing."""
    d = _radius_grid(shape)
    m = (d <= cutoff).astype(np.float32)
    return 1.0 - m if high else m


def mask_butterworth(shape, cutoff: float, order: int = 2, high: bool = False) -> np.ndarray:
    """``1 / (1 + (D/D0)^(2n))`` — a tunable transition steepness.

    ``order`` interpolates between the two extremes: order 1 is gentler than a
    Gaussian, and as order rises it approaches the ideal filter and the ringing
    comes back. That makes it the right control for the whole claim, because one
    parameter moves continuously between "no ringing" and "ringing".
    """
    d = _radius_grid(shape)
    m = 1.0 / (1.0 + (d / max(cutoff, EPS)) ** (2 * order))
    return 1.0 - m if high else m


def mask_gaussian(shape, cutoff: float, high: bool = False) -> np.ndarray:
    """``exp(-D^2 / 2 D0^2)`` — the smoothest transition, so no ringing at all.

    A Gaussian's inverse transform is another Gaussian, which is strictly
    positive and has no oscillation. That is the reason, in one sentence.
    """
    d = _radius_grid(shape)
    m = np.exp(-(d**2) / (2.0 * max(cutoff, EPS) ** 2)).astype(np.float32)
    return 1.0 - m if high else m


MASKS: dict[str, Callable] = {
    "Ideal": mask_ideal,
    "Butterworth (n=2)": lambda s, c, high=False: mask_butterworth(s, c, 2, high),
    "Butterworth (n=8)": lambda s, c, high=False: mask_butterworth(s, c, 8, high),
    "Gaussian": mask_gaussian,
}


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return ifft(fft(img) * mask)


# --------------------------------------------------------------------------- #
# notch filtering — targeted removal of periodic interference
# --------------------------------------------------------------------------- #


def add_periodic_noise(img: np.ndarray, fx: int = 40, fy: int = 25, amplitude: float = 0.25):
    """Add a sinusoidal interference pattern at a known spatial frequency.

    Returns ``(noisy, peak_positions)``. The peaks are where the notch filter
    must cut, so detection can be scored against their true location.
    """
    g = to_float(to_gray(img))
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pattern = amplitude * np.sin(2 * np.pi * (fx * xx / w + fy * yy / h))
    cy, cx = h // 2, w // 2
    peaks = [(cy + fy, cx + fx), (cy - fy, cx - fx)]
    return to_uint8(g + pattern), peaks


def mask_notch(shape, peaks, radius: float = 8.0, order: int = 4) -> np.ndarray:
    """Butterworth notches at given spectrum coordinates, and their conjugates.

    The conjugate is not optional: a real image has a Hermitian-symmetric
    spectrum, so every peak has a mirror through DC. Notching only one of the
    pair leaves the interference at half amplitude and looks like the filter
    barely worked.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.ones(shape, np.float32)
    for py, px in peaks:
        for sy, sx in ((py, px), (2 * cy - py, 2 * cx - px)):
            d = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
            # At the peak itself d -> 0, so radius/d -> inf and float32 overflows
            # raising it to 2*order. The limit is well defined -- the notch is
            # fully closed there -- so the ratio is capped at a value whose
            # power is still finite and whose reciprocal is already zero to
            # float precision. Overflow here produced a RuntimeWarning and an
            # inf that numpy then turned into the correct answer by accident.
            ratio = np.minimum(max(radius, EPS) / np.maximum(d, EPS),
                               np.float32(1e6) ** (1.0 / max(2 * order, 1)))
            mask *= 1.0 / (1.0 + ratio ** (2 * order))
    return mask


def detect_noise_peaks(img: np.ndarray, exclude_dc: int = 12, n_peaks: int = 2):
    """Find the brightest spectrum peaks away from DC — blind notch placement."""
    spectrum = np.abs(fft(img))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    work = spectrum.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    work[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < exclude_dc] = 0

    peaks = []
    for _ in range(n_peaks):
        idx = int(np.argmax(work))
        py, px = divmod(idx, w)
        peaks.append((py, px))
        work[max(0, py - 10) : py + 10, max(0, px - 10) : px + 10] = 0
    return peaks


# --------------------------------------------------------------------------- #
# homomorphic filtering
# --------------------------------------------------------------------------- #


def homomorphic(
    img: np.ndarray, cutoff: float = 30.0, gamma_low: float = 0.5, gamma_high: float = 2.0
) -> np.ndarray:
    """Separate illumination from reflectance in the **log** domain.

    An image is illumination times reflectance, which frequency filtering cannot
    separate because it is multiplicative. Taking the log makes it additive, and
    then illumination (smooth, low frequency) and reflectance (detailed, high
    frequency) really are separable by a filter. Suppressing the low band and
    boosting the high band corrects uneven lighting while enhancing detail — in
    one linear operation.
    """
    g = to_float(to_gray(img))
    log_img = np.log1p(g)
    spectrum = np.fft.fftshift(np.fft.fft2(log_img))
    d = _radius_grid(g.shape)
    h_filter = (gamma_high - gamma_low) * (
        1.0 - np.exp(-(d**2) / (2.0 * max(cutoff, EPS) ** 2))
    ) + gamma_low
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * h_filter)))
    return to_uint8(np.expm1(filtered))


# --------------------------------------------------------------------------- #
# ringing measurement
# --------------------------------------------------------------------------- #


def ringing_score(img: np.ndarray, reference: np.ndarray) -> float:
    """Oscillation energy near edges — kept, and **not** used for the headline.

    Counts sign changes of the error in a band beside the reference's edges. On
    a synthetic step that is a clean measure of Gibbs ringing. On a photograph it
    is not, and the difference is worth stating rather than hiding: a natural
    image has edges everywhere, so the band beside one edge is full of other
    edges, and the score ends up dominated by ordinary blur error.

    Measured on this project's twelve photographs at cutoff 40, it ranks Gaussian
    as the *worst* ringer (0.302) and Ideal as the best (0.233), which is exactly
    backwards. Several alternatives were tried -- error amplitude beside edges
    minus amplitude far from edges, sign changes weighted by local error -- and
    none separated the filters by more than the noise between images.

    So ringing is measured where it can be measured, on a step edge, by
    `ringing_on_step`. This function stays because the failed attempt is worth
    keeping visible: a plausible metric that ranks the answer backwards is the
    thing this repository is about.
    """
    ref = to_float(to_gray(reference))
    out = to_float(to_gray(img))
    edges = cv2.Canny(to_gray(reference), 60, 160)
    band = cv2.dilate(edges, np.ones((9, 9), np.uint8)) > 0
    band &= edges == 0  # beside the edge, not on it

    err = out - ref
    sign_changes = np.abs(np.diff(np.sign(err), axis=1))[:, :]
    band_x = band[:, 1:]
    if not band_x.any():
        return 0.0
    return float((sign_changes[band_x] > 0).mean())


#: Side of the synthetic step image the ringing measurement runs on.
STEP_SIZE = 256

#: Minimum swing, in [0, 1] intensity, for a turn in the step profile to count
#: as an oscillation rather than as floating-point residue. 1/255 is one uint8
#: level -- below that it could not be seen in the image anyway.
RINGING_FLOOR = 1.0 / 255.0


def step_edge(size: int = STEP_SIZE) -> np.ndarray:
    """A single vertical step — the one setting where ringing is unambiguous."""
    img = np.zeros((size, size), np.uint8)
    img[:, size // 2:] = 255
    return img


def ringing_on_step(mask_fn, cutoff: float = 20.0, size: int = STEP_SIZE) -> dict:
    """Gibbs ringing, measured where a natural image cannot measure it.

    Returns the number of oscillations in the filtered step's profile and the
    peak excursion outside the original's range, computed **before** clipping to
    uint8 -- clipping hides the overshoot entirely, which is why the first
    version of this reported 0.0000 for every filter.

    A step edge contains every frequency, so truncating the spectrum with a hard
    cutoff produces the classic Gibbs oscillation and a smooth cutoff does not.
    That is the whole content of "ideal filters ring".

    **Oscillations are counted only above `RINGING_FLOOR`.** Counting every turn
    in the profile does not work: a Gaussian-filtered step is almost perfectly
    flat away from the edge, and its floating-point wiggle produced *147* turns
    against the ideal filter's 42 -- ranking the non-ringing filter as the worst
    ringer, for the second time in this file. The amplitude threshold is what
    makes the count mean what its name says.
    """
    img = step_edge(size)
    f = to_float(img)
    spectrum = np.fft.fftshift(np.fft.fft2(f))
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * mask_fn(f.shape, cutoff))))

    profile = filtered[size // 2]
    # A turn counts only if the swing either side of it is a real excursion.
    turns = np.flatnonzero(np.abs(np.diff(np.sign(np.diff(profile)))) > 0) + 1
    swing = [min(abs(profile[i] - profile[i - 1]), abs(profile[i] - profile[i + 1]))
             for i in turns if 0 < i < len(profile) - 1]
    return {
        "oscillations": int(sum(s > RINGING_FLOOR for s in swing)),
        "overshoot": round(float(max(profile.max() - f.max(), f.min() - profile.min())), 5),
    }


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **detail density** — mean gradient magnitude, a
#: direct proxy for how much of the picture lives in the high frequencies a
#: low-pass throws away. Selected by `tools/select_images.py --axis detail`;
#: the range is 59 to 688, a factor of 12.
IMAGES = (
    "hazy_ridges",           # detail  59 — almost pure low frequency
    "whitewashed_chapel",    # detail 158
    "child_red_jumper",      # detail 193
    "geese_and_goslings",    # detail 223
    "fjord_harbour",         # detail 250
    "woman_hanbok",          # detail 277
    "graffiti_wall",         # detail 300
    "stone_wellhead",        # detail 321
    "lizard_on_gravel",      # detail 358
    "hawk_and_chick",        # detail 406
    "tower_and_spire",       # detail 453 — strong regular periodic structure
    "couple_autumn_bank",    # detail 688 — the busiest frame in the pool
)


def load_scene(name: str) -> np.ndarray:
    """Load one of this project's photographs by name."""
    from shared import io

    return io.real_photo(name)
CUTOFFS = (10.0, 20.0, 40.0, 80.0, 150.0)
BUTTERWORTH_ORDERS = (1, 2, 4, 8, 16)


def evaluate_lowpass(cutoff: float = 40.0, images=IMAGES, runs: int = 3):
    """Compare filter shapes at one cutoff, measuring ringing explicitly."""
    from shared import io

    acc = {n: {"psnr": [], "ssim": [], "ring": [], "ms": []} for n in MASKS}
    for name in images:
        img = load_scene(name)
        gray = to_gray(img)
        for label, builder in MASKS.items():
            mask = builder(gray.shape, cutoff)
            out, timing = timeit(lambda a=img, m=mask: apply_mask(a, m), runs=runs, warmup=1)
            acc[label]["psnr"].append(psnr(out, gray))
            acc[label]["ssim"].append(ssim(out, gray))
            acc[label]["ring"].append(ringing_score(out, gray))
            acc[label]["ms"].append(timing.median_ms)

    return [
        {
            "filter": n,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "ringing": round(float(np.mean(a["ring"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for n, a in acc.items()
    ]


def sweep_butterworth_order(images=IMAGES, orders=BUTTERWORTH_ORDERS, cutoff: float = 40.0):
    """Ringing against transition steepness — the continuous version of the claim.

    As the order rises the Butterworth filter approaches the ideal one, so if the
    sharp-cutoff explanation is right the ringing must rise with it.
    """
    from shared import io

    rows = []
    for order in orders:
        rings, psnrs = [], []
        for name in images:
            img = load_scene(name)
            gray = to_gray(img)
            out = apply_mask(img, mask_butterworth(gray.shape, cutoff, order))
            rings.append(ringing_score(out, gray))
            psnrs.append(psnr(out, gray))
        rows.append(
            {
                "order": order,
                "ringing": round(float(np.mean(rings)), 4),
                "psnr_db": round(float(np.mean(psnrs)), 3),
            }
        )
    return rows


def sweep_cutoff(images=IMAGES, cutoffs=CUTOFFS):
    """How much of the image survives each cutoff, per filter shape."""
    rows = []
    for c in cutoffs:
        scored = evaluate_lowpass(cutoff=c, images=images, runs=1)
        row: dict[str, float] = {"cutoff": c}
        for r in scored:
            row[r["filter"]] = r["psnr_db"]
        rows.append(row)
    return rows


def evaluate_notch(images=IMAGES, fx: int = 40, fy: int = 25, amplitude: float = 0.25):
    """Periodic noise removal, with the notch placed truly and blindly."""
    from shared import io

    rows = []
    true_p, blind_p, spatial_p, noisy_p = [], [], [], []
    peak_errors = []

    for name in images:
        img = load_scene(name)
        gray = to_gray(img)
        noisy, peaks = add_periodic_noise(img, fx, fy, amplitude)
        noisy_p.append(psnr(to_gray(noisy), gray))

        true_p.append(psnr(apply_mask(noisy, mask_notch(gray.shape, peaks)), gray))

        found = detect_noise_peaks(noisy)
        blind_p.append(psnr(apply_mask(noisy, mask_notch(gray.shape, found)), gray))
        peak_errors.append(
            float(np.min([np.hypot(found[0][0] - p[0], found[0][1] - p[1]) for p in peaks]))
        )

        # the spatial control: a median filter is the usual reflex and cannot
        # remove a global periodic pattern at all
        spatial_p.append(psnr(cv2.medianBlur(to_gray(noisy), 5), gray))

    rows.append({"method": "Noisy input", "psnr_db": round(float(np.mean(noisy_p)), 3)})
    rows.append({"method": "Median filter (spatial)", "psnr_db": round(float(np.mean(spatial_p)), 3)})
    rows.append({"method": "Notch, blind peaks", "psnr_db": round(float(np.mean(blind_p)), 3)})
    rows.append({"method": "Notch, true peaks (oracle)", "psnr_db": round(float(np.mean(true_p)), 3)})
    return rows, {"peak_localisation_error_px": round(float(np.mean(peak_errors)), 2)}


def match_mean(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``img`` so its mean matches ``reference``, bisected on the clipped result.

    Scaling then clipping is not linear, so a closed-form factor overshoots
    whenever anything saturates.
    """
    target = float(to_gray(reference).mean())
    f = to_float(img)
    lo, hi = 0.05, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        got = float(to_gray(to_uint8(np.clip(f * mid, 0, 1))).mean())
        if got < target:
            lo = mid
        else:
            hi = mid
    return to_uint8(np.clip(f * 0.5 * (lo + hi), 0, 1))


def evaluate_homomorphic(images=IMAGES, illum_min: float = 0.3):
    """Homomorphic filtering against an uneven-illumination degradation.

    Reported **twice**, raw and after matching the output's mean back to the
    original's, because the raw number is not measuring what it appears to.

    `gamma_low = 0.5` multiplies the low-frequency band of the *log* image by a
    half, and the DC term lives in that band -- so the whole picture comes back
    at roughly a third of its brightness. Scored raw, homomorphic filtering
    lands at 7.8 dB against the 12.3 dB uneven input it was supposed to fix,
    which reads as a method that does not work.

    Matched, it reaches 15.1 dB and is clearly ahead. The formula is left exactly
    as Gonzalez & Woods write it; what changed is the scoring. The same trap
    caught the high-boost row in project 16.
    """
    from shared import synth

    rows = []
    before, after, after_matched, clahe = [], [], [], []
    for name in images:
        clean = to_gray(load_scene(name))
        h, w = clean.shape
        shade = synth.scene_shading((w, h), illum_min)
        lit = to_uint8(to_float(clean) * shade)
        out = homomorphic(lit)
        before.append(psnr(lit, clean))
        after.append(psnr(out, clean))
        after_matched.append(psnr(match_mean(out, clean), clean))
        clahe.append(psnr(cv2.createCLAHE(3.0, (8, 8)).apply(lit), clean))

    def row(name, raw, matched=None):
        r = {"method": name, "psnr_db": round(float(np.mean(raw)), 3)}
        r["psnr_matched_db"] = round(float(np.mean(matched if matched else raw)), 3)
        return r

    rows.append(row("Uneven input", before))
    rows.append(row("CLAHE (spatial)", clahe))
    rows.append(row("Homomorphic", after, after_matched))
    return rows


def filter_image(img: np.ndarray, shape: str = "Gaussian", cutoff: float = 40.0, high: bool = False):
    gray = to_gray(img)
    return apply_mask(img, MASKS[shape](gray.shape, cutoff, high))
