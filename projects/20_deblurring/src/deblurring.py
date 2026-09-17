"""Deblurring: can you undo a *known* blur, and where does it stop?

The question
------------
Blur is a convolution, and convolution is invertible in the Fourier domain — on
paper. In practice naive inverse filtering is catastrophic, and the reason is
worth measuring rather than asserting.

> **The claim under test:** Richardson-Lucy has an **iteration optimum**. More
> iterations is not better: the likelihood keeps improving while the image gets
> worse, because the extra iterations are fitting noise. There is a specific
> iteration count where PSNR peaks and then falls.

And the deeper one:

> Blur destroys frequencies where ``|H(f)|`` approaches zero. Dividing by a
> near-zero number amplifies noise without bound, which is why every practical
> method is a *regularised* inverse. The regularisation parameter, not the
> algorithm, is what decides the result.

The blur kernel is generated here, so both the true image and the exact PSF are
known — which makes it possible to separate "this method is weak" from "the
information is gone".
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim

EPS = 1e-8


def _psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad and circularly shift a PSF so its FFT is the optical transfer function.

    The shift matters: without centring the kernel at the origin the result is
    correct but translated, and the deblurred image comes out offset by half the
    kernel — a bug that looks like the method failing.
    """
    pad = np.zeros(shape, np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)
    return np.fft.fft2(pad)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def deblur_inverse(img: np.ndarray, psf: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Naive inverse filter: divide by the OTF.

    Included because it is the textbook starting point and because it fails
    spectacularly. ``epsilon`` is the only thing standing between it and division
    by zero, and no value of it produces a usable image at realistic noise — that
    failure is the motivation for everything below.
    """
    f = to_float(to_gray(img))
    H = _psf_to_otf(psf, f.shape)
    H_safe = np.where(np.abs(H) < epsilon, epsilon, H)
    return to_uint8(np.real(np.fft.ifft2(np.fft.fft2(f) / H_safe)))


#: Noise-to-signal ratio Wiener runs at in the main table. The textbook example
#: value is 0.01; swept against the truth on this project's twelve photographs
#: the PSNR optimum is **0.05**, and 0.01 costs 0.75 dB.
#:
#: Reported at its own best setting rather than at the textbook one, because the
#: finding here is that Wiener loses to doing nothing *even when tuned* -- and
#: that claim is only worth making about a fairly tuned Wiener. Note that SSIM
#: prefers 0.1, so the two metrics do not agree on this number either.
WIENER_NSR = 0.05


def deblur_wiener(img: np.ndarray, psf: np.ndarray, nsr: float = WIENER_NSR) -> np.ndarray:
    """Wiener filter: the minimum-mean-squared-error linear inverse.

        G = conj(H) / (|H|^2 + NSR)

    Where the blur preserved signal (``|H|`` large) it inverts; where the blur
    destroyed it (``|H|`` near zero) the NSR term dominates and it backs off
    instead of amplifying noise. ``nsr`` is the noise-to-signal ratio, and
    guessing it wrong dominates the result — which is why it is swept rather
    than fixed.
    """
    f = to_float(to_gray(img))
    H = _psf_to_otf(psf, f.shape)
    G = np.conj(H) / (np.abs(H) ** 2 + nsr)
    return to_uint8(np.real(np.fft.ifft2(np.fft.fft2(f) * G)))


def deblur_richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Richardson-Lucy: iterative maximum-likelihood deconvolution for Poisson noise.

    Each iteration multiplies the estimate by a correction derived from the ratio
    of observed to re-blurred data. It is non-negative by construction, which is
    physically right for photon counts.

    **It does not converge to the truth.** It converges to the maximum-likelihood
    solution, and past some iteration count that means fitting the noise. The
    optimum is the project's headline measurement.
    """
    f = to_float(to_gray(img))
    psf = psf.astype(np.float32)
    psf_mirror = psf[::-1, ::-1]
    estimate = np.full_like(f, 0.5)

    for _ in range(iterations):
        conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        relative = f / np.maximum(conv, EPS)
        estimate = estimate * cv2.filter2D(
            relative, -1, psf_mirror, borderType=cv2.BORDER_REFLECT
        )
        estimate = np.clip(estimate, 0.0, 1.0)
    return to_uint8(estimate)


def deblur_regularised(img: np.ndarray, psf: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """Constrained least squares with a Laplacian smoothness prior.

        G = conj(H) / (|H|^2 + lambda * |L|^2)

    The same shape as Wiener, but the penalty grows with frequency instead of
    being flat. That suppresses exactly the high-frequency noise the inverse
    would amplify most, at the cost of softening genuine fine detail.
    """
    f = to_float(to_gray(img))
    H = _psf_to_otf(psf, f.shape)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
    L = _psf_to_otf(laplacian, f.shape)
    G = np.conj(H) / (np.abs(H) ** 2 + lam * np.abs(L) ** 2)
    return to_uint8(np.real(np.fft.ifft2(np.fft.fft2(f) * G)))


def deblur_unsharp(img: np.ndarray, psf: np.ndarray = None, amount: float = 1.5) -> np.ndarray:
    """Unsharp masking — the control that does not know the kernel.

    Sharpening is constantly confused with deblurring. This row measures the
    difference: it has no access to the PSF and cannot invert anything, it can
    only amplify what survived.
    """
    f = to_float(to_gray(img))
    blurred = cv2.GaussianBlur(f, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    return to_uint8(f + amount * (f - blurred))


def deblur_none(img: np.ndarray, psf: np.ndarray = None) -> np.ndarray:
    """Do nothing — the floor every method must clear."""
    return to_gray(img).copy()


METHODS: dict[str, Callable] = {
    "Inverse filter": deblur_inverse,
    "Wiener": deblur_wiener,
    "Richardson-Lucy": deblur_richardson_lucy,
    "Regularised LS": deblur_regularised,
    "Unsharp (no kernel)": deblur_unsharp,
    "Do nothing (control)": deblur_none,
}

#: Methods that are handed the true PSF. The others are controls that are not,
#: and the table must say which is which or the comparison is dishonest.
KNOWS_KERNEL = {"Inverse filter", "Wiener", "Richardson-Lucy", "Regularised LS"}


# --------------------------------------------------------------------------- #
# blind estimation — because a real pipeline is not given the PSF
# --------------------------------------------------------------------------- #


#: Radius band of the log-spectrum the angle search samples, as a fraction of
#: the shortest half-axis. The very centre is the DC blob, which is bright at
#: every angle and swamps the signal; the outermost ring is mostly noise.
ANGLE_RADII = (0.06, 1.0)


def estimate_motion_angle(img: np.ndarray, n_angles: int = 180) -> float:
    """Estimate a motion-blur direction from the log power spectrum, blind.

    Linear motion blur multiplies the spectrum by a sinc whose zero-crossings
    form parallel stripes **perpendicular** to the motion. Averaging the log
    spectrum along rays from the centre gives a profile over angle, and the
    motion direction falls out of where that profile peaks.

    Two things about this were wrong before, and together they made the
    estimator look like it worked at one angle and nowhere else.

    **It used `cv2.HoughLines` on a thresholded spectrum.** Hough found the FFT's
    own axis-aligned and diagonal structure rather than the sinc stripes, so the
    answer snapped to multiples of 45 degrees: exact at 0, 45, 90 and 135, and
    20 degrees or more out at every angle between.

    **And the perpendicular was never taken.** The stripes run across the motion,
    so the reported angle was the stripe orientation, not the blur direction.
    Scored against the truth that produced errors of 50 to 80 degrees -- worse
    than guessing, on a quantity with only 180 degrees to be wrong in -- while
    the one angle where the two errors happened to cancel (90) came out at 3.25
    and made the whole thing look plausible.

    The radial profile below carries no preference for any particular direction,
    and gets within about 10 degrees at every angle tested.
    """
    f = to_float(to_gray(img))
    h, w = f.shape
    # Windowing first: a rectangular crop has a hard edge at the frame boundary,
    # whose spectrum is a bright cross through the origin at exactly 0 and 90
    # degrees -- the two answers the estimator is most likely to be asked for.
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f * window))))

    cy, cx = h // 2, w // 2
    lo, hi = ANGLE_RADII
    limit = min(cy, cx)
    radii = np.arange(max(2, int(lo * limit)), int(hi * limit))
    thetas = np.deg2rad(np.arange(n_angles))
    ys = np.clip((cy + radii[None, :] * np.sin(thetas)[:, None]).astype(int), 0, h - 1)
    xs = np.clip((cx + radii[None, :] * np.cos(thetas)[:, None]).astype(int), 0, w - 1)
    profile = spectrum[ys, xs].mean(axis=1)

    stripe_angle = float(np.argmax(profile))
    return float((90.0 - stripe_angle) % 180.0)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **detail density** — mean gradient magnitude.
#: Deblurring is judged on how much fine structure it puts back, so an image
#: with none cannot show a difference between methods and an image made of it
#: shows the largest difference there is. Selected by
#: `tools/select_images.py --axis detail`; the range is 57 to 732, a factor of 13.
IMAGES = (
    "paraglider_peak",    # detail  57 — a smooth sky, almost nothing to restore
    "ox_in_pasture",      # detail 154
    "surfer_barrel",      # detail 193
    "three_owlets",       # detail 222
    "geisha_costume",     # detail 249
    "child_fur_hood",     # detail 278
    "tiger_wading",       # detail 299 — stripes, a direction a motion blur hides in
    "woman_white_fence",  # detail 322
    "castle_gatehouse",   # detail 361 — hard man-made edges
    "man_laying_paving",  # detail 406
    "marmot_boulder",     # detail 454
    "owl_in_grass",       # detail 732 — the finest detail in the pool
)


def detail_of(img) -> float:
    """Mean gradient magnitude x1000 — the axis this pool was selected on.

    Repeated here so a figure can label each row with the quantity that decides
    how much there was to restore in the first place.
    """
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(dx, dy)) * 1000)


def load_scene(name: str):
    """Load one of this project's photographs by name."""
    from shared import io

    return io.real_photo(name)
RL_ITERATIONS = (1, 3, 5, 10, 20, 30, 50, 80, 120, 200)
NSR_LEVELS = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3)
NOISE_LEVELS = (0.0, 1.0, 3.0, 6.0, 12.0)


def make_blurred(image: str, kind: str = "motion", noise_sigma: float = 3.0, seed: int = 0):
    """Blur an image with a known PSF and add read noise.

    Returns ``(clean_gray, blurred, psf)``. Noise is what makes the problem hard:
    without it even the naive inverse filter works.
    """
    from shared import synth

    clean = load_scene(image)
    psf = synth.motion_blur_kernel(15, 30.0) if kind == "motion" else synth.defocus_kernel(7)
    blurred = synth.apply_kernel(clean, psf)
    if noise_sigma > 0:
        blurred = synth.gaussian_noise(blurred, sigma=noise_sigma, seed=seed)
    return to_gray(clean), blurred, psf


def evaluate_methods(kind: str = "motion", noise_sigma: float = 3.0, images=IMAGES, runs: int = 3):
    """Score every method with the true PSF supplied where applicable."""
    acc = {n: {"psnr": [], "ssim": [], "ms": []} for n in METHODS}
    blurred_stats = {"psnr": [], "ssim": []}

    for i, name in enumerate(images):
        clean, blurred, psf = make_blurred(name, kind, noise_sigma, seed=i)
        blurred_stats["psnr"].append(psnr(to_gray(blurred), clean))
        blurred_stats["ssim"].append(ssim(to_gray(blurred), clean))

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(blurred, psf), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["ms"].append(timing.median_ms)

    return (
        [
            {
                "method": m,
                "knows_kernel": m in KNOWS_KERNEL,
                "psnr_db": round(float(np.mean(a["psnr"])), 3),
                "ssim": round(float(np.mean(a["ssim"])), 4),
                "median_ms": round(float(np.median(a["ms"])), 3),
            }
            for m, a in acc.items()
        ],
        {k: round(float(np.mean(v)), 4) for k, v in blurred_stats.items()},
    )


def sweep_rl_iterations(images=IMAGES, iterations=RL_ITERATIONS, noise_sigma: float = 3.0):
    """The headline experiment: PSNR against iteration count.

    If the optimum exists, this curve rises then falls, and the peak is a
    specific number of iterations rather than "as many as you can afford".
    """
    rows = []
    for n in iterations:
        p, s = [], []
        for i, name in enumerate(images):
            clean, blurred, psf = make_blurred(name, noise_sigma=noise_sigma, seed=i)
            out = deblur_richardson_lucy(blurred, psf, iterations=n)
            p.append(psnr(out, clean))
            s.append(ssim(out, clean))
        rows.append(
            {
                "iterations": n,
                "psnr_db": round(float(np.mean(p)), 3),
                "ssim": round(float(np.mean(s)), 4),
            }
        )
    return rows


def sweep_nsr(images=IMAGES, levels=NSR_LEVELS, noise_sigma: float = 3.0):
    """Wiener's noise-to-signal ratio: guessing it wrong dominates the result."""
    rows = []
    for nsr in levels:
        p, s = [], []
        for i, name in enumerate(images):
            clean, blurred, psf = make_blurred(name, noise_sigma=noise_sigma, seed=i)
            out = deblur_wiener(blurred, psf, nsr=nsr)
            p.append(psnr(out, clean))
            s.append(ssim(out, clean))
        rows.append(
            {
                "nsr": nsr,
                "psnr_db": round(float(np.mean(p)), 3),
                "ssim": round(float(np.mean(s)), 4),
            }
        )
    return rows


def sweep_noise(images=IMAGES, levels=NOISE_LEVELS):
    """How quickly each method collapses as noise rises.

    The naive inverse should fail first and hardest, because it is the only one
    with no regularisation at all.
    """
    rows = []
    for sigma in levels:
        scored, blurred = evaluate_methods(noise_sigma=sigma, images=images, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma, "blurred_input": blurred["psnr"]}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def evaluate_blind_angle(images=IMAGES, angles=(0.0, 30.0, 60.0, 90.0, 135.0)):
    """Can the motion direction be recovered from the image alone?"""
    from shared import synth

    rows = []
    for true_angle in angles:
        errs = []
        for i, name in enumerate(images):
            clean = load_scene(name)
            psf = synth.motion_blur_kernel(21, true_angle)
            blurred = synth.gaussian_noise(synth.apply_kernel(clean, psf), sigma=2.0, seed=i)
            est = estimate_motion_angle(blurred)
            diff = abs(est - true_angle) % 180.0
            errs.append(min(diff, 180.0 - diff))
        rows.append(
            {"true_angle_deg": true_angle, "mean_abs_error_deg": round(float(np.mean(errs)), 2)}
        )
    return rows


def deblur(img: np.ndarray, psf: np.ndarray, method: str = "Wiener") -> np.ndarray:
    return METHODS[method](img, psf)
