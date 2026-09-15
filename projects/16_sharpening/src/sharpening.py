"""Sharpening: does it add information, or only contrast?

The question
------------
Sharpening is the one enhancement where the popular intuition is most clearly
wrong, and it is testable.

> **The claim under test:** sharpening a blurred image adds *contrast*, not
> *information*. If that is true, PSNR against the true original should **fall**
> as sharpening strength rises, even while the picture looks better.

Two experiments make that answerable:

* **Sharpen a clean image.** There is nothing to recover, so any change is pure
  distortion. PSNR can only fall. How fast it falls, against how much
  acutance is gained, is the whole trade-off in one curve.
* **Sharpen a blurred image, where the blur kernel is known.** Now there *is*
  something to recover, and a deconvolution oracle shows how much of it a
  sharpening filter actually gets back — which is the honest comparison, because
  sharpening and deblurring are constantly confused.

The Laplacian sign trap
-----------------------
Whether you **add** or **subtract** the Laplacian depends on the sign of its
kernel centre. Get it backwards and you blur instead of sharpen — silently, with
no error. Both signs are implemented and measured rather than asserted.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, rms_contrast, ssim

EPS = 1e-6

#: Positive-centre Laplacian: the centre is +4 and the neighbours -1, so the
#: kernel already estimates the *negative* second derivative. You ADD this one.
LAPLACIAN_POS = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)

#: Negative-centre Laplacian: the sign convention is flipped, so the same
#: operation requires SUBTRACTING it. Mixing the two up is the classic bug.
LAPLACIAN_NEG = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)


# --------------------------------------------------------------------------- #
# acutance: what sharpening actually increases
# --------------------------------------------------------------------------- #


def acutance(img: np.ndarray) -> float:
    """Mean gradient magnitude — a direct measure of edge steepness.

    This is what sharpening genuinely raises. Reporting it beside PSNR is what
    turns "looks better but scores worse" from a paradox into two numbers moving
    in opposite directions, exactly as the model predicts.
    """
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def overshoot(img: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of pixels pushed outside the reference's local range — haloing.

    Sharpening produces bright and dark rims either side of an edge. This
    quantifies that halo, which is the visible artefact people describe as
    "oversharpened" and which no global metric captures.
    """
    f, r = to_float(img), to_float(reference)
    lo = cv2.erode(r, np.ones((3, 3), np.float32))
    hi = cv2.dilate(r, np.ones((3, 3), np.float32))
    return float(np.mean((f < lo - 0.02) | (f > hi + 0.02)))


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def sharpen_laplacian_add(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """ADD a positive-centre Laplacian. The correct pairing."""
    f = to_float(img)
    lap = cv2.filter2D(f, cv2.CV_32F, LAPLACIAN_POS)
    return to_uint8(f + amount * lap)


def sharpen_laplacian_sub(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """SUBTRACT a negative-centre Laplacian. Also correct — and identical.

    Included to demonstrate the equivalence, because the two conventions cause
    endless confusion and the measured result settles it: these two rows should
    be the same to within floating-point noise.
    """
    f = to_float(img)
    lap = cv2.filter2D(f, cv2.CV_32F, LAPLACIAN_NEG)
    return to_uint8(f - amount * lap)


def sharpen_laplacian_wrong_sign(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """The bug: ADD a negative-centre Laplacian.

    This *blurs*. It is in the comparison as a measured row rather than a warning
    in a comment, because "it looks slightly soft" is exactly how this bug
    survives code review.
    """
    f = to_float(img)
    lap = cv2.filter2D(f, cv2.CV_32F, LAPLACIAN_NEG)
    return to_uint8(f + amount * lap)


def sharpen_unsharp(img: np.ndarray, amount: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Unsharp mask: ``I + amount * (I - blur(I))``.

    The name is backwards and the mechanism is not: subtracting a blurred copy
    leaves the high frequencies, and adding those back amplifies them. ``sigma``
    chooses *which* frequencies, which is why it has a radius control and the
    Laplacian does not.
    """
    f = to_float(img)
    blurred = cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return to_uint8(f + amount * (f - blurred))


def sharpen_high_boost(img: np.ndarray, boost: float = 1.5, sigma: float = 1.5) -> np.ndarray:
    """High-boost: ``A*I - blur(I)``, i.e. unsharp with the original amplified.

    For ``A = 1`` it is a pure high-pass. Above 1 it keeps progressively more of
    the original, which is the knob that stops a high-pass result looking like an
    embossing.
    """
    f = to_float(img)
    blurred = cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return to_uint8(boost * f - blurred)


def sharpen_none(img: np.ndarray) -> np.ndarray:
    """Do nothing — the control that any "improvement" must beat on PSNR."""
    return img.copy()


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Laplacian (add, +centre)": sharpen_laplacian_add,
    "Laplacian (sub, -centre)": sharpen_laplacian_sub,
    "Laplacian (WRONG sign)": sharpen_laplacian_wrong_sign,
    "Unsharp mask": sharpen_unsharp,
    "High-boost": sharpen_high_boost,
    "Do nothing (control)": sharpen_none,
}


# --------------------------------------------------------------------------- #
# the deblurring oracle — sharpening is not deconvolution
# --------------------------------------------------------------------------- #

ORACLE_NAME = "Wiener deconvolution (oracle)"


def deconvolve_oracle(img: np.ndarray, kernel: np.ndarray, nsr: float = 0.01) -> np.ndarray:
    """**Oracle**: Wiener deconvolution with the *true* blur kernel.

    Not a sharpening filter and not usable without knowing the kernel. It is here
    to make the distinction concrete: recovering detail that a known blur removed
    is a different operation from amplifying the detail that survived, and the
    gap between this row and the best sharpening row is the size of that
    difference.
    """
    f = to_float(to_gray(img))
    h, w = f.shape
    pad = np.zeros((h, w), np.float32)
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)

    F = np.fft.fft2(f)
    H = np.fft.fft2(pad)
    G = np.conj(H) / (np.abs(H) ** 2 + nsr)
    return to_uint8(np.real(np.fft.ifft2(F * G)))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "camera", "moon", "brick")
AMOUNTS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
BLUR_SIGMAS = (0.0, 0.8, 1.5, 2.5, 4.0)


def evaluate_on_clean(amount: float = 1.0, images=IMAGES, runs: int = 3):
    """Sharpen an image that is already perfect.

    There is nothing to recover, so PSNR can only fall. Any method scoring above
    "do nothing" here would falsify the project's central claim.
    """
    from shared import io

    acc = {n: {"psnr": [], "ssim": [], "acut": [], "over": [], "contrast": [], "ms": []}
           for n in METHODS}

    for name in images:
        clean = io.sample(name)
        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(clean), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["acut"].append(acutance(out))
            acc[method]["over"].append(overshoot(out, clean))
            acc[method]["contrast"].append(rms_contrast(out))
            acc[method]["ms"].append(timing.median_ms)

    return [
        {
            "method": m,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "acutance": round(float(np.mean(a["acut"])), 5),
            "overshoot": round(float(np.mean(a["over"])), 4),
            "rms_contrast": round(float(np.mean(a["contrast"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for m, a in acc.items()
    ]


def sweep_amount(images=IMAGES, amounts=AMOUNTS):
    """PSNR and acutance against sharpening strength, on a clean image.

    The two curves moving in opposite directions is the finding, stated as a
    picture: the image gets measurably sharper and measurably less correct at the
    same time.
    """
    from shared import io

    rows = []
    for amt in amounts:
        psnrs, acuts, overs = [], [], []
        for name in images:
            clean = io.sample(name)
            out = sharpen_unsharp(clean, amount=amt)
            psnrs.append(psnr(out, clean))
            acuts.append(acutance(out))
            overs.append(overshoot(out, clean))
        rows.append(
            {
                "amount": amt,
                "psnr_db": round(float(np.mean(psnrs)), 3),
                "acutance": round(float(np.mean(acuts)), 5),
                "overshoot": round(float(np.mean(overs)), 4),
            }
        )
    return rows


def evaluate_on_blurred(sigma: float = 1.5, amount: float = 1.0, images=IMAGES):
    """Sharpen a *blurred* image, where there genuinely is detail to recover.

    The oracle deconvolves with the true kernel. The gap between it and the best
    sharpening filter is the amount of information sharpening cannot restore
    because it never had the kernel.
    """
    from shared import io

    acc = {n: {"psnr": [], "ssim": [], "acut": []} for n in list(METHODS) + [ORACLE_NAME]}
    blurred_stats = {"psnr": [], "acutance": []}

    ksize = int(max(3, round(sigma * 6) | 1))
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel2d = (kernel @ kernel.T).astype(np.float32)

    for name in images:
        clean = io.sample(name)
        blurred = cv2.GaussianBlur(clean, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        blurred_stats["psnr"].append(psnr(blurred, clean))
        blurred_stats["acutance"].append(acutance(blurred))

        for method, fn in METHODS.items():
            out = fn(blurred)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["acut"].append(acutance(out))

        gray_clean = to_gray(clean)
        rec = deconvolve_oracle(blurred, kernel2d)
        acc[ORACLE_NAME]["psnr"].append(psnr(rec, gray_clean))
        acc[ORACLE_NAME]["ssim"].append(ssim(rec, gray_clean))
        acc[ORACLE_NAME]["acut"].append(acutance(rec))

    rows = [
        {
            "method": m,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "acutance": round(float(np.mean(a["acut"])), 5),
        }
        for m, a in acc.items()
    ]
    return rows, {k: round(float(np.mean(v)), 4) for k, v in blurred_stats.items()}


def sweep_blur(images=IMAGES, sigmas=BLUR_SIGMAS, amount: float = 1.0):
    """How much of a blur can sharpening undo, as the blur gets worse?"""
    rows = []
    for s in sigmas:
        if s <= 0:
            continue
        scored, blurred = evaluate_on_blurred(sigma=s, amount=amount, images=images)
        row: dict[str, float] = {"blur_sigma": s, "blurred_input": blurred["psnr"]}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def sharpen(img: np.ndarray, method: str = "Unsharp mask") -> np.ndarray:
    return METHODS[method](img)
