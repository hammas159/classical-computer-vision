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
kernel centre. Get it backwards and the failure is silent — no error, no
exception, an image that still looks processed. Both signs are implemented and
measured rather than asserted, and measuring them corrected the folklore: on a
sharp photograph the wrong sign does **not** blur, it raises acutance while
collapsing SSIM from 0.644 to **0.149**.
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

    It is in the comparison as a measured row rather than a warning in a comment,
    and measuring it corrected the comment. **This does not blur**, which is what
    this docstring used to claim. Adding a ``-4``-centre kernel subtracts the
    second derivative, and the result is not a smoothed image but a differently
    wrong one: edges are pushed the opposite way, leaving a soft inverted rim
    either side of every contour.

    The consequence is worse than blurring would be. On a **sharp** image it
    scores **SSIM 0.149** against the correct pairing's 0.644 — it is visibly
    damage — while its **acutance still rises**, 0.302 against the original's
    0.245. So the no-reference metric everyone reaches for to confirm "this looks
    sharper" reports success on a sign error.

    On an **already-blurred** image it does soften further (acutance 0.090
    against the blurred input's 0.114), which is where the folklore comes from.
    Both directions are in the tables. Neither is visible without a reference,
    which is exactly why this project keeps a control row.
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
    """High-boost: ``A*I - blur(I)`` — the textbook formula, including its flaw.

    For ``A = 1`` it is a pure high-pass. Above 1 it keeps progressively more of
    the original, which is the knob that stops a high-pass result looking like an
    embossing.

    **It also darkens everything, and that is not a bug here.** In a flat region
    ``blur(I) == I``, so the output is ``(A - 1) * I`` — at the usual ``A = 1.5``
    every flat area comes back at **half brightness**. On a sigma 1.5 blur it
    scores **13.47 dB** raw and **30.08 dB** once the mean is matched back — a
    16.61 dB swing, from last place to first, with not one pixel of its output
    changed. That number was almost entirely the brightness decision.

    The formula is left exactly as the textbooks write it, and the *scoring* is
    what changed: every method is reported twice, raw and after matching the
    output's mean back to the input. The gap between those two columns is this
    formulation's brightness shift, and it belongs in the open.
    """
    f = to_float(img)
    blurred = cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return to_uint8(boost * f - blurred)


def match_mean(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``img`` so its mean luminance matches ``reference``.

    Bisected on the *clipped* result rather than solved in closed form, because
    scaling then clipping is not linear: a closed-form factor overshoots
    whenever any pixel saturates, which sharpened images routinely do.
    """
    target = float(to_gray(reference).mean())
    lo, hi = 0.05, 20.0
    f = to_float(img)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        got = float(to_gray(to_uint8(np.clip(f * mid, 0, 1))).mean())
        if got < target:
            lo = mid
        else:
            hi = mid
    return to_uint8(np.clip(f * 0.5 * (lo + hi), 0, 1))


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


#: Noise-to-signal ratios the oracle searches. A single fixed value is not an
#: oracle: at nsr=0.01 the Wiener result scored 32.86 dB while a plain Laplacian
#: reached 33.28, so the "ceiling" was being beaten by a 3x3 kernel. An oracle
#: is allowed to use the ground truth -- that is what makes it a ceiling -- so
#: it picks the nsr that actually maximises PSNR.
ORACLE_NSR_VALUES = (0.0005, 0.001, 0.003, 0.01, 0.03, 0.1)


def deconvolve_oracle_best(img: np.ndarray, kernel: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """**Oracle**: Wiener deconvolution with the true kernel *and* the best nsr.

    Not a method. It is handed both the blur kernel and the clean image, and it
    exists to answer one question: how much of what the blur removed was
    recoverable at all? The gap between this row and the best sharpening row is
    the difference between *restoring* detail and *amplifying* what survived.
    """
    best, best_psnr = None, -1.0
    for nsr in ORACLE_NSR_VALUES:
        out = deconvolve_oracle(img, kernel, nsr=nsr)
        score = psnr(out, truth)
        if score > best_psnr:
            best, best_psnr = out, score
    return best


#: Reflect-padding added on every side before the FFT, then cropped away. See
#: `deconvolve_oracle` for why a bare FFT is not good enough here.
ORACLE_PAD = 32


def deconvolve_oracle(img: np.ndarray, kernel: np.ndarray, nsr: float = 0.01) -> np.ndarray:
    """Wiener deconvolution with the true kernel at one fixed nsr.

    Runs per channel on a colour image. It used to convert to grayscale, which
    made the oracle's PSNR incomparable with every other row in the table: a
    grayscale reconstruction is scored against a grayscale reference and has one
    third as much to get wrong. The ceiling has to be measured on the same thing
    the methods are.

    **The image is reflect-padded before the transform and cropped after.** An
    FFT treats the image as periodic, so the last row is deconvolved as though
    the first row were its neighbour. Where those two edges differ — a bright sky
    at the top, dark water at the bottom — that seam is a step edge the size of
    the whole frame, and the ringing it produces is not confined to the border.

    On `elk_water` it cost the oracle enough to score **30.68 dB against the
    30.84 dB blurred image it was handed**: a ceiling below its own floor, while
    a 3x3 Laplacian reached 32.33. That reads as "deconvolution does not help
    here", which is the opposite of true — with the padding it reaches 34.41.
    Circular wraparound is an artefact of computing the convolution with an FFT,
    not a property of Wiener deconvolution, so removing it is a fix and not a
    favour. Pinned by `test_the_oracle_is_a_ceiling_over_every_method`.
    """
    if img.ndim == 3:
        return np.stack(
            [deconvolve_oracle(img[..., c], kernel, nsr) for c in range(img.shape[2])],
            axis=-1,
        )
    f = cv2.copyMakeBorder(to_float(img), ORACLE_PAD, ORACLE_PAD, ORACLE_PAD,
                           ORACLE_PAD, cv2.BORDER_REFLECT)
    h, w = f.shape
    pad = np.zeros((h, w), np.float32)
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)

    F = np.fft.fft2(f)
    H = np.fft.fft2(pad)
    G = np.conj(H) / (np.abs(H) ** 2 + nsr)
    out = np.real(np.fft.ifft2(F * G))
    return to_uint8(out[ORACLE_PAD:h - ORACLE_PAD, ORACLE_PAD:w - ORACLE_PAD])


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Six photographs spanning a wide range of DETAIL DENSITY. Sharpening is
#: judged on what it does to fine structure, so a pool of images with similar
#: detail measures the same thing six times.
IMAGES = (
    "albatross_pair",      # detail  7  — smooth, almost nothing to sharpen
    "lionesses",           # detail  9
    "elk_water",           # detail 11
    "deer_water",          # detail 17
    "rhino_road",          # detail 23
    "stone_face_leaves",   # detail 45 — dense fine structure
)


def load_scene(name: str):
    """Load a scene by name from either image source."""
    from shared import io

    if name in io.REAL_PHOTOS:
        return io.real_photo(name)
    return io.sample(name)
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
        clean = load_scene(name)
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
            clean = load_scene(name)
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

    acc = {n: {"psnr": [], "matched": [], "ssim": [], "acut": []}
           for n in list(METHODS) + [ORACLE_NAME]}
    blurred_stats = {"psnr": [], "acutance": []}

    ksize = int(max(3, round(sigma * 6) | 1))
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel2d = (kernel @ kernel.T).astype(np.float32)

    for name in images:
        clean = load_scene(name)
        blurred = cv2.GaussianBlur(clean, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        blurred_stats["psnr"].append(psnr(blurred, clean))
        blurred_stats["acutance"].append(acutance(blurred))

        for method, fn in METHODS.items():
            out = fn(blurred)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["matched"].append(psnr(match_mean(out, clean), clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["acut"].append(acutance(out))

        rec = deconvolve_oracle_best(blurred, kernel2d, clean)
        acc[ORACLE_NAME]["psnr"].append(psnr(rec, clean))
        acc[ORACLE_NAME]["matched"].append(psnr(match_mean(rec, clean), clean))
        acc[ORACLE_NAME]["ssim"].append(ssim(rec, clean))
        acc[ORACLE_NAME]["acut"].append(acutance(rec))

    rows = [
        {
            "method": m,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "psnr_matched_db": round(float(np.mean(a["matched"])), 3),
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
