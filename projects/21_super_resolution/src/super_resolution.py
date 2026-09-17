"""Single-image super-resolution: can you invent pixels that were never captured?

The question
------------
> **The claim under test:** no single-image interpolation can add information.
> PSNR should therefore **plateau** across nearest, bilinear, bicubic and
> Lanczos — a few tenths of a dB apart — because they are all just different
> weighted averages of the same surviving samples.

That is the honest counterweight to "AI upscaling" intuition, and it sets up
project 40 (multi-frame SR), where extra *frames* genuinely do add information
and the plateau is broken.

🚨 The degradation must be **blur then decimate**, not ``cv2.resize`` down.
``INTER_AREA`` averages over the whole receptive field, which is a different
operator from anti-alias-blur followed by point sampling. Using resize makes the
comparison measure how well each method inverts ``INTER_AREA`` — a question
nobody meant to ask. :func:`shared.synth.downsample_for_sr` does it correctly,
and the difference is measured here rather than taken on trust.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim

EPS = 1e-6


# --------------------------------------------------------------------------- #
# interpolation methods
# --------------------------------------------------------------------------- #


def _resize(img: np.ndarray, scale: int, interpolation: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=interpolation)


def sr_nearest(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Nearest neighbour: replicate. Blocky, and the only one with no blur."""
    return _resize(img, scale, cv2.INTER_NEAREST)


def sr_bilinear(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Bilinear: linear in both axes. Smooth, and softens every edge."""
    return _resize(img, scale, cv2.INTER_LINEAR)


def sr_bicubic(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Bicubic: a cubic kernel over 4x4 neighbours. The de facto default."""
    return _resize(img, scale, cv2.INTER_CUBIC)


def sr_lanczos(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Lanczos-4: a windowed sinc, the closest practical thing to ideal
    band-limited reconstruction. If interpolation could recover detail, this is
    the one that would.
    """
    return _resize(img, scale, cv2.INTER_LANCZOS4)


def sr_back_projection(
    img: np.ndarray, scale: int = 4, iterations: int = 12, blur_sigma: float | None = None
) -> np.ndarray:
    """Iterative back-projection — the only method here that uses the forward model.

    Upsample, then repeatedly: simulate the degradation on the current estimate,
    compare with the actual low-resolution input, and push the residual back up.
    It enforces consistency with what was observed, which interpolation never
    checks.

    This is the interesting row. It should beat the interpolators — but by a
    small margin, because consistency with the low-resolution data still does not
    determine the missing frequencies.
    """
    if blur_sigma is None:
        blur_sigma = 0.5 * scale
    lr = to_float(img)
    estimate = to_float(_resize(img, scale, cv2.INTER_CUBIC))

    for _ in range(iterations):
        blurred = cv2.GaussianBlur(estimate, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
        simulated = blurred[::scale, ::scale]
        h = min(simulated.shape[0], lr.shape[0])
        w = min(simulated.shape[1], lr.shape[1])
        residual = np.zeros_like(simulated)
        residual[:h, :w] = lr[:h, :w] - simulated[:h, :w]
        up = cv2.resize(
            residual, (estimate.shape[1], estimate.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        estimate = estimate + 0.6 * cv2.GaussianBlur(
            up, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT
        )
        estimate = np.clip(estimate, 0.0, 1.0)
    return to_uint8(estimate)


def sr_edi(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Edge-directed style upscale: bicubic, then an edge-aware sharpen.

    A cheap stand-in for NEDI. It cannot add information either, but it
    redistributes the softening away from edges — which is what makes it *look*
    better while scoring about the same. A useful demonstration that perceived
    sharpness and fidelity are different axes.
    """
    up = _resize(img, scale, cv2.INTER_CUBIC)
    f = to_float(up)
    blurred = cv2.GaussianBlur(f, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    mask = np.abs(f - blurred)
    return to_uint8(f + 0.8 * mask * (f - blurred) * 4.0)


METHODS: dict[str, Callable[..., np.ndarray]] = {
    "Nearest": sr_nearest,
    "Bilinear": sr_bilinear,
    "Bicubic": sr_bicubic,
    "Lanczos-4": sr_lanczos,
    "Edge-directed": sr_edi,
    "Back-projection": sr_back_projection,
}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **texture density** — the fraction of spectral
#: energy above a quarter Nyquist. That is precisely the content downsampling
#: destroys and upsampling has to invent, so a pool that did not vary along it
#: would be twelve runs of the same experiment. Selected by
#: `tools/select_images.py --axis texture`; the range is 53.5 to 84.3.
IMAGES = (
    "hawk_in_scrub",        # texture 53.5
    "helicopter_dusk",      # texture 60.4 — large smooth sky, little to lose
    "rider_and_herd",       # texture 63.8
    "indian_corn",          # texture 66.3
    "man_green_parka",      # texture 67.6
    "longtail_boats",       # texture 68.9
    "moated_chateau",       # texture 70.2 — fine roof detail and a reflection
    "woman_wading",         # texture 71.5
    "parasols_willows",     # texture 72.7
    "steam_train_viaduct",  # texture 74.4
    "shark_shallows",       # texture 77.0
    "raked_zen_garden",     # texture 84.3 — the finest repeating texture here
)
SCALES = (2, 3, 4, 6, 8)

ORACLE_NAME = "Band-limited reference"


def load_scene(name: str) -> np.ndarray:
    """Load one of this project's photographs by name."""
    from shared import io

    return io.real_photo(name)


def oracle_band_limited(hr: np.ndarray, scale: int = 4) -> np.ndarray:
    """The original, low-passed by the same blur the downsample applied.

    Handed the high-resolution image, so it is not a method. It answers the
    question the table otherwise cannot: *how much of this picture was still
    present in the low-resolution file at all?* Downsampling blurs and then
    throws away every other pixel; applying the blur without the decimation
    gives the image a reconstruction is aiming at.

    **It is a reference, not a ceiling, and the difference is measurable.** A
    Gaussian is not a brick-wall filter: it attenuates the frequencies above the
    new Nyquist rather than removing them, so the decimated samples still carry
    a folded, weakened copy of them. A method that models the degradation can
    partly invert that attenuation and score *above* this row. Back-projection
    does exactly that on 1 of the 12 photographs here, by 0.35 dB, and comes in
    0.2 to 0.5 dB below it on the other eleven.

    Calling it an oracle would have been the tidier story and the false one.
    What it actually marks is the point past which an interpolator has to start
    modelling the degradation rather than just resampling -- which is precisely
    the line back-projection crosses and the other five do not.
    """
    from shared import synth

    sigma = 0.5 * scale
    return cv2.GaussianBlur(hr, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def evaluate_methods(scale: int = 4, images=IMAGES, runs: int = 3, noise_sigma: float = 0.0):
    """Score every method against the true high-resolution original."""
    from shared import io, synth

    acc = {n: {"psnr": [], "ssim": [], "ms": []} for n in list(METHODS) + [ORACLE_NAME]}

    for i, name in enumerate(images):
        hr = load_scene(name)
        # crop so the dimensions divide exactly by the scale; otherwise the
        # comparison silently includes a resampling mismatch at the edge
        h = (hr.shape[0] // scale) * scale
        w = (hr.shape[1] // scale) * scale
        hr = hr[:h, :w]

        lr = synth.downsample_for_sr(hr, scale=scale)
        if noise_sigma > 0:
            lr = synth.gaussian_noise(lr, sigma=noise_sigma, seed=i)

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(lr, scale), runs=runs, warmup=1)
            out = out[:h, :w]
            acc[method]["psnr"].append(psnr(out, hr))
            acc[method]["ssim"].append(ssim(out, hr))
            acc[method]["ms"].append(timing.median_ms)

        rec, timing = timeit(lambda: oracle_band_limited(hr, scale), runs=1, warmup=0)
        acc[ORACLE_NAME]["psnr"].append(psnr(rec, hr))
        acc[ORACLE_NAME]["ssim"].append(ssim(rec, hr))
        acc[ORACLE_NAME]["ms"].append(timing.median_ms)

    rows = [
        {
            "method": m,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for m, a in acc.items()
    ]
    real = [r for r in rows if r["method"] != ORACLE_NAME]
    best = max(r["psnr_db"] for r in real)
    worst = min(r["psnr_db"] for r in real)
    oracle = next(r["psnr_db"] for r in rows if r["method"] == ORACLE_NAME)
    return rows, {
        "psnr_spread_db": round(best - worst, 3),
        "oracle_headroom_db": round(oracle - best, 3),
    }


def sweep_scale(images=IMAGES, scales=SCALES):
    """The plateau, traced across scale factors.

    If interpolation cannot add information, the *spread* between methods should
    stay small at every scale even as all of them get worse together.
    """
    rows = []
    for s in scales:
        scored, extra = evaluate_methods(scale=s, images=images, runs=1)
        row: dict[str, float | int] = {
            "scale": s,
            "spread_db": extra["psnr_spread_db"],
            "oracle_headroom_db": extra["oracle_headroom_db"],
        }
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def compare_degradations(images=IMAGES, scale: int = 4):
    """Prove the degradation choice matters, rather than asserting it.

    Scores bicubic upsampling against the same original under two different
    downsampling operators. If the numbers differ materially, then any SR
    comparison that used ``cv2.resize`` to build its low-resolution inputs was
    measuring the wrong inverse problem.
    """
    from shared import io, synth

    correct, naive, difference = [], [], []
    for name in images:
        hr = load_scene(name)
        h = (hr.shape[0] // scale) * scale
        w = (hr.shape[1] // scale) * scale
        hr = hr[:h, :w]

        lr_correct = synth.downsample_for_sr(hr, scale=scale)
        lr_naive = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

        correct.append(psnr(sr_bicubic(lr_correct, scale)[:h, :w], hr))
        naive.append(psnr(sr_bicubic(lr_naive, scale)[:h, :w], hr))
        difference.append(psnr(lr_correct, lr_naive))

    return {
        "blur_then_decimate_psnr_db": round(float(np.mean(correct)), 3),
        "cv2_resize_area_psnr_db": round(float(np.mean(naive)), 3),
        "difference_db": round(float(np.mean(naive)) - float(np.mean(correct)), 3),
        "lr_images_differ_psnr_db": round(float(np.mean(difference)), 3),
    }


def frequency_content(img: np.ndarray) -> float:
    """Fraction of spectral energy above half the Nyquist frequency.

    The direct test of "did anything get added": upsampling cannot create energy
    in bands the low-resolution image did not contain, so this number should stay
    far below the original's for every method.
    """
    f = to_float(to_gray(img))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = radius > min(h, w) / 4.0
    return float(spectrum[high].sum() / max(spectrum.sum(), EPS))


def evaluate_frequency(scale: int = 4, images=IMAGES):
    """High-frequency energy recovered by each method, against the original."""
    from shared import io, synth

    rows = []
    originals = []
    per = {n: [] for n in METHODS}
    for name in images:
        hr = load_scene(name)
        h = (hr.shape[0] // scale) * scale
        w = (hr.shape[1] // scale) * scale
        hr = hr[:h, :w]
        originals.append(frequency_content(hr))
        lr = synth.downsample_for_sr(hr, scale=scale)
        for method, fn in METHODS.items():
            per[method].append(frequency_content(fn(lr, scale)[:h, :w]))

    rows.append({"method": "Original (truth)", "high_freq_energy": round(float(np.mean(originals)), 5)})
    for method, vals in per.items():
        rows.append({"method": method, "high_freq_energy": round(float(np.mean(vals)), 5)})
    return rows


def upscale(img: np.ndarray, method: str = "Bicubic", scale: int = 4) -> np.ndarray:
    return METHODS[method](img, scale)
