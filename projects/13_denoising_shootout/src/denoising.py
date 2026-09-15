"""Denoising shootout: six filters, three noise types, and the cost of each.

The question
------------
Everyone knows "median kills salt-and-pepper". Fewer people can say what it costs
on Gaussian noise, or how much you pay in milliseconds for the edge-preserving
filters that win on quality.

> **The claim under test:** there is no best denoiser — the winner changes with
> the noise *type*, and the method that wins on quality loses on speed by two
> orders of magnitude.

The noise is generated, so sigma and density are known exactly and PSNR/SSIM are
measured against the true clean image rather than against another algorithm's
output.

An **oracle** is included: the clean image itself, scored against itself, is
meaningless — so instead the oracle here is *the best achievable by any of these
filters at its own best parameter*, found by sweeping. That separates "this
filter is weak" from "this filter was badly tuned", which is the usual reason
denoising comparisons disagree with each other.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.metrics import psnr, ssim

EPS = 1e-6


# --------------------------------------------------------------------------- #
# the filters
# --------------------------------------------------------------------------- #


def denoise_box(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Unweighted local average. The cheapest thing that works, and it blurs."""
    return cv2.blur(img, (ksize, ksize))


def denoise_gaussian(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian average — optimal for additive Gaussian noise *if* the image were flat.

    It is the linear minimum-mean-squared-error filter for white noise on a
    constant signal. Real images are not constant, which is exactly why it blurs
    edges: it has no way to know an edge is not noise.
    """
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def denoise_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median of the neighbourhood — an order statistic, not an average.

    An outlier cannot drag a median the way it drags a mean, so impulse noise is
    rejected outright rather than smeared. On Gaussian noise, where every pixel
    is slightly wrong rather than a few being completely wrong, that advantage
    disappears.
    """
    return cv2.medianBlur(img, ksize)


def denoise_bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 50.0,
                      sigma_space: float = 9.0) -> np.ndarray:
    """Gaussian in space *and* in intensity — averages only similar pixels.

    ``sigma_color`` is the parameter that matters: it decides how different two
    pixels can be and still be averaged together. Set it above the noise level
    and it behaves like a Gaussian blur; set it below and it does nothing.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def denoise_nlm(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-local means — average over similar *patches* anywhere in the image.

    The insight is that an image repeats itself: a patch of brick wall has
    hundreds of near-identical patches elsewhere. Averaging those preserves
    detail a local filter must destroy, at a cost of one or two orders of
    magnitude in time.
    """
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def denoise_wiener(img: np.ndarray, ksize: int = 5, noise_var: float | None = None) -> np.ndarray:
    """Local adaptive Wiener filter.

    Estimates the local mean and variance and shrinks each pixel toward its local
    mean in proportion to how much of the local variance is noise. Where the
    image is flat it smooths hard; where it is detailed it barely touches
    anything. That adaptivity is the whole idea.
    """
    f = img.astype(np.float32) / 255.0
    mean = cv2.blur(f, (ksize, ksize))
    sq = cv2.blur(f * f, (ksize, ksize))
    var = np.maximum(sq - mean * mean, 0.0)
    nv = float(np.mean(var)) if noise_var is None else noise_var
    out = mean + np.maximum(var - nv, 0.0) / np.maximum(var, nv + EPS) * (f - mean)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def denoise_identity(img: np.ndarray) -> np.ndarray:
    """Do nothing — the control.

    At low noise this beats several real filters, because their blurring costs
    more than the noise does. Knowing where that crossover is, is the practically
    useful part of the whole comparison.
    """
    return img.copy()


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Box": denoise_box,
    "Gaussian": denoise_gaussian,
    "Median": denoise_median,
    "Bilateral": denoise_bilateral,
    "Non-local means": denoise_nlm,
    "Wiener (adaptive)": denoise_wiener,
    "Do nothing (control)": denoise_identity,
}


# --------------------------------------------------------------------------- #
# noise models
# --------------------------------------------------------------------------- #


def make_noisy(clean: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    from shared import synth

    if kind == "gaussian":
        return synth.gaussian_noise(clean, sigma=level, seed=seed)
    if kind == "salt_pepper":
        return synth.salt_pepper_noise(clean, density=level, seed=seed)
    if kind == "poisson":
        return synth.poisson_noise(clean, lam=level, seed=seed)
    raise ValueError(f"unknown noise kind {kind!r}")


#: (kind, level, human label). Levels chosen so all three are visibly corrupted.
NOISE_TYPES = (
    ("gaussian", 25.0, "Gaussian sigma=25"),
    ("salt_pepper", 0.06, "Salt & pepper 6%"),
    ("poisson", 30.0, "Poisson lambda=30"),
)

IMAGES = ("astronaut", "coffee", "chelsea", "camera", "brick", "moon")


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_methods(kind: str = "gaussian", level: float = 25.0, images=IMAGES, runs: int = 3):
    """Score every filter on one noise type."""
    from shared import io

    acc = {n: {"psnr": [], "ssim": [], "ms": []} for n in METHODS}
    noisy_stats = {"psnr": [], "ssim": []}

    for i, name in enumerate(images):
        clean = io.sample(name)
        noisy = make_noisy(clean, kind, level, seed=i)
        noisy_stats["psnr"].append(psnr(noisy, clean))
        noisy_stats["ssim"].append(ssim(noisy, clean))
        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(noisy), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["ms"].append(timing.median_ms)

    return (
        [
            {
                "method": m,
                "psnr_db": round(float(np.mean(a["psnr"])), 3),
                "ssim": round(float(np.mean(a["ssim"])), 4),
                "median_ms": round(float(np.median(a["ms"])), 3),
            }
            for m, a in acc.items()
        ],
        {k: round(float(np.mean(v)), 4) for k, v in noisy_stats.items()},
    )


def compare_noise_types(images=IMAGES):
    """The central table: every filter against every noise type.

    This is where "median kills salt-and-pepper" becomes a number, and where the
    claim that it is *worse* on Gaussian noise gets tested rather than repeated.
    """
    rows = []
    for kind, level, label in NOISE_TYPES:
        scored, _ = evaluate_methods(kind=kind, level=level, images=images, runs=1)
        row: dict[str, float | str] = {"noise": label}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def sweep_level(kind: str = "gaussian", levels=(5.0, 10.0, 20.0, 35.0, 50.0), images=IMAGES):
    """Trace every filter as the noise gets worse, including the do-nothing control.

    The crossing point where each filter overtakes "do nothing" is the number
    that tells you when to bother denoising at all.
    """
    rows = []
    for lv in levels:
        scored, noisy = evaluate_methods(kind=kind, level=lv, images=images, runs=1)
        row: dict[str, float] = {"level": lv, "noisy_input": noisy["psnr"]}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def tune_parameter(images=IMAGES, kind: str = "gaussian", level: float = 25.0):
    """Best achievable PSNR per filter at its own best parameter.

    Denoising comparisons disagree with each other mostly because of tuning. This
    sweeps each filter's main knob and reports its best, so a poor score means
    the filter is genuinely weak on this noise rather than badly configured.
    """
    from shared import io

    grids: dict[str, tuple[str, tuple]] = {
        "Box": ("ksize", (3, 5, 7, 9)),
        "Gaussian": ("sigma", (0.8, 1.2, 1.8, 2.5, 3.5)),
        "Median": ("ksize", (3, 5, 7, 9)),
        "Bilateral": ("sigma_color", (15.0, 35.0, 55.0, 80.0, 120.0)),
        "Non-local means": ("h", (4.0, 8.0, 12.0, 18.0, 25.0)),
        "Wiener (adaptive)": ("ksize", (3, 5, 7, 9)),
    }
    fns = {
        "Box": denoise_box,
        "Gaussian": denoise_gaussian,
        "Median": denoise_median,
        "Bilateral": denoise_bilateral,
        "Non-local means": denoise_nlm,
        "Wiener (adaptive)": denoise_wiener,
    }

    rows = []
    for method, (param, values) in grids.items():
        best_val, best_psnr = None, -np.inf
        for v in values:
            scores = []
            for i, name in enumerate(images):
                clean = io.sample(name)
                noisy = make_noisy(clean, kind, level, seed=i)
                scores.append(psnr(fns[method](noisy, **{param: v}), clean))
            mean = float(np.mean(scores))
            if mean > best_psnr:
                best_psnr, best_val = mean, v
        rows.append(
            {
                "method": method,
                "parameter": param,
                "best_value": best_val,
                "best_psnr_db": round(best_psnr, 3),
            }
        )
    return rows


def denoise(img: np.ndarray, method: str = "Non-local means") -> np.ndarray:
    return METHODS[method](img)
