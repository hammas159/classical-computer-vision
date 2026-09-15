"""Histogram equalisation family: five ways to redistribute tone.

The question
------------
Equalisation flattens the intensity histogram. That is a *statistical* goal, not
a perceptual one, and the gap between them is the project.

> **The claim under test:** global HE maximises entropy and often makes the image
> worse, because flattening a histogram is not the same as improving it. CLAHE's
> clip limit is the parameter that decides whether local equalisation helps or
> just amplifies noise — and it has an optimum, not a "higher is better".

Scoring uses three different kinds of measure on purpose:

* **full reference** (PSNR, SSIM against a known original) where a true image
  exists — the degradation is generated, so it does;
* **no reference** (entropy, RMS contrast) which is what you are stuck with on a
  real photo, and which *disagrees* with the full-reference metrics — that
  disagreement is itself a finding;
* **noise amplification**, because every contrast operation multiplies whatever
  noise was already there.

Colour handling is not incidental: equalising R, G and B independently shifts the
colour balance and produces the lurid output people associate with HE. Every
method here works on luminance only, which is the fair version.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import entropy, estimate_noise_sigma, psnr, rms_contrast, ssim

EPS = 1e-6


def _on_luma(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a grayscale operation to the Y channel of YCrCb only.

    Doing this once, here, is what stops every method in the comparison from
    having its own colour bug.
    """
    if img.ndim == 2:
        return fn(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = fn(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def eq_none(img: np.ndarray) -> np.ndarray:
    """Do nothing — the control."""
    return img.copy()


def eq_global(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation: map the CDF onto a straight line.

    Provably maximises entropy for a discrete histogram. It is also completely
    blind to *where* the pixels are, so a large flat region dictates the mapping
    for the whole image.
    """
    return _on_luma(img, cv2.equalizeHist)


def eq_ahe(img: np.ndarray, grid: int = 8) -> np.ndarray:
    """Adaptive HE — CLAHE with the clip limit effectively disabled.

    Included to isolate what the clip limit actually buys. Unclipped local
    equalisation amplifies noise without bound in flat tiles, which is the exact
    failure CLAHE exists to prevent.
    """
    return _on_luma(
        img, lambda g: cv2.createCLAHE(clipLimit=40.0, tileGridSize=(grid, grid)).apply(g)
    )


def eq_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited AHE. The clip limit caps each tile's histogram."""
    return _on_luma(
        img, lambda g: cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(g)
    )


def eq_match(img: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Histogram matching: reshape the histogram to match a reference image.

    More useful than equalisation in practice — matching a well-exposed
    reference is a *targeted* transform, where equalisation aims at a flat
    histogram nobody actually wants.
    """
    from shared import io as shared_io

    ref = reference if reference is not None else shared_io.sample("coffee")

    def _match(g: np.ndarray) -> np.ndarray:
        r = to_gray(ref)
        src_hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        ref_hist = np.bincount(r.ravel(), minlength=256).astype(np.float64)
        src_cdf = np.cumsum(src_hist) / max(src_hist.sum(), 1)
        ref_cdf = np.cumsum(ref_hist) / max(ref_hist.sum(), 1)
        lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
        return lut[g]

    return _on_luma(img, _match)


def eq_gamma(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """A power-law curve — a fixed, monotonic tone map that adapts to nothing.

    The cheapest possible contrast operation, and the honest control for whether
    any of the adaptive machinery is earning its keep.
    """
    return to_uint8(np.power(to_float(img), gamma))


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Do nothing (control)": eq_none,
    "Global HE": eq_global,
    "AHE (unclipped)": eq_ahe,
    "CLAHE (clip 2.0)": eq_clahe,
    "Histogram matching": eq_match,
    "Gamma 0.6": eq_gamma,
}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("moon", "coffee", "chelsea", "astronaut", "camera", "retina")
CLIP_LEVELS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)
GRID_LEVELS = (2, 4, 8, 16, 32)


def degrade(clean: np.ndarray, kind: str = "low_contrast", noise_sigma: float = 3.0, seed: int = 0):
    """Produce an image that genuinely needs equalising, with truth retained.

    ``low_contrast`` compresses the tone range toward mid-grey, which is what a
    hazy or flat-lit photograph actually looks like. ``gamma`` darkens instead.
    """
    from shared import synth

    f = to_float(clean)
    if kind == "low_contrast":
        out = to_uint8(f * 0.35 + 0.33)
    elif kind == "gamma":
        out = to_uint8(np.power(f, 2.2))
    else:
        raise ValueError(f"unknown degradation {kind!r}")
    if noise_sigma > 0:
        out = synth.gaussian_noise(out, sigma=noise_sigma, seed=seed)
    return out


def evaluate_methods(kind: str = "low_contrast", images=IMAGES, noise_sigma: float = 3.0, runs: int = 3):
    """Score every method with full-reference *and* no-reference metrics."""
    from shared import io

    keys = ("psnr", "ssim", "entropy", "contrast", "noise", "ms")
    acc = {n: {k: [] for k in keys} for n in METHODS}
    degraded_stats = {"psnr": [], "entropy": [], "contrast": [], "noise": []}

    for i, name in enumerate(images):
        clean = io.sample(name)
        bad = degrade(clean, kind=kind, noise_sigma=noise_sigma, seed=i)
        degraded_stats["psnr"].append(psnr(bad, clean))
        degraded_stats["entropy"].append(entropy(bad))
        degraded_stats["contrast"].append(rms_contrast(bad))
        degraded_stats["noise"].append(estimate_noise_sigma(bad))

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(bad), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["entropy"].append(entropy(out))
            acc[method]["contrast"].append(rms_contrast(out))
            acc[method]["noise"].append(estimate_noise_sigma(out))
            acc[method]["ms"].append(timing.median_ms)

    rows = [
        {
            "method": m,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "entropy_bits": round(float(np.mean(a["entropy"])), 3),
            "rms_contrast": round(float(np.mean(a["contrast"])), 4),
            "noise_sigma": round(float(np.mean(a["noise"])), 3),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for m, a in acc.items()
    ]
    return rows, {k: round(float(np.mean(v)), 4) for k, v in degraded_stats.items()}


def sweep_clip_limit(images=IMAGES, clips=CLIP_LEVELS, noise_sigma: float = 3.0):
    """The parameter that decides whether CLAHE helps.

    PSNR against the truth should peak at a moderate clip and fall away either
    side, while entropy rises monotonically — so the two metrics point at
    different "best" settings, which is the point.
    """
    from shared import io

    rows = []
    for clip in clips:
        p, s, e, n = [], [], [], []
        for i, name in enumerate(images):
            clean = io.sample(name)
            bad = degrade(clean, noise_sigma=noise_sigma, seed=i)
            out = eq_clahe(bad, clip=clip)
            p.append(psnr(out, clean))
            s.append(ssim(out, clean))
            e.append(entropy(out))
            n.append(estimate_noise_sigma(out))
        rows.append(
            {
                "clip_limit": clip,
                "psnr_db": round(float(np.mean(p)), 3),
                "ssim": round(float(np.mean(s)), 4),
                "entropy_bits": round(float(np.mean(e)), 3),
                "noise_sigma": round(float(np.mean(n)), 3),
            }
        )
    return rows


def sweep_grid(images=IMAGES, grids=GRID_LEVELS, clip: float = 2.0):
    """Tile size: too few tiles is global HE, too many is per-pixel noise."""
    from shared import io

    rows = []
    for grid in grids:
        p, e = [], []
        for i, name in enumerate(images):
            clean = io.sample(name)
            bad = degrade(clean, seed=i)
            out = eq_clahe(bad, clip=clip, grid=grid)
            p.append(psnr(out, clean))
            e.append(entropy(out))
        rows.append(
            {
                "grid": grid,
                "psnr_db": round(float(np.mean(p)), 3),
                "entropy_bits": round(float(np.mean(e)), 3),
            }
        )
    return rows


def metric_disagreement(images=IMAGES):
    """Rank the methods by each metric and report where the rankings differ.

    If entropy and PSNR disagree about the winner — and they should — then every
    "our method improves contrast" claim that quotes only a no-reference metric
    is unfalsifiable.
    """
    rows, _ = evaluate_methods(images=images, runs=1)
    by_psnr = [r["method"] for r in sorted(rows, key=lambda r: -r["psnr_db"])]
    by_entropy = [r["method"] for r in sorted(rows, key=lambda r: -r["entropy_bits"])]
    by_contrast = [r["method"] for r in sorted(rows, key=lambda r: -r["rms_contrast"])]
    return {
        "rank_by_psnr": by_psnr,
        "rank_by_entropy": by_entropy,
        "rank_by_contrast": by_contrast,
        "psnr_winner": by_psnr[0],
        "entropy_winner": by_entropy[0],
        "contrast_winner": by_contrast[0],
        "winners_agree": by_psnr[0] == by_entropy[0] == by_contrast[0],
    }


def equalise(img: np.ndarray, method: str = "CLAHE (clip 2.0)") -> np.ndarray:
    return METHODS[method](img)
