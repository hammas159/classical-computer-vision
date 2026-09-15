"""Do quality metrics agree? The meta-project.

The question
------------
Every other project in this repo quotes PSNR or SSIM or IoU. This one asks
whether those numbers can be trusted to *rank* anything.

> **The claim under test:** PSNR and SSIM rank the same images differently, and
> the disagreement is systematic rather than random — it depends on the
> **type** of degradation. If true, then every "our method has better PSNR"
> claim is conditional on a distortion type that is usually left unstated.

The design is what makes this answerable. Take one clean image. Apply several
different degradations, each tuned so that **PSNR is held constant**. If PSNR is
a complete description of quality, all of those images are equally good. Then ask
the other metrics — and ask whether the images look equally good, which they
plainly do not.

That is a controlled experiment rather than a correlation plot, and it isolates
exactly one thing: what PSNR cannot see.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.io import to_float, to_gray, to_uint8
from shared.metrics import entropy, mse, psnr, rms_contrast, ssim

EPS = 1e-9


# --------------------------------------------------------------------------- #
# the metrics
# --------------------------------------------------------------------------- #


def metric_mse(a, b) -> float:
    return mse(a, b)


def metric_psnr(a, b) -> float:
    return psnr(a, b)


def metric_ssim(a, b) -> float:
    return ssim(a, b)


def metric_ms_ssim(a, b, levels: int = 3) -> float:
    """Multi-scale SSIM: SSIM computed over an image pyramid, then combined.

    Single-scale SSIM depends on viewing distance in disguise — its window is a
    fixed pixel size, so what counts as "structure" changes with resolution.
    Averaging across scales removes that dependence, which is why MS-SSIM
    correlates better with human judgement than SSIM does.
    """
    scores = []
    x, y = a.copy(), b.copy()
    for _ in range(levels):
        if min(x.shape[:2]) < 32:
            break
        scores.append(ssim(x, y))
        x = cv2.pyrDown(x)
        y = cv2.pyrDown(y)
    return float(np.mean(scores)) if scores else float("nan")


def metric_gmsd(a, b) -> float:
    """Gradient Magnitude Similarity Deviation. **Lower is better.**

    Compares gradient magnitude maps and reports their standard deviation rather
    than their mean. The deviation is the point: an image with a few badly
    damaged regions and a mostly perfect rest scores worse than one damaged
    evenly, which matches how people actually judge images and which no
    mean-based metric can express.
    """
    ga = cv2.Sobel(to_float(to_gray(a)), cv2.CV_32F, 1, 0, 3)
    gb = cv2.Sobel(to_float(to_gray(b)), cv2.CV_32F, 1, 0, 3)
    ga2 = cv2.Sobel(to_float(to_gray(a)), cv2.CV_32F, 0, 1, 3)
    gb2 = cv2.Sobel(to_float(to_gray(b)), cv2.CV_32F, 0, 1, 3)
    ma = np.sqrt(ga**2 + ga2**2)
    mb = np.sqrt(gb**2 + gb2**2)
    c = 0.0026
    gms = (2 * ma * mb + c) / (ma**2 + mb**2 + c)
    return float(np.std(gms))


def metric_vif_approx(a, b) -> float:
    """A multi-scale information-fidelity approximation. Higher is better.

    Full VIF models the human visual system with a Gaussian scale mixture. This
    keeps the core idea — the ratio of mutual information surviving in the
    distorted image to that in the reference — using local variances at several
    scales, and is labelled an approximation rather than passed off as VIF.
    """
    num, den = 0.0, 0.0
    x, y = to_float(to_gray(a)), to_float(to_gray(b))
    for scale in range(4):
        if scale > 0:
            x = cv2.pyrDown(x)
            y = cv2.pyrDown(y)
        if min(x.shape) < 16:
            break
        k = (7, 7)
        mu_x, mu_y = cv2.blur(x, k), cv2.blur(y, k)
        var_x = cv2.blur(x * x, k) - mu_x * mu_x
        var_y = cv2.blur(y * y, k) - mu_y * mu_y
        cov = cv2.blur(x * y, k) - mu_x * mu_y
        g = cov / (var_x + EPS)
        sv = np.maximum(var_y - g * cov, 0.0)
        sigma_nsq = 0.004
        num += float(np.sum(np.log10(1.0 + (g**2) * var_x / (sv + sigma_nsq) + EPS)))
        den += float(np.sum(np.log10(1.0 + var_x / sigma_nsq + EPS)))
    return float(num / max(den, EPS))


#: name -> (function, higher_is_better)
METRICS: dict[str, tuple[Callable, bool]] = {
    "MSE": (metric_mse, False),
    "PSNR": (metric_psnr, True),
    "SSIM": (metric_ssim, True),
    "MS-SSIM": (metric_ms_ssim, True),
    "GMSD": (metric_gmsd, False),
    "VIF (approx)": (metric_vif_approx, True),
}


# --------------------------------------------------------------------------- #
# degradations, each with one strength knob
# --------------------------------------------------------------------------- #


def degrade_gaussian_noise(img, strength: float):
    from shared import synth

    return synth.gaussian_noise(img, sigma=strength, seed=0)


def degrade_blur(img, strength: float):
    return cv2.GaussianBlur(img, (0, 0), max(strength, 0.01), borderType=cv2.BORDER_REFLECT)


def degrade_jpeg(img, strength: float):
    """JPEG compression at a given quality. ``strength`` is 100 - quality."""
    quality = int(np.clip(100 - strength, 1, 100))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img.copy()
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade_contrast(img, strength: float):
    """Reduce contrast toward mid-grey. A purely global, structure-preserving change."""
    f = to_float(img)
    k = np.clip(1.0 - strength, 0.0, 1.0)
    return to_uint8((f - 0.5) * k + 0.5)


def degrade_shift(img, strength: float):
    """Translate the image by a few pixels.

    The adversarial case for PSNR: a one-pixel shift destroys per-pixel
    correspondence while preserving every bit of the image's content. A human
    would call it unchanged; PSNR calls it badly damaged.
    """
    dx = int(round(strength))
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def degrade_salt_pepper(img, strength: float):
    from shared import synth

    return synth.salt_pepper_noise(img, density=strength, seed=0)


DEGRADATIONS: dict[str, tuple[Callable, tuple]] = {
    "Gaussian noise": (degrade_gaussian_noise, (1.0, 80.0)),
    "Blur": (degrade_blur, (0.2, 8.0)),
    "JPEG": (degrade_jpeg, (1.0, 97.0)),
    "Contrast loss": (degrade_contrast, (0.02, 0.9)),
    "Sub-pixel shift": (degrade_shift, (1.0, 8.0)),
    "Salt & pepper": (degrade_salt_pepper, (0.001, 0.2)),
}


# --------------------------------------------------------------------------- #
# the controlled experiment: equalise PSNR, then ask the others
# --------------------------------------------------------------------------- #


def find_strength_for_psnr(
    img: np.ndarray, degradation: str, target_psnr: float, tolerance: float = 0.05
):
    """Bisect a degradation's strength until it hits a target PSNR.

    This is what makes the comparison controlled. Every degraded image in the
    experiment is, by construction, exactly as good as every other *according to
    PSNR* — so any disagreement between the remaining metrics is attributable to
    the distortion type and nothing else.
    """
    fn, (lo, hi) = DEGRADATIONS[degradation]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        value = psnr(fn(img, mid), img)
        if not np.isfinite(value):
            lo = mid
            continue
        if abs(value - target_psnr) < tolerance:
            return mid, value
        if value > target_psnr:
            lo = mid   # too good, degrade harder
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, psnr(fn(img, mid), img)


IMAGES = ("astronaut", "coffee", "chelsea", "camera", "brick", "moon")
TARGET_PSNRS = (24.0, 28.0, 32.0)


def equal_psnr_comparison(target_psnr: float = 28.0, images=IMAGES):
    """Build images of equal PSNR and score them with every metric.

    If PSNR were sufficient, every row would be identical under every metric.
    """
    from shared import io

    rows = []
    for degradation in DEGRADATIONS:
        scores = {name: [] for name in METRICS}
        achieved, strengths = [], []
        for image in images:
            clean = io.sample(image)
            strength, got = find_strength_for_psnr(clean, degradation, target_psnr)
            fn, _ = DEGRADATIONS[degradation]
            bad = fn(clean, strength)
            achieved.append(got)
            strengths.append(strength)
            for name, (metric, _) in METRICS.items():
                scores[name].append(metric(bad, clean))
        row: dict[str, float | str] = {
            "degradation": degradation,
            "strength": round(float(np.mean(strengths)), 4),
            "achieved_psnr": round(float(np.mean(achieved)), 3),
        }
        for name, vals in scores.items():
            row[name] = round(float(np.nanmean(vals)), 5)
        rows.append(row)
    return rows


def ranking_disagreement(target_psnr: float = 28.0, images=IMAGES):
    """Rank the equal-PSNR images by each metric and measure how much they differ.

    Reported as Kendall's tau against the PSNR ranking. A tau well below 1 means
    the metrics genuinely disagree about which distortion is worse, at equal PSNR.
    """
    rows = equal_psnr_comparison(target_psnr, images)
    names = [r["degradation"] for r in rows]

    rankings: dict[str, list[str]] = {}
    for metric, (_, higher_better) in METRICS.items():
        order = sorted(rows, key=lambda r: r[metric], reverse=higher_better)
        rankings[metric] = [r["degradation"] for r in order]

    def kendall_tau(a: list[str], b: list[str]) -> float:
        pos_a = {n: i for i, n in enumerate(a)}
        pos_b = {n: i for i, n in enumerate(b)}
        concordant = discordant = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = names[i], names[j]
                s = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
                if s > 0:
                    concordant += 1
                elif s < 0:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total else 1.0

    reference = rankings["PSNR"]
    return {
        "rankings": rankings,
        "kendall_tau_vs_psnr": {
            m: round(kendall_tau(reference, r), 4) for m, r in rankings.items()
        },
        "worst_by_psnr": reference[-1],
        "worst_by_ssim": rankings["SSIM"][-1],
        "psnr_and_ssim_agree": reference == rankings["SSIM"],
    }


def sweep_strength(degradation: str = "Blur", images=IMAGES, steps: int = 9):
    """Every metric traced across one degradation's full strength range.

    Normalised to [0, 1] per metric so curves with different units can be
    compared for *shape*. A metric that saturates early is one that stops being
    able to tell bad from terrible.
    """
    from shared import io

    fn, (lo, hi) = DEGRADATIONS[degradation]
    rows = []
    for strength in np.linspace(lo, hi, steps):
        scores = {name: [] for name in METRICS}
        for image in images:
            clean = io.sample(image)
            bad = fn(clean, float(strength))
            for name, (metric, _) in METRICS.items():
                scores[name].append(metric(bad, clean))
        row: dict[str, float] = {"strength": round(float(strength), 4)}
        for name, vals in scores.items():
            v = float(np.nanmean(vals))
            row[name] = round(v, 5) if np.isfinite(v) else None
        rows.append(row)
    return rows


def shift_sensitivity(images=IMAGES, shifts=(0, 1, 2, 4, 8)):
    """The clearest single demonstration that PSNR is not perceptual.

    A translated image contains exactly the same content. PSNR collapses anyway,
    because it compares pixel to pixel with no notion of correspondence.
    """
    from shared import io

    rows = []
    for dx in shifts:
        scores = {name: [] for name in METRICS}
        for image in images:
            clean = io.sample(image)
            shifted = degrade_shift(clean, dx)
            for name, (metric, _) in METRICS.items():
                scores[name].append(metric(shifted, clean))
        row: dict[str, float] = {"shift_px": dx}
        for name, vals in scores.items():
            v = float(np.nanmean(vals))
            row[name] = round(v, 5) if np.isfinite(v) else None
        rows.append(row)
    return rows


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """All metrics at once, for the UI."""
    return {name: float(fn(pred, truth)) for name, (fn, _) in METRICS.items()}
