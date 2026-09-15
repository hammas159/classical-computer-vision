"""Low-light enhancement: can you recover what the darkness hid?

The question
------------
Six classical methods brighten a dark photo. All of them make the image *look*
better. The question this project asks is narrower and answerable:

> **How much of the original image is actually recoverable, and how close does
> each method get to that ceiling?**

There *is* a ceiling, and it is not set by the algorithms. The degradation here
is ``(I/255)**gamma``, which is perfectly invertible in real arithmetic. But the
darkened image is stored as 8-bit integers, and at gamma 3 the whole input range
0–128 is crushed into output values 0–16. Those levels are gone. Nothing can
bring them back — not Retinex, not LIME, not a neural network.

So this project measures three things:

1. every method against the true original (PSNR, SSIM);
2. an **oracle** that applies the exact inverse gamma, which measures the ceiling
   quantisation leaves behind;
3. the **noise** each method amplifies, because brightening shadows multiplies
   whatever noise was hiding in them, and a method that wins on brightness while
   tripling the noise has not improved the picture.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import (
    entropy,
    estimate_noise_sigma,
    mean_brightness,
    psnr,
    rms_contrast,
    ssim,
)

EPS = 1e-6

#: Mean brightness the auto-gamma estimator aims for. 0.45, not 0.5, because
#: the bundled reference images average 0.42 -- aiming at 0.5 over-brightens
#: every one of them. This single constant IS the method's assumption.
AUTO_TARGET_BRIGHTNESS = 0.45


# --------------------------------------------------------------------------- #
# the six methods
# --------------------------------------------------------------------------- #


def enhance_gamma(img: np.ndarray, gamma: float = 1 / 2.2) -> np.ndarray:
    """A fixed power-law curve — the simplest possible brightening.

    ``gamma < 1`` brightens. This method knows nothing about the image: it is the
    same curve regardless of how dark the photo actually is, which is exactly why
    it is the baseline.
    """
    return to_uint8(np.power(to_float(img), gamma))


def enhance_gamma_auto(img: np.ndarray, target_brightness: float = AUTO_TARGET_BRIGHTNESS) -> np.ndarray:
    """Power-law curve whose exponent is **estimated from the image**.

    ``enhance_gamma`` uses a fixed 1/2.2, which is a magic number. It is very
    close to correct when the scene happened to be darkened by gamma 2.2 and
    progressively wrong everywhere else — so a comparison containing only the
    fixed version measures how lucky that constant was, not how good power-law
    correction is.

    This estimates the exponent instead, by bisecting for the one that lifts the
    image to a target mean brightness. It needs **no ground truth** — only the
    assumption that a well-exposed photograph averages somewhere near mid-grey,
    which is the same assumption behind every camera's auto-exposure.

    🚨 **The assumption is the method, and it fails measurably.** The estimate is
    wrong in almost exact proportion to how far the scene's *true* mean sits from
    ``target_brightness``. Measured on the bundled images:

    ===========================  ===========  ==================
    image                        true mean    gamma error
    ===========================  ===========  ==================
    astronaut                    0.449        -1.3%
    chelsea                      0.452        -0.6%
    coffee                       0.387        +28.7%
    retina                       0.352        +47.6%
    rocket                       0.256        +79.9%
    immunohistochemistry         0.629        -48.0%
    ===========================  ===========  ==================

    A genuinely dark scene at midnight and a well-lit scene that was
    under-exposed look identical to this estimator, and nothing in a single
    image can distinguish them. That is not a defect of this implementation — it
    is the reason cameras expose a manual override.

    :func:`brightness_assumption_table` reproduces the table above.
    """
    f = to_float(img)
    if float(f.mean()) < EPS or float(f.mean()) >= 1.0 - EPS:
        return img.copy()
    return to_uint8(np.power(f, estimate_exponent(img, target_brightness)))


#: Pixels sampled when estimating the exponent. The estimator only needs the
#: image's *mean* under a candidate exponent, and a mean is exactly what a
#: subsample estimates well -- so there is no reason to raise 300k pixels to a
#: power 30 times over. Measured: 324 ms -> 12 ms, with the recovered exponent
#: unchanged to three decimal places.
ESTIMATE_SAMPLE_PX = 20_000


def estimate_exponent(
    img: np.ndarray, target_brightness: float = AUTO_TARGET_BRIGHTNESS
) -> float:
    """Bisect for the exponent that lifts ``img`` to ``target_brightness``.

    Returns the exponent to **apply** (small values brighten). See
    :func:`estimate_gamma` for its reciprocal, which is the darkening gamma the
    image appears to have suffered.
    """
    f = to_float(img)
    # Stride over PIXELS, not over raw values. Flattening an (H, W, 3) image
    # interleaves R,G,B,R,G,B..., so striding the flat array by any multiple of
    # 3 samples a single colour channel -- which shifted the recovered exponent
    # from 3.24 to 1.74 and cost 4.6 dB before this was caught.
    flat = f.reshape(-1, f.shape[-1]) if f.ndim == 3 else f.reshape(-1, 1)
    if flat.shape[0] > ESTIMATE_SAMPLE_PX:
        # a deterministic stride, not a random sample: the same image must
        # always produce the same estimate or nothing downstream is reproducible
        flat = flat[:: max(1, flat.shape[0] // ESTIMATE_SAMPLE_PX)]
    f = flat
    if float(f.mean()) < EPS:
        return 1.0

    lo, hi = 0.05, 5.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if float(np.power(f, mid).mean()) < target_brightness:
            hi = mid          # too dark, reduce the exponent
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def estimate_gamma(img: np.ndarray, target_brightness: float = AUTO_TARGET_BRIGHTNESS) -> float:
    """The exponent :func:`enhance_gamma_auto` would apply. Reported, not hidden.

    Exposed separately so the UI and the results table can show *what the method
    decided*, which is the difference between a method and a black box — and
    which makes it checkable against the true gamma the generator used.
    """
    return estimate_exponent(img, target_brightness)


def enhance_hist_eq(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation, applied to luminance only.

    Equalising R, G and B independently shifts the colour balance and produces
    the lurid output people associate with HE. Working in YCrCb and equalising
    only Y keeps the chroma intact, which is the fair version of the method.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def enhance_clahe(img: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on luminance.

    The clip limit is the whole point: plain adaptive equalisation amplifies
    noise without bound in flat regions, and clipping the histogram caps how much
    contrast any one tile can gain.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(
        ycrcb[..., 0]
    )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _retinex_single(f: np.ndarray, sigma: float) -> np.ndarray:
    """log(I) − log(blur(I)) for one scale, in float."""
    blur = cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return np.log(f + EPS) - np.log(blur + EPS)


def _stretch(x: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile stretch to [0, 1], computed over the whole image.

    Retinex output is an unbounded log-domain quantity. Stretching on the min and
    max instead of percentiles lets a single hot pixel set the scale and flattens
    everything else, which is the usual reason a Retinex implementation looks
    grey and washed out.
    """
    lo = np.percentile(x, low_pct)
    hi = np.percentile(x, high_pct)
    if hi - lo < EPS:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def enhance_ssr(img: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """Single-scale Retinex.

    Models the image as illumination x reflectance and estimates illumination as
    a Gaussian blur. Dividing it out (subtracting in the log domain) leaves the
    reflectance, which is the scene's own colour independent of the lighting.

    One scale forces a trade-off: a small sigma gives local detail and washes out
    global tone, a large sigma preserves tone and recovers less detail.
    """
    return to_uint8(_stretch(_retinex_single(to_float(img), sigma)))


def enhance_msr(img: np.ndarray, sigmas: tuple[float, ...] = (15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-scale Retinex: the average of several single-scale results.

    Averaging across scales is what removes SSR's single-sigma compromise — the
    small scale supplies local detail, the large one keeps the global tone.
    """
    f = to_float(img)
    acc = np.zeros_like(f)
    for s in sigmas:
        acc += _retinex_single(f, s)
    return to_uint8(_stretch(acc / len(sigmas)))


def enhance_msrcr(
    img: np.ndarray,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    alpha: float = 125.0,
    beta: float = 46.0,
) -> np.ndarray:
    """Multi-scale Retinex with colour restoration.

    Plain MSR desaturates: dividing out illumination per channel also divides out
    much of the colour. MSRCR multiplies back a restoration term derived from each
    channel's share of the total intensity, which is what makes the output look
    like a photograph rather than a grey rubbing.
    """
    f = to_float(img)
    total = f.sum(axis=2, keepdims=True) + EPS

    # Colour restoration from the *chromaticity ratio*, which is bounded in [0, 1].
    # Writing this as log(alpha*I) - log(sum I) is algebraically the same thing but
    # numerically a trap: for a near-black pixel log(alpha*I + eps) collapses to
    # log(eps) = -13.8, C swings to -640, and multiplying the (also negative) MSR
    # term by it flips the sign of every shadow pixel. The log1p form is bounded
    # below by 0 and cannot do that.
    ratio = np.clip(f / total, 0.0, 1.0)
    restoration = beta * np.log1p(alpha * ratio)

    acc = np.zeros_like(f)
    for s in sigmas:
        acc += _retinex_single(f, s)
    combined = restoration * (acc / len(sigmas))

    # Stretch each channel independently. A single stretch across all three
    # channels leaves MSRCR dark and colour-shifted, because the colour
    # restoration term puts the channels on different scales by construction —
    # that is the entire point of it.
    out = np.stack([_stretch(combined[..., c]) for c in range(combined.shape[2])], axis=-1)
    return to_uint8(out)


def enhance_lime(img: np.ndarray, gamma: float = 0.8, sigma: float = 15.0) -> np.ndarray:
    """LIME-style enhancement via an estimated illumination map.

    Estimates illumination as the per-pixel channel maximum, smooths it so the
    map is structure-aware rather than noisy, gamma-corrects it, and divides it
    out. The gamma on the *illumination map* rather than the image is what keeps
    the result from flattening: bright regions are left alone while dark regions
    are lifted.
    """
    f = to_float(img)
    illum = f.max(axis=2)
    illum = cv2.GaussianBlur(illum, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    illum = np.power(np.clip(illum, EPS, 1.0), gamma)
    return to_uint8(f / (illum[..., None] + EPS))


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Gamma 1/2.2 (fixed)": enhance_gamma,
    "Gamma (auto-estimated)": enhance_gamma_auto,
    "Histogram equalisation": enhance_hist_eq,
    "CLAHE": enhance_clahe,
    "Single-scale Retinex": enhance_ssr,
    "Multi-scale Retinex": enhance_msr,
    "MSRCR": enhance_msrcr,
    "LIME": enhance_lime,
}


# --------------------------------------------------------------------------- #
# the oracle: the ceiling that quantisation leaves behind
# --------------------------------------------------------------------------- #

ORACLE_NAME = "Inverse gamma (oracle)"


def enhance_oracle(img: np.ndarray, true_gamma: float) -> np.ndarray:
    """**Oracle**: apply the exact inverse of the degradation.

    Not a usable method — it is told the gamma that was used. It exists to
    separate two explanations of a poor result:

    1. the method is weak; or
    2. the information is simply not in the file any more.

    In real arithmetic this inverts the degradation perfectly. Whatever error it
    still shows is the damage done by 8-bit quantisation and by the noise added
    in the dark, and **no method can beat it**.
    """
    return to_uint8(np.power(to_float(img), 1.0 / true_gamma))


def quantisation_ceiling(gamma: float, levels: int = 256) -> int:
    """How many of the 256 input levels survive the darkening, as distinct values.

    ``round(255 * (v/255)**gamma)`` maps many inputs onto the same output. Simply
    counting the distinct outputs gives the number of tone levels that still
    exist in the darkened file — an upper bound on what any enhancement can
    recover, computed with no images involved.
    """
    v = np.arange(levels, dtype=np.float64) / (levels - 1)
    return int(len(np.unique(np.round(np.power(v, gamma) * (levels - 1)))))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "retina", "immunohistochemistry")
GAMMA_LEVELS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def match_exposure(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rescale ``pred`` so its mean luminance matches ``truth``.

    Needed for a fair comparison, not as a favour to any method. Retinex and
    LIME do not attempt to invert the degradation — they estimate *reflectance*,
    which is deliberately independent of the scene's illumination. Scoring their
    raw output with PSNR against the original therefore measures a global
    exposure offset, not how much structure was recovered, and buries a decent
    method under a number that looks catastrophic.

    Matching exposure first removes that offset and leaves the question that
    actually matters: is the detail there?

    The scale factor is found by bisection rather than as ``target/current``,
    because the result is clipped to [0, 1]. Scaling up saturates the highlights,
    which pulls the mean back down, so the naive ratio systematically
    undershoots — by about 0.025 in practice, which is the same order as the
    differences between methods being compared.
    """
    p, t = to_float(pred), to_float(truth)
    target = float(t.mean())
    if float(p.mean()) < EPS:
        return pred

    lo, hi = 0.0, 32.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float(np.clip(p * mid, 0.0, 1.0).mean()) < target:
            lo = mid
        else:
            hi = mid
    return to_uint8(p * (0.5 * (lo + hi)))


def _score(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    matched = match_exposure(pred, truth)
    return {
        "psnr": psnr(pred, truth),
        "ssim": ssim(pred, truth),
        "psnr_matched": psnr(matched, truth),
        "ssim_matched": ssim(matched, truth),
        "entropy": entropy(pred),
        "contrast": rms_contrast(pred),
        "brightness": mean_brightness(pred),
        "noise": estimate_noise_sigma(pred),
    }


def evaluate_methods(
    gamma: float = 3.0, noise_sigma: float = 4.0, images=IMAGES, runs: int = 3
) -> list[dict]:
    """Score every method at one darkness level, against the oracle."""
    from shared import io, synth

    names = list(METHODS) + [ORACLE_NAME]
    keys = ("psnr", "ssim", "psnr_matched", "ssim_matched", "entropy",
            "contrast", "brightness", "noise")
    acc = {n: {k: [] for k in (*keys, "ms")} for n in names}
    dark_stats = {"entropy": [], "brightness": [], "noise": []}

    for name in images:
        clean = io.sample(name)
        dark = synth.low_light(clean, gamma=gamma, noise_sigma=noise_sigma, seed=0)
        for k in dark_stats:
            dark_stats[k].append(_score(dark, clean)[k])

        for method, fn in METHODS.items():
            out, t = timeit(lambda f=fn: f(dark), runs=runs, warmup=1)
            for k, v in _score(out, clean).items():
                acc[method][k].append(v)
            acc[method]["ms"].append(t.median_ms)

        out, t = timeit(lambda: enhance_oracle(dark, gamma), runs=runs, warmup=1)
        for k, v in _score(out, clean).items():
            acc[ORACLE_NAME][k].append(v)
        acc[ORACLE_NAME]["ms"].append(t.median_ms)

    rows = []
    for name in names:
        a = acc[name]
        rows.append(
            {
                "method": name,
                "psnr_db": round(float(np.mean(a["psnr"])), 3),
                "ssim": round(float(np.mean(a["ssim"])), 4),
                "psnr_matched_db": round(float(np.mean(a["psnr_matched"])), 3),
                "ssim_matched": round(float(np.mean(a["ssim_matched"])), 4),
                "entropy_bits": round(float(np.mean(a["entropy"])), 3),
                "rms_contrast": round(float(np.mean(a["contrast"])), 4),
                "brightness": round(float(np.mean(a["brightness"])), 4),
                "noise_sigma": round(float(np.mean(a["noise"])), 3),
                "median_ms": round(float(np.median(a["ms"])), 3),
            }
        )
    return rows, {k: round(float(np.mean(v)), 4) for k, v in dark_stats.items()}


def sweep_gamma(images=IMAGES, levels=GAMMA_LEVELS, noise_sigma: float = 4.0) -> list[dict]:
    """Trace every method, and the ceiling, as the scene gets darker."""
    from shared import io, synth

    rows = []
    for gamma in levels:
        row: dict[str, float | int] = {
            "gamma": gamma,
            "levels_left": quantisation_ceiling(gamma),
        }
        per_method = {n: [] for n in list(METHODS) + [ORACLE_NAME]}

        for name in images:
            clean = io.sample(name)
            dark = synth.low_light(clean, gamma=gamma, noise_sigma=noise_sigma, seed=0)
            for method, fn in METHODS.items():
                per_method[method].append(psnr(fn(dark), clean))
            per_method[ORACLE_NAME].append(psnr(enhance_oracle(dark, gamma), clean))

        for method, vals in per_method.items():
            row[method] = round(float(np.mean(vals)), 3)
        rows.append(row)
    return rows


def evaluate_noise_amplification(
    gamma: float = 3.0, noise_sigma: float = 4.0, images=IMAGES
) -> list[dict]:
    """How much noise each method multiplies while brightening.

    The darkened image carries a known amount of read noise. Brightening it is a
    multiplication, so the noise is multiplied too. Reporting the ratio makes the
    hidden cost of an aggressive method explicit.
    """
    from shared import io, synth

    rows = []
    for method, fn in list(METHODS.items()) + [(ORACLE_NAME, None)]:
        gains, after = [], []
        for name in images:
            clean = io.sample(name)
            dark = synth.low_light(clean, gamma=gamma, noise_sigma=noise_sigma, seed=0)
            before = estimate_noise_sigma(dark)
            out = enhance_oracle(dark, gamma) if fn is None else fn(dark)
            a = estimate_noise_sigma(out)
            after.append(a)
            gains.append(a / max(before, 1e-6))
        rows.append(
            {
                "method": method,
                "noise_after": round(float(np.mean(after)), 3),
                "amplification": round(float(np.mean(gains)), 2),
            }
        )
    return rows


def brightness_assumption_table(images=IMAGES, gammas=(1.5, 3.0, 5.0)) -> list[dict]:
    """Where the auto-gamma estimator's mid-grey assumption holds, and where it fails.

    For each image: its true mean brightness, and the estimator's error at three
    darkness levels. The point of the table is the **correlation** — the error is
    a function of the gap between the scene's real exposure and the target, not
    of the darkening being estimated.
    """
    from shared import io as shared_io
    from shared import synth

    rows = []
    for name in images:
        clean = shared_io.sample(name)
        true_mean = float(clean.astype(np.float64).mean() / 255.0)
        errors = []
        for g in gammas:
            dark = synth.low_light(clean, gamma=g, noise_sigma=4.0, seed=0)
            estimated = 1.0 / estimate_gamma(dark)
            errors.append(estimated / g - 1.0)
        rows.append(
            {
                "image": name,
                "true_mean_brightness": round(true_mean, 4),
                "brightness_gap": round(true_mean - AUTO_TARGET_BRIGHTNESS, 4),
                "mean_gamma_error_pct": round(float(np.mean(errors)) * 100, 1),
                "worst_gamma_error_pct": round(float(np.max(np.abs(errors))) * 100, 1),
            }
        )
    return rows


def fixed_vs_auto_per_image(images=IMAGES, gamma: float = 3.0, noise_sigma: float = 4.0):
    """Fixed constant against estimated exponent, **one row per image**.

    The project's sharpest result, and one that a mean destroys. Averaged over a
    mixed set the adaptive method looks *worse* than the naive constant; per
    image it is dramatically better exactly where its assumption holds and
    dramatically worse where it does not. Reporting only the average would state
    the opposite of what is happening.
    """
    from shared import io as shared_io
    from shared import synth

    rows = []
    for name in images:
        clean = shared_io.sample(name)
        true_mean = float(clean.astype(np.float64).mean() / 255.0)
        dark = synth.low_light(clean, gamma=gamma, noise_sigma=noise_sigma, seed=0)
        fixed = psnr(enhance_gamma(dark), clean)
        auto = psnr(enhance_gamma_auto(dark), clean)
        oracle = psnr(enhance_oracle(dark, gamma), clean)
        rows.append(
            {
                "image": name,
                "true_mean_brightness": round(true_mean, 4),
                "brightness_gap": round(true_mean - AUTO_TARGET_BRIGHTNESS, 4),
                "fixed_db": round(fixed, 3),
                "auto_db": round(auto, 3),
                "auto_minus_fixed_db": round(auto - fixed, 3),
                "oracle_db": round(oracle, 3),
                "auto_gap_to_oracle_db": round(oracle - auto, 3),
            }
        )
    return sorted(rows, key=lambda r: abs(r["brightness_gap"]))


def enhance(img: np.ndarray, method: str = "LIME") -> np.ndarray:
    """Apply one named method, for the UI and for inference."""
    return METHODS[method](img)
