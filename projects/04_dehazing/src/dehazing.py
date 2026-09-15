"""Dehazing: can you remove a known scattering model?

The question
------------
Project 03 asked whether darkness-hidden detail is recoverable. This asks a
different question, and the distinction matters or the two read as one project
split in half: haze is not a loss of light, it is an **additive veil** with a
known physical model.

    I(x) = J(x) * t(x) + A * (1 - t(x))          t(x) = exp(-beta * d(x))

``I`` is what the camera saw, ``J`` the true scene, ``A`` the airlight, and ``t``
the transmission. Every unknown is on the right. The scene generator builds the
hazy image from this equation, so ``J`` and ``t`` are both known exactly — which
means a method can be scored on **how well it recovered the transmission map**,
not merely on whether the output looks clearer.

That is the whole point: "looks clearer" is satisfied by any contrast stretch.
Recovering ``t`` is only satisfied by actually inverting the scattering.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, rms_contrast, ssim

EPS = 1e-6


# --------------------------------------------------------------------------- #
# transmission estimation — the part that is actually physics
# --------------------------------------------------------------------------- #


def dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    """Minimum over colour channels, then a local minimum filter.

    The dark channel prior (He, Sun & Tang, 2009): in almost any outdoor
    haze-free patch, at least one colour channel has some pixel close to zero.
    Where that is not true, the patch is veiled — so the dark channel is a direct
    read-out of haze thickness.
    """
    f = to_float(img)
    min_channel = f.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_channel, kernel)


def estimate_airlight(img: np.ndarray, patch: int = 15, top_fraction: float = 0.001) -> np.ndarray:
    """Airlight A, taken from the haziest pixels rather than the brightest.

    Picking the single brightest pixel is the usual shortcut and it is wrong: a
    white car or a specular highlight is brighter than the sky and is not
    airlight. Selecting within the top of the *dark channel* restricts the search
    to genuinely veiled regions first.
    """
    f = to_float(img)
    dc = dark_channel(img, patch)
    n = max(1, int(dc.size * top_fraction))
    idx = np.argpartition(dc.ravel(), -n)[-n:]
    candidates = f.reshape(-1, 3)[idx]
    brightest = candidates.sum(axis=1).argmax()
    return candidates[brightest]


def transmission_dcp(
    img: np.ndarray, airlight: np.ndarray, omega: float = 0.95, patch: int = 15
) -> np.ndarray:
    """Transmission from the dark channel prior.

    ``omega`` deliberately keeps a little haze (0.95, not 1.0). Removing all of
    it looks unnatural: distant objects with zero haze read as pasted-on, because
    the eye uses aerial perspective as a depth cue.
    """
    normalised = to_float(img) / np.maximum(airlight, EPS)
    dc = cv2.erode(
        normalised.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    )
    return 1.0 - omega * dc


def refine_transmission_guided(
    img: np.ndarray, t: np.ndarray, radius: int = 40, eps: float = 1e-3
) -> np.ndarray:
    """Guided-filter refinement of a blocky transmission map.

    The patch-wise minimum makes ``t`` piecewise constant, so depth edges land on
    patch boundaries instead of object boundaries and the output shows halos.
    A guided filter pushes the map back onto the image's own edges.

    Implemented directly rather than via ``cv2.ximgproc``, which lives in
    opencv-contrib and is not guaranteed to be installed.
    """
    guide = to_float(to_gray(img))
    t = t.astype(np.float32)
    d = radius * 2 + 1

    mean_i = cv2.blur(guide, (d, d))
    mean_t = cv2.blur(t, (d, d))
    corr_i = cv2.blur(guide * guide, (d, d))
    corr_it = cv2.blur(guide * t, (d, d))

    var_i = corr_i - mean_i * mean_i
    cov_it = corr_it - mean_i * mean_t

    a = cov_it / (var_i + eps)
    b = mean_t - a * mean_i
    return cv2.blur(a, (d, d)) * guide + cv2.blur(b, (d, d))


def recover_scene(
    img: np.ndarray, airlight: np.ndarray, t: np.ndarray, t_min: float = 0.1
) -> np.ndarray:
    """Invert the scattering model: ``J = (I - A) / max(t, t_min) + A``.

    ``t_min`` is not cosmetic. As ``t`` approaches zero the division explodes and
    the deepest haze turns into saturated noise, so the transmission is floored.
    That floor is also the reason no dehazing method fully recovers a distant
    horizon: there, the signal genuinely is almost all airlight.
    """
    f = to_float(img)
    t3 = np.maximum(t, t_min)[..., None]
    return to_uint8((f - airlight) / t3 + airlight)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def dehaze_dcp(img: np.ndarray, patch: int = 15, refine: bool = False) -> np.ndarray:
    """Dark channel prior with a blocky transmission map."""
    a = estimate_airlight(img, patch)
    t = transmission_dcp(img, a, patch=patch)
    if refine:
        t = refine_transmission_guided(img, t)
    return recover_scene(img, a, t)


def dehaze_dcp_refined(img: np.ndarray, patch: int = 15) -> np.ndarray:
    """Dark channel prior with guided-filter refinement — the full method."""
    return dehaze_dcp(img, patch=patch, refine=True)


def dehaze_clahe(img: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    """CLAHE on luminance — a contrast baseline that models nothing.

    Included precisely because it *looks* like dehazing. It raises local
    contrast, which is most of the visible effect, while having no notion of
    transmission at all. If a metric cannot separate this from a physical method,
    the metric is the problem.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(
        ycrcb[..., 0]
    )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def dehaze_retinex(img: np.ndarray, sigmas: tuple[float, ...] = (15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-scale Retinex — an illumination model applied to a scattering problem.

    Retinex assumes image = illumination x reflectance, a *multiplicative* model.
    Haze is *additive*. Retinex therefore cannot represent the degradation, and
    how badly it does is worth measuring rather than assuming.
    """
    f = to_float(img)
    acc = np.zeros_like(f)
    for s in sigmas:
        blur = cv2.GaussianBlur(f, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        acc += np.log(f + EPS) - np.log(blur + EPS)
    acc /= len(sigmas)
    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    return to_uint8(np.clip((acc - lo) / max(hi - lo, EPS), 0, 1))


def dehaze_gamma(img: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """A plain power-law curve — the "did you need any of this?" control."""
    return to_uint8(np.power(to_float(img), gamma))


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Dark channel prior": dehaze_dcp,
    "DCP + guided refine": dehaze_dcp_refined,
    "CLAHE (contrast only)": dehaze_clahe,
    "Multi-scale Retinex": dehaze_retinex,
    "Gamma curve (control)": dehaze_gamma,
}


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #

ORACLE_NAME = "True transmission (oracle)"


def dehaze_oracle(img: np.ndarray, airlight: float, t: np.ndarray) -> np.ndarray:
    """**Oracle**: invert the model using the transmission that was actually used.

    Not a method — it is handed the ground truth. It separates "this method
    estimates transmission badly" from "the scattering model itself cannot be
    inverted here", which the ``t_min`` floor guarantees at some depth.
    """
    return recover_scene(img, np.array([airlight] * 3, np.float32), t)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("rocket", "coffee", "astronaut", "chelsea", "immunohistochemistry", "retina")
BETA_LEVELS = (0.4, 0.8, 1.2, 1.6, 2.2, 3.0)
AIRLIGHT = 0.88


def transmission_error(pred_t: np.ndarray, true_t: np.ndarray) -> float:
    """Mean absolute error of the transmission map, in transmission units."""
    return float(np.abs(pred_t.astype(np.float64) - true_t.astype(np.float64)).mean())


def evaluate_methods(beta: float = 1.4, images=IMAGES, runs: int = 3):
    """Score every method against the true clear image, plus the oracle."""
    from shared import io, synth

    names = list(METHODS) + [ORACLE_NAME]
    acc = {n: {"psnr": [], "ssim": [], "contrast": [], "ms": []} for n in names}
    t_err = {"Dark channel prior": [], "DCP + guided refine": []}
    hazy_stats = {"psnr": [], "contrast": []}

    for name in images:
        clean = io.sample(name)
        hazy, true_t = synth.add_haze(clean, beta=beta, airlight=AIRLIGHT)
        hazy_stats["psnr"].append(psnr(hazy, clean))
        hazy_stats["contrast"].append(rms_contrast(hazy))

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(hazy), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["contrast"].append(rms_contrast(out))
            acc[method]["ms"].append(timing.median_ms)

        # how well did the two physical methods recover the transmission itself?
        a = estimate_airlight(hazy)
        t_blocky = transmission_dcp(hazy, a)
        t_err["Dark channel prior"].append(transmission_error(t_blocky, true_t))
        t_err["DCP + guided refine"].append(
            transmission_error(refine_transmission_guided(hazy, t_blocky), true_t)
        )

        out, timing = timeit(lambda: dehaze_oracle(hazy, AIRLIGHT, true_t), runs=runs, warmup=1)
        acc[ORACLE_NAME]["psnr"].append(psnr(out, clean))
        acc[ORACLE_NAME]["ssim"].append(ssim(out, clean))
        acc[ORACLE_NAME]["contrast"].append(rms_contrast(out))
        acc[ORACLE_NAME]["ms"].append(timing.median_ms)

    rows = []
    for name in names:
        a = acc[name]
        rows.append(
            {
                "method": name,
                "psnr_db": round(float(np.mean(a["psnr"])), 3),
                "ssim": round(float(np.mean(a["ssim"])), 4),
                "rms_contrast": round(float(np.mean(a["contrast"])), 4),
                "transmission_mae": (
                    round(float(np.mean(t_err[name])), 4) if name in t_err else None
                ),
                "median_ms": round(float(np.median(a["ms"])), 3),
            }
        )
    hazy_summary = {k: round(float(np.mean(v)), 4) for k, v in hazy_stats.items()}
    return rows, hazy_summary


def sweep_beta(images=IMAGES, levels=BETA_LEVELS):
    """Trace every method as the haze thickens, against the oracle ceiling."""
    from shared import io, synth

    rows = []
    for beta in levels:
        row: dict[str, float] = {"beta": beta}
        per = {n: [] for n in list(METHODS) + [ORACLE_NAME]}
        for name in images:
            clean = io.sample(name)
            hazy, true_t = synth.add_haze(clean, beta=beta, airlight=AIRLIGHT)
            for method, fn in METHODS.items():
                per[method].append(psnr(fn(hazy), clean))
            per[ORACLE_NAME].append(psnr(dehaze_oracle(hazy, AIRLIGHT, true_t), clean))
        row["min_transmission"] = round(float(np.exp(-beta)), 4)
        for method, vals in per.items():
            row[method] = round(float(np.mean(vals)), 3)
        rows.append(row)
    return rows


def dehaze(img: np.ndarray, method: str = "DCP + guided refine") -> np.ndarray:
    """Apply one named method, for the UI and for inference."""
    return METHODS[method](img)
