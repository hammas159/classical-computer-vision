"""White balance: four algorithms, one known colour cast, exact ground truth.

The question
------------
White balance corrects a colour cast caused by the scene's illuminant. Because
the cast here is *applied* — known per-channel gains — the correct answer is
known exactly, and the interesting measurement is not "does the image look
better" but:

> **How accurately does each method recover the illuminant itself?**

That is measured as **angular error in degrees** between the estimated and true
illuminant vectors, which is the standard metric in the colour-constancy
literature and is far more informative than image PSNR: it separates *estimating
the light* from *applying the correction*.

> **The claim under test:** grey-world assumes the scene averages to grey, so it
> fails exactly when that assumption fails — a scene dominated by one colour.
> White-patch assumes something in the scene is white, so it fails when nothing
> is, and is destroyed by a single blown highlight. Neither failure is a bug;
> both are the assumption showing through, and both can be triggered deliberately.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_uint8

EPS = 1e-9


# --------------------------------------------------------------------------- #
# illuminant estimation — each returns an RGB gain vector
# --------------------------------------------------------------------------- #


def est_grey_world(img: np.ndarray) -> np.ndarray:
    """Grey-world: assume the average of the scene is achromatic.

    Estimates the illuminant as the per-channel mean. Cheap, assumption-heavy,
    and the failure is sharp rather than gradual: a scene that genuinely is
    mostly red produces an estimate that says the light was red.
    """
    f = to_float(img)
    return f.reshape(-1, 3).mean(axis=0)


def est_white_patch(img: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """White-patch (max-RGB): assume the brightest response per channel is white.

    Uses a high percentile rather than the true maximum. The raw max is decided by
    a handful of pixels, so a single specular highlight or a hot pixel sets the
    entire white balance — which is worth demonstrating by *also* offering the
    100th percentile.
    """
    f = to_float(img)
    return np.percentile(f.reshape(-1, 3), percentile, axis=0)


def est_shades_of_grey(img: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Shades-of-grey: the Minkowski p-norm, which contains both of the above.

    ``p = 1`` is exactly grey-world (the mean); ``p -> infinity`` is exactly
    white-patch (the max). Intermediate values interpolate, and ``p = 6`` is the
    value the literature settles on. Having one parameter span both extremes is
    what makes the sweep in this project meaningful.
    """
    f = to_float(img).reshape(-1, 3).astype(np.float64)
    return np.power(np.mean(np.power(f, p), axis=0), 1.0 / p)


def est_grey_edge(img: np.ndarray, p: float = 6.0, sigma: float = 1.0) -> np.ndarray:
    """Grey-edge: apply the grey assumption to **derivatives**, not intensities.

    The insight is that the *average edge* in a scene is achromatic, which is a
    much weaker assumption than the average pixel being achromatic. It survives a
    large uniform coloured region — precisely the case that breaks grey-world.
    """
    f = to_float(img)
    blurred = cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    grads = []
    for c in range(3):
        gx = cv2.Sobel(blurred[..., c], cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred[..., c], cv2.CV_64F, 0, 1, ksize=3)
        grads.append(np.abs(gx) + np.abs(gy))
    g = np.stack(grads, axis=-1).reshape(-1, 3)
    return np.power(np.mean(np.power(g, p), axis=0), 1.0 / p)


def est_none(img: np.ndarray) -> np.ndarray:
    """Do nothing — the control, as a neutral illuminant."""
    return np.ones(3, np.float64) / np.sqrt(3)


ESTIMATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Do nothing (control)": est_none,
    "Grey-world": est_grey_world,
    "White-patch (99th pct)": est_white_patch,
    "White-patch (true max)": lambda a: est_white_patch(a, percentile=100.0),
    "Shades-of-grey (p=6)": est_shades_of_grey,
    "Grey-edge (p=6)": est_grey_edge,
}


# --------------------------------------------------------------------------- #
# correction and scoring
# --------------------------------------------------------------------------- #


def normalise_illuminant(v: np.ndarray) -> np.ndarray:
    """Unit-length illuminant vector.

    Only the *direction* is recoverable, never the magnitude: a scene lit twice
    as brightly by the same light is indistinguishable from the same scene at
    twice the exposure. Normalising makes that explicit, and it is why the metric
    is an angle and not a distance.
    """
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), EPS)


def apply_correction(img: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """Divide out the estimated illuminant and restore the overall brightness."""
    e = normalise_illuminant(illuminant)
    gains = float(np.mean(e)) / np.maximum(e, EPS)
    return to_uint8(to_float(img) * gains.astype(np.float32))


def angular_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Angle in degrees between two illuminant directions.

    The standard colour-constancy metric. Under 3 degrees is generally considered
    imperceptible; over 10 is an obviously wrong image.
    """
    a, b = normalise_illuminant(estimate), normalise_illuminant(truth)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# scenes, including the ones designed to break each assumption
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry")
CASTS = {
    "tungsten (warm)": (1.35, 1.0, 0.65),
    "daylight (neutral)": (1.02, 1.0, 0.98),
    "shade (cool)": (0.75, 1.0, 1.30),
    "strong green": (0.80, 1.35, 0.80),
}
P_VALUES = (1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0)


def make_case(image: str, gains=(1.25, 1.0, 0.75), noise_sigma: float = 0.0, seed: int = 0):
    """Apply a known colour cast. Returns ``(cast_image, clean, true_illuminant)``."""
    from shared import io, synth

    clean = io.sample(image)
    cast = synth.colour_cast(clean, gains=tuple(float(g) for g in gains))
    if noise_sigma > 0:
        cast = synth.gaussian_noise(cast, sigma=noise_sigma, seed=seed)
    return cast, clean, normalise_illuminant(np.asarray(gains, np.float64))


def dominant_colour_scene(size: int = 384, fraction: float = 0.7, seed: int = 0):
    """A scene dominated by one colour — grey-world's designed failure case.

    ``fraction`` of the frame is a single saturated colour. Grey-world must
    interpret that as a coloured light and "correct" it away; grey-edge should
    not, because a large flat region contributes almost no edges.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    img[:] = (0.45, 0.45, 0.45)

    n = int(size * size * fraction)
    side = int(np.sqrt(n))
    img[:side, :side] = (0.80, 0.18, 0.18)

    for _ in range(14):
        x, y = int(rng.integers(0, size - 40)), int(rng.integers(0, size - 40))
        colour = rng.random(3).astype(np.float32)
        cv2.rectangle(img, (x, y), (x + 36, y + 36), colour.tolist(), -1)
    return to_uint8(img)


def blown_highlight_scene(size: int = 384, seed: int = 0):
    """A scene with one saturated highlight — white-patch's failure case.

    A clipped specular highlight is not white, it is *clipped*, and its channel
    ratios carry no information about the illuminant. A method that trusts the
    brightest pixel trusts exactly that.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for _ in range(40):
        x, y = int(rng.integers(0, size - 50)), int(rng.integers(0, size - 50))
        cv2.rectangle(img, (x, y), (x + 48, y + 48), rng.random(3).tolist(), -1)
    cv2.circle(img, (size // 2, size // 2), 14, (1.0, 1.0, 1.0), -1)
    return to_uint8(img)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_estimators(cast: str = "tungsten (warm)", images=IMAGES, seeds=(0,), runs: int = 3):
    """Angular error and corrected-image quality for every estimator."""
    from shared.metrics import psnr, ssim

    gains = CASTS[cast]
    acc = {n: {"angle": [], "psnr": [], "ssim": [], "ms": []} for n in ESTIMATORS}

    for image in images:
        for seed in seeds:
            cast_img, clean, truth = make_case(image, gains=gains, seed=seed)
            for name, fn in ESTIMATORS.items():
                estimate, timing = timeit(lambda f=fn, a=cast_img: f(a), runs=runs, warmup=1)
                corrected = apply_correction(cast_img, estimate)
                acc[name]["angle"].append(angular_error(estimate, truth))
                acc[name]["psnr"].append(psnr(corrected, clean))
                acc[name]["ssim"].append(ssim(corrected, clean))
                acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": n,
            "angular_error_deg": round(float(np.mean(a["angle"])), 3),
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for n, a in acc.items()
    ]


def compare_casts(images=IMAGES, casts=CASTS):
    """Does any method handle every illuminant, or do they trade?"""
    rows = []
    for cast in casts:
        scored = evaluate_estimators(cast=cast, images=images, runs=1)
        row: dict[str, float | str] = {"cast": cast}
        for r in scored:
            row[r["method"]] = r["angular_error_deg"]
        rows.append(row)
    return rows


def sweep_minkowski_p(images=IMAGES, ps=P_VALUES, cast: str = "tungsten (warm)"):
    """The parameter that spans grey-world and white-patch.

    At ``p = 1`` shades-of-grey *is* grey-world; as ``p`` grows it becomes
    white-patch. If both endpoints are worse than the middle, then the optimum is
    a genuine blend rather than either assumption being right.
    """
    gains = CASTS[cast]
    rows = []
    for p in ps:
        errors, edge_errors = [], []
        for image in images:
            cast_img, _, truth = make_case(image, gains=gains)
            errors.append(angular_error(est_shades_of_grey(cast_img, p=p), truth))
            edge_errors.append(angular_error(est_grey_edge(cast_img, p=p), truth))
        rows.append(
            {
                "p": p,
                "shades_of_grey_deg": round(float(np.mean(errors)), 3),
                "grey_edge_deg": round(float(np.mean(edge_errors)), 3),
            }
        )
    return rows


def assumption_failure_tests(seeds=(0, 1, 2)):
    """Trigger each method's designed failure mode deliberately.

    Both scenes are lit **neutrally** — there is no colour cast at all, so the
    correct answer is "the light is white" and any angular error is the method
    hallucinating an illuminant from the scene's own content.
    """
    neutral = normalise_illuminant(np.ones(3))
    rows = []
    for scene_name, builder in (
        ("Dominant colour (breaks grey-world)", dominant_colour_scene),
        ("Blown highlight (breaks white-patch)", blown_highlight_scene),
    ):
        row: dict[str, float | str] = {"scene": scene_name}
        for name, fn in ESTIMATORS.items():
            errors = [angular_error(fn(builder(seed=s)), neutral) for s in seeds]
            row[name] = round(float(np.mean(errors)), 3)
        rows.append(row)
    return rows


def sweep_dominance(fractions=(0.0, 0.2, 0.4, 0.6, 0.8), seeds=(0, 1)):
    """How much single-colour dominance grey-world tolerates before it breaks.

    Grey-edge should stay flat across this sweep while grey-world climbs, because
    a large flat region adds pixels but almost no edges.
    """
    neutral = normalise_illuminant(np.ones(3))
    rows = []
    for frac in fractions:
        row: dict[str, float] = {"dominant_fraction": frac}
        for name, fn in ESTIMATORS.items():
            errors = [
                angular_error(fn(dominant_colour_scene(fraction=frac, seed=s)), neutral)
                for s in seeds
            ]
            row[name] = round(float(np.mean(errors)), 3)
        rows.append(row)
    return rows


def sweep_noise(images=IMAGES, levels=(0.0, 5.0, 15.0, 30.0), cast: str = "tungsten (warm)"):
    """White-patch depends on extreme values, so noise should hurt it most."""
    gains = CASTS[cast]
    rows = []
    for sigma in levels:
        row: dict[str, float] = {"noise_sigma": sigma}
        for name, fn in ESTIMATORS.items():
            errors = []
            for image in images:
                cast_img, _, truth = make_case(image, gains=gains, noise_sigma=sigma)
                errors.append(angular_error(fn(cast_img), truth))
            row[name] = round(float(np.mean(errors)), 3)
        rows.append(row)
    return rows


def balance(img: np.ndarray, method: str = "Grey-edge (p=6)"):
    """Estimate and correct in one call, for the UI."""
    estimate = ESTIMATORS[method](img)
    return apply_correction(img, estimate), normalise_illuminant(estimate)
