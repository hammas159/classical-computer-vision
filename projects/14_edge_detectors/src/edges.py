"""Edge detectors: six operators, scored against an exact edge map.

The question
------------
Sobel, Prewitt, Roberts, Scharr, LoG and Canny all "find edges". Comparing them
by eye is how they are usually compared, and it settles nothing.

> **The claim under test:** with a pixel tolerance applied — and it must be
> applied — the differences between the first-derivative operators are far
> smaller than their reputations suggest, and almost all of Canny's advantage
> comes from two post-processing steps that could be bolted onto any of them.

Two measurement decisions carry the whole project:

* **A pixel tolerance is mandatory.** Edge operators legitimately place a
  boundary one pixel to either side of the nominal truth. Scoring at zero
  tolerance measures sub-pixel luck and gives every detector a near-zero F1.
* **Every operator gets the same threshold sweep.** Comparing a tuned Canny
  against an untuned Sobel measures the tuning, not the operator, so each is
  scored at *its own best threshold* as well as at a common default.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray
from shared.metrics import edge_prf, pratt_fom

EPS = 1e-6

# --------------------------------------------------------------------------- #
# the operators — each returns a float gradient magnitude, not a binary map
# --------------------------------------------------------------------------- #

#: Roberts cross kernels: the original 1963 operator, a 2x2 diagonal difference.
#: It is the smallest possible derivative estimate and the most noise-sensitive,
#: which is why it lost to 3x3 operators and is here as the historical floor.
ROBERTS_X = np.array([[1, 0], [0, -1]], np.float32)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], np.float32)

#: Prewitt: a 3x3 difference with a *box* smoothing column. Sobel is the same
#: shape with a [1 2 1] column instead, which is the entire difference between
#: them — Sobel weights the centre row more, giving slightly better rotational
#: symmetry.
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)


def _magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    m = np.sqrt(gx * gx + gy * gy)
    return m / max(float(m.max()), EPS)


def grad_sobel(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    g = to_float(gray)
    return _magnitude(
        cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize),
        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize),
    )


def grad_prewitt(gray: np.ndarray) -> np.ndarray:
    g = to_float(gray)
    return _magnitude(
        cv2.filter2D(g, cv2.CV_32F, PREWITT_X), cv2.filter2D(g, cv2.CV_32F, PREWITT_Y)
    )


def grad_roberts(gray: np.ndarray) -> np.ndarray:
    g = to_float(gray)
    return _magnitude(
        cv2.filter2D(g, cv2.CV_32F, ROBERTS_X), cv2.filter2D(g, cv2.CV_32F, ROBERTS_Y)
    )


def grad_scharr(gray: np.ndarray) -> np.ndarray:
    """Scharr: a 3x3 kernel optimised for rotational symmetry.

    Sobel's 3x3 kernel has a measurable angular error — it responds differently
    to a diagonal edge than to a horizontal one. Scharr's weights ([3 10 3]
    rather than [1 2 1]) minimise that error, so it should win on the diagonal
    edges in the test scene specifically.
    """
    g = to_float(gray)
    return _magnitude(cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1))


def grad_log(gray: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    """Laplacian of Gaussian — a *second* derivative, so edges are zero crossings.

    Returned here as |LoG| so it can share the thresholding machinery, which is
    slightly unfair to it: the principled version detects sign changes, not
    magnitude peaks. That caveat belongs in the results, not hidden in the code.
    """
    g = cv2.GaussianBlur(to_float(gray), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    return lap / max(float(lap.max()), EPS)


GRADIENTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Roberts": grad_roberts,
    "Prewitt": grad_prewitt,
    "Sobel": grad_sobel,
    "Scharr": grad_scharr,
    "LoG": grad_log,
}


# --------------------------------------------------------------------------- #
# binarisation, and Canny
# --------------------------------------------------------------------------- #


def threshold_magnitude(mag: np.ndarray, t: float) -> np.ndarray:
    """Plain threshold on a normalised gradient magnitude."""
    return (mag >= t).astype(np.uint8) * 255


def edges_canny(gray: np.ndarray, low: int = 50, high: int = 150, blur: float = 1.4) -> np.ndarray:
    """Canny: Sobel plus non-maximum suppression plus hysteresis.

    The gradient underneath is an ordinary Sobel. Everything that makes Canny
    better is post-processing:

    * **non-maximum suppression** thins a ridge several pixels wide to one pixel;
    * **hysteresis** keeps a weak edge only if it connects to a strong one, which
      is what suppresses isolated noise responses without also deleting genuine
      faint edges.

    Because the gradient is shared, any advantage Canny shows over Sobel here is
    attributable to those two steps and nothing else.
    """
    g = cv2.GaussianBlur(gray, (0, 0), blur, borderType=cv2.BORDER_REFLECT)
    return cv2.Canny(g, low, high)  # Canny requires 8-bit input


def edges_canny_auto(gray: np.ndarray, sigma_frac: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the image median — the common recipe."""
    v = float(np.median(gray))
    low = int(max(0, (1.0 - sigma_frac) * v))
    high = int(min(255, (1.0 + sigma_frac) * v))
    return edges_canny(gray, low, high)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

TOLERANCE = 2
THRESHOLDS = tuple(np.round(np.linspace(0.02, 0.5, 25), 4))
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0, 50.0)


def scene(size: int = 512, noise_sigma: float = 0.0, seed: int = 0):
    """Synthetic shapes plus their exact edge map, optionally noised."""
    from shared import synth

    img, truth = synth.shapes(size=size, seed=seed)
    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return img, truth


def best_threshold(
    gray: np.ndarray, truth: np.ndarray, grad_fn, thresholds=THRESHOLDS, tolerance: int = TOLERANCE
):
    """Sweep the threshold and return the best (f1, threshold, edge map).

    Every operator is given this same courtesy, which is the only way the
    comparison measures operators rather than tuning.
    """
    mag = grad_fn(gray)
    best = (-1.0, thresholds[0], None)
    for t in thresholds:
        e = threshold_magnitude(mag, float(t))
        f1 = edge_prf(e, truth, tolerance)["f1"]
        if f1 > best[0]:
            best = (f1, float(t), e)
    return best


def evaluate_operators(noise_sigma: float = 0.0, seeds=(0, 1, 2), runs: int = 3):
    """Score every operator at its own best threshold, plus both Canny variants."""
    acc = {n: {"f1": [], "p": [], "r": [], "fom": [], "t": [], "ms": []}
           for n in list(GRADIENTS) + ["Canny (fixed 50/150)", "Canny (auto median)"]}

    for seed in seeds:
        img, truth = scene(noise_sigma=noise_sigma, seed=seed)
        gray = to_gray(img)

        for name, fn in GRADIENTS.items():
            (_, _), timing = timeit(lambda f=fn: (f(gray), None), runs=runs, warmup=1)
            f1, t, e = best_threshold(gray, truth, fn)
            prf = edge_prf(e, truth, TOLERANCE)
            acc[name]["f1"].append(prf["f1"])
            acc[name]["p"].append(prf["precision"])
            acc[name]["r"].append(prf["recall"])
            acc[name]["fom"].append(pratt_fom(e, truth))
            acc[name]["t"].append(t)
            acc[name]["ms"].append(timing.median_ms)

        for label, fn in (
            ("Canny (fixed 50/150)", lambda g: edges_canny(g)),
            ("Canny (auto median)", lambda g: edges_canny_auto(g)),
        ):
            e, timing = timeit(lambda f=fn: f(gray), runs=runs, warmup=1)
            prf = edge_prf(e, truth, TOLERANCE)
            acc[label]["f1"].append(prf["f1"])
            acc[label]["p"].append(prf["precision"])
            acc[label]["r"].append(prf["recall"])
            acc[label]["fom"].append(pratt_fom(e, truth))
            acc[label]["t"].append(float("nan"))
            acc[label]["ms"].append(timing.median_ms)

    return [
        {
            "operator": name,
            "f1": round(float(np.mean(a["f1"])), 4),
            "precision": round(float(np.mean(a["p"])), 4),
            "recall": round(float(np.mean(a["r"])), 4),
            "pratt_fom": round(float(np.mean(a["fom"])), 4),
            "best_threshold": (
                None if np.isnan(np.mean(a["t"])) else round(float(np.mean(a["t"])), 4)
            ),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for name, a in acc.items()
    ]


def sweep_tolerance(tolerances=(0, 1, 2, 3, 5), seeds=(0,)):
    """Why the tolerance is not optional.

    At zero tolerance every operator scores near zero, which says nothing about
    any of them. Showing the curve makes the choice defensible instead of
    arbitrary.
    """
    rows = []
    for tol in tolerances:
        row: dict[str, float | int] = {"tolerance_px": tol}
        for name, fn in GRADIENTS.items():
            scores = []
            for seed in seeds:
                img, truth = scene(seed=seed)
                gray = to_gray(img)
                f1, _, _ = best_threshold(gray, truth, fn, tolerance=tol)
                scores.append(f1)
            row[name] = round(float(np.mean(scores)), 4)
        canny = []
        for seed in seeds:
            img, truth = scene(seed=seed)
            canny.append(edge_prf(edges_canny(to_gray(img)), truth, tol)["f1"])
        row["Canny (fixed 50/150)"] = round(float(np.mean(canny)), 4)
        rows.append(row)
    return rows


def sweep_noise(levels=NOISE_LEVELS, seeds=(0, 1)):
    """Robustness: Roberts is a 2x2 kernel and should collapse first."""
    rows = []
    for sigma in levels:
        scored = evaluate_operators(noise_sigma=sigma, seeds=seeds, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["operator"]] = r["f1"]
        rows.append(row)
    return rows


def detect(img: np.ndarray, operator: str = "Canny (fixed 50/150)", threshold: float = 0.1):
    """Run one named operator, for the UI and for inference."""
    gray = to_gray(img)
    if operator.startswith("Canny (auto"):
        return edges_canny_auto(gray)
    if operator.startswith("Canny"):
        return edges_canny(gray)
    return threshold_magnitude(GRADIENTS[operator](gray), threshold)
