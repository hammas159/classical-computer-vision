"""Canny parameter sensitivity: which knob actually matters?

The question
------------
Canny has three parameters — the smoothing sigma and the two hysteresis
thresholds. Everyone tunes the thresholds. Almost nobody touches sigma.

> **The claim under test:** F1 moves more with **sigma** than with either
> threshold. If true, the universal tuning habit is aimed at the wrong knob.

This is a sensitivity study, not a method comparison, so the design is different
from the other projects: one method, a full grid over its parameters, and the
output is a *response surface* rather than a ranking.

Two things make the answer trustworthy:

* **A pixel tolerance**, without which every configuration scores near zero and
  the surface is noise.
* **Variance decomposition** — for each parameter, how much of the total spread
  in F1 it accounts for when the others are marginalised out. That turns "sigma
  matters more" from an impression about a plot into a number.
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import edge_prf, pratt_fom

EPS = 1e-9

#: Canny takes 8-bit input. Passing a float image silently produces garbage or
#: raises, depending on the OpenCV build -- one of the few genuinely
#: version-dependent traps in the library.
TOLERANCE = 2

SIGMAS = (0.0, 0.5, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0)
LOW_THRESHOLDS = (10, 25, 50, 75, 100, 150)
RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0)
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0, 50.0)


def canny(gray: np.ndarray, sigma: float, low: int, ratio: float) -> np.ndarray:
    """Canny with explicit smoothing, and the high threshold set by a ratio.

    Parameterising as ``(low, ratio)`` rather than ``(low, high)`` is deliberate:
    the two thresholds are strongly coupled — a high threshold below the low one
    is meaningless — and sweeping them independently spends most of the grid on
    invalid combinations. The 1:2 to 1:3 ratio convention comes from Canny's own
    paper.
    """
    work = gray
    if sigma > 0:
        work = cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    high = int(np.clip(low * ratio, low + 1, 255))
    return cv2.Canny(work, int(low), high)


#: Twelve photographs with **human** boundary annotations, spanning edge
#: density 1.6% to 36.7%. Canny is an edge detector and BSDS is the edge
#: benchmark, so unlike most of this repository the real-image half of this
#: project has a target a person actually drew. Selected by
#: `tools/select_images.py --axis edges`.
IMAGES = (
    "acacia_and_herd",      # edges  1.6 — one silhouette and open sky
    "clownfish_anemone",    # edges  9.0
    "alpine_church",        # edges 11.6
    "three_astronauts",     # edges 13.2
    "man_floral_shirt",     # edges 15.7
    "hawk_on_stump",        # edges 17.4
    "baboon_in_foliage",    # edges 19.1
    "deer_and_fawn",        # edges 20.5
    "firefighter_debris",   # edges 22.8
    "blossom_pavilion",     # edges 25.4
    "deer_bare_branches",   # edges 29.7
    "bears_on_hillside",    # edges 36.7 — the busiest frame in the pool
)


def load_scene(name: str):
    from shared import io

    return io.real_photo(name)


def photo_grid(images=None, sigmas=(1.0, 1.4, 2.0), lows=(25, 50, 100),
               ratios=(2.0, 3.0), min_annotators: int = 2):
    """Canny's parameter grid scored against **human** boundaries.

    The synthetic half of this project has exact geometry and an unambiguous
    answer. This half does not, and the difference is the point: on a generated
    scene of shapes there is one correct edge map, and on a photograph five
    people draw five different ones.

    Scored with the same boundary F-measure project 24 uses, against what at
    least `min_annotators` of them agreed on.
    """
    from shared import bsds

    images = images if images is not None else IMAGES
    rows = []
    for sigma in sigmas:
        for low in lows:
            for ratio in ratios:
                p, r, f = [], [], []
                for name in images:
                    gray = to_gray(load_scene(name))
                    target = bsds.consensus_boundaries(name, min_annotators)
                    score = bsds.boundary_f_measure(
                        canny(gray, sigma, low, ratio), target, TOLERANCE)
                    p.append(score["precision"])
                    r.append(score["recall"])
                    f.append(score["f"])
                rows.append({
                    "sigma": sigma, "low": low, "ratio": ratio,
                    "precision": round(float(np.mean(p)), 4),
                    "recall": round(float(np.mean(r)), 4),
                    "f": round(float(np.mean(f)), 4),
                })
    return rows


def photo_human_ceiling(images=None, min_annotators: int = 2):
    """How well one annotator reproduces what the others agreed on.

    The ceiling for the photograph half, measured rather than assumed. Canny has
    no colour, no semantics and no idea what an object is, so the gap between it
    and this number is not a tuning problem.
    """
    from shared import bsds

    images = images if images is not None else IMAGES
    best, mean = [], []
    for name in images:
        ann = bsds.load_annotations(name)
        target = bsds.consensus_boundaries(name, min_annotators)
        scores = [bsds.boundary_f_measure(a["boundaries"], target, TOLERANCE)["f"]
                  for a in ann]
        best.append(max(scores))
        mean.append(float(np.mean(scores)))
    return {"best": round(float(np.mean(best)), 4),
            "mean": round(float(np.mean(mean)), 4)}


def scene(size: int = 512, noise_sigma: float = 0.0, seed: int = 0):
    from shared import synth

    img, truth = synth.shapes(size=size, seed=seed)
    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return to_gray(img), truth


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #


def full_grid(noise_sigma: float = 15.0, seeds=(0, 1), tolerance: int = TOLERANCE):
    """Every (sigma, low, ratio) combination, scored on F1 and Pratt's FOM.

    Returns one row per configuration. This is the raw material for both the
    response surface and the variance decomposition.
    """
    rows = []
    for sigma in SIGMAS:
        for low in LOW_THRESHOLDS:
            for ratio in RATIOS:
                f1s, foms, precisions, recalls = [], [], [], []
                for seed in seeds:
                    gray, truth = scene(noise_sigma=noise_sigma, seed=seed)
                    edges = canny(gray, sigma, low, ratio)
                    prf = edge_prf(edges, truth, tolerance)
                    f1s.append(prf["f1"])
                    precisions.append(prf["precision"])
                    recalls.append(prf["recall"])
                    foms.append(pratt_fom(edges, truth))
                rows.append(
                    {
                        "sigma": sigma,
                        "low": low,
                        "ratio": ratio,
                        "high": int(np.clip(low * ratio, low + 1, 255)),
                        "f1": round(float(np.mean(f1s)), 4),
                        "precision": round(float(np.mean(precisions)), 4),
                        "recall": round(float(np.mean(recalls)), 4),
                        "pratt_fom": round(float(np.mean(foms)), 4),
                    }
                )
    return rows


def variance_decomposition(rows: list[dict], metric: str = "f1"):
    """How much of the spread in ``metric`` each parameter accounts for.

    For each parameter: group the grid by that parameter's value, take the mean
    within each group, and measure the variance *between* group means. A
    parameter that barely changes the outcome produces group means that are all
    alike, and therefore a small between-group variance.

    Reported as a fraction of the total, so the three numbers are comparable and
    answer the project's question directly.
    """
    values = np.array([r[metric] for r in rows], float)
    total_var = float(np.var(values))
    out = {}
    for param in ("sigma", "low", "ratio"):
        levels = sorted({r[param] for r in rows})
        group_means = [
            float(np.mean([r[metric] for r in rows if r[param] == lv])) for lv in levels
        ]
        out[param] = {
            "between_group_variance": round(float(np.var(group_means)), 8),
            "fraction_of_total": round(float(np.var(group_means) / max(total_var, EPS)), 4),
            "best_value": levels[int(np.argmax(group_means))],
            "range": round(float(max(group_means) - min(group_means)), 4),
        }
    out["_total_variance"] = round(total_var, 8)
    return out


def marginal_response(rows: list[dict], param: str, metric: str = "f1"):
    """Mean and best ``metric`` at each level of one parameter.

    The mean shows how much that knob moves the *typical* result; the best shows
    what is achievable if the other knobs are tuned for it. They can disagree,
    and when they do it means the parameter interacts rather than acting alone.
    """
    levels = sorted({r[param] for r in rows})
    return [
        {
            param: lv,
            "mean_" + metric: round(float(np.mean([r[metric] for r in rows if r[param] == lv])), 4),
            "best_" + metric: round(float(np.max([r[metric] for r in rows if r[param] == lv])), 4),
        }
        for lv in levels
    ]


def best_configuration(rows: list[dict], metric: str = "f1"):
    return max(rows, key=lambda r: r[metric])


def sweep_noise(levels=NOISE_LEVELS, seeds=(0, 1)):
    """Does the best sigma track the noise level?

    It should — smoothing exists to suppress noise, so the optimum should climb
    as noise rises. If it does, that is a second, independent confirmation that
    sigma is the parameter carrying the load.
    """
    rows = []
    for noise in levels:
        grid = full_grid(noise_sigma=noise, seeds=seeds)
        best = best_configuration(grid)
        decomposition = variance_decomposition(grid)
        rows.append(
            {
                "noise_sigma": noise,
                "best_f1": best["f1"],
                "best_sigma": best["sigma"],
                "best_low": best["low"],
                "best_ratio": best["ratio"],
                "sigma_variance_share": decomposition["sigma"]["fraction_of_total"],
                "low_variance_share": decomposition["low"]["fraction_of_total"],
                "ratio_variance_share": decomposition["ratio"]["fraction_of_total"],
            }
        )
    return rows


def sweep_tolerance(tolerances=(0, 1, 2, 3, 5), noise_sigma: float = 15.0, seeds=(0,)):
    """Why the tolerance is not an arbitrary choice.

    At zero tolerance the best achievable F1 is so low that the whole surface is
    noise, and any conclusion drawn from it would be about sub-pixel luck.
    """
    rows = []
    for tol in tolerances:
        grid = full_grid(noise_sigma=noise_sigma, seeds=seeds, tolerance=tol)
        best = best_configuration(grid)
        rows.append(
            {
                "tolerance_px": tol,
                "best_f1": best["f1"],
                "mean_f1": round(float(np.mean([r["f1"] for r in grid])), 4),
                "best_sigma": best["sigma"],
            }
        )
    return rows


def precision_recall_tradeoff(sigma: float = 1.4, seeds=(0,), noise_sigma: float = 15.0):
    """The thresholds trade precision against recall rather than improving both.

    If that is what they do, then "tuning the thresholds" is choosing an
    operating point, not raising quality — which is a different activity from
    what sigma does, and explains why sigma dominates the variance.
    """
    rows = []
    for low in LOW_THRESHOLDS:
        for ratio in RATIOS:
            p, r, f = [], [], []
            for seed in seeds:
                gray, truth = scene(noise_sigma=noise_sigma, seed=seed)
                prf = edge_prf(canny(gray, sigma, low, ratio), truth, TOLERANCE)
                p.append(prf["precision"])
                r.append(prf["recall"])
                f.append(prf["f1"])
            rows.append(
                {
                    "low": low,
                    "ratio": ratio,
                    "precision": round(float(np.mean(p)), 4),
                    "recall": round(float(np.mean(r)), 4),
                    "f1": round(float(np.mean(f)), 4),
                }
            )
    return rows


def timing(seeds=(0,), runs: int = 5):
    """Does sigma cost anything? Smoothing is a separable convolution, so barely."""
    rows = []
    for sigma in SIGMAS:
        ms = []
        for seed in seeds:
            gray, _ = scene(seed=seed)
            _, t = timeit(lambda g=gray, s=sigma: canny(g, s, 50, 3.0), runs=runs, warmup=1)
            ms.append(t.median_ms)
        rows.append({"sigma": sigma, "median_ms": round(float(np.median(ms)), 3)})
    return rows


def response_surface(rows: list[dict], ratio: float = 3.0, metric: str = "f1"):
    """A sigma x low-threshold grid at one ratio, as a 2-D array for plotting."""
    sigmas = sorted({r["sigma"] for r in rows})
    lows = sorted({r["low"] for r in rows})
    surface = np.full((len(sigmas), len(lows)), np.nan)
    for r in rows:
        if r["ratio"] != ratio:
            continue
        surface[sigmas.index(r["sigma"]), lows.index(r["low"])] = r[metric]
    return surface, sigmas, lows
