"""Template matching: SSD, NCC, ZNCC, and what each one is blind to.

The question
------------
Template matching is the simplest detector there is: slide a patch, find the best
score. The interesting part is that the *scoring function* determines which
changes it survives, and the differences are exact and predictable.

> **The claim under test:** SSD fails the moment brightness changes, NCC survives
> a brightness *scale* but not an *offset*, and ZNCC survives both — because
> subtracting the mean is precisely what removes an additive offset and dividing
> by the standard deviation is precisely what removes a multiplicative one.

That is a mathematical statement, so it should hold to near-machine precision,
and a measurement that disagrees means a bug rather than a nuance.

The template is cropped from the image itself and then transformed by a known
amount, so the true location is known exactly and error is reported in **pixels**
rather than as a detection rate.

Multi-scale is the practical extension: plain matching fails completely if the
target is a different size, and a pyramid search fixes it at a measurable cost.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8

EPS = 1e-9


# --------------------------------------------------------------------------- #
# the scoring methods
# --------------------------------------------------------------------------- #

#: OpenCV's match methods, with whether a *low* score means a better match.
METHODS: dict[str, tuple[int, bool]] = {
    "SSD (SQDIFF)": (cv2.TM_SQDIFF, True),
    "SSD normalised": (cv2.TM_SQDIFF_NORMED, True),
    "Cross-correlation": (cv2.TM_CCORR, False),
    "NCC (CCORR_NORMED)": (cv2.TM_CCORR_NORMED, False),
    "ZNCC (CCOEFF_NORMED)": (cv2.TM_CCOEFF_NORMED, False),
}


def match(image: np.ndarray, template: np.ndarray, method: str = "ZNCC (CCOEFF_NORMED)"):
    """Run one scoring method and return ``(top_left, score, response_map)``.

    The min/max choice is not cosmetic. ``TM_SQDIFF`` is a *distance*, so the
    best match is its **minimum**; the correlation methods are similarities and
    want the maximum. Taking the max of a SQDIFF map finds the worst possible
    location while looking like working code.
    """
    flag, lower_is_better = METHODS[method]
    result = cv2.matchTemplate(to_gray(image), to_gray(template), flag)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if lower_is_better:
        return min_loc, float(min_val), result
    return max_loc, float(max_val), result


def match_multiscale(
    image: np.ndarray, template: np.ndarray, method: str = "ZNCC (CCOEFF_NORMED)",
    scales=np.linspace(0.5, 1.5, 21),
):
    """Search over template scales and keep the best response.

    Plain matching has no scale invariance whatsoever — a target 20% larger than
    the template produces a near-zero response. Rescaling the template and
    retrying is the standard fix, and it multiplies the cost by the number of
    scales tried, which the timing column makes explicit.
    """
    flag, lower_is_better = METHODS[method]
    gray_img = to_gray(image)
    gray_tpl = to_gray(template)

    best = None
    for scale in scales:
        h = int(gray_tpl.shape[0] * scale)
        w = int(gray_tpl.shape[1] * scale)
        if h < 8 or w < 8 or h >= gray_img.shape[0] or w >= gray_img.shape[1]:
            continue
        resized = cv2.resize(gray_tpl, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray_img, resized, flag)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        value, loc = (min_val, min_loc) if lower_is_better else (max_val, max_loc)
        if best is None:
            best = (loc, value, float(scale))
        elif (lower_is_better and value < best[1]) or (not lower_is_better and value > best[1]):
            best = (loc, value, float(scale))
    return best


# --------------------------------------------------------------------------- #
# the test scene
# --------------------------------------------------------------------------- #


def make_case(
    image: str = "astronaut", size: int = 64, brightness_gain: float = 1.0,
    brightness_offset: float = 0.0, noise_sigma: float = 0.0, scale: float = 1.0,
    rotation: float = 0.0, seed: int = 0,
):
    """Crop a template, then degrade the *search image* by a known amount.

    Returns ``(search_image, template, true_top_left)``. Degrading the search
    image rather than the template is the realistic direction: the template is
    what you stored, the scene is what the camera gave you today.
    """
    from shared import synth

    rng = np.random.default_rng(seed)
    base = load_scene(image)
    h, w = base.shape[:2]

    x = int(rng.integers(size, w - 2 * size))
    y = int(rng.integers(size, h - 2 * size))
    template = base[y : y + size, x : x + size].copy()

    scene = to_float(base)
    if brightness_gain != 1.0 or brightness_offset != 0.0:
        scene = scene * brightness_gain + brightness_offset
    scene = to_uint8(scene)

    if scale != 1.0 or rotation != 0.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, scale)
        scene = cv2.warpAffine(scene, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        centre = np.array([x + size / 2, y + size / 2, 1.0])
        moved = m @ centre
        x = int(moved[0] - size * scale / 2)
        y = int(moved[1] - size * scale / 2)

    if noise_sigma > 0:
        scene = synth.gaussian_noise(scene, sigma=noise_sigma, seed=seed)

    return scene, template, (x, y)


def localisation_error(found, truth) -> float:
    return float(np.hypot(found[0] - truth[0], found[1] - truth[1]))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs selected by `tools/select_images.py --axis entropy`, which
#: is the axis that decides whether template matching has a chance. A template
#: cut from a flat region is ambiguous no matter which scoring function reads it,
#: and one cut from a busy region is not — so a pool at one end of this axis
#: would measure the pictures rather than the methods.
#:
#: The spread is 5.2 to 7.9 bits. Two of the twelve are in the pool for a
#: specific reason rather than for their entropy: `cricket_on_the_green` has
#: eleven near-identical white figures on a plain field, and
#: `boy_with_a_fish_trap` is most of a frame of woven lattice. Both create
#: genuine repeated structure, which is the failure mode a single best-peak
#: search cannot report.
IMAGES = (
    "hippo_in_green_water",    # entropy 5.2 - the flattest scene here
    "porcupine_on_a_branch",   #         6.7
    "leopard_in_bare_tree",    #         6.9
    "cricket_on_the_green",    #         7.1 - eleven near-identical figures
    "weaver_at_her_loom",      #         7.2
    "bugling_elk",             #         7.3
    "elephants_and_grooms",    #         7.4
    "boy_in_a_wide_hat",       #         7.4
    "boy_with_a_fish_trap",    #         7.5 - a repetitive woven lattice
    "two_at_a_railing",        #         7.6
    "street_piper",            #         7.7
    "monks_at_a_noticeboard",  #         7.9 - a grid of near-identical notices
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs.

    Named so that `run.py`, the tests and `infer.py` all read the same pixels —
    the template is cropped from these, so a different loader would move the
    ground truth.
    """
    from shared import io

    return io.real_photo(name)


def entropy(img: np.ndarray) -> float:
    """Shannon entropy of the grey histogram, in bits.

    The axis the pool was selected on, recomputed here so the README's numbers
    come from the project rather than from the selection tool.
    """
    hist = np.bincount(to_gray(img).ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
#: Brightness *scale*. Pushed to a tenth rather than the usual gentle 0.9, because
#: NCC's scale invariance is exact and nothing below 10x shows it: at gain 0.1 NCC
#: still localises every template and SSD-normalised localises none.
GAINS = (1.0, 0.5, 0.3, 0.1, 2.0, 3.0)

#: Brightness *offset*, and the negative half is the half that matters. NCC is
#: not offset invariant, but a *positive* offset hides that — it clips at white
#: and destroys the structure for every method equally, so NCC and ZNCC fail
#: together. Darkening clips only in the shadows, and there NCC falls to 0.458
#: while ZNCC holds 1.000.
OFFSETS = (0.0, -0.1, -0.2, -0.3, -0.4, 0.2, 0.4)
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0, 50.0)
SCALES = (1.0, 0.9, 0.8, 1.1, 1.25)
ROTATIONS = (0.0, 2.0, 5.0, 10.0, 20.0)
SUCCESS_PX = 5.0


def evaluate_methods(images=IMAGES, seeds=(0, 1, 2), runs: int = 3, **degradation):
    """Score every method under one degradation condition."""
    acc = {n: {"err": [], "hit": [], "ms": []} for n in METHODS}
    for image in images:
        for seed in seeds:
            scene, template, truth = make_case(image, seed=seed, **degradation)
            for name in METHODS:
                (loc, _, _), timing = timeit(
                    lambda n=name, s=scene, t=template: match(s, t, n), runs=runs, warmup=1
                )
                err = localisation_error(loc, truth)
                acc[name]["err"].append(err)
                acc[name]["hit"].append(err <= SUCCESS_PX)
                acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": n,
            "mean_error_px": round(float(np.mean(a["err"])), 3),
            "median_error_px": round(float(np.median(a["err"])), 3),
            "success_rate": round(float(np.mean(a["hit"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for n, a in acc.items()
    ]


def sweep_brightness_gain(images=IMAGES, gains=GAINS, seeds=(0, 1)):
    """A multiplicative change. NCC and ZNCC divide by the norm, so both survive."""
    rows = []
    for gain in gains:
        scored = evaluate_methods(images=images, seeds=seeds, runs=1, brightness_gain=gain)
        row: dict[str, float] = {"gain": gain}
        for r in scored:
            row[r["method"]] = r["success_rate"]
        rows.append(row)
    return rows


def sweep_brightness_offset(images=IMAGES, offsets=OFFSETS, seeds=(0, 1)):
    """An additive change — the one that separates NCC from ZNCC.

    Only ZNCC subtracts the mean, so only ZNCC should be unaffected. This is the
    cleanest demonstration in the project of a formula's consequence showing up
    directly in a measurement.
    """
    rows = []
    for offset in offsets:
        scored = evaluate_methods(images=images, seeds=seeds, runs=1, brightness_offset=offset)
        row: dict[str, float] = {"offset": offset}
        for r in scored:
            row[r["method"]] = r["success_rate"]
        rows.append(row)
    return rows


def sweep_noise(images=IMAGES, levels=NOISE_LEVELS, seeds=(0, 1)):
    rows = []
    for sigma in levels:
        scored = evaluate_methods(images=images, seeds=seeds, runs=1, noise_sigma=sigma)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["method"]] = r["success_rate"]
        rows.append(row)
    return rows


def sweep_scale(images=IMAGES, scales=SCALES, seeds=(0, 1)):
    """Scale is where single-scale matching simply stops working."""
    rows = []
    for s in scales:
        scored = evaluate_methods(images=images, seeds=seeds, runs=1, scale=s)
        row: dict[str, float] = {"scale": s}
        for r in scored:
            row[r["method"]] = r["success_rate"]
        rows.append(row)
    return rows


def sweep_rotation(images=IMAGES, rotations=ROTATIONS, seeds=(0, 1)):
    """No method here is rotation invariant — the question is how fast each dies."""
    rows = []
    for rot in rotations:
        scored = evaluate_methods(images=images, seeds=seeds, runs=1, rotation=rot)
        row: dict[str, float] = {"rotation_deg": rot}
        for r in scored:
            row[r["method"]] = r["success_rate"]
        rows.append(row)
    return rows


def multiscale_benefit(images=IMAGES, scales=SCALES, seeds=(0, 1), runs: int = 1):
    """What a pyramid search buys, and what it costs."""
    rows = []
    for s in scales:
        single_hits, multi_hits, single_ms, multi_ms, scale_err = [], [], [], [], []
        for image in images:
            for seed in seeds:
                scene, template, truth = make_case(image, scale=s, seed=seed)
                (loc, _, _), t1 = timeit(
                    lambda a=scene, b=template: match(a, b), runs=runs, warmup=0
                )
                single_hits.append(localisation_error(loc, truth) <= SUCCESS_PX)
                single_ms.append(t1.median_ms)

                best, t2 = timeit(
                    lambda a=scene, b=template: match_multiscale(a, b), runs=runs, warmup=0
                )
                multi_ms.append(t2.median_ms)
                if best is not None:
                    multi_hits.append(localisation_error(best[0], truth) <= SUCCESS_PX)
                    scale_err.append(abs(best[2] - s))
                else:
                    multi_hits.append(False)

        rows.append(
            {
                "true_scale": s,
                "single_scale_success": round(float(np.mean(single_hits)), 4),
                "multi_scale_success": round(float(np.mean(multi_hits)), 4),
                "scale_estimate_error": round(float(np.mean(scale_err)), 4) if scale_err else None,
                "single_ms": round(float(np.median(single_ms)), 2),
                "multi_ms": round(float(np.median(multi_ms)), 2),
            }
        )
    return rows


def response_sharpness(images=IMAGES, seeds=(0,)):
    """How *peaked* each method's response surface is.

    A method can find the right location and still have a nearly flat response,
    which means it is one noisy pixel away from being wrong. The ratio of the
    peak to the surrounding mean is a confidence measure that the raw match score
    does not provide.
    """
    rows = []
    for name in METHODS:
        ratios = []
        for image in images:
            for seed in seeds:
                scene, template, _ = make_case(image, seed=seed)
                _, best, result = match(scene, template, name)
                flag, lower_is_better = METHODS[name]
                r = result.astype(np.float64)
                if lower_is_better:
                    r = r.max() - r
                    best = r.max()
                mean = float(r.mean())
                ratios.append(float(best) / max(abs(mean), EPS))
        rows.append(
            {
                "method": name,
                "peak_to_mean": round(float(np.mean(ratios)), 4),
            }
        )
    return rows


def verify_formula_invariance():
    """Check the algebra directly, with no image search involved.

    ZNCC of a patch against ``a*patch + b`` must be 1.0 for any positive ``a``
    and any ``b``. If it is not, the implementation is wrong — no experiment
    needed.
    """
    rng = np.random.default_rng(0)
    patch = to_uint8(rng.random((32, 32)).astype(np.float32))
    rows = []
    for gain, offset in ((1.0, 0.0), (0.6, 0.0), (1.0, 0.2), (0.6, 0.2)):
        transformed = to_uint8(to_float(patch) * gain + offset)
        row: dict[str, float | str] = {"gain": gain, "offset": offset}
        for name, (flag, lower_is_better) in METHODS.items():
            score = float(cv2.matchTemplate(transformed, patch, flag)[0, 0])
            row[name] = round(score, 6)
        rows.append(row)
    return rows
