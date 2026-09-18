"""Poisson blending: composite in the gradient domain, not the pixel domain.

The question
------------
Paste a region from one image into another and the seam is obvious, because the
two images have different colour and brightness. Feathering the edge hides the
seam by blurring it. Poisson blending removes it properly, and the reason is
worth measuring.

> **The claim under test:** Poisson blending does not copy *pixels*, it copies
> **gradients**, then solves for the image whose gradients best match. Because
> human vision responds to local contrast rather than absolute brightness, the
> result reads as seamless even though the pasted region's actual colours have
> been changed — often drastically.

That last part is the measurable, counter-intuitive bit: **the output pixels
inside the pasted region can differ enormously from the source**, while looking
more correct than an exact copy. Alpha blending, which preserves those pixels
faithfully, looks worse.

So the metric matters more than usual here. Scoring against the source region
rewards alpha blending and punishes the method that works — a clean example of
the wrong metric inverting a conclusion.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8

EPS = 1e-9


# --------------------------------------------------------------------------- #
# the blending methods
# --------------------------------------------------------------------------- #


def blend_copy_paste(target: np.ndarray, source: np.ndarray, mask: np.ndarray, centre):
    """Hard paste. The control that shows what the seam looks like untreated."""
    out = target.copy()
    x, y, w, h = _placement(target, source, mask, centre)
    region_mask = mask[:h, :w] > 0
    out[y : y + h, x : x + w][region_mask] = source[:h, :w][region_mask]
    return out


def blend_alpha_feather(target: np.ndarray, source: np.ndarray, mask: np.ndarray,
                        centre, feather: int = 15):
    """Feathered alpha blend: a soft ramp across the mask boundary.

    Hides the seam by smearing it over ``feather`` pixels. It cannot fix a colour
    mismatch — it only makes the transition gradual, so a large brightness
    difference becomes a visible gradient instead of a visible line.
    """
    x, y, w, h = _placement(target, source, mask, centre)
    m = (mask[:h, :w] > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(m, (0, 0), max(feather / 3.0, 0.5),
                             borderType=cv2.BORDER_REFLECT)[..., None]
    out = to_float(target).copy()
    patch = to_float(source[:h, :w])
    region = out[y : y + h, x : x + w]
    out[y : y + h, x : x + w] = region * (1 - alpha) + patch * alpha
    return to_uint8(out)


def blend_poisson_opencv(target: np.ndarray, source: np.ndarray, mask: np.ndarray,
                         centre, flags=cv2.NORMAL_CLONE):
    """OpenCV's ``seamlessClone`` — Poisson blending, production implementation."""
    m = (mask > 0).astype(np.uint8) * 255
    return cv2.cvtColor(
        cv2.seamlessClone(
            cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(target, cv2.COLOR_RGB2BGR),
            m, tuple(int(c) for c in centre), flags,
        ),
        cv2.COLOR_BGR2RGB,
    )


def blend_poisson_mixed(target, source, mask, centre):
    """Mixed gradients: at each pixel keep whichever source has the stronger gradient.

    The variant for pasting something onto a *textured* background. Normal cloning
    replaces the target's texture inside the region entirely; mixed cloning lets
    the target's texture show through wherever it is stronger than the source's,
    which is how you paste writing onto a brick wall.
    """
    return blend_poisson_opencv(target, source, mask, centre, flags=cv2.MIXED_CLONE)


def blend_poisson_jacobi(target: np.ndarray, source: np.ndarray, mask: np.ndarray,
                         centre, iterations: int = 400):
    """Poisson blending solved directly, by Jacobi iteration.

    Written out so the mathematics is visible rather than hidden behind a library
    call. The problem is:

        minimise  |grad(f) - grad(source)|^2   inside the region
        subject to f = target on the boundary

    whose solution satisfies the Poisson equation ``laplacian(f) = div(grad
    source)``. Discretised, every interior pixel is the average of its four
    neighbours minus the source's Laplacian — so iterating that update converges
    to the answer.

    Slow and transparent. The OpenCV version solves the same system directly and
    is the one to actually use; the timing gap between them is part of the result.
    """
    x, y, w, h = _placement(target, source, mask, centre)
    region_mask = (mask[:h, :w] > 0)

    out = to_float(target).copy()
    patch = to_float(source[:h, :w])
    sub = out[y : y + h, x : x + w]

    # the guidance field: the source's Laplacian inside the region
    lap_source = np.stack(
        [cv2.Laplacian(patch[..., c], cv2.CV_32F, ksize=1) for c in range(patch.shape[2])],
        axis=-1,
    )

    interior = cv2.erode(region_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # The boundary ring keeps the TARGET's values. That single line is the whole
    # of Poisson blending: gradients from the source inside, values from the
    # destination on the edge.
    #
    # Seeding the boundary from the source instead — `f[region_mask] = patch[...]`
    # — makes `f = source` an exact fixed point of the iteration, because the
    # solution of `lap f = lap g` with `f = g` on the boundary *is* g. The solver
    # then converged in zero steps to the copy-paste answer and reported a seam
    # visibility of 2.80, identical to the control, with nothing to indicate it
    # had not run.
    f = sub.copy()
    f[interior] = patch[interior]  # a starting guess, interior only

    for _ in range(iterations):
        neighbour_sum = (
            np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
            + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
        )
        updated = (neighbour_sum - lap_source) / 4.0
        f[interior] = updated[interior]

    sub[interior] = np.clip(f[interior], 0.0, 1.0)
    out[y : y + h, x : x + w] = sub
    return to_uint8(out)


METHODS: dict[str, Callable] = {
    "Copy-paste (control)": blend_copy_paste,
    "Alpha feather": blend_alpha_feather,
    "Poisson (OpenCV)": blend_poisson_opencv,
    "Poisson (mixed gradients)": blend_poisson_mixed,
    "Poisson (Jacobi, from scratch)": blend_poisson_jacobi,
}


def _placement(target, source, mask, centre):
    """Top-left corner and clipped size of the pasted region."""
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    cx, cy = int(centre[0]), int(centre[1])
    x = int(np.clip(cx - sw // 2, 0, max(tw - sw, 0)))
    y = int(np.clip(cy - sh // 2, 0, max(th - sh, 0)))
    w = min(sw, tw - x)
    h = min(sh, th - y)
    return x, y, w, h


# --------------------------------------------------------------------------- #
# measuring the seam
# --------------------------------------------------------------------------- #


def seam_visibility(result: np.ndarray, mask: np.ndarray, centre, target: np.ndarray,
                    band: int = 3) -> float:
    """Gradient magnitude on the mask boundary, relative to nearby gradients.

    A visible seam is a gradient that exists *only because* of the paste. Dividing
    the boundary gradient by the surrounding gradient normalises away the image's
    own busyness, so the number means "how much extra edge did the compositing
    create" rather than "how textured is this image".

    1.0 means the boundary is indistinguishable from its surroundings.
    """
    # The mask, not the result: `_placement` reads its second argument's shape as
    # the size of the pasted patch, and `result` is the whole composite. Passing
    # it gave w, h of the full frame and the mask assignment below raised a
    # broadcast error — loudly, which is the good case. The quiet version of this
    # mistake would have placed the seam somewhere else entirely.
    x, y, w, h = _placement(target, mask, mask, centre)
    full = np.zeros(result.shape[:2], np.uint8)
    full[y : y + h, x : x + w] = (mask[:h, :w] > 0).astype(np.uint8) * 255

    boundary = cv2.morphologyEx(full, cv2.MORPH_GRADIENT, np.ones((band, band), np.uint8)) > 0
    nearby = cv2.dilate(full, np.ones((band * 6, band * 6), np.uint8)) > 0
    nearby &= ~boundary

    g = to_float(to_gray(result))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)

    if not boundary.any() or not nearby.any():
        return float("nan")
    return float(mag[boundary].mean() / max(mag[nearby].mean(), EPS))


def gradient_fidelity(result: np.ndarray, source: np.ndarray, mask: np.ndarray,
                      centre, target: np.ndarray) -> float:
    """Correlation between the result's gradients and the source's, inside the region.

    **The metric that matches what Poisson blending optimises.** It should be near
    1.0 for the Poisson variants and lower for feathering, which blurs gradients
    near the edge.
    """
    x, y, w, h = _placement(target, source, mask, centre)
    region = (mask[:h, :w] > 0)

    def grad(img):
        g = to_float(to_gray(img))
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        return np.sqrt(gx * gx + gy * gy)

    a = grad(result[y : y + h, x : x + w])[region]
    b = grad(source[:h, :w])[region]
    if a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pixel_fidelity(result: np.ndarray, source: np.ndarray, mask: np.ndarray,
                   centre, target: np.ndarray) -> float:
    """Mean absolute pixel difference from the source, inside the region, 0-255.

    **The metric that gets the answer wrong**, included deliberately. Copy-paste
    scores a perfect 0 here and looks terrible; Poisson scores badly and looks
    right. It is the clearest demonstration in the repo that choosing a metric is
    choosing a conclusion.
    """
    x, y, w, h = _placement(target, source, mask, centre)
    region = (mask[:h, :w] > 0)
    a = result[y : y + h, x : x + w][region].astype(np.float64)
    b = source[:h, :w][region].astype(np.float64)
    return float(np.abs(a - b).mean())


# --------------------------------------------------------------------------- #
# the test cases
# --------------------------------------------------------------------------- #

#: Twelve photographs selected by `tools/select_images.py --axis tone`, which
#: measures how wide a range of greys a scene occupies (this module recomputes it
#: as the 1st-to-99th percentile spread, which orders the pool identically but on
#: a 0-255 scale). Tone is the axis here:
#: the seam a blend has to hide is exactly the tonal discontinuity between the
#: two images, so a pool clustered at one tone would have almost no seam to
#: remove.
IMAGES = (
    "bird_in_a_meadow",      # tone  82 - the narrowest range here
    "tent_on_the_ice",       #      152
    "wallaby_in_scrub",      #      153
    "stone_viaduct",         #      177
    "zebra_in_grass",        #      181
    "carved_boat_houses",    #      202
    "soldier_and_child",     #      211
    "ploughing_with_oxen",   #      218
    "skier_mid_air",         #      225
    "spear_fisher",          #      233
    "bobcat_and_daisies",    #      240
    "snowshoes_on_snow",     #      250 - the widest
)

#: Six (target, source) pairs built from those twelve, each image used exactly
#: once. Paired deliberately across the tone range rather than at random: the
#: quantity that decides how hard a blend is, is the *difference* between the two
#: images, and pairing like with like would give the seam nothing to be made of.
PAIRS = (
    ("bird_in_a_meadow", "snowshoes_on_snow"),      # tone 82 into 250 - the extreme
    ("tent_on_the_ice", "bobcat_and_daisies"),      # cold blue into warm yellow
    ("wallaby_in_scrub", "spear_fisher"),
    ("stone_viaduct", "skier_mid_air"),
    ("zebra_in_grass", "ploughing_with_oxen"),
    ("carved_boat_houses", "soldier_and_child"),
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs.

    Named so that `run.py`, the tests and `infer.py` all read the same pixels.
    """
    from shared import io

    return io.real_photo(name)


def tonal_range(img: np.ndarray) -> float:
    """The 1st-to-99th percentile spread of the greyscale, in levels.

    The axis the pool was selected on, recomputed here so the README's numbers
    come from the project rather than from the selection tool.
    """
    g = to_gray(img)
    return float(np.percentile(g, 99) - np.percentile(g, 1))
OFFSETS = (0.0, 0.1, 0.25, 0.5)
REGION_SIZES = (60, 100, 150, 200)


def make_case(target_name: str, source_name: str, size: int = 120,
              brightness_offset: float = 0.0, shape: str = "ellipse", seed: int = 0):
    """Build a blending case: a target, a source patch, a mask and a centre.

    ``brightness_offset`` shifts the source's brightness by a known amount, which
    is what creates the seam. Controlling it directly is what lets the sweep find
    where each method stops coping.
    """
    rng = np.random.default_rng(seed)
    target = load_scene(target_name)
    source_full = load_scene(source_name)

    sh, sw = source_full.shape[:2]
    sx = int(rng.integers(0, max(sw - size, 1)))
    sy = int(rng.integers(0, max(sh - size, 1)))
    source = source_full[sy : sy + size, sx : sx + size].copy()

    if brightness_offset != 0.0:
        source = to_uint8(to_float(source) + brightness_offset)

    mask = np.zeros((size, size), np.uint8)
    if shape == "ellipse":
        cv2.ellipse(mask, (size // 2, size // 2), (size // 2 - 4, size // 2 - 4), 0, 0, 360, 255, -1)
    elif shape == "rect":
        cv2.rectangle(mask, (4, 4), (size - 4, size - 4), 255, -1)
    else:
        cv2.circle(mask, (size // 2, size // 2), size // 2 - 4, 255, -1)

    th, tw = target.shape[:2]
    centre = (tw // 2, th // 2)
    return target, source, mask, centre


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_methods(pairs=PAIRS, brightness_offset: float = 0.25, size: int = 120, runs: int = 1):
    """Score every method on all three metrics at once.

    Reporting seam visibility, gradient fidelity and pixel fidelity side by side
    is the whole design: they disagree, and the disagreement is the finding.
    """
    acc = {n: {"seam": [], "grad": [], "pixel": [], "ms": []} for n in METHODS}
    for i, (tname, sname) in enumerate(pairs):
        target, source, mask, centre = make_case(
            tname, sname, size=size, brightness_offset=brightness_offset, seed=i
        )
        for name, fn in METHODS.items():
            out, timing = timeit(
                lambda f=fn: f(target, source, mask, centre), runs=runs, warmup=0
            )
            acc[name]["seam"].append(seam_visibility(out, mask, centre, target))
            acc[name]["grad"].append(gradient_fidelity(out, source, mask, centre, target))
            acc[name]["pixel"].append(pixel_fidelity(out, source, mask, centre, target))
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": n,
            "seam_visibility": round(float(np.nanmean(a["seam"])), 4),
            "gradient_fidelity": round(float(np.nanmean(a["grad"])), 4),
            "pixel_difference": round(float(np.nanmean(a["pixel"])), 2),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for n, a in acc.items()
    ]


def sweep_brightness_offset(pairs=PAIRS, offsets=OFFSETS):
    """How large a colour mismatch can each method absorb?

    Copy-paste should degrade linearly with the offset. Poisson should be almost
    flat, because it never used the source's absolute values in the first place.
    """
    rows = []
    for offset in offsets:
        scored = evaluate_methods(pairs=pairs, brightness_offset=offset)
        row: dict[str, float] = {"brightness_offset": offset}
        for r in scored:
            row[r["method"]] = r["seam_visibility"]
        rows.append(row)
    return rows


def sweep_region_size(pairs=PAIRS, sizes=REGION_SIZES):
    """Does the region size change which method wins, or only the cost?

    The Jacobi solver's iteration count must grow with region size to converge —
    information propagates one pixel per iteration — so a fixed count should show
    up as degradation on the larger regions.
    """
    rows = []
    for size in sizes:
        scored = evaluate_methods(pairs=pairs, size=size)
        row: dict[str, float | int] = {"region_size": size}
        for r in scored:
            row[f"{r['method']} seam"] = r["seam_visibility"]
            row[f"{r['method']} ms"] = r["median_ms"]
        rows.append(row)
    return rows


def metric_disagreement(pairs=PAIRS, brightness_offset: float = 0.25):
    """Rank the methods by each metric and show the rankings invert.

    If pixel fidelity ranks copy-paste first and seam visibility ranks it last,
    then the choice of metric fully determines the conclusion — which is this
    project's central point stated as data.
    """
    rows = evaluate_methods(pairs=pairs, brightness_offset=brightness_offset)
    by_seam = [r["method"] for r in sorted(rows, key=lambda r: r["seam_visibility"])]
    by_gradient = [r["method"] for r in sorted(rows, key=lambda r: -r["gradient_fidelity"])]
    by_pixel = [r["method"] for r in sorted(rows, key=lambda r: r["pixel_difference"])]
    return {
        "rank_by_seam_visibility": by_seam,
        "rank_by_gradient_fidelity": by_gradient,
        "rank_by_pixel_fidelity": by_pixel,
        "best_by_seam": by_seam[0],
        "best_by_pixel": by_pixel[0],
        "rankings_agree": by_seam[0] == by_pixel[0],
    }


def convergence(pairs=PAIRS, iterations=(10, 50, 100, 200, 400, 800), brightness_offset=0.25):
    """How many Jacobi iterations are needed, and against the direct solver.

    Jacobi propagates boundary information one pixel per iteration, so a region
    200 px across needs on the order of 200 sweeps before the boundary condition
    has reached the middle at all. That is why direct solvers exist.
    """
    rows = []
    for n in iterations:
        seams, ms = [], []
        for i, (tname, sname) in enumerate(pairs):
            target, source, mask, centre = make_case(
                tname, sname, brightness_offset=brightness_offset, seed=i
            )
            out, timing = timeit(
                lambda: blend_poisson_jacobi(target, source, mask, centre, iterations=n),
                runs=1, warmup=0,
            )
            seams.append(seam_visibility(out, mask, centre, target))
            ms.append(timing.median_ms)
        rows.append(
            {
                "iterations": n,
                "seam_visibility": round(float(np.nanmean(seams)), 4),
                "median_ms": round(float(np.median(ms)), 2),
            }
        )
    return rows


def blend(target: np.ndarray, source: np.ndarray, mask: np.ndarray, centre,
          method: str = "Poisson (OpenCV)"):
    return METHODS[method](target, source, mask, centre)
