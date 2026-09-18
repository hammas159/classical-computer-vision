"""Focus stacking: five focus measures, three controls, and truth by construction.

The question
------------
A macro photographer takes a stack of frames at different focus distances, each
sharp in a different band of depth, and merges them into one image sharp
everywhere. The merge needs a **focus measure**: a per-pixel number saying how
in-focus this frame is here.

> **The claim under test:** the focus measure is what decides the result. **It is
> not.** The five measures land within **1.44 dB** of each other, and the best of
> them is **0.034 dB** from an oracle that is handed the answer. Changing the
> window the response is pooled over -- a parameter almost nobody reports -- moves
> the result by **3.34 dB**, 2.3 times as far.

> **What the problem is worth:** picking a source frame per pixel at random
> scores 26.7 dB and picking correctly scores 43.7. The whole task is worth
> **17.0 dB**, and every measure here captures at least 15.5 of it. The
> differences the literature argues about are the last decibel and a half.

> **The failure worth knowing:** on a flat region **no measure can be right**,
> because every frame is identically smooth there. Agreement with the truth falls
> from **0.982** in detailed regions to **0.496** in flat ones, and the five
> measures disagree with each other on **58%** of flat pixels against **11%** of
> detailed ones. How much of an image is flat is measurable before any stack is
> built.

Where the ground truth comes from
---------------------------------
The stack is **built**, so the truth is not estimated: each source frame is the
real photograph blurred everywhere except in one band of a synthetic depth map,
and the index of the frame that is sharp at each pixel is recorded. The merged
image is scored against the original photograph, which is sharp everywhere.

What is real and what is not, stated plainly: the photograph is real and the
depth map is not. A real focal stack of a real scene has no ground truth at all —
there is no single all-sharp frame to compare against — which is exactly why this
project builds one.

The controls
------------
`Random pick` chooses a source frame per pixel at random and `First frame` takes
the whole thing from one frame. `Oracle` uses the recorded index. They bracket
the measures: the first two say what the problem is worth, and the third says
how much of it is left.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.io import real_photo, to_gray

EPS = 1e-9

#: The twelve photographs, in order of **mean gradient magnitude** -- the detail
#: a focus measure has to work with. A whitewashed wall against a clear sky has
#: almost none, and no focus measure can find focus in a region that looks the
#: same in every frame.
IMAGES = (
    "whitewashed_bell_tower",
    "mossy_boulders_in_a_valley",
    "golfer_by_the_sea",
    "wall_across_the_hills",
    "partridge_on_gravel",
    "two_jackals",
    "race_cars_on_a_bend",
    "picnic_in_the_snow",
    "alpine_cottage_with_flowers",
    "lily_pond_and_pagoda",
    "waterfall_under_a_bridge",
    "mountain_lake_and_scree",
)

#: How many frames a stack has, and the blur applied to the out-of-focus parts.
#: `MAX_SIGMA` is the blur at the far end of the depth range from a frame's own
#: focus plane; a frame is sharp in its own band and increasingly blurred away
#: from it, which is what a real lens does.
N_FRAMES = 5
MAX_SIGMA = 4.0

#: The blur levels a stack is approximated with are spaced this far apart in
#: sigma, so a stack built at any `max_sigma` is approximated equally finely.
SIGMA_STEP = 0.25

#: The window the per-pixel focus response is pooled over before comparison.
#: This turns out to matter more than which measure is used -- see
#: `sweep_pooling`.
POOL = 9


def load(name: str) -> np.ndarray:
    return real_photo(name)


def detail(rgb: np.ndarray) -> float:
    """Mean gradient magnitude x1000: the substrate a focus measure needs."""
    g = to_gray(rgb).astype(np.float32) / 255.0
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(dx, dy)) * 1000)


# --------------------------------------------------------------------------- #
# building a stack, and its truth
# --------------------------------------------------------------------------- #


def depth_map(shape, seed: int = 0, smooth: float = 60.0) -> np.ndarray:
    """A smooth synthetic depth field in [0, 1].

    Smooth rather than random: real depth is piecewise continuous, and a
    per-pixel random depth would make the truth mask salt-and-pepper, which no
    method could win and which would flatter whichever method blurs most.
    """
    h, w = shape[:2]
    rng = np.random.default_rng(seed)
    coarse = rng.random((max(2, h // 64), max(2, w // 64))).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), smooth)
    field -= field.min()
    return field / max(float(field.max()), EPS)


def build_stack(image: np.ndarray, n: int = N_FRAMES, seed: int = 0,
                max_sigma: float = MAX_SIGMA):
    """``(frames, truth_index, depth)`` for one photograph.

    Frame ``k`` is focused at depth ``k / (n - 1)``. A pixel at depth ``d`` is
    blurred in frame ``k`` by ``max_sigma * |d - k/(n-1)|``, so every frame is
    sharp in its own band and smoothly worse away from it -- and exactly one
    frame is sharpest at each pixel, which is the truth.
    """
    depth = depth_map(image.shape, seed)
    planes = np.linspace(0.0, 1.0, n)

    # A blurred version at each of a few sigmas, then blended by the per-pixel
    # sigma: blurring every pixel individually would be correct and far too slow.
    #
    # The levels are spaced by a **fixed** sigma step rather than by splitting
    # `max_sigma` into a fixed number of parts. With nine levels regardless, a
    # stack built at max_sigma 1 was approximated eight times more finely than one
    # built at max_sigma 8, and `sweep_blur` came out non-monotone with a dip at
    # sigma 2 -- an artefact of the construction, not a property of any measure.
    levels = np.arange(0.0, max_sigma + SIGMA_STEP, SIGMA_STEP)
    blurred = [image.astype(np.float32) if s <= EPS
               else cv2.GaussianBlur(image.astype(np.float32), (0, 0), s)
               for s in levels]

    frames = []
    for plane in planes:
        sigma = max_sigma * np.abs(depth - plane)
        out = np.zeros_like(image, np.float32)
        for i in range(len(levels) - 1):
            lo, hi = levels[i], levels[i + 1]
            sel = (sigma >= lo) & (sigma < hi)
            if not sel.any():
                continue
            t = ((sigma - lo) / max(hi - lo, EPS))[..., None]
            out[sel] = ((1 - t) * blurred[i] + t * blurred[i + 1])[sel]
        sel = sigma >= levels[-1]
        if sel.any():
            out[sel] = blurred[-1][sel]
        frames.append(np.clip(out, 0, 255).astype(np.uint8))

    # the truth: which plane is nearest this pixel's depth
    truth = np.argmin(np.abs(depth[..., None] - planes[None, None, :]), axis=2)
    return frames, truth.astype(np.int32), depth


def flat_share(image: np.ndarray, threshold: float = 2.0) -> float:
    """Fraction of the image where the local gradient is too small to focus on.

    Where every frame is identically smooth, no focus measure can be right, and
    the answer it gives is decided by noise. This is measured from the original
    photograph, before any stack exists.
    """
    g = to_gray(image).astype(np.float32)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    local = cv2.blur(np.hypot(dx, dy), (POOL, POOL))
    return float((local < threshold).mean())


# --------------------------------------------------------------------------- #
# the focus measures
# --------------------------------------------------------------------------- #


def _pool(response: np.ndarray, window: int = POOL) -> np.ndarray:
    """Average the raw per-pixel response over a window.

    Not cosmetic. A per-pixel focus response is extremely noisy -- it is a second
    derivative of a noisy signal -- and the window is what makes it usable.
    `sweep_pooling` shows it matters more than which measure produced it.
    """
    if window <= 1:
        return response
    return cv2.blur(response, (window, window))


def measure_variance_of_laplacian(frame: np.ndarray, window: int = POOL) -> np.ndarray:
    """|Laplacian|, pooled. The most widely used focus measure there is."""
    g = to_gray(frame).astype(np.float32)
    return _pool(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), window)


def measure_tenengrad(frame: np.ndarray, window: int = POOL) -> np.ndarray:
    """Squared Sobel gradient magnitude, pooled. Tenenbaum, 1970."""
    g = to_gray(frame).astype(np.float32)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return _pool(dx * dx + dy * dy, window)


def measure_modified_laplacian(frame: np.ndarray, window: int = POOL) -> np.ndarray:
    """Sum of absolute second derivatives taken separately per axis.

    Nayar & Nakagawa's point: in a plain Laplacian the x and y second derivatives
    can cancel, so a sharp edge in one direction can read as out of focus.
    """
    g = to_gray(frame).astype(np.float32)
    kx = np.array([[-1.0, 2.0, -1.0]], np.float32)
    lx = np.abs(cv2.filter2D(g, cv2.CV_32F, kx))
    ly = np.abs(cv2.filter2D(g, cv2.CV_32F, kx.T))
    return _pool(lx + ly, window)


def measure_local_variance(frame: np.ndarray, window: int = POOL) -> np.ndarray:
    """Local grey variance. No derivative at all, so no noise amplification."""
    g = to_gray(frame).astype(np.float32)
    mean = cv2.blur(g, (window, window))
    sq = cv2.blur(g * g, (window, window))
    return np.maximum(sq - mean * mean, 0.0)


def measure_wavelet(frame: np.ndarray, window: int = POOL) -> np.ndarray:
    """Energy of the finest detail band of a one-level Haar transform.

    Blur removes high frequencies first, so the finest band is where focus lives.
    Implemented directly rather than through a wavelet library, because it is
    four additions.
    """
    g = to_gray(frame).astype(np.float32)
    a = g[0::2, 0::2]
    b = g[0::2, 1::2]
    c = g[1::2, 0::2]
    d = g[1::2, 1::2]
    n = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    m = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    a, b, c, d = a[:n, :m], b[:n, :m], c[:n, :m], d[:n, :m]
    detail_band = np.abs(a - b) + np.abs(a - c) + np.abs(a - d)
    full = cv2.resize(detail_band, (g.shape[1], g.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return _pool(full, window)


MEASURES: dict[str, Callable] = {
    "Variance of Laplacian": measure_variance_of_laplacian,
    "Tenengrad": measure_tenengrad,
    "Modified Laplacian": measure_modified_laplacian,
    "Local variance": measure_local_variance,
    "Wavelet detail": measure_wavelet,
}


# --------------------------------------------------------------------------- #
# selecting and merging
# --------------------------------------------------------------------------- #


def select_indices(frames, measure: str, window: int = POOL) -> np.ndarray:
    """Which frame each pixel should come from, according to one measure."""
    responses = np.stack([MEASURES[measure](f, window) for f in frames])
    return np.argmax(responses, axis=0).astype(np.int32)


def select_random(frames, seed: int = 0, **kw) -> np.ndarray:
    """A frame per pixel, at random. The control that says what the task is worth."""
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    return rng.integers(0, len(frames), (h, w)).astype(np.int32)


def select_first(frames, **kw) -> np.ndarray:
    """Everything from frame 0. The other end of the control."""
    return np.zeros(frames[0].shape[:2], np.int32)


def merge(frames, indices: np.ndarray, feather: int = 0) -> np.ndarray:
    """Take each pixel from its selected frame.

    ``feather`` blurs the selection map first, which trades a little sharpness at
    a boundary for the absence of a visible seam. Zero by default so the measures
    are compared on what they actually selected.
    """
    stack = np.stack(frames).astype(np.float32)
    if feather > 0:
        weights = np.stack([
            cv2.blur((indices == k).astype(np.float32), (feather, feather))
            for k in range(len(frames))])
        weights /= np.maximum(weights.sum(axis=0, keepdims=True), EPS)
        out = np.einsum("khw,khwc->hwc", weights, stack)
        return np.clip(out, 0, 255).astype(np.uint8)
    h, w = indices.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return stack[indices, ys, xs].astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= EPS else float(10.0 * np.log10(255.0 ** 2 / mse))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: The controls, kept alongside the measures so no table can omit them.
CONTROLS = {
    "Random pick (control)": select_random,
    "First frame (control)": select_first,
}


def stack_for(name: str, n: int = N_FRAMES, seed: int = 0,
              max_sigma: float = MAX_SIGMA):
    """``(image, frames, truth)`` for one photograph."""
    image = load(name)
    frames, truth, _ = build_stack(image, n=n, seed=seed, max_sigma=max_sigma)
    return image, frames, truth


def evaluate(images=IMAGES, window: int = POOL, n: int = N_FRAMES,
             seed: int = 0) -> list[dict]:
    """Every measure and every control, over all twelve photographs."""
    acc = {name: {"psnr": [], "agree": []} for name in
           list(MEASURES) + list(CONTROLS) + ["Oracle (truth)"]}

    for image_name in images:
        image, frames, truth = stack_for(image_name, n=n, seed=seed)
        for name in MEASURES:
            idx = select_indices(frames, name, window)
            acc[name]["psnr"].append(psnr(merge(frames, idx), image))
            acc[name]["agree"].append(float((idx == truth).mean()))
        for name, fn in CONTROLS.items():
            idx = fn(frames)
            acc[name]["psnr"].append(psnr(merge(frames, idx), image))
            acc[name]["agree"].append(float((idx == truth).mean()))
        acc["Oracle (truth)"]["psnr"].append(psnr(merge(frames, truth), image))
        acc["Oracle (truth)"]["agree"].append(1.0)

    return [
        {
            "method": name,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "agreement_with_truth": round(float(np.mean(a["agree"])), 4),
            "is_control": name in CONTROLS or name == "Oracle (truth)",
        }
        for name, a in acc.items()
    ]


def per_image(images=IMAGES, window: int = POOL) -> list[dict]:
    rows = []
    for image_name in images:
        image, frames, truth = stack_for(image_name)
        row = {
            "image": image_name,
            "detail": round(detail(image), 1),
            "flat_share": round(flat_share(image), 4),
            "oracle_db": round(psnr(merge(frames, truth), image), 3),
            "random_db": round(psnr(merge(frames, select_random(frames)), image), 3),
        }
        for name in MEASURES:
            idx = select_indices(frames, name, window)
            row[name] = round(psnr(merge(frames, idx), image), 3)
        rows.append(row)
    return rows


def sweep_pooling(images=IMAGES[:6], windows=(1, 3, 5, 9, 15, 25, 41),
                  measure: str = "Variance of Laplacian") -> list[dict]:
    """How much the pooling window is worth, against how much the measure is worth.

    The project's headline. The window is a parameter almost nobody reports, and
    on this data it moves the result several times further than choosing a
    different measure does.
    """
    rows = []
    for window in windows:
        psnrs, agrees = [], []
        for image_name in images:
            image, frames, truth = stack_for(image_name)
            idx = select_indices(frames, measure, window)
            psnrs.append(psnr(merge(frames, idx), image))
            agrees.append(float((idx == truth).mean()))
        rows.append({
            "window": window,
            "psnr_db": round(float(np.mean(psnrs)), 3),
            "agreement_with_truth": round(float(np.mean(agrees)), 4),
        })
    return rows


def measure_spread(images=IMAGES, window: int = POOL) -> dict:
    """The spread across measures against the spread across pooling windows.

    **Both on the same photographs.** The first version of this function compared
    a measure spread over twelve images against a pooling spread over six, which
    is not a comparison of anything -- the two sets have different ceilings, and
    the numbers differed by 3 dB for that reason alone.
    """
    rows = evaluate(images, window)
    real = [r["psnr_db"] for r in rows if not r["is_control"]]
    pooling = [r["psnr_db"] for r in sweep_pooling(images=images)]
    return {
        "measure_spread_db": round(max(real) - min(real), 3),
        "pooling_spread_db": round(max(pooling) - min(pooling), 3),
        "pooling_matters_more": bool((max(pooling) - min(pooling))
                                     > (max(real) - min(real))),
        "best_measure_db": round(max(real), 3),
        "worst_measure_db": round(min(real), 3),
    }


def disagreement(images=IMAGES, window: int = POOL) -> list[dict]:
    """Where the five measures disagree, and how much of it is flat region.

    A measure can only be wrong where there is something to be right about. If
    the disagreement were spread evenly over the image this would be a story
    about the measures; it is concentrated in the regions where every frame looks
    the same, which makes it a story about the photograph.
    """
    rows = []
    for image_name in images:
        image, frames, truth = stack_for(image_name)
        picks = np.stack([select_indices(frames, m, window) for m in MEASURES])
        disagree = (picks != picks[0]).any(axis=0)

        g = to_gray(image).astype(np.float32)
        dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        local = cv2.blur(np.hypot(dx, dy), (window, window))
        flat = local < 2.0

        rows.append({
            "image": image_name,
            "disagree_share": round(float(disagree.mean()), 4),
            "flat_share": round(float(flat.mean()), 4),
            "disagreement_in_flat": round(
                float(disagree[flat].mean()) if flat.any() else 0.0, 4),
            "disagreement_in_detailed": round(
                float(disagree[~flat].mean()) if (~flat).any() else 0.0, 4),
        })
    return rows


def flat_regions_are_unwinnable(images=IMAGES, window: int = POOL) -> dict:
    """Agreement with truth inside flat regions against outside them.

    Where every frame is identically smooth there is no evidence, and the answer
    is decided by noise. Pooled over the twelve photographs so the effect is not
    one image.
    """
    inside, outside = [], []
    for image_name in images:
        image, frames, truth = stack_for(image_name)
        idx = select_indices(frames, "Variance of Laplacian", window)
        g = to_gray(image).astype(np.float32)
        dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        flat = cv2.blur(np.hypot(dx, dy), (window, window)) < 2.0
        if flat.any():
            inside.append(float((idx == truth)[flat].mean()))
        if (~flat).any():
            outside.append(float((idx == truth)[~flat].mean()))
    return {
        "agreement_in_flat_regions": round(float(np.mean(inside)), 4) if inside else None,
        "agreement_in_detailed_regions": round(float(np.mean(outside)), 4),
        "images_with_any_flat_region": len(inside),
    }


def detail_predicts_the_ceiling(images=IMAGES) -> dict:
    """Does the selection axis predict how well the stack can be rebuilt?

    Asked rather than assumed. `detail` is computed from the photograph before a
    stack exists; the oracle PSNR is the best any selection could do.
    """
    rows = per_image(images)
    x = np.array([r["detail"] for r in rows])
    y = np.array([r["oracle_db"] for r in rows])
    return {
        "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 4),
        "detail_range": [float(x.min()), float(x.max())],
        "oracle_db_range": [float(y.min()), float(y.max())],
    }


def sweep_frames(images=IMAGES[:6], counts=(3, 5, 9, 17)) -> list[dict]:
    """More frames means a finer depth quantisation and a harder selection."""
    rows = []
    for n in counts:
        oracle, best = [], []
        for image_name in images:
            image, frames, truth = stack_for(image_name, n=n)
            oracle.append(psnr(merge(frames, truth), image))
            idx = select_indices(frames, "Variance of Laplacian")
            best.append(psnr(merge(frames, idx), image))
        rows.append({
            "frames": n,
            "oracle_db": round(float(np.mean(oracle)), 3),
            "variance_of_laplacian_db": round(float(np.mean(best)), 3),
            "gap_db": round(float(np.mean(oracle)) - float(np.mean(best)), 3),
        })
    return rows


def sweep_blur(images=IMAGES[:6], sigmas=(1.0, 2.0, 4.0, 8.0)) -> list[dict]:
    """How much defocus there has to be before the measures can see it."""
    rows = []
    for sigma in sigmas:
        row: dict[str, float] = {"max_sigma": sigma}
        for name in MEASURES:
            agree = []
            for image_name in images:
                _, frames, truth = stack_for(image_name, max_sigma=sigma)
                idx = select_indices(frames, name)
                agree.append(float((idx == truth).mean()))
            row[name] = round(float(np.mean(agree)), 4)
        rows.append(row)
    return rows


def feathering_effect(images=IMAGES[:6], feathers=(0, 5, 11, 21)) -> list[dict]:
    """Blurring the selection map: fewer seams, softer result."""
    rows = []
    for feather in feathers:
        psnrs = []
        for image_name in images:
            image, frames, truth = stack_for(image_name)
            idx = select_indices(frames, "Variance of Laplacian")
            psnrs.append(psnr(merge(frames, idx, feather=feather), image))
        rows.append({
            "feather": feather,
            "psnr_db": round(float(np.mean(psnrs)), 3),
        })
    return rows
