"""Optical flow: five methods, and the number nobody quotes.

The question
------------
"Lucas-Kanade fails for large motion" is in every textbook. None of them say
*how* large.

> **The claim under test:** plain LK breaks down past roughly one to two pixels
> of displacement, and each pyramid level roughly doubles the motion it can
> handle — so an L-level pyramid extends the range by about 2^L.

That is a falsifiable, quantitative statement, and the flow field is generated
here, so endpoint error is measured against the exact truth rather than against
another algorithm's output.

Endpoint error (EPE) is the standard metric and it is reported in **pixels**,
which makes it directly comparable to the displacement being tested — the whole
point is the ratio between them.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import endpoint_error

EPS = 1e-6

#: `cv2.Sobel` with ksize=3 applies an **unnormalised** kernel: its positive
#: coefficients sum to 4 and the separable pair multiplies out to a factor of 8.
#: The result is 8x the actual intensity derivative.
#:
#: That matters here more than almost anywhere else, because Lucas-Kanade and
#: Horn-Schunck both divide a temporal difference by a spatial one. Leaving the
#: gradient 8x too large makes the recovered flow 8x too *small* -- and it stays
#: pointing in the right direction, so it looks like a method that half-works
#: rather than a scale error. Measured before the fix: a true 1.00 px
#: displacement came back as 0.123 px, which is 1/8.1.
SOBEL_SCALE = 1.0 / 8.0


# --------------------------------------------------------------------------- #
# dense flow methods
# --------------------------------------------------------------------------- #


def flow_lucas_kanade_dense(
    prev: np.ndarray, nxt: np.ndarray, window: int = 15
) -> np.ndarray:
    """Dense Lucas-Kanade, solved per pixel from local image derivatives.

    LK assumes the flow is constant in a small window and solves the resulting
    least-squares system. The assumption that breaks is not that one — it is the
    *linearisation*: LK uses a first-order Taylor expansion of the brightness
    constancy equation, which is only valid while the displacement is small
    compared with the image structure. That is the real source of the
    one-to-two-pixel limit.

    Implemented directly rather than via a sparse tracker so that the dense
    endpoint error is comparable with the other methods.
    """
    p = to_gray(prev).astype(np.float32) / 255.0
    n = to_gray(nxt).astype(np.float32) / 255.0

    ix = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3, scale=SOBEL_SCALE)
    iy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3, scale=SOBEL_SCALE)
    it = n - p

    k = (window, window)
    ixx = cv2.blur(ix * ix, k)
    iyy = cv2.blur(iy * iy, k)
    ixy = cv2.blur(ix * iy, k)
    ixt = cv2.blur(ix * it, k)
    iyt = cv2.blur(iy * it, k)

    # Solve the 2x2 system per pixel. The determinant is the aperture problem
    # made numerical: it vanishes
    # wherever the local structure is 1-D. The threshold is on the *scaled*
    # gradients, so it is 8^4 = 4096 times smaller than it would be on raw
    # Sobel output -- lowered to match rather than left where it silently
    # rejected well-conditioned pixels.
    det = ixx * iyy - ixy * ixy
    valid = np.abs(det) > 1e-10
    u = np.zeros_like(p)
    v = np.zeros_like(p)
    u[valid] = (-iyy[valid] * ixt[valid] + ixy[valid] * iyt[valid]) / det[valid]
    v[valid] = (ixy[valid] * ixt[valid] - ixx[valid] * iyt[valid]) / det[valid]
    return np.stack([u, v], axis=-1)


def flow_lk_pyramid(prev: np.ndarray, nxt: np.ndarray, levels: int = 3, window: int = 15):
    """Coarse-to-fine LK: solve on a shrunken image, upscale, refine.

    At level L the image is 2^L smaller, so a 16 px motion becomes a 1 px motion
    — back inside the linearisation's valid range. This is the entire trick, and
    it predicts the 2^L scaling the project sets out to measure.
    """
    p = to_gray(prev)
    n = to_gray(nxt)

    pyr_p = [p]
    pyr_n = [n]
    for _ in range(levels):
        pyr_p.append(cv2.pyrDown(pyr_p[-1]))
        pyr_n.append(cv2.pyrDown(pyr_n[-1]))

    flow = np.zeros((*pyr_p[-1].shape, 2), np.float32)
    for lvl in range(levels, -1, -1):
        pl, nl = pyr_p[lvl], pyr_n[lvl]
        if flow.shape[:2] != pl.shape:
            flow = cv2.resize(flow, (pl.shape[1], pl.shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0

        # warp the second frame back by the flow so far, then solve for the residual
        yy, xx = np.mgrid[0 : pl.shape[0], 0 : pl.shape[1]].astype(np.float32)
        warped = cv2.remap(
            nl, xx + flow[..., 0], yy + flow[..., 1],
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        flow = flow + flow_lucas_kanade_dense(pl, warped, window=window)
    return flow


def flow_farneback(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Farneback: fit a local quadratic polynomial to each neighbourhood.

    Instead of linearising the brightness, it models the image itself as a
    quadratic surface and derives displacement from how the polynomial
    coefficients change. That is a second-order model, so it handles larger
    motion than plain LK without any pyramid — though OpenCV pyramids it anyway.
    """
    return cv2.calcOpticalFlowFarneback(
        to_gray(prev), to_gray(nxt), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_dis(prev: np.ndarray, nxt: np.ndarray, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
    """DIS: dense inverse search. Modern, fast, and the practical default.

    Patch-based inverse search plus variational refinement. It is the method to
    reach for when you want dense flow in real time on a CPU, and it belongs in
    the comparison as the "what you would actually use" row.
    """
    dis = cv2.DISOpticalFlow.create(preset)
    return dis.calc(to_gray(prev), to_gray(nxt), None)


#: Jacobi iterations Horn-Schunck runs. The textbook figure is 100, and at 100
#: this implementation scores an EPE of **0.78 px on a 1.00 px displacement** --
#: barely distinguishable from predicting no motion at all, and it would have
#: gone into the table as "Horn-Schunck does not work".
#:
#: It is not that it does not work; it is that it has not finished. Jacobi
#: iteration propagates information one pixel per sweep, so filling a 481 px
#: frame from its edges takes hundreds of sweeps by construction. Measured on
#: the same pair: 100 -> 0.784, 300 -> 0.532, 1000 -> 0.268, 3000 -> 0.154.
#:
#: 1000 is the compromise reported in the main table -- converged enough to be
#: a fair representation of the method, at 10x the textbook cost. The full curve
#: is in `sweep_hs_iterations`, because "how slowly does it converge" is a more
#: useful fact about Horn-Schunck than any single row.
HS_ITERS = 1000
HS_ALPHA = 1.0


def flow_horn_schunck(prev: np.ndarray, nxt: np.ndarray,
                      alpha: float = HS_ALPHA, iters: int = HS_ITERS):
    """Horn-Schunck: global smoothness, solved by Jacobi iteration.

    Where LK assumes flow is constant in a window, HS assumes it varies smoothly
    over the *whole image* and minimises a global energy. That fills in
    textureless regions LK leaves blank, and over-smooths genuine motion
    boundaries — the classic local-versus-global trade-off.

    See `HS_ITERS` for why the iteration count is not the textbook 100.
    """
    p = to_gray(prev).astype(np.float32) / 255.0
    n = to_gray(nxt).astype(np.float32) / 255.0
    ix = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3, scale=SOBEL_SCALE)
    iy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3, scale=SOBEL_SCALE)
    it = n - p

    u = np.zeros_like(p)
    v = np.zeros_like(p)
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], np.float32)
    for _ in range(iters):
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)
        d = (ix * u_avg + iy * v_avg + it) / (alpha**2 + ix**2 + iy**2)
        u = u_avg - ix * d
        v = v_avg - iy * d
    return np.stack([u, v], axis=-1)


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Lucas-Kanade (dense)": flow_lucas_kanade_dense,
    "LK pyramid (3 levels)": flow_lk_pyramid,
    "Horn-Schunck": flow_horn_schunck,
    "Farneback": flow_farneback,
    "DIS": flow_dis,
}


def flow_zero(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Predict no motion at all — the control.

    Its EPE equals the true displacement by definition, which makes it the exact
    line a method must beat to be doing anything. A method scoring above this is
    worse than useless.
    """
    return np.zeros((*to_gray(prev).shape, 2), np.float32)


# --------------------------------------------------------------------------- #
# flow visualisation
# --------------------------------------------------------------------------- #


def flow_to_colour(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Standard hue=direction, saturation=magnitude colour wheel encoding."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # OpenCV hue is 0-179
    hsv[..., 1] = 255
    m = max_mag if max_mag is not None else max(float(mag.max()), EPS)
    hsv[..., 2] = np.clip(mag / m * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **texture density** — the fraction of spectral
#: energy above a quarter Nyquist. That is the axis optical flow lives or dies
#: on: every method here solves for displacement from local image structure, and
#: where there is none the system is singular. A pool of uniformly textured
#: images would hide the failure the aperture problem predicts. Selected by
#: `tools/select_images.py --axis texture`; the range is 45.8 to 84.7.
IMAGES = (
    "eagle_in_flight",     # texture 45.8 — blown-out sky, nothing to track
    "man_striped_shirt",   # texture 60.5 — strong 1-D structure
    "angelfish_reef",      # texture 63.8
    "iceberg_watcher",     # texture 66.2 — flat sea and sky
    "woman_by_tree",       # texture 67.7
    "bighorn_rock",        # texture 68.8
    "polar_bears_snow",    # texture 70.2 — white on white
    "firefighters_map",    # texture 71.5 — man-made lines in every direction
    "stone_archway",       # texture 72.6
    "mare_foal_meadow",    # texture 74.4
    "elephant_waterhole",  # texture 77.1
    "carved_mask_thatch",  # texture 84.7 — the most textured frame in the pool
)


def texture_of(img: np.ndarray) -> float:
    """Fraction of spectral energy above a quarter Nyquist, as a percentage.

    The same measure `tools/select_images.py --axis texture` selected this
    pool on, repeated here so the figure can label each row with the quantity
    that decides how well flow can be recovered from it.
    """
    g = cv2.resize(to_gray(img), (256, 256)).astype(np.float32) / 255.0
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    y, x = np.ogrid[:256, :256]
    total = f.sum()
    return float(f[np.hypot(y - 128, x - 128) > 32].sum() / total * 100) if total else 0.0


def load_scene(name: str) -> np.ndarray:
    """Load one of this project's photographs by name."""
    from shared import io

    return io.real_photo(name)
MAGNITUDES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
PYRAMID_LEVELS = (0, 1, 2, 3, 4)


def make_pair(image: str = "angelfish_reef", magnitude: float = 4.0,
              kind: str = "translation"):
    """Two frames related by a **known** dense flow field.

    Returns ``(first, second, flow)`` where ``flow`` is the **forward** flow: the
    displacement that carries a pixel of ``first`` to where it appears in
    ``second``. That is the convention every method in `METHODS` predicts, and
    the convention `endpoint_error` is scored in.

    The negation is not cosmetic. `synth.warp_by_flow` is a *backward* warp — it
    builds the output by sampling the input at ``(x + dx, y + dy)``, which is
    what `cv2.remap` wants — so warping by ``f`` moves image content by ``-f``.
    Passing ``flow`` straight through scored every method against the exact
    opposite of the truth, and made all five score **worse than predicting zero
    motion**: DIS came out at EPE 8.93 px on a 4 px displacement. Against the
    correct sign it scores **0.013 px**.

    That is the failure mode this whole comparison exists to detect, produced by
    the harness rather than by any method, and it looked exactly like a result.
    """
    from shared import synth

    first = load_scene(image)
    flow = synth.synthetic_flow(first.shape[:2], magnitude=magnitude, kind=kind)
    second = synth.warp_by_flow(first, -flow)
    return first, second, flow


def evaluate_methods(magnitude: float = 4.0, kind: str = "translation", images=IMAGES, runs: int = 1):
    """Endpoint error for every method at one displacement."""
    entries = list(METHODS.items()) + [("Predict zero (control)", flow_zero)]
    acc = {n: {"epe": [], "ms": []} for n, _ in entries}

    for image in images:
        first, second, truth = make_pair(image, magnitude, kind)
        for name, fn in entries:
            pred, timing = timeit(lambda f=fn: f(first, second), runs=runs, warmup=0)
            # crop the border: every method's edge handling differs and the
            # frame is not what is being compared
            m = 16
            acc[name]["epe"].append(endpoint_error(pred[m:-m, m:-m], truth[m:-m, m:-m]))
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": n,
            "epe_px": round(float(np.mean(a["epe"])), 4),
            "epe_relative": round(float(np.mean(a["epe"])) / max(magnitude, EPS), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for n, a in acc.items()
    ]


def sweep_magnitude(images=IMAGES, magnitudes=MAGNITUDES, kind: str = "translation"):
    """The central experiment: where does each method break?

    A method has "failed" once its EPE approaches the displacement itself, since
    that is what predicting zero would score.
    """
    rows = []
    for mag in magnitudes:
        scored = evaluate_methods(magnitude=mag, kind=kind, images=images)
        row: dict[str, float] = {"magnitude_px": mag}
        for r in scored:
            row[r["method"]] = r["epe_px"]
        rows.append(row)
    return rows


def sweep_pyramid_levels(images=IMAGES, levels=PYRAMID_LEVELS, magnitudes=MAGNITUDES):
    """Does each pyramid level really double the tractable motion?

    For each level count, report the largest displacement still handled with an
    EPE below half the displacement. If the doubling story is right, that
    threshold should itself roughly double per level.
    """
    rows = []
    for lvl in levels:
        breaking = None
        for mag in magnitudes:
            epes = []
            for image in images:
                first, second, truth = make_pair(image, mag)
                pred = (
                    flow_lucas_kanade_dense(first, second)
                    if lvl == 0
                    else flow_lk_pyramid(first, second, levels=lvl)
                )
                m = 16
                epes.append(endpoint_error(pred[m:-m, m:-m], truth[m:-m, m:-m]))
            if float(np.mean(epes)) < 0.5 * mag:
                breaking = mag
        rows.append({"pyramid_levels": lvl, "max_tractable_px": breaking})
    return rows


HS_ITER_LEVELS = (30, 100, 300, 1000, 3000)


def sweep_hs_iterations(images=None, magnitude: float = 1.0, iters=HS_ITER_LEVELS):
    """How slowly does Horn-Schunck converge?

    The answer is the most useful single fact about it. Jacobi iteration moves
    information one pixel per sweep, so a global method on a 481 px frame needs
    hundreds of sweeps before the smoothness term has reached the middle. Quoted
    at its textbook 100 iterations it looks like a failed method rather than an
    unfinished one.
    """
    images = images if images is not None else IMAGES[:4]
    rows = []
    for n in iters:
        epes, times = [], []
        for image in images:
            first, second, truth = make_pair(image, magnitude)
            pred, timing = timeit(
                lambda: flow_horn_schunck(first, second, iters=n), runs=1, warmup=0)
            m = 16
            epes.append(endpoint_error(pred[m:-m, m:-m], truth[m:-m, m:-m]))
            times.append(timing.median_ms)
        rows.append({
            "iterations": n,
            "epe_px": round(float(np.mean(epes)), 4),
            "median_ms": round(float(np.median(times)), 1),
        })
    return rows


def compute(prev: np.ndarray, nxt: np.ndarray, method: str = "DIS") -> np.ndarray:
    return METHODS[method](prev, nxt)
