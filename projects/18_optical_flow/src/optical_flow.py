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

    ix = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    it = n - p

    k = (window, window)
    ixx = cv2.blur(ix * ix, k)
    iyy = cv2.blur(iy * iy, k)
    ixy = cv2.blur(ix * iy, k)
    ixt = cv2.blur(ix * it, k)
    iyt = cv2.blur(iy * it, k)

    # solve the 2x2 system per pixel; the determinant is the aperture problem
    # made numerical -- it vanishes wherever the local structure is 1-D
    det = ixx * iyy - ixy * ixy
    valid = np.abs(det) > 1e-6
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


def flow_horn_schunck(prev: np.ndarray, nxt: np.ndarray, alpha: float = 1.0, iters: int = 100):
    """Horn-Schunck: global smoothness, solved by Jacobi iteration.

    Where LK assumes flow is constant in a window, HS assumes it varies smoothly
    over the *whole image* and minimises a global energy. That fills in
    textureless regions LK leaves blank, and over-smooths genuine motion
    boundaries — the classic local-versus-global trade-off.
    """
    p = to_gray(prev).astype(np.float32) / 255.0
    n = to_gray(nxt).astype(np.float32) / 255.0
    ix = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
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

IMAGES = ("astronaut", "coffee", "chelsea", "brick")
MAGNITUDES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
PYRAMID_LEVELS = (0, 1, 2, 3, 4)


def make_pair(image: str = "astronaut", magnitude: float = 4.0, kind: str = "translation"):
    """Two frames related by a **known** dense flow field."""
    from shared import io, synth

    first = io.sample(image)
    flow = synth.synthetic_flow(first.shape[:2], magnitude=magnitude, kind=kind)
    second = synth.warp_by_flow(first, flow)
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


def compute(prev: np.ndarray, nxt: np.ndarray, method: str = "DIS") -> np.ndarray:
    return METHODS[method](prev, nxt)
