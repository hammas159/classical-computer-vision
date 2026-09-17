"""Hough transforms: lines, circles, and the cost of a vote.

The question
------------
The Hough transform converts detection into voting: every edge pixel votes for
every shape that could pass through it, and peaks in the accumulator are the
shapes. It is elegant and it is expensive, and the expense scales badly.

> **The claim under test:** the accumulator grows with the *number of shape
> parameters*, so a line (2 parameters) is cheap and a circle (3) costs an order
> of magnitude more — and a generalised shape is worse again. That scaling, not
> accuracy, is what decides whether Hough is usable.

Because the shapes are drawn here, their true parameters are known exactly.
Detection becomes measurable in the units that matter: **degrees of angle error**
and **pixels of radius error**, rather than a count of "found" shapes.

A second question the probabilistic variant raises:

> ``HoughLinesP`` samples a random subset of edge pixels. Does that cost accuracy,
> and how much time does it save?
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray

EPS = 1e-9


# --------------------------------------------------------------------------- #
# scenes with known geometry
# --------------------------------------------------------------------------- #


def line_scene(size: int = 512, n_lines: int = 5, noise_sigma: float = 0.0,
               clutter: int = 0, seed: int = 0):
    """Draw lines at known (rho, theta) and return both image and parameters.

    Lines are specified in the *same* polar form Hough uses, so the comparison is
    direct and needs no conversion that could itself introduce error.
    """
    from shared import synth

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    truth = []
    for _ in range(n_lines):
        theta = float(rng.uniform(0, np.pi))
        rho = float(rng.uniform(-size / 3, size / 3))
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho + size / 2, b * rho + size / 2
        p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(img, p1, p2, 255, 2)
        truth.append((rho, theta))

    for _ in range(clutter):
        cv2.circle(
            img, (int(rng.integers(0, size)), int(rng.integers(0, size))),
            int(rng.integers(5, 30)), 255, 2,
        )

    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return img, truth


def circle_scene(size: int = 512, n_circles: int = 5, noise_sigma: float = 0.0,
                 clutter: int = 0, seed: int = 0):
    """Draw circles at known (x, y, r)."""
    from shared import synth

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    truth = []
    for _ in range(n_circles):
        r = int(rng.integers(size // 16, size // 6))
        cx = int(rng.integers(r + 5, size - r - 5))
        cy = int(rng.integers(r + 5, size - r - 5))
        cv2.circle(img, (cx, cy), r, 255, 2)
        truth.append((float(cx), float(cy), float(r)))

    for _ in range(clutter):
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.line(img, p1, p2, 255, 2)

    if noise_sigma > 0:
        img = synth.gaussian_noise(img, sigma=noise_sigma, seed=seed)
    return img, truth


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def detect_lines_standard(gray: np.ndarray, threshold: int = 120):
    """Standard Hough: every edge pixel votes in a (rho, theta) accumulator."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return []
    # Hough returns rho relative to the image origin; the scene defines it
    # relative to the centre, so shift once here rather than in every comparison
    h, w = gray.shape
    out = []
    for rho, theta in lines[:, 0]:
        shift = (w / 2) * np.cos(theta) + (h / 2) * np.sin(theta)
        out.append((float(rho - shift), float(theta)))
    return out


def detect_lines_probabilistic(gray: np.ndarray, threshold: int = 60,
                               min_length: int = 60, max_gap: int = 10):
    """Probabilistic Hough: vote with a random subset, return finite segments.

    Two differences that matter. It is faster because it votes with fewer pixels,
    and it returns **segments with endpoints** rather than infinite lines — which
    is usually what an application actually wants, and is information the
    standard transform throws away.
    """
    edges = cv2.Canny(gray, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                           minLineLength=min_length, maxLineGap=max_gap)
    if segs is None:
        return [], []
    h, w = gray.shape
    params = []
    for x1, y1, x2, y2 in segs[:, 0]:
        theta = (np.arctan2(y2 - y1, x2 - x1) + np.pi / 2) % np.pi
        rho = (x1 - w / 2) * np.cos(theta) + (y1 - h / 2) * np.sin(theta)
        params.append((float(rho), float(theta)))
    return params, segs[:, 0]


def detect_circles(gray: np.ndarray, dp: float = 1.2, min_dist: int = 30,
                   param1: int = 120, param2: int = 40,
                   min_radius: int = 10, max_radius: int = 120):
    """Hough circles, via the gradient method.

    OpenCV does not build a full 3-D accumulator. It uses the *gradient
    direction* at each edge pixel to vote only along the normal, collapsing the
    centre search to 2-D and finding radii afterwards. That optimisation is the
    only reason circle Hough is tractable at all, and it is why ``param2``
    behaves so differently from a plain vote threshold.
    """
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1.2), cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist, param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


# --------------------------------------------------------------------------- #
# the accumulator, built by hand
# --------------------------------------------------------------------------- #


def build_line_accumulator(edges: np.ndarray, rho_step: float = 1.0, theta_steps: int = 180):
    """Build the (rho, theta) accumulator explicitly, for the figures.

    Worth doing once by hand: the accumulator *is* the algorithm, and seeing a
    sinusoid per edge pixel — intersecting where the line is — explains Hough
    better than any description.
    """
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))
    thetas = np.linspace(0, np.pi, theta_steps, endpoint=False)
    acc = np.zeros((int(2 * diag / rho_step) + 1, theta_steps), np.int32)

    ys, xs = np.nonzero(edges)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for x, y in zip(xs, ys):
        rhos = x * cos_t + y * sin_t
        idx = ((rhos + diag) / rho_step).astype(np.int32)
        np.add.at(acc, (idx, np.arange(theta_steps)), 1)
    return acc, thetas, diag


def accumulator_size(n_params: int, resolution: int = 180) -> int:
    """Accumulator cells for a shape with ``n_params`` parameters.

    The scaling claim, as arithmetic: a line needs ``R x T`` cells, a circle
    ``X x Y x R``, a generalised shape adds rotation and scale on top. Each extra
    parameter multiplies the cost, and this makes that concrete before any timing
    is measured.
    """
    return int(resolution**n_params)


# --------------------------------------------------------------------------- #
# matching detections to truth
# --------------------------------------------------------------------------- #


def match_lines(detected, truth, rho_tol: float = 10.0, theta_tol_deg: float = 3.0):
    """Greedily match detections to true lines; report angle and offset error.

    Angles wrap at pi, so the comparison has to handle a line at 179 degrees and
    one at 1 degree being nearly identical. Getting that wrong makes a correct
    detector look badly broken on a handful of lines.
    """
    theta_tol = np.deg2rad(theta_tol_deg)
    used = set()
    angle_errors, rho_errors = [], []
    for t_rho, t_theta in truth:
        best, best_cost = None, None
        for i, (d_rho, d_theta) in enumerate(detected):
            if i in used:
                continue
            dtheta = abs(d_theta - t_theta) % np.pi
            dtheta = min(dtheta, np.pi - dtheta)
            drho = abs(abs(d_rho) - abs(t_rho))
            cost = dtheta / max(theta_tol, EPS) + drho / max(rho_tol, EPS)
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        if best is not None and best_cost is not None and best_cost < 2.0:
            used.add(best)
            d_rho, d_theta = detected[best]
            dtheta = abs(d_theta - t_theta) % np.pi
            angle_errors.append(np.rad2deg(min(dtheta, np.pi - dtheta)))
            rho_errors.append(abs(abs(d_rho) - abs(t_rho)))

    return {
        "detected": len(detected),
        "true": len(truth),
        "matched": len(used),
        "recall": len(used) / max(len(truth), 1),
        "precision": len(used) / max(len(detected), 1),
        "angle_error_deg": float(np.mean(angle_errors)) if angle_errors else float("nan"),
        "rho_error_px": float(np.mean(rho_errors)) if rho_errors else float("nan"),
    }


def match_circles(detected, truth, centre_tol: float = 15.0):
    used = set()
    centre_errors, radius_errors = [], []
    for tx, ty, tr in truth:
        best, best_d = None, None
        for i, (dx, dy, dr) in enumerate(detected):
            if i in used:
                continue
            d = float(np.hypot(dx - tx, dy - ty))
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None and best_d is not None and best_d < centre_tol:
            used.add(best)
            dx, dy, dr = detected[best]
            centre_errors.append(best_d)
            radius_errors.append(abs(dr - tr))

    return {
        "detected": len(detected),
        "true": len(truth),
        "matched": len(used),
        "recall": len(used) / max(len(truth), 1),
        "precision": len(used) / max(len(detected), 1),
        "centre_error_px": float(np.mean(centre_errors)) if centre_errors else float("nan"),
        "radius_error_px": float(np.mean(radius_errors)) if radius_errors else float("nan"),
    }


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **edge density** 1.9% to 36.3%, from BSDS500 so
#: they carry human boundary annotations. Hough finds straight structure and a
#: photograph's straight structure is mostly man-made, so the pool deliberately
#: mixes glazed roofs and window grids with cheetahs and rock faces.
IMAGES = (
    "camel_at_sunset",     # edges  1.9 - one long horizon and little else
    "woman_black_beret",   # edges  9.2
    "held_sunfish",        # edges 11.6
    "clouded_leopard",     # edges 13.1
    "palms_at_dusk",       # edges 15.3
    "cheetah_walking",     # edges 17.5
    "two_women_street",    # edges 19.1
    "rocky_cove",          # edges 20.5
    "glass_roof_trees",    # edges 23.0 - dense man-made straight lines
    "barges_and_blocks",   # edges 25.4
    "hotel_rossiya",       # edges 29.9 - a grid of windows
    "ocelot_on_rock",      # edges 36.3 - the busiest frame in the pool
)


def load_scene(name: str):
    from shared import io

    return io.real_photo(name)


def rasterise_lines(lines, shape) -> np.ndarray:
    """Draw detected lines as a binary mask, so they can be compared to a boundary map.

    `detect_lines_standard` returns infinite (rho, theta) lines and
    `detect_lines_probabilistic` returns finite segments. Drawing both onto a
    canvas is what puts them on the same footing -- and it is also the honest
    representation of the difference, because an infinite line crosses the whole
    frame including the parts where nothing supported it.
    """
    canvas = np.zeros(shape[:2], np.uint8)
    h, w = shape[:2]
    for item in lines:
        if len(item) == 4:
            x1, y1, x2, y2 = (int(v) for v in item)
        else:
            rho, theta = float(item[0]), float(item[1])
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            big = max(h, w) * 2
            x1, y1 = int(x0 + big * (-b)), int(y0 + big * a)
            x2, y2 = int(x0 - big * (-b)), int(y0 - big * a)
        cv2.line(canvas, (x1, y1), (x2, y2), 255, 1, cv2.LINE_8)
    return canvas


def evaluate_photo_lines(images=None, tolerance: int = 2, min_annotators: int = 2):
    """How many of Hough's lines land on something a person actually drew.

    **Precision only, and deliberately so.** A photograph's human boundary map
    contains every contour a person traced, most of which are not straight;
    asking a line detector to recall them would be asking it to be a different
    algorithm. Precision -- of the line pixels Hough drew, how many sit on a real
    boundary -- is exactly the question a line detector should be asked.

    The `Canny edges` row is the control. Hough is built on Canny's output, so a
    Hough precision below Canny's means the voting stage *added* error rather
    than removing it.
    """
    from shared import bsds

    images = images if images is not None else IMAGES
    variants = {
        "Canny edges (control)": lambda g: cv2.Canny(g, 50, 150),
        "Standard Hough": lambda g: rasterise_lines(detect_lines_standard(g), g.shape),
        "Probabilistic Hough": lambda g: rasterise_lines(
            detect_lines_probabilistic(g)[0], g.shape),
    }
    rows = []
    for label, fn in variants.items():
        precisions, coverage = [], []
        for name in images:
            gray = to_gray(load_scene(name))
            target = bsds.consensus_boundaries(name, min_annotators)
            drawn = fn(gray)
            score = bsds.boundary_f_measure(drawn, target, tolerance)
            precisions.append(score["precision"])
            coverage.append(float((drawn > 0).mean()))
        rows.append({
            "method": label,
            "precision": round(float(np.mean(precisions)), 4),
            "pixels_drawn": round(float(np.mean(coverage)) * 100, 3),
        })
    return rows


NOISE_LEVELS = (0.0, 10.0, 25.0, 45.0)
CLUTTER_LEVELS = (0, 3, 8, 15)
LINE_THRESHOLDS = (60, 90, 120, 160, 220)


def evaluate_lines(noise_sigma: float = 0.0, clutter: int = 0, seeds=(0, 1, 2), runs: int = 3):
    """Standard against probabilistic Hough, on accuracy and on time."""
    rows = []
    for label, fn in (
        ("Standard Hough", lambda g: detect_lines_standard(g)),
        ("Probabilistic Hough", lambda g: detect_lines_probabilistic(g)[0]),
    ):
        acc = {k: [] for k in ("recall", "precision", "angle_error_deg", "rho_error_px", "ms")}
        for seed in seeds:
            img, truth = line_scene(noise_sigma=noise_sigma, clutter=clutter, seed=seed)
            gray = to_gray(img)
            detected, timing = timeit(lambda f=fn, g=gray: f(g), runs=runs, warmup=1)
            m = match_lines(detected, truth)
            for k in ("recall", "precision", "angle_error_deg", "rho_error_px"):
                acc[k].append(m[k])
            acc["ms"].append(timing.median_ms)
        rows.append(
            {
                "method": label,
                "recall": round(float(np.mean(acc["recall"])), 4),
                "precision": round(float(np.mean(acc["precision"])), 4),
                "angle_error_deg": round(float(np.nanmean(acc["angle_error_deg"])), 4),
                "rho_error_px": round(float(np.nanmean(acc["rho_error_px"])), 3),
                "median_ms": round(float(np.median(acc["ms"])), 3),
            }
        )
    return rows


def evaluate_circles(noise_sigma: float = 0.0, clutter: int = 0, seeds=(0, 1, 2), runs: int = 3):
    acc = {k: [] for k in ("recall", "precision", "centre_error_px", "radius_error_px", "ms")}
    for seed in seeds:
        img, truth = circle_scene(noise_sigma=noise_sigma, clutter=clutter, seed=seed)
        gray = to_gray(img)
        detected, timing = timeit(lambda g=gray: detect_circles(g), runs=runs, warmup=1)
        m = match_circles(detected, truth)
        for k in ("recall", "precision", "centre_error_px", "radius_error_px"):
            acc[k].append(m[k])
        acc["ms"].append(timing.median_ms)
    return [
        {
            "method": "Hough circles (gradient)",
            "recall": round(float(np.mean(acc["recall"])), 4),
            "precision": round(float(np.mean(acc["precision"])), 4),
            "centre_error_px": round(float(np.nanmean(acc["centre_error_px"])), 3),
            "radius_error_px": round(float(np.nanmean(acc["radius_error_px"])), 3),
            "median_ms": round(float(np.median(acc["ms"])), 3),
        }
    ]


def cost_scaling(seeds=(0,), runs: int = 3):
    """Lines against circles: parameters, accumulator cells and measured time.

    The three columns should tell one consistent story — an extra parameter costs
    an order of magnitude — and if the timing does *not* match the accumulator
    arithmetic, that gap is OpenCV's gradient optimisation showing up.
    """
    rows = []
    for label, n_params, fn, scene_fn in (
        ("Lines (rho, theta)", 2, detect_lines_standard, line_scene),
        ("Circles (x, y, r)", 3, detect_circles, circle_scene),
    ):
        ms = []
        for seed in seeds:
            img, _ = scene_fn(seed=seed)
            _, timing = timeit(lambda f=fn, g=to_gray(img): f(g), runs=runs, warmup=1)
            ms.append(timing.median_ms)
        rows.append(
            {
                "shape": label,
                "parameters": n_params,
                "accumulator_cells": accumulator_size(n_params),
                "median_ms": round(float(np.median(ms)), 3),
            }
        )
    rows.append(
        {
            "shape": "Generalised (x, y, scale, rotation)",
            "parameters": 4,
            "accumulator_cells": accumulator_size(4),
            "median_ms": None,
        }
    )
    return rows


def sweep_threshold(thresholds=LINE_THRESHOLDS, seeds=(0, 1), clutter: int = 5):
    """The vote threshold trades recall against precision, as expected — but the
    *angle error* of the matched lines barely moves, which says the threshold
    controls which lines survive rather than how well they are localised."""
    rows = []
    for t in thresholds:
        acc = {k: [] for k in ("recall", "precision", "angle_error_deg")}
        for seed in seeds:
            img, truth = line_scene(clutter=clutter, seed=seed)
            m = match_lines(detect_lines_standard(to_gray(img), threshold=t), truth)
            for k in acc:
                acc[k].append(m[k])
        rows.append(
            {
                "threshold": t,
                "recall": round(float(np.mean(acc["recall"])), 4),
                "precision": round(float(np.mean(acc["precision"])), 4),
                "angle_error_deg": round(float(np.nanmean(acc["angle_error_deg"])), 4),
            }
        )
    return rows


def sweep_noise(levels=NOISE_LEVELS, seeds=(0, 1)):
    """Hough votes, and votes are robust — degradation should be gentle."""
    rows = []
    for sigma in levels:
        lines = evaluate_lines(noise_sigma=sigma, seeds=seeds, runs=1)
        circles = evaluate_circles(noise_sigma=sigma, seeds=seeds, runs=1)
        rows.append(
            {
                "noise_sigma": sigma,
                "line_recall": lines[0]["recall"],
                "line_angle_error_deg": lines[0]["angle_error_deg"],
                "circle_recall": circles[0]["recall"],
                "circle_radius_error_px": circles[0]["radius_error_px"],
            }
        )
    return rows


def sweep_clutter(levels=CLUTTER_LEVELS, seeds=(0, 1)):
    """Clutter is the harder test: it adds *structured* false votes, not noise."""
    rows = []
    for c in levels:
        lines = evaluate_lines(clutter=c, seeds=seeds, runs=1)
        rows.append(
            {
                "clutter": c,
                "standard_precision": lines[0]["precision"],
                "standard_recall": lines[0]["recall"],
                "probabilistic_precision": lines[1]["precision"],
                "probabilistic_recall": lines[1]["recall"],
            }
        )
    return rows
