"""Camera calibration: the number everyone reports, and why it lies.

The question
------------
Point a camera at a known planar grid from several angles and you can recover the
focal length, the principal point and the lens distortion. `cv2.calibrateCamera`
returns a single RMS reprojection error, and that number is what gets quoted.

> **The claim under test:** a lower reprojection error is a better calibration.
> **It is not**, and the gap is not subtle. Adding distortion coefficients drives
> the error on the views you fitted *down* while the error on views you did not
> fit goes *up*. With three views the fitted RMS is the lowest in this project
> and the held-out RMS is the worst.

Where the ground truth comes from
---------------------------------
Three separate sources, and they are independent of each other, which is the
point:

1. **The board's own geometry.** Nine by six inner corners on a square grid, which
   is exact by construction. This is what `calibrateCamera` optimises against, so
   it cannot also be a fair test of the result.
2. **Held-out views.** Calibrate on some of the thirteen poses, measure on the
   others. Same objective, unseen data -- the standard answer to overfitting, and
   here it is decisive.
3. **Straightness after undistortion.** A straight line in the world must be
   straight in the image once the lens model is removed. `calibrateCamera` never
   optimises this, so it is an **out-of-objective** check: a model can lower
   reprojection error and bend lines at the same time, and that is exactly what
   the over-parameterised one does.

The two controls
----------------
`No distortion model` fixes every coefficient at zero, and
`Everything (14 coefficients)` turns on the rational, thin-prism and tilted-sensor
terms. They bracket the real models: the first says how much of the error the
lens actually causes, and the second says what happens when you let the optimiser
have whatever it wants.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

EPS = 1e-12

#: OpenCV's stereo calibration set: a 9x6 chessboard shot from thirteen poses by
#: two cameras. Cached by `tools/fetch_assets.py --set calibration`.
ASSETS = Path.home() / ".cache" / "classical-cv-images" / "assets" / "calibration"

#: Inner corners, not squares. The board is 10x7 squares.
BOARD = (9, 6)

#: left10/right10 are 404 on the upstream host and were never cached, so the set
#: is thirteen poses rather than fourteen. Stated rather than silently skipped.
VIEW_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)

#: The square size is not published with these images. Every length here is
#: therefore in **board squares**: focal length in pixels is unaffected by the
#: choice, and the stereo baseline is reported in squares rather than invented
#: millimetres.
SQUARE = 1.0

CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def assets_available() -> bool:
    return ASSETS.exists() and any(ASSETS.glob("left*.jpg"))


def image_path(side: str, index: int) -> Path:
    return ASSETS / f"{side}{index:02d}.jpg"


def load_view(side: str, index: int) -> np.ndarray:
    path = image_path(side, index)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"{path} is missing. Run `python tools/fetch_assets.py --set calibration`.")
    return img


def object_points(board=BOARD, square: float = SQUARE) -> np.ndarray:
    """The board's corners in its own plane, z = 0. Exact by construction."""
    grid = np.zeros((board[0] * board[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
    return grid * square


_CORNERS: dict[tuple[str, int], np.ndarray | None] = {}


def find_corners(side: str, index: int, refine: bool = True):
    """Sub-pixel corner locations for one view, or None if the board is not found.

    Cached: every experiment here re-uses the same detections, so a difference
    between two calibrations is never a difference in what was detected.
    """
    key = (side, index, refine)
    if key in _CORNERS:
        return _CORNERS[key]
    img = load_view(side, index)
    found, corners = cv2.findChessboardCorners(
        img, BOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        _CORNERS[key] = None
        return None
    if refine:
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    _CORNERS[key] = corners
    return corners


def usable_views(side: str = "left") -> list[int]:
    """The views where the board is actually found, reported rather than assumed."""
    return [i for i in VIEW_IDS if find_corners(side, i) is not None]


def image_size(side: str = "left") -> tuple[int, int]:
    img = load_view(side, usable_views(side)[0])
    return (img.shape[1], img.shape[0])


# --------------------------------------------------------------------------- #
# the models
# --------------------------------------------------------------------------- #

#: Every model is the same call with different flags, so nothing but the
#: parameterisation changes between rows of the results table.
MODELS: dict[str, int] = {
    "No distortion model (control)":
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3,
    "k1 only":
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
    "k1, k2":
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "k1, k2, p1, p2 (OpenCV default)":
        cv2.CALIB_FIX_K3,
    "k1, k2, k3, p1, p2":
        0,
    "Rational (8 coefficients)":
        cv2.CALIB_RATIONAL_MODEL,
    "Everything (14 coefficients, control)":
        cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
        | cv2.CALIB_TILTED_MODEL,
}


def coefficient_count(name: str) -> int:
    """How many distortion numbers each model is actually free to move."""
    return {
        "No distortion model (control)": 0,
        "k1 only": 1,
        "k1, k2": 2,
        "k1, k2, p1, p2 (OpenCV default)": 4,
        "k1, k2, k3, p1, p2": 5,
        "Rational (8 coefficients)": 8,
        "Everything (14 coefficients, control)": 14,
    }[name]


def calibrate(views, side: str = "left", flags: int = cv2.CALIB_FIX_K3):
    """Calibrate on the given view indices. Returns the full OpenCV result."""
    objp = object_points()
    obj, img = [], []
    for i in views:
        corners = find_corners(side, i)
        if corners is None:
            continue
        obj.append(objp)
        img.append(corners)
    if len(obj) < 3:
        raise ValueError(f"{len(obj)} usable views; calibration needs at least 3")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size(side), None, None, flags=flags)
    return {"rms": float(rms), "K": K, "dist": dist, "rvecs": rvecs,
            "tvecs": tvecs, "views": list(views)}


def reprojection_error(K, dist, views, side: str = "left") -> float:
    """RMS reprojection error on views the calibration may or may not have seen.

    The pose of each board is re-solved with `solvePnP` before measuring, because
    a held-out view has no pose from the calibration. That is the only fair way
    to score one: the intrinsics are being tested, not the poses.
    """
    objp = object_points()
    squared, count = 0.0, 0
    for i in views:
        corners = find_corners(side, i)
        if corners is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if not ok:
            continue
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        diff = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        squared += float(np.sum(diff ** 2))
        count += len(diff)
    return float(np.sqrt(squared / max(count, 1)))


# --------------------------------------------------------------------------- #
# the out-of-objective check
# --------------------------------------------------------------------------- #


def line_straightness(K, dist, views, side: str = "left") -> float:
    """Mean perpendicular deviation, in pixels, of each board row after undistortion.

    A row of chessboard corners is a straight line in the world, so it must be a
    straight line in an undistorted image. **`calibrateCamera` never optimises
    this** -- it minimises reprojection error, and the two are only the same thing
    when the model is right. A model that lowers reprojection error while bending
    lines is overfitting, and this is the number that says so.

    Measured by fitting a total-least-squares line to each row of nine corners
    and averaging the residual.
    """
    residuals = []
    for i in views:
        corners = find_corners(side, i)
        if corners is None:
            continue
        undistorted = cv2.undistortPoints(corners, K, dist, P=K).reshape(-1, 2)
        rows = undistorted.reshape(BOARD[1], BOARD[0], 2)
        for row in rows:
            centred = row - row.mean(axis=0)
            # the smallest singular vector is the line's normal
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            normal = vt[-1]
            residuals.append(float(np.sqrt(np.mean((centred @ normal) ** 2))))
    return float(np.mean(residuals)) if residuals else float("nan")


def distorted_straightness(views, side: str = "left") -> float:
    """The same measurement with no undistortion at all: what the lens does."""
    identity = np.eye(3, dtype=np.float64)
    identity[0, 0] = identity[1, 1] = 1.0
    residuals = []
    for i in views:
        corners = find_corners(side, i)
        if corners is None:
            continue
        rows = corners.reshape(BOARD[1], BOARD[0], 2)
        for row in rows:
            centred = row - row.mean(axis=0)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            residuals.append(float(np.sqrt(np.mean((centred @ vt[-1]) ** 2))))
    del identity
    return float(np.mean(residuals)) if residuals else float("nan")


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate_models(side: str = "left", fit_views: int = 7,
                    seed: int = 0) -> list[dict]:
    """Every distortion model, scored three ways.

    `rms_fitted` is what `calibrateCamera` returns and what gets quoted.
    `rms_held_out` is the same measure on views the model never saw.
    `straightness` is the out-of-objective check.
    """
    views = usable_views(side)
    rng = np.random.default_rng(seed)
    order = list(views)
    rng.shuffle(order)
    fit, held = sorted(order[:fit_views]), sorted(order[fit_views:])

    rows = []
    for name, flags in MODELS.items():
        result = calibrate(fit, side, flags)
        K, dist = result["K"], result["dist"]
        rows.append({
            "model": name,
            "coefficients": coefficient_count(name),
            "rms_fitted": round(result["rms"], 4),
            "rms_held_out": round(reprojection_error(K, dist, held, side), 4),
            "straightness_px": round(line_straightness(K, dist, views, side), 4),
            "fx": round(float(K[0, 0]), 2),
            "fy": round(float(K[1, 1]), 2),
            "cx": round(float(K[0, 2]), 2),
            "cy": round(float(K[1, 2]), 2),
        })
    return {"fit_views": fit, "held_out_views": held, "rows": rows}


def sweep_view_count(side: str = "left", counts=(3, 4, 5, 6, 8, 10, 13),
                     flags: int = cv2.CALIB_FIX_K3, trials: int = 5,
                     seed: int = 0) -> list[dict]:
    """How the two errors move as views are added.

    The one that matters: with three views the fitted RMS is the **lowest** in the
    project and the held-out RMS is the **worst**. Repeated over several random
    draws so the effect is not one unlucky triple.
    """
    views = usable_views(side)
    rng = np.random.default_rng(seed)
    rows = []
    for n in counts:
        fitted, held, straight, focals = [], [], [], []
        for _ in range(trials):
            order = list(views)
            rng.shuffle(order)
            fit, rest = sorted(order[:n]), sorted(order[n:])
            if len(fit) < 3:
                continue
            result = calibrate(fit, side, flags)
            fitted.append(result["rms"])
            if rest:
                held.append(reprojection_error(result["K"], result["dist"], rest, side))
            straight.append(line_straightness(result["K"], result["dist"], views, side))
            focals.append(float(result["K"][0, 0]))
            if n == len(views):
                break
        rows.append({
            "views": n,
            "rms_fitted": round(float(np.mean(fitted)), 4),
            "rms_held_out": round(float(np.mean(held)), 4) if held else float("nan"),
            "straightness_px": round(float(np.mean(straight)), 4),
            "fx_mean": round(float(np.mean(focals)), 2),
            "fx_spread": round(float(np.std(focals)), 2),
        })
    return rows


def intrinsic_stability(side: str = "left", n: int = 5, trials: int = 12,
                        flags: int = cv2.CALIB_FIX_K3, seed: int = 1) -> list[dict]:
    """The same camera, calibrated from different subsets of the same photographs.

    Every row here is the same physical lens. The spread is how much of the
    answer came from which five poses happened to be used, and it is the honest
    error bar on any single calibration.
    """
    views = usable_views(side)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(trials):
        order = list(views)
        rng.shuffle(order)
        fit = sorted(order[:n])
        result = calibrate(fit, side, flags)
        K = result["K"]
        rows.append({
            "views": fit,
            "rms_fitted": round(result["rms"], 4),
            "fx": round(float(K[0, 0]), 2),
            "fy": round(float(K[1, 1]), 2),
            "cx": round(float(K[0, 2]), 2),
            "cy": round(float(K[1, 2]), 2),
            "k1": round(float(result["dist"].ravel()[0]), 5),
        })
    return rows


def full_calibration(side: str = "left", flags: int = cv2.CALIB_FIX_K3) -> dict:
    """All thirteen views with the default model: the answer this project ships."""
    views = usable_views(side)
    result = calibrate(views, side, flags)
    K, dist = result["K"], result["dist"]
    return {
        "side": side,
        "views": views,
        "rms": round(result["rms"], 4),
        "fx": round(float(K[0, 0]), 3),
        "fy": round(float(K[1, 1]), 3),
        "cx": round(float(K[0, 2]), 3),
        "cy": round(float(K[1, 2]), 3),
        "k1": round(float(dist.ravel()[0]), 6),
        "k2": round(float(dist.ravel()[1]), 6),
        "p1": round(float(dist.ravel()[2]), 6),
        "p2": round(float(dist.ravel()[3]), 6),
        "straightness_px": round(line_straightness(K, dist, views, side), 4),
        "straightness_before_px": round(distorted_straightness(views, side), 4),
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
    }


def stereo_calibration(flags: int = cv2.CALIB_FIX_K3) -> dict:
    """Both cameras at once, on the views where both boards are found.

    The baseline is a length, so it comes out in whatever unit the board's square
    is. The square size is not published with these images, so it is reported in
    **squares** rather than in invented millimetres -- and the ratio of baseline to
    square is a real, checkable number either way.
    """
    shared = [i for i in VIEW_IDS
              if find_corners("left", i) is not None
              and find_corners("right", i) is not None]
    objp = object_points()
    obj = [objp for _ in shared]
    left = [find_corners("left", i) for i in shared]
    right = [find_corners("right", i) for i in shared]

    kl = calibrate(shared, "left", flags)
    kr = calibrate(shared, "right", flags)

    rms, K1, d1, K2, d2, R, T, _, _ = cv2.stereoCalibrate(
        obj, left, right, kl["K"], kl["dist"], kr["K"], kr["dist"],
        image_size("left"), flags=cv2.CALIB_FIX_INTRINSIC)

    baseline = float(np.linalg.norm(T))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {
        "views": shared,
        "rms": round(float(rms), 4),
        "baseline_squares": round(baseline, 4),
        "rotation_deg": round(angle, 4),
        "translation": [round(float(v), 4) for v in T.ravel()],
        "left_fx": round(float(K1[0, 0]), 2),
        "right_fx": round(float(K2[0, 0]), 2),
    }


def baseline_stability(trials: int = 10, n: int = 6, seed: int = 2) -> list[dict]:
    """The stereo baseline from different subsets of the same pairs.

    A physical distance between two lenses cannot change between subsets. How much
    it appears to is the calibration's real uncertainty.
    """
    shared = [i for i in VIEW_IDS
              if find_corners("left", i) is not None
              and find_corners("right", i) is not None]
    objp = object_points()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(trials):
        order = list(shared)
        rng.shuffle(order)
        subset = sorted(order[:n])
        kl = calibrate(subset, "left")
        kr = calibrate(subset, "right")
        rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            [objp] * len(subset),
            [find_corners("left", i) for i in subset],
            [find_corners("right", i) for i in subset],
            kl["K"], kl["dist"], kr["K"], kr["dist"], image_size("left"),
            flags=cv2.CALIB_FIX_INTRINSIC)
        rows.append({
            "views": subset,
            "rms": round(float(rms), 4),
            "baseline_squares": round(float(np.linalg.norm(T)), 4),
        })
    return rows


def undistorted_view(index: int, K, dist, side: str = "left",
                     alpha: float = 0.0):
    """One view with the lens model removed, for looking at."""
    img = load_view(side, index)
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    return cv2.undistort(img, K, dist, None, newK)
