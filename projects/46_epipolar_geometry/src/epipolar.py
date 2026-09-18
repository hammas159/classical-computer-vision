"""Epipolar geometry: five estimators of F, and a truth that shares nothing with them.

The question
------------
Two cameras look at the same scene. The fundamental matrix **F** says that a point
in one image must lie on a particular *line* in the other. Estimating it from
point correspondences is a textbook linear problem, and the textbook is unusually
emphatic about one detail.

> **The claim under test:** Hartley's normalisation is not a refinement, it is the
> difference between a working algorithm and a broken one. **True** — the same
> eight-point code on the same correspondences scores 144 px on raw pixels and
> 37.5 px normalised, a factor of four, and on some pairs a factor of twenty.
>
> **And the result that outranks it:** a control with **one parameter** — assume a
> rectified rig, fit a single constant vertical offset by taking a median — scores
> **0.74 px median**, better than RANSAC's 1.46 and thirty times better than the
> normalised eight-point algorithm. On a nearly parallel rig, estimating a general
> projective F from noisy features is worse than using the structure you already
> know.

Where the ground truth comes from
---------------------------------
This is the rare case where there is a real one. The same two cameras also
photographed a **9x6 chessboard from thirteen poses**, and project 35 calibrates
them from those board corners alone. That calibration yields

    F = K2^-T [t]x R K1^-1

from the stereo extrinsics — and it is built from **nothing but the board**, while
every estimator here is built from **nothing but scene features**. The two share
no measurement.

How good is it? It places all **702 board corners within 0.145 px** of their
epipolar lines. That is the floor this project measures against.

The planar trap
---------------
The chessboard is the largest flat thing in the frame and it attracts features.
**A set of correspondences that all lie on one plane does not determine F** — any
homography-compatible F fits them — so an estimator fed mostly board matches is
solving a degenerate problem while reporting a healthy inlier count.
`exclude_board` is the control that measures it.

The controls
------------
`Random 8 matches` fits F to eight correspondences chosen without regard to
quality: what a careless estimate looks like.

`Assume rectified` (zero parameters) asserts that corresponding points share a y
coordinate, and `Assume rectified + offset` (one parameter) allows a constant
vertical shift. They are not meaningless matrices -- they are the structure this
rig nearly has, and they are here to say how much of the answer is available
before any estimator runs. The honest reading of this project is that on this rig
it is most of it: the measured vertical disparity of the board corners is
12.93 px with a standard deviation of **0.79**.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

EPS = 1e-12

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "projects" / "35_camera_calibration" / "src"))

import calibration as cal  # noqa: E402

#: The stereo pairs, shared with project 35. Thirteen poses of a chessboard in an
#: office; `left10`/`right10` are 404 upstream and were never cached.
VIEW_IDS = cal.VIEW_IDS
BOARD = cal.BOARD


def assets_available() -> bool:
    return cal.assets_available()


def usable_pairs() -> list[int]:
    return [i for i in VIEW_IDS
            if cal.find_corners("left", i) is not None
            and cal.find_corners("right", i) is not None]


# --------------------------------------------------------------------------- #
# the truth
# --------------------------------------------------------------------------- #

_TRUTH: dict | None = None


def calibration_truth() -> dict:
    """F, the intrinsics and the board correspondences, from the chessboard alone.

    Cached. `cv2.stereoCalibrate` returns F directly, built from the extrinsics
    rather than from any scene correspondence — which is what makes it usable as
    truth for estimators that only see scene correspondences.
    """
    global _TRUTH
    if _TRUTH is not None:
        return _TRUTH

    views = usable_pairs()
    objp = cal.object_points()
    kl = cal.calibrate(views, "left")
    kr = cal.calibrate(views, "right")
    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        [objp] * len(views),
        [cal.find_corners("left", i) for i in views],
        [cal.find_corners("right", i) for i in views],
        kl["K"], kl["dist"], kr["K"], kr["dist"], cal.image_size("left"),
        flags=cv2.CALIB_FIX_INTRINSIC)

    left_pts, right_pts = [], []
    for i in views:
        left_pts.append(cv2.undistortPoints(
            cal.find_corners("left", i), K1, d1, P=K1).reshape(-1, 2))
        right_pts.append(cv2.undistortPoints(
            cal.find_corners("right", i), K2, d2, P=K2).reshape(-1, 2))

    _TRUTH = {
        "F": F / F[2, 2],
        "K1": K1, "d1": d1, "K2": K2, "d2": d2, "R": R, "T": T,
        "stereo_rms": float(rms),
        "views": views,
        "board_left": np.vstack(left_pts).astype(np.float64),
        "board_right": np.vstack(right_pts).astype(np.float64),
    }
    return _TRUTH


def undistorted_pair(index: int):
    """One left/right pair with the lens model removed, so F is a pinhole F."""
    t = calibration_truth()
    left = cv2.undistort(cal.load_view("left", index), t["K1"], t["d1"], None, t["K1"])
    right = cv2.undistort(cal.load_view("right", index), t["K2"], t["d2"], None,
                          t["K2"])
    return left, right


# --------------------------------------------------------------------------- #
# correspondences
# --------------------------------------------------------------------------- #

#: Lowe's ratio. 0.75 is the usual value and is kept rather than tuned, because
#: tuning it per estimator would make the comparison one of match quality.
RATIO = 0.75


def match_features(index: int, detector: str = "SIFT", max_features: int = 3000):
    """Ratio-tested matches between one undistorted pair.

    Returns ``(points_left, points_right)`` as Nx2 float arrays.
    """
    left, right = undistorted_pair(index)
    if detector == "SIFT":
        det = cv2.SIFT_create(max_features)
        norm = cv2.NORM_L2
    elif detector == "ORB":
        det = cv2.ORB_create(max_features)
        norm = cv2.NORM_HAMMING
    elif detector == "AKAZE":
        det = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(detector)

    k1, d1 = det.detectAndCompute(left, None)
    k2, d2 = det.detectAndCompute(right, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]
    p1 = np.array([k1[m.queryIdx].pt for m in good], np.float64)
    p2 = np.array([k2[m.trainIdx].pt for m in good], np.float64)
    return p1, p2


def board_hull(index: int, side: str = "left", pad: float = 12.0):
    """The convex hull of the board's corners, in the undistorted image."""
    t = calibration_truth()
    K, d = (t["K1"], t["d1"]) if side == "left" else (t["K2"], t["d2"])
    corners = cal.find_corners(side, index)
    if corners is None:
        return None
    pts = cv2.undistortPoints(corners, K, d, P=K).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts)
    centre = hull.reshape(-1, 2).mean(axis=0)
    grown = centre + (hull.reshape(-1, 2) - centre) * (
        1.0 + pad / max(np.linalg.norm(hull.reshape(-1, 2) - centre, axis=1).mean(), 1.0))
    return grown.astype(np.float32)


def board_share(index: int, detector: str = "SIFT") -> dict:
    """How much of the correspondence set sits on the chessboard.

    The number the planar-degeneracy argument needs. A matched pair counts as
    "on the board" when both of its points fall inside the board's hull.
    """
    p1, p2 = match_features(index, detector)
    on = _on_board(index, p1, p2)
    return {
        "view": int(index),
        "matches": int(len(p1)),
        "on_board": int(on.sum()),
        "share_on_board": round(float(on.mean()), 4) if len(p1) else 0.0,
    }


def _on_board(index: int, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    left_hull = board_hull(index, "left")
    right_hull = board_hull(index, "right")
    if left_hull is None or right_hull is None or not len(p1):
        return np.zeros(len(p1), bool)
    inside = np.zeros(len(p1), bool)
    for i, (a, b) in enumerate(zip(p1, p2)):
        in_left = cv2.pointPolygonTest(left_hull, (float(a[0]), float(a[1])), False) >= 0
        in_right = cv2.pointPolygonTest(right_hull, (float(b[0]), float(b[1])), False) >= 0
        inside[i] = in_left and in_right
    return inside


def correspondences(index: int, detector: str = "SIFT", exclude_board: bool = False):
    """Scene matches for one pair, optionally with the chessboard removed."""
    p1, p2 = match_features(index, detector)
    if exclude_board and len(p1):
        keep = ~_on_board(index, p1, p2)
        return p1[keep], p2[keep]
    return p1, p2


# --------------------------------------------------------------------------- #
# the estimators
# --------------------------------------------------------------------------- #


def _normalise(points: np.ndarray):
    """Hartley's isotropic normalisation: centroid at the origin, mean distance sqrt(2).

    This is the whole argument of Hartley 1997. Image coordinates run to 640, so
    the entries of the 9-vector the eight-point algorithm solves for differ by
    ~10^4 in magnitude, and the smallest singular vector of that matrix is decided
    by rounding rather than by geometry.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    scale = np.sqrt(2.0) / max(float(np.mean(np.linalg.norm(centred, axis=1))), EPS)
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1.0]])
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (T @ homogeneous.T).T[:, :2], T


def eight_point_raw(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """The eight-point algorithm on raw pixel coordinates. Written out, not called.

    Implemented here rather than through OpenCV because OpenCV **always**
    normalises internally, so the un-normalised version — the one the textbooks
    warn about — cannot be obtained from it. Showing what it costs requires
    writing it.
    """
    return _eight_point(p1, p2, normalise=False)


def eight_point_normalised(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """The same code with Hartley's normalisation switched on."""
    return _eight_point(p1, p2, normalise=True)


def _eight_point(p1: np.ndarray, p2: np.ndarray, normalise: bool) -> np.ndarray:
    if len(p1) < 8:
        raise ValueError(f"{len(p1)} correspondences; the eight-point algorithm needs 8")
    if normalise:
        q1, T1 = _normalise(p1)
        q2, T2 = _normalise(p2)
    else:
        q1, q2 = p1, p2
        T1 = T2 = np.eye(3)

    x1, y1 = q1[:, 0], q1[:, 1]
    x2, y2 = q2[:, 0], q2[:, 1]
    A = np.column_stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1,
                         np.ones(len(q1))])
    _, _, vt = np.linalg.svd(A)
    F = vt[-1].reshape(3, 3)

    # enforce rank 2: a fundamental matrix is singular by construction, and the
    # linear solution never is
    u, s, vt2 = np.linalg.svd(F)
    s[2] = 0.0
    F = u @ np.diag(s) @ vt2

    F = T2.T @ F @ T1
    return F / (F[2, 2] if abs(F[2, 2]) > EPS else np.linalg.norm(F))


def opencv_ransac(p1, p2, threshold: float = 1.0):
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, threshold, 0.999)
    return _normalised_or_none(F), mask


def opencv_lmeds(p1, p2, threshold: float = 1.0):
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS, threshold, 0.999)
    return _normalised_or_none(F), mask


def opencv_seven_point(p1, p2, threshold: float = 1.0):
    """The seven-point algorithm on the seven best-spread correspondences.

    It can return up to three matrices; the first is taken, which is a stated
    simplification rather than a claim that it is the right one.
    """
    idx = _spread_subset(p1, 7)
    F, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_7POINT)
    if F is None:
        return None, None
    return _normalised_or_none(F[:3]), None


def random_eight(p1, p2, threshold: float = 1.0, seed: int = 0):
    """Eight correspondences chosen at random. The control."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p1), size=8, replace=False)
    return eight_point_normalised(p1[idx], p2[idx]), None


def assume_rectified(p1, p2, threshold: float = 1.0):
    """No estimation at all: assume corresponding points have the same y.

    Zero parameters. This is not a meaningless matrix -- it is the assumption a
    rectified stereo rig satisfies exactly, and this rig nearly does. It is in the
    table to say how much of the answer is available before any estimator runs.
    """
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), None


def assume_rectified_offset(p1, p2, threshold: float = 1.0):
    """The same assumption with **one** parameter: a constant vertical offset.

    The epipolar line for ``(x1, y1)`` is ``y2 = y1 + c``, and ``c`` is the median
    vertical disparity of the correspondences -- estimated from the matches, like
    every other row of the table, and never from the board.

    One number, fitted by taking a median. If it beats a seven-parameter
    projective estimate, the estimate was not earning its parameters.
    """
    c = float(np.median(p2[:, 1] - p1[:, 1])) if len(p1) else 0.0
    return np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [0.0, -1.0, -c]]), None


def _spread_subset(points: np.ndarray, n: int) -> np.ndarray:
    """Pick n points spread over the image rather than n adjacent ones.

    Seven correspondences from one corner of the frame determine nothing useful,
    and picking the first seven would measure detector ordering.
    """
    chosen = [int(np.argmin(points[:, 0] + points[:, 1]))]
    for _ in range(n - 1):
        d = np.min(np.linalg.norm(points[:, None, :] - points[chosen][None, :, :],
                                  axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return np.array(chosen)


def _normalised_or_none(F):
    if F is None or F.shape[0] < 3:
        return None
    F = np.asarray(F[:3], np.float64)
    return F / (F[2, 2] if abs(F[2, 2]) > EPS else np.linalg.norm(F))


ESTIMATORS = {
    "8-point, raw pixels": lambda p1, p2, t: (eight_point_raw(p1, p2), None),
    "8-point, normalised (Hartley)": lambda p1, p2, t: (
        eight_point_normalised(p1, p2), None),
    "7-point": opencv_seven_point,
    "LMedS": opencv_lmeds,
    "RANSAC": opencv_ransac,
    "Random 8 matches (control)": lambda p1, p2, t: random_eight(p1, p2, t),
    "Assume rectified (0 parameters, control)": assume_rectified,
    "Assume rectified + offset (1 parameter, control)": assume_rectified_offset,
}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def symmetric_epipolar_distance(F, p1: np.ndarray, p2: np.ndarray) -> float:
    """Mean distance, in pixels, from each point to the other's epipolar line.

    Symmetric because F is not: a matrix can put every left point close to its
    line in the right image and be badly wrong the other way round.
    """
    if F is None or not len(p1):
        return float("nan")
    h1 = np.hstack([p1, np.ones((len(p1), 1))])
    h2 = np.hstack([p2, np.ones((len(p2), 1))])
    l2 = (F @ h1.T).T
    l1 = (F.T @ h2.T).T
    numerator = np.abs(np.sum(h2 * l2, axis=1))
    d2 = numerator / np.maximum(np.hypot(l2[:, 0], l2[:, 1]), EPS)
    d1 = numerator / np.maximum(np.hypot(l1[:, 0], l1[:, 1]), EPS)
    return float(np.mean(0.5 * (d1 + d2)))


def epipole(F) -> np.ndarray | None:
    """The right epipole: the null vector of F, in pixels."""
    if F is None:
        return None
    _, _, vt = np.linalg.svd(F)
    e = vt[-1]
    if abs(e[2]) < 1e-10:
        return None  # at infinity, which is a real answer for a parallel rig
    return e[:2] / e[2]


def board_error(F) -> float:
    """The headline metric: how well F explains the 702 chessboard corners.

    Those corners are what the *truth* was built from and what **no estimator
    here ever sees**, provided the board is excluded from the correspondences.
    """
    t = calibration_truth()
    return symmetric_epipolar_distance(F, t["board_left"], t["board_right"])


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def evaluate(views=None, detector: str = "SIFT", exclude_board: bool = True,
             threshold: float = 1.0) -> list[dict]:
    """Every estimator on every pair, scored on the board corners it never saw."""
    views = views or usable_pairs()
    truth = calibration_truth()
    truth_epipole = epipole(truth["F"])

    acc = {name: {"board_px": [], "match_px": [], "epipole_px": [], "inliers": [],
                  "failures": 0} for name in ESTIMATORS}

    for index in views:
        p1, p2 = correspondences(index, detector, exclude_board)
        if len(p1) < 8:
            for name in ESTIMATORS:
                acc[name]["failures"] += 1
            continue
        for name, fn in ESTIMATORS.items():
            try:
                F, mask = fn(p1, p2, threshold)
            except Exception:  # noqa: BLE001
                F, mask = None, None
            if F is None or not np.all(np.isfinite(F)):
                acc[name]["failures"] += 1
                continue
            acc[name]["board_px"].append(board_error(F))
            acc[name]["match_px"].append(symmetric_epipolar_distance(F, p1, p2))
            e = epipole(F)
            if e is not None and truth_epipole is not None:
                acc[name]["epipole_px"].append(
                    float(np.linalg.norm(e - truth_epipole)))
            acc[name]["inliers"].append(
                int(mask.sum()) if mask is not None else len(p1))

    def summarise(name, a):
        return {
            "estimator": name,
            "board_px": round(float(np.mean(a["board_px"])), 4) if a["board_px"]
            else float("nan"),
            "board_median_px": round(float(np.median(a["board_px"])), 4)
            if a["board_px"] else float("nan"),
            "match_px": round(float(np.mean(a["match_px"])), 4) if a["match_px"]
            else float("nan"),
            "epipole_px": round(float(np.median(a["epipole_px"])), 2)
            if a["epipole_px"] else float("nan"),
            "inliers": round(float(np.mean(a["inliers"])), 1) if a["inliers"] else 0.0,
            "failures": a["failures"],
        }

    rows = [summarise(n, a) for n, a in acc.items()]
    rows.append({
        "estimator": "Calibration (truth)",
        "board_px": round(board_error(truth["F"]), 4),
        "board_median_px": round(board_error(truth["F"]), 4),
        "match_px": float("nan"),
        "epipole_px": 0.0,
        "inliers": float("nan"),
        "failures": 0,
    })
    return rows


def normalisation_effect(views=None, detector: str = "SIFT") -> list[dict]:
    """The same eight-point code with normalisation on and off, pair by pair.

    Written as its own experiment because it is the project's headline and a mean
    over thirteen views hides how large the per-view ratio gets.
    """
    views = views or usable_pairs()
    rows = []
    for index in views:
        p1, p2 = correspondences(index, detector, exclude_board=True)
        if len(p1) < 8:
            continue
        raw = board_error(eight_point_raw(p1, p2))
        norm = board_error(eight_point_normalised(p1, p2))
        rows.append({
            "view": int(index),
            "matches": int(len(p1)),
            "raw_px": round(raw, 3),
            "normalised_px": round(norm, 3),
            "ratio": round(raw / max(norm, EPS), 1),
        })
    return rows


def planar_degeneracy(views=None, detector: str = "SIFT") -> list[dict]:
    """F from board matches only, against F from everything but the board.

    The chessboard is one plane. A correspondence set confined to a plane does
    not determine F, so this is not a hard case — it is an ill-posed one, and the
    estimator has no way to say so.
    """
    views = views or usable_pairs()
    rows = []
    for index in views:
        p1, p2 = match_features(index, detector)
        if len(p1) < 16:
            continue
        on = _on_board(index, p1, p2)
        row = {"view": int(index), "on_board": int(on.sum()),
               "off_board": int((~on).sum())}
        for label, keep in (("board only", on), ("board excluded", ~on),
                            ("everything", np.ones(len(p1), bool))):
            if keep.sum() < 8:
                row[label] = float("nan")
                continue
            F, mask = opencv_ransac(p1[keep], p2[keep], 1.0)
            row[label] = round(board_error(F), 3) if F is not None else float("nan")
            if label == "board only":
                row["board_only_inlier_rate"] = (
                    round(float(mask.mean()), 3) if mask is not None else float("nan"))
        rows.append(row)
    return rows


def inliers_are_not_quality(views=None, detector: str = "SIFT",
                            thresholds=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)) -> list[dict]:
    """RANSAC's inlier count against the error it never measures.

    The inlier count is what a pipeline logs and what a person reads as
    confidence. It is a function of the threshold, and the threshold is a
    parameter — so it can be made to look like anything.
    """
    views = views or usable_pairs()
    rows = []
    for t in thresholds:
        errors, rates = [], []
        for index in views:
            p1, p2 = correspondences(index, detector, exclude_board=True)
            if len(p1) < 8:
                continue
            F, mask = opencv_ransac(p1, p2, t)
            if F is None:
                continue
            errors.append(board_error(F))
            rates.append(float(mask.mean()) if mask is not None else float("nan"))
        rows.append({
            "threshold_px": t,
            "inlier_rate": round(float(np.mean(rates)), 4),
            "board_px": round(float(np.mean(errors)), 4),
        })
    return rows


def detector_comparison(views=None, detectors=("SIFT", "ORB", "AKAZE")) -> list[dict]:
    """Which feature detector gives RANSAC the best correspondences here."""
    views = views or usable_pairs()
    rows = []
    for d in detectors:
        counts, errors = [], []
        for index in views:
            p1, p2 = correspondences(index, d, exclude_board=True)
            counts.append(len(p1))
            if len(p1) < 8:
                continue
            F, _ = opencv_ransac(p1, p2, 1.0)
            if F is not None:
                errors.append(board_error(F))
        rows.append({
            "detector": d,
            "matches": round(float(np.mean(counts)), 1),
            "board_px": round(float(np.mean(errors)), 4) if errors else float("nan"),
            "board_median_px": round(float(np.median(errors)), 4) if errors
            else float("nan"),
        })
    return rows


def epipolar_lines(F, points: np.ndarray, shape) -> list[tuple]:
    """Endpoints of each point's epipolar line, clipped to the image."""
    h, w = shape[:2]
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    lines = (F @ homogeneous.T).T
    out = []
    for a, b, c in lines:
        if abs(b) < EPS:
            continue
        out.append(((0, int(-c / b)), (w, int(-(c + a * w) / b))))
    return out
