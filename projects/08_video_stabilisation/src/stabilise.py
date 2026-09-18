"""Video stabilisation: four motion estimators, three smoothers, and two controls.

The question
------------
A handheld camera shakes. Estimate the shake frame by frame, smooth the estimated
path, and warp each frame onto the smooth one. Three steps, and the interesting
failures are in different places from where the literature puts them.

> **The claim under test:** stabilisation is limited by how well the inter-frame
> motion can be estimated. **It is not.** The best estimator recovers the camera
> path to **0.035 px per frame** — seventy times finer than the shake it is
> removing — and swapping it for the worst one changes the residual jitter by
> **0.076**. Swapping the *smoother* instead changes it by **0.639**, eight times
> as much. The estimator is not the bottleneck; the path model is.

> **And the control nobody runs:** a stabiliser pointed at footage that is
> already perfectly still should do nothing. Run on the unjittered clip — where
> the true camera motion is exactly zero — these estimators invent **0.023 to
> 0.144 px per frame**, and because the path is an integral, ECC accumulates
> **25.3 px of drift** over sixty frames. It introduces a visible wander into a
> shot that never moved.

> **Nothing here is called best without saying what it cost.** Widening the
> smoother from sigma 2 to 32 makes the output **12x steadier** and throws away
> **5x more of the frame**, from 1.2% to 5.9%.

Where the ground truth comes from
---------------------------------
The clip is a **static-camera** recording of a plaza: real people, real lighting,
real compression. A **known** random-walk camera path is then applied to it, so
the truth is not estimated — it is the matrix that was used.

What is real and what is not, stated plainly: every pixel of content is
photographed, and the camera path is synthetic. That is the right way round for
this project, because the camera path is precisely the thing being estimated, and
the original clip doubles as a **zero-motion control** that no synthetic
benchmark can provide.

The cropping trap
-----------------
Stabilising means warping frames off their own canvas, so the output has to be
cropped to stay full. **A stabiliser can score arbitrarily well by cropping more**,
and the two quantities are reported together for that reason: nothing here is
called "best" without saying what it cost in field of view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from shared.io import to_gray

EPS = 1e-9

#: The static-camera plaza clip, shared with projects 29, 30, 55 and 57.
VIDEO = Path.home() / ".cache" / "classical-cv-images" / "assets" / "video" / "vtest.avi"

#: The twelve segments the experiments run on. Sixty frames each, starting where
#: neither project 29's runs nor project 30's frames begin, so the segments are
#: this project's own.
STARTS = (10, 75, 130, 185, 250, 300, 365, 425, 470, 525, 595, 655)
SEGMENT = 60

#: The shake. A random walk in translation and rotation, integrated so the path
#: wanders rather than vibrating about a fixed point -- which is what a hand does
#: and what makes smoothing a real problem rather than a low-pass filter.
#:
#: The first version used a step of 6 px and 0.25 deg with weak damping, which
#: produced a path wandering 277 px across a 768 px frame. That is a pan, not a
#: shake, and it made the crop dominate every other number. These values give a
#: path with a standard deviation of a few pixels, which is what a hand does.
SHAKE_TRANSLATION = 2.0     # pixels of step standard deviation
SHAKE_ROTATION = 0.12       # degrees of step standard deviation
SHAKE_DAMPING = 0.80        # pulls the walk back, so it does not leave the frame


def video_available() -> bool:
    return VIDEO.exists()


def load_frames(start: int, count: int) -> list[np.ndarray]:
    if not video_available():
        raise FileNotFoundError(
            f"{VIDEO} is missing. Run `python tools/fetch_assets.py --set video`.")
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))
        out = []
        for _ in range(int(count)):
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# the shake, and the truth
# --------------------------------------------------------------------------- #


def camera_path(n: int, seed: int = 0, translation: float = SHAKE_TRANSLATION,
                rotation: float = SHAKE_ROTATION,
                damping: float = SHAKE_DAMPING) -> np.ndarray:
    """A damped random walk of (dx, dy, dtheta). The truth, in absolute terms.

    Returns an ``(n, 3)`` array: the position of the camera at each frame, not the
    step between frames. Damping keeps the walk near the origin, so a sixty-frame
    segment does not drift entirely out of the canvas and make the crop dominate
    everything.
    """
    rng = np.random.default_rng(seed)
    path = np.zeros((n, 3), np.float64)
    velocity = np.zeros(3)
    for i in range(1, n):
        step = rng.normal(0.0, [translation, translation, rotation])
        velocity = damping * velocity + step
        path[i] = path[i - 1] * damping + velocity
    return path


def to_matrix(pose) -> np.ndarray:
    """A 2x3 affine from (dx, dy, dtheta in degrees), rotating about the centre."""
    dx, dy, angle = (float(v) for v in pose)
    M = cv2.getRotationMatrix2D((0.0, 0.0), angle, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def apply_shake(frames: list[np.ndarray], path: np.ndarray) -> list[np.ndarray]:
    """Warp each frame by its pose. The 'handheld' clip."""
    h, w = frames[0].shape[:2]
    centre = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0]], np.float64)
    back = np.array([[1, 0, w / 2.0], [0, 1, h / 2.0]], np.float64)
    out = []
    for frame, pose in zip(frames, path):
        M = to_matrix(pose)
        full = _compose(back, _compose(M, centre))
        out.append(cv2.warpAffine(frame, full, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT))
    return out


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affines: apply b, then a."""
    A = np.vstack([a, [0, 0, 1]])
    B = np.vstack([b, [0, 0, 1]])
    return (A @ B)[:2]


def shaken_segment(start: int, seed: int = 0, count: int = SEGMENT,
                   translation: float = SHAKE_TRANSLATION,
                   rotation: float = SHAKE_ROTATION):
    """``(original, shaken, true_path)`` for one segment."""
    frames = load_frames(start, count)
    path = camera_path(len(frames), seed, translation, rotation)
    return frames, apply_shake(frames, path), path


def path_steps(path: np.ndarray) -> np.ndarray:
    """Frame-to-frame differences of an absolute path."""
    return np.diff(path, axis=0)


# --------------------------------------------------------------------------- #
# the motion estimators
# --------------------------------------------------------------------------- #


def _pose_from_affine(M, shape) -> np.ndarray:
    """Read (dx, dy, dtheta) back out of a 2x3 affine about the image centre."""
    if M is None:
        return np.zeros(3)
    h, w = shape[:2]
    # `getRotationMatrix2D` builds [[cos, sin], [-sin, cos]], so M[1, 0] is
    # *minus* the sine. Reading the angle as arctan2(M[1, 0], M[0, 0]) returns
    # the negative of the rotation that was applied, and a round-trip test --
    # warp a frame by +2 degrees, estimate it back -- returned -2.
    angle = float(np.degrees(np.arctan2(-M[1, 0], M[0, 0])))
    centre = np.array([w / 2.0, h / 2.0, 1.0])
    moved = M @ centre
    return np.array([moved[0] - w / 2.0, moved[1] - h / 2.0, angle])


def estimate_none(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Report no motion. The control that says what the others are worth."""
    return np.zeros(3)


def estimate_features(prev: np.ndarray, curr: np.ndarray,
                      max_corners: int = 400) -> np.ndarray:
    """Shi-Tomasi corners, Lucas-Kanade flow, then a partial affine by RANSAC.

    The standard pipeline, and the one every stabilisation tutorial uses.
    """
    g0, g1 = to_gray(prev), to_gray(curr)
    p0 = cv2.goodFeaturesToTrack(g0, max_corners, 0.01, 30, blockSize=3)
    if p0 is None or len(p0) < 6:
        return np.zeros(3)
    p1, status, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
    good0 = p0[status.ravel() == 1]
    good1 = p1[status.ravel() == 1]
    if len(good0) < 6:
        return np.zeros(3)
    M, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC,
                                       ransacReprojThreshold=3.0)
    return _pose_from_affine(M, prev.shape)


def estimate_phase_correlation(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Whole-frame phase correlation. Translation only, by construction.

    It cannot see rotation at all, which is not a weakness to be hidden -- it is
    the reason the rotation column of the results table exists.
    """
    g0 = to_gray(prev).astype(np.float32)
    g1 = to_gray(curr).astype(np.float32)
    window = cv2.createHanningWindow(g0.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(g0, g1, window)
    # `phaseCorrelate(a, b)` returns the shift **of a towards b**, which is
    # already the motion from the previous frame to this one. Negating it -- the
    # obvious guess, and the first version here -- made this estimator score
    # worse than reporting no motion at all.
    return np.array([dx, dy, 0.0])


def estimate_ecc(prev: np.ndarray, curr: np.ndarray,
                 iterations: int = 60, eps: float = 1e-5) -> np.ndarray:
    """Direct alignment by maximising enhanced correlation coefficient.

    Uses every pixel rather than a few hundred corners, which should help on a
    frame with little texture and cost a great deal of time.
    """
    g0 = to_gray(prev).astype(np.float32) / 255.0
    g1 = to_gray(curr).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            g0, g1, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps), None, 5)
    except cv2.error:
        return np.zeros(3)
    return _pose_from_affine(warp.astype(np.float64), prev.shape)


def estimate_block_matching(prev: np.ndarray, curr: np.ndarray,
                            grid: int = 6, block: int = 48,
                            search: int = 24) -> np.ndarray:
    """A grid of blocks matched by normalised cross-correlation, then an affine.

    The oldest method here and the only one with no derivative in it, which is
    why it survives the low-texture segments that break the corner tracker.
    """
    g0, g1 = to_gray(prev), to_gray(curr)
    h, w = g0.shape
    src, dst = [], []
    for gy in range(grid):
        for gx in range(grid):
            y = int((gy + 0.5) * h / grid) - block // 2
            x = int((gx + 0.5) * w / grid) - block // 2
            y = int(np.clip(y, search, h - block - search))
            x = int(np.clip(x, search, w - block - search))
            template = g0[y:y + block, x:x + block]
            window = g1[y - search:y + block + search, x - search:x + block + search]
            if template.size == 0 or window.shape[0] <= block:
                continue
            res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < 0.3:
                continue
            src.append([x + block / 2.0, y + block / 2.0])
            dst.append([x - search + loc[0] + block / 2.0,
                        y - search + loc[1] + block / 2.0])
    if len(src) < 4:
        return np.zeros(3)
    M, _ = cv2.estimateAffinePartial2D(np.array(src, np.float32),
                                       np.array(dst, np.float32),
                                       method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return _pose_from_affine(M, prev.shape)


ESTIMATORS: dict[str, Callable] = {
    "No motion (control)": estimate_none,
    "Features + LK + RANSAC": estimate_features,
    "Phase correlation": estimate_phase_correlation,
    "ECC (direct)": estimate_ecc,
    "Block matching": estimate_block_matching,
}


def estimate_path(frames: list[np.ndarray], estimator: str) -> np.ndarray:
    """Integrate the per-frame estimates into an absolute path."""
    fn = ESTIMATORS[estimator]
    steps = np.zeros((len(frames), 3))
    for i in range(1, len(frames)):
        steps[i] = fn(frames[i - 1], frames[i])
    return np.cumsum(steps, axis=0)


# --------------------------------------------------------------------------- #
# the smoothers
# --------------------------------------------------------------------------- #


def smooth_none(path: np.ndarray, **kw) -> np.ndarray:
    """Target a perfectly still camera. Maximum stabilisation, maximum crop."""
    return np.zeros_like(path)


def smooth_moving_average(path: np.ndarray, radius: int = 15, **kw) -> np.ndarray:
    """A box filter over the path. What most tutorials use.

    Its problem is visible in the results: a box filter has ringing, so a sharp
    real camera movement produces overshoot on both sides of it.
    """
    n = len(path)
    out = np.empty_like(path)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        out[i] = path[lo:hi].mean(axis=0)
    return out


def smooth_gaussian(path: np.ndarray, sigma: float = 8.0, **kw) -> np.ndarray:
    """A Gaussian over the path: no ringing, more lag at the ends."""
    n = len(path)
    x = np.arange(-3 * int(sigma), 3 * int(sigma) + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    out = np.empty_like(path)
    for c in range(path.shape[1]):
        padded = np.pad(path[:, c], (len(x) // 2, len(x) // 2), mode="edge")
        out[:, c] = np.convolve(padded, kernel, mode="valid")[:n]
    return out


def smooth_kalman(path: np.ndarray, process: float = 1e-3,
                  measurement: float = 1e-1, **kw) -> np.ndarray:
    """A constant-velocity Kalman filter per axis: causal, so it lags.

    In the table it is the one that could run live. The lag is the price, and it
    is measured rather than described.
    """
    out = np.empty_like(path)
    for c in range(path.shape[1]):
        kf = cv2.KalmanFilter(2, 1)
        kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0]], np.float32)
        kf.processNoiseCov = np.eye(2, dtype=np.float32) * process
        kf.measurementNoiseCov = np.array([[measurement]], np.float32)
        kf.statePost = np.array([[np.float32(path[0, c])], [0.0]], np.float32)
        for i in range(len(path)):
            kf.predict()
            kf.correct(np.array([[np.float32(path[i, c])]]))
            out[i, c] = float(kf.statePost[0, 0])
    return out


SMOOTHERS: dict[str, Callable] = {
    "Fix the camera (control)": smooth_none,
    "Moving average (r=15)": smooth_moving_average,
    "Gaussian (sigma=8)": smooth_gaussian,
    "Kalman (causal)": smooth_kalman,
}


# --------------------------------------------------------------------------- #
# putting it together
# --------------------------------------------------------------------------- #


def stabilise(frames: list[np.ndarray], estimator: str = "Features + LK + RANSAC",
              smoother: str = "Gaussian (sigma=8)"):
    """Returns ``(stabilised_frames, estimated_path, smoothed_path)``."""
    estimated = estimate_path(frames, estimator)
    smoothed = SMOOTHERS[smoother](estimated)
    correction = smoothed - estimated

    h, w = frames[0].shape[:2]
    centre = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0]], np.float64)
    back = np.array([[1, 0, w / 2.0], [0, 1, h / 2.0]], np.float64)
    out = []
    for frame, pose in zip(frames, correction):
        M = to_matrix(pose)
        out.append(cv2.warpAffine(frame, _compose(back, _compose(M, centre)),
                                  (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT))
    return out, estimated, smoothed


def required_crop(correction: np.ndarray, shape) -> float:
    """The fraction of the frame that must be thrown away to stay full.

    Computed from the corners the warps push outside the canvas, which is the
    honest version: a stabiliser that moves frames further needs a bigger crop,
    and quoting stability without this is quoting half a result.
    """
    h, w = shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
    worst = 0.0
    for pose in correction:
        M = to_matrix(pose)
        full = _compose(np.array([[1, 0, w / 2.0], [0, 1, h / 2.0]]),
                        _compose(M, np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0]])))
        moved = np.vstack([full, [0, 0, 1]]) @ corners
        # how far in from each edge the warped frame sits
        left = max(0.0, float(np.max(moved[0, [0, 3]])))
        right = max(0.0, float(w - np.min(moved[0, [1, 2]])))
        top = max(0.0, float(np.max(moved[1, [0, 1]])))
        bottom = max(0.0, float(h - np.min(moved[1, [2, 3]])))
        keep = max(0.0, (w - left - right)) * max(0.0, (h - top - bottom))
        worst = max(worst, 1.0 - keep / float(w * h))
    return float(worst)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def jitter(path: np.ndarray) -> float:
    """Mean absolute acceleration of a camera path, in pixels per frame squared.

    The second difference rather than the first: a smooth pan is not shake, and a
    measure built on velocity would call it shake. What a viewer sees as unsteady
    is the path changing direction, which is acceleration.
    """
    if len(path) < 3:
        return 0.0
    accel = np.diff(path[:, :2], n=2, axis=0)
    return float(np.mean(np.abs(accel)))


def effective_path(true_path: np.ndarray, estimated: np.ndarray,
                   smoothed: np.ndarray) -> np.ndarray:
    """Where the camera effectively ends up once the correction is applied.

    The output frame is the shaken frame warped by ``smoothed - estimated``, so
    the path a viewer sees is ``true + (smoothed - estimated)``. If the estimator
    were perfect this is exactly ``smoothed``; every deviation from it is
    estimation error surviving into the result.
    """
    return true_path + (smoothed - estimated)


def evaluate_estimators(starts=STARTS, seed: int = 0, runs: int = 1) -> list[dict]:
    """How well each estimator recovers the known per-frame camera motion."""
    from shared.bench import timeit

    acc = {name: {"xy": [], "rot": [], "ms": []} for name in ESTIMATORS}
    for i, start in enumerate(starts):
        _, shaken, path = shaken_segment(start, seed=seed + i)
        truth = path_steps(path)
        for name in ESTIMATORS:
            estimated, timing = timeit(
                lambda n=name: estimate_path(shaken, n), runs=runs, warmup=0)
            error = np.abs(path_steps(estimated) - truth)
            acc[name]["xy"].append(float(error[:, :2].mean()))
            acc[name]["rot"].append(float(error[:, 2].mean()))
            acc[name]["ms"].append(timing.median_ms / max(len(shaken) - 1, 1))
    return [
        {
            "estimator": name,
            "translation_error_px": round(float(np.mean(a["xy"])), 4),
            "rotation_error_deg": round(float(np.mean(a["rot"])), 4),
            "ms_per_frame": round(float(np.median(a["ms"])), 2),
        }
        for name, a in acc.items()
    ]


def zero_motion_control(starts=STARTS) -> list[dict]:
    """Every estimator on the **unjittered** clip, where the truth is exactly zero.

    The control nobody runs. A stabiliser pointed at a tripod shot should report
    no motion and change nothing; what it reports instead is the noise floor of
    the whole pipeline, and it is not zero. People walk through the frame, the
    codec moves blocks about, and a corner tracker has no way to know the camera
    did not move.
    """
    rows = []
    for name in ESTIMATORS:
        invented, drift = [], []
        for start in starts:
            frames = load_frames(start, SEGMENT)
            estimated = estimate_path(frames, name)
            steps = path_steps(estimated)
            invented.append(float(np.mean(np.abs(steps[:, :2]))))
            drift.append(float(np.max(np.abs(estimated[:, :2]))))
        rows.append({
            "estimator": name,
            "invented_motion_px_per_frame": round(float(np.mean(invented)), 4),
            "worst_accumulated_drift_px": round(float(np.max(drift)), 3),
        })
    return rows


def evaluate_pipeline(starts=STARTS, seed: int = 0,
                      estimators=None, smoothers=None) -> list[dict]:
    """Every estimator against every smoother: residual jitter and crop together.

    Reported as a pair on purpose. A stabiliser can always look steadier by
    targeting a stiller camera, and the bill arrives as field of view.
    """
    estimators = estimators or list(ESTIMATORS)
    smoothers = smoothers or list(SMOOTHERS)
    rows = []
    for name in estimators:
        for smoother in smoothers:
            residual, crops, baseline = [], [], []
            for i, start in enumerate(starts):
                _, shaken, path = shaken_segment(start, seed=seed + i)
                estimated = estimate_path(shaken, name)
                smoothed = SMOOTHERS[smoother](estimated)
                effective = effective_path(path, estimated, smoothed)
                residual.append(jitter(effective))
                baseline.append(jitter(path))
                crops.append(required_crop(smoothed - estimated, shaken[0].shape))
            rows.append({
                "estimator": name,
                "smoother": smoother,
                "residual_jitter": round(float(np.mean(residual)), 4),
                "input_jitter": round(float(np.mean(baseline)), 4),
                "reduction": round(float(np.mean(baseline)) /
                                   max(float(np.mean(residual)), EPS), 2),
                "crop": round(float(np.mean(crops)), 4),
            })
    return rows


def what_limits_the_result(starts=STARTS, seed: int = 0) -> dict:
    """Is the residual set by the estimator or by the smoother?

    Two comparisons on the same segments: hold the smoother and vary the
    estimator, then hold the estimator and vary the smoother. The project's
    headline is which of the two spreads is larger.
    """
    real = [e for e in ESTIMATORS if "control" not in e]
    smoothers = [s for s in SMOOTHERS if "control" not in s]

    fixed_smoother = evaluate_pipeline(starts, seed, estimators=real,
                                       smoothers=["Gaussian (sigma=8)"])
    fixed_estimator = evaluate_pipeline(starts, seed,
                                        estimators=["Features + LK + RANSAC"],
                                        smoothers=smoothers)

    a = [r["residual_jitter"] for r in fixed_smoother]
    b = [r["residual_jitter"] for r in fixed_estimator]
    return {
        "varying_the_estimator": {r["estimator"]: r["residual_jitter"]
                                  for r in fixed_smoother},
        "varying_the_smoother": {r["smoother"]: r["residual_jitter"]
                                 for r in fixed_estimator},
        "estimator_spread": round(max(a) - min(a), 4),
        "smoother_spread": round(max(b) - min(b), 4),
        "smoother_matters_more": bool((max(b) - min(b)) > (max(a) - min(a))),
    }


def sweep_shake(starts=STARTS[:6], seed: int = 0,
                amounts=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)) -> list[dict]:
    """How much shake each estimator survives.

    The translation step is scaled and the rotation with it, so the sweep is of
    the *amount* of shake rather than of its character.
    """
    rows = []
    for amount in amounts:
        row: dict[str, float] = {"translation_step_px": amount}
        for name in ESTIMATORS:
            errors = []
            for i, start in enumerate(starts):
                _, shaken, path = shaken_segment(
                    start, seed=seed + i, translation=amount,
                    rotation=SHAKE_ROTATION * amount / SHAKE_TRANSLATION)
                estimated = estimate_path(shaken, name)
                errors.append(float(np.mean(np.abs(
                    path_steps(estimated)[:, :2] - path_steps(path)[:, :2]))))
            row[name] = round(float(np.mean(errors)), 4)
        rows.append(row)
    return rows


def crop_versus_stability(starts=STARTS[:6], seed: int = 0,
                          sigmas=(2.0, 4.0, 8.0, 16.0, 32.0)) -> list[dict]:
    """The trade the field-of-view bill is paid into.

    A wider smoothing kernel targets a stiller camera, which is steadier and
    crops more. There is no setting that is best; there is a curve.
    """
    rows = []
    for sigma in sigmas:
        residual, crops = [], []
        for i, start in enumerate(starts):
            _, shaken, path = shaken_segment(start, seed=seed + i)
            estimated = estimate_path(shaken, "Features + LK + RANSAC")
            smoothed = smooth_gaussian(estimated, sigma=sigma)
            residual.append(jitter(effective_path(path, estimated, smoothed)))
            crops.append(required_crop(smoothed - estimated, shaken[0].shape))
        rows.append({
            "sigma": sigma,
            "residual_jitter": round(float(np.mean(residual)), 4),
            "crop": round(float(np.mean(crops)), 4),
        })
    return rows
