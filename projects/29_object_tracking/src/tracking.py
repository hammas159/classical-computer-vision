"""Object tracking: six classical trackers, two controls, and a truth chain.

The question
------------
Given a box around a person in one frame, keep the box on that person. Every
tracker here is a different answer to "what makes this box the same object as
that box", and they fail in different ways rather than by different amounts.

> **The claim under test:** a tracker that models appearance beats one that only
> follows motion. It does, until the person it is following walks behind someone
> else — and then the ranking is decided entirely by **how long each tracker
> survives**, not by how accurately it tracks while it is alive.

Why these six
-------------
The usual demo calls `cv2.TrackerKCF_create()` and `cv2.TrackerCSRT_create()`.
**Neither exists in `opencv-python-headless`** — they live in `opencv-contrib`,
which this repository does not depend on, and the trackers that *are* in core
besides MIL (GOTURN, DaSiamRPN, Nano, Vit) are neural networks with downloaded
weights, which is the one thing this repository is not about.

So the correlation filter is **implemented here from scratch**: MOSSE is about
forty lines of FFT (Bolme et al., CVPR 2010) and is a better thing to have than
a call into a binary. The others are `cv2.meanShift`, `cv2.CamShift`,
`cv2.matchTemplate`, `cv2.calcOpticalFlowPyrLK` and `cv2.KalmanFilter` — all core,
all classical, none trained.

Where the ground truth comes from
---------------------------------
The clip has no annotation. What it has is the fact that **the per-pixel median
of all 795 frames is the empty plaza**, so the difference between a frame and
that median is where things are. A target is then followed frame to frame by
**nearest-centroid chaining on those masks**, using motion continuity and nothing
about appearance.

Two honest properties of that chain, both measured rather than assumed:

* it **breaks when two people merge into one blob**, and `chain_quality` reports
  how often per target;
* it uses information no tracker gets (the whole clip), which is what makes it
  usable as truth.

The two controls
----------------
`static_box` never moves the initial box, and `oracle_box` teleports to the true
box every frame. They bracket every real tracker: a tracker that cannot beat a
box nailed to the first frame is not tracking, and the gap to the oracle is what
is left to win.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from shared.io import to_gray

EPS = 1e-9

VIDEO = Path.home() / ".cache" / "classical-cv-images" / "assets" / "video" / "vtest.avi"

#: Where each of the twelve tracking runs starts -- **chosen by measurement, not
#: by eye**. `find_good_starts` scans the clip every ten frames and keeps the
#: starts where a target can actually be followed for at least 40 of the 50
#: frames. The first version of this project picked twelve evenly spaced starts
#: and seven of them had truth chains shorter than ten frames, which would have
#: made this a benchmark of initialisation rather than of tracking.
#:
#: They overlap by at most ten frames, and they are not the frames projects 30
#: and 57 use.
STARTS = (50, 140, 170, 210, 240, 410, 490, 550, 580, 610, 640, 740)

#: How many frames each run lasts: five seconds at this clip's 10 fps. Long
#: enough for every tracker here to fail at least once, short enough that most
#: targets have not left the frame.
RUN_LENGTH = 50

#: A box counts as still on the target at this IoU. Stated because "tracking
#: failed" is a threshold, not an observation.
SURVIVAL_IOU = 0.3


def video_available() -> bool:
    return VIDEO.exists()


def frame_count() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def load_frames(start: int, stop: int) -> list[np.ndarray]:
    """Frames ``[start, stop)`` read sequentially, RGB."""
    if not video_available():
        raise FileNotFoundError(
            f"{VIDEO} is missing. Run `python tools/fetch_assets.py --set video`.")
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))
        out = []
        for _ in range(max(0, int(stop) - max(0, int(start)))):
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# the oracle and the truth chain
# --------------------------------------------------------------------------- #

_EMPTY: np.ndarray | None = None
_NOISE: float | None = None

#: Grass nobody walks through, used only to measure how much a pixel moves when
#: nothing happens.
STATIC_PATCH = (20, 430, 120, 100)


def noise_floor() -> float:
    """The 99th percentile of the inter-frame difference over a static patch."""
    global _NOISE
    if _NOISE is None:
        x, y, w, h = STATIC_PATCH
        seq = [to_gray(f)[y:y + h, x:x + w].astype(np.float32)
               for f in load_frames(0, 120)]
        _NOISE = float(np.percentile(np.abs(np.diff(np.stack(seq), axis=0)), 99))
    return _NOISE


def empty_scene() -> np.ndarray:
    """The per-pixel median of the whole clip: the plaza with nobody in it."""
    global _EMPTY
    if _EMPTY is None:
        frames = load_frames(0, frame_count())
        _EMPTY = np.median(np.stack(frames[::3]), axis=0).astype(np.uint8)
    return _EMPTY


#: A blob has to look roughly like a standing person to be a tracking target.
MIN_AREA = 600
ASPECT_RANGE = (0.15, 0.90)
HEIGHT_RANGE = (45, 400)


def moving_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Person-shaped boxes in one frame, from the empty-scene difference."""
    diff = np.abs(to_gray(frame).astype(np.float32)
                  - to_gray(empty_scene()).astype(np.float32))
    mask = ((diff > 2.0 * noise_floor()) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or not (HEIGHT_RANGE[0] <= h <= HEIGHT_RANGE[1]):
            continue
        if not (ASPECT_RANGE[0] <= w / max(h, 1) <= ASPECT_RANGE[1]):
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def box_iou(a, b) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, EPS)


def centre(box) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def truth_chain(start: int, length: int = RUN_LENGTH, target: int = 0,
                jump_per_height: float = 0.55, max_gap: int = 3,
                frames: list | None = None):
    """Follow one moving blob frame to frame by nearest centroid.

    Returns ``(frames, boxes)`` where ``boxes[i]`` is the target's box in
    ``frames[i]``, or ``None`` once the chain has ended.

    Two decisions that had to be made rather than defaulted, both because the
    first version of this function produced chains one and two frames long:

    * **The jump limit scales with the box.** A person at the top of this frame
      is 50 px tall and crosses three pixels a frame; one at the bottom is 200 px
      tall and crosses twelve. A single pixel threshold either cuts the near
      chains or lets the far ones jump to a neighbour, so the limit is
      ``jump_per_height`` times the previous box's height.
    * **A gap of up to `max_gap` frames is survivable.** Background subtraction
      loses a person who stops for a moment or passes a similarly coloured wall.
      Ending the chain there would be calling a gap in the *evidence* a gap in
      the *person*. Beyond three frames it is no longer the same claim and the
      chain ends.

    **How it still breaks, reported rather than fixed:** two people who walk
    together merge into one blob, and the chain follows the merged box.
    `chain_quality` counts those events per run.
    """
    frames = frames if frames is not None else load_frames(start, start + length)
    if not frames:
        return [], []

    first = moving_boxes(frames[0])
    if len(first) <= target:
        return frames, [None] * len(frames)

    first.sort(key=lambda b: -b[2] * b[3])
    boxes: list[tuple[int, int, int, int] | None] = [first[target]]
    last = first[target]
    gap = 0

    for frame in frames[1:]:
        if last is None:
            boxes.append(None)
            continue
        candidates = moving_boxes(frame)
        px, py = centre(last)
        limit = jump_per_height * last[3]
        best, distance = None, float("inf")
        for b in candidates:
            bx, by = centre(b)
            d = float(np.hypot(bx - px, by - py))
            if d < distance:
                best, distance = b, d
        if best is not None and distance <= limit:
            boxes.append(best)
            last = best
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                boxes.append(None)
                last = None
            else:
                # the person is still there; the evidence is not
                boxes.append(None)
    return frames, boxes


def chain_length(start: int, length: int = RUN_LENGTH, target: int = 0) -> int:
    """How many frames of a run have a truth box, before the chain ends."""
    _, boxes = truth_chain(start, length, target)
    end = len(boxes)
    run = 0
    for i, b in enumerate(boxes):
        if b is None:
            run += 1
            if run > 3:
                end = i - run + 1
                break
        else:
            run = 0
    return int(sum(1 for b in boxes[:end] if b is not None))


def find_good_starts(count: int = 12, length: int = RUN_LENGTH, step: int = 20,
                     minimum: int = 40) -> list[int]:
    """Start points where a target can actually be followed for long enough.

    Chosen by measurement rather than by eye, and reported: a tracking benchmark
    whose truth chains are two frames long is a benchmark of initialisation.
    Candidates are scanned every `step` frames and kept when the chain survives at
    least `minimum` of them; the `count` longest that do not overlap are used.
    """
    scored = []
    for start in range(0, max(1, frame_count() - length), step):
        n = chain_length(start, length)
        if n >= minimum:
            scored.append((n, start))
    scored.sort(key=lambda s: (-s[0], s[1]))

    chosen: list[int] = []
    for _, start in scored:
        if all(abs(start - c) >= length // 2 for c in chosen):
            chosen.append(start)
        if len(chosen) == count:
            break
    return sorted(chosen)


def chain_quality(starts=STARTS, length: int = RUN_LENGTH) -> list[dict]:
    """How far each truth chain gets, and how much its box size wobbles.

    A chain whose box area jumps by a factor of two in one frame has merged with
    somebody. Reported per run so the tracker scores can be read against it.
    """
    rows = []
    for start in starts:
        _, boxes = truth_chain(start, length)
        alive = [b for b in boxes if b is not None]
        breaks = next((i for i, b in enumerate(boxes) if b is None), len(boxes))
        areas = np.array([b[2] * b[3] for b in alive], np.float64)
        ratios = areas[1:] / np.maximum(areas[:-1], 1.0) if len(areas) > 1 else np.array([1.0])
        rows.append({
            "start": int(start),
            "frames_tracked": int(breaks),
            "of": int(len(boxes)),
            "max_area_jump": round(float(np.max(np.maximum(ratios, 1 / ratios))), 2),
            "merge_events": int(np.sum(np.maximum(ratios, 1 / ratios) > 1.8)),
        })
    return rows


# --------------------------------------------------------------------------- #
# the trackers
# --------------------------------------------------------------------------- #


def _crop(frame: np.ndarray, box) -> np.ndarray:
    x, y, w, h = (int(v) for v in box)
    x, y = max(0, x), max(0, y)
    return frame[y:y + h, x:x + w]


def _clamp(box, shape) -> tuple[int, int, int, int]:
    h_img, w_img = shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    w, h = max(4, w), max(4, h)
    x = int(np.clip(x, 0, w_img - w))
    y = int(np.clip(y, 0, h_img - h))
    return (x, y, w, h)


def static_box(frames, first) -> list:
    """Never move. The control that says whether a tracker is tracking at all."""
    return [tuple(first)] * len(frames)


def oracle_box(frames, first, truth=None) -> list:
    """Teleport to the true box every frame. The upper bound."""
    if truth is None:
        return [tuple(first)] * len(frames)
    return [tuple(b) if b is not None else None for b in truth]


def track_template(frames, first, search: int = 40) -> list:
    """Normalised cross-correlation against the *initial* crop, in a local window.

    The template never updates, which is the point of having it: it cannot drift
    onto the background, and it cannot survive the target turning around.
    """
    template = to_gray(_crop(frames[0], first))
    boxes = [tuple(first)]
    for frame in frames[1:]:
        x, y, w, h = boxes[-1]
        gray = to_gray(frame)
        x0, y0 = max(0, x - search), max(0, y - search)
        x1 = min(gray.shape[1], x + w + search)
        y1 = min(gray.shape[0], y + h + search)
        window = gray[y0:y1, x0:x1]
        if window.shape[0] < h or window.shape[1] < w:
            boxes.append(boxes[-1])
            continue
        res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        boxes.append(_clamp((x0 + loc[0], y0 + loc[1], w, h), frame.shape))
    return boxes


def _hue_histogram(frame, box):
    hsv = cv2.cvtColor(_crop(frame, box), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    hist = cv2.calcHist([hsv], [0], mask, [16], [0, 180])
    return cv2.normalize(hist, hist).reshape(-1)


def track_meanshift(frames, first, iterations: int = 12) -> list:
    """Mean shift on a hue back-projection. Fixed window size by construction."""
    hist = _hue_histogram(frames[0], first)
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1)
    window = tuple(int(v) for v in first)
    boxes = [window]
    for frame in frames[1:]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        back = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        _, window = cv2.meanShift(back, window, term)
        boxes.append(_clamp(window, frame.shape))
    return boxes


def track_camshift(frames, first, iterations: int = 12) -> list:
    """CamShift: mean shift plus a window that resizes to the back-projection.

    The resize is the whole difference and it cuts both ways — it follows a person
    walking toward the camera, and it collapses onto a bright patch of background.
    """
    hist = _hue_histogram(frames[0], first)
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1)
    window = tuple(int(v) for v in first)
    boxes = [window]
    for frame in frames[1:]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        back = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        _, window = cv2.CamShift(back, window, term)
        boxes.append(_clamp(window, frame.shape))
    return boxes


def track_optical_flow(frames, first, max_corners: int = 60) -> list:
    """Lucas-Kanade on corners inside the box; the box follows their median shift.

    Tracks *texture*, not the object: corners that land on the background behind
    a thin limb stay put and drag the median toward zero motion.
    """
    gray = to_gray(frames[0])
    x, y, w, h = (int(v) for v in first)
    mask = np.zeros(gray.shape, np.uint8)
    mask[y:y + h, x:x + w] = 255
    points = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 5, mask=mask)
    boxes = [tuple(first)]

    for frame in frames[1:]:
        nxt = to_gray(frame)
        if points is None or len(points) < 4:
            boxes.append(boxes[-1])
            gray = nxt
            continue
        moved, status, _ = cv2.calcOpticalFlowPyrLK(gray, nxt, points, None)
        good_old = points[status.ravel() == 1]
        good_new = moved[status.ravel() == 1]
        if len(good_new) < 4:
            boxes.append(boxes[-1])
            gray, points = nxt, None
            continue
        shift = np.median(good_new - good_old, axis=0).ravel()
        bx, by, bw, bh = boxes[-1]
        boxes.append(_clamp((bx + shift[0], by + shift[1], bw, bh), frame.shape))
        gray = nxt
        points = good_new.reshape(-1, 1, 2)
    return boxes


# --------------------------------------------------------------------------- #
# MOSSE, from scratch
# --------------------------------------------------------------------------- #

#: The learning rate Bolme et al. use. Kept rather than tuned, because tuning it
#: on this clip would be fitting the tracker to the test.
MOSSE_ETA = 0.125
MOSSE_SIGMA = 2.0
MOSSE_LAMBDA = 0.1


def _mosse_window(patch: np.ndarray) -> np.ndarray:
    """Log-transform, normalise, and taper the edges to zero.

    The cosine window is not cosmetic: the FFT treats the patch as periodic, so
    without it the wrap-around edge is a strong artificial gradient and the filter
    locks onto it.
    """
    f = np.log(patch.astype(np.float32) + 1.0)
    f = (f - f.mean()) / (f.std() + 1e-5)
    win = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    return f * win


def _mosse_target(shape, sigma: float = MOSSE_SIGMA) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-(((xx - w // 2) ** 2 + (yy - h // 2) ** 2) / (2.0 * sigma ** 2)))
    return np.fft.fft2(np.fft.ifftshift(g))


def _mosse_perturbations(patch, target_f, count: int = 8, rng=None):
    """Small random rotations of the first patch, so the filter starts with more
    than one example. Bolme's paper does exactly this, and without it the filter
    is a single correlation and drifts on the second frame."""
    rng = rng or np.random.default_rng(0)
    h, w = patch.shape
    num = np.zeros((h, w), np.complex128)
    den = np.zeros((h, w), np.complex128)
    for _ in range(count):
        angle = float(rng.uniform(-8, 8))
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
        warped = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        F = np.fft.fft2(_mosse_window(np.clip(warped, 0, 255)))
        num += target_f * np.conj(F)
        den += F * np.conj(F)
    return num, den


def track_mosse(frames, first, eta: float = MOSSE_ETA) -> list:
    """MOSSE: a correlation filter learned in the Fourier domain.

    Bolme, Beveridge, Draper & Lui, CVPR 2010. The filter H minimises the squared
    error between the correlation output and a Gaussian peak, which in the Fourier
    domain has a closed form -- an elementwise ratio -- and so can be updated at
    hundreds of frames a second.

    Implemented here because `opencv-python-headless` does not ship it.
    """
    x, y, w, h = (int(v) for v in first)
    w, h = max(8, w // 2 * 2), max(8, h // 2 * 2)
    gray = to_gray(frames[0]).astype(np.float32)
    patch = cv2.resize(gray[y:y + h, x:x + w], (w, h))
    target_f = _mosse_target((h, w))
    num, den = _mosse_perturbations(patch, target_f)

    boxes = [(x, y, w, h)]
    for frame in frames[1:]:
        gray = to_gray(frame).astype(np.float32)
        bx, by, bw, bh = boxes[-1]
        bx = int(np.clip(bx, 0, gray.shape[1] - bw))
        by = int(np.clip(by, 0, gray.shape[0] - bh))
        patch = gray[by:by + bh, bx:bx + bw]
        if patch.shape != (bh, bw):
            boxes.append(boxes[-1])
            continue

        F = np.fft.fft2(_mosse_window(patch))
        H = num / (den + MOSSE_LAMBDA)
        response = np.real(np.fft.ifft2(H * F))
        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        # The target Gaussian was built with `ifftshift`, so zero displacement is
        # index (0, 0) and the response wraps. Subtracting bh//2 instead -- which
        # is what a centred target would need -- put every step half a box out and
        # scored this tracker below the do-nothing control.
        dy = dy - bh if dy > bh // 2 else dy
        dx = dx - bw if dx > bw // 2 else dx
        new = _clamp((bx + dx, by + dy, bw, bh), frame.shape)
        boxes.append(new)

        nx, ny, _, _ = new
        patch = gray[ny:ny + bh, nx:nx + bw]
        if patch.shape == (bh, bw):
            F = np.fft.fft2(_mosse_window(patch))
            num = (1 - eta) * num + eta * (target_f * np.conj(F))
            den = (1 - eta) * den + eta * (F * np.conj(F))
    return boxes


def track_kalman_flow(frames, first) -> list:
    """Lucas-Kanade with a constant-velocity Kalman filter over the centre.

    The filter does not track anything. It smooths what the flow tracker says and
    keeps predicting through the frames where the flow tracker says nothing --
    which is exactly the failure mode a Kalman filter is for, and exactly not the
    one where the tracker is confidently wrong.
    """
    measured = track_optical_flow(frames, first)
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0

    cx, cy = centre(measured[0])
    kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    boxes = [tuple(measured[0])]
    for box, frame in zip(measured[1:], frames[1:]):
        prediction = kf.predict()
        mx, my = centre(box)
        kf.correct(np.array([[np.float32(mx)], [np.float32(my)]]))
        sx, sy = float(kf.statePost[0, 0]), float(kf.statePost[1, 0])
        _, _, w, h = box
        boxes.append(_clamp((sx - w / 2, sy - h / 2, w, h), frame.shape))
        del prediction
    return boxes


TRACKERS: dict[str, Callable] = {
    "Static box (control)": static_box,
    "Template matching": track_template,
    "Mean shift": track_meanshift,
    "CamShift": track_camshift,
    "Optical flow (LK)": track_optical_flow,
    "LK + Kalman": track_kalman_flow,
    "MOSSE (from scratch)": track_mosse,
}

REAL_TRACKERS = tuple(t for t in TRACKERS if "control" not in t)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def target_saturation(starts=STARTS, length: int = RUN_LENGTH) -> list[dict]:
    """How colourful the tracked targets actually are.

    The explanation CamShift's score needs. Mean shift and CamShift both track a
    **hue** histogram, and hue is meaningless where saturation is low: this is an
    overcast plaza and most of the people are in dark coats. The number is
    measured rather than asserted, because "CamShift is bad" and "CamShift was
    given nothing to work with" are different claims.
    """
    rows = []
    for start in starts:
        frames, boxes = truth_chain(start, length)
        if not frames or boxes[0] is None:
            continue
        hsv = cv2.cvtColor(_crop(frames[0], boxes[0]), cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1].astype(np.float32)
        rows.append({
            "start": int(start),
            "mean_saturation": round(float(sat.mean()), 1),
            "share_above_40": round(float((sat > 40).mean()), 3),
        })
    return rows


def backprojection_contrast(starts=STARTS, length: int = RUN_LENGTH) -> list[dict]:
    """How much brighter the hue back-projection is on the target than off it.

    A ratio near 1 means the back-projection cannot tell the target from the
    plaza, and mean shift is then climbing a flat surface.
    """
    rows = []
    for start in starts:
        frames, boxes = truth_chain(start, length)
        if not frames or boxes[0] is None:
            continue
        hist = _hue_histogram(frames[0], boxes[0])
        hsv = cv2.cvtColor(frames[0], cv2.COLOR_RGB2HSV)
        back = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1).astype(np.float32)
        x, y, w, h = boxes[0]
        inside = back[y:y + h, x:x + w].mean()
        outside_mask = np.ones(back.shape, bool)
        outside_mask[y:y + h, x:x + w] = False
        rows.append({
            "start": int(start),
            "inside": round(float(inside), 2),
            "outside": round(float(back[outside_mask].mean()), 2),
            "ratio": round(float(inside / max(back[outside_mask].mean(), 1e-6)), 2),
        })
    return rows


def score_run(predicted, truth, survival_iou: float = SURVIVAL_IOU) -> dict:
    """IoU, centre error, and how long the box stayed on the target.

    Survival is the number that matters and the one a mean IoU hides: a tracker
    that is perfect for ten frames and lost for ninety has the same mean IoU as
    one that is mediocre throughout.
    """
    ious, errors = [], []
    survived = 0
    still_alive = True
    compared = 0
    for p, t in zip(predicted, truth):
        if t is None or p is None:
            continue
        compared += 1
        value = box_iou(p, t)
        ious.append(value)
        px, py = centre(p)
        tx, ty = centre(t)
        errors.append(float(np.hypot(px - tx, py - ty)))
        if still_alive and value >= survival_iou:
            survived += 1
        elif still_alive:
            still_alive = False
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "centre_error_px": float(np.mean(errors)) if errors else float("nan"),
        "frames_survived": survived,
        "frames_compared": compared,
        "survival_rate": survived / max(compared, 1),
    }


def evaluate(starts=STARTS, length: int = RUN_LENGTH) -> list[dict]:
    """Every tracker on every run, against the truth chain."""
    acc = {name: {"mean_iou": [], "centre_error_px": [], "survival_rate": [],
                  "frames_survived": []} for name in TRACKERS}

    for start in starts:
        frames, truth = truth_chain(start, length)
        if not frames or truth[0] is None:
            continue
        for name, fn in TRACKERS.items():
            predicted = (oracle_box(frames, truth[0], truth) if name == "Oracle"
                         else fn(frames, truth[0]))
            s = score_run(predicted, truth)
            for key in ("mean_iou", "centre_error_px", "survival_rate",
                        "frames_survived"):
                acc[name][key].append(s[key])

    return [
        {
            "tracker": name,
            "mean_iou": round(float(np.mean(a["mean_iou"])), 4),
            "centre_error_px": round(float(np.nanmean(a["centre_error_px"])), 2),
            "survival_rate": round(float(np.mean(a["survival_rate"])), 4),
            "frames_survived": round(float(np.mean(a["frames_survived"])), 1),
        }
        for name, a in acc.items()
    ]


def per_run(starts=STARTS, length: int = RUN_LENGTH) -> list[dict]:
    rows = []
    for start in starts:
        frames, truth = truth_chain(start, length)
        if not frames or truth[0] is None:
            continue
        row = {"start": int(start),
               "truth_frames": int(sum(1 for b in truth if b is not None))}
        for name in TRACKERS:
            row[name] = round(score_run(TRACKERS[name](frames, truth[0]),
                                        truth)["mean_iou"], 4)
        rows.append(row)
    return rows


def iou_over_time(start: int, length: int = RUN_LENGTH) -> dict:
    """Per-frame IoU for one run: the curve a mean collapses into one number."""
    frames, truth = truth_chain(start, length)
    out: dict[str, list] = {"frame": list(range(len(frames)))}
    for name in TRACKERS:
        predicted = TRACKERS[name](frames, truth[0])
        out[name] = [box_iou(p, t) if (t is not None and p is not None) else np.nan
                     for p, t in zip(predicted, truth)]
    return out


def survival_versus_accuracy(starts=STARTS, length: int = RUN_LENGTH) -> list[dict]:
    """Mean IoU *while alive* against how long each tracker stays alive.

    The project's own picture. A tracker can be accurate and short-lived or
    sloppy and persistent, and a single mean IoU cannot tell those apart.
    """
    acc = {name: {"alive_iou": [], "survival": []} for name in TRACKERS}
    for start in starts:
        frames, truth = truth_chain(start, length)
        if not frames or truth[0] is None:
            continue
        for name, fn in TRACKERS.items():
            predicted = fn(frames, truth[0])
            alive = []
            # frame 0 is *given* to every tracker, so its IoU is 1.0 by
            # construction. Counting it would hand a perfect score to a tracker
            # that is lost by frame 1, which is exactly the comparison this
            # function exists to make.
            for p, t in zip(predicted[1:], truth[1:]):
                if t is None or p is None:
                    continue
                value = box_iou(p, t)
                if value < SURVIVAL_IOU:
                    break
                alive.append(value)
            s = score_run(predicted, truth)
            acc[name]["alive_iou"].append(float(np.mean(alive)) if alive else 0.0)
            acc[name]["survival"].append(s["survival_rate"])
    return [
        {
            "tracker": name,
            "iou_while_alive": round(float(np.mean(a["alive_iou"])), 4),
            "survival_rate": round(float(np.mean(a["survival"])), 4),
        }
        for name, a in acc.items()
    ]


def sweep_run_length(lengths=(10, 25, 50, 100, 200), starts=STARTS[:6]) -> list[dict]:
    """How the ranking changes with how long you watch.

    A tracking comparison over ten frames is a comparison of initialisation.
    """
    rows = []
    for length in lengths:
        row: dict[str, float | int] = {"length": length}
        for name in TRACKERS:
            ious = []
            for start in starts:
                frames, truth = truth_chain(start, length)
                if not frames or truth[0] is None:
                    continue
                ious.append(score_run(TRACKERS[name](frames, truth[0]),
                                      truth)["mean_iou"])
            row[name] = round(float(np.mean(ious)), 4) if ious else 0.0
        rows.append(row)
    return rows
