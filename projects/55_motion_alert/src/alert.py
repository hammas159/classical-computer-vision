"""Motion-triggered security alert: six decision rules, three controls, one clip.

The question
------------
A camera watches a place. Motion is detected. **When should it call somebody?**

That is not the question project 30 asks. Project 30 asks which pixels changed
and scores masks. This project holds the mask **fixed** and varies only what is
done with it -- because a security system that answers "which pixels changed"
perfectly and then telephones you four hundred times is not a working system.

> **The mask is not the system. The decision is.** Given the *oracle* mask --
> the exact per-pixel truth project 30 had to earn -- the rule "alert if anything
> moves in the zone" fires on hundreds of frames for a handful of intrusions.
> Nothing about the pixels is wrong. The rule is wrong.

Shared footage, different question
----------------------------------
This uses `vtest.avi`, which projects 29, 30 and 57 also use. **That is stated
rather than hidden**, and it is the reason this project was left until last. The
pixels are the same 795 frames; what differs is that nothing here is scored per
pixel. The unit is an **event**, the metrics are *alerts per minute* and
*frames from entry to alert*, and the figures are timelines rather than frames,
so nothing in this project looks like or duplicates anything in that one.

Where the ground truth comes from
---------------------------------
Two things, and the awkward part is named rather than smoothed over.

**1 -- Planted intrusions.** A real pedestrian is cut out of the clip using the
oracle mask, and pasted into the restricted zone across a **recorded span of
frames** on a recorded path. The entry and exit frames are therefore exact. The
person is real pixels from this camera at this exposure; only the schedule is
invented, and the schedule is precisely what is being measured.

**2 -- Verified-quiet frames.** The zone is not empty for the whole clip: real
people walk across it in about 3% of frames. So "no intrusion" cannot be
assumed, it has to be **checked**, frame by frame, with the oracle. Frames where
the zone is genuinely still are the only ones on which a false alert is counted;
the ambiguous ones are excluded from both arms and the count of exclusions is
reported.

The three controls
------------------
`Always alert` fires on every frame and catches every intrusion. `Never alert`
fires on none and has a perfect false-alert rate. `Random, rate-matched` fires at
the same rate as the best real rule but at random times, which separates "this
rule fires at the right moments" from "this rule fires the right number of
times".
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

EPS = 1e-9

VIDEO = Path.home() / ".cache" / "classical-cv-images" / "assets" / "video" / "vtest.avi"

#: Frames per second, from the file. Every duration in this project is reported
#: in seconds or alerts per minute rather than in frames, because "412 alerts"
#: means nothing until it is "31 alerts a minute".
FPS = 10.0

#: The restricted zone: the grass at the bottom right, as fractions of the frame.
#: Chosen by measuring how often each candidate region contains motion at all --
#: the paving at the centre is busy in 94% of frames and this is busy in 3.4% --
#: so it is the only part of this scene that behaves like somewhere you would put
#: an alarm.
ZONE = (0.55, 0.80, 1.00, 1.00)

#: Robust noise floor of this clip, measured on a patch of building wall that
#: never changes. Every motion threshold here is a multiple of it, so the numbers
#: transfer to a differently-encoded copy of the same scene.
NOISE_PATCH = (20, 20, 80, 60)
ORACLE_MULTIPLE = 4.0

#: How the oracle background is built: the per-pixel median of every eighth
#: frame. The same device project 30 uses, and it is the reason the truth here
#: can be called exact rather than assumed.
BACKGROUND_STRIDE = 8

_frames: list[np.ndarray] | None = None
_background: np.ndarray | None = None
_noise: float | None = None


def available() -> bool:
    return VIDEO.exists()


def frames() -> list[np.ndarray]:
    """Every frame of the clip, RGB, decoded once and kept."""
    global _frames
    if _frames is None:
        if not VIDEO.exists():
            raise FileNotFoundError(
                f"{VIDEO} is missing. Run `python tools/fetch_assets.py --set video`.")
        cap = cv2.VideoCapture(str(VIDEO))
        out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        _frames = out
    return _frames


def frame_count() -> int:
    return len(frames())


def noise_floor() -> float:
    """Robust sigma of a patch that never changes, in grey levels."""
    global _noise
    if _noise is None:
        x0, y0, x1, y1 = NOISE_PATCH
        patch = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
                          .astype(np.float32) for f in frames()[::3]])
        _noise = float(np.median(np.abs(patch - patch.mean(axis=0))) * 1.4826)
    return _noise


def background() -> np.ndarray:
    """The empty scene: per-pixel median over the whole clip."""
    global _background
    if _background is None:
        stack = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
                          for f in frames()[::BACKGROUND_STRIDE]])
        _background = np.median(stack, axis=0)
    return _background


def zone_mask(shape=None) -> np.ndarray:
    h, w = (shape or frames()[0].shape)[:2]
    x0, y0, x1, y1 = ZONE
    m = np.zeros((h, w), np.uint8)
    m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    return m


def _clean(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def oracle_mask(frame: np.ndarray, multiple: float = ORACLE_MULTIPLE) -> np.ndarray:
    """Where this frame differs from the empty scene, by more than the noise.

    Not causal -- it uses the median of the whole clip, including the future --
    so it is a truth rather than a method. It is here to make the project's
    central point: even with this mask, the alerting rule decides everything.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return _clean((np.abs(g - background()) > multiple * noise_floor())
                  .astype(np.uint8) * 255)


def causal_mask(index: int, history: int = 60,
                multiple: float = ORACLE_MULTIPLE, seq=None) -> np.ndarray:
    """What a real system could compute: the median of the last `history` frames.

    Uses no frame later than the current one, which the oracle does. Reported
    beside the oracle so the gap between "the best mask obtainable" and "the best
    mask obtainable *live*" is visible.

    ``seq`` **must** be the same sequence the rest of the arm is scoring. It
    defaults to the unmodified clip, and the first version of this project let
    that default stand while the oracle arm ran on the *planted* clip -- so the
    causal mask was differencing frames that had no intruder in them and found
    0 of 7. A comparison where the two arms look at different pixels is not a
    comparison.
    """
    seq = frames() if seq is None else seq
    start = max(0, index - history)
    if index - start < 5:
        return np.zeros(seq[0].shape[:2], np.uint8)
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
                      for f in seq[start:index:3]])
    bg = np.median(stack, axis=0)
    g = cv2.cvtColor(seq[index], cv2.COLOR_RGB2GRAY).astype(np.float32)
    return _clean((np.abs(g - bg) > multiple * noise_floor()).astype(np.uint8) * 255)


# --------------------------------------------------------------------------- #
# what is in the zone, per frame
# --------------------------------------------------------------------------- #


def largest_blob_in_zone(mask: np.ndarray) -> int:
    """Area of the biggest connected blob inside the restricted zone."""
    inside = cv2.bitwise_and(mask, zone_mask(mask.shape))
    n, _, stats, _ = cv2.connectedComponentsWithStats((inside > 0).astype(np.uint8), 8)
    if n <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


#: A blob smaller than this is a leaf, a bird or a compression artefact rather
#: than a person. Measured against the planted intruder, which is about 3500 px.
QUIET_AREA = 150


def zone_activity(masks=None) -> np.ndarray:
    """Largest in-zone blob for every frame of the unmodified clip."""
    seq = frames()
    if masks is None:
        masks = (oracle_mask(f) for f in seq)
    return np.array([largest_blob_in_zone(m) for m in masks], np.int64)


def quiet_frames(activity=None) -> np.ndarray:
    """Frames where the zone is **verified** still, not merely assumed still.

    The zone is not empty for the whole clip -- real people cross it -- so a
    false alert can only be counted where the oracle says nothing is there. The
    frames in between are excluded from both arms, and `excluded_frames` counts
    them.
    """
    activity = zone_activity() if activity is None else activity
    return np.nonzero(activity < QUIET_AREA)[0]


def busy_frames(activity=None) -> np.ndarray:
    activity = zone_activity() if activity is None else activity
    return np.nonzero(activity >= QUIET_AREA)[0]


# --------------------------------------------------------------------------- #
# planting an intruder, with a recorded schedule
# --------------------------------------------------------------------------- #

#: The donor: a well-isolated pedestrian, found by searching the clip for the
#: largest person-shaped blob the oracle produces. Recorded as a constant so the
#: composite is reproducible rather than re-searched.
DONOR_FRAME = 401
DONOR_MIN_AREA, DONOR_MAX_AREA = 900, 4000
DONOR_MIN_RATIO, DONOR_MAX_RATIO = 1.4, 3.6

#: Four intrusions, each a (first frame, last frame) span and a path across the
#: zone in fractions of the frame. These spans **are** the ground truth: they are
#: applied by this code, so entry and exit are exact.
#:
#: They are placed where the zone is verified quiet -- see `plan_intrusions`,
#: which refuses to plant over real activity rather than silently overlapping it.
INTRUSIONS = (
    (120, 150, (0.62, 0.86), (0.92, 0.86)),
    (230, 258, (0.95, 0.92), (0.62, 0.84)),
    (300, 325, (0.95, 0.90), (0.60, 0.82)),
    (400, 430, (0.60, 0.84), (0.90, 0.92)),
    # Deliberately overlaps frames 502-505, where a real person walks through the
    # zone. `plan_intrusions` refuses it. It is left in the list rather than
    # quietly removed, because a guard nobody can see firing is a guard nobody
    # can trust.
    (470, 505, (0.60, 0.82), (0.95, 0.92)),
    (545, 575, (0.92, 0.88), (0.62, 0.86)),
    (650, 672, (0.90, 0.84), (0.62, 0.90)),
    (700, 730, (0.62, 0.90), (0.95, 0.84)),
)


def cut_donor() -> tuple[np.ndarray, np.ndarray]:
    """A real pedestrian and their mask, cut out of the clip by the oracle.

    The mask includes the person's **cast shadow**, because the oracle sees the
    shadow as a change too. That is stated rather than trimmed: a planted
    intruder therefore brings a little paving with it onto the grass, which
    makes it slightly easier to detect than a real one would be.
    """
    frame = frames()[DONOR_FRAME]
    mask = oracle_mask(frame)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    best, best_area = None, 0
    for j in range(1, n):
        a = stats[j, cv2.CC_STAT_AREA]
        w, h = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        if (DONOR_MIN_AREA < a < DONOR_MAX_AREA
                and DONOR_MIN_RATIO < h / max(w, 1) < DONOR_MAX_RATIO and a > best_area):
            best, best_area = j, a
    if best is None:
        raise RuntimeError("no donor pedestrian found in the donor frame")
    x, y = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP]
    w, h = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    person = frame[y:y + h, x:x + w].copy()
    cut = ((labels[y:y + h, x:x + w] == best).astype(np.uint8) * 255)
    return person, cut


def plan_intrusions(intrusions=INTRUSIONS, activity=None) -> list[dict]:
    """Turn the recorded spans into per-frame positions, refusing bad spans.

    A span that overlaps frames where the zone already has real activity would
    make "was the alert caused by the intruder" unanswerable, so such a span is
    dropped and reported rather than used.
    """
    activity = zone_activity() if activity is None else activity
    out = []
    for first, last, start_xy, end_xy in intrusions:
        span = np.arange(first, last + 1)
        overlaps = int((activity[span] >= QUIET_AREA).sum())
        out.append({
            "first": int(first), "last": int(last),
            "frames": len(span), "overlaps_real_activity": overlaps,
            "usable": overlaps == 0,
            "start": start_xy, "end": end_xy,
            "seconds": len(span) / FPS,
        })
    return out


def planted_clip(intrusions=INTRUSIONS, activity=None):
    """The clip with intruders composited in, and the per-frame truth.

    Returns ``(frames, present)`` where ``present[i]`` is True when an intruder
    is in the zone in frame ``i``. Only spans that `plan_intrusions` marks usable
    are planted.
    """
    seq = [f.copy() for f in frames()]
    person, cut = cut_donor()
    ph, pw = cut.shape[:2]
    H, W = seq[0].shape[:2]
    present = np.zeros(len(seq), bool)
    plan = plan_intrusions(intrusions, activity)
    for spec, info in zip(intrusions, plan):
        if not info["usable"]:
            continue
        first, last, (sx, sy), (ex, ey) = spec
        span = np.arange(first, last + 1)
        for t, i in enumerate(span):
            u = t / max(len(span) - 1, 1)
            cx = (1 - u) * sx + u * ex
            cy = (1 - u) * sy + u * ey
            x = int(np.clip(cx * W - pw / 2, 0, W - pw))
            y = int(np.clip(cy * H - ph / 2, 0, H - ph))
            region = seq[i][y:y + ph, x:x + pw]
            sel = cut > 0
            region[sel] = person[sel]
            present[i] = True
    return seq, present


# --------------------------------------------------------------------------- #
# the decision layer: what turns a mask into an alert
# --------------------------------------------------------------------------- #


def alerts_from(signal: np.ndarray, min_area: int = 0, persistence: int = 1,
                cooldown: int = 0) -> np.ndarray:
    """Frames on which an alert fires, from a per-frame in-zone blob area.

    Three knobs, and they are the whole project:

    * ``min_area``    -- ignore anything smaller than a person.
    * ``persistence`` -- require the motion to last this many consecutive frames,
      which is what separates a person from a gust of wind.
    * ``cooldown``    -- having alerted, stay quiet for this many frames, so one
      intruder produces one telephone call rather than thirty.
    """
    over = signal >= max(min_area, 1)
    if persistence > 1:
        run = np.zeros(len(over), int)
        count = 0
        for i, v in enumerate(over):
            count = count + 1 if v else 0
            run[i] = count
        firing = run >= persistence
    else:
        firing = over
    out = []
    last = -10 ** 9
    for i in np.nonzero(firing)[0]:
        if i - last > cooldown:
            out.append(int(i))
            last = i
    return np.array(out, int)


#: The six rules, as (min_area, persistence, cooldown). Each adds one knob to the
#: one before, so the table reads as an ablation rather than as six unrelated
#: configurations.
RULES: dict[str, tuple[int, int, int]] = {
    "Any motion in the zone": (1, 1, 0),
    "+ minimum area": (600, 1, 0),
    "+ persistence 3": (600, 3, 0),
    "+ persistence 8": (600, 8, 0),
    "+ cooldown 50": (600, 8, 50),
    "+ cooldown 100": (600, 8, 100),
}


# --------------------------------------------------------------------------- #
# scoring, at the level of events
# --------------------------------------------------------------------------- #


def score_alerts(fired: np.ndarray, present: np.ndarray, quiet: np.ndarray,
                 spans=None) -> dict:
    """Event-level scoring. No pixel is counted anywhere in this function.

    * an intrusion is **detected** if any alert falls inside its span,
    * **latency** is frames from the intruder entering to the first alert,
    * a **false alert** is one on a frame where the zone is *verified* quiet,
    * everything else -- alerts during real, unlabelled activity -- is counted
      separately as ``unscored`` rather than being quietly called correct or
      incorrect.
    """
    spans = spans if spans is not None else present_spans(present)
    fired_set = set(int(i) for i in fired)

    # `quiet` is measured on the **unmodified** clip, so it still contains the
    # frames an intruder was later planted into. Subtracting the spans here, in
    # the scorer, is not a nicety: without it every correct alert during an
    # intrusion is also counted as a false alert, which made the first run of
    # this project report 214 false alerts where there are 44.
    planted = {i for first, last in spans for i in range(first, last + 1)}
    quiet_set = set(int(i) for i in quiet) - planted

    detected, latencies = 0, []
    for first, last in spans:
        hits = [i for i in fired_set if first <= i <= last]
        if hits:
            detected += 1
            latencies.append(min(hits) - first)

    false = sum(1 for i in fired_set if i in quiet_set)
    inside = sum(1 for i in fired_set
                 for first, last in spans if first <= i <= last)
    unscored = len(fired_set) - false - inside
    minutes = len(present) / FPS / 60.0
    return {
        "alerts": len(fired_set),
        "intrusions": len(spans),
        "detected": detected,
        "missed": len(spans) - detected,
        "median_latency_frames": float(np.median(latencies)) if latencies else float("nan"),
        "median_latency_seconds": float(np.median(latencies)) / FPS if latencies else float("nan"),
        "false_alerts": false,
        "false_per_minute": false / max(minutes, EPS),
        "alerts_per_minute": len(fired_set) / max(minutes, EPS),
        "unscored": unscored,
    }


def present_spans(present: np.ndarray) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, v in enumerate(present):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(present) - 1))
    return spans


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def control_always(signal: np.ndarray, **kw) -> np.ndarray:
    """**Control.** Alert on every frame. Catches everything, useless."""
    return np.arange(len(signal))


def control_never(signal: np.ndarray, **kw) -> np.ndarray:
    """**Control.** Never alert. A perfect false-alert rate, and no security."""
    return np.array([], int)


def control_random(signal: np.ndarray, count: int = 20, seed: int = 0) -> np.ndarray:
    """**Control.** Alert at random, the same number of times as a real rule.

    Rate-matched on purpose: it separates *firing the right number of times* from
    *firing at the right moments*, which a raw alert count cannot.
    """
    rng = np.random.default_rng(seed)
    count = min(count, len(signal))
    return np.sort(rng.choice(len(signal), size=count, replace=False))
