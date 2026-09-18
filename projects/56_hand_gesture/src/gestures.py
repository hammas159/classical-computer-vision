"""Hand gesture recognition: seven segmenters, four controls, and an exact matte.

The question
------------
The classical hand-gesture pipeline is two stages. **Segment the hand** -- almost
always by skin colour -- then **describe its shape**, usually by counting the deep
convexity defects between the fingers. Every tutorial spends its words on the
second stage, because that is the clever part.

> **The second stage is not where the problem is.** Handed a perfect mask, the
> shape stage answers correctly. Handed a mask recovered from a photograph, the
> same shape code answers differently -- and how differently is predicted almost
> entirely by **the background**, which is the one thing nobody in the pipeline
> is looking at.

Where the ground truth comes from
---------------------------------
The 27 hand photographs are **real**, of eight different people, and they arrive
already cut out against pure black. That is somebody else's segmentation, and it
is inherited rather than invented -- but it means a threshold at grey level 20
recovers an **exact foreground mask** for free.

So the construction is:

1. take the real hand and its exact mask,
2. **composite it onto a real photograph** at a recorded position and scale,
3. ask each segmenter to recover the mask it was built from.

The truth is recorded because the composite is performed here. Nothing about the
hand is synthetic -- it is a photograph of a hand on a photograph of a place --
and the only invented quantity is *where* it was put, which is exactly the
quantity being scored.

**The gesture label is real too**: the dataset names each file after the sign it
shows, so `3_P` is three fingers and `B_P` is the letter B. Five of the
twenty-seven are numbers, which gives a small but genuine finger-count accuracy.

The difficulty axis
-------------------
`skin_likeness(background)` is the share of a background that a standard YCrCb
skin rule accepts **before any hand is put on it**. Across the twelve
backgrounds it runs from **0%** (dolphins in open water) to **99%** (a sunlit
sandy wall). Two of the twelve contain real human skin and one contains a bronze
human figure, which are the cases a skin detector cannot win.

The four controls
-----------------
`Oracle mask` is handed the exact matte and is not a method: it is the ceiling,
and what the shape stage scores there is what the shape stage is worth.
`Whole frame`, `Centre ellipse` and `Nothing` fence in the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

EPS = 1e-9

#: 27 photographs of hands, eight people, from the HGR1 gesture set. Each is
#: 224x224 with the hand cut out against black, and the gesture is the prefix of
#: the filename.
HANDS = Path.home() / ".cache" / "classical-cv-images" / "assets" / "hands_hgr1"

#: The BSDS500 photographs used as backgrounds, none of them used by any other
#: project. Ordered here by how much of each a skin-colour rule already accepts,
#: which is the project's difficulty axis and is measured, not guessed.
BSDS = Path.home() / ".cache" / "classical-cv-images" / "bsds"

BACKGROUNDS = {
    "314016": "dolphins in open water",
    "119082": "a painted mural and a city street",
    "268002": "a cormorant on a branch",
    "247012": "a black dog on grass",
    "206062": "a polo player on a horse",
    "33044": "the steps of a Mayan pyramid",
    "372019": "a bronze human figure on a rock",
    "81090": "a woman on a beach — real skin in the background",
    "188025": "people unloading a plane on snow",
    "65132": "orange koi carp under water",
    "372047": "a guard beside a sentry box",
    "293029": "a man sweeping beside a sunlit sandy wall",
}

#: Grey level above which a pixel of a HGR1 photograph is hand rather than
#: backdrop. The backdrop is pure black: on eleven of the twelve sampled images
#: under 1.5% of pixels fall between 8 and 48, so the cut is not delicate.
MATTE_CUT = 20

#: Where the hand is placed in the background, as a fraction of the background,
#: and how tall it is made. Fixed so the composite is reproducible and so no
#: method can be helped by a lucky position.
PLACE = (0.30, 0.22)
HAND_HEIGHT = 0.62

#: A convexity defect counts as a gap between fingers when it is deeper than
#: this fraction of the contour's perimeter. Shared by every method, so the
#: comparison is of masks rather than of who tuned their defect threshold.
DEFECT_DEPTH = 0.02


def available() -> bool:
    return (HANDS.exists() and len(list(HANDS.glob("*.jpg"))) >= 20
            and all((BSDS / f"{b}.jpg").exists() for b in BACKGROUNDS))


def hand_names() -> list[str]:
    return sorted(p.stem for p in HANDS.glob("*.jpg"))


def gesture_of(name: str) -> str:
    """The sign this photograph shows, from the folder the dataset put it in."""
    return name.split("__")[0].replace("_P", "")


def person_of(name: str) -> str:
    """Which of the eight people this is, from the filename."""
    return "id" + name.split("_id")[1].split("_")[0]


def fingers_of(name: str) -> int | None:
    """How many fingers are held up, where the dataset says so.

    Only the five numeric gestures carry this. The letters are **not** given a
    finger count here: whether the thumb counts as extended in `A` or `S` is a
    judgement, and inventing one would turn the dataset's label into mine.
    """
    g = gesture_of(name)
    return int(g) if g.isdigit() else None


def load_hand(name: str) -> np.ndarray:
    img = cv2.imread(str(HANDS / f"{name}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"{name} is missing. Run `python tools/fetch_assets.py --set hands`.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_background(name: str) -> np.ndarray:
    img = cv2.imread(str(BSDS / f"{name}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{name} is missing. Run `python tools/fetch_images.py`.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# the matte, and the composite that makes it a truth
# --------------------------------------------------------------------------- #


def matte(hand: np.ndarray) -> np.ndarray:
    """The exact foreground mask, from the black backdrop the dataset supplies.

    Largest connected component only: a few of the photographs carry a speck of
    sleeve or a compression fleck in a corner, and including those would make the
    "exact" truth quietly inexact.
    """
    g = cv2.cvtColor(hand, cv2.COLOR_RGB2GRAY)
    m = (g >= MATTE_CUT).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == keep).astype(np.uint8) * 255
    return m


def matte_is_clean(name: str) -> float:
    """Share of the photograph in the halo between backdrop and hand.

    A number that says how much the "exact" matte is really an approximation.
    It is reported per image rather than assumed away, and one photograph in the
    set is markedly worse than the rest.
    """
    g = cv2.cvtColor(load_hand(name), cv2.COLOR_RGB2GRAY)
    return float(((g >= 8) & (g < 48)).mean())


def composite(hand_name: str, background_name: str):
    """Put a real hand on a real photograph and return the frame and its mask.

    Returns ``(frame, mask)``. The mask is what the frame was built from, so it
    is exact by construction -- this is the same device project 30 uses to get a
    truth for background subtraction and project 54 uses for defects.
    """
    hand = load_hand(hand_name)
    mask = matte(hand)
    bg = load_background(background_name).copy()
    bh, bw = bg.shape[:2]

    target_h = max(16, int(HAND_HEIGHT * bh))
    scale = target_h / hand.shape[0]
    target_w = max(16, int(round(hand.shape[1] * scale)))
    hand_s = cv2.resize(hand, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    x0 = int(PLACE[0] * bw)
    y0 = int(PLACE[1] * bh)
    x0 = max(0, min(x0, bw - target_w))
    y0 = max(0, min(y0, bh - target_h))

    out_mask = np.zeros((bh, bw), np.uint8)
    region = bg[y0:y0 + target_h, x0:x0 + target_w]
    sel = mask_s > 0
    region[sel] = hand_s[sel]
    out_mask[y0:y0 + target_h, x0:x0 + target_w][sel] = 255
    return bg, out_mask


def skin_likeness(background_name: str) -> float:
    """How much of a background a standard skin rule accepts with no hand on it.

    Computed from the background alone, before any composite is made and before
    any method runs, so it predicts difficulty rather than explaining it
    afterwards.
    """
    y = cv2.cvtColor(load_background(background_name), cv2.COLOR_RGB2YCrCb)
    return float((cv2.inRange(y, (0, 133, 77), (255, 173, 127)) > 0).mean())


# --------------------------------------------------------------------------- #
# stage one: find the hand
# --------------------------------------------------------------------------- #


def _largest(mask: np.ndarray) -> np.ndarray:
    """Keep the largest blob and close the gaps inside it.

    Every segmenter gets the identical cleanup, so the comparison is of the
    colour rule rather than of the morphology behind it.
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8) * 255


def segment_ycrcb(frame, **kw):
    """The textbook rule: Cr in [133, 173], Cb in [77, 127].

    Chrominance-only, so it is meant to survive a change of illumination. What
    it cannot survive is a background that happens to be that colour.
    """
    y = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    return _largest(cv2.inRange(y, (0, 133, 77), (255, 173, 127)))


def segment_hsv(frame, **kw):
    """The other textbook rule, in HSV: a narrow hue band at moderate saturation."""
    h = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return _largest(cv2.inRange(h, (0, 40, 60), (25, 180, 255)))


def segment_lab(frame, **kw):
    """Skin in Lab: positive a (red) and positive b (yellow)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    return _largest(cv2.inRange(lab, (40, 135, 135), (250, 180, 185)))


def segment_ycrcb_and_hsv(frame, **kw):
    """Both rules, intersected -- the standard way to cut the false positives.

    Worth having because it is what a practitioner reaches for when a single
    rule over-fires, and because intersecting two rules that fail on the *same*
    backgrounds buys nothing.
    """
    a = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    b = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    both = cv2.bitwise_and(cv2.inRange(a, (0, 133, 77), (255, 173, 127)),
                           cv2.inRange(b, (0, 40, 60), (25, 180, 255)))
    return _largest(both)


def segment_otsu(frame, **kw):
    """No colour model at all: Otsu on greyscale, dark side taken as the hand.

    The control for the colour rules. If a global brightness cut does as well,
    the colour model is not what is doing the work.

    **Both polarities were measured**: taking the bright side as the hand scores
    a mean IoU of 0.055 over the set and taking the dark side 0.135, so the
    better of the two is used here. Neither works, and that is the point -- a
    brightness threshold has no way to know which side of the cut is the object,
    and choosing per image by looking at the answer would not be a method.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _largest(m)


def segment_grabcut(frame, **kw):
    """GrabCut initialised from a generous box around where the hand was put.

    **This method is told something the others are not** -- roughly where to
    look -- so it is expected to win and its margin is the interesting number
    rather than the win itself. The box is derived from the placement constants,
    not from the mask, so it does not leak the answer's shape.
    """
    h, w = frame.shape[:2]
    x0 = int(PLACE[0] * w)
    y0 = int(PLACE[1] * h)
    bw = int(0.75 * HAND_HEIGHT * h)
    bh = int(HAND_HEIGHT * h)
    rect = (max(1, x0 - 10), max(1, y0 - 10),
            min(bw + 20, w - x0 - 1), min(bh + 20, h - y0 - 1))
    if rect[2] < 20 or rect[3] < 20:
        return np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), mask, rect,
                    bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _largest(out)


def segment_adaptive(frame, **kw):
    """Adaptive threshold on the Cr channel: a local version of the skin rule.

    Both offsets were measured, as for Otsu: C = -6 scores 0.031 and C = +6
    scores 0.104, and the better is used. A local threshold cannot help here,
    because the thing that makes a background hard is that it is *globally* the
    same colour as skin, not that it is unevenly lit.
    """
    cr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)[..., 1]
    m = cv2.adaptiveThreshold(cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 51, 6)
    return _largest(m)


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def control_oracle(frame, truth=None, **kw):
    """**Not a method.** Handed the exact matte the composite was built from.

    It exists to separate the two stages: whatever the shape code scores here is
    the ceiling, and every real segmenter's shortfall is the cost of stage one.
    """
    if truth is None:
        raise ValueError("the oracle control must be given the true mask")
    return truth.copy()


def control_whole_frame(frame, **kw):
    """**Control.** Everything is hand."""
    return np.full(frame.shape[:2], 255, np.uint8)


def control_centre_ellipse(frame, **kw):
    """**Control.** A hand-sized ellipse where a hand usually is, image unread."""
    h, w = frame.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int((PLACE[0] + 0.18) * w), int((PLACE[1] + 0.31) * h)),
                (int(0.16 * w), int(0.30 * h)), 0, 0, 360, 255, -1)
    return m


def control_nothing(frame, **kw):
    """**Control.** Nothing is hand."""
    return np.zeros(frame.shape[:2], np.uint8)


SEGMENTERS: dict[str, Callable] = {
    "Oracle mask (control)": control_oracle,
    "Whole frame (control)": control_whole_frame,
    "Centre ellipse (control)": control_centre_ellipse,
    "Nothing (control)": control_nothing,
    "YCrCb skin": segment_ycrcb,
    "HSV skin": segment_hsv,
    "Lab skin": segment_lab,
    "YCrCb and HSV": segment_ycrcb_and_hsv,
    "Adaptive Cr": segment_adaptive,
    "Otsu on grey": segment_otsu,
    "GrabCut from a box": segment_grabcut,
}

REAL_SEGMENTERS = tuple(k for k in SEGMENTERS if "control" not in k)


# --------------------------------------------------------------------------- #
# stage two: describe the shape
# --------------------------------------------------------------------------- #


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= 64 else None


def deep_defects(mask: np.ndarray, depth: float = DEFECT_DEPTH):
    """The gaps between fingers: convexity defects deeper than a share of the
    perimeter.

    Scaling the depth to the contour's own perimeter rather than to pixels is
    what lets one threshold serve hands photographed at different sizes -- the
    same fractions-not-pixels point project 06 makes about a region of interest.
    """
    c = largest_contour(mask)
    if c is None or len(c) < 5:
        return []
    hull = cv2.convexHull(c, returnPoints=False)
    if hull is None or len(hull) < 4:
        return []
    try:
        defects = cv2.convexityDefects(c, hull)
    except cv2.error:
        return []
    if defects is None:
        return []
    limit = depth * cv2.arcLength(c, True)
    out = []
    for s, e, f, d in defects[:, 0]:
        if d / 256.0 > limit:
            out.append((tuple(c[s][0]), tuple(c[e][0]), tuple(c[f][0]), d / 256.0))
    return out


#: Radius of the finger-counting circle, as a multiple of the palm radius.
#: Swept from 1.6 to 3.0 on perfect masks; 1.8 is the best value on this set and
#: the sweep is reported rather than hidden, because the best value is only
#: 3 of 5 and a single number would imply the rule works.
PALM_RADIUS_K = 1.8


def palm_circle(mask: np.ndarray):
    """Centre and radius of the largest circle that fits inside the silhouette.

    The palm, by construction -- it is the widest part of a hand. Everything the
    finger counter does is relative to it, which is what makes the count
    independent of how large the hand was photographed.
    """
    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, radius, _, centre = cv2.minMaxLoc(d)
    # A mask with no background pixels at all has nothing to measure a distance
    # to, and OpenCV returns FLT_MAX (3.4e38) for every pixel. That is finite, so
    # it passes a naive guard and then overflows the integer cast downstream --
    # which is precisely what the `Whole frame` control produces. A palm cannot
    # be larger than the frame, so the radius is capped at half the short side.
    limit = 0.5 * min(mask.shape[:2])
    return centre, float(min(radius, limit))


def count_fingers(mask: np.ndarray, k: float = PALM_RADIUS_K) -> int:
    """Fingers, by counting what a circle round the palm crosses.

    Malima's method: draw a circle at `k` palm radii and count the runs of
    foreground it passes through. Each extended finger is one run; the wrist is
    one more, so the count is runs minus one.

    This replaces counting convexity defects directly, which does not work at
    all on this set: the photographs include the **forearm**, and the wrist
    contributes defects of its own, so the defect count came out at three or
    four whatever the hand was doing.
    """
    if not np.any(mask):
        return 0
    centre, radius = palm_circle(mask)
    # An empty or degenerate mask sends a NaN through the cos/sin below, which
    # numpy casts silently to a huge negative int rather than raising. The
    # controls produce exactly that, so it is guarded rather than left to warn.
    if not np.isfinite(radius) or radius < 3 or not all(np.isfinite(centre)):
        return 0
    r = k * radius
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    xs = np.clip((centre[0] + r * np.cos(angles)).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip((centre[1] + r * np.sin(angles)).astype(int), 0, mask.shape[0] - 1)
    on = (mask[ys, xs] > 0).astype(np.int8)
    runs = int(((on == 1) & (np.roll(on, 1) == 0)).sum())
    return max(0, runs - 1)


def shape_features(mask: np.ndarray) -> dict:
    """Everything the second stage computes, from a mask and nothing else.

    `fingers` is the classical rule -- one more than the number of deep gaps --
    and it is reported whatever the mask, so a mask covering the whole frame
    produces a finger count too. That is the point of keeping the controls.
    """
    c = largest_contour(mask)
    if c is None:
        return {"area": 0.0, "solidity": 0.0, "defects": 0, "fingers": 0,
                "aspect": 0.0, "extent": 0.0, "hu1": 0.0}
    area = float(cv2.contourArea(c))
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(c)
    defects = deep_defects(mask)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    return {
        "area": area / max(mask.size, 1),
        "solidity": area / max(hull_area, EPS),
        "defects": len(defects),
        "fingers": count_fingers(mask),
        "aspect": w / max(h, 1),
        "extent": area / max(w * h, EPS),
        "hu1": float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + EPS)),
    }


def feature_gap(a: dict, b: dict) -> float:
    """How far two feature vectors sit apart, in units of the scale-free ones.

    Used to say what a segmentation error costs the *description*, which is the
    quantity that actually reaches a classifier. Area is excluded because a mask
    covering the whole frame would dominate the distance and say nothing about
    shape.
    """
    keys = ("solidity", "aspect", "extent")
    return float(np.sqrt(sum((a[k] - b[k]) ** 2 for k in keys)))


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def iou(a, b) -> float:
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def evaluate(segmenter: str, hands=None, backgrounds=None) -> dict:
    """One segmenter over every hand-on-background pair.

    Both stages are scored on the same pass: the mask against the matte it was
    built from, and the shape features against the ones the **oracle** mask
    gives on the identical frame.
    """
    hands = hands or hand_names()
    backgrounds = backgrounds or list(BACKGROUNDS)
    fn = SEGMENTERS[segmenter]
    rows = []
    for i, hand in enumerate(hands):
        bg = backgrounds[i % len(backgrounds)]
        frame, truth = composite(hand, bg)
        got = fn(frame, truth=truth)
        true_features = shape_features(truth)
        got_features = shape_features(got)
        want = fingers_of(hand)
        rows.append({
            "hand": hand, "gesture": gesture_of(hand), "person": person_of(hand),
            "background": bg, "skin_likeness": skin_likeness(bg),
            "iou": iou(got, truth),
            "defects": got_features["defects"],
            "true_defects": true_features["defects"],
            "fingers": got_features["fingers"],
            "true_fingers": true_features["fingers"],
            "want_fingers": want,
            "feature_gap": feature_gap(got_features, true_features),
            "solidity": got_features["solidity"],
            "true_solidity": true_features["solidity"],
        })
    numeric = [r for r in rows if r["want_fingers"] is not None]
    return {
        "segmenter": segmenter,
        "frames": len(rows),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "median_iou": float(np.median([r["iou"] for r in rows])),
        "found": sum(r["iou"] >= 0.5 for r in rows),
        "median_feature_gap": float(np.median([r["feature_gap"] for r in rows])),
        "defects_match": sum(r["defects"] == r["true_defects"] for r in rows),
        "finger_correct": sum(r["fingers"] == r["want_fingers"] for r in numeric),
        "finger_frames": len(numeric),
        "per_image": rows,
    }


def evaluate_all(hands=None, backgrounds=None) -> list[dict]:
    return [evaluate(k, hands, backgrounds) for k in SEGMENTERS]


def stage_split(hands=None, backgrounds=None) -> dict:
    """The project's central number: what each of the two stages is worth.

    The oracle is handed the exact matte, so its shape answers are the ceiling
    the second stage can reach on this data. Every real segmenter's shortfall is
    the price of the first stage, and the two are reported separately because a
    single accuracy merges them.
    """
    results = {r["segmenter"]: r for r in evaluate_all(hands, backgrounds)}
    oracle = results["Oracle mask (control)"]
    best = max((results[k] for k in REAL_SEGMENTERS), key=lambda r: r["mean_iou"])
    return {
        "ceiling_defects_match": oracle["defects_match"],
        "ceiling_finger_correct": oracle["finger_correct"],
        "finger_frames": oracle["finger_frames"],
        "best_real": best["segmenter"],
        "best_real_iou": best["mean_iou"],
        "best_real_defects_match": best["defects_match"],
        "best_real_finger_correct": best["finger_correct"],
        "frames": oracle["frames"],
    }


def background_effect(segmenter: str = "YCrCb skin", hands=None) -> list[dict]:
    """Every hand on **every** background, so the background is the only variable.

    The per-image table elsewhere pairs one hand with one background; this runs
    the full 27 x 12 grid for one segmenter, which is the only way to say that
    the background rather than the hand is what decides the answer.
    """
    hands = hands or hand_names()
    fn = SEGMENTERS[segmenter]
    rows = []
    for bg in BACKGROUNDS:
        scores = []
        for hand in hands:
            frame, truth = composite(hand, bg)
            scores.append(iou(fn(frame, truth=truth), truth))
        scores = np.asarray(scores)
        rows.append({
            "background": bg, "what": BACKGROUNDS[bg],
            "skin_likeness": skin_likeness(bg),
            "mean_iou": float(scores.mean()),
            "median_iou": float(np.median(scores)),
            "worst_iou": float(scores.min()),
            "best_iou": float(scores.max()),
            # The count that stops the mean lying. On the hardest backgrounds the
            # result is bimodal -- the hand is either found well or missed
            # completely -- and a background whose mean is 0.363 has a median of
            # 0.000. Reporting only the mean would describe neither half.
            "total_failures": int((scores < 0.05).sum()),
            "recovered": int((scores >= 0.5).sum()),
            "of": len(scores),
        })
    return rows


def size_contest(background: str, segmenter: str = "Lab skin") -> dict:
    """Does the largest-component rule simply pick whichever blob is bigger?

    The mechanism behind the total failures. Every segmenter here keeps the
    largest connected component, so when a background contains a skin-coloured
    object larger than the hand, the rule selects **that** and the score is not
    degraded but zero. If this is right, the score on such a background should
    track how much of the frame the hand covers, and nothing else.

    On `a black dog on grass` -- where the skin-coloured object is the varnished
    wooden cart behind the dog, not the dog -- it tracks it at r = +0.83.
    """
    fn = SEGMENTERS[segmenter]
    areas, scores = [], []
    for hand in hand_names():
        frame, truth = composite(hand, background)
        areas.append(float((truth > 0).mean()))
        scores.append(iou(fn(frame, truth=truth), truth))
    areas, scores = np.asarray(areas), np.asarray(scores)
    failed = scores < 0.05
    return {
        "background": background, "what": BACKGROUNDS[background],
        "area_vs_iou_r": float(np.corrcoef(areas, scores)[0, 1]),
        "total_failures": int(failed.sum()), "of": len(scores),
        "mean_area_when_failed": float(areas[failed].mean()) if failed.any() else 0.0,
        "mean_area_when_not": float(areas[~failed].mean()) if (~failed).any() else 0.0,
        "mean_iou": float(scores.mean()), "median_iou": float(np.median(scores)),
    }


def hand_effect(segmenter: str = "YCrCb skin", backgrounds=None) -> list[dict]:
    """The same grid collapsed the other way: how much does the *hand* matter?

    Reported next to `background_effect` because the comparison between the two
    spreads is the claim. If the hand mattered more, this project would be about
    hands.
    """
    backgrounds = backgrounds or list(BACKGROUNDS)
    fn = SEGMENTERS[segmenter]
    rows = []
    for hand in hand_names():
        scores = []
        for bg in backgrounds:
            frame, truth = composite(hand, bg)
            scores.append(iou(fn(frame, truth=truth), truth))
        rows.append({
            "hand": hand, "gesture": gesture_of(hand), "person": person_of(hand),
            "mean_iou": float(np.mean(scores)),
            "worst_iou": float(np.min(scores)),
        })
    return rows


def which_matters_more(segmenter: str = "YCrCb skin") -> dict:
    """Spread across backgrounds against spread across hands, same grid."""
    bg = background_effect(segmenter)
    hd = hand_effect(segmenter)
    bg_means = [r["mean_iou"] for r in bg]
    hd_means = [r["mean_iou"] for r in hd]
    return {
        "segmenter": segmenter,
        "background_spread": float(max(bg_means) - min(bg_means)),
        "hand_spread": float(max(hd_means) - min(hd_means)),
        "background_std": float(np.std(bg_means)),
        "hand_std": float(np.std(hd_means)),
        "skin_likeness_r": float(np.corrcoef([r["skin_likeness"] for r in bg],
                                             bg_means)[0, 1]),
    }


def finger_rule_ceiling(hands=None) -> dict:
    """How well finger counting can possibly do here, given **perfect** masks.

    Two families are swept, both on the oracle mask, so nothing in this table is
    contaminated by segmentation. It exists to answer a question that has to be
    settled before any segmenter is blamed for anything: does the second stage
    work at all on these photographs?

    It does not. The best setting of the best rule gets **3 of the 5** numbered
    gestures, and the failures are not marginal -- the four-finger gesture is
    read as one at every setting tried.

    The reason is visible in the photographs: HGR1 hands are at arbitrary
    orientations with the forearm in frame, and both classical rules assume an
    upright, palm-forward hand with the wrist at the bottom.
    """
    hands = hands or [h for h in hand_names() if fingers_of(h) is not None]
    by_depth, by_radius = [], []
    for depth in (0.01, 0.02, 0.04, 0.08):
        detail = []
        for hand in hands:
            _, truth = composite(hand, list(BACKGROUNDS)[0])
            got = min(5, len(deep_defects(truth, depth)) + 1)
            detail.append({"hand": hand, "want": fingers_of(hand), "got": got})
        by_depth.append({"rule": "convexity defects + 1", "parameter": depth,
                         "correct": sum(d["got"] == d["want"] for d in detail),
                         "of": len(hands), "detail": detail})
    for k in (1.6, 1.8, 2.0, 2.2, 2.6, 3.0):
        detail = []
        for hand in hands:
            _, truth = composite(hand, list(BACKGROUNDS)[0])
            detail.append({"hand": hand, "want": fingers_of(hand),
                           "got": count_fingers(truth, k)})
        by_radius.append({"rule": "palm-circle crossings", "parameter": k,
                          "correct": sum(d["got"] == d["want"] for d in detail),
                          "of": len(hands), "detail": detail})
    rows = by_depth + by_radius
    best = max(rows, key=lambda r: r["correct"])
    return {"sweeps": rows, "best_rule": best["rule"],
            "best_parameter": best["parameter"], "best_correct": best["correct"],
            "of": len(hands)}


def matte_quality(hands=None) -> list[dict]:
    """How exact the "exact" matte is, per photograph."""
    hands = hands or hand_names()
    return sorted(({"hand": h, "gesture": gesture_of(h), "halo": matte_is_clean(h)}
                   for h in hands), key=lambda r: -r["halo"])


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def overlay(frame, mask, truth=None, colour=(60, 220, 90)):
    out = frame.copy()
    sel = mask > 0
    out[sel] = (0.45 * out[sel] + 0.55 * np.array(colour, np.float32)).astype(np.uint8)
    if truth is not None:
        contours, _ = cv2.findContours((truth > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (250, 190, 40),
                         max(2, frame.shape[1] // 300))
    return out


def draw_defects(frame, mask, colour=(235, 70, 70)):
    out = frame.copy()
    c = largest_contour(mask)
    if c is None:
        return out
    cv2.drawContours(out, [cv2.convexHull(c)], -1, (60, 160, 235),
                     max(2, frame.shape[1] // 400))
    for start, end, far, _ in deep_defects(mask):
        cv2.circle(out, far, max(4, frame.shape[1] // 120), colour, -1)
    return out
