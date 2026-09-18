"""Colour space robustness: which space survives a change of illumination?

The question
------------
"Use HSV for colour detection, it's robust to lighting" is repeated everywhere.
It is half true, and the half that is false causes real bugs.

> **The claim under test:** HSV's hue is invariant to a *brightness scale* but
> **not** to a change in the illuminant's colour. Lab's a/b channels are closer
> to perceptually uniform but no more invariant to a colour cast. Normalised RGB
> is invariant to intensity by construction. None of them is invariant to a
> genuine illuminant change without an explicit white-balance step.

Testing it is straightforward because the degradations are applied here: take a
colour target, apply a known brightness scale or a known colour cast, and measure
how far each space's coordinates move.

The practical form of the question — the one that decides whether a colour
threshold written today still works tomorrow — is: **does a threshold tuned under
one illuminant still segment the object under another?** That is measured as IoU,
with the threshold fixed.

🚨 OpenCV's hue is **0-179**, not 0-359 — halved to fit in a uint8. A threshold
written for degrees silently selects the wrong half of the colour wheel.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.io import to_float, to_uint8
from shared.metrics import iou

EPS = 1e-9

#: OpenCV stores hue in 0-179 for 8-bit images. Anything written against a
#: 0-359 assumption is wrong by a factor of two.
OPENCV_HUE_MAX = 179


# --------------------------------------------------------------------------- #
# the colour spaces
# --------------------------------------------------------------------------- #


def to_rgb(img):
    return img.copy()


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def to_normalised_rgb(img):
    """r = R/(R+G+B), g = G/(...), b = B/(...), scaled to 0-255.

    Invariant to a brightness *scale* by construction: multiplying all three
    channels by any constant leaves the ratios unchanged. It is the cleanest
    possible demonstration of what intensity invariance does and does not buy —
    it survives dimming and fails on a colour cast, because a cast changes the
    channels by *different* factors and the ratios move.
    """
    f = to_float(img)
    total = f.sum(axis=2, keepdims=True) + EPS
    return to_uint8(np.clip(f / total, 0.0, 1.0))


SPACES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "RGB": to_rgb,
    "HSV": to_hsv,
    "Lab": to_lab,
    "YCrCb": to_ycrcb,
    "Normalised RGB": to_normalised_rgb,
}

#: Which channels of each space are supposed to be chromatic — the ones a
#: "lighting robust" method would threshold on, dropping the intensity channel.
CHROMA_CHANNELS: dict[str, tuple[int, ...]] = {
    "RGB": (0, 1, 2),
    "HSV": (0, 1),
    "Lab": (1, 2),
    "YCrCb": (1, 2),
    "Normalised RGB": (0, 1, 2),
}


# --------------------------------------------------------------------------- #
# the test scene
# --------------------------------------------------------------------------- #

PATCH_COLOURS = (
    (200, 40, 40), (40, 180, 60), (50, 70, 200), (220, 200, 40),
    (200, 90, 180), (60, 190, 200), (230, 230, 230), (35, 35, 35),
)


def colour_chart(size: int = 320, seed: int = 0):
    """A grid of known colour patches plus a target mask for one of them.

    Returns ``(image, patch_masks)``. Using flat known colours means every
    measured change is caused by the degradation and not by the scene.
    """
    img = np.zeros((size, size, 3), np.uint8)
    masks = []
    cols = 4
    rows = int(np.ceil(len(PATCH_COLOURS) / cols))
    ph, pw = size // rows, size // cols
    for i, colour in enumerate(PATCH_COLOURS):
        r, c = divmod(i, cols)
        y, x = r * ph, c * pw
        img[y : y + ph, x : x + pw] = colour
        m = np.zeros((size, size), np.uint8)
        m[y : y + ph, x : x + pw] = 255
        masks.append(m)
    return img, masks


def object_scene(size: int = 320, target_colour=(200, 40, 40), seed: int = 0):
    """A coloured object on a mixed background, with its exact mask.

    The realistic form of the question: can a colour threshold find this object?
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(0, size - 60)), int(rng.integers(0, size - 60))
        cv2.rectangle(img, (x, y), (x + 55, y + 55), rng.integers(20, 235, 3).tolist(), -1)

    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 5, 255, -1)
    img[mask > 0] = target_colour
    return img, mask


# --------------------------------------------------------------------------- #
# degradations
# --------------------------------------------------------------------------- #


def degrade_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale all channels equally — a pure intensity change, no colour shift."""
    return to_uint8(to_float(img) * factor)


def degrade_colour_cast(img: np.ndarray, gains=(1.25, 1.0, 0.75)) -> np.ndarray:
    """Per-channel gains — a change in the illuminant's *colour*."""
    from shared import synth

    return synth.colour_cast(img, gains=gains)


def degrade_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """A non-linear tone curve, as a display or camera would apply."""
    return to_uint8(np.power(to_float(img), gamma))


DEGRADATIONS: dict[str, Callable] = {
    "Brightness scale": degrade_brightness,
    "Colour cast": lambda a, s: degrade_colour_cast(a, (1.0 + s, 1.0, 1.0 - s)),
    "Gamma": degrade_gamma,
}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def channel_shift(before: np.ndarray, after: np.ndarray, space: str,
                  mask: np.ndarray, chroma_only: bool = True) -> float:
    """Mean absolute coordinate change inside ``mask``, in that space's own units.

    Hue is handled separately because it is **circular**: a shift from 179 to 0
    is one unit, not 179. Treating it linearly would report an enormous, fake
    change every time a red object's hue crosses the wrap point.
    """
    conv = SPACES[space]
    a = conv(before).astype(np.float64)
    b = conv(after).astype(np.float64)
    channels = CHROMA_CHANNELS[space] if chroma_only else tuple(range(a.shape[2]))
    sel = mask > 0
    if not sel.any():
        return 0.0

    diffs = []
    for c in channels:
        d = np.abs(a[..., c][sel] - b[..., c][sel])
        if space == "HSV" and c == 0:
            d = np.minimum(d, (OPENCV_HUE_MAX + 1) - d)  # circular distance
        diffs.append(float(d.mean()))
    return float(np.mean(diffs))


def threshold_in_space(img: np.ndarray, space: str, reference: np.ndarray,
                       mask: np.ndarray, tolerance: float = 25.0) -> np.ndarray:
    """Segment by nearest-colour threshold, using statistics from a reference image.

    The threshold is built **once** from the reference and then applied unchanged
    to the degraded image — which is the whole point. Re-tuning it per image would
    measure nothing.
    """
    conv = SPACES[space]
    ref = conv(reference).astype(np.float64)
    target = ref[mask > 0].mean(axis=0)

    test = conv(img).astype(np.float64)
    channels = CHROMA_CHANNELS[space]

    dist = np.zeros(test.shape[:2], np.float64)
    for c in channels:
        d = np.abs(test[..., c] - target[c])
        if space == "HSV" and c == 0:
            d = np.minimum(d, (OPENCV_HUE_MAX + 1) - d)
        dist += d**2
    dist = np.sqrt(dist)
    return (dist <= tolerance).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs, each with a **human-traced region whose colour is distinct
#: from its surroundings** — the thing a colour threshold is written to find.
#: Selected by the Lab chroma distance between the region's mean and the rest of
#: the frame, which is the axis that decides whether any colour space has
#: something to separate; none of `tools/select_images.py`'s stock axes measures
#: it, because it is a property of a region rather than of a picture.
#:
#: Each entry is ``(annotator, region label)`` — which person's segmentation and
#: which region of it — so the target is reproducible and attributable.
SEGMENTS: dict[str, tuple[int, int]] = {
    "lobsters_and_wine": (0, 7),      # chroma distance 63.6 - red on grey quay
    "stacked_timber": (4, 1),         #                  63.1 - orange timber
    "anteater_at_sunset": (0, 2),     #                  59.2
    "kabuki_pair": (2, 29),           #                  55.3 - a yellow kimono
    "tomato_stall": (0, 18),          #                  54.6 - a crate of tomatoes
    "kalmar_castle": (3, 30),         #                  53.5
    "yellow_trousers": (4, 23),       #                  48.1
    "woman_in_blue_dress": (1, 1),    #                  46.8
    "red_robed_figures": (2, 6),      #                  46.8
    "red_sports_car": (3, 2),         #                  46.1
    "westminster_pair": (5, 6),       #                  43.9
    "green_field_worker": (3, 10),    #                  40.2 - the least distinct here
}

IMAGES = tuple(SEGMENTS)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs, RGB."""
    from shared import io

    return io.real_photo(name)


def load_target(name: str) -> np.ndarray:
    """The human-traced region a colour threshold is supposed to find.

    One person's segmentation, one region of it, named in `SEGMENTS`. Not the
    largest region — on a photograph that is usually the sky.
    """
    from shared import bsds

    annotator, label = SEGMENTS[name]
    segmentation = bsds.load_annotations(name)[annotator]["segmentation"]
    return ((segmentation == label).astype(np.uint8)) * 255


def chroma_distance(img: np.ndarray, mask: np.ndarray) -> float:
    """Lab a/b distance between the masked region's mean colour and the rest.

    The selection axis, recomputed here. It is the honest predictor of whether a
    colour threshold can work at all: a region whose chroma matches its surround
    cannot be separated by colour in any space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    inside = mask > 0
    if not inside.any() or inside.all():
        return 0.0
    return float(np.linalg.norm(lab[inside][:, 1:].mean(0) - lab[~inside][:, 1:].mean(0)))


BRIGHTNESS_LEVELS = (1.0, 0.8, 0.6, 0.4, 1.3)
CAST_LEVELS = (0.0, 0.1, 0.2, 0.35, 0.5)
GAMMA_LEVELS = (1.0, 1.4, 2.0, 0.7, 0.5)


def stability_table(degradation: str = "Brightness scale", levels=BRIGHTNESS_LEVELS, seed: int = 0):
    """How far each space's chroma coordinates move under a degradation.

    Lower is more invariant. Because the units differ between spaces, the
    *shape* across levels matters more than the absolute value — a space that is
    genuinely invariant stays flat.
    """
    img, masks = colour_chart(seed=seed)
    fn = DEGRADATIONS[degradation]
    rows = []
    for level in levels:
        degraded = fn(img, level)
        row: dict[str, float] = {"level": level}
        for space in SPACES:
            shifts = [channel_shift(img, degraded, space, m) for m in masks]
            row[space] = round(float(np.mean(shifts)), 3)
        rows.append(row)
    return rows


def segmentation_robustness(degradation: str = "Brightness scale", levels=BRIGHTNESS_LEVELS,
                            tolerance: float = 25.0, seeds=(0, 1, 2)):
    """The practical test: does a threshold tuned once still work after the change?

    IoU against the true object mask, with the threshold frozen at its
    reference-image value. This is what actually breaks in deployed code.
    """
    fn = DEGRADATIONS[degradation]
    rows = []
    for level in levels:
        row: dict[str, float] = {"level": level}
        for space in SPACES:
            scores = []
            for seed in seeds:
                img, mask = object_scene(seed=seed)
                degraded = fn(img, level)
                pred = threshold_in_space(degraded, space, img, mask, tolerance)
                scores.append(iou(pred, mask))
            row[space] = round(float(np.mean(scores)), 4)
        rows.append(row)
    return rows


def compare_all_degradations(seeds=(0, 1, 2), tolerance: float = 25.0):
    """One row per (space, degradation): the worst IoU across that sweep.

    The summary that answers "which space should I use", and shows that the
    answer depends on which degradation you expect.
    """
    sweeps = {
        "Brightness scale": BRIGHTNESS_LEVELS,
        "Colour cast": CAST_LEVELS,
        "Gamma": GAMMA_LEVELS,
    }
    rows = []
    for space in SPACES:
        row: dict[str, float | str] = {"space": space}
        for degradation, levels in sweeps.items():
            scores = []
            fn = DEGRADATIONS[degradation]
            for level in levels:
                per_seed = []
                for seed in seeds:
                    img, mask = object_scene(seed=seed)
                    pred = threshold_in_space(fn(img, level), space, img, mask, tolerance)
                    per_seed.append(iou(pred, mask))
                scores.append(float(np.mean(per_seed)))
            row[f"{degradation} worst"] = round(float(np.min(scores)), 4)
            row[f"{degradation} mean"] = round(float(np.mean(scores)), 4)
        rows.append(row)
    return rows


def white_balance_rescue(levels=CAST_LEVELS, seeds=(0, 1, 2), tolerance: float = 25.0):
    """Does white-balancing first fix what no colour space fixes alone?

    The constructive conclusion. If a grey-world correction restores the IoU that
    the cast destroyed, then the right answer is not "pick a better space" but
    "correct the illuminant, then pick any space".
    """
    from shared import io as _io  # noqa: F401

    rows = []
    for level in levels:
        row: dict[str, float] = {"cast_level": level}
        for space in SPACES:
            raw, corrected = [], []
            for seed in seeds:
                img, mask = object_scene(seed=seed)
                cast = degrade_colour_cast(img, (1.0 + level, 1.0, 1.0 - level))
                raw.append(iou(threshold_in_space(cast, space, img, mask, tolerance), mask))

                f = to_float(cast)
                means = f.reshape(-1, 3).mean(axis=0)
                balanced = to_uint8(f * (float(means.mean()) / np.maximum(means, EPS)))
                corrected.append(
                    iou(threshold_in_space(balanced, space, img, mask, tolerance), mask)
                )
            row[f"{space} raw"] = round(float(np.mean(raw)), 4)
            row[f"{space} balanced"] = round(float(np.mean(corrected)), 4)
        rows.append(row)
    return rows


#: Tolerance used on the photographs, **per space**. A single shared value would
#: decide the comparison on its own: `sweep_photo_tolerance` finds Lab and YCrCb
#: peaking at 25 and RGB and HSV at 60, so any one number hands the result to
#: whichever space it happens to suit.
#:
#: They differ because the spaces do not share units. Lab's a/b run about
#: +-100 around a neutral axis while RGB spans 0-255 in three correlated
#: channels, so "distance 25" is a far wider net in one than the other. Every
#: space is therefore given its own best value and compared at its own best.
PHOTO_TOLERANCE = {
    "RGB": 60.0,
    "HSV": 60.0,
    "Lab": 25.0,
    "YCrCb": 25.0,
    "Normalised RGB": 25.0,
}

PHOTO_DEGRADATIONS = {
    "Brightness x0.6": lambda img: degrade_brightness(img, 0.6),
    "Brightness x1.3": lambda img: degrade_brightness(img, 1.3),
    "Warm cast": lambda img: degrade_colour_cast(img, (1.25, 1.0, 0.75)),
    "Cool cast": lambda img: degrade_colour_cast(img, (0.78, 1.0, 1.28)),
    "Gamma 2.0": lambda img: degrade_gamma(img, 2.0),
}


def photo_robustness(images=None, tolerance=None, degradations=None):
    """The practical question, asked of human-traced regions in real photographs.

    A colour threshold is tuned on the original — its target colour is the mean
    of the traced region — and then applied **unchanged** to a degraded copy. The
    score is IoU against the person's own mask, so the undegraded column is the
    ceiling each space can reach at all and every other column is what a change
    of light costs it.

    This is the arm the project was missing. Flat synthetic patches make every
    space look better than it is: a real region has shadow, highlight and
    texture in it, and the spaces separate differently once it does.
    """
    from shared.metrics import iou

    images = IMAGES if images is None else images
    degradations = PHOTO_DEGRADATIONS if degradations is None else degradations
    tolerance = PHOTO_TOLERANCE if tolerance is None else tolerance

    rows = []
    for space in SPACES:
        tol = tolerance[space] if isinstance(tolerance, dict) else float(tolerance)
        acc: dict[str, list[float]] = {"none": []}
        for name in images:
            clean = load_scene(name)
            truth = load_target(name)
            acc["none"].append(iou(threshold_in_space(clean, space, clean, truth,
                                                      tol), truth))
            for label, fn in degradations.items():
                degraded = fn(clean)
                found = threshold_in_space(degraded, space, clean, truth, tol)
                acc.setdefault(label, []).append(iou(found, truth))

        row: dict[str, float | str] = {"space": space, "tolerance": tol}
        row["undegraded_iou"] = round(float(np.mean(acc["none"])), 4)
        for label in degradations:
            row[label] = round(float(np.mean(acc[label])), 4)
        row["worst_case"] = round(min(row[label] for label in degradations), 4)
        row["mean_loss"] = round(row["undegraded_iou"]
                                 - float(np.mean([row[label] for label in degradations])), 4)
        rows.append(row)
    return rows


def photo_robustness_per_image(images=None, tolerance=None):
    """Per-photograph IoU under each degradation, for the front figure."""
    from shared.metrics import iou

    images = IMAGES if images is None else images
    tolerance = PHOTO_TOLERANCE if tolerance is None else tolerance
    rows = []
    for name in images:
        clean = load_scene(name)
        truth = load_target(name)
        row: dict[str, float | str] = {
            "image": name,
            "chroma_distance": round(chroma_distance(clean, truth), 1),
        }
        for space in SPACES:
            tol = tolerance[space] if isinstance(tolerance, dict) else float(tolerance)
            row[f"{space} clean"] = round(
                iou(threshold_in_space(clean, space, clean, truth, tol), truth), 4)
            warm = degrade_colour_cast(clean, (1.25, 1.0, 0.75))
            row[f"{space} warm"] = round(
                iou(threshold_in_space(warm, space, clean, truth, tol), truth), 4)
        rows.append(row)
    return rows


def sweep_photo_tolerance(tolerances=(15.0, 25.0, 35.0, 45.0, 60.0, 80.0), images=None):
    """Which tolerance to use on photographs, chosen rather than assumed.

    Too tight and a textured region is missed entirely; too loose and everything
    is selected. Reported so the operating point is a measured choice.
    """
    from shared.metrics import iou

    images = IMAGES if images is None else images
    rows = []
    for tol in tolerances:
        row: dict[str, float] = {"tolerance": tol}
        for space in SPACES:
            scores = []
            for name in images:
                clean = load_scene(name)
                truth = load_target(name)
                scores.append(iou(threshold_in_space(clean, space, clean, truth, tol),
                                  truth))
            row[space] = round(float(np.mean(scores)), 4)
        rows.append(row)
    return rows


def hue_range_demonstration():
    """Show the 0-179 trap producing a concrete wrong answer.

    A pure red object has hue 0. Code written for a 0-359 wheel that looks for
    "around 120 for green" lands on 120 in OpenCV's scale, which is **240
    degrees** — blue. The table makes the off-by-two explicit.
    """
    rows = []
    for name, rgb, degrees in (
        ("red", (255, 0, 0), 0),
        ("yellow", (255, 255, 0), 60),
        ("green", (0, 255, 0), 120),
        ("cyan", (0, 255, 255), 180),
        ("blue", (0, 0, 255), 240),
        ("magenta", (255, 0, 255), 300),
    ):
        patch = np.zeros((8, 8, 3), np.uint8)
        patch[:] = rgb
        hue = int(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)[0, 0, 0])
        rows.append(
            {
                "colour": name,
                "true_degrees": degrees,
                "opencv_hue": hue,
                "naive_degrees_reading": hue,
                "correct_degrees": hue * 2,
            }
        )
    return rows


def convert(img: np.ndarray, space: str = "HSV") -> np.ndarray:
    return SPACES[space](img)
