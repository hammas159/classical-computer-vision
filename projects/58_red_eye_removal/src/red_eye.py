"""Red-eye removal: colour thresholding, morphology, and where it goes wrong.

The question
------------
Red-eye is the flash reflecting off the retina. The fix is simple — find the red
pupils, desaturate them — and the failure is the interesting part.

> **The claim under test:** a red-eye detector that only looks at colour cannot
> distinguish a red pupil from anything else red in the frame. Adding the
> **geometric** constraint that a pupil is small, round and inside a detected
> face eliminates almost all of those false positives — and the size of that
> improvement is the measurement.

The correction itself has a subtlety worth measuring too:

> Setting the red channel to zero is what most tutorials do, and it produces a
> **black hole** where the pupil was. Replacing red with the *mean of green and
> blue* preserves the pupil's luminance, so the eye still looks like an eye. The
> difference is visible and scoreable against the pre-flash original.

What is measured on what
------------------------
There are two arms, and they disagree, which is the point.

The **generated** scene is a face with known pupil positions, red-eye painted in
at a known intensity, and drawn red distractors. It gives an exact truth mask,
and it flatters the geometric detectors: a drawn disc is something the shape
filter is entitled to reject.

The **real** arm is twelve photographs. Six are portraits with red-eye planted at
recorded pupil positions — the face, the skin and every red object in the frame
are photographed, and only the pupil recolouring is synthetic, because a
photograph that already has red-eye carries no record of what it looked like
before. The other six contain no red-eye at all, so truth is **empty** and every
detection is a false positive with no metric choice to argue about.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import ensure_rgb, to_float, to_gray, to_uint8
from shared.metrics import iou, psnr

EPS = 1e-9

FACE_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_eye.xml"
_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def cascade(path: str) -> cv2.CascadeClassifier:
    """Load and cache a bundled Haar cascade.

    🚨 OpenCV 5 removed these XML files from ``cv2/data/``. This repo pins
    ``opencv-python-headless<5`` for that reason; on OpenCV 5 the classifier
    loads *empty* and silently detects nothing, which looks like a broken
    detector rather than a missing file.
    """
    if path not in _CASCADES:
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(
                f"Haar cascade failed to load from {path}. On OpenCV 5 the bundled "
                "cascade XMLs are gone; this repo pins opencv-python-headless<5."
            )
        _CASCADES[path] = c
    return _CASCADES[path]


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def redness(img: np.ndarray) -> np.ndarray:
    """Redness map: how much the red channel exceeds green and blue.

    Not simply "the red channel is high" — a white highlight has a high red
    channel too. What distinguishes red-eye is red being high **relative to** the
    other two, which is what this computes.
    """
    f = to_float(img).astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return np.clip(r - np.maximum(g, b), 0.0, 1.0)


def detect_colour_only(img: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """Threshold the redness map. The naive detector, and the control.

    Finds every red thing in the picture: a red jumper, a red sign, a red flower.
    Its false-positive rate is the number the geometric methods have to beat.
    """
    mask = (redness(img) > threshold).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_colour_shape(img: np.ndarray, threshold: float = 0.25,
                        min_area: int = 12, max_area: int = 4000,
                        min_circularity: float = 0.5) -> np.ndarray:
    """Redness plus geometry: keep only small, round, compact blobs.

    A pupil is approximately circular and occupies a small, bounded area. A red
    jumper is neither. Filtering on circularity and area is cheap and removes the
    large-region false positives without needing a face at all.
    """
    raw = detect_colour_only(img, threshold)
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(raw)
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / max(perimeter**2, EPS)
        if circularity < min_circularity:
            continue
        cv2.drawContours(out, [c], -1, 255, -1)
    return out


def detect_face_constrained(img: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """Redness and geometry, restricted to inside a detected face.

    The strongest constraint available without learning anything: eyes are in
    faces. One pre-trained, non-deep component is used — OpenCV's bundled Haar
    cascade, trained by someone else — and it is stated rather than hidden.

    The search is narrowed further to the **upper 60%** of the face box, because
    eyes are never in the chin.
    """
    gray = to_gray(img)
    faces = cascade(FACE_CASCADE_FILE).detectMultiScale(gray, 1.1, 5)
    out = np.zeros(gray.shape, np.uint8)
    if len(faces) == 0:
        return out

    candidate = detect_colour_shape(img, threshold)
    for (x, y, w, h) in faces:
        region = np.zeros_like(out)
        region[y : y + int(h * 0.6), x : x + w] = 255
        out = cv2.bitwise_or(out, cv2.bitwise_and(candidate, region))
    return out


def detect_eye_constrained(img: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """Restricted to inside detected *eye* regions — the tightest constraint.

    More precise than the face box and more fragile: the eye cascade fails more
    often than the face cascade, so this should trade recall for precision in a
    measurable way.
    """
    gray = to_gray(img)
    eyes = cascade(EYE_CASCADE_FILE).detectMultiScale(gray, 1.1, 6)
    out = np.zeros(gray.shape, np.uint8)
    if len(eyes) == 0:
        return out

    candidate = detect_colour_shape(img, threshold)
    for (x, y, w, h) in eyes:
        region = np.zeros_like(out)
        region[y : y + h, x : x + w] = 255
        out = cv2.bitwise_or(out, cv2.bitwise_and(candidate, region))
    return out


DETECTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Colour only (control)": detect_colour_only,
    "Colour + shape": detect_colour_shape,
    "Face-constrained": detect_face_constrained,
    "Eye-constrained": detect_eye_constrained,
}


# --------------------------------------------------------------------------- #
# correction
# --------------------------------------------------------------------------- #


def correct_zero_red(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set the red channel to zero inside the mask. What most tutorials do.

    The pupil loses roughly a third of its luminance and turns into a dark blue
    hole. It removes the red and it does not look like an eye.
    """
    out = img.copy()
    sel = mask > 0
    out[..., 0][sel] = 0
    return out


def correct_mean_gb(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace red with the mean of green and blue.

    Keeps the pixel's luminance approximately unchanged while removing the colour
    cast, so the pupil stays a plausible dark grey rather than becoming a void.
    """
    out = img.copy()
    sel = mask > 0
    g = out[..., 1][sel].astype(np.float32)
    b = out[..., 2][sel].astype(np.float32)
    out[..., 0][sel] = np.clip((g + b) / 2.0, 0, 255).astype(np.uint8)
    return out


def correct_desaturate(img: np.ndarray, mask: np.ndarray, keep: float = 0.15) -> np.ndarray:
    """Blend the masked region toward its own grayscale value.

    ``keep`` retains a little colour so the correction does not read as a flat
    patch. Blending with a feathered mask also avoids a hard edge at the pupil
    boundary, which is the giveaway that an image has been retouched.
    """
    out = to_float(img)
    gray = to_float(to_gray(img))[..., None]
    alpha = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.2)[..., None]
    blended = out * keep + gray * (1.0 - keep)
    return to_uint8(out * (1 - alpha) + blended * alpha)


CORRECTIONS: dict[str, Callable] = {
    "Zero red channel": correct_zero_red,
    "Mean of G and B": correct_mean_gb,
    "Desaturate (feathered)": correct_desaturate,
}


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #


def make_scene(size: int = 480, red_strength: float = 0.75, distractors: int = 4,
               pupil_radius: int = 9, seed: int = 0):
    """A face with red-eye, plus red distractor objects.

    Returns ``(flash_image, pre_flash_image, pupil_mask)``. The pre-flash image
    is the correction's ground truth: it is what the photo *should* look like.

    The distractors are red, round and the wrong size or in the wrong place —
    designed so that colour alone cannot reject them but geometry can.
    """
    from shared import io, synth

    rng = np.random.default_rng(seed)

    # a real face gives the Haar cascades something genuine to find
    face = synth._astronaut_face_crop()
    face = cv2.resize(face, (int(size * 0.55), int(size * 0.65)), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:] = (90, 105, 120)
    fy, fx = int(size * 0.12), int(size * 0.22)
    canvas[fy : fy + face.shape[0], fx : fx + face.shape[1]] = face

    gray = to_gray(canvas)
    eyes = cascade(EYE_CASCADE_FILE).detectMultiScale(gray, 1.1, 6)

    pupils = []
    if len(eyes) >= 1:
        for (ex, ey, ew, eh) in eyes[:2]:
            pupils.append((ex + ew // 2, ey + eh // 2))
    else:
        # fall back to anatomical proportions if the cascade misses
        pupils = [
            (fx + int(face.shape[1] * 0.33), fy + int(face.shape[0] * 0.42)),
            (fx + int(face.shape[1] * 0.67), fy + int(face.shape[0] * 0.42)),
        ]

    pre_flash = canvas.copy()
    mask = np.zeros((size, size), np.uint8)
    for (px, py) in pupils:
        cv2.circle(pre_flash, (px, py), pupil_radius, (28, 26, 30), -1)
        cv2.circle(mask, (px, py), pupil_radius, 255, -1)

    flash = pre_flash.copy()
    red = to_float(flash)
    sel = mask > 0
    red[..., 0][sel] = np.clip(red[..., 0][sel] + red_strength, 0, 1)
    flash = to_uint8(red)

    for _ in range(distractors):
        x = int(rng.integers(0, size - 60))
        y = int(rng.integers(int(size * 0.75), size - 40))
        kind = int(rng.integers(0, 2))
        if kind == 0:
            cv2.circle(flash, (x + 25, y + 20), int(rng.integers(20, 34)), (215, 40, 40), -1)
            cv2.circle(pre_flash, (x + 25, y + 20), int(rng.integers(20, 34)), (215, 40, 40), -1)
        else:
            cv2.rectangle(flash, (x, y), (x + 55, y + 35), (200, 45, 50), -1)
            cv2.rectangle(pre_flash, (x, y), (x + 55, y + 35), (200, 45, 50), -1)

    flash = synth.gaussian_noise(flash, sigma=2.0, seed=seed)
    return flash, pre_flash, mask


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# real photographs
# --------------------------------------------------------------------------- #

#: The six portraits red-eye is planted in, with the pupil centres and radii.
#:
#: The centres were found once with OpenCV's eye cascade and then checked by
#: eye against a pixel grid. The ones the cascade **missed** are marked `"hand"`
#: and were read off that grid directly. Recording which is which
#: matters: a table that mixes measured and hand-placed positions without saying
#: so is not reproducible, and the cascade's misses are themselves a result (it
#: found 12 of 16 pupils across these six photographs).
PORTRAITS: dict[str, dict] = {
    "two_women_in_headdress": {
        "pupils": [(324, 113, 5), (376, 107, 5), (72, 159, 5), (128, 162, 5)],
        "source": ["cascade"] * 4,
        "note": "two frontal faces, four pupils; the headdresses are dense red and white",
    },
    "girl_with_tulips": {
        "pupils": [(266, 87, 4), (303, 78, 4)],
        "source": ["cascade", "hand"],
        "note": "three red tulips fill the lower half of the frame",
    },
    "woman_in_red_scarf": {
        "pupils": [(95, 107, 5), (127, 109, 5)],
        "source": ["cascade", "cascade"],
        "note": "a large red scarf directly below the face",
    },
    "girl_in_pink_shirt": {
        "pupils": [(158, 101, 5), (187, 104, 4)],
        "source": ["cascade", "cascade"],
        "note": "an old scan with a magenta cast, which suppresses the redness cue",
    },
    "two_firefighters": {
        "pupils": [(114, 98, 3), (144, 98, 3), (322, 144, 3), (347, 147, 3)],
        "source": ["cascade", "hand", "hand", "hand"],
        "note": "small faces under helmets, in front of a red fire engine",
    },
    "woman_with_curly_hair": {
        "pupils": [(107, 169, 4), (153, 172, 4)],
        "source": ["cascade", "cascade"],
        "note": "red lipstick, the textbook false positive, in the same face",
    },
}

#: Photographs with no face and no red-eye at all. Truth here is **empty**, so
#: every detection is a false positive and no threshold choice can hide one.
CLEAN_PHOTOGRAPHS = (
    "orange_lichen_on_rock",
    "children_carrying_pots",
    "feather_duster_worms",
    "runners_in_the_stadium",
    "scattered_sweets",
    "red_brick_house",
)

#: All twelve of this project's photographs.
PHOTOGRAPHS = tuple(PORTRAITS) + CLEAN_PHOTOGRAPHS


def pupil_like_blobs(img: np.ndarray, threshold: float = 0.25) -> int:
    """How many small round red blobs a photograph already contains.

    This project's selection axis, and it is the **method's own response** rather
    than a proxy for it: `detect_colour_shape` up to the point where it would
    accept a blob. A photograph scoring high here is one where the geometric
    detector has somewhere to go wrong, and that is what the twelve were chosen
    to span — from 0 on the brick house to 68 on the lichen.
    """
    shaped = detect_colour_shape(img, threshold)
    n, _, _, _ = cv2.connectedComponentsWithStats((shaped > 0).astype(np.uint8), 8)
    return int(n - 1)


def plant_red_eye(img: np.ndarray, pupils, red_strength: float = 0.75):
    """Put flash red-eye into a real photograph, at stated pupil positions.

    Returns ``(flash, pre_flash, mask)`` to match `make_scene`, so both arms feed
    the same experiments.

    What is real and what is not, stated plainly: the face, the skin, the hair,
    the lighting and every red object in the frame are photographed. The pupil
    recolouring is synthetic, and it has to be — a photograph *with* red-eye
    carries no record of what it looked like before, so there would be nothing to
    score a correction against.

    The effect is modelled the way the physics works: the retina returns the
    flash, so red is *added* to the pupil rather than the pupil being painted a
    flat colour. The falloff is radial and the existing specular highlight is
    left alone, both of which a flat disc destroys.
    """
    pre_flash = ensure_rgb(img).copy()
    mask = np.zeros(pre_flash.shape[:2], np.uint8)
    glow = np.zeros(pre_flash.shape[:2], np.float32)

    for (px, py, radius) in pupils:
        cv2.circle(mask, (int(px), int(py)), int(radius), 255, -1)
        # radial falloff: strongest at the pupil centre, gone at its edge
        y0, y1 = max(0, py - radius), min(pre_flash.shape[0], py + radius + 1)
        x0, x1 = max(0, px - radius), min(pre_flash.shape[1], px + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(yy - py, xx - px) / max(radius, 1)
        glow[y0:y1, x0:x1] = np.maximum(glow[y0:y1, x0:x1],
                                        np.clip(1.0 - dist**2, 0.0, 1.0))

    flash = to_float(pre_flash).astype(np.float32)
    flash[..., 0] = np.clip(flash[..., 0] + red_strength * glow, 0.0, 1.0)
    # the retina reflects red, so green and blue fall slightly rather than hold
    flash[..., 1] = np.clip(flash[..., 1] - 0.15 * red_strength * glow, 0.0, 1.0)
    flash[..., 2] = np.clip(flash[..., 2] - 0.15 * red_strength * glow, 0.0, 1.0)
    return to_uint8(flash), pre_flash, mask


def portrait_scene(name: str, red_strength: float = 0.75):
    """One of the six portraits with red-eye planted, and its ground truth."""
    from shared import io

    entry = PORTRAITS[name]
    return plant_red_eye(io.real_photo(name), entry["pupils"], red_strength)


def evaluate_on_portraits(threshold: float = 0.25, red_strength: float = 0.75,
                          runs: int = 1):
    """Every detector on the six real portraits, scored against the true pupils."""
    acc = {n: {"iou": [], "recall": [], "fp_area": [], "ms": []} for n in DETECTORS}

    for name in PORTRAITS:
        flash, _, truth = portrait_scene(name, red_strength)
        truth_area = max(float((truth > 0).sum()), 1.0)
        for det, fn in DETECTORS.items():
            pred, timing = timeit(lambda f=fn: f(flash, threshold), runs=runs, warmup=0)
            acc[det]["iou"].append(iou(pred, truth))
            acc[det]["recall"].append(float((pred[truth > 0] > 0).mean()))
            acc[det]["fp_area"].append(
                float(((pred > 0) & (truth == 0)).sum()) / truth_area)
            acc[det]["ms"].append(timing.median_ms)

    return [
        {
            "detector": n,
            "iou": round(float(np.mean(a["iou"])), 4),
            "pupil_recall": round(float(np.mean(a["recall"])), 4),
            "false_positive_area_ratio": round(float(np.mean(a["fp_area"])), 3),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for n, a in acc.items()
    ]


def per_portrait(threshold: float = 0.25, red_strength: float = 0.75):
    """The same measurement, one row per photograph rather than averaged.

    Averages hide the two cases worth looking at: the magenta-cast scan where the
    redness cue is suppressed, and the portrait whose own lipstick is redder than
    the planted pupils.
    """
    rows = []
    for name, entry in PORTRAITS.items():
        flash, _, truth = portrait_scene(name, red_strength)
        row = {
            "photograph": name,
            "pupils": len(entry["pupils"]),
            "cascade_found": entry["source"].count("cascade"),
            "pupil_like_blobs": pupil_like_blobs(io_real(name)),
        }
        for det, fn in DETECTORS.items():
            pred = fn(flash, threshold)
            row[det] = round(iou(pred, truth), 4)
        rows.append(row)
    return rows


def io_real(name: str) -> np.ndarray:
    """`shared.io.real_photo`, imported lazily so the module stays cheap."""
    from shared import io

    return io.real_photo(name)


def false_positives_on_clean_photographs(threshold: float = 0.25):
    """Every detector on six photographs with **no red-eye in them at all**.

    The control that the generated scene cannot provide. Truth is empty, so
    every detected pixel is wrong and there is no metric choice to argue about.
    A red-eye remover is run over ordinary holiday photographs far more often
    than over photographs that need it, and what it does to them is the number
    a user actually experiences.
    """
    rows = []
    for name in CLEAN_PHOTOGRAPHS:
        img = io_real(name)
        row = {"photograph": name, "pupil_like_blobs": pupil_like_blobs(img, threshold)}
        for det, fn in DETECTORS.items():
            pred = fn(img, threshold)
            row[det] = int((pred > 0).sum())
        rows.append(row)
    return rows


def generated_versus_real_background(threshold: float = 0.25, scenes: int = 6):
    """False-positive area on the generated scene against the real photographs.

    The comparison that says whether the generated distractors are a fair stand-in
    for the red things in a real photograph. They are drawn discs and rectangles,
    which the shape filter is entitled to reject; lipstick, brake lights and
    sugar-shelled sweets are round, small and red.
    """
    generated = {r["detector"]: r for r in evaluate_detectors(
        scenes=scenes, threshold=threshold, runs=1)}
    real = {r["detector"]: r for r in evaluate_on_portraits(threshold=threshold)}
    return [
        {
            "detector": det,
            "generated_fp_area": generated[det]["false_positive_area_ratio"],
            "real_fp_area": real[det]["false_positive_area_ratio"],
            "generated_iou": generated[det]["iou"],
            "real_iou": real[det]["iou"],
        }
        for det in DETECTORS
    ]


def corrections_on_portraits(red_strength: float = 0.75):
    """Each correction on real skin, scored against the real photograph.

    The pre-flash here is a **photograph**, not a rendering, so the corrections
    are being asked to restore a real pupil rather than a flat grey disc. The
    true mask is used, so this measures correction alone.
    """
    rows = []
    for name in PORTRAITS:
        flash, pre_flash, truth = portrait_scene(name, red_strength)
        sel = truth > 0
        row = {"photograph": name}
        for corr, fn in CORRECTIONS.items():
            fixed = fn(flash, truth)
            row[corr] = round(float(psnr(fixed[sel].reshape(-1, 1, 3),
                                         pre_flash[sel].reshape(-1, 1, 3))), 2)
        row["did nothing"] = round(float(psnr(flash[sel].reshape(-1, 1, 3),
                                              pre_flash[sel].reshape(-1, 1, 3))), 2)
        rows.append(row)
    return rows


THRESHOLDS = (0.1, 0.18, 0.25, 0.35, 0.5)
STRENGTHS = (0.35, 0.5, 0.65, 0.8, 0.95)
DISTRACTOR_COUNTS = (0, 2, 4, 8)
#: Chosen to straddle **both** ends of the shape filter's working range: at
#: radius 2 the blob is below `min_area`, at 45 it is above `max_area`, and
#: both report 0. A sweep that only spans the middle would show a filter that
#: always works.
PUPIL_RADII = (2, 4, 9, 20, 32, 45)


def evaluate_detectors(scenes: int = 6, threshold: float = 0.25, distractors: int = 4,
                       red_strength: float = 0.75, runs: int = 3):
    """IoU, recall and false-positive area for every detector.

    False-positive **area** matters more than a count here: one huge wrongly
    detected red jumper is worse than several small mistakes, and a count treats
    them the same.
    """
    acc = {n: {"iou": [], "recall": [], "fp_area": [], "ms": []} for n in DETECTORS}

    for seed in range(scenes):
        flash, _, truth = make_scene(
            red_strength=red_strength, distractors=distractors, seed=seed
        )
        truth_area = max(float((truth > 0).sum()), 1.0)

        for name, fn in DETECTORS.items():
            pred, timing = timeit(
                lambda f=fn: f(flash, threshold) if f is not detect_colour_only
                else f(flash, threshold),
                runs=runs, warmup=1,
            )
            acc[name]["iou"].append(iou(pred, truth))
            acc[name]["recall"].append(float((pred[truth > 0] > 0).mean()))
            acc[name]["fp_area"].append(float(((pred > 0) & (truth == 0)).sum()) / truth_area)
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "detector": n,
            "iou": round(float(np.mean(a["iou"])), 4),
            "pupil_recall": round(float(np.mean(a["recall"])), 4),
            "false_positive_area_ratio": round(float(np.mean(a["fp_area"])), 3),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for n, a in acc.items()
    ]


def evaluate_corrections(scenes: int = 6, runs: int = 3):
    """Score each correction against the **pre-flash** original.

    Using the true pupil mask, so this measures the correction alone and cannot
    be blamed on detection. PSNR is computed over the pupil pixels only —
    whole-image PSNR barely moves when a few hundred pixels change.
    """
    acc = {n: {"pupil_psnr": [], "luminance_error": [], "ms": []} for n in CORRECTIONS}

    for seed in range(scenes):
        flash, pre_flash, mask = make_scene(seed=seed)
        sel = mask > 0
        for name, fn in CORRECTIONS.items():
            out, timing = timeit(lambda f=fn: f(flash, mask), runs=runs, warmup=1)
            a = to_float(out)[sel]
            b = to_float(pre_flash)[sel]
            mse = float(np.mean((a - b) ** 2))
            acc[name]["pupil_psnr"].append(
                float("inf") if mse <= EPS else float(10.0 * np.log10(1.0 / mse))
            )
            lum_out = float(to_float(to_gray(out))[sel].mean())
            lum_ref = float(to_float(to_gray(pre_flash))[sel].mean())
            acc[name]["luminance_error"].append(abs(lum_out - lum_ref))
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "correction": n,
            "pupil_psnr_db": round(float(np.mean(a["pupil_psnr"])), 3),
            "luminance_error": round(float(np.mean(a["luminance_error"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for n, a in acc.items()
    ]


def sweep_threshold(scenes: int = 6, thresholds=THRESHOLDS):
    """The redness threshold trades recall against false positives."""
    rows = []
    for t in thresholds:
        scored = evaluate_detectors(scenes=scenes, threshold=t, runs=1)
        row: dict[str, float] = {"threshold": t}
        for r in scored:
            row[f"{r['detector']} IoU"] = r["iou"]
            row[f"{r['detector']} FP"] = r["false_positive_area_ratio"]
        rows.append(row)
    return rows


def sweep_distractors(scenes: int = 6, counts=DISTRACTOR_COUNTS):
    """The central experiment: how much does each constraint buy?

    With no distractors every method looks fine. The colour-only detector should
    degrade linearly with the number of red objects while the geometric and
    face-constrained ones stay flat.
    """
    rows = []
    for n in counts:
        scored = evaluate_detectors(scenes=scenes, distractors=n, runs=1)
        row: dict[str, float | int] = {"distractors": n}
        for r in scored:
            row[r["detector"]] = r["false_positive_area_ratio"]
        rows.append(row)
    return rows


def sweep_red_strength(scenes: int = 6, strengths=STRENGTHS):
    """Faint red-eye is harder to find and easier to leave alone."""
    rows = []
    for s in strengths:
        scored = evaluate_detectors(scenes=scenes, red_strength=s, runs=1)
        row: dict[str, float] = {"red_strength": s}
        for r in scored:
            row[r["detector"]] = r["pupil_recall"]
        rows.append(row)
    return rows


def sweep_pupil_size(scenes: int = 6, radii=PUPIL_RADII):
    """The shape filter has area bounds, so very large or small pupils escape it.

    Any fixed geometric constraint has a working range, and finding it is more
    useful than claiming the filter works.
    """
    rows = []
    for r in radii:
        ious = {n: [] for n in DETECTORS}
        for seed in range(scenes):
            flash, _, truth = make_scene(pupil_radius=r, seed=seed)
            for name, fn in DETECTORS.items():
                ious[name].append(iou(fn(flash), truth))
        row: dict[str, float | int] = {"pupil_radius": r}
        for name, vals in ious.items():
            row[name] = round(float(np.mean(vals)), 4)
        rows.append(row)
    return rows


def end_to_end(scenes: int = 6, detector_name: str = "Face-constrained",
               correction: str = "Mean of G and B"):
    """Full pipeline with a *detected* mask rather than the true one.

    The realistic number. Comparing it with the correction-only table shows how
    much of the remaining error is detection's fault.
    """
    rows = []
    for name in DETECTORS:
        psnrs = []
        for seed in range(scenes):
            flash, pre_flash, truth = make_scene(seed=seed)
            mask = DETECTORS[name](flash)
            out = CORRECTIONS[correction](flash, mask)
            sel = truth > 0
            a = to_float(out)[sel]
            b = to_float(pre_flash)[sel]
            mse = float(np.mean((a - b) ** 2))
            psnrs.append(float("inf") if mse <= EPS else float(10.0 * np.log10(1.0 / mse)))
        rows.append(
            {
                "detector": name,
                "correction": correction,
                "pupil_psnr_db": round(float(np.mean(psnrs)), 3),
            }
        )
    return rows


def remove(img: np.ndarray, detector_name: str = "Face-constrained",
           correction: str = "Mean of G and B"):
    """Detect and correct in one call, for the UI."""
    mask = DETECTORS[detector_name](img)
    return mask, CORRECTIONS[correction](img, mask)


def sweep_threshold_on_portraits(thresholds=THRESHOLDS):
    """The redness threshold, measured on real photographs.

    Worth having separately from the generated sweep: the generated pupils are
    painted at one known intensity, so the threshold that suits them is the one
    that was painted in. Real skin, real pupils and real ambient colour move it.
    """
    rows = []
    for t in thresholds:
        scored = evaluate_on_portraits(threshold=t)
        clean = false_positives_on_clean_photographs(threshold=t)
        row: dict[str, float] = {"threshold": t}
        for r in scored:
            row[f"{r['detector']} IoU"] = r["iou"]
        row["clean-photo false positives"] = int(
            sum(c["Face-constrained"] for c in clean))
        row["colour-only false positives"] = int(
            sum(c["Colour only (control)"] for c in clean))
        rows.append(row)
    return rows


def end_to_end_on_portraits(correction: str = "Mean of G and B"):
    """Full pipeline on real photographs, with a *detected* mask.

    The number a user would actually get. Compared with `corrections_on_portraits`
    — which uses the true mask — it separates what detection costs from what the
    correction costs.
    """
    rows = []
    for name in DETECTORS:
        psnrs = []
        for photo in PORTRAITS:
            flash, pre_flash, truth = portrait_scene(photo)
            out = CORRECTIONS[correction](flash, DETECTORS[name](flash))
            sel = truth > 0
            mse = float(np.mean((to_float(out)[sel] - to_float(pre_flash)[sel]) ** 2))
            psnrs.append(float("inf") if mse <= EPS else float(10.0 * np.log10(1.0 / mse)))
        rows.append({
            "detector": name,
            "correction": correction,
            "pupil_psnr_db": round(float(np.mean(psnrs)), 3),
        })

    # the control that makes the rest readable
    nothing = []
    for photo in PORTRAITS:
        flash, pre_flash, truth = portrait_scene(photo)
        sel = truth > 0
        mse = float(np.mean((to_float(flash)[sel] - to_float(pre_flash)[sel]) ** 2))
        nothing.append(float("inf") if mse <= EPS else float(10.0 * np.log10(1.0 / mse)))
    rows.append({
        "detector": "Did nothing (control)",
        "correction": "none",
        "pupil_psnr_db": round(float(np.mean(nothing)), 3),
    })
    return rows
