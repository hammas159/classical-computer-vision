"""Industrial defect detection: six detectors, two controls, and an empty-truth arm.

The question
------------
An inspection camera looks at a surface that is supposed to be uniform. Anything
that breaks the uniformity is a defect. The whole family of classical methods is
one idea -- **model what the surface should look like, and flag the residual** --
and they differ only in what "should look like" means.

> **The claim under test:** a defect detector is scored by how well it finds
> defects. **That is half a metric.** Run on the same twelve surfaces with **no
> defect in them at all**, where the true answer is an empty mask, these
> detectors mark between **3.7% and 14.3%** of the surface. On an inspection line
> that is the number that stops production, and it is invisible to any benchmark
> built only on defective parts.

> **The detectors are complementary, not competing.** A **smear** -- a local loss
> of texture at unchanged brightness -- is found by the local standard deviation
> on 0.58 of attempts and by **four of the other five exactly never**, because no
> intensity residual can see it at all. A **scratch** is found by almost
> everything. Picking "the best defect detector" is picking which defects to
> miss.

> **And the surface decides more than the detector does.** The best detection
> rate any of the six achieves spans **0.25 to 1.00** across the twelve surfaces
> -- a wider spread than between any two detectors on one surface.

> **Pixel accuracy is unusable here** and is never reported: a defect covers
> **1.07%** of a surface, so flagging nothing is right **98.92%** of the time.

Where the ground truth comes from
---------------------------------
The surfaces are photographs; the defects are planted, with the mask recorded.
That is the only way to have an exact truth here -- a photograph of a *real*
defective part carries no mask, and drawing one by hand would make the truth a
matter of opinion.

The defect shapes are deliberately the kinds of thing that go wrong on a
production line: a **scratch** (a thin bright or dark line), a **blob** (a stain
or a dent), a **hole** (missing material), and a **smear** (a local loss of
texture). Each is applied at a stated strength so that the *severity* can be
swept rather than argued about.

The two controls
----------------
`Flag nothing` returns an empty mask and `Flag everything` returns a full one.
They bracket the detectors and make the class imbalance explicit: a defect covers
well under one per cent of a surface, so a detector that finds nothing scores
over 0.99 on pixel accuracy.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.io import real_photo, to_gray

EPS = 1e-9

#: The twelve surfaces, in order of **uniformity**: the standard deviation of a
#: heavily blurred copy, which is how much large-scale variation the surface has
#: before anything is done to it. A defect detector assumes there is none, so
#: this is the assumption itself, measured.
IMAGES = (
    "surface_fine_weave",
    "surface_brick_paving",
    "surface_sand_ripple",
    "surface_water_ripples",
    "surface_field_mosaic",
    "surface_pebbled_render",
    "surface_brick_wall",
    "surface_coarse_cloth",
    "surface_dry_grass",
    "surface_roof_slates",
    "surface_straw_thatch",
    "surface_knitted_fabric",
)

#: The blur used to measure large-scale variation. Wide enough that the surface's
#: own repeating unit is averaged away and only lighting and shape remain.
UNIFORMITY_SIGMA = 24.0

#: Defect strength in multiples of the surface's own residual noise ceiling
#: (`noise_ceiling`). At 8 the defect is far outside anything the texture
#: produces and every detector should find it; `sweep_severity` goes down to 1,
#: where it is indistinguishable from the surface by construction.
SEVERITY = 8.0


def load(name: str) -> np.ndarray:
    return real_photo(name)


def uniformity(image: np.ndarray, sigma: float = UNIFORMITY_SIGMA) -> float:
    """Standard deviation of a heavily blurred copy, in grey levels.

    Low means the surface really is uniform at large scale, which is what every
    detector here assumes. High means there is shading or structure that a
    residual method will report as a defect whether or not one is present.
    """
    g = to_gray(image).astype(np.float32)
    return float(cv2.GaussianBlur(g, (0, 0), sigma).std())


def surface_contrast(image: np.ndarray) -> float:
    """The surface's own local contrast, in grey levels."""
    g = to_gray(image).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 3.0)).std())


def noise_ceiling(image: np.ndarray, size: int = 9) -> float:
    """Robust spread of the surface's own median-filter residual.

    **This is what a defect has to clear**, and it is not the same as the
    surface's contrast. A first version scaled defects by `surface_contrast`,
    which made every defect about two and a half times the texture it was hiding
    in on every surface -- so every detector scored near zero everywhere and the
    comparison measured nothing but the construction.

    Scaling by this instead makes `severity` mean something: severity 1.0 is a
    defect one robust sigma above the surface's own residual, severity 8 is a
    defect no texture could produce.
    """
    g = to_gray(image)
    r = np.abs(g.astype(np.float32) - cv2.medianBlur(g, size).astype(np.float32))
    med = float(np.median(r))
    return 1.4826 * float(np.median(np.abs(r - med)))


# --------------------------------------------------------------------------- #
# planting defects
# --------------------------------------------------------------------------- #

DEFECTS = ("scratch", "blob", "hole", "smear")


def plant(image: np.ndarray, kind: str, seed: int = 0,
          severity: float = SEVERITY):
    """Put one defect into a surface. Returns ``(defective, mask)``.

    The strength is scaled by the surface's **own** local contrast, so a defect
    of severity 0.6 is equally visible on a smooth weave and on tree bark. Using
    a fixed number of grey levels instead would make the comparison across
    surfaces a comparison of their contrast.
    """
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32).copy()
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    amplitude = severity * noise_ceiling(image)

    if kind == "scratch":
        x0, y0 = rng.integers(w // 6, 5 * w // 6), rng.integers(h // 6, 5 * h // 6)
        angle = rng.uniform(0, np.pi)
        length = int(min(h, w) * rng.uniform(0.25, 0.45))
        x1 = int(np.clip(x0 + length * np.cos(angle), 0, w - 1))
        y1 = int(np.clip(y0 + length * np.sin(angle), 0, h - 1))
        thickness = int(rng.integers(2, 5))
        cv2.line(mask, (int(x0), int(y0)), (x1, y1), 255, thickness)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        out[mask > 0] += sign * amplitude

    elif kind == "blob":
        cx, cy = rng.integers(w // 5, 4 * w // 5), rng.integers(h // 5, 4 * h // 5)
        radius = int(min(h, w) * rng.uniform(0.05, 0.10))
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0),
                                radius * 0.3)[..., None]
        out -= soft * amplitude

    elif kind == "hole":
        cx, cy = rng.integers(w // 5, 4 * w // 5), rng.integers(h // 5, 4 * h // 5)
        radius = int(min(h, w) * rng.uniform(0.04, 0.08))
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
        # missing material: flat dark, with no texture at all
        out[mask > 0] = float(np.percentile(to_gray(image), 5))

    elif kind == "smear":
        cx, cy = rng.integers(w // 5, 4 * w // 5), rng.integers(h // 5, 4 * h // 5)
        radius = int(min(h, w) * rng.uniform(0.07, 0.12))
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
        blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 6.0)
        sel = mask > 0
        out[sel] = blurred[sel]

    else:
        raise ValueError(kind)

    return np.clip(out, 0, 255).astype(np.uint8), mask


def clean(image: np.ndarray):
    """The surface with nothing wrong with it, and an empty truth mask.

    The arm that makes a false-alarm rate measurable. On a real line most parts
    are good, so this is the case the detector spends almost all of its time on.
    """
    return image.copy(), np.zeros(image.shape[:2], np.uint8)


# --------------------------------------------------------------------------- #
# the detectors
# --------------------------------------------------------------------------- #

#: Every detector flags a pixel whose residual exceeds this many robust standard
#: deviations of that detector's own residual. Sharing the rule is what makes the
#: comparison one of *models* rather than one of thresholds.
K_SIGMA = 4.0

#: Minimum blob area, in pixels. Below this a detection is noise, and every
#: detector gets the same floor.
MIN_AREA = 40


def _flag(residual: np.ndarray, k: float = K_SIGMA,
          min_area: int = MIN_AREA) -> np.ndarray:
    """Threshold a residual at k robust sigmas and drop specks.

    The spread is the median absolute deviation rather than the standard
    deviation, because the defect is in the data being measured and would inflate
    a standard deviation enough to hide itself.
    """
    r = np.abs(residual)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    sigma = 1.4826 * mad
    mask = (r > med + k * max(sigma, EPS)).astype(np.uint8) * 255
    # CLOSE before the area filter, never OPEN. Thresholding a thin scratch
    # leaves a dotted line; a 3x3 opening erodes a two-pixel-wide line out of
    # existence, which is what the first version of this function did -- 2011
    # thresholded pixels became 82 and every detector scored zero on scratches.
    # Closing reconnects the dots, and the area filter then removes real specks.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, 4] >= min_area:
            out[labels == i] = 255
    return out


def detect_nothing(image: np.ndarray, **kw) -> np.ndarray:
    """Flag nothing. The control that exposes the class imbalance."""
    return np.zeros(image.shape[:2], np.uint8)


def detect_everything(image: np.ndarray, **kw) -> np.ndarray:
    """Flag everything. The other end."""
    return np.full(image.shape[:2], 255, np.uint8)


def detect_median_residual(image: np.ndarray, size: int = 9, **kw) -> np.ndarray:
    """The surface is what a median filter says it is; the defect is the rest.

    The simplest model that survives a textured surface: a median is unmoved by a
    thin scratch crossing its window, so the scratch appears in the residual.
    """
    g = to_gray(image)
    return _flag(g.astype(np.float32) - cv2.medianBlur(g, size).astype(np.float32))


def detect_gaussian_residual(image: np.ndarray, sigma: float = 4.0,
                             **kw) -> np.ndarray:
    """The same idea with a Gaussian. It smooths *across* a defect, so part of
    the defect ends up in the model and is subtracted from itself."""
    g = to_gray(image).astype(np.float32)
    return _flag(g - cv2.GaussianBlur(g, (0, 0), sigma))


def detect_morphological(image: np.ndarray, size: int = 15, **kw) -> np.ndarray:
    """Top-hat plus black-hat: what is too bright and too dark for its surround.

    The classical inspection operator. Unlike a symmetric residual it is explicit
    about polarity, which matters when only one kind of defect is possible.
    """
    g = to_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    black = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    return _flag(top + black)


def detect_local_statistics(image: np.ndarray, window: int = 21,
                            **kw) -> np.ndarray:
    """Flag where the local standard deviation departs from the surface's own.

    A *smear* has the same mean as its surround and less texture, so no intensity
    residual sees it. This is the detector that can, which is why the defect
    catalogue includes one.
    """
    g = to_gray(image).astype(np.float32)
    mean = cv2.blur(g, (window, window))
    sq = cv2.blur(g * g, (window, window))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return _flag(std - float(np.median(std)))


def detect_fourier(image: np.ndarray, cut: float = 0.06, **kw) -> np.ndarray:
    """Remove the surface's periodic structure in the frequency domain.

    A woven or laid surface is periodic, so its energy sits in a few spikes.
    Notching those out leaves whatever is *not* part of the weave -- which is the
    one model here that uses the surface's regularity rather than working around
    it.
    """
    g = to_gray(image).astype(np.float32)
    h, w = g.shape
    F = np.fft.fftshift(np.fft.fft2(g))
    magnitude = np.abs(F)

    # keep the DC neighbourhood, notch the strongest periodic peaks outside it
    yy, xx = np.ogrid[:h, :w]
    radius = np.hypot(yy - h / 2, xx - w / 2)
    outside = radius > max(h, w) * 0.02
    if outside.any():
        threshold = float(np.percentile(magnitude[outside], 100 * (1 - cut)))
        F[outside & (magnitude > threshold)] = 0
    residual = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    return _flag(g - residual)


def detect_phase_only(image: np.ndarray, **kw) -> np.ndarray:
    """Spectral residual saliency: reconstruct from phase with a flattened
    log-magnitude, which suppresses whatever is statistically ordinary.

    Hou and Zhang's saliency detector, used here as a defect detector because a
    defect is exactly what is not ordinary on a uniform surface.
    """
    g = to_gray(image).astype(np.float32)
    F = np.fft.fft2(g)
    log_magnitude = np.log(np.abs(F) + 1.0)
    phase = np.angle(F)
    smooth = cv2.blur(log_magnitude, (3, 3))
    spectral_residual = log_magnitude - smooth
    reconstructed = np.abs(np.fft.ifft2(
        np.exp(spectral_residual + 1j * phase))) ** 2
    return _flag(cv2.GaussianBlur(reconstructed, (0, 0), 3.0))


DETECTORS: dict[str, Callable] = {
    "Flag nothing (control)": detect_nothing,
    "Flag everything (control)": detect_everything,
    "Median residual": detect_median_residual,
    "Gaussian residual": detect_gaussian_residual,
    "Morphological top/black-hat": detect_morphological,
    "Local standard deviation": detect_local_statistics,
    "Fourier notch": detect_fourier,
    "Spectral residual": detect_phase_only,
}

REAL_DETECTORS = tuple(d for d in DETECTORS if "control" not in d)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def fill(mask: np.ndarray, close: int = 21) -> np.ndarray:
    """Close a detection and fill its interior.

    A residual detector responds to the **boundary** of a smooth defect, not to
    its area: inside a soft blob the surface is locally unchanged, so there is
    nothing to flag. Filling was added on the expectation that converting that
    edge response into a region would recover most of the IoU.

    **It does not.** Measured, it is worth between -0.006 and +0.002 IoU --
    nothing. The edges a residual detector produces are broken enough that a
    21-pixel closing does not enclose them, so there is no interior to fill. The
    step is kept, and `filling_helps` reports the number, because "we tried the
    obvious repair and it did not work" is a result and deleting it would leave
    the reader to try it again.
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (close, close)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(closed)
    cv2.drawContours(out, contours, -1, 255, -1)
    return out


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    p = pred > 0
    t = truth > 0
    union = float((p | t).sum())
    return float((p & t).sum()) / max(union, EPS) if union else 1.0


def found(pred: np.ndarray, truth: np.ndarray, fraction: float = 0.10) -> bool:
    """Did the detector find the defect *at all*?

    The question an inspection line actually asks. A defect counts as found when
    the detector marks at least `fraction` of it -- enough to raise an alarm and
    put the part in front of a person, which is what the system is for. Pixel IoU
    answers a different question: how precisely the extent was traced, which
    matters for a repair robot and not for a reject gate.
    """
    t = truth > 0
    if not t.any():
        return False
    return float((pred > 0)[t].mean()) >= fraction


def false_alarm_rate(pred: np.ndarray) -> float:
    """Fraction of a clean surface that was marked. Truth is empty, so this is
    the entire score on the clean arm."""
    return float((pred > 0).mean())


def evaluate(images=IMAGES, defects=DEFECTS, severity: float = SEVERITY,
             seed: int = 0, use_fill: bool = True) -> list[dict]:
    """Every detector on defective surfaces and on clean ones.

    Both arms, always, because the project's point is that they disagree.
    """
    acc = {name: {"iou": [], "found": [], "alarm": []} for name in DETECTORS}

    for image_name in images:
        image = load(image_name)
        for kind in defects:
            bad, mask = plant(image, kind, seed=seed, severity=severity)
            for name, fn in DETECTORS.items():
                raw = fn(bad)
                pred = fill(raw) if (use_fill and "control" not in name) else raw
                acc[name]["iou"].append(iou(pred, mask))
                acc[name]["found"].append(float(found(pred, mask)))
        ok, _ = clean(image)
        for name, fn in DETECTORS.items():
            raw = fn(ok)
            pred = fill(raw) if (use_fill and "control" not in name) else raw
            acc[name]["alarm"].append(false_alarm_rate(pred))

    return [
        {
            "detector": name,
            "iou": round(float(np.mean(a["iou"])), 4),
            "detection_rate": round(float(np.mean(a["found"])), 4),
            "false_alarm_share": round(float(np.mean(a["alarm"])), 5),
            "is_control": "control" in name,
        }
        for name, a in acc.items()
    ]


def per_defect(images=IMAGES, severity: float = SEVERITY,
               seed: int = 0) -> list[dict]:
    """Which detector finds which kind of defect. They do not overlap."""
    rows = []
    for kind in DEFECTS:
        row: dict[str, float | str] = {"defect": kind}
        for name in REAL_DETECTORS:
            hits = []
            for image_name in images:
                image = load(image_name)
                bad, mask = plant(image, kind, seed=seed, severity=severity)
                hits.append(float(found(fill(DETECTORS[name](bad)), mask)))
            row[name] = round(float(np.mean(hits)), 4)
        rows.append(row)
    return rows


def per_surface(images=IMAGES, severity: float = SEVERITY,
                seed: int = 0) -> list[dict]:
    """How much the *surface* decides the answer."""
    rows = []
    for image_name in images:
        image = load(image_name)
        ok, _ = clean(image)
        row = {
            "surface": image_name,
            "uniformity": round(uniformity(image), 2),
            "noise_ceiling": round(noise_ceiling(image), 2),
        }
        best_found, best_alarm = [], []
        for name in REAL_DETECTORS:
            hits = []
            for kind in DEFECTS:
                bad, mask = plant(image, kind, seed=seed, severity=severity)
                hits.append(float(found(fill(DETECTORS[name](bad)), mask)))
            best_found.append(float(np.mean(hits)))
            best_alarm.append(false_alarm_rate(fill(DETECTORS[name](ok))))
        row["best_detection_rate"] = round(max(best_found), 4)
        row["lowest_false_alarm"] = round(min(best_alarm), 5)
        row["worst_false_alarm"] = round(max(best_alarm), 5)
        rows.append(row)
    return rows


def arms_disagree(images=IMAGES, severity: float = SEVERITY) -> dict:
    """Does the defective arm rank the detectors the same way the clean arm does?

    The project's headline. A benchmark built only on defective parts picks a
    detector by one column of a two-column problem.
    """
    rows = [r for r in evaluate(images, severity=severity) if not r["is_control"]]
    by_detection = [r["detector"] for r in sorted(rows,
                                                  key=lambda r: -r["detection_rate"])]
    by_alarm = [r["detector"] for r in sorted(rows,
                                              key=lambda r: r["false_alarm_share"])]
    best = by_detection[0]
    return {
        "by_detection_rate": by_detection,
        "by_false_alarm": by_alarm,
        "detection_winner": best,
        "detection_winner_alarm_rank": by_alarm.index(best) + 1,
        "alarm_winner": by_alarm[0],
        "rankings_agree": by_detection == by_alarm,
    }


def filling_helps(images=IMAGES[:6], severity: float = SEVERITY) -> list[dict]:
    """What the fill step is worth, per detector.

    It converts an edge response into a region, which is the difference between
    scoring a smooth defect at its boundary and scoring it over its area.
    """
    rows = []
    for name in REAL_DETECTORS:
        raw_iou, filled_iou = [], []
        for image_name in images:
            image = load(image_name)
            for kind in DEFECTS:
                bad, mask = plant(image, kind, severity=severity)
                raw = DETECTORS[name](bad)
                raw_iou.append(iou(raw, mask))
                filled_iou.append(iou(fill(raw), mask))
        rows.append({
            "detector": name,
            "iou_raw": round(float(np.mean(raw_iou)), 4),
            "iou_filled": round(float(np.mean(filled_iou)), 4),
            "gain": round(float(np.mean(filled_iou)) - float(np.mean(raw_iou)), 4),
        })
    return rows


def sweep_severity(images=IMAGES[:8], severities=(1, 2, 4, 8, 16)) -> list[dict]:
    """How large a defect has to be before anything finds it."""
    rows = []
    for severity in severities:
        row: dict[str, float] = {"severity": severity}
        for name in REAL_DETECTORS:
            hits = []
            for image_name in images:
                image = load(image_name)
                for kind in DEFECTS:
                    bad, mask = plant(image, kind, severity=severity)
                    hits.append(float(found(fill(DETECTORS[name](bad)), mask)))
            row[name] = round(float(np.mean(hits)), 4)
        rows.append(row)
    return rows


def pixel_accuracy_is_broken(images=IMAGES[:6], severity: float = SEVERITY) -> dict:
    """What the do-nothing control scores on pixel accuracy.

    A defect covers about one per cent of a surface, so flagging nothing is right
    about ninety-nine per cent of the time.
    """
    shares, accuracies = [], []
    for image_name in images:
        image = load(image_name)
        for kind in DEFECTS:
            _, mask = plant(image, kind, severity=severity)
            shares.append(float((mask > 0).mean()))
            accuracies.append(float((mask == 0).mean()))
    return {
        "mean_defect_share": round(float(np.mean(shares)), 5),
        "flag_nothing_pixel_accuracy": round(float(np.mean(accuracies)), 5),
    }
