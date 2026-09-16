"""Portrait mode: separate the subject, blur the background, composite.

The question
------------
A phone's portrait mode uses a depth sensor or a trained segmentation network.
With **neither**, how good a portrait can classical methods produce, and *where*
exactly does the result break?

Three things are measured, because "portrait mode" is really three problems:

1. **The matte** — six segmentation methods, scored against an exact alpha.
   Scored separately on the solid body and on **hair**, because hair is only
   about 2.5% of the subject's pixels: a method can lose every strand and still
   report a whole-image IoU above 0.97.
2. **The composite** — blurring the whole image and then pasting the subject
   back pulls subject colour into the background. That halo is measured in
   pixels and in colour error against the true blurred background.
3. **The bokeh kernel** — a real lens turns a point highlight into a *flat disc*,
   not a Gaussian bump. Four kernels are compared on a synthetic point light.

One pre-trained component is used and it is not hidden: OpenCV's **Haar cascade**
for frontal faces, which ships inside the library. It was trained by someone
else, it is not a neural network, and nothing here trains anything.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared import synth
from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import dice, edge_prf, iou

FACE_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_CASCADE: cv2.CascadeClassifier | None = None


def face_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face Haar cascade once and cache it.

    Loading the XML costs a few milliseconds. Doing it inside a method that is
    about to be timed would put file I/O into the benchmark and make the whole
    timing column meaningless.
    """
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(FACE_CASCADE_FILE)
        if _CASCADE.empty():
            raise RuntimeError(
                f"Haar cascade failed to load from {FACE_CASCADE_FILE}. "
                "OpenCV 5 removed the bundled cascade XMLs; this repo pins "
                "opencv-python-headless<5 for exactly this reason."
            )
    return _CASCADE


def detect_face(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest frontal face as ``(x, y, w, h)``, or None."""
    faces = face_cascade().detectMultiScale(to_gray(img), scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def body_box_from_face(
    face: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Expand a face box into a plausible head-and-shoulders box.

    The proportions are anthropometric rules of thumb, not fitted parameters:
    shoulders are roughly three head-widths across, and the body continues to the
    bottom of the frame in a portrait.
    """
    x, y, w, h = face
    H, W = shape[:2]
    cx = x + w // 2
    bw = int(w * 3.2)
    bx = max(0, cx - bw // 2)
    by = max(0, y - int(h * 0.75))
    return bx, by, min(bw, W - bx), H - by


# --------------------------------------------------------------------------- #
# the six matting methods
# --------------------------------------------------------------------------- #


def matte_face_rect(img: np.ndarray) -> np.ndarray | None:
    """Baseline: the expanded face box, filled. Cannot follow a silhouette at all."""
    face = detect_face(img)
    if face is None:
        return None
    x, y, w, h = body_box_from_face(face, img.shape)
    m = np.zeros(img.shape[:2], np.uint8)
    m[y : y + h, x : x + w] = 255
    return m


def matte_face_ellipse(img: np.ndarray) -> np.ndarray | None:
    """A head ellipse plus a shoulder trapezoid — a shape prior, still no image evidence."""
    face = detect_face(img)
    if face is None:
        return None
    x, y, w, h = face
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    cx, cy = x + w // 2, y + h // 2
    cv2.ellipse(m, (cx, cy), (int(w * 0.72), int(h * 0.92)), 0, 0, 360, 255, -1)
    shoulder_top = cy + int(h * 0.75)
    pts = np.array(
        [
            [cx - int(w * 0.62), shoulder_top],
            [cx + int(w * 0.62), shoulder_top],
            [min(W - 1, cx + int(w * 2.0)), H - 1],
            [max(0, cx - int(w * 2.0)), H - 1],
        ],
        np.int32,
    )
    cv2.fillPoly(m, [pts], 255)
    return m


#: Default RNG seed for GrabCut. See :func:`_grabcut` — without this, repeated
#: runs on the *same* image return materially different masks.
GRABCUT_SEED = 0


def _grabcut(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    iters: int = 5,
    rng_seed: int | None = GRABCUT_SEED,
) -> np.ndarray:
    """Run GrabCut from a rectangle and return a binary mask.

    Five iterations, not one. GrabCut alternates between fitting colour mixtures
    and re-cutting the graph; a single pass has barely moved away from "the
    rectangle" and looks like the method is broken.

    **GrabCut is not deterministic.** It initialises its foreground and
    background Gaussian mixtures with k-means, seeded from OpenCV's *global* RNG.
    Run it twice on identical input and you get different masks — measured here
    at up to 0.24 IoU apart on one image. Any single reported GrabCut number is
    therefore one draw from a distribution, not a measurement.

    ``rng_seed`` is set before each call so results are reproducible. Pass None
    to sample the distribution instead; :func:`evaluate_grabcut_stability` does
    exactly that, deliberately.
    """
    if rng_seed is not None:
        cv2.setRNGSeed(rng_seed)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # GrabCut assumes a BGR-ordered image
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def matte_grabcut_face(
    img: np.ndarray, rng_seed: int | None = GRABCUT_SEED
) -> np.ndarray | None:
    """The method the plan prescribes: Haar to locate the subject, GrabCut to cut it out."""
    face = detect_face(img)
    if face is None:
        return None
    x, y, w, h = body_box_from_face(face, img.shape)
    H, W = img.shape[:2]
    # GrabCut needs a margin of guaranteed background, or it has no negative
    # examples to build a background colour model from
    x = max(1, min(x, W - 3))
    y = max(1, min(y, H - 3))
    w = max(2, min(w, W - x - 1))
    h = max(2, min(h, H - y - 1))
    return _grabcut(img, (x, y, w, h), rng_seed=rng_seed)


def matte_grabcut_centre(img: np.ndarray, rng_seed: int | None = GRABCUT_SEED) -> np.ndarray | None:
    """GrabCut with no face detection — a centred rectangle covering the middle 70%.

    Included to separate two contributions: how much of GrabCut's performance
    comes from the graph cut, and how much comes merely from being told roughly
    where the subject is.
    """
    H, W = img.shape[:2]
    rect = (int(W * 0.15), int(H * 0.10), int(W * 0.70), int(H * 0.88))
    return _grabcut(img, rect, rng_seed=rng_seed)


def matte_skin_colour(img: np.ndarray) -> np.ndarray | None:
    """Skin detection in YCrCb, then morphology and largest-component selection.

    YCrCb separates luma from chroma, so a skin rule in Cr/Cb is far more robust
    to brightness changes than the same rule in RGB. It can only ever find skin,
    which is the point: clothing and hair are invisible to it.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
    if n <= 1:
        return skin
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def matte_watershed(img: np.ndarray) -> np.ndarray | None:
    """Watershed, with sure-foreground and sure-background markers from the face box.

    Watershed over-segments badly without markers — it finds a basin for every
    local minimum. The face box supplies the seeds that make it behave.
    """
    face = detect_face(img)
    if face is None:
        return None
    x, y, w, h = body_box_from_face(face, img.shape)
    H, W = img.shape[:2]

    markers = np.zeros((H, W), np.int32)
    markers[:] = 0
    border = max(4, int(min(H, W) * 0.02))
    markers[:border, :] = 1          # sure background: the frame edge
    markers[-border:, :] = 1
    markers[:, :border] = 1
    markers[:, -border:] = 1
    fx, fy, fw, fh = face
    markers[fy : fy + fh, fx : fx + fw] = 2                      # sure foreground: the face
    markers[max(0, y + int(h * 0.6)) : H, x + w // 3 : x + 2 * w // 3] = 2  # and the torso

    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)
    return (markers == 2).astype(np.uint8) * 255


MATTES: dict[str, Callable[[np.ndarray], np.ndarray | None]] = {
    "Face rect (baseline)": matte_face_rect,
    "Face ellipse prior": matte_face_ellipse,
    "Haar + GrabCut": matte_grabcut_face,
    "GrabCut (centre rect)": matte_grabcut_centre,
    "Skin colour (YCrCb)": matte_skin_colour,
    "Watershed + markers": matte_watershed,
}


# --------------------------------------------------------------------------- #
# bokeh kernels
# --------------------------------------------------------------------------- #


def kernel_gaussian(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    k1 = cv2.getGaussianKernel(size, radius / 2.0)
    k = k1 @ k1.T
    return (k / k.sum()).astype(np.float32)


def kernel_box(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    k = np.ones((size, size), np.float32)
    return k / k.sum()


def kernel_disc(radius: int) -> np.ndarray:
    """A filled circle — the physically correct model for a circular aperture."""
    return synth.defocus_kernel(radius)


def kernel_hexagon(radius: int) -> np.ndarray:
    """A hexagonal aperture, as produced by a six-bladed diaphragm."""
    size = radius * 2 + 1
    k = np.zeros((size, size), np.float32)
    pts = np.array(
        [
            [radius + radius * np.cos(a), radius + radius * np.sin(a)]
            for a in np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
        ],
        np.int32,
    )
    cv2.fillPoly(k, [pts], 1.0)
    total = k.sum()
    return k / total if total > 0 else k


BOKEH_KERNELS: dict[str, Callable[[int], np.ndarray]] = {
    "Gaussian": kernel_gaussian,
    "Box": kernel_box,
    "Disc (circular aperture)": kernel_disc,
    "Hexagon (6-blade)": kernel_hexagon,
}


def highlight_profile(kernel: np.ndarray) -> dict[str, float]:
    """Measure how a kernel renders a point highlight.

    A real out-of-focus highlight is a **flat disc with a hard edge**. A Gaussian
    produces a soft bump instead, which is why Gaussian-blurred backgrounds read
    as "smudged" rather than "out of focus".

    * ``peak_to_mean`` — peak value over the mean of the non-zero support.
      A perfect disc is 1.0; a Gaussian is much larger.
    * ``edge_sharpness`` — fraction of total energy inside the outer 25% of the
      support radius. A disc keeps energy right out to its rim; a Gaussian has
      almost none there.
    """
    support = kernel > kernel.max() * 1e-3
    if not support.any():
        return {"peak_to_mean": float("nan"), "edge_sharpness": float("nan")}

    peak = float(kernel.max())
    mean = float(kernel[support].mean())

    r = kernel.shape[0] // 2
    yy, xx = np.mgrid[: kernel.shape[0], : kernel.shape[1]]
    dist = np.sqrt((xx - r) ** 2 + (yy - r) ** 2)
    rim = (dist >= 0.75 * r) & (dist <= r)

    return {
        "peak_to_mean": round(peak / mean if mean > 0 else float("inf"), 4),
        "edge_sharpness": round(float(kernel[rim].sum() / kernel.sum()), 4),
    }


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #


def composite_naive(img: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur the whole image, then paste the sharp subject back on top.

    This is what nearly every tutorial does, and it is wrong: the blur kernel
    reaches across the subject's boundary, so subject colour is smeared into the
    background just outside the silhouette. The result is a halo the subject
    appears to glow with.
    """
    blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    out = blurred.copy()
    out[mask > 0] = img[mask > 0]
    return out


def composite_masked(img: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur the background **with the subject excluded**, then composite.

    Implemented as a normalised convolution: blur the background-only image and
    blur the background indicator with the same kernel, then divide. Every output
    pixel is then an average of background pixels only, weighted correctly, and
    no subject colour can leak outward.
    """
    keep = (mask == 0).astype(np.float32)
    bg_only = to_float(img) * keep[..., None]

    num = cv2.filter2D(bg_only, -1, kernel, borderType=cv2.BORDER_REFLECT)
    den = cv2.filter2D(keep, -1, kernel, borderType=cv2.BORDER_REFLECT)
    den = np.maximum(den, 1e-4)[..., None]

    out = to_uint8(num / den)
    out[mask > 0] = img[mask > 0]
    return out


def composite_reference(scene: synth.PortraitScene, kernel: np.ndarray) -> np.ndarray:
    """The ideal result: blur the **true clean background plate**, then paste the subject.

    Only available because the scene is generated. It is the ground truth the two
    real strategies above are scored against.
    """
    blurred_bg = cv2.filter2D(scene.background, -1, kernel, borderType=cv2.BORDER_REFLECT)
    out = blurred_bg.copy()
    out[scene.mask > 0] = scene.image[scene.mask > 0]
    return out


COMPOSITORS: dict[str, Callable] = {
    "Naive (blur all, paste back)": composite_naive,
    "Masked (normalised convolution)": composite_masked,
}


def matte_confusion(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """2x2 confusion counts for a matte: rows actual, columns predicted.

    Layout::

        [[background->background, background->subject],
         [subject->background,    subject->subject   ]]

    Worth looking at in raw counts rather than rates: background is the majority
    class by a wide margin, so a method can lose the entire subject boundary and
    still show a high overall accuracy.
    """
    p = pred > 0
    t = truth > 0
    return np.array(
        [
            [int((~t & ~p).sum()), int((~t & p).sum())],
            [int((t & ~p).sum()), int((t & p).sum())],
        ],
        dtype=np.int64,
    )


def region_recall(pred: np.ndarray, scene: synth.PortraitScene) -> dict[str, float]:
    """Fraction of each region the prediction got right.

    Splitting the score by region is the entire point of this project: body,
    hair and background are wildly different in size and in difficulty, and one
    blended number hides which of them a method actually failed at.
    """
    return {
        "body": float((pred[scene.body > 0] > 0).mean()),
        "fine": float((pred[scene.fine > 0] > 0).mean()) if (scene.fine > 0).any() else 0.0,
        "background": float((pred[scene.mask == 0] == 0).mean()),
    }


def halo_error_map(
    composited: np.ndarray, ideal: np.ndarray
) -> np.ndarray:
    """Per-pixel absolute error against the ideal composite, in 0-255 units."""
    return np.abs(to_float(composited) - to_float(ideal)).mean(axis=-1) * 255.0


def halo_ring(mask: np.ndarray, width: int) -> np.ndarray:
    """The band of background pixels within ``width`` px outside the subject."""
    dilated = cv2.dilate(mask, np.ones((width * 2 + 1, width * 2 + 1), np.uint8))
    return cv2.bitwise_and(dilated, cv2.bitwise_not(mask))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

BACKGROUNDS = ("coffee", "rocket", "grass", "brick", "chelsea", "gravel")


def evaluate_mattes(n_scenes: int = 1, runs: int = 3) -> list[dict]:
    """Score every matting method, overall and separately on body and hair."""
    acc = {
        name: {
            "iou": [], "dice": [], "fine": [], "body": [], "bf1": [], "fpr": [],
            "ms": [], "found": 0,
        }
        for name in MATTES
    }

    for i in range(n_scenes):
        scene = synth.portrait_scene()
        truth_edges = cv2.Canny(scene.mask, 50, 150)

        for name, fn in MATTES.items():
            pred, timing = timeit(lambda f=fn: f(scene.image), runs=runs, warmup=1)
            acc[name]["ms"].append(timing.median_ms)
            if pred is None:
                continue
            acc[name]["found"] += 1
            acc[name]["iou"].append(iou(pred, scene.mask))
            acc[name]["dice"].append(dice(pred, scene.mask))
            # restrict scoring to each region: how much thin structure survived
            acc[name]["fine"].append(
                float((pred[scene.fine > 0] > 0).mean()) if (scene.fine > 0).any() else 0.0
            )
            acc[name]["body"].append(float((pred[scene.body > 0] > 0).mean()))
            # Recall alone is gameable: a mask covering the whole frame "recovers"
            # 100% of the fine detail. The false-positive rate on true background is
            # reported beside it so that cheat is visible in the table.
            acc[name]["fpr"].append(float((pred[scene.mask == 0] > 0).mean()))
            acc[name]["bf1"].append(edge_prf(cv2.Canny(pred, 50, 150), truth_edges, 2)["f1"])

    rows = []
    for name, a in acc.items():
        ok = bool(a["iou"])
        rows.append(
            {
                "method": name,
                "detected_pct": f"{a['found'] / n_scenes * 100:.0f}%",
                "iou": round(float(np.mean(a["iou"])), 4) if ok else None,
                "dice": round(float(np.mean(a["dice"])), 4) if ok else None,
                "body_recall": round(float(np.mean(a["body"])), 4) if ok else None,
                "fine_recall": round(float(np.mean(a["fine"])), 4) if ok else None,
                "background_fpr": round(float(np.mean(a["fpr"])), 4) if ok else None,
                "boundary_f1": round(float(np.mean(a["bf1"])), 4) if ok else None,
                "median_ms": round(float(np.median(a["ms"])), 3) if a["ms"] else None,
            }
        )
    return rows


def stability_scenes() -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """The images the seed sweep runs on: ``(label, image, truth or None)``.

    Five genuinely different photographs, not one image relabelled five times.
    Only the footballer has a reference matte, so the other four are scored by
    self-agreement alone — see :func:`evaluate_grabcut_stability`.
    """
    from shared import io as _io

    scene = synth.portrait_scene()
    return [
        ("footballer · person, crowd behind", scene.image, scene.mask),
        ("girl · person, soft background", _io.real_photo("girl"), None),
        ("dog · animal, head on", _io.real_photo("dog"), None),
        ("butterfly · insect, busy background", _io.real_photo("butterfly"), None),
        ("coffee cup · object, table top", _io.real_photo("coffee_cup"), None),
    ]


def seed_self_agreement(masks: list[np.ndarray]) -> float:
    """Mean IoU between every pair of masks the same image produced.

    This is the measurement that makes the stability question answerable on an
    ordinary photograph. Scoring seed-to-seed spread against a *ground truth*
    restricts the experiment to the one image in this project that has one; but
    "did the algorithm return the same answer twice" needs no truth at all —
    only the masks agreeing with each other. 1.0 means every seed produced an
    identical cut-out; 0.5 means two seeds typically disagree about a third of
    the pixels they claim.
    """
    if len(masks) < 2:
        return 1.0
    scores = [
        iou(masks[i], masks[j])
        for i in range(len(masks))
        for j in range(i + 1, len(masks))
    ]
    return float(np.mean(scores))


def evaluate_grabcut_stability(n_seeds: int = 24, scenes=None) -> list[dict]:
    """Quantify how much a single GrabCut number is worth.

    Each image is segmented ``n_seeds`` times with a different RNG seed. The
    input never changes, so any spread is **pure algorithmic noise** — a direct
    measure of how much of a reported GrabCut score is luck. GrabCut initialises
    its foreground and background colour mixtures with k-means seeded from
    OpenCV's **global** RNG, which is where the noise enters.

    Two separate facts came out of this, and they are easy to confuse:

    * With a seed pinned, GrabCut is perfectly reproducible — the same seed
      gives the same mask every time, on any thread count.
    * *Across* seeds it need not be stable at all.

    So a GrabCut result is reproducible and still not a measurement, unless the
    seed is reported alongside it or the distribution is summarised. This exists
    because the first version of this project reported IoU 0.87 for GrabCut from
    a single unseeded run.

    🚨 **This used to sweep one image and label it four times.** It read
    ``BACKGROUNDS[i]`` for the row label but called ``synth.portrait_scene()``
    with no argument, and ``portrait_scene`` ignores ``background`` anyway — it
    returns one fixed photograph. So four rows claiming to be coffee, rocket,
    grass and brick were the same segmentation repeated, and they printed
    identical numbers to four decimal places. It now runs on five genuinely
    different photographs.

    ``agreement`` is always reported; ``iou_*`` only where a truth matte exists.
    """
    rows = []
    for label, image, truth in scenes if scenes is not None else stability_scenes():
        masks = []
        for s in range(n_seeds):
            m = matte_grabcut_centre(image, rng_seed=s)
            if m is not None:
                masks.append(m)
        if not masks:
            continue

        row = {
            "scene": label,
            "n_seeds": len(masks),
            "agreement": round(seed_self_agreement(masks), 4),
        }
        if truth is not None:
            scores = [iou(m, truth) for m in masks]
            row.update(
                {
                    "iou_mean": round(float(np.mean(scores)), 4),
                    "iou_std": round(float(np.std(scores)), 4),
                    "iou_min": round(float(np.min(scores)), 4),
                    "iou_max": round(float(np.max(scores)), 4),
                    "iou_spread": round(float(np.max(scores) - np.min(scores)), 4),
                }
            )
        rows.append(row)
    return rows


def evaluate_bokeh(radius: int = 15) -> list[dict]:
    """Characterise each bokeh kernel's highlight rendering."""
    rows = []
    for name, make in BOKEH_KERNELS.items():
        k = make(radius)
        prof = highlight_profile(k)
        rows.append({"kernel": name, "radius_px": radius, **prof})
    return rows


def evaluate_compositing(n_scenes: int = 1, radius: int = 15, ring: int = 12) -> list[dict]:
    """Measure halo bleed for each compositing strategy against the ideal result.

    Scored with the **true** matte, so the halo measured here is a property of the
    compositing step alone and cannot be blamed on a poor segmentation.
    """
    kernel = kernel_disc(radius)
    acc = {name: {"ring_err": [], "bg_err": [], "ms": []} for name in COMPOSITORS}

    for i in range(n_scenes):
        scene = synth.portrait_scene()
        ideal = composite_reference(scene, kernel)
        band = halo_ring(scene.mask, ring) > 0
        outside = scene.mask == 0

        for name, fn in COMPOSITORS.items():
            out, timing = timeit(
                lambda f=fn: f(scene.image, scene.mask, kernel), runs=2, warmup=1
            )
            diff = np.abs(to_float(out) - to_float(ideal)).mean(axis=-1) * 255.0
            acc[name]["ring_err"].append(float(diff[band].mean()))
            acc[name]["bg_err"].append(float(diff[outside].mean()))
            acc[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": name,
            "halo_err_0_255": round(float(np.mean(a["ring_err"])), 3),
            "background_err_0_255": round(float(np.mean(a["bg_err"])), 3),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for name, a in acc.items()
    ]


def portrait(
    img: np.ndarray,
    matte: str = "Haar + GrabCut",
    bokeh: str = "Disc (circular aperture)",
    radius: int = 15,
    compositor: str = "Masked (normalised convolution)",
):
    """End-to-end portrait mode, for the UI and for inference.

    Returns ``(mask, output)``; ``mask`` is None when no subject was found.
    """
    mask = MATTES[matte](img)
    if mask is None:
        return None, None
    kernel = BOKEH_KERNELS[bokeh](radius)
    return mask, COMPOSITORS[compositor](img, mask, kernel)
