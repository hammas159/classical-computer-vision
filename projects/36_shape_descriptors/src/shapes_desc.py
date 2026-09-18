"""Shape descriptors: Hu moments, Fourier descriptors, chain codes.

The question
------------
Shape descriptors are sold on their invariances — "invariant to translation,
rotation and scale". Those claims are checkable arithmetic, not marketing.

> **The claim under test:** verify each invariance *directly*. Apply a known
> transform to a known shape and measure how much the descriptor moves. A truly
> invariant descriptor changes by rounding error; anything else has a caveat that
> is usually left unstated.

The caveats worth surfacing:

* **Hu moments are invariant in continuous mathematics.** On a rasterised
  discrete image they are not — but the usual explanation, that a bigger raster
  fixes it, is measurably false here. The error bounces around (0.236 at 64 px,
  0.042 at 96, 0.391 at 128) instead of converging, and for a drawn ellipse it
  gets **seventeen times worse** going from 64 px to 256. A finer raster
  computes a symmetric shape's near-zero moments more accurately, which pushes
  them *closer* to the floor where the log transform is least stable.
* **The usual log transform, `sign(h)·log10(|h|+eps)`, is broken at zero.**
  `np.sign(0)` is `0`, so a moment that is exactly zero maps to 0 rather than to
  the floor, and the first rotation that makes it 1e-62 instead of 0 moves the
  descriptor by 12 units. `desc_hu_textbook` keeps the broken recipe as a
  control; `desc_hu` floors the magnitude instead.
* **The 7th Hu moment changes sign under reflection.** That is a feature, not a
  bug: it is the only one that can distinguish a shape from its mirror image.
* **Fourier descriptors need a *resampled* contour.** Different shapes yield
  different numbers of boundary points, and comparing spectra of different
  lengths is meaningless.

Shapes are generated here, so every transform is exact.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_gray

EPS = 1e-12


# --------------------------------------------------------------------------- #
# shape generation
# --------------------------------------------------------------------------- #

SHAPE_NAMES = ("circle", "square", "triangle", "star", "cross", "ellipse")


def make_shape(name: str, size: int = 256, scale: float = 1.0,
               rotation: float = 0.0, translate=(0, 0)) -> np.ndarray:
    """Draw a named binary shape under an exact known transform."""
    img = np.zeros((size, size), np.uint8)
    c = size // 2
    r = int(size * 0.3 * scale)

    if name == "circle":
        cv2.circle(img, (c, c), r, 255, -1)
    elif name == "ellipse":
        cv2.ellipse(img, (c, c), (r, int(r * 0.55)), 0, 0, 360, 255, -1)
    elif name == "square":
        cv2.rectangle(img, (c - r, c - r), (c + r, c + r), 255, -1)
    elif name == "triangle":
        pts = np.array([[c, c - r], [c - r, c + r], [c + r, c + r]], np.int32)
        cv2.fillPoly(img, [pts], 255)
    elif name == "cross":
        t = max(2, r // 3)
        cv2.rectangle(img, (c - t, c - r), (c + t, c + r), 255, -1)
        cv2.rectangle(img, (c - r, c - t), (c + r, c + t), 255, -1)
    elif name == "star":
        pts = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            rad = r if i % 2 == 0 else r * 0.45
            pts.append([int(c + rad * np.cos(angle)), int(c + rad * np.sin(angle))])
        cv2.fillPoly(img, [np.array(pts, np.int32)], 255)
    else:
        raise ValueError(f"unknown shape {name!r}")

    if rotation:
        m = cv2.getRotationMatrix2D((c, c), rotation, 1.0)
        img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_NEAREST)
    if translate != (0, 0):
        m = np.float32([[1, 0, translate[0]], [0, 1, translate[1]]])
        img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_NEAREST)
    return img


# --------------------------------------------------------------------------- #
# real silhouettes — the shapes people actually traced
# --------------------------------------------------------------------------- #

#: Twelve silhouettes cut from BSDS500 **human segmentations**. Generated shapes
#: are the right way to measure an invariance, because the transform is exact and
#: any movement in the descriptor is the descriptor's fault. They are the wrong
#: way to find out whether the descriptor is useful: a drawn circle has a perfect
#: boundary, and nothing a person traced does.
#:
#: Each entry is ``(annotator, region label)`` — which person's segmentation, and
#: which region of it. That pair is recorded rather than "the biggest region"
#: because the biggest region of a photograph is usually the sky.
#:
#: Selected by a silhouette-quality score rather than by one of
#: `tools/select_images.py`'s axes: a single connected region covering 8-55% of
#: the frame, not glued to the border. The pool deliberately keeps two shapes
#: that break the usual assumptions — `made_up_face` fragments into fifteen
#: contours, and `lizard_on_a_leaf` is a leaf with a lizard-shaped hole in it.
SILHOUETTES: dict[str, tuple[int, int]] = {
    "greek_amphora": (3, 2),          # two handles, holes between handle and body
    "collie_standing": (2, 4),        # the most articulated outline here
    "flatfish_on_sand": (0, 2),       # a smooth oval with a tail
    "man_in_a_fez": (0, 3),           # head and shoulders
    "made_up_face": (0, 30),          # fragments into fifteen contours
    "roman_amphitheatre": (1, 7),     # an elongated oval
    "green_mountain_ridge": (5, 5),   # a long jagged boundary
    "basket_of_grain": (2, 2),        # near-circular
    "buttressed_trunk": (3, 9),       # a straight-sided polygon
    "brain_coral": (3, 5),            # a rounded dome
    "lizard_on_a_leaf": (5, 4),       # a leaf with a lizard-shaped hole
    "spotted_fish_head": (1, 4),      # an angular wedge
}

REAL_IMAGES = tuple(SILHOUETTES)


def load_scene(name: str) -> np.ndarray:
    """The photograph a silhouette was traced from."""
    from shared import io

    return io.real_photo(name)


def load_silhouette(name: str) -> np.ndarray:
    """One region of one person's segmentation, as a binary mask.

    Not `bsds.dominant_foreground`, which takes the largest region — on a
    photograph that is usually the sky. The annotator and region are named in
    `SILHOUETTES` so the shape is reproducible and attributable to a person.
    """
    from shared import bsds

    annotator, label = SILHOUETTES[name]
    segmentation = bsds.load_annotations(name)[annotator]["segmentation"]
    return ((segmentation == label).astype(np.uint8)) * 255


def silhouettes_by_annotator(name: str, min_iou: float = 0.5) -> list[np.ndarray]:
    """The same object as every annotator who traced it drew it.

    BSDS ships five to seven independent segmentations per image, and the region
    labels are not shared between them — annotator 3's region 9 has nothing to do
    with annotator 0's. So each other annotator's version of *this* object is
    found by taking the region that overlaps the reference most, and kept only if
    the overlap is convincing.

    This is the measurement that gives the invariance numbers a scale. A
    descriptor whose rotation error is smaller than the spread between two people
    tracing the same vase is being quoted to more precision than the shape has.
    """
    from shared import bsds

    reference = load_silhouette(name) > 0
    out = [reference.astype(np.uint8) * 255]
    annotator, _ = SILHOUETTES[name]

    for i, entry in enumerate(bsds.load_annotations(name)):
        if i == annotator:
            continue
        segmentation = entry["segmentation"]
        best, best_iou = None, 0.0
        for label in np.unique(segmentation):
            mask = segmentation == label
            union = np.logical_or(mask, reference).sum()
            if union == 0:
                continue
            score = np.logical_and(mask, reference).sum() / union
            if score > best_iou:
                best, best_iou = mask, float(score)
        if best is not None and best_iou >= min_iou:
            out.append(best.astype(np.uint8) * 255)
    return out


def largest_contour(binary: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# --------------------------------------------------------------------------- #
# the descriptors
# --------------------------------------------------------------------------- #


def desc_hu_textbook(binary: np.ndarray) -> np.ndarray:
    """The log-Hu recipe as it is usually written — and it is broken.

    ``sign(h) * log10(|h| + eps)`` appears in every tutorial. It has a hole at
    exactly zero, and symmetric shapes land in it:

    * a drawn circle's 5th, 6th and 7th Hu moments are **exactly 0.0**;
    * ``np.sign(0)`` is ``0``, so those components come out as 0 rather than at
      the floor ``log10(eps) = -12``;
    * rotate the circle by any angle and they become 4.7e-62 instead of exactly
      zero — still numerically zero, but now ``sign`` is 1 and the component
      jumps to -12. **The descriptor moves 12 units because a quantity went from
      zero to 1e-62.**

    Worse for a five-pointed star, whose h5-h7 hover at 1e-16: rotation flips
    their *sign*, so the component swings from -12 to +12 and the descriptor
    moves 24.

    Kept as a control. It is what makes Hu moments look catastrophically
    non-invariant on generated shapes (mean relative change 5.06) while being
    the most invariant descriptor here on real ones (0.00099).
    """
    m = cv2.moments(binary, binaryImage=True)
    hu = cv2.HuMoments(m).ravel()
    return np.sign(hu) * np.log10(np.abs(hu) + EPS)


def desc_hu(binary: np.ndarray) -> np.ndarray:
    """Seven Hu moments, log-transformed with the zero handled.

    The raw values span many orders of magnitude, so a Euclidean distance between
    them is decided entirely by the first moment. The log compresses the range
    while **keeping the sign**, which matters because the 7th moment's sign is
    the reflection detector.

    Everything below ``EPS`` is mapped to a single floor value regardless of
    sign. A Hu moment of 1e-62 computed from a binary raster is not a small
    number, it is zero with rounding on it, and giving it a sign invents a
    distinction the image cannot support. See `desc_hu_textbook` for what
    happens without this.

    **This is a trade, not a free fix.** Flooring the magnitude also floors the
    *sign*, and the sign of h7 is the only reflection detector in this project.
    For a near-symmetric shape whose h7 legitimately sits below the floor — the
    basket of grain has h7 = +3.7e-13 upright and exactly -3.7e-13 mirrored — the
    floored version reports **no** change under reflection while the textbook one
    reports 1.151. That 3.7e-13 is structured, not noise: it negates to four
    significant figures.

    So for a nearly mirror-symmetric shape, rotation stability and reflection
    sensitivity are in direct conflict and no single epsilon resolves both. Both
    variants are reported for exactly this reason.
    """
    m = cv2.moments(binary, binaryImage=True)
    hu = cv2.HuMoments(m).ravel()
    magnitude = np.log10(np.maximum(np.abs(hu), EPS))
    sign = np.where(np.abs(hu) < EPS, 1.0, np.sign(hu))
    return sign * magnitude


def desc_fourier(binary: np.ndarray, n_descriptors: int = 20, n_points: int = 256) -> np.ndarray:
    """Fourier descriptors of the boundary, made invariant explicitly.

    The contour is treated as a complex signal ``x + iy`` and transformed. Then:

    * dropping the zeroth coefficient removes **translation** (it is the centroid);
    * dividing by the magnitude of the first removes **scale**;
    * taking magnitudes removes **rotation and starting point** together, since
      both appear only as phase.

    Resampling to a fixed ``n_points`` first is not optional — spectra of
    different lengths are not comparable.
    """
    contour = largest_contour(binary)
    if contour is None or len(contour) < 8:
        return np.zeros(n_descriptors, np.float32)

    pts = contour.reshape(-1, 2).astype(np.float64)
    idx = np.linspace(0, len(pts) - 1, n_points)
    resampled = np.stack(
        [np.interp(idx, np.arange(len(pts)), pts[:, 0]),
         np.interp(idx, np.arange(len(pts)), pts[:, 1])], axis=-1
    )

    signal = resampled[:, 0] + 1j * resampled[:, 1]
    spectrum = np.fft.fft(signal)
    spectrum[0] = 0
    scale = np.abs(spectrum[1]) + EPS
    return (np.abs(spectrum[1 : n_descriptors + 1]) / scale).astype(np.float32)


def desc_chain_code_histogram(binary: np.ndarray, bins: int = 8) -> np.ndarray:
    """Freeman chain code direction histogram.

    Encodes the boundary as a sequence of 8-connected steps and histograms the
    directions. Translation invariant for free and scale invariant once
    normalised — but **not** rotation invariant, because rotating the shape
    permutes the direction bins. Included precisely so that failure shows up as a
    number in the rotation column.
    """
    contour = largest_contour(binary)
    if contour is None or len(contour) < 3:
        return np.zeros(bins, np.float32)
    pts = contour.reshape(-1, 2)
    deltas = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0]) % (2 * np.pi)
    hist, _ = np.histogram(angles, bins=bins, range=(0, 2 * np.pi))
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def desc_simple_geometry(binary: np.ndarray) -> np.ndarray:
    """Circularity, aspect ratio, solidity, extent — the "boring" baseline.

    Four ratios anyone could write in ten minutes. Worth including because it
    frequently performs comparably to the sophisticated descriptors on simple
    shapes, which is a useful thing to know before reaching for Fourier analysis.
    """
    contour = largest_contour(binary)
    if contour is None:
        return np.zeros(4, np.float32)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    x, y, w, h = cv2.boundingRect(contour)
    return np.array(
        [
            4 * np.pi * area / max(perimeter**2, EPS),   # circularity
            w / max(h, EPS),                             # aspect ratio
            area / max(hull_area, EPS),                  # solidity
            area / max(w * h, EPS),                      # extent
        ],
        np.float32,
    )


DESCRIPTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Hu moments (log)": desc_hu,
    "Hu moments (textbook log)": desc_hu_textbook,
    "Fourier descriptors": desc_fourier,
    "Chain code histogram": desc_chain_code_histogram,
    "Simple geometry": desc_simple_geometry,
}


# --------------------------------------------------------------------------- #
# invariance measurement
# --------------------------------------------------------------------------- #


def relative_change(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised distance between two descriptor vectors.

    Normalised by the reference's magnitude so descriptors with wildly different
    scales — log-Hu spans tens, a histogram spans one — can be compared on the
    same axis.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), EPS))


ROTATIONS = (0, 15, 30, 45, 90, 180)
SCALES = (1.0, 0.8, 0.6, 0.4, 1.3)
TRANSLATIONS = ((0, 0), (10, 0), (0, 25), (30, 30))
SIZES = (64, 96, 128, 192, 256)


def invariance_table(transform: str, levels, shapes=SHAPE_NAMES, size: int = 256):
    """How much each descriptor moves under a transform it claims not to notice."""
    rows = []
    for name, fn in DESCRIPTORS.items():
        changes = []
        for shape in shapes:
            reference = fn(make_shape(shape, size=size))
            for level in levels:
                if transform == "rotation":
                    img = make_shape(shape, size=size, rotation=float(level))
                elif transform == "scale":
                    img = make_shape(shape, size=size, scale=float(level))
                elif transform == "translation":
                    img = make_shape(shape, size=size, translate=level)
                else:
                    raise ValueError(f"unknown transform {transform!r}")
                if transform == "rotation" and level == 0:
                    continue
                if transform == "scale" and level == 1.0:
                    continue
                if transform == "translation" and level == (0, 0):
                    continue
                changes.append(relative_change(reference, fn(img)))
        rows.append(
            {
                "descriptor": name,
                "mean_relative_change": round(float(np.mean(changes)), 5) if changes else 0.0,
                "max_relative_change": round(float(np.max(changes)), 5) if changes else 0.0,
            }
        )
    return rows


def discretisation_error(shapes=SHAPE_NAMES, sizes=SIZES, rotation: float = 30.0):
    """Hu moments are invariant in continuous maths — but images are discrete.

    The same rotation is applied at several raster sizes. The usual story is that
    the invariance failure is a discretisation artefact and shrinks as the shape
    gets larger. **It does not.** The error bounces — 0.236 at 64 px, 0.042 at
    96, 0.391 at 128 — and for a drawn ellipse it gets seventeen times worse from
    64 px to 256.

    The reason is the log floor. A symmetric shape's h5-h7 are zero in theory; a
    finer raster computes them *more* accurately, which moves them closer to the
    epsilon where the log transform is least stable. Resolution does not rescue a
    descriptor whose instability lives at zero.
    """
    rows = []
    for size in sizes:
        per_descriptor = {}
        for name, fn in DESCRIPTORS.items():
            changes = []
            for shape in shapes:
                ref = fn(make_shape(shape, size=size))
                rot = fn(make_shape(shape, size=size, rotation=rotation))
                changes.append(relative_change(ref, rot))
            per_descriptor[name] = round(float(np.mean(changes)), 5)
        rows.append({"raster_size": size, **per_descriptor})
    return rows


def reflection_test(shapes=SHAPE_NAMES, size: int = 256):
    """Only the 7th Hu moment should flip sign under reflection.

    A shape and its mirror image are genuinely different objects. Every other
    descriptor here is blind to the difference, which is a real limitation and is
    worth stating as a measurement.
    """
    rows = []
    for shape in shapes:
        img = make_shape(shape, size=size)
        mirrored = cv2.flip(img, 1)
        hu_a, hu_b = desc_hu(img), desc_hu(mirrored)
        sign_flips = [bool(np.sign(a) != np.sign(b)) for a, b in zip(hu_a, hu_b)]
        rows.append(
            {
                "shape": shape,
                "hu7_sign_flipped": sign_flips[6],
                "other_hu_flipped": sum(sign_flips[:6]),
                "fourier_change": round(
                    relative_change(desc_fourier(img), desc_fourier(mirrored)), 5
                ),
                "geometry_change": round(
                    relative_change(desc_simple_geometry(img), desc_simple_geometry(mirrored)), 5
                ),
            }
        )
    return rows


def classification_accuracy(shapes=SHAPE_NAMES, size: int = 256, runs: int = 1):
    """Can each descriptor tell the shapes apart, across all transforms?

    Nearest-neighbour with leave-one-out — no training, no parameters. Every
    shape appears under many transforms, so a descriptor only scores well if its
    invariance holds *and* it still distinguishes different shapes. Those two
    requirements pull against each other, which is what makes the number
    informative.
    """
    features, labels = [], []
    for label, shape in enumerate(shapes):
        for rot in (0, 30, 75, 180):
            for scale in (1.0, 0.7):
                img = make_shape(shape, size=size, rotation=float(rot), scale=scale)
                features.append(img)
                labels.append(label)
    labels = np.asarray(labels)

    rows = []
    for name, fn in DESCRIPTORS.items():
        _, timing = timeit(lambda f=fn, a=features[0]: f(a), runs=runs, warmup=1)
        vectors = np.stack([fn(img) for img in features])
        mean = vectors.mean(axis=0, keepdims=True)
        std = vectors.std(axis=0, keepdims=True)
        v = (vectors - mean) / np.maximum(std, EPS)
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        predicted = labels[np.argmin(d, axis=1)]
        rows.append(
            {
                "descriptor": name,
                "accuracy": round(float((predicted == labels).mean()), 4),
                "dimensions": int(vectors.shape[1]),
                "median_ms": round(float(timing.median_ms), 4),
            }
        )
    return rows


def noise_robustness(shapes=SHAPE_NAMES, levels=(0.0, 0.01, 0.03, 0.08), size: int = 256):
    """Boundary noise: Fourier descriptors keep low frequencies, so they should hold.

    Salt-and-pepper on the *boundary* roughens the contour. A descriptor built
    from low-frequency coefficients discards that roughness; one built from the
    raw boundary sequence cannot.
    """
    rows = []
    for level in levels:
        per = {}
        for name, fn in DESCRIPTORS.items():
            changes = []
            for shape in shapes:
                clean = make_shape(shape, size=size)
                noisy = clean.copy()
                if level > 0:
                    rng = np.random.default_rng(0)
                    edge = cv2.morphologyEx(clean, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
                    flip = (rng.random(clean.shape) < level) & (edge > 0)
                    noisy[flip] = 255 - noisy[flip]
                changes.append(relative_change(fn(clean), fn(noisy)))
            per[name] = round(float(np.mean(changes)), 5)
        rows.append({"boundary_noise": level, **per})
    return rows


# --------------------------------------------------------------------------- #
# the same questions, on shapes a person drew
# --------------------------------------------------------------------------- #


def transform_mask(mask: np.ndarray, rotation: float = 0.0, scale: float = 1.0,
                   translate=(0, 0)) -> np.ndarray:
    """Rotate/scale a real silhouette about its own centroid, without clipping it.

    The canvas is padded first. A generated shape sits in the middle of its
    frame by construction; a traced one does not, and rotating it in place would
    slice off whatever crossed the border — which a descriptor would read as a
    change of shape rather than a change of pose.
    """
    h, w = mask.shape
    pad = int(max(h, w) * (0.5 * max(scale, 1.0) + 0.6))
    canvas = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    m = cv2.moments(canvas, binaryImage=True)
    cx = m["m10"] / max(m["m00"], EPS)
    cy = m["m01"] / max(m["m00"], EPS)

    matrix = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    matrix[0, 2] += translate[0]
    matrix[1, 2] += translate[1]
    out = cv2.warpAffine(canvas, matrix, (canvas.shape[1], canvas.shape[0]),
                         flags=cv2.INTER_NEAREST)
    return (out > 127).astype(np.uint8) * 255


def real_invariance_table(transform: str, levels, images=REAL_IMAGES):
    """The invariance question, asked of human-traced outlines.

    Same protocol as `invariance_table`, same exact transforms — the only change
    is that the boundary was drawn by a person instead of by `cv2.fillPoly`.
    """
    rows = []
    masks = {name: load_silhouette(name) for name in images}
    for name, fn in DESCRIPTORS.items():
        changes = []
        for image in images:
            base = transform_mask(masks[image])
            reference = fn(base)
            for level in levels:
                if transform == "rotation":
                    if level == 0:
                        continue
                    moved = transform_mask(masks[image], rotation=float(level))
                elif transform == "scale":
                    if level == 1.0:
                        continue
                    moved = transform_mask(masks[image], scale=float(level))
                elif transform == "translation":
                    if level == (0, 0):
                        continue
                    moved = transform_mask(masks[image], translate=level)
                else:
                    raise ValueError(f"unknown transform {transform!r}")
                changes.append(relative_change(reference, fn(moved)))
        rows.append({
            "descriptor": name,
            "mean_relative_change": round(float(np.mean(changes)), 5),
            "max_relative_change": round(float(np.max(changes)), 5),
        })
    return rows


def annotator_variation(images=REAL_IMAGES):
    """How much each descriptor moves between two people tracing the same object.

    This is the scale the invariance numbers have to be read against. An
    invariance error far below this is real arithmetic and no practical use: the
    shape itself is not defined that precisely. An invariance error far above it
    is a genuine failure.
    """
    rows = []
    versions = {name: silhouettes_by_annotator(name) for name in images}
    for name, fn in DESCRIPTORS.items():
        changes, counted = [], 0
        for image in images:
            traced = versions[image]
            if len(traced) < 2:
                continue
            counted += 1
            reference = fn(transform_mask(traced[0]))
            for other in traced[1:]:
                changes.append(relative_change(reference, fn(transform_mask(other))))
        rows.append({
            "descriptor": name,
            "mean_relative_change": round(float(np.mean(changes)), 5),
            "max_relative_change": round(float(np.max(changes)), 5),
            "images_with_two_or_more_tracings": counted,
            "pairs": len(changes),
        })
    return rows


def real_classification_accuracy(images=REAL_IMAGES, runs: int = 1):
    """Can each descriptor tell twelve traced silhouettes apart under transforms?

    The generated-shape version of this saturates — six drawn shapes are very
    different from each other. Twelve outlines a person traced out of
    photographs are not, and the number stops being a formality.
    """
    masks = {name: load_silhouette(name) for name in images}
    features, labels = [], []
    for label, image in enumerate(images):
        for rot in (0, 30, 75, 180):
            for scale in (1.0, 0.7):
                features.append(transform_mask(masks[image], rotation=float(rot), scale=scale))
                labels.append(label)
    labels = np.asarray(labels)

    rows = []
    for name, fn in DESCRIPTORS.items():
        _, timing = timeit(lambda f=fn, a=features[0]: f(a), runs=runs, warmup=1)
        vectors = np.stack([fn(img) for img in features])
        mean = vectors.mean(axis=0, keepdims=True)
        std = vectors.std(axis=0, keepdims=True)
        v = (vectors - mean) / np.maximum(std, EPS)
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        predicted = labels[np.argmin(d, axis=1)]
        rows.append({
            "descriptor": name,
            "accuracy": round(float((predicted == labels).mean()), 4),
            "dimensions": int(vectors.shape[1]),
            "median_ms": round(float(timing.median_ms), 4),
        })
    return rows


def real_reflection_test(images=REAL_IMAGES):
    """The 7th Hu moment on shapes that are genuinely not mirror-symmetric.

    On generated shapes this looks like a failure: h7 flips for the triangle and
    for none of the other five. It is not a failure — a circle, a square, a
    cross, an ellipse and a five-pointed star all *have* a mirror axis, so their
    reflection is the same shape and h7 is right not to move. The test only means
    anything on shapes without that symmetry, which is most things a person
    traces.
    """
    rows = []
    for image in images:
        mask = transform_mask(load_silhouette(image))
        mirrored = cv2.flip(mask, 1)
        hu_a, hu_b = desc_hu(mask), desc_hu(mirrored)
        flips = [bool(np.sign(a) != np.sign(b)) for a, b in zip(hu_a, hu_b)]

        # how far the shape is from being its own mirror image, for context
        moments = cv2.moments(mask, binaryImage=True)
        cx = int(moments["m10"] / max(moments["m00"], EPS))
        shifted = np.roll(cv2.flip(mask, 1), cx - (mask.shape[1] - 1 - cx), axis=1)
        union = np.logical_or(mask > 0, shifted > 0).sum()
        symmetry = float(np.logical_and(mask > 0, shifted > 0).sum() / max(union, 1))

        rows.append({
            "image": image,
            "mirror_symmetry_iou": round(symmetry, 4),
            "hu7_sign_flipped": flips[6],
            "other_hu_flipped": sum(flips[:6]),
            "fourier_change": round(relative_change(desc_fourier(mask),
                                                    desc_fourier(mirrored)), 5),
            "geometry_change": round(relative_change(desc_simple_geometry(mask),
                                                     desc_simple_geometry(mirrored)), 5),
        })
    return rows


def cross_annotator_classification(images=REAL_IMAGES, rotation: float = 0.0):
    """Recognise an object from *another person's* tracing of it.

    The gallery is each object as one annotator drew it; the probes are the same
    objects as everyone else drew them. Nothing is transformed unless
    ``rotation`` says so, so this measures only one thing: whether the descriptor
    survives the difference between two people's ideas of where the edge is.

    It is the honest version of the classification number. Classifying an object
    against transformed copies of *itself* saturates at 1.000 for every
    descriptor here, because a rotated copy of a silhouette is still that exact
    silhouette.
    """
    gallery, gallery_labels = [], []
    probes, probe_labels = [], []
    for label, image in enumerate(images):
        traced = silhouettes_by_annotator(image)
        if len(traced) < 2:
            continue
        gallery.append(transform_mask(traced[0]))
        gallery_labels.append(label)
        for other in traced[1:]:
            probes.append(transform_mask(other, rotation=rotation))
            probe_labels.append(label)

    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)

    rows = []
    for name, fn in DESCRIPTORS.items():
        g = np.stack([fn(m) for m in gallery])
        p = np.stack([fn(m) for m in probes])
        mean = g.mean(axis=0, keepdims=True)
        std = np.maximum(g.std(axis=0, keepdims=True), EPS)
        d = np.linalg.norm(((p - mean) / std)[:, None, :]
                           - ((g - mean) / std)[None, :, :], axis=2)
        predicted = gallery_labels[np.argmin(d, axis=1)]
        rows.append({
            "descriptor": name,
            "accuracy": round(float((predicted == probe_labels).mean()), 4),
            "probes": len(probes),
            "classes": len(gallery),
        })
    return rows


def describe(binary: np.ndarray, descriptor: str = "Hu moments (log)") -> np.ndarray:
    return DESCRIPTORS[descriptor](to_gray(binary) if binary.ndim == 3 else binary)
