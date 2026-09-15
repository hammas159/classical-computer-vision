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
  discrete image they are not, and the error grows as the shape gets small —
  because rotation resamples the pixel grid.
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


def largest_contour(binary: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# --------------------------------------------------------------------------- #
# the descriptors
# --------------------------------------------------------------------------- #


def desc_hu(binary: np.ndarray) -> np.ndarray:
    """Seven Hu moments, log-transformed.

    The raw values span many orders of magnitude, so a Euclidean distance between
    them is decided entirely by the first moment. ``sign(h) * log|h|`` compresses
    the range while **keeping the sign**, which matters because the 7th moment's
    sign is the reflection detector.
    """
    m = cv2.moments(binary, binaryImage=True)
    hu = cv2.HuMoments(m).ravel()
    return np.sign(hu) * np.log10(np.abs(hu) + EPS)


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

    The same rotation is applied at several raster sizes. If the invariance
    failure is a discretisation artefact rather than a flaw in the theory, the
    error must shrink as the shape gets larger.
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


def describe(binary: np.ndarray, descriptor: str = "Hu moments (log)") -> np.ndarray:
    return DESCRIPTORS[descriptor](to_gray(binary) if binary.ndim == 3 else binary)
