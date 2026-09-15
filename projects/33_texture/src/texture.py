"""Texture descriptors: GLCM, LBP, Gabor and Laws, as a classification problem.

The question
------------
Texture descriptors are usually presented as feature *extractors* and evaluated
by showing a feature map. A feature map proves nothing.

> **The claim under test:** turn it into a classification task with a nearest-
> neighbour classifier and no training, and the descriptors separate sharply —
> and they separate *differently* depending on the degradation. LBP should be
> nearly immune to illumination change by construction, because it encodes only
> the sign of local differences. GLCM should not be.

That is the useful output: not "which descriptor is best" but **which invariance
each one actually has**, measured rather than claimed.

No training is involved anywhere. Classification is nearest-neighbour in feature
space with a leave-one-out protocol, which is a pure distance computation — so
this stays inside the repo's no-learning rule while still producing an accuracy.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8

EPS = 1e-9


# --------------------------------------------------------------------------- #
# the descriptors — each returns a 1-D feature vector
# --------------------------------------------------------------------------- #


def feat_glcm(patch: np.ndarray, distances=(1, 3, 5), levels: int = 32) -> np.ndarray:
    """Grey-level co-occurrence matrix statistics (Haralick features).

    Counts how often pairs of grey levels occur at a given offset, then
    summarises that matrix with contrast, homogeneity, energy and correlation.

    Quantising to 32 levels is not a shortcut — a 256x256 co-occurrence matrix
    from a small patch is almost all zeros, and its statistics become noise.
    """
    from skimage.feature import graycomatrix, graycoprops

    q = (to_gray(patch).astype(np.float32) / 256.0 * levels).astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(q, distances=list(distances), angles=angles,
                        levels=levels, symmetric=True, normed=True)
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props]).astype(np.float32)


def feat_lbp(patch: np.ndarray, radius: int = 3, n_points: int | None = None) -> np.ndarray:
    """Local binary patterns, uniform variant, as a normalised histogram.

    Each pixel is compared with its circular neighbours and the *signs* of the
    differences form a binary code. Because only signs are kept, any monotonic
    change in illumination leaves the code untouched — LBP is invariant to
    brightness and contrast by construction, not by tuning.

    The "uniform" variant keeps only codes with at most two bitwise transitions,
    which covers the vast majority of real texture patterns and collapses the
    rest into one bin, making the histogram far more stable on small patches.
    """
    from skimage.feature import local_binary_pattern

    n_points = n_points or 8 * radius
    codes = local_binary_pattern(to_gray(patch), n_points, radius, method="uniform")
    n_bins = int(n_points + 2)
    hist, _ = np.histogram(codes.ravel(), bins=n_bins, range=(0, n_bins))
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def _gabor_bank(ksize: int = 21, sigmas=(2.0, 4.0), lambdas=(6.0, 12.0), n_orientations: int = 6):
    """A bank of Gabor kernels over orientation, scale and wavelength."""
    kernels = []
    for theta in np.linspace(0, np.pi, n_orientations, endpoint=False):
        for sigma in sigmas:
            for lam in lambdas:
                k = cv2.getGaborKernel((ksize, ksize), sigma, float(theta), lam, 0.5, 0,
                                       ktype=cv2.CV_32F)
                kernels.append(k / max(np.abs(k).sum(), EPS))
    return kernels


_GABOR_KERNELS = None


def feat_gabor(patch: np.ndarray) -> np.ndarray:
    """Gabor energy: mean and standard deviation of each filter's response.

    A Gabor filter is a sinusoid under a Gaussian envelope — it is tuned to one
    orientation *and* one spatial frequency at once, which is exactly what
    distinguishes texture from edge detection. The bank is cached because
    rebuilding 24 kernels per patch would put kernel construction into the timing.
    """
    global _GABOR_KERNELS
    if _GABOR_KERNELS is None:
        _GABOR_KERNELS = _gabor_bank()
    g = to_float(to_gray(patch))
    feats = []
    for k in _GABOR_KERNELS:
        response = cv2.filter2D(g, cv2.CV_32F, k)
        feats.extend([float(np.mean(np.abs(response))), float(np.std(response))])
    return np.asarray(feats, np.float32)


#: Laws' 1-D kernels. Their outer products give 25 separable 5x5 masks, each
#: tuned to a primitive: Level, Edge, Spot, Wave, Ripple.
LAWS_VECTORS = {
    "L5": np.array([1, 4, 6, 4, 1], np.float32),
    "E5": np.array([-1, -2, 0, 2, 1], np.float32),
    "S5": np.array([-1, 0, 2, 0, -1], np.float32),
    "W5": np.array([-1, 2, 0, -2, 1], np.float32),
    "R5": np.array([1, -4, 6, -4, 1], np.float32),
}


def feat_laws(patch: np.ndarray) -> np.ndarray:
    """Laws' texture energy from 25 separable 5x5 masks.

    The oldest method here and still competitive. Pairs that are transposes of
    each other (L5E5 and E5L5) are averaged, which is what makes the descriptor
    rotation-*symmetric* — not invariant, but no longer arbitrary about which
    axis is which.
    """
    g = to_float(to_gray(patch))
    # remove local mean first, or L5L5 dominates everything with plain brightness
    g = g - cv2.blur(g, (15, 15))

    names = list(LAWS_VECTORS)
    maps: dict[str, np.ndarray] = {}
    for a in names:
        for b in names:
            kernel = np.outer(LAWS_VECTORS[a], LAWS_VECTORS[b])
            maps[a + b] = np.abs(cv2.filter2D(g, cv2.CV_32F, kernel))

    feats = []
    done = set()
    for a in names:
        for b in names:
            key = tuple(sorted((a + b, b + a)))
            if key in done:
                continue
            done.add(key)
            energy = (maps[a + b] + maps[b + a]) / 2.0 if a != b else maps[a + b]
            feats.extend([float(np.mean(energy)), float(np.std(energy))])
    return np.asarray(feats, np.float32)


DESCRIPTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "GLCM (Haralick)": feat_glcm,
    "LBP (uniform)": feat_lbp,
    "Gabor bank": feat_gabor,
    "Laws energy": feat_laws,
    "Raw histogram (control)": lambda p: (
        np.histogram(to_gray(p).ravel(), bins=32, range=(0, 256))[0]
        / max(to_gray(p).size, 1)
    ).astype(np.float32),
}


# --------------------------------------------------------------------------- #
# the dataset — patches cut from the bundled texture images
# --------------------------------------------------------------------------- #

TEXTURES = ("brick", "grass", "gravel")
PATCH = 96
PER_CLASS = 12


def build_patches(patch: int = PATCH, per_class: int = PER_CLASS, seed: int = 0,
                  degradation: str | None = None, strength: float = 0.0):
    """Cut labelled patches from each bundled texture, optionally degraded.

    Returns ``(patches, labels)``. Patches are cut on a grid rather than at random
    so the same spatial content is used across every degradation — otherwise a
    difference between conditions could be the crop, not the degradation.
    """
    from shared import io, synth

    rng = np.random.default_rng(seed)
    patches, labels = [], []
    for label, name in enumerate(TEXTURES):
        img = to_gray(io.sample(name))
        h, w = img.shape
        coords = []
        step_y = max(1, (h - patch) // max(int(np.sqrt(per_class)), 1))
        step_x = max(1, (w - patch) // max(int(np.sqrt(per_class)), 1))
        for y in range(0, h - patch + 1, step_y):
            for x in range(0, w - patch + 1, step_x):
                coords.append((y, x))
        rng.shuffle(coords)
        for y, x in coords[:per_class]:
            p = img[y : y + patch, x : x + patch].copy()
            if degradation == "illumination":
                p = to_uint8(to_float(p) * (1.0 - strength) + strength * 0.1)
            elif degradation == "gamma":
                p = to_uint8(np.power(to_float(p), 1.0 + strength))
            elif degradation == "noise":
                p = synth.gaussian_noise(p, sigma=strength, seed=int(x + y))
            elif degradation == "rotation":
                m = cv2.getRotationMatrix2D((patch / 2, patch / 2), strength, 1.0)
                p = cv2.warpAffine(p, m, (patch, patch), borderMode=cv2.BORDER_REFLECT)
            elif degradation == "scale":
                s = max(0.2, 1.0 + strength)
                m = cv2.getRotationMatrix2D((patch / 2, patch / 2), 0.0, s)
                p = cv2.warpAffine(p, m, (patch, patch), borderMode=cv2.BORDER_REFLECT)
            patches.append(p)
            labels.append(label)
    return patches, np.asarray(labels)


# --------------------------------------------------------------------------- #
# classification without training
# --------------------------------------------------------------------------- #


def normalise_features(features: np.ndarray) -> np.ndarray:
    """Z-score each dimension across the dataset.

    Mandatory here. Gabor energies and GLCM correlations differ by orders of
    magnitude, and a Euclidean distance without normalisation is decided entirely
    by whichever dimension happens to have the largest units.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, EPS)


def leave_one_out_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Nearest-neighbour accuracy, each sample classified by all the others.

    No training, no parameters, no train/test split to get wrong. Every sample is
    a test sample exactly once, and the "model" is the distance function.
    """
    f = normalise_features(features)
    d = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    predicted = labels[np.argmin(d, axis=1)]
    return float((predicted == labels).mean())


def class_separability(features: np.ndarray, labels: np.ndarray) -> float:
    """Between-class scatter over within-class scatter (Fisher-style ratio).

    Accuracy saturates at 1.0 and then stops distinguishing descriptors. This
    keeps discriminating: a descriptor with a wide margin scores higher than one
    that classifies correctly but barely.
    """
    f = normalise_features(features)
    overall = f.mean(axis=0)
    between, within = 0.0, 0.0
    for label in np.unique(labels):
        sel = f[labels == label]
        centre = sel.mean(axis=0)
        between += len(sel) * float(np.sum((centre - overall) ** 2))
        within += float(np.sum((sel - centre) ** 2))
    return float(between / max(within, EPS))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

ILLUMINATION_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
GAMMA_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)
NOISE_LEVELS = (0.0, 5.0, 15.0, 30.0, 50.0)
ROTATION_LEVELS = (0.0, 15.0, 30.0, 45.0, 90.0)


def evaluate_descriptors(degradation: str | None = None, strength: float = 0.0,
                         seed: int = 0, runs: int = 1):
    """Accuracy, separability, dimensionality and cost for every descriptor."""
    patches, labels = build_patches(seed=seed, degradation=degradation, strength=strength)
    rows = []
    for name, fn in DESCRIPTORS.items():
        _, timing = timeit(lambda f=fn, p=patches[0]: f(p), runs=runs, warmup=1)
        features = np.stack([fn(p) for p in patches])
        rows.append(
            {
                "descriptor": name,
                "accuracy": round(leave_one_out_accuracy(features, labels), 4),
                "separability": round(class_separability(features, labels), 4),
                "dimensions": int(features.shape[1]),
                "median_ms": round(float(timing.median_ms), 3),
            }
        )
    return rows


def sweep_degradation(degradation: str, levels, seed: int = 0):
    """Accuracy against one degradation, for every descriptor.

    This is where the invariance claims get tested. LBP keeps only the *sign* of
    local differences, so the illumination and gamma columns should barely move
    for it and should move for GLCM and the raw histogram.
    """
    rows = []
    for strength in levels:
        scored = evaluate_descriptors(degradation=degradation, strength=strength, seed=seed)
        row: dict[str, float] = {degradation: strength}
        for r in scored:
            row[r["descriptor"]] = r["accuracy"]
        rows.append(row)
    return rows


def invariance_summary(seed: int = 0):
    """One row per descriptor: how much accuracy each degradation costs it.

    Reported as the drop from the clean baseline to the strongest degradation, so
    the numbers are directly comparable across descriptors with different
    baselines.
    """
    baseline = {r["descriptor"]: r["accuracy"] for r in evaluate_descriptors(seed=seed)}
    degradations = {
        "illumination": ILLUMINATION_LEVELS[-1],
        "gamma": GAMMA_LEVELS[-1],
        "noise": NOISE_LEVELS[-1],
        "rotation": ROTATION_LEVELS[-1],
    }
    rows = []
    for name in DESCRIPTORS:
        row: dict[str, float | str] = {"descriptor": name, "clean_accuracy": baseline[name]}
        for deg, strength in degradations.items():
            scored = evaluate_descriptors(degradation=deg, strength=strength, seed=seed)
            after = next(r["accuracy"] for r in scored if r["descriptor"] == name)
            row[f"{deg}_accuracy"] = after
            row[f"{deg}_drop"] = round(baseline[name] - after, 4)
        rows.append(row)
    return rows


def feature_map(img: np.ndarray, descriptor: str = "LBP (uniform)", window: int = 32):
    """A per-window descriptor response, for the figures.

    Reduced to one number per window (the feature vector's norm) so it can be
    displayed as an image. Useful as an illustration, and explicitly *not* what
    the descriptors are scored on — a picture of features is what this project
    exists to replace.
    """
    gray = to_gray(img)
    h, w = gray.shape
    fn = DESCRIPTORS[descriptor]
    out = np.zeros((h // window, w // window), np.float32)
    for i, y in enumerate(range(0, h - window + 1, window)):
        for j, x in enumerate(range(0, w - window + 1, window)):
            if i < out.shape[0] and j < out.shape[1]:
                out[i, j] = float(np.linalg.norm(fn(gray[y : y + window, x : x + window])))
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
