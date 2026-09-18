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

#: Twelve textures, selected by `tools/select_images.py --axis texture` from the
#: 58 USC-SIPI Brodatz-style plates and then pruned by hand so that no two are
#: the same *kind* of surface. Chosen to put the invariance claims under real
#: pressure rather than to be easy:
#:
#:   * three are strongly oriented (wood grain, straw, thatch) and rotation
#:     should cost a rotation-variant descriptor on exactly those;
#:   * two are exactly periodic (perforated metal, brick paving), which is what
#:     a Gabor bank is built for;
#:   * `sand_ripples` is the lowest-contrast plate in the set and `crushed_gravel`
#:     the highest, so an illumination change does not mean the same thing to
#:     every class;
#:   * `coarse_stucco` and `crushed_gravel` are both blobby and isotropic — a
#:     deliberate confusable pair, because twelve unmistakable classes would make
#:     every descriptor look good.
#:
#: A thirteenth candidate (SIPI 1.2.04, a coarse linen weave) was rejected by
#: `tools/check_image_reuse.py`: its perceptual hash is within 5 bits of the
#: herringbone plate. Two Brodatz textures with different numbers can be near
#: duplicates, which is worth knowing before using them as separate classes.
TEXTURES = (
    "tree_bark_ridged",     # coarse, irregular, unoriented
    "herringbone_weave",    # fine, regular, two diagonals
    "wood_grain",           # strongly oriented, low contrast
    "brick_paving",         # periodic lattice + mortar lines
    "packed_cobbles",       # cellular blobs with dark grout
    "thatch_fibres",        # one diagonal, uneven illumination
    "coarse_stucco",        # high contrast, blobby, unoriented
    "dry_straw",            # thin strands, one orientation
    "sand_ripples",         # the lowest-contrast plate here
    "stipple_plaster",      # fine isotropic bumps, bright
    "perforated_metal",     # an exactly periodic dot lattice
    "crushed_gravel",       # large angular stones, unoriented
)

#: The operating point, chosen by `sweep_patch_size` rather than by taste. At
#: 96 px all four descriptors score a perfect 1.000 on twelve classes, and a
#: saturated benchmark cannot rank anything — the same trap project 28 fell into
#: with its synthetic scene. 32 px leaves every descriptor short of ceiling and
#: spread over 0.764 to 0.993, which is where a comparison means something.
PATCH = 32
PER_CLASS = 12


def load_scene(name: str) -> np.ndarray:
    """One texture plate, greyscale.

    Kept as a named function rather than inlined so that `run.py`, the tests and
    `infer.py` all read the same pixels — the patches below are cut on a grid,
    and a different loader would silently give them different content.
    """
    from shared import io

    return to_gray(io.real_photo(name))


def crop_centres(shape: tuple[int, int], patch: int, count: int, seed: int = 0,
                 headroom: float = 1.0) -> list[tuple[int, int]]:
    """Patch centres on a grid, restricted to a disc about the plate's centre.

    The disc is what makes a rotation test honest. A rotation about the plate
    centre preserves distance from it, so a centre inside a disc of radius
    ``min(h, w) / 2 - patch`` stays inside the plate no matter the angle, and the
    rotated patch is cut from real pixels rather than from a reflected border.

    ``headroom`` shrinks that radius for a scale change, which does not preserve
    distance from the centre.
    """
    h, w = shape
    radius = min(h, w) / 2.0 / max(headroom, 1.0) - patch
    if radius <= patch:
        raise ValueError(f"patch {patch} is too large for a {w}x{h} plate")

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    step = max(patch // 2, 4)
    grid = [(y, x)
            for y in range(patch, h - patch + 1, step)
            for x in range(patch, w - patch + 1, step)
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2]
    if len(grid) < count:
        raise ValueError(f"only {len(grid)} usable centres for {count} patches")

    rng = np.random.default_rng(seed)
    rng.shuffle(grid)
    return grid[:count]


def _cut(img: np.ndarray, centre: tuple[float, float], patch: int) -> np.ndarray:
    # A patch occupying rows [t, t + patch) is centred on t + (patch - 1) / 2,
    # not on t + patch / 2. The half-pixel matters: with the wrong convention a
    # 180 degree rotation lands one row off its own mirror image.
    y = int(round(centre[0] - (patch - 1) / 2.0))
    x = int(round(centre[1] - (patch - 1) / 2.0))
    y = int(np.clip(y, 0, img.shape[0] - patch))
    x = int(np.clip(x, 0, img.shape[1] - patch))
    return img[y : y + patch, x : x + patch].copy()


def _warp_plate(img: np.ndarray, degrees: float, scale: float):
    """Warp the whole plate, and return it with the matrix that moved it.

    Geometry is applied to the plate and the crop centre is carried through the
    same matrix, so a rotated probe shows the **same surface region** as its
    clean counterpart, turned. Rotating a 32 px patch in place instead would
    invent its corners from a reflected border, and a descriptor could then be
    scoring the seam.
    """
    h, w = img.shape
    # About ((w-1)/2, (h-1)/2), the centre of the *pixel grid*. Rotating about
    # (w/2, h/2) — the usual half-pixel mistake — shifts every rotated patch one
    # pixel off, which is not nothing when the patch is 32 across.
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), degrees, scale)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return out, m


def _move(m: np.ndarray, centre: tuple[float, float]) -> tuple[float, float]:
    y, x = centre
    return (float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
            float(m[0, 0] * x + m[0, 1] * y + m[0, 2]))


GEOMETRIC = ("rotation", "scale")


def build_split(patch: int = PATCH, per_class: int = PER_CLASS, seed: int = 0,
                degradation: str | None = None, strength: float = 0.0,
                textures: tuple[str, ...] = TEXTURES):
    """Return ``(gallery, probes, labels)`` — clean gallery, degraded probes.

    Two things here are the whole experiment, and both were wrong in the first
    version of this project:

    **The two sets are cut from disjoint crops.** Leave-one-out on a single set
    is a different and easier task, and comparing a cross-condition score
    against a leave-one-out baseline made a 90 degree rotation appear to *raise*
    accuracy (LBP 0.674 to 0.847). It did not: the probes were simply landing on
    different parts of the plate. Splitting the crops up front removes the
    confound instead of correcting for it afterwards.

    **Only the probes are degraded.** Degrading both sides measures nothing —
    nearest-neighbour matching is invariant to anything that moves every sample
    the same way, so a global brightness change cost even a plain intensity
    histogram nothing at all.
    """
    from shared import synth

    headroom = 1.0
    if degradation == "scale" and strength:
        headroom = max(1.0, 1.0 + strength)

    gallery, probes, labels = [], [], []
    for label, name in enumerate(textures):
        plate = load_scene(name)
        centres = crop_centres(plate.shape, patch, 2 * per_class, seed=seed,
                               headroom=headroom)
        gallery_centres = centres[:per_class]
        probe_centres = centres[per_class:]

        warped, matrix = plate, None
        if degradation in GEOMETRIC and strength:
            degrees = strength if degradation == "rotation" else 0.0
            scale = 1.0 if degradation == "rotation" else max(0.2, 1.0 + strength)
            warped, matrix = _warp_plate(plate, degrees, scale)

        for centre in gallery_centres:
            gallery.append(_cut(plate, centre, patch))

        for i, centre in enumerate(probe_centres):
            if matrix is not None:
                p = _cut(warped, _move(matrix, centre), patch)
            else:
                p = _cut(plate, centre, patch)
            if degradation == "illumination" and strength:
                p = to_uint8(to_float(p) * (1.0 - strength) + strength * 0.1)
            elif degradation == "gamma" and strength:
                p = to_uint8(np.power(to_float(p), 1.0 + strength))
            elif degradation == "noise" and strength:
                p = synth.gaussian_noise(p, sigma=strength, seed=seed * 1000 + label * 50 + i)
            probes.append(p)

        labels.extend([label] * per_class)

    return gallery, probes, np.asarray(labels)


def build_patches(patch: int = PATCH, per_class: int = PER_CLASS, seed: int = 0,
                  degradation: str | None = None, strength: float = 0.0,
                  textures: tuple[str, ...] = TEXTURES):
    """The probe half of `build_split`, for callers that want one labelled set.

    Used by the figures and by anything measuring a property of the feature space
    itself (separability), where there is no gallery to match against.
    """
    _, probes, labels = build_split(patch=patch, per_class=per_class, seed=seed,
                                    degradation=degradation, strength=strength,
                                    textures=textures)
    return probes, labels


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


def cross_condition_accuracy(gallery: np.ndarray, probe: np.ndarray,
                             labels: np.ndarray) -> float:
    """Classify **degraded** probes against a **clean** gallery.

    This is the protocol the invariance question actually needs, and getting it
    wrong is silent. Degrading every patch and then running leave-one-out
    compares degraded against degraded — and a nearest-neighbour classifier is
    invariant to anything that moves every sample the same way, so a global
    brightness change scores *zero* cost for every descriptor including a plain
    intensity histogram, which has no such invariance at all.

    The gallery and the probes are cut from disjoint crops (`build_split`), so
    no probe can match its own pixels and nothing has to be masked out.

    Both sides are z-scored with the **gallery's** statistics, because the
    gallery is what stands in for known data; normalising the probes by their
    own mean would quietly undo part of the degradation.
    """
    mean = gallery.mean(axis=0, keepdims=True)
    std = np.maximum(gallery.std(axis=0, keepdims=True), EPS)
    g = (gallery - mean) / std
    p = (probe - mean) / std

    d = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2)
    return float((labels[np.argmin(d, axis=1)] == labels).mean())


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
PATCH_LEVELS = (24, 32, 48, 64, 96)

#: Every headline number is averaged over five crop seeds. One seed is not
#: enough and the project found that out the hard way: fixing a half-pixel
#: convention in `_cut` changed *which* 32 px crops were used and moved GLCM's
#: clean accuracy from 0.993 to 0.958, reversing the clean winner. Neither
#: number was wrong; a single seed simply does not resolve a 0.025 difference.
#: Five seeds is 720 classifications per cell and a standard deviation to print
#: next to it, which is what makes a 0.02 gap readable as "tied".
SEEDS = (0, 1, 2, 3, 4)


def _features(patches, fn) -> np.ndarray:
    return np.stack([fn(p) for p in patches])


def evaluate_descriptors(degradation: str | None = None, strength: float = 0.0,
                         seed: int = 0, runs: int = 1, patch: int = PATCH):
    """Accuracy, separability, dimensionality and cost for every descriptor.

    Accuracy is always cross-condition — degraded probes against a clean
    gallery, on disjoint crops — so that the clean row is measured by exactly the
    same protocol as every degraded row and the two are comparable. Separability
    is measured within the probe set, because it is a property of the feature
    space rather than of a matching protocol.
    """
    gallery, probes, labels = build_split(seed=seed, patch=patch,
                                          degradation=degradation, strength=strength)
    rows = []
    for name, fn in DESCRIPTORS.items():
        _, timing = timeit(lambda f=fn, p=gallery[0]: f(p), runs=runs, warmup=1)
        probe_features = _features(probes, fn)
        accuracy = cross_condition_accuracy(_features(gallery, fn), probe_features, labels)
        rows.append(
            {
                "descriptor": name,
                "accuracy": round(accuracy, 4),
                "separability": round(class_separability(probe_features, labels), 4),
                "dimensions": int(probe_features.shape[1]),
                "median_ms": round(float(timing.median_ms), 3),
            }
        )
    return rows


def evaluate_over_seeds(degradation: str | None = None, strength: float = 0.0,
                        seeds=SEEDS, patch: int = PATCH, runs: int = 1):
    """`evaluate_descriptors` averaged over crop seeds, with the spread kept.

    The spread is the point. Without it a 0.953 and a 0.951 read as a ranking,
    and both carry a standard deviation of 0.025.
    """
    per_seed = [evaluate_descriptors(degradation=degradation, strength=strength,
                                     seed=s, patch=patch, runs=runs)
                for s in seeds]
    rows = []
    for i, name in enumerate(DESCRIPTORS):
        acc = np.array([p[i]["accuracy"] for p in per_seed])
        sep = np.array([p[i]["separability"] for p in per_seed])
        rows.append({
            "descriptor": name,
            "accuracy": round(float(acc.mean()), 4),
            "accuracy_sd": round(float(acc.std()), 4),
            "accuracy_min": round(float(acc.min()), 4),
            "accuracy_max": round(float(acc.max()), 4),
            "separability": round(float(sep.mean()), 4),
            "dimensions": per_seed[0][i]["dimensions"],
            "median_ms": round(float(np.median([p[i]["median_ms"] for p in per_seed])), 3),
            "seeds": len(seeds),
        })
    return rows


def sweep_degradation(degradation: str, levels, seeds=SEEDS, patch: int = PATCH):
    """Accuracy against one degradation, for every descriptor.

    This is where the invariance claims get tested. LBP keeps only the *sign* of
    local differences, so the illumination and gamma columns should barely move
    for it and should move for GLCM and the raw histogram.
    """
    if isinstance(seeds, int):
        seeds = (seeds,)
    rows = []
    for strength in levels:
        scored = evaluate_over_seeds(degradation=degradation, strength=strength,
                                     seeds=seeds, patch=patch)
        row: dict[str, float] = {degradation: strength}
        for r in scored:
            row[r["descriptor"]] = r["accuracy"]
        rows.append(row)
    return rows


def sweep_patch_size(levels=PATCH_LEVELS, seeds=SEEDS):
    """Clean accuracy against patch size — how much surface each one needs.

    This is what chooses the operating point for every other experiment here.
    At 96 px every real descriptor scores a perfect 1.000 and the benchmark
    stops being able to rank them; the differences between descriptors only
    exist while the task is still hard.
    """
    if isinstance(seeds, int):
        seeds = (seeds,)
    rows = []
    for size in levels:
        scored = evaluate_over_seeds(seeds=seeds, patch=size)
        row: dict[str, float] = {"patch": size}
        for r in scored:
            row[r["descriptor"]] = r["accuracy"]
        rows.append(row)
    return rows


#: The strongest level of each degradation, plus the 45 degree rotation, which
#: is kept as its own column because it is the worst angle for a square-sampled
#: operator and the only column the do-nothing control wins.
INVARIANCE_COLUMNS = {
    "illumination": ("illumination", ILLUMINATION_LEVELS[-1]),
    "gamma": ("gamma", GAMMA_LEVELS[-1]),
    "noise": ("noise", NOISE_LEVELS[-1]),
    "rotation45": ("rotation", 45.0),
    "rotation": ("rotation", ROTATION_LEVELS[-1]),
}


def invariance_summary(seeds=SEEDS):
    """One row per descriptor: how much accuracy each degradation costs it.

    Reported as the drop from the clean baseline to the strongest degradation, so
    the numbers are directly comparable across descriptors with different
    baselines. Every cell is a mean over `seeds` crop seeds and carries its own
    standard deviation, because several of the gaps here are smaller than the
    seed-to-seed spread and should not be read as rankings.
    """
    if isinstance(seeds, int):
        seeds = (seeds,)
    base_rows = evaluate_over_seeds(seeds=seeds)
    baseline = {r["descriptor"]: r for r in base_rows}
    scored = {
        column: {r["descriptor"]: r
                 for r in evaluate_over_seeds(degradation=deg, strength=s, seeds=seeds)}
        for column, (deg, s) in INVARIANCE_COLUMNS.items()
    }

    rows = []
    for name in DESCRIPTORS:
        row: dict[str, float | str | int] = {
            "descriptor": name,
            "clean_accuracy": baseline[name]["accuracy"],
            "clean_sd": baseline[name]["accuracy_sd"],
        }
        for column in INVARIANCE_COLUMNS:
            after = scored[column][name]
            row[f"{column}_accuracy"] = after["accuracy"]
            row[f"{column}_sd"] = after["accuracy_sd"]
            row[f"{column}_drop"] = round(baseline[name]["accuracy"] - after["accuracy"], 4)
        row["seeds"] = len(seeds)
        rows.append(row)
    return rows


def confusion(descriptor: str = "Laws energy", seed: int = 0, patch: int = PATCH,
              degradation: str | None = None, strength: float = 0.0):
    """Which textures get mistaken for which, for one descriptor.

    Returns a ``(12, 12)`` integer matrix, rows true and columns predicted. The
    per-descriptor accuracy says how often it is wrong; this says *what* it is
    wrong about, which is the only way to tell a descriptor that is confused
    between two genuinely similar surfaces from one that is merely noisy.
    """
    gallery, probes, labels = build_split(seed=seed, patch=patch,
                                          degradation=degradation, strength=strength)
    fn = DESCRIPTORS[descriptor]
    g = _features(gallery, fn)
    p = _features(probes, fn)
    mean, std = g.mean(0, keepdims=True), np.maximum(g.std(0, keepdims=True), EPS)
    d = np.linalg.norm(((p - mean) / std)[:, None, :] - ((g - mean) / std)[None, :, :], axis=2)
    predicted = labels[np.argmin(d, axis=1)]

    n = len(TEXTURES)
    matrix = np.zeros((n, n), np.int32)
    for true, pred in zip(labels, predicted):
        matrix[true, pred] += 1
    return matrix


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
