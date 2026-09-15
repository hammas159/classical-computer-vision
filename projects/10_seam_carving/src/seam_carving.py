"""Seam carving: content-aware resizing, and whether it is worth the cost.

The question
------------
Seam carving (Avidan & Shamir, 2007) removes the lowest-energy connected path of
pixels from an image, repeatedly, so that resizing removes *boring* pixels rather
than squashing everything equally. It is a dynamic-programming algorithm and a
genuinely elegant one.

> **The question worth measuring:** at what amount of resizing does it stop
> beating a plain rescale, and what does it cost per percent?

Two things make this scoreable without a subjective judgement:

* **Object preservation.** Paste a known object into the image. After carving to
  a target width, measure how much of that object's area survived. A plain
  rescale preserves 100% of it in proportion but distorts its aspect ratio; seam
  carving should preserve its *shape* until it runs out of low-energy paths.
* **Straight-line distortion.** Draw a known straight line. Carving bends it.
  Measuring the bend gives a distortion number that needs no human.

The energy function is the other axis: the algorithm is only as good as its
definition of "boring", and that definition is a one-line choice most write-ups
never examine.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray

EPS = 1e-6


# --------------------------------------------------------------------------- #
# energy functions — the definition of "boring"
# --------------------------------------------------------------------------- #


def energy_gradient(img: np.ndarray) -> np.ndarray:
    """|dx| + |dy| via Sobel — the energy from the original paper."""
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(dx) + np.abs(dy)


def energy_sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """sqrt(dx^2 + dy^2) — the true gradient magnitude rather than the L1 sum."""
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(dx * dx + dy * dy)


def energy_laplacian(img: np.ndarray) -> np.ndarray:
    """|Laplacian| — responds to second-order change, so it peaks on fine detail.

    More sensitive to texture and noise than a first-derivative energy, which
    should make it protect textured regions and carve through smooth gradients
    more readily.
    """
    g = to_float(to_gray(img))
    return np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))


def energy_entropy(img: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Local standard deviation — a cheap stand-in for local information content.

    Unlike a gradient, this is high across a *textured area* rather than only on
    its edges, so it should keep whole busy regions intact instead of protecting
    their outlines and hollowing them out.
    """
    g = to_float(to_gray(img))
    mean = cv2.blur(g, (ksize, ksize))
    sq = cv2.blur(g * g, (ksize, ksize))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


ENERGIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Gradient |dx|+|dy|": energy_gradient,
    "Sobel magnitude": energy_sobel_magnitude,
    "Laplacian": energy_laplacian,
    "Local std (entropy-like)": energy_entropy,
}


# --------------------------------------------------------------------------- #
# the dynamic program
# --------------------------------------------------------------------------- #


def cumulative_energy(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-up DP table of minimum path cost, plus the backtrack pointers.

    ``M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])``

    Vectorised row by row: the inner minimum over three neighbours is done with
    two shifted copies rather than a Python loop over columns, which is the
    difference between this running in milliseconds and in seconds.
    """
    h, w = energy.shape
    M = energy.astype(np.float32).copy()
    backtrack = np.zeros((h, w), np.int32)

    for i in range(1, h):
        prev = M[i - 1]
        left = np.roll(prev, 1)
        right = np.roll(prev, -1)
        left[0] = np.inf   # no wraparound: column 0 has no left neighbour
        right[-1] = np.inf

        stacked = np.stack([left, prev, right])
        idx = np.argmin(stacked, axis=0)
        backtrack[i] = idx - 1  # -1, 0, +1 relative to the current column
        M[i] += stacked[idx, np.arange(w)]

    return M, backtrack


def find_seam(energy: np.ndarray) -> np.ndarray:
    """Column index of the minimum-energy vertical seam, one entry per row."""
    M, backtrack = cumulative_energy(energy)
    h, w = energy.shape
    seam = np.zeros(h, np.int32)
    seam[-1] = int(np.argmin(M[-1]))
    for i in range(h - 2, -1, -1):
        seam[i] = int(np.clip(seam[i + 1] + backtrack[i + 1, seam[i + 1]], 0, w - 1))
    return seam


def remove_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one pixel per row, returning an image one column narrower."""
    h, w = img.shape[:2]
    if img.ndim == 3:
        out = np.zeros((h, w - 1, img.shape[2]), img.dtype)
    else:
        out = np.zeros((h, w - 1), img.dtype)
    for i in range(h):
        j = seam[i]
        out[i] = np.concatenate([img[i, :j], img[i, j + 1 :]], axis=0)
    return out


def carve(
    img: np.ndarray,
    target_width: int,
    energy_fn: Callable[[np.ndarray], np.ndarray] = energy_gradient,
    track: np.ndarray | None = None,
):
    """Carve vertical seams until the image reaches ``target_width``.

    ``track`` is an optional mask carried through the identical removals, which
    is how object preservation is measured: the same seams are deleted from the
    mask, so what remains is exactly the part of the object that survived.
    """
    out = img.copy()
    tracked = None if track is None else track.copy()
    while out.shape[1] > target_width:
        seam = find_seam(energy_fn(out))
        out = remove_seam(out, seam)
        if tracked is not None:
            tracked = remove_seam(tracked, seam)
    return (out, tracked) if track is not None else (out, None)


def seam_overlay(img: np.ndarray, energy_fn=energy_gradient, n: int = 40) -> np.ndarray:
    """Draw the next ``n`` seams the algorithm would remove, for the figures."""
    work = img.copy()
    overlay = img.copy()
    offsets = np.zeros(img.shape[0], np.int32)
    keep = np.ones(img.shape[:2], bool)
    for _ in range(n):
        if work.shape[1] < 3:
            break
        seam = find_seam(energy_fn(work))
        for i, j in enumerate(seam):
            # map the column in the shrunken image back to the original
            original_col = np.flatnonzero(keep[i])[j]
            overlay[i, original_col] = (255, 0, 0)
            keep[i, original_col] = False
        work = remove_seam(work, seam)
    return overlay


# --------------------------------------------------------------------------- #
# the scoreable scene
# --------------------------------------------------------------------------- #


def scene_with_object(name: str = "coffee", size: int = 90, seed: int = 0):
    """Paste a solid object and a straight line into an image.

    Returns ``(image, object_mask, line_rows)``. The object gives an area to
    preserve and the line gives a straightness to distort, so both metrics come
    from the same scene with no annotation.
    """
    from shared import io

    rng = np.random.default_rng(seed)
    img = io.sample(name).copy()
    h, w = img.shape[:2]

    x = int(rng.integers(w // 4, 3 * w // 4 - size))
    y = int(rng.integers(h // 4, 3 * h // 4 - size))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x, y), (x + size, y + size), 255, -1)
    img[mask > 0] = (220, 40, 40)

    line_x = w // 3
    cv2.line(img, (line_x, 0), (line_x, h - 1), (20, 220, 20), 3)
    return img, mask, line_x


def line_straightness(img: np.ndarray) -> float:
    """Standard deviation, in px, of the green line's column position per row.

    A perfectly straight line scores 0. Seam carving bends it, and the amount of
    bend is a distortion measure that needs no human judgement.
    """
    f = img.astype(np.int32)
    greenness = f[..., 1] - (f[..., 0] + f[..., 2]) // 2
    cols = []
    for row in greenness:
        if row.max() > 60:
            cols.append(float(np.argmax(row)))
    return float(np.std(cols)) if len(cols) > 2 else float("nan")


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("coffee", "rocket", "chelsea", "astronaut")
REDUCTION_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.45)


def evaluate_energies(reduction: float = 0.25, images=IMAGES, runs: int = 1):
    """Compare energy functions, plus a plain rescale control."""
    rows = []
    for name, fn in ENERGIES.items():
        kept, straight, ms = [], [], []
        for i, image_name in enumerate(images):
            img, mask, _ = scene_with_object(image_name, seed=i)
            target = int(img.shape[1] * (1.0 - reduction))
            (out, out_mask), timing = timeit(
                lambda a=img, m=mask, t=target, g=fn: carve(a, t, g, track=m),
                runs=runs,
                warmup=0,
            )
            kept.append(float((out_mask > 0).sum()) / max(float((mask > 0).sum()), 1.0))
            straight.append(line_straightness(out))
            ms.append(timing.median_ms)
        rows.append(
            {
                "energy": name,
                "object_kept": round(float(np.mean(kept)), 4),
                "line_bend_px": round(float(np.nanmean(straight)), 3),
                "median_ms": round(float(np.median(ms)), 1),
            }
        )

    # the control: a plain resize keeps every object pixel in proportion but
    # squashes the aspect ratio, and never bends a straight vertical line
    kept, straight, ms = [], [], []
    for i, image_name in enumerate(images):
        img, mask, _ = scene_with_object(image_name, seed=i)
        target = int(img.shape[1] * (1.0 - reduction))
        out, timing = timeit(
            lambda a=img, t=target: cv2.resize(a, (t, a.shape[0]), interpolation=cv2.INTER_AREA),
            runs=runs,
            warmup=0,
        )
        out_mask = cv2.resize(mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        kept.append(float((out_mask > 0).sum()) / max(float((mask > 0).sum()), 1.0))
        straight.append(line_straightness(out))
        ms.append(timing.median_ms)
    rows.append(
        {
            "energy": "Plain rescale (control)",
            "object_kept": round(float(np.mean(kept)), 4),
            "line_bend_px": round(float(np.nanmean(straight)), 3),
            "median_ms": round(float(np.median(ms)), 1),
        }
    )
    return rows


def sweep_reduction(images=IMAGES, levels=REDUCTION_LEVELS, energy_fn=energy_gradient):
    """How far can you carve before the image gives up?"""
    rows = []
    for red in levels:
        kept, straight = [], []
        for i, image_name in enumerate(images):
            img, mask, _ = scene_with_object(image_name, seed=i)
            target = int(img.shape[1] * (1.0 - red))
            out, out_mask = carve(img, target, energy_fn, track=mask)
            kept.append(float((out_mask > 0).sum()) / max(float((mask > 0).sum()), 1.0))
            straight.append(line_straightness(out))
        rows.append(
            {
                "reduction": red,
                "object_kept": round(float(np.mean(kept)), 4),
                "line_bend_px": round(float(np.nanmean(straight)), 3),
            }
        )
    return rows
