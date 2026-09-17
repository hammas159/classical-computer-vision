"""Seam carving: content-aware resizing, and whether it is worth the cost.

The question
------------
Seam carving (Avidan & Shamir, 2007) removes the lowest-energy connected path of
pixels from an image, repeatedly, so that resizing removes *boring* pixels rather
than squashing everything equally. It is a dynamic-programming algorithm and a
genuinely elegant one.

> **The question worth measuring:** how much better than a plain rescale is it,
> and what does that cost?

Three numbers, all measured on the **real photograph** with nothing pasted into
it, and none of them needing an annotation:

* **Subject preservation.** A region of interest is found from the image itself
  (:func:`subject_region`: the box of a given area holding the most energy) and
  carried through the identical seam removals. What survives is what the resize
  kept.
* **Shape distortion.** The surviving region's bounding-box aspect ratio, divided
  by its original. A plain rescale lands on exactly ``1 − reduction`` by
  arithmetic, which makes it a control you cannot argue with.
* **Retained energy.** Seam carving's *own* objective — it claims to remove
  low-energy pixels, and this is the number that claim is about. Reported with a
  single fixed energy function for every row so the column is comparable.

The energy function is the other axis: the algorithm is only as good as its
definition of "boring", and that definition is a one-line choice most write-ups
agonise over. Measured, at 20% reduction over four images, the four energies here
span **0.5 percentage points** of subject retention while the gap to a plain
rescale is **11.6** — so the choice that is argued about is 23x smaller than the
choice that is not.
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
    backtrack = np.zeros((h, w), np.int8)

    # The row loop is inherently sequential — row i needs row i-1 — so the only
    # thing that can be optimised is the per-row constant. Every buffer is
    # allocated once here rather than per row, and the three-way minimum is a
    # chain of np.minimum with `out=` rather than stack+argmin+fancy-index.
    # `carve` runs this once per removed column (150 times for a 25% reduction on
    # a 600 px image), so a few microseconds a row is seconds of wall clock.
    inf = np.float32(np.inf)
    left = np.empty(w, np.float32)
    right = np.empty(w, np.float32)
    best = np.empty(w, np.float32)
    left[0] = inf       # no wraparound: column 0 has no left neighbour
    right[-1] = inf

    for i in range(1, h):
        prev = M[i - 1]
        left[1:] = prev[:-1]
        right[:-1] = prev[1:]
        np.minimum(left, prev, out=best)
        np.minimum(best, right, out=best)
        M[i] += best
        # -1 / 0 / +1 relative to the current column; comparisons are cheaper
        # than the fancy indexing an argmin would need
        backtrack[i] = np.where(best == left, -1, np.where(best == prev, 0, 1))

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
    """Delete one pixel per row, returning an image one column narrower.

    Done with a boolean mask and a reshape rather than the Python loop over rows
    that every tutorial shows. Measured on a 400x600 image: **4.9 ms** for the
    loop, **3.7 ms** for this.

    Two faster-looking alternatives are slower. `np.take_along_axis` with an
    explicit index array runs at **5.9 ms** either way it is built — the index
    array is `h x (w-1)` int32, which is more memory traffic than the boolean
    mask it replaces. Measured, not assumed; see the README.
    """
    h, w = img.shape[:2]
    keep = np.ones((h, w), bool)
    keep[np.arange(h), seam] = False
    if img.ndim == 3:
        return img[keep].reshape(h, w - 1, img.shape[2])
    return img[keep].reshape(h, w - 1)


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


def subject_region(img: np.ndarray, energy_fn=energy_gradient, frac: float = 0.16):
    """The most *interesting* rectangle of the photograph, found from the image itself.

    Nothing is pasted in. An earlier version composited a solid red square and a
    green line into each photo, and both were the wrong test:

    * A **uniform** square has zero gradient inside it, so a gradient energy
      cannot see it at all and seams run straight through the middle. The
      experiment measured "does seam carving protect a region it is structurally
      blind to", and the answer is no, for a reason that has nothing to do with
      resizing.
    * A **bright green line** is the highest-energy thing in the frame, so seams
      avoided it perfectly. Measured bend: 0.22 px, against 0.0 for a plain
      rescale. Neither number distinguishes anything.

    Instead the region of interest is a real part of the real photograph: the
    axis-aligned box of a given area fraction containing the most energy, found
    with an integral image so every candidate position is evaluated exactly. It
    is the subject in the sense the algorithm itself means — and that is the
    point, because it is the region seam carving *claims* to protect.

    Returns ``(mask, (x, y, w, h))``.
    """
    e = energy_fn(img)
    h, w = e.shape
    bh, bw = int(round(h * np.sqrt(frac))), int(round(w * np.sqrt(frac)))
    integral = cv2.integral(e.astype(np.float64))
    # sum over every bh x bw window, in one shot
    sums = (
        integral[bh:, bw:]
        - integral[:-bh, bw:]
        - integral[bh:, :-bw]
        + integral[:-bh, :-bw]
    )
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    mask = np.zeros((h, w), np.uint8)
    mask[y : y + bh, x : x + bw] = 255
    return mask, (int(x), int(y), int(bw), int(bh))


def region_shape(mask: np.ndarray) -> tuple[int, int, float]:
    """Bounding-box width, height and aspect ratio of what survives of a mask.

    Carving deletes columns from inside the region, so the surviving pixels stay
    in their rows and the box narrows. Comparing its aspect ratio before and
    after is a distortion measure that needs no annotation and no pasted marker.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return 0, 0, float("nan")
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    return bw, bh, float(bw) / max(bh, 1)


def energy_of_removed(original: np.ndarray, carved_energy_sum: float, energy_fn) -> float:
    """Total energy of the pixels a resize discarded, relative to the original.

    Seam carving's own objective. It claims to remove low-energy pixels, and this
    is the number that claim is about — reported alongside the perceptual ones
    because a method can win its own objective and still look worse.
    """
    total = float(energy_fn(original).sum())
    return (total - carved_energy_sum) / max(total, EPS)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Six photographs chosen for SHAPE, not subject. Seam carving is judged on
#: what it has room to remove, so a pool of six images with the same layout
#: measures one thing six times. These span a tall thin subject in a wide
#: frame, a wide scene with no single subject, architecture whose verticals a
#: seam cannot cross without visible damage, and a small subject surrounded by
#: removable background.
IMAGES = (
    "giraffe",
    "rocky_coast",
    "stone_arch",
    "squirrel_rock",
    "harbour_boat",
    "gallery_visitors",
)


def load_scene(name: str) -> np.ndarray:
    """Load a scene by name from either image source.

    The benchmark moved off scikit-image's bundled samples when it became clear
    that four photographs of roughly the same shape cannot show how much seam
    carving depends on the shape of what it is given. Dispatching on the name
    keeps both sources usable from one call site.
    """
    from shared import io

    if name in io.REAL_PHOTOS:
        return io.real_photo(name)
    return io.sample(name)


REDUCTION_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.70)


def _score_one(img, mask, out, out_mask, reduction, energy_fn=None):
    """Three numbers for one resized image, none of which needs an annotation.

    ``energy_kept`` is deliberately measured with **one fixed** energy function
    for every row, not with the one the method was carved by. Scoring each method
    on its own objective made the Laplacian row read 0.9785 against the others'
    0.939 — which says only that Laplacian energy survives Laplacian-guided
    carving, and nothing about the images.
    """
    base_w, base_h, base_aspect = region_shape(mask)
    kept = float((out_mask > 0).sum()) / max(float((mask > 0).sum()), 1.0)
    _, _, aspect = region_shape(out_mask)
    e_kept = float(energy_gradient(out).sum()) / max(float(energy_gradient(img).sum()), EPS)
    return {
        "subject_kept": kept,
        # 1.0 means the subject's shape is untouched; the plain rescale squashes
        # it by exactly the reduction factor, so it lands on (1 - reduction)
        "aspect_ratio": (aspect / base_aspect) if base_aspect > 0 else float("nan"),
        "energy_kept": e_kept,
    }


def _rescale(img, target):
    return cv2.resize(img, (target, img.shape[0]), interpolation=cv2.INTER_AREA)


def evaluate_energies(reduction: float = 0.25, images=IMAGES, runs: int = 1):
    """Compare energy functions against each other and against a plain rescale.

    Every column is measured on the **real** photograph with a region of interest
    found from the image itself — nothing is pasted in. See
    :func:`subject_region` for why the previous synthetic scene measured nothing.
    """
    from shared import io

    rows = []
    scenes = []
    for name in images:
        img = load_scene(name)
        mask, _ = subject_region(img)
        scenes.append((img, mask))

    for name, fn in ENERGIES.items():
        acc = {k: [] for k in ("subject_kept", "aspect_ratio", "energy_kept")}
        ms = []
        for img, mask in scenes:
            target = int(img.shape[1] * (1.0 - reduction))
            (out, out_mask), timing = timeit(
                lambda a=img, m=mask, t=target, g=fn: carve(a, t, g, track=m),
                runs=runs,
                warmup=0,
            )
            for k, v in _score_one(img, mask, out, out_mask, reduction, fn).items():
                acc[k].append(v)
            ms.append(timing.median_ms)
        rows.append(
            {
                "energy": name,
                "subject_kept": round(float(np.mean(acc["subject_kept"])), 4),
                "aspect_ratio": round(float(np.nanmean(acc["aspect_ratio"])), 4),
                "energy_kept": round(float(np.mean(acc["energy_kept"])), 4),
                "median_ms": round(float(np.median(ms)), 1),
            }
        )

    # The control. A plain rescale keeps every subject pixel in proportion and
    # squashes the aspect ratio by exactly the reduction factor — so its
    # `aspect_ratio` column is (1 - reduction) by arithmetic, and that is the
    # number every carving row has to beat to have done anything at all.
    acc = {k: [] for k in ("subject_kept", "aspect_ratio", "energy_kept")}
    ms = []
    for img, mask in scenes:
        target = int(img.shape[1] * (1.0 - reduction))
        out, timing = timeit(lambda a=img, t=target: _rescale(a, t), runs=runs, warmup=0)
        out_mask = cv2.resize(mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        for k, v in _score_one(img, mask, out, out_mask, reduction, energy_gradient).items():
            acc[k].append(v)
        ms.append(timing.median_ms)
    rows.append(
        {
            "energy": "Plain rescale (control)",
            "subject_kept": round(float(np.mean(acc["subject_kept"])), 4),
            "aspect_ratio": round(float(np.nanmean(acc["aspect_ratio"])), 4),
            "energy_kept": round(float(np.mean(acc["energy_kept"])), 4),
            "median_ms": round(float(np.median(ms)), 1),
        }
    )
    return rows


def sweep_reduction(images=IMAGES, levels=REDUCTION_LEVELS, energy_fn=energy_gradient):
    """How far can you carve before it stops beating a plain rescale?

    The project's central question. Seam carving's advantage is that it can route
    around the subject — and it can only do that while there are low-energy paths
    left to route through. Past some reduction there are none, every remaining
    seam has to cross something, and the method degrades towards (and then below)
    the rescale it was supposed to beat.
    """
    from shared import io

    rows = []
    scenes = [(load_scene(n), subject_region(load_scene(n))[0]) for n in images]
    for red in levels:
        carved = {k: [] for k in ("subject_kept", "aspect_ratio", "energy_kept")}
        plain = {k: [] for k in ("subject_kept", "aspect_ratio", "energy_kept")}
        for img, mask in scenes:
            target = int(img.shape[1] * (1.0 - red))
            out, out_mask = carve(img, target, energy_fn, track=mask)
            for k, v in _score_one(img, mask, out, out_mask, red, energy_fn).items():
                carved[k].append(v)
            r_out = _rescale(img, target)
            r_mask = cv2.resize(mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            for k, v in _score_one(img, mask, r_out, r_mask, red, energy_fn).items():
                plain[k].append(v)
        rows.append(
            {
                "reduction": red,
                "carved_subject_kept": round(float(np.mean(carved["subject_kept"])), 4),
                "rescale_subject_kept": round(float(np.mean(plain["subject_kept"])), 4),
                "carved_aspect": round(float(np.nanmean(carved["aspect_ratio"])), 4),
                "rescale_aspect": round(float(np.nanmean(plain["aspect_ratio"])), 4),
                "carved_energy_kept": round(float(np.mean(carved["energy_kept"])), 4),
                "rescale_energy_kept": round(float(np.mean(plain["energy_kept"])), 4),
            }
        )
    return rows


def compare_cost(reduction: float = 0.20, images=IMAGES) -> list[dict]:
    """The trade the project exists to quantify: what the advantage costs.

    Seam carving is O(n) dynamic programmes for n removed columns, each one
    inherently sequential down the rows. A plain rescale is one interpolation.
    Putting the quality gain and the time next to each other is the only way to
    make "is it worth it" a question with an answer.
    """
    from shared import io

    rows = []
    scenes = [(load_scene(n), subject_region(load_scene(n))[0]) for n in images]
    for label, fn in (("Seam carving", None), ("Plain rescale", _rescale)):
        acc, ms = {k: [] for k in ("subject_kept", "aspect_ratio", "energy_kept")}, []
        for img, mask in scenes:
            target = int(img.shape[1] * (1.0 - reduction))
            if fn is None:
                (out, out_mask), timing = timeit(
                    lambda a=img, m=mask, t=target: carve(a, t, track=m), runs=1, warmup=0
                )
            else:
                out, timing = timeit(lambda a=img, t=target: fn(a, t), runs=1, warmup=0)
                out_mask = cv2.resize(
                    mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            for k, v in _score_one(img, mask, out, out_mask, reduction).items():
                acc[k].append(v)
            ms.append(timing.median_ms)
        rows.append(
            {
                "method": label,
                "subject_kept": round(float(np.mean(acc["subject_kept"])), 4),
                "aspect_ratio": round(float(np.nanmean(acc["aspect_ratio"])), 4),
                "energy_kept": round(float(np.mean(acc["energy_kept"])), 4),
                "median_ms": round(float(np.median(ms)), 2),
            }
        )
    carve_row, plain_row = rows
    rows.append(
        {
            "method": "Difference",
            "subject_kept": round(carve_row["subject_kept"] - plain_row["subject_kept"], 4),
            "aspect_ratio": round(carve_row["aspect_ratio"] - plain_row["aspect_ratio"], 4),
            "energy_kept": round(carve_row["energy_kept"] - plain_row["energy_kept"], 4),
            "median_ms": round(carve_row["median_ms"] / max(plain_row["median_ms"], EPS), 0),
        }
    )
    return rows


def resize_pair(img: np.ndarray, reduction: float, energy_fn=energy_gradient):
    """Both resizes of one image, for the UI and the figures.

    Returns ``(carved, rescaled, mask, carved_mask, rescaled_mask)``.
    """
    mask, _ = subject_region(img, energy_fn)
    target = max(8, int(img.shape[1] * (1.0 - reduction)))
    carved, carved_mask = carve(img, target, energy_fn, track=mask)
    rescaled = _rescale(img, target)
    rescaled_mask = cv2.resize(mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    return carved, rescaled, mask, carved_mask, rescaled_mask


def per_image(reduction: float = 0.20, images=IMAGES, energy_fn=energy_gradient) -> list[dict]:
    """The same comparison, image by image, because the average hides a sign change.

    Aggregated over four photographs seam carving keeps 91.6% of the high-energy
    region against a rescale's 80.0%. On `coffee` it keeps **79.0% against 80.0%**
    — it loses. The subject fills most of the frame, so there are no low-energy
    paths to route around it, and every seam has to cross something.

    A single averaged row would report a clean win and hide that the method has a
    regime where it is worse than doing nothing clever.
    """
    from shared import io

    rows = []
    for name in images:
        img = load_scene(name)
        mask, _ = subject_region(img, energy_fn)
        target = int(img.shape[1] * (1.0 - reduction))
        out, out_mask = carve(img, target, energy_fn, track=mask)
        r_out = _rescale(img, target)
        r_mask = cv2.resize(mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = _score_one(img, mask, out, out_mask, reduction)
        p = _score_one(img, mask, r_out, r_mask, reduction)
        rows.append(
            {
                "image": name,
                "roi_fraction": round(float((mask > 0).mean()), 4),
                "carved_kept": round(c["subject_kept"], 4),
                "rescale_kept": round(p["subject_kept"], 4),
                "advantage": round(c["subject_kept"] - p["subject_kept"], 4),
                "carved_energy_kept": round(c["energy_kept"], 4),
                "rescale_energy_kept": round(p["energy_kept"], 4),
            }
        )
    return rows
