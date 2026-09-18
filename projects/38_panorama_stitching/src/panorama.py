"""Panorama stitching: five blenders, a real pair, and an exposure difference.

The question
------------
Warp one photograph onto another and paste them together. Projects 25 and 46
already measured the *estimating* of the homography, so this one is about what
happens afterwards: the compositing, which is where a panorama visibly succeeds
or fails and which almost no write-up measures.

> **The first result is that the obvious metric is a trap.** Scored by PSNR
> against the original photograph, the winner is the control that **does not
> stitch at all** -- it keeps the first frame wherever the first frame has data
> (32.7 dB against every blender's 30.9-32.2). That is not a quirk: the first
> frame *is* the original in its own region, so any blending that mixes the second
> frame in can only move away from it. **A panorama cannot be scored by fidelity
> to one of its inputs.**

> **The metric and the goal point in opposite directions.** Measured at the seam
> -- the brightness step straight across the join -- the PSNR winner is the
> *worst* method. With a 20% exposure difference between the frames, which is
> what auto-exposure does between two shots seconds apart, not stitching leaves a
> **25 grey-level** step and a multi-band blend leaves **11**, while their PSNRs
> run the other way (24.4 against 23.0).

> **And with the exposure difference removed, blending is worth nothing.** Every
> method, including the two controls, leaves a seam step of 6-7 grey levels --
> which is simply the scene's own texture. A blender comparison run on frames of
> equal exposure measures nothing at all.

> **The failure the selection axis predicts:** a wall of identical brick courses
> yields **19 RANSAC inliers per megapixel** against 11,414 for an aerial town --
> a factor of 600. Every correspondence on the brick wall is as good as every
> other, so there is no consistent homography to find, and no blender can rescue
> a panorama that was never registered.

Where the ground truth comes from
---------------------------------
Two sources, deliberately different.

**Synthetic homographies on real photographs.** A real photograph is warped by a
**known** homography, so the truth is the matrix that was used and the answer for
the overlap is the original image. What is real is the content; what is
constructed is the viewpoint change, which is exactly the thing being estimated —
the same arrangement project 08 uses for a camera path, and defensible for the
same reason.

**One genuinely real pair.** `graf1` and `graf3` from the Oxford graffiti set are
two photographs of one wall from two viewpoints, and they give 355 RANSAC inliers
of 520 matches. The set's *published* homographies 404 upstream, so there is no
external truth for this pair; what can be measured without one is **cycle
consistency** -- warping there and back must return every point to itself -- and
that is what `cycle_error` reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from shared.io import real_photo, to_gray

EPS = 1e-9

#: The twelve photographs, in order of **registrability**: RANSAC inliers per
#: megapixel between the image and a known warp of itself. That is the project's
#: own response, and it spans four orders of magnitude -- from a brick wall where
#: every correspondence is ambiguous to an aerial town where nothing is.
IMAGES = (
    "brick_wall_courses",
    "roof_shingles",
    "coastal_city_from_the_air",
    "irrigated_fields_from_the_air",
    "tugboat_under_the_bridge",
    "coarse_woven_fabric",
    "elephant_crossing_a_road",
    "cyclists_on_a_track",
    "tank_in_scrub",
    "street_grid_from_the_air",
    "fibrous_matting",
    "hillside_town_from_the_air",
)

#: The real pair. Cached by `tools/fetch_assets.py --set graf`; the set's
#: published homographies are 404 upstream, so only 2 of its 11 files exist.
GRAF = Path.home() / ".cache" / "classical-cv-images" / "assets" / "graf"

#: The viewpoint change between the two frames: a rotation, a scale and some
#: perspective, the same for every photograph so the comparison is of scenes
#: rather than of transforms.
VIEWPOINT = np.array([[0.94, -0.08, 12.0],
                      [0.06, 0.97, -9.0],
                      [1.6e-4, 7.0e-5, 1.0]], np.float64)

#: What fraction of the canvas each frame covers. 0.65 each leaves a 30% overlap
#: strip, which is about what a person shooting a panorama leaves.
COVERAGE = 0.65

#: How much brighter the second frame is, as a multiplier. Auto-exposure between
#: two frames of a real panorama routinely differs by this much.
EXPOSURE = 1.20

RATIO = 0.75


def load(name: str) -> np.ndarray:
    return real_photo(name)


def graf_available() -> bool:
    return (GRAF / "graf1.png").exists() and (GRAF / "graf3.png").exists()


def load_graf():
    if not graf_available():
        raise FileNotFoundError(
            f"{GRAF} is missing graf1/graf3. Run "
            "`python tools/fetch_assets.py --set graf`.")
    a = cv2.imread(str(GRAF / "graf1.png"), cv2.IMREAD_COLOR)
    b = cv2.imread(str(GRAF / "graf3.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def match(a: np.ndarray, b: np.ndarray, max_features: int = 4000):
    """Ratio-tested SIFT correspondences between two images."""
    sift = cv2.SIFT_create(max_features)
    k0, d0 = sift.detectAndCompute(to_gray(a), None)
    k1, d1 = sift.detectAndCompute(to_gray(b), None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]
    p0 = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    p1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return p0, p1


def estimate_homography(a: np.ndarray, b: np.ndarray, threshold: float = 3.0):
    """``(H, inliers, total)`` mapping points of ``a`` onto ``b``."""
    p0, p1 = match(a, b)
    if len(p0) < 4:
        return None, 0, len(p0)
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, threshold)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers, len(p0)


def registrability(image: np.ndarray, H: np.ndarray = VIEWPOINT) -> dict:
    """Inliers per megapixel between the image and a known warp of itself.

    The project's selection axis. A scene of repeated identical structure gives
    matches that are all equally good, so RANSAC has no consistent homography to
    find -- which is the single most common way a real stitch fails, and it is a
    property of the scene rather than of the algorithm.
    """
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, H, (w, h))
    _, inliers, total = estimate_homography(image, warped)
    return {
        "matches": total,
        "inliers": inliers,
        "inlier_rate": round(inliers / max(total, 1), 4),
        "per_megapixel": round(inliers / (h * w / 1e6), 1),
    }


def cycle_error(a: np.ndarray, b: np.ndarray, n: int = 500,
                seed: int = 0) -> float:
    """Warp there and back: every point must return to itself.

    The only check available on the real pair, whose published homography 404s
    upstream. It cannot detect a *consistently* wrong homography -- H and its
    own inverse always compose to the identity -- so it is reported as a sanity
    check and not as an accuracy.
    """
    Hab, _, _ = estimate_homography(a, b)
    Hba, _, _ = estimate_homography(b, a)
    if Hab is None or Hba is None:
        return float("nan")
    rng = np.random.default_rng(seed)
    h, w = a.shape[:2]
    pts = np.stack([rng.uniform(0, w, n), rng.uniform(0, h, n)], axis=1)
    there = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), Hab)
    back = cv2.perspectiveTransform(there, Hba).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(back - pts, axis=1)))


# --------------------------------------------------------------------------- #
# building a pair
# --------------------------------------------------------------------------- #


def make_pair(image: np.ndarray, H: np.ndarray = VIEWPOINT,
              exposure: float = EXPOSURE, coverage: float = COVERAGE):
    """Two frames that each see part of one scene, and must be joined.

    Returns ``(left, left_cover, right, right_cover, H, truth)``.

    **Each frame covers only part of the canvas**, so neither alone is the
    answer. `left` is the photograph with its right-hand portion missing;
    `right` is the same photograph from the warped viewpoint with its left-hand
    portion missing, and at a different exposure -- which is what auto-exposure
    does between two frames of a real panorama. `truth` is the whole photograph.

    An earlier version made `left` the entire photograph, which made `truth` and
    `left` identical: the do-nothing control scored an infinite PSNR and the
    comparison measured nothing.
    """
    h, w = image.shape[:2]
    cut = int(w * coverage)

    left = image.copy()
    left_cover = np.zeros((h, w), bool)
    left_cover[:, :cut] = True
    left[~left_cover] = 0

    warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)
    right_cover_full = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H,
                                           (w, h)) > 128
    right_cover = right_cover_full.copy()
    right_cover[:, :w - cut] = False
    right = warped.copy()
    right[~right_cover] = 0
    if abs(exposure - 1.0) > EPS:
        bright = np.clip(right.astype(np.float32) * exposure, 0, 255).astype(np.uint8)
        right = np.where(right_cover[..., None], bright, right)

    return left, left_cover, right, right_cover, H, image


def align(right: np.ndarray, right_cover: np.ndarray, H: np.ndarray, shape):
    """The second frame and its coverage, warped back onto the first's canvas."""
    h, w = shape[:2]
    Hi = np.linalg.inv(H)
    aligned = cv2.warpPerspective(right, Hi, (w, h), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(right_cover.astype(np.uint8) * 255, Hi, (w, h)) > 128
    return aligned, cover


def regions(left_cover: np.ndarray, aligned_cover: np.ndarray):
    """``(overlap, right_only, covered)`` -- the three parts of the canvas."""
    overlap = left_cover & aligned_cover
    right_only = aligned_cover & ~left_cover
    return overlap, right_only, left_cover | aligned_cover


# --------------------------------------------------------------------------- #
# the blenders
# --------------------------------------------------------------------------- #


def blend_overwrite(left, aligned, mask, right_only=None, **kw):
    """Paste the second frame over the first wherever it has data. The control.

    No blending at all. Every visible seam in the results is this one's fault,
    and its score is what the others have to beat.
    """
    out = left.copy()
    paint = mask if right_only is None else (mask | right_only)
    out[paint] = aligned[paint]
    return out


def blend_keep_first(left, aligned, mask, right_only=None, **kw):
    """Use the second frame only where the first has nothing. The other control.

    It does the *joining* and none of the *blending*, so it isolates what
    blending is worth: on a pair of equal exposure it is very hard to beat, and
    with an exposure difference it leaves a hard step at the seam.
    """
    out = left.copy()
    if right_only is not None:
        out[right_only] = aligned[right_only]
    return out


def blend_average(left, aligned, mask, right_only=None, **kw):
    """Mean of the two frames in the overlap. Ghosts anything that moved."""
    out = left.astype(np.float32).copy()
    if right_only is not None:
        out[right_only] = aligned[right_only].astype(np.float32)
    out[mask] = 0.5 * (left[mask].astype(np.float32)
                       + aligned[mask].astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_feather(left, aligned, mask, right_only=None, width: int = 41, **kw):
    """Linear alpha ramp across the overlap boundary.

    The distance transform gives a weight that is 0 at the seam and 1 well
    inside, so the transition is spread over ``width`` pixels instead of
    happening at one.
    """
    both = mask if right_only is None else (mask | right_only)
    # 1 deep inside the second frame, 0 at the first frame's edge of the overlap
    alpha = cv2.distanceTransform((~left_only_of(mask, right_only)).astype(np.uint8),
                                  cv2.DIST_L2, 3)
    alpha = np.clip(alpha / max(width, 1), 0.0, 1.0)[..., None]
    out = (1 - alpha) * left.astype(np.float32) + alpha * aligned.astype(np.float32)
    out[~both] = left[~both]
    if right_only is not None:
        out[right_only] = aligned[right_only]
    return np.clip(out, 0, 255).astype(np.uint8)


def left_only_of(mask, right_only):
    """Where only the first frame has data: the far side of the overlap ramp."""
    if right_only is None:
        return ~mask
    return ~(mask | right_only)


#: Levels in the Laplacian pyramid. Enough that the coarsest level is a few
#: pixels across, which is what lets a low-frequency exposure difference be
#: blended over the whole overlap instead of over a 41-pixel ramp.
BANDS = 5


def blend_multiband(left, aligned, mask, right_only=None, bands: int = BANDS,
                    **kw):
    """Burt and Adelson's multi-band blend.

    The idea in one sentence: **blend each spatial frequency over a distance
    proportional to its wavelength.** A coarse exposure difference is blended
    across the whole overlap, so it disappears; fine detail is blended over a few
    pixels, so it stays sharp. A single ramp cannot do both, which is why
    feathering either leaves a band or smears the texture.
    """
    both = mask if right_only is None else (mask | right_only)
    alpha = cv2.distanceTransform((~left_only_of(mask, right_only)).astype(np.uint8),
                                  cv2.DIST_L2, 3)
    alpha = np.clip(alpha / 41.0, 0.0, 1.0)

    a = left.astype(np.float32)
    b = aligned.astype(np.float32)

    ga, gb, gm = [a], [b], [alpha]
    for _ in range(bands - 1):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))

    la, lb = [], []
    for i in range(bands - 1):
        size = (ga[i].shape[1], ga[i].shape[0])
        la.append(ga[i] - cv2.pyrUp(ga[i + 1], dstsize=size))
        lb.append(gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size))
    la.append(ga[-1])
    lb.append(gb[-1])

    blended = []
    for i in range(bands):
        w = gm[i][..., None]
        blended.append(la[i] * (1 - w) + lb[i] * w)

    out = blended[-1]
    for i in range(bands - 2, -1, -1):
        size = (blended[i].shape[1], blended[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + blended[i]

    out[~both] = left[~both]
    if right_only is not None:
        out[right_only] = aligned[right_only]
    return np.clip(out, 0, 255).astype(np.uint8)


BLENDERS: dict[str, Callable] = {
    "Keep the first frame (control)": blend_keep_first,
    "Overwrite (control)": blend_overwrite,
    "Average": blend_average,
    "Feather (41 px)": blend_feather,
    "Multi-band (5 levels)": blend_multiband,
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def psnr_on(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """PSNR over the overlap only.

    Whole-image PSNR would be dominated by the region only one frame covers,
    where every method is identical by construction.
    """
    if not mask.any():
        return float("nan")
    a = pred[mask].astype(np.float64)
    b = truth[mask].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse <= EPS else float(10.0 * np.log10(255.0 ** 2 / mse))


def seam_step(image: np.ndarray, boundary: np.ndarray, reach: int = 6) -> float:
    """Median brightness step straight across a seam line, in grey levels.

    The measure that says what blending is *for*. `seam_visibility` compares
    gradient energy to its surroundings, which a textured scene can drown out;
    this samples the pixels a few steps either side of the boundary and reports
    how far apart they are. A perfect blend gives roughly the difference between
    neighbouring pixels anywhere else in the picture; a hard paste gives the
    exposure difference.
    """
    if not boundary.any():
        return float("nan")
    g = to_gray(image).astype(np.float32)
    ys, xs = np.nonzero(boundary)
    h, w = g.shape
    left = np.clip(xs - reach, 0, w - 1)
    right = np.clip(xs + reach, 0, w - 1)
    steps = np.abs(g[ys, right] - g[ys, left])
    return float(np.median(steps))


def seam_line(mask: np.ndarray, side: str = "left") -> np.ndarray:
    """The vertical boundary of a region, as a thin mask.

    ``side`` picks which edge: a stitched canvas has two, and the blenders put
    their transition at different ones.
    """
    if not mask.any():
        return np.zeros_like(mask)
    cols = np.nonzero(mask.any(axis=0))[0]
    x = int(cols[0] if side == "left" else cols[-1])
    line = np.zeros_like(mask)
    line[:, max(0, x - 1):min(mask.shape[1], x + 2)] = True
    return line & mask.any(axis=1)[:, None]


def seam_visibility(image: np.ndarray, mask: np.ndarray, band: int = 5) -> float:
    """Gradient energy on the overlap boundary against its own surroundings.

    A ratio of 1.0 means the boundary is indistinguishable from the rest of the
    picture. Above 1 there is a visible edge where no edge belongs.
    """
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((band, band), np.uint8)) > 0
    inside = cv2.erode(mask.astype(np.uint8),
                       np.ones((band * 4, band * 4), np.uint8)) > 0
    if not edge.any() or not inside.any():
        return float("nan")
    g = to_gray(image).astype(np.float32)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.hypot(dx, dy)
    return float(energy[edge].mean() / max(energy[inside].mean(), EPS))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def stitch(image: np.ndarray, blender: str, exposure: float = EXPOSURE,
           H: np.ndarray = VIEWPOINT):
    """Build a pair from one photograph and join it. Returns everything scored."""
    left, lc, right, rc, H, truth = make_pair(image, H=H, exposure=exposure)
    aligned, ac = align(right, rc, H, image.shape)
    overlap, right_only, covered = regions(lc, ac)
    out = BLENDERS[blender](left, aligned, overlap, right_only=right_only)
    return {
        "output": out, "truth": truth, "left": left, "right": right,
        "aligned": aligned, "overlap": overlap, "right_only": right_only,
        "covered": covered,
    }


def worst_seam_step(out: np.ndarray, overlap: np.ndarray) -> float:
    """The larger of the two seam steps: a stitched canvas has two joins."""
    near = seam_step(out, seam_line(overlap, "left"))
    far = seam_step(out, seam_line(overlap, "right"))
    values = [v for v in (near, far) if v == v]
    return max(values) if values else float("nan")


def evaluate(images=IMAGES, exposure: float = EXPOSURE) -> list[dict]:
    """Every blender on every photograph, by both metrics."""
    acc = {name: {"psnr": [], "seam": [], "visibility": []} for name in BLENDERS}
    for name in images:
        image = load(name)
        for blender in BLENDERS:
            r = stitch(image, blender, exposure)
            acc[blender]["psnr"].append(psnr_on(r["output"], r["truth"],
                                                r["covered"]))
            acc[blender]["seam"].append(worst_seam_step(r["output"], r["overlap"]))
            acc[blender]["visibility"].append(
                seam_visibility(r["output"], r["overlap"]))
    return [
        {
            "blender": name,
            "psnr_db": round(float(np.nanmean(a["psnr"])), 3),
            "seam_step": round(float(np.nanmean(a["seam"])), 2),
            "seam_visibility": round(float(np.nanmean(a["visibility"])), 3),
            "is_control": "control" in name,
        }
        for name, a in acc.items()
    ]


def per_image(images=IMAGES, exposure: float = EXPOSURE) -> list[dict]:
    rows = []
    for name in images:
        image = load(name)
        reg = registrability(image)
        row = {"image": name, "inliers_per_mpx": reg["per_megapixel"],
               "inlier_rate": reg["inlier_rate"], "matches": reg["matches"]}
        for blender in BLENDERS:
            r = stitch(image, blender, exposure)
            row[blender] = round(worst_seam_step(r["output"], r["overlap"]), 2)
        rows.append(row)
    return rows


def metrics_disagree(images=IMAGES, exposure: float = EXPOSURE) -> dict:
    """Do PSNR and the seam step rank the blenders the same way?

    The project's headline. They do not, and the disagreement is total at the
    top: the method with the best PSNR is the one that does not stitch.
    """
    rows = evaluate(images, exposure)
    by_psnr = [r["blender"] for r in sorted(rows, key=lambda r: -r["psnr_db"])]
    by_seam = [r["blender"] for r in sorted(rows, key=lambda r: r["seam_step"])]
    return {
        "by_psnr": by_psnr,
        "by_seam_step": by_seam,
        "psnr_winner": by_psnr[0],
        "seam_winner": by_seam[0],
        "psnr_winner_is_the_non_stitching_control": "Keep the first" in by_psnr[0],
        "psnr_winner_seam_rank": by_seam.index(by_psnr[0]) + 1,
    }


def exposure_is_the_whole_story(images=IMAGES,
                                exposures=(1.0, 1.05, 1.1, 1.2, 1.4)) -> list[dict]:
    """How much each blender is worth as the exposure difference grows.

    At 1.0 there is nothing to hide and every method scores the same, which is
    the control that says a comparison on matched exposures measures nothing.
    """
    rows = []
    for exposure in exposures:
        row: dict[str, float] = {"exposure": exposure}
        for blender in BLENDERS:
            steps = []
            for name in images:
                r = stitch(load(name), blender, exposure)
                steps.append(worst_seam_step(r["output"], r["overlap"]))
            row[blender] = round(float(np.nanmean(steps)), 2)
        rows.append(row)
    return rows


def registrability_table(images=IMAGES) -> list[dict]:
    """The selection axis, per photograph."""
    rows = []
    for name in images:
        image = load(name)
        r = registrability(image)
        rows.append({"image": name, **r})
    return rows


def sweep_bands(images=IMAGES[:6], bands=(2, 3, 5, 7),
                exposure: float = EXPOSURE) -> list[dict]:
    """How many pyramid levels a multi-band blend needs.

    Each level doubles the distance the coarsest frequency is blended over, so
    this is really a sweep of "how wide is the widest thing being hidden".
    """
    rows = []
    for n in bands:
        steps = []
        for name in images:
            image = load(name)
            left, lc, right, rc, H, truth = make_pair(image, exposure=exposure)
            aligned, ac = align(right, rc, H, image.shape)
            overlap, right_only, _ = regions(lc, ac)
            out = blend_multiband(left, aligned, overlap, right_only=right_only,
                                  bands=n)
            steps.append(worst_seam_step(out, overlap))
        rows.append({"bands": n, "seam_step": round(float(np.nanmean(steps)), 2)})
    return rows


def sweep_feather_width(images=IMAGES[:6], widths=(5, 21, 41, 81, 161),
                        exposure: float = EXPOSURE) -> list[dict]:
    """The feather ramp width: the one parameter feathering has."""
    rows = []
    for width in widths:
        steps = []
        for name in images:
            image = load(name)
            left, lc, right, rc, H, truth = make_pair(image, exposure=exposure)
            aligned, ac = align(right, rc, H, image.shape)
            overlap, right_only, _ = regions(lc, ac)
            out = blend_feather(left, aligned, overlap, right_only=right_only,
                                width=width)
            steps.append(worst_seam_step(out, overlap))
        rows.append({"width": width, "seam_step": round(float(np.nanmean(steps)), 2)})
    return rows


def real_pair() -> dict:
    """The one genuinely real pair: two photographs of one graffiti wall.

    There is no published homography for it here -- the Oxford set's homography
    files 404 upstream -- so what is reported is the match count, the inlier
    rate, and cycle consistency. None of those is an accuracy, and the section
    says so rather than implying otherwise.
    """
    a, b = load_graf()
    H, inliers, total = estimate_homography(a, b)
    return {
        "matches": total,
        "inliers": inliers,
        "inlier_rate": round(inliers / max(total, 1), 4),
        "cycle_error_px": round(cycle_error(a, b), 4),
        "shape": list(a.shape[:2]),
    }


def stitch_real(blender: str = "Multi-band (5 levels)"):
    """Stitch the real pair, on a canvas big enough to hold both."""
    a, b = load_graf()
    H, _, _ = estimate_homography(b, a)
    if H is None:
        raise RuntimeError("no homography between the graffiti pair")
    h, w = a.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    allpts = np.vstack([moved, [[0, 0], [w, 0], [w, h], [0, h]]])
    x0, y0 = np.floor(allpts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(allpts.max(axis=0)).astype(int)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    size = (int(x1 - x0), int(y1 - y0))

    left = cv2.warpPerspective(a, shift, size)
    left_cover = cv2.warpPerspective(np.full((h, w), 255, np.uint8), shift, size) > 128
    aligned = cv2.warpPerspective(b, shift @ H, size)
    aligned_cover = cv2.warpPerspective(np.full((h, w), 255, np.uint8),
                                        shift @ H, size) > 128

    overlap = left_cover & aligned_cover
    right_only = aligned_cover & ~left_cover
    out = BLENDERS[blender](left, aligned, overlap, right_only=right_only)
    return {"output": out, "left": left, "aligned": aligned, "overlap": overlap,
            "right_only": right_only, "covered": left_cover | aligned_cover}
