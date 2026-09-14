"""Document scanner: find the page, rectify it, make it readable.

The question
------------
A phone photo of a page is taken at an angle, on a cluttered desk, under uneven
light. Every tutorial solves this with "Canny, find the biggest contour, warp".
**How many pixels wrong is that, and does anything beat it?**

Six classical page-boundary detectors are compared on scenes whose four true
corner positions are known exactly, so the answer is a number in pixels rather
than a screenshot that looks about right.

Pipeline
--------
``detect corners -> rectify by homography -> binarise``

Each stage is scored on its own terms:

* corners  -> mean corner error in px, and success rate
* rectify  -> IoU of the recovered page area against the true page area
* binarise -> IoU of the recovered text against the true text mask

No training, no learned model, no downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from shared import synth
from shared.bench import timeit
from shared.io import to_gray
from shared.metrics import corner_error, iou

# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Uses the sum/difference trick: for a convex quad the top-left has the
    smallest ``x + y`` and the top-right the smallest ``y - x``. Ordering is not
    cosmetic — an unordered quad produces a homography that mirrors or rotates
    the page, and the corner error then reports a huge number for a detection
    that was actually correct.
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.float32(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]]
    )


def quad_from_contour(contour: np.ndarray, image_area: float) -> np.ndarray | None:
    """Reduce a contour to a convex 4-gon, sweeping the approximation tolerance.

    A single fixed ``epsilon`` for ``approxPolyDP`` is the usual reason a scanner
    "sometimes works": too small and the page keeps 6 vertices from sensor noise,
    too large and a corner is cut off entirely. Sweeping is cheap and removes the
    magic number.
    """
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None
    for frac in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 0.05 * image_area < area < 0.98 * image_area:
                return order_corners(approx.reshape(4, 2))
    return None


def _largest_quad(mask: np.ndarray, top_k: int = 6) -> np.ndarray | None:
    """Find the biggest plausible 4-gon in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_area = float(mask.shape[0] * mask.shape[1])
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]:
        quad = quad_from_contour(c, image_area)
        if quad is not None:
            return quad
    return None


# --------------------------------------------------------------------------- #
# the six detectors
# --------------------------------------------------------------------------- #


def detect_canny_contour(img: np.ndarray) -> np.ndarray | None:
    """The textbook method: blur -> Canny -> dilate -> largest convex 4-gon.

    The dilation is not decoration. Canny leaves one-pixel gaps wherever the page
    edge crosses a shadow, ``findContours`` then walks straight through the gap,
    and the "page" contour becomes the whole image.
    """
    gray = to_gray(img)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)  # 1:3 ratio, the standard convention
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _largest_quad(edges)


def detect_otsu_contour(img: np.ndarray) -> np.ndarray | None:
    """Brightness segmentation: the page is much lighter than the desk.

    Otsu assumes a bimodal histogram. A page under a lighting gradient is *two*
    bright modes plus the desk, and this is where the method is expected to slip.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _largest_quad(mask)


def detect_morph_gradient(img: np.ndarray) -> np.ndarray | None:
    """Morphological gradient (dilate - erode) instead of a derivative operator.

    Responds to any intensity step regardless of sign or orientation, so unlike
    Canny it needs no threshold pair chosen up front.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    return _largest_quad(mask)


def detect_saturation(img: np.ndarray) -> np.ndarray | None:
    """Colour, not brightness: paper is bright and **unsaturated**, the desk is brown.

    HSV saturation is ``(max - min) / max``, which is scale-invariant: multiplying
    a pixel by a shading factor leaves S unchanged. A saturation test should
    therefore be immune to the lighting gradient that troubles every grayscale
    method. That is the hypothesis this detector exists to test.

    The split point comes from Otsu on the saturation channel rather than a
    hand-picked constant, so this method is tuned no more carefully than the
    others it is being compared against.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return _largest_quad(mask)


def detect_hough_lines(img: np.ndarray) -> np.ndarray | None:
    """Fit four straight page edges and intersect them.

    A page boundary *is* four lines, so fitting lines uses more evidence per
    corner than tracing a contour: every pixel along an edge votes, and the
    corner is recovered even where the actual corner is occluded or blurred.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=70, minLineLength=min(h, w) // 5, maxLineGap=40
    )
    if segs is None or len(segs) < 4:
        return None

    lines = []  # (angle in [0, pi), signed offset from the image centre)
    cx, cy = w / 2.0, h / 2.0
    for x1, y1, x2, y2 in segs[:, 0]:
        theta = np.arctan2(y2 - y1, x2 - x1) % np.pi
        nx, ny = -np.sin(theta), np.cos(theta)  # unit normal
        lines.append((theta, nx * (x1 - cx) + ny * (y1 - cy), (nx, ny), (x1, y1)))

    # Split into two orientation groups around the dominant angle. The two edge
    # families of a quadrilateral differ by roughly 90 degrees, so one split is
    # enough — no clustering algorithm needed.
    thetas = np.array([t for t, _, _, _ in lines])
    dominant = thetas[np.argmax([np.sum(np.cos(2 * (thetas - t))) for t in thetas])]
    delta = np.abs(np.angle(np.exp(1j * 2 * (thetas - dominant)))) / 2
    group_a = [ln for ln, d in zip(lines, delta) if d < np.pi / 4]
    group_b = [ln for ln, d in zip(lines, delta) if d >= np.pi / 4]
    if len(group_a) < 2 or len(group_b) < 2:
        return None

    def extremes(group):
        offs = [o for _, o, _, _ in group]
        return group[int(np.argmin(offs))], group[int(np.argmax(offs))]

    a1, a2 = extremes(group_a)
    b1, b2 = extremes(group_b)

    def intersect(la, lb):
        (_, _, (nax, nay), (ax, ay)) = la
        (_, _, (nbx, nby), (bx, by)) = lb
        A = np.array([[nax, nay], [nbx, nby]], np.float64)
        if abs(np.linalg.det(A)) < 1e-6:  # parallel: no corner exists
            return None
        rhs = np.array([nax * ax + nay * ay, nbx * bx + nby * by], np.float64)
        return np.linalg.solve(A, rhs)

    pts = [intersect(a, b) for a in (a1, a2) for b in (b1, b2)]
    if any(p is None for p in pts):
        return None
    pts = np.float32(pts)
    if not (
        (-0.2 * w < pts[:, 0]).all()
        and (pts[:, 0] < 1.2 * w).all()
        and (-0.2 * h < pts[:, 1]).all()
        and (pts[:, 1] < 1.2 * h).all()
    ):
        return None
    return order_corners(pts)


def detect_minarea_rect(img: np.ndarray) -> np.ndarray | None:
    """Deliberate baseline: fit a rotated **rectangle** to the bright blob.

    Included to be beaten. A page seen at an angle is a general quadrilateral,
    not a rotated rectangle, so this method cannot be right no matter how well
    the blob is segmented — and the size of its error is the measurement of how
    much perspective there actually is.
    """
    gray = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))
    return order_corners(box)


#: Every detector, in the order they appear in the results table.
DETECTORS: dict[str, Callable[[np.ndarray], np.ndarray | None]] = {
    "Canny + contour": detect_canny_contour,
    "Otsu + contour": detect_otsu_contour,
    "Morph gradient": detect_morph_gradient,
    "Saturation (HSV)": detect_saturation,
    "Hough lines": detect_hough_lines,
    "minAreaRect (baseline)": detect_minarea_rect,
}


# --------------------------------------------------------------------------- #
# rectification
# --------------------------------------------------------------------------- #


def aspect_from_edges(corners: np.ndarray) -> float:
    """Naive aspect ratio: longest top/bottom edge over longest left/right edge.

    This is what essentially every document-scanner tutorial does, and it is
    **wrong by construction**. The page is seen in perspective, so its far edge
    is foreshortened; measuring pixel lengths in the image measures the
    projection, not the page. A portrait page photographed from a low angle comes
    out square or landscape.
    """
    c = order_corners(corners)
    w = max(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3]))
    h = max(np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1]))
    return float(w / h) if h > 0 else 1.0


def aspect_from_perspective(corners: np.ndarray, image_shape: tuple[int, int]) -> float | None:
    """Recover the page's **true** width/height ratio from the four corners.

    A rectangle viewed in perspective carries enough information to recover both
    the camera's focal length and the rectangle's real aspect ratio, provided the
    principal point is assumed to be the image centre. This is the standard
    closed-form solution (Zhang & He, *Whiteboard Scanning and Image
    Enhancement*, 2007).

    Returns None for the degenerate cases, which are real and must be handled
    rather than ignored:

    * the quad is a parallelogram — the view is effectively affine, the vanishing
      points are at infinity, and the focal length is unrecoverable;
    * the recovered ``f**2`` is negative, which means the quad cannot be the
      image of any rectangle under this camera model.

    In both cases the caller falls back to :func:`aspect_from_edges`.
    """
    c = order_corners(corners)
    # rectangle corners (0,0), (w,0), (0,h), (w,h) map to TL, TR, BL, BR
    m1, m2, m3, m4 = (np.append(p, 1.0).astype(np.float64) for p in (c[0], c[1], c[3], c[2]))

    h_img, w_img = image_shape[:2]
    u0, v0 = w_img / 2.0, h_img / 2.0

    den2 = np.dot(np.cross(m2, m4), m3)
    den3 = np.dot(np.cross(m3, m4), m2)
    if abs(den2) < 1e-9 or abs(den3) < 1e-9:
        return None

    k2 = np.dot(np.cross(m1, m4), m3) / den2
    k3 = np.dot(np.cross(m1, m4), m2) / den3

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    if abs(n2[2]) < 1e-7 or abs(n3[2]) < 1e-7:
        return None  # parallelogram: affine view, no perspective to exploit

    f_sq = -(1.0 / (n2[2] * n3[2])) * (
        (n2[0] - n2[2] * u0) * (n3[0] - n3[2] * u0)
        + (n2[1] - n2[2] * v0) * (n3[1] - n3[2] * v0)
    )
    if f_sq <= 0:
        return None

    f = float(np.sqrt(f_sq))
    A = np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]])
    A_inv = np.linalg.inv(A)
    M = A_inv.T @ A_inv

    num = float(n2 @ M @ n2)
    den = float(n3 @ M @ n3)
    if num <= 0 or den <= 0:
        return None
    return float(np.sqrt(num / den))


def rectify(img: np.ndarray, corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp the quad defined by ``corners`` onto a flat ``(width, height)`` page."""
    w, h = size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(order_corners(corners), dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)


def quad_mask(corners: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled binary mask of a quadrilateral, for area IoU."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(corners)], 255)
    return m


# --------------------------------------------------------------------------- #
# binarisation (the readability stage)
# --------------------------------------------------------------------------- #


def binarise_otsu(gray: np.ndarray) -> np.ndarray:
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def binarise_adaptive_mean(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 12
    )


def binarise_adaptive_gaussian(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )


def binarise_sauvola(gray: np.ndarray) -> np.ndarray:
    """Sauvola: a local threshold that adapts to local *contrast*, not just mean.

    Designed for exactly this case — a document under uneven illumination, where
    a single global cut cannot serve both the bright and the shaded half.
    """
    from skimage.filters import threshold_sauvola

    t = threshold_sauvola(gray, window_size=31, k=0.2)
    return ((gray > t).astype(np.uint8)) * 255


def binarise_best_global(gray: np.ndarray, truth_text: np.ndarray) -> tuple[np.ndarray, int]:
    """**Oracle**: the best possible single global threshold, found by exhaustive search.

    This is not a usable method — it needs the ground truth it is being scored
    against. It exists to separate two very different explanations of an Otsu
    failure:

    1. *No global threshold can work here* — the shadowed paper is darker than
       the lit ink, and the two classes genuinely overlap; or
    2. *A global threshold would work, but Otsu picks the wrong one* — Otsu
       maximises between-class variance, which is not the same objective as
       separating ink from paper.

    Without this control, every collapse looks like case 1.
    """
    best_iou, best_t = -1.0, 0
    for t in range(1, 255):
        pred = (gray <= t).astype(np.uint8) * 255  # dark pixels are ink
        score = iou(pred, truth_text)
        if score > best_iou:
            best_iou, best_t = score, t
    return ((gray > best_t).astype(np.uint8) * 255), best_t


BINARISERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Otsu (global)": binarise_otsu,
    "Adaptive mean": binarise_adaptive_mean,
    "Adaptive Gaussian": binarise_adaptive_gaussian,
    "Sauvola": binarise_sauvola,
}


# --------------------------------------------------------------------------- #
# the experiment
# --------------------------------------------------------------------------- #


@dataclass
class DetectorScore:
    """Aggregated performance of one detector over many scenes."""

    name: str
    n_scenes: int
    n_success: int
    corner_errors: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    times_ms: list[float] = field(default_factory=list)

    #: A detection is "usable" if every corner lands within this many pixels.
    #: Beyond it the rectified page is visibly cropped or skewed.
    USABLE_PX = 10.0

    @property
    def success_rate(self) -> float:
        """Fraction of scenes where the detector returned *a* quadrilateral."""
        return self.n_success / self.n_scenes if self.n_scenes else 0.0

    @property
    def usable_rate(self) -> float:
        """Fraction of scenes where the quad it returned was actually right.

        Reported separately from :attr:`success_rate` because the dangerous
        failure is not "no page found" — that is visible and recoverable — but a
        detector confidently returning the wrong quadrilateral.
        """
        if not self.n_scenes:
            return 0.0
        return sum(e <= self.USABLE_PX for e in self.corner_errors) / self.n_scenes

    def summary(self) -> dict[str, float | str | int]:
        ok = bool(self.corner_errors)
        return {
            "method": self.name,
            "success_rate": round(self.success_rate, 4),
            "usable_rate": round(self.usable_rate, 4),
            "success_pct": f"{self.success_rate * 100:.0f}%",
            "usable_pct": f"{self.usable_rate * 100:.0f}%",
            "n_success": self.n_success,
            "n_scenes": self.n_scenes,
            "corner_error_px_mean": round(float(np.mean(self.corner_errors)), 3) if ok else None,
            "corner_error_px_median": round(float(np.median(self.corner_errors)), 3) if ok else None,
            "corner_error_px_p90": round(float(np.percentile(self.corner_errors, 90)), 3) if ok else None,
            "area_iou_mean": round(float(np.mean(self.ious)), 4) if ok else None,
            "median_ms": round(float(np.median(self.times_ms)), 3) if self.times_ms else None,
        }


def evaluate_detectors(
    n_scenes: int = 30, runs: int = 3, warmup: int = 1, illum_min: float = 0.62
) -> list[dict]:
    """Run every detector over ``n_scenes`` scenes with known corners.

    Failures are excluded from the error statistics and reported separately as
    ``success_rate``. Averaging a failure in as "error = 0" would reward a
    detector for giving up, and averaging it as some large number would make the
    figure depend on an arbitrary penalty.
    """
    scores = {name: DetectorScore(name, n_scenes, 0) for name in DETECTORS}

    for seed in range(n_scenes):
        photo, _page, truth = synth.document_scene(seed=seed, illum_min=illum_min)
        truth_mask = quad_mask(truth, photo.shape)

        for name, fn in DETECTORS.items():
            corners, timing = timeit(lambda f=fn: f(photo), runs=runs, warmup=warmup)
            scores[name].times_ms.append(timing.median_ms)
            if corners is None:
                continue
            scores[name].n_success += 1
            scores[name].corner_errors.append(corner_error(corners, truth))
            scores[name].ious.append(iou(quad_mask(corners, photo.shape), truth_mask))

    return [scores[name].summary() for name in DETECTORS]


def text_mask(gray_page: np.ndarray) -> np.ndarray:
    """Ground-truth text pixels of a clean page: anything clearly darker than paper."""
    return (gray_page < 128).astype(np.uint8) * 255


def evaluate_binarisers(n_scenes: int = 30, illum_min: float = 0.62) -> list[dict]:
    """Score each binariser on pages rectified with the *true* corners.

    Rectifying with ground-truth corners is deliberate: it isolates the
    binarisation stage, so a poor score here cannot be blamed on bad detection.

    Scoring is IoU against the true text mask, which turns "readable" into a
    number. The rectified page still carries the illumination gradient from the
    photo, which is what separates a global threshold from a local one.
    """
    results = {name: {"iou": [], "ms": []} for name in BINARISERS}

    for seed in range(n_scenes):
        photo, page, truth = synth.document_scene(seed=seed, illum_min=illum_min)
        page_h, page_w = page.shape[:2]
        truth_text = text_mask(to_gray(page))

        rect = rectify(photo, truth, (page_w, page_h))
        gray = to_gray(rect)

        for name, fn in BINARISERS.items():
            binary, timing = timeit(lambda f=fn: f(gray), runs=3, warmup=1)
            # binarisers return white paper / black text; invert to get text pixels
            results[name]["iou"].append(iou(255 - binary, truth_text))
            results[name]["ms"].append(timing.median_ms)

    return [
        {
            "method": name,
            "text_iou_mean": round(float(np.mean(r["iou"])), 4),
            "text_iou_median": round(float(np.median(r["iou"])), 4),
            "median_ms": round(float(np.median(r["ms"])), 3),
        }
        for name, r in results.items()
    ]


#: Scene-level illumination settings swept by :func:`sweep_illumination`.
#: ``1.0`` is flat studio light; ``0.02`` is a deep shadow across the frame.
ILLUM_LEVELS = (1.0, 0.7, 0.5, 0.35, 0.22, 0.12, 0.06, 0.03, 0.015)

#: Label for the oracle series in the sweep. Not a method — see
#: :func:`binarise_best_global`.
ORACLE_NAME = "Best global (oracle)"

#: Ink is this fraction of paper reflectance in the generated page (45/245).
#: A global threshold can separate them only while the *darkest* paper stays
#: brighter than the *brightest* ink, i.e. while the page ratio exceeds this.
INK_REFLECTANCE = 45.0 / 245.0


def page_illumination_ratio(
    corners: np.ndarray, image_size: tuple[int, int], illum_min: float
) -> float:
    """Illumination ratio **across the page**, darkest corner over brightest.

    This, not the scene-level setting, is the quantity that decides whether a
    global threshold can work: the page occupies only part of the frame, and how
    much of the lighting gradient falls across it depends on where it landed.
    Because the shading is affine, evaluating the four corners is exact.
    """
    vals = synth.shading_at(corners, image_size, illum_min)
    return float(vals.min() / vals.max())


def sweep_illumination(n_scenes: int = 12, levels=ILLUM_LEVELS) -> list[dict]:
    """Re-score every binariser across a range of illumination ratios.

    This is the experiment the single-condition table cannot do. A global
    threshold is not simply "worse" than a local one — it is *optimal* under flat
    light and collapses past some ratio. Sweeping locates that ratio instead of
    asserting it.

    Each row also carries the mean **measured page ratio**, so the x-axis is a
    physical property of the image rather than a knob in the generator.
    """
    rows = []
    for level in levels:
        scored = evaluate_binarisers(n_scenes=n_scenes, illum_min=level)

        ratios, oracle_ious = [], []
        for seed in range(n_scenes):
            photo, page, truth = synth.document_scene(seed=seed, illum_min=level)
            H, W = photo.shape[:2]
            ratios.append(page_illumination_ratio(truth, (W, H), level))

            page_h, page_w = page.shape[:2]
            gray = to_gray(rectify(photo, truth, (page_w, page_h)))
            truth_text = text_mask(to_gray(page))
            best, _t = binarise_best_global(gray, truth_text)
            oracle_ious.append(iou(255 - best, truth_text))

        row: dict[str, float | str] = {
            "illum_min": level,
            "page_ratio": round(float(np.mean(ratios)), 4),
            ORACLE_NAME: round(float(np.mean(oracle_ious)), 4),
        }
        for r in scored:
            row[r["method"]] = r["text_iou_mean"]
        rows.append(row)
    return rows


def evaluate_aspect_recovery(n_scenes: int = 30) -> list[dict]:
    """Compare the two ways of choosing the output aspect ratio.

    The page generator builds a 400x560 page, so the true width/height ratio is
    known exactly. Each estimator is scored by relative error against it.
    """
    _photo, page, _ = synth.document_scene(seed=0)
    true_ratio = page.shape[1] / page.shape[0]

    rows = {"Edge lengths (tutorial method)": [], "Perspective (closed form)": []}
    n_fallback = 0

    for seed in range(n_scenes):
        photo, _page, truth = synth.document_scene(seed=seed)
        rows["Edge lengths (tutorial method)"].append(aspect_from_edges(truth))
        persp = aspect_from_perspective(truth, photo.shape)
        if persp is None:
            n_fallback += 1
        else:
            rows["Perspective (closed form)"].append(persp)

    out = []
    for name, vals in rows.items():
        if not vals:
            out.append({"method": name, "mean_ratio": None, "rel_error_pct": None, "n": 0})
            continue
        errs = [abs(v - true_ratio) / true_ratio * 100 for v in vals]
        out.append(
            {
                "method": name,
                "true_ratio": round(true_ratio, 4),
                "mean_ratio": round(float(np.mean(vals)), 4),
                "rel_error_pct": round(float(np.mean(errs)), 2),
                "rel_error_pct_max": round(float(np.max(errs)), 2),
                "n": len(vals),
            }
        )
    out[1]["n_degenerate_fallback"] = n_fallback
    return out


def scan(
    img: np.ndarray,
    detector: str = "Otsu + contour",
    binariser: str = "Sauvola",
    use_perspective_aspect: bool = True,
):
    """End-to-end scan of a single image, for the UI and for inference.

    Returns ``(corners, rectified_rgb, binary)``; ``corners`` is None when the
    detector could not find a page, in which case nothing downstream is produced.

    The output size is chosen from the **recovered** aspect ratio when the
    closed-form solution is available, falling back to the edge-length heuristic
    for degenerate (near-affine) views.
    """
    corners = DETECTORS[detector](img)
    if corners is None:
        return None, None, None

    diag = float(max(np.linalg.norm(corners[2] - corners[0]), np.linalg.norm(corners[3] - corners[1])))
    ratio = aspect_from_perspective(corners, img.shape) if use_perspective_aspect else None
    if ratio is None:
        ratio = aspect_from_edges(corners)
    ratio = float(np.clip(ratio, 0.1, 10.0))

    # keep the diagonal roughly constant so the output resolution is stable
    h = int(round(diag / np.sqrt(1.0 + ratio**2)))
    w = int(round(h * ratio))
    w, h = max(w, 32), max(h, 32)

    rect = rectify(img, corners, (w, h))
    return corners, rect, BINARISERS[binariser](to_gray(rect))
