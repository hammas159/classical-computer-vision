"""Lane detection: six pipelines, two controls, and a truth that needs no annotation.

The question
------------
A forward-facing camera on a car. Find the two painted lines bounding the lane it
is in. The pipeline in every tutorial is the same four steps -- mask by colour,
detect edges, crop to a region of interest, fit lines with a Hough transform --
and the interesting question is which of those four is doing the work.

> **The claim under test:** the region of interest is a preprocessing
> convenience. **It is the algorithm.** Replacing everything inside it -- colour,
> Canny, Hough -- with "fit a line to the bright pixels in each half of the
> trapezoid" loses almost nothing, because the trapezoid already says where the
> lane is. The ROI was drawn by a person who knew the answer.

> **And the ROI is where the pipeline breaks when nothing else changes.** These
> fourteen photographs come from two cameras at two resolutions. A region of
> interest written in pixels -- which is how every tutorial writes it -- is
> correct on one and wrong on the other, and the failure is silent: lines are
> still returned, they are just the wrong lines.

Where the ground truth comes from
---------------------------------
There is no annotation for these photographs and **none is invented**. What there
is instead is geometry that any correct answer must satisfy, and that no method
is told about:

* the two lane lines of a straight road **converge at the vanishing point**, and
  the vanishing point is a property of *where the camera is bolted*, not of the
  frame. Across frames from one camera it must not move.
* the lane has a **width at the car**, and that is a property of the road and the
  camera together. It must not move either.
* a lane line is not horizontal and not vertical, and the left one leans the
  opposite way to the right one.

A method that locks onto a guard rail, a shadow edge or the concrete seam of a
bridge satisfies none of these, and it fails them without anyone drawing a line
on a photograph.

The two controls
----------------
`Bright pixels in the ROI` skips every step except the region of interest.
`Fixed guess` returns the same two lines for every frame -- the average lane, with
the image never looked at. They say what the ROI is worth and what the whole
pipeline is worth over guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from shared.io import to_gray

EPS = 1e-9

#: Fourteen dashcam photographs of real roads, cached by
#: `tools/fetch_assets.py --set lanes`. Six are 960x540 and eight are 1280x720 --
#: two cameras, which is the point rather than an inconvenience.
LANES = Path.home() / ".cache" / "classical-cv-images" / "assets" / "lanes"

#: The region of interest, as **fractions of the frame**: a trapezoid from the
#: bottom corners up to a narrow top near the horizon.
#:
#: Written in fractions on purpose. Every tutorial writes it in pixels, which
#: silently means "for this camera at this resolution", and `roi_in_pixels`
#: measures what that costs on the eight frames it was not written for.
ROI = ((0.10, 1.00), (0.45, 0.60), (0.55, 0.60), (0.95, 1.00))

#: The same trapezoid as the pixel literal a tutorial would write for a 960x540
#: frame. Kept so the failure can be reproduced rather than described.
ROI_PIXELS_960 = ((96, 540), (432, 324), (528, 324), (912, 540))

#: A lane line is never horizontal and never vertical. Hough segments outside
#: this band are road markings across the lane, shadows, or the horizon.
MIN_SLOPE = 0.5
MAX_SLOPE = 3.0


def available() -> bool:
    return LANES.exists() and len(list(LANES.glob("*.jpg"))) >= 12


def image_names() -> list[str]:
    return sorted(p.stem for p in LANES.glob("*.jpg"))


def load(name: str) -> np.ndarray:
    path = LANES / f"{name}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"{path} is missing. Run `python tools/fetch_assets.py --set lanes`.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def camera_of(name: str) -> str:
    """Which of the two cameras took this frame.

    The vanishing point may only be compared within a group, because it is a
    property of how the camera is mounted.
    """
    h, w = load(name).shape[:2]
    return f"{w}x{h}"


def roi_mask(shape, roi=ROI) -> np.ndarray:
    """The trapezoid, as a mask, from fractional coordinates."""
    h, w = shape[:2]
    pts = np.array([[int(x * w), int(y * h) - 1] for x, y in roi], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def roi_mask_pixels(shape, roi=ROI_PIXELS_960) -> np.ndarray:
    """The same trapezoid as a pixel literal, applied to whatever frame it is given.

    This is what a tutorial's `vertices = np.array([[(96, 540), ...]])` does on a
    frame it was not written for. It is not corrected; it is measured.
    """
    h, w = shape[:2]
    pts = np.array([[int(x), int(y)] for x, y in roi], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #


def colour_mask_hls(image: np.ndarray) -> np.ndarray:
    """White and yellow paint, in HLS.

    Lane paint is defined by being white or yellow, and HLS separates those from
    grey asphalt better than RGB does: white is high L at any H, yellow is a
    narrow H at moderate S.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white = cv2.inRange(hls, (0, 195, 0), (180, 255, 255))
    yellow = cv2.inRange(hls, (12, 80, 90), (32, 210, 255))
    return cv2.bitwise_or(white, yellow)


def colour_mask_lab(image: np.ndarray) -> np.ndarray:
    """The same two colours in Lab: L for white, b for yellow.

    The b channel of Lab is blue-to-yellow by construction, so yellow paint on
    grey asphalt is a single-channel threshold rather than a range in a hue
    wheel that wraps.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    white = cv2.inRange(lab[..., 0], 200, 255)
    yellow = cv2.inRange(lab[..., 2], 150, 255)
    return cv2.bitwise_or(white, yellow)


def edges_canny(image: np.ndarray, blur: int = 5) -> np.ndarray:
    g = cv2.GaussianBlur(to_gray(image), (blur, blur), 0)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(g, 0.5 * t, t)


def hough_segments(mask: np.ndarray, threshold: int = 30,
                   min_length: int = 25, max_gap: int = 120):
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold,
                            minLineLength=min_length, maxLineGap=max_gap)
    return [] if lines is None else [tuple(int(v) for v in l[0]) for l in lines]


def _fit_side(points: np.ndarray, shape, top: float, bottom: float = 1.0):
    """Least-squares line through points, returned as two endpoints.

    Fitted as ``x = m*y + c`` rather than ``y = m*x + c``: a lane line is closer
    to vertical than to horizontal, and the usual form is ill-conditioned exactly
    where the lane is straightest.
    """
    if len(points) < 2:
        return None
    h, w = shape[:2]
    ys = points[:, 1].astype(np.float64)
    xs = points[:, 0].astype(np.float64)
    if ys.max() - ys.min() < 5:
        return None
    m, c = np.polyfit(ys, xs, 1)
    y0, y1 = top * h, bottom * h
    return (float(m * y0 + c), float(y0), float(m * y1 + c), float(y1))


def segments_to_lanes(segments, shape, top: float = 0.62):
    """Split Hough segments by slope sign and fit one line to each side."""
    h, w = shape[:2]
    left, right = [], []
    for x1, y1, x2, y2 in segments:
        if abs(x2 - x1) < EPS:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (MIN_SLOPE <= abs(slope) <= MAX_SLOPE):
            continue
        target = left if slope < 0 else right
        target.extend([(x1, y1), (x2, y2)])
    lanes = []
    for side in (left, right):
        fitted = _fit_side(np.array(side), shape, top) if side else None
        lanes.append(fitted)
    return tuple(lanes)


# --------------------------------------------------------------------------- #
# the pipelines
# --------------------------------------------------------------------------- #


def detect_canny_hough(image, roi=ROI, **kw):
    """Grayscale, Canny, ROI, Hough. The pipeline with no colour step."""
    mask = cv2.bitwise_and(edges_canny(image), roi_mask(image.shape, roi))
    return segments_to_lanes(hough_segments(mask), image.shape)


def detect_colour_canny_hough(image, roi=ROI, **kw):
    """HLS colour mask, then Canny on it, then ROI and Hough. The full tutorial."""
    colour = colour_mask_hls(image)
    edges = cv2.Canny(cv2.GaussianBlur(colour, (5, 5), 0), 50, 150)
    mask = cv2.bitwise_and(edges, roi_mask(image.shape, roi))
    return segments_to_lanes(hough_segments(mask), image.shape)


def detect_lab_canny_hough(image, roi=ROI, **kw):
    """The same with the Lab colour mask instead of HLS."""
    colour = colour_mask_lab(image)
    edges = cv2.Canny(cv2.GaussianBlur(colour, (5, 5), 0), 50, 150)
    mask = cv2.bitwise_and(edges, roi_mask(image.shape, roi))
    return segments_to_lanes(hough_segments(mask), image.shape)


def detect_colour_hough(image, roi=ROI, **kw):
    """Colour mask straight into Hough, with no edge detector at all.

    Worth having because paint is a *region*, not an edge, and a Hough transform
    will happily vote on a filled stripe.
    """
    mask = cv2.bitwise_and(colour_mask_hls(image), roi_mask(image.shape, roi))
    return segments_to_lanes(hough_segments(mask), image.shape)


def detect_sobel_x(image, roi=ROI, **kw):
    """Threshold the horizontal gradient only.

    A lane line is a near-vertical stripe, so its gradient is horizontal. This
    is the cheapest way to say that, and it says it more directly than Canny,
    which has no orientation preference at all.
    """
    g = to_gray(image).astype(np.float32)
    dx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5))
    dx = (dx / max(dx.max(), EPS) * 255).astype(np.uint8)
    strong = cv2.inRange(dx, 40, 255)
    mask = cv2.bitwise_and(strong, roi_mask(image.shape, roi))
    return segments_to_lanes(hough_segments(mask), image.shape)


def detect_bright_in_roi(image, roi=ROI, **kw):
    """**The control.** No colour test, no edges, no Hough.

    Take the brightest pixels inside the trapezoid, split them down the middle,
    and fit a line to each half. Everything this knows about lanes is in the
    shape of the region of interest.
    """
    h, w = image.shape[:2]
    region = roi_mask(image.shape, roi)
    g = to_gray(image)
    inside = g[region > 0]
    if inside.size == 0:
        return (None, None)
    cut = float(np.percentile(inside, 96))
    bright = ((g >= cut) & (region > 0))

    ys, xs = np.nonzero(bright)
    if len(xs) < 4:
        return (None, None)
    middle = w / 2.0
    lanes = []
    for sel in (xs < middle, xs >= middle):
        pts = np.stack([xs[sel], ys[sel]], axis=1)
        lanes.append(_fit_side(pts, image.shape, 0.62) if len(pts) >= 2 else None)
    return tuple(lanes)


#: The average lane over the fourteen frames, in fractions of the frame. Fitted
#: once from the ROI control's own output and then frozen, so this really is a
#: guess that never looks at the image.
FIXED_LANE = ((0.435, 0.62, 0.175, 1.00), (0.565, 0.62, 0.885, 1.00))


def detect_fixed(image, roi=ROI, **kw):
    """**The other control.** The same two lines for every frame.

    A lane detector that never looks at the picture. Its score is what any of
    the others has to beat to have done anything at all.
    """
    h, w = image.shape[:2]
    return tuple((x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1 in FIXED_LANE)


DETECTORS: dict[str, Callable] = {
    "Fixed guess (control)": detect_fixed,
    "Bright pixels in the ROI (control)": detect_bright_in_roi,
    "Canny + Hough": detect_canny_hough,
    "HLS colour + Canny + Hough": detect_colour_canny_hough,
    "Lab colour + Canny + Hough": detect_lab_canny_hough,
    "HLS colour + Hough": detect_colour_hough,
    "Sobel-x + Hough": detect_sobel_x,
}

REAL_DETECTORS = tuple(d for d in DETECTORS if "control" not in d)


# --------------------------------------------------------------------------- #
# geometry: what a correct answer has to satisfy
# --------------------------------------------------------------------------- #


def vanishing_point(lanes):
    """Where the two fitted lines meet.

    For a straight road this is the vanishing point of the road direction, and
    it depends on **how the camera is mounted**, not on the frame. It is the
    single most useful annotation-free check available here: across frames from
    one camera it must not move.
    """
    left, right = lanes
    if left is None or right is None:
        return None
    (x1, y1, x2, y2), (x3, y3, x4, y4) = left, right
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    den = d1x * d2y - d1y * d2x
    if abs(den) < EPS:
        return None
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / den
    return (x1 + t * d1x, y1 + t * d1y)


def lane_width(lanes, shape):
    """Distance between the two lines at the bottom of the frame, as a fraction
    of the frame width.

    A property of the road and the camera together, so it is also constant
    within a camera group -- and expressing it as a fraction is what makes the
    two resolutions comparable at all.
    """
    left, right = lanes
    if left is None or right is None:
        return None
    return abs(right[2] - left[2]) / shape[1]


def plausible(lanes, shape) -> bool:
    """Does this answer satisfy what any lane pair must satisfy?

    Both lines present, leaning towards each other, meeting above the middle of
    the frame and not off the side of it, and a lane between a fifth and the
    whole of the frame wide. Nothing here needs an annotation; it is all
    geometry that follows from a camera pointing down a road.
    """
    left, right = lanes
    if left is None or right is None:
        return False
    h, w = shape[:2]
    if not (left[2] < left[0] and right[2] > right[0]):
        return False  # the lines must lean towards each other going up the frame
    vp = vanishing_point(lanes)
    if vp is None:
        return False
    if not (0.15 * w < vp[0] < 0.85 * w):
        return False
    if not (0.35 * h < vp[1] < 0.75 * h):
        return False
    width = lane_width(lanes, shape)
    return width is not None and 0.20 <= width <= 1.10


def _spread(values) -> float:
    a = np.asarray([v for v in values if v is not None], float)
    return float("nan") if len(a) < 2 else float(a.std())


def consistency(detector: str, names=None) -> dict:
    """Per camera: how far the vanishing point and the lane width wander.

    This is the score. It uses **no annotation at all** -- only the fact that a
    camera bolted to a car cannot have a vanishing point that jumps between
    frames -- and it punishes exactly the failure that matters: locking onto a
    guard rail, a shadow edge, or the seam of a bridge.
    """
    names = names or image_names()
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(camera_of(n), []).append(n)

    out = {"detector": detector, "cameras": {}}
    fn = DETECTORS[detector]
    all_ok = 0
    total = 0
    for camera, members in sorted(groups.items()):
        vps_x, vps_y, widths, ok = [], [], [], 0
        for n in members:
            img = load(n)
            lanes = fn(img)
            total += 1
            if plausible(lanes, img.shape):
                ok += 1
                all_ok += 1
                vp = vanishing_point(lanes)
                vps_x.append(vp[0])
                vps_y.append(vp[1])
                widths.append(lane_width(lanes, img.shape))
        out["cameras"][camera] = {
            "frames": len(members),
            "plausible": ok,
            "vp_x_spread": _spread(vps_x),
            "vp_y_spread": _spread(vps_y),
            "width_spread": _spread(widths),
        }
    out["plausible_rate"] = all_ok / max(total, 1)
    spreads = [c["vp_x_spread"] for c in out["cameras"].values()]
    out["vp_x_spread"] = float(np.nanmean(spreads)) if spreads else float("nan")
    widths = [c["width_spread"] for c in out["cameras"].values()]
    out["width_spread"] = float(np.nanmean(widths)) if widths else float("nan")
    return out


def evaluate_all(names=None) -> list[dict]:
    return [consistency(d, names) for d in DETECTORS]


# --------------------------------------------------------------------------- #
# the ROI written in pixels
# --------------------------------------------------------------------------- #


def detect_pixel_roi(image, **kw):
    """The full pipeline with the region of interest as a **pixel literal**.

    Identical to `detect_colour_canny_hough` in every respect except that the
    trapezoid is the one a tutorial hard-codes for a 960x540 frame. On a
    960x540 frame it is the same trapezoid. On a 1280x720 frame it is a
    trapezoid over the bottom-left quadrant, and the detector still returns two
    lines rather than complaining.
    """
    colour = colour_mask_hls(image)
    edges = cv2.Canny(cv2.GaussianBlur(colour, (5, 5), 0), 50, 150)
    mask = cv2.bitwise_and(edges, roi_mask_pixels(image.shape))
    return segments_to_lanes(hough_segments(mask), image.shape)


def roi_in_pixels() -> list[dict]:
    """What the pixel literal costs, per camera.

    The comparison is against the identical pipeline with the identical
    trapezoid expressed in fractions, so the **only** difference between the two
    columns is how the region was written down.
    """
    rows = []
    for camera in sorted({camera_of(n) for n in image_names()}):
        members = [n for n in image_names() if camera_of(n) == camera]
        frac_ok = px_ok = 0
        frac_vp, px_vp = [], []
        for n in members:
            img = load(n)
            for fn, is_frac, vps in ((detect_colour_canny_hough, True, frac_vp),
                                     (detect_pixel_roi, False, px_vp)):
                lanes = fn(img)
                if plausible(lanes, img.shape):
                    vps.append(vanishing_point(lanes)[0])
                    if is_frac:
                        frac_ok += 1
                    else:
                        px_ok += 1
        shape = load(members[0]).shape
        rows.append({
            "camera": camera,
            "frames": len(members),
            "fractional_plausible": frac_ok,
            "pixel_plausible": px_ok,
            "fractional_vp_spread": _spread(frac_vp),
            "pixel_vp_spread": _spread(px_vp),
            "pixel_roi_covers": float((roi_mask_pixels(shape) > 0).mean()),
            "fractional_roi_covers": float((roi_mask(shape) > 0).mean()),
        })
    return rows


# --------------------------------------------------------------------------- #
# how much of the answer is the ROI
# --------------------------------------------------------------------------- #


def bottom_endpoints(lanes):
    left, right = lanes
    return (None if left is None else left[2], None if right is None else right[2])


def distance_to_control(detector: str, names=None) -> dict:
    """How far each pipeline's answer sits from the ROI control's.

    Measured at the bottom of the frame, in **percent of frame width**, so the
    two resolutions can be averaged. This is the number behind the claim that
    the region of interest is doing the work: if a five-step pipeline lands
    within a percent of "the bright pixels in the trapezoid", the four steps
    after the trapezoid are decoration.
    """
    names = names or image_names()
    gaps = []
    for n in names:
        img = load(n)
        w = img.shape[1]
        a = detect_bright_in_roi(img)
        b = DETECTORS[detector](img)
        for pa, pb in zip(bottom_endpoints(a), bottom_endpoints(b)):
            if pa is not None and pb is not None:
                gaps.append(abs(pa - pb) / w * 100.0)
    return {
        "detector": detector,
        "median_gap_pct": float(np.median(gaps)) if gaps else float("nan"),
        "max_gap_pct": float(np.max(gaps)) if gaps else float("nan"),
        "n": len(gaps),
    }


def control_gap_by_camera(detector: str = "HLS colour + Canny + Hough") -> list[dict]:
    """The same gap, split by which camera took the frame.

    Worth splitting because the answer is not the same on both, and averaging
    the two hides the only interesting thing about it: the trapezoid is written
    in fractions of the frame, but the horizon is not at the same fraction of
    the frame on both cameras, and the control has nothing else to fall back on.
    """
    rows = []
    for camera in sorted({camera_of(n) for n in image_names()}):
        members = [n for n in image_names() if camera_of(n) == camera]
        gaps = []
        for n in members:
            img = load(n)
            w = img.shape[1]
            a = detect_bright_in_roi(img)
            b = DETECTORS[detector](img)
            for pa, pb in zip(bottom_endpoints(a), bottom_endpoints(b)):
                if pa is not None and pb is not None:
                    gaps.append(abs(pa - pb) / w * 100.0)
        rows.append({
            "camera": camera,
            "frames": len(members),
            "median_gap_pct": float(np.median(gaps)),
            "max_gap_pct": float(np.max(gaps)),
            "bright_is_paint": float(np.mean([bright_is_paint(n) for n in members])),
        })
    return rows


def bright_is_paint(name: str) -> float:
    """Of the pixels the ROI control picks, what share are actually lane-coloured.

    The control has no way to tell paint from any other bright thing inside the
    trapezoid, so this is the quantity it is blind to -- and it is measured with
    the colour mask the control does not run.

    It is a **partial** explanation of where the control and the pipeline part
    company, not a complete one: across the fourteen frames it correlates with
    the gap at r = -0.54. Reported as the partial explanation it is.
    """
    img = load(name)
    g = to_gray(img)
    region = roi_mask(img.shape) > 0
    cut = np.percentile(g[region], 96)
    bright = (g >= cut) & region
    paint = colour_mask_hls(img) > 0
    return float((bright & paint).sum() / max(bright.sum(), 1))


# --------------------------------------------------------------------------- #
# a truth that is recorded rather than inferred
# --------------------------------------------------------------------------- #


def yaw_homography(shape, shift_px: float):
    """A small camera yaw: the horizon slides sideways, the bottom edge does not.

    This is a **recorded** transform -- it is applied here, so it is known
    exactly -- and it is the only thing that changes between the two frames the
    repeatability test compares. A detector following the paint has to return a
    line that maps through this homography.
    """
    h, w = shape[:2]
    src = np.float32([[0, h - 1], [w - 1, h - 1], [0, 0.55 * h], [w - 1, 0.55 * h]])
    dst = np.float32([[0, h - 1], [w - 1, h - 1],
                      [shift_px, 0.55 * h], [w - 1 + shift_px, 0.55 * h]])
    return cv2.getPerspectiveTransform(src, dst)


def _apply(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def yaw_repeatability(detector: str, shift_px: float = 40.0, names=None) -> dict:
    """Warp by a known yaw, detect again, map the first answer forward, compare.

    The error is taken at the top of the region of interest -- where the yaw has
    actually moved things -- in percent of frame width. Zero would mean the
    detector tracked the paint through the warp exactly.
    """
    names = names or image_names()
    fn = DETECTORS[detector]
    errors = []
    for n in names:
        img = load(n)
        h, w = img.shape[:2]
        H = yaw_homography(img.shape, shift_px)
        warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        before, after = fn(img), fn(warped)
        for a, b in zip(before, after):
            if a is None or b is None:
                continue
            ax, _ = _apply(H, a[0], a[1])
            errors.append(abs(ax - b[0]) / w * 100.0)
    return {
        "detector": detector,
        "shift_px": shift_px,
        "median_error_pct": float(np.median(errors)) if errors else float("nan"),
        "n": len(errors),
    }


# --------------------------------------------------------------------------- #
# ablation: which of the four steps matters
# --------------------------------------------------------------------------- #


def no_roi(image, **kw):
    """The full pipeline with the region of interest removed entirely."""
    colour = colour_mask_hls(image)
    edges = cv2.Canny(cv2.GaussianBlur(colour, (5, 5), 0), 50, 150)
    return segments_to_lanes(hough_segments(edges), image.shape)


def no_slope_filter(image, **kw):
    """The full pipeline without the slope band, keeping every Hough segment."""
    colour = colour_mask_hls(image)
    edges = cv2.Canny(cv2.GaussianBlur(colour, (5, 5), 0), 50, 150)
    mask = cv2.bitwise_and(edges, roi_mask(image.shape))
    left, right = [], []
    for x1, y1, x2, y2 in hough_segments(mask):
        if abs(x2 - x1) < EPS:
            continue
        slope = (y2 - y1) / (x2 - x1)
        (left if slope < 0 else right).extend([(x1, y1), (x2, y2)])
    return tuple(_fit_side(np.array(s), image.shape, 0.62) if s else None
                 for s in (left, right))


ABLATIONS = {
    "Full pipeline": detect_colour_canny_hough,
    "without the colour mask": detect_canny_hough,
    "without the edge detector": detect_colour_hough,
    "without the slope filter": no_slope_filter,
    "without the region of interest": no_roi,
}


def ablate(names=None) -> list[dict]:
    """Remove one step at a time and score the result on the same geometry."""
    names = names or image_names()
    rows = []
    for label, fn in ABLATIONS.items():
        ok, vps, widths = 0, [], []
        for n in names:
            img = load(n)
            lanes = fn(img)
            if plausible(lanes, img.shape):
                ok += 1
                vps.append(vanishing_point(lanes)[0] / img.shape[1])
                widths.append(lane_width(lanes, img.shape))
        rows.append({
            "variant": label,
            "plausible": ok,
            "frames": len(names),
            "vp_x_spread_pct": 100 * _spread(vps),
            "width_spread": _spread(widths),
        })
    return rows


# --------------------------------------------------------------------------- #
# what makes a frame hard
# --------------------------------------------------------------------------- #


def paint_contrast(name: str) -> float:
    """How far the paint stands out from the road, in grey levels.

    Measured inside the region of interest as the gap between the 98th
    percentile (paint) and the median (asphalt). It is computed from the image
    alone, before any detector runs, so it can be used to predict difficulty
    rather than to explain it afterwards.
    """
    img = load(name)
    g = to_gray(img)
    inside = g[roi_mask(img.shape) > 0].astype(np.float32)
    return float(np.percentile(inside, 98) - np.median(inside))


def shadow_fraction(name: str) -> float:
    """Share of the region of interest darker than 60% of its median.

    Tree shadow across a lane is the classic failure, and this is the cheapest
    way to say how much of it there is.
    """
    img = load(name)
    g = to_gray(img)
    inside = g[roi_mask(img.shape) > 0].astype(np.float32)
    return float((inside < 0.6 * np.median(inside)).mean())


def per_image(names=None) -> list[dict]:
    """One row per photograph: its difficulty, and how far the detectors spread."""
    names = names or image_names()
    rows = []
    for n in names:
        img = load(n)
        w = img.shape[1]
        agree = []
        ok = 0
        for d in REAL_DETECTORS:
            lanes = DETECTORS[d](img)
            if plausible(lanes, img.shape):
                ok += 1
                agree.append(bottom_endpoints(lanes))
        lefts = [a[0] for a in agree if a[0] is not None]
        rights = [a[1] for a in agree if a[1] is not None]
        spread = 0.0
        if len(lefts) > 1 and len(rights) > 1:
            spread = float((np.ptp(lefts) + np.ptp(rights)) / 2 / w * 100)
        rows.append({
            "image": n,
            "camera": camera_of(n),
            "paint_contrast": paint_contrast(n),
            "shadow_fraction": shadow_fraction(n),
            "bright_is_paint": bright_is_paint(n),
            "plausible": ok,
            "detectors": len(REAL_DETECTORS),
            "disagreement_pct": spread,
        })
    return rows


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def draw(image, lanes, colour=(60, 220, 90), thickness=None):
    out = image.copy()
    h, w = image.shape[:2]
    thickness = thickness or max(3, w // 240)
    for line in lanes:
        if line is None:
            continue
        x0, y0, x1, y1 = line
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), colour, thickness)
    vp = vanishing_point(lanes)
    if vp is not None and 0 <= vp[0] < w and 0 <= vp[1] < h:
        cv2.circle(out, (int(vp[0]), int(vp[1])), max(5, w // 160), (250, 190, 40), -1)
    return out


def draw_roi(image, roi=ROI, pixels=False):
    out = image.copy()
    mask = roi_mask_pixels(image.shape) if pixels else roi_mask(image.shape, roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (240, 120, 60), max(2, image.shape[1] // 300))
    return out
