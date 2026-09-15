"""Synthetic degradations with **exact** ground truth.

Why this module exists
----------------------
Almost every project in this repo needs a "before" and a known-correct "after".
Downloading a benchmark gives you someone else's ground truth; generating the
degradation yourself gives you a *perfect* one. Darken an image and you know the
true brightness. Add haze and you know the true clear image. Paste a region and
you know exactly which pixels were forged.

That is why 41 of the 58 candidate projects need no download at all.

Every function here
  * takes and returns **RGB uint8** (see :mod:`shared.io`),
  * accepts ``seed`` so a result is reproducible,
  * returns the *parameters it used* alongside the degraded image, because those
    parameters are the ground truth the project is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import to_float, to_uint8


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


def gaussian_noise(img: np.ndarray, sigma: float = 25.0, seed: int | None = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise. ``sigma`` is in 0-255 units.

    Note the arithmetic is done in float and clipped on the way back. Doing
    ``uint8_img + noise`` in numpy **wraps around**: 250 + 10 becomes 4, turning
    a highlight into a black speck with no warning.
    """
    f = to_float(img)
    noisy = f + _rng(seed).normal(0.0, sigma / 255.0, f.shape)
    return to_uint8(noisy)


def salt_pepper_noise(
    img: np.ndarray, density: float = 0.05, seed: int | None = 0
) -> np.ndarray:
    """Set a fraction ``density`` of pixels to pure black or pure white.

    Applied per-pixel (not per-channel) so the corruption looks like real
    impulse noise rather than colour confetti.
    """
    out = img.copy()
    r = _rng(seed).random(img.shape[:2])
    out[r < density / 2] = 0
    out[r > 1.0 - density / 2] = 255
    return out


def poisson_noise(img: np.ndarray, lam: float = 30.0, seed: int | None = 0) -> np.ndarray:
    """Shot noise: ``poisson(img * lam) / lam``. Lower ``lam`` is noisier.

    This is the physically correct model for photon-limited imaging, and unlike
    Gaussian noise its magnitude depends on the signal.
    """
    f = to_float(img)
    return to_uint8(_rng(seed).poisson(f * lam) / lam)


# --------------------------------------------------------------------------- #
# blur (the kernel is the ground truth)
# --------------------------------------------------------------------------- #


def motion_blur_kernel(length: int = 15, angle_deg: float = 30.0) -> np.ndarray:
    """A normalised line point-spread function of ``length`` px at ``angle_deg``."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    total = k.sum()
    return k / total if total > 0 else k


def defocus_kernel(radius: int = 7) -> np.ndarray:
    """A normalised filled-disc PSF — the classic out-of-focus model."""
    size = radius * 2 + 1
    yy, xx = np.mgrid[:size, :size]
    k = ((xx - radius) ** 2 + (yy - radius) ** 2 <= radius**2).astype(np.float32)
    return k / k.sum()


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``img`` with ``kernel``.

    ``cv2.filter2D`` computes *correlation*, not convolution — it does not flip
    the kernel. For the symmetric PSFs above that makes no difference, but for an
    asymmetric kernel it silently blurs in the mirrored direction, so the flip is
    done explicitly here and the function is honest about its name.
    """
    flipped = cv2.flip(kernel, -1)
    return to_uint8(cv2.filter2D(to_float(img), -1, flipped, borderType=cv2.BORDER_REFLECT))


# --------------------------------------------------------------------------- #
# photometric degradations
# --------------------------------------------------------------------------- #


def low_light(
    img: np.ndarray, gamma: float = 3.0, noise_sigma: float = 4.0, seed: int | None = 0
) -> np.ndarray:
    """Darken by ``(I/255) ** gamma`` and add read noise.

    ``gamma > 1`` darkens. Real low-light images are noisy as well as dark, and
    an enhancement method that ignores the noise will amplify it — which is the
    whole point of the low-light project.
    """
    f = to_float(img)
    dark = np.power(f, gamma)
    if noise_sigma > 0:
        dark = dark + _rng(seed).normal(0.0, noise_sigma / 255.0, dark.shape)
    return to_uint8(dark)


def add_haze(
    img: np.ndarray,
    beta: float = 1.4,
    airlight: float = 0.88,
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the atmospheric scattering model ``I = J*t + A*(1-t)``.

    ``t = exp(-beta * d)`` is the transmission. Returns ``(hazy, transmission)``;
    the caller already holds ``J`` (the input), so between them the ground truth
    is complete.

    If ``depth`` is None a smooth vertical gradient is used with the **far plane
    at the top** of the frame, which is how an outdoor scene actually recedes:
    the horizon is distant and hazy, the foreground at your feet is close and
    clear. Getting this upside down puts the densest haze on the nearest object,
    which no dehazing method can then explain.

    Pass a constant ``depth`` array for uniform haze, where the contrast
    reduction is exactly ``t``.
    """
    f = to_float(img)
    h, w = f.shape[:2]
    if depth is None:
        depth = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    t = np.exp(-beta * depth).astype(np.float32)
    t3 = t[..., None] if f.ndim == 3 else t
    hazy = f * t3 + airlight * (1.0 - t3)
    return to_uint8(hazy), t


def colour_cast(
    img: np.ndarray, gains: tuple[float, float, float] = (1.25, 1.0, 0.75)
) -> np.ndarray:
    """Multiply the R, G, B channels by known gains — ground truth for white balance."""
    f = to_float(img)
    return to_uint8(f * np.asarray(gains, np.float32))


# --------------------------------------------------------------------------- #
# geometric degradations
# --------------------------------------------------------------------------- #


def random_homography(
    shape: tuple[int, int], jitter: float = 0.08, seed: int | None = 0
) -> np.ndarray:
    """A 3x3 homography built by perturbing the four image corners.

    ``jitter`` is a fraction of the image size. Corner perturbation is used
    rather than a random matrix because it is guaranteed non-degenerate: four
    random *matrix* entries can easily produce a collinear configuration, and a
    collinear homography fails **silently**.
    """
    h, w = shape[:2]
    rng = _rng(seed)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    offs = rng.uniform(-jitter, jitter, (4, 2)) * np.float32([w, h])
    return cv2.getPerspectiveTransform(src, (src + offs).astype(np.float32))


def warp_by_homography(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp ``img`` by ``H``, keeping the original canvas size."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)


def synthetic_flow(
    shape: tuple[int, int], magnitude: float = 6.0, kind: str = "sinusoidal"
) -> np.ndarray:
    """A smooth dense flow field, shape ``(H, W, 2)`` holding ``(dx, dy)``."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "sinusoidal":
        dx = magnitude * np.sin(2 * np.pi * yy / h)
        dy = magnitude * np.cos(2 * np.pi * xx / w)
    elif kind == "translation":
        dx = np.full((h, w), magnitude, np.float32)
        dy = np.full((h, w), magnitude * 0.5, np.float32)
    else:
        raise ValueError(f"unknown flow kind {kind!r}")
    return np.stack([dx, dy], axis=-1).astype(np.float32)


def warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp ``img`` by a dense flow field via ``cv2.remap``."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        img,
        (xx + flow[..., 0]).astype(np.float32),
        (yy + flow[..., 1]).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def downsample_for_sr(img: np.ndarray, scale: int = 4, blur_sigma: float | None = None):
    """Blur **then** decimate — the correct low-resolution degradation.

    This is not pedantry. ``cv2.resize(..., INTER_AREA)`` averages over the whole
    receptive field and is a *different* operator from anti-alias-blur followed by
    point sampling. Using resize makes a super-resolution comparison measure how
    well each method inverts ``INTER_AREA``, which is not the question anyone
    means to ask.
    """
    if blur_sigma is None:
        blur_sigma = 0.5 * scale
    blurred = cv2.GaussianBlur(img, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
    return blurred[::scale, ::scale].copy()


# --------------------------------------------------------------------------- #
# damage and forgery (the mask is the ground truth)
# --------------------------------------------------------------------------- #


@dataclass
class Forgery:
    """A copy-move forgery and the mask of exactly which pixels were pasted."""

    image: np.ndarray
    mask: np.ndarray
    src_xy: tuple[int, int]
    dst_xy: tuple[int, int]
    size: int
    angle_deg: float
    scale: float


def copy_move_forgery(
    img: np.ndarray,
    size: int = 96,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    seed: int | None = 0,
) -> Forgery:
    """Copy a square patch and paste it elsewhere, optionally rotated/scaled.

    The returned mask marks the pasted pixels exactly, which makes detection
    scoreable with IoU instead of eyeballed.
    """
    rng = _rng(seed)
    h, w = img.shape[:2]
    out = img.copy()

    sx = int(rng.integers(0, max(1, w - size)))
    sy = int(rng.integers(0, max(1, h - size)))
    patch = img[sy : sy + size, sx : sx + size].copy()

    if angle_deg or scale != 1.0:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, scale)
        patch = cv2.warpAffine(patch, m, (size, size), borderMode=cv2.BORDER_REFLECT)

    for _ in range(64):  # find a destination that does not overlap the source
        dx = int(rng.integers(0, max(1, w - size)))
        dy = int(rng.integers(0, max(1, h - size)))
        if abs(dx - sx) > size or abs(dy - sy) > size:
            break

    out[dy : dy + size, dx : dx + size] = patch
    mask = np.zeros((h, w), np.uint8)
    mask[dy : dy + size, dx : dx + size] = 255
    return Forgery(out, mask, (sx, sy), (dx, dy), size, angle_deg, scale)


def add_scratches(
    img: np.ndarray,
    n_strokes: int = 14,
    thickness: int = 3,
    blotches: int = 6,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random white strokes and blotches, as on a damaged print.

    Returns ``(damaged, mask)``. The mask is the exact inpainting target, so a
    restoration can be scored against the untouched original rather than judged
    by eye.
    """
    rng = _rng(seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    for _ in range(n_strokes):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        pts = [(x, y)]
        for _ in range(int(rng.integers(3, 7))):  # a wandering polyline, not a straight line
            x = int(np.clip(x + rng.integers(-w // 6, w // 6), 0, w - 1))
            y = int(np.clip(y + rng.integers(-h // 8, h // 8), 0, h - 1))
            pts.append((x, y))
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, thickness)

    for _ in range(blotches):
        cv2.circle(
            mask,
            (int(rng.integers(0, w)), int(rng.integers(0, h))),
            int(rng.integers(4, 12)),
            255,
            -1,
        )

    damaged = img.copy()
    damaged[mask > 0] = 255
    return damaged, mask


# --------------------------------------------------------------------------- #
# synthetic scenes
# --------------------------------------------------------------------------- #


def shapes(size: int = 512, seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic scene of solid shapes plus its **exact** edge map.

    Returned edges are the true object boundaries, so edge detectors can be
    scored with precision/recall rather than compared by eye.
    """
    rng = _rng(seed)
    img = np.full((size, size), 40, np.uint8)
    filled = np.zeros((size, size), np.uint8)

    cv2.rectangle(img, (60, 60), (220, 200), 200, -1)
    cv2.rectangle(filled, (60, 60), (220, 200), 255, -1)
    cv2.circle(img, (360, 150), 80, 140, -1)
    cv2.circle(filled, (360, 150), 80, 255, -1)
    tri = np.array([[120, 460], [260, 300], [400, 460]], np.int32)
    cv2.fillPoly(img, [tri], 230)
    cv2.fillPoly(filled, [tri], 255)

    edges = cv2.Canny(filled, 50, 150)  # exact boundaries of a noiseless binary mask
    img = to_uint8(to_float(img) + rng.normal(0, 2 / 255, img.shape))
    return img, edges


_FACE_CROP_CACHE: np.ndarray | None = None


def _astronaut_face_crop() -> np.ndarray:
    """A tight crop of the real face in ``skimage.data.astronaut``.

    The crop box is found by running OpenCV's own frontal-face cascade rather
    than hard-coding pixel coordinates: hard-coded numbers silently included the
    white helmet behind the head, which made the composited subject look like a
    face floating in a white oval. Deriving the box means the crop is correct by
    construction and self-documents what it contains.
    """
    global _FACE_CROP_CACHE
    if _FACE_CROP_CACHE is not None:
        return _FACE_CROP_CACHE

    from .io import sample, to_gray

    src = sample("astronaut")
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(to_gray(src), scaleFactor=1.1, minNeighbors=5)

    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # expand to include chin, forehead and a little hair
        pad_x, pad_y = int(w * 0.30), int(h * 0.42)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(src.shape[1], x + w + pad_x), min(src.shape[0], y + h + pad_y)
    else:  # pragma: no cover - only if the bundled cascade ever changes
        x0, y0, x1, y1 = 170, 40, 350, 250

    _FACE_CROP_CACHE = src[y0:y1, x0:x1].copy()
    return _FACE_CROP_CACHE


@dataclass
class PortraitScene:
    """A composited portrait with an exact alpha matte.

    ``fine`` is tracked separately from ``body`` because **thin structures are
    where every segmentation method actually fails**, and a single whole-image
    IoU hides that completely: thin structures are a small fraction of the
    pixels, so losing all of them barely moves the overall score.
    """

    image: np.ndarray
    mask: np.ndarray       # full subject: body + fine
    body: np.ndarray       # the thick core (torso) alone
    fine: np.ndarray       # thin structure: outstretched limbs + the boundary band
    face_box: tuple[int, int, int, int]  # (x, y, w, h) of the subject's face
    background: np.ndarray  # the clean plate, with no subject composited onto it


def portrait_scene(
    size: tuple[int, int] = (640, 640),
    background: str = "coffee",
    seed: int | None = 0,
    **_legacy,
) -> PortraitScene:
    """A **real photograph of a real person**, with a stored reference matte.

    Nothing is composited. The image is
    ``assets/real/messi5.jpg`` exactly as photographed — real person, real hair,
    real kit, real depth, a real crowd behind them — and the ground truth is a
    reference matte stored beside it.

    🚨 **This replaced a synthetic subject, and the old one was indefensible.**
    It was an oval face crop with a *navy triangle* for a body, drawn hair
    strands, pasted onto a coffee cup. It did not look like a portrait, and an
    oval edge against a triangle is not what makes matting hard — so the table
    was scoring methods on a shape that does not resemble the problem.

    ``background`` and ``seed`` are accepted and ignored. They are kept so the
    older call sites still run; there is now exactly one scene, because portrait
    mode needs one photograph of one person and nothing else.

    **The reference matte was produced once by GrabCut plus morphological
    cleanup, then frozen.** That is an ordinary annotation, but it is not
    neutral: it will flatter GrabCut-based methods. The bias is stated here
    rather than buried, and it is why the numbers in this project are read as
    "which methods agree with a careful annotation", not as an absolute ranking.
    """
    from .io import imread

    assets = Path(__file__).resolve().parent.parent / "assets" / "real"
    image = imread(assets / "messi5.jpg")
    matte = imread(assets / "messi5_matte.png")
    if matte.ndim == 3:
        matte = cv2.cvtColor(matte, cv2.COLOR_RGB2GRAY)
    mask = (matte > 127).astype(np.uint8) * 255

    # Split the subject into its thick core and its thin structure. Eroding by a
    # radius wider than an arm strips the limbs away, so what remains is the
    # torso and what was removed is exactly the part every method loses.
    body = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fine = cv2.bitwise_and(mask, cv2.bitwise_not(body))

    face_box = _find_face_box(image, mask)

    # There is no clean background plate for a real photograph -- the pixels
    # behind the subject were never seen. The subject region is inpainted so the
    # compositing reference has *something* plausible there, and the fact that it
    # is a reconstruction rather than ground truth is stated wherever it is used.
    plate = cv2.inpaint(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        cv2.dilate(mask, np.ones((5, 5), np.uint8)),
        7,
        cv2.INPAINT_TELEA,
    )
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

    return PortraitScene(
        image=image,
        mask=mask,
        body=body,
        fine=fine,
        face_box=face_box,
        background=plate,
    )

def _find_face_box(img: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the subject's face, falling back to the top of the silhouette.

    The fallback matters: on a full-body action shot the face is small and a
    cascade often misses it. Returning the head region of the known matte keeps
    the field meaningful instead of returning something arbitrary.
    """
    from .io import to_gray

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if not cascade.empty():
            faces = cascade.detectMultiScale(to_gray(img), 1.1, 5)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return int(x), int(y), int(w), int(h)
    except cv2.error:
        pass

    ys, xs = np.nonzero(mask)
    if not len(ys):
        return (0, 0, 1, 1)
    top = ys.min()
    band = (ys < top + (ys.max() - top) * 0.22)
    bx0, bx1 = xs[band].min(), xs[band].max()
    return int(bx0), int(top), int(bx1 - bx0 + 1), int((ys.max() - top) * 0.22)

def _rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rotation matrix from XYZ Euler angles in degrees."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def camera_homography(
    page_px: tuple[int, int],
    image_size: tuple[int, int],
    focal_ratio: float = 0.9,
    angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fill: float = 0.72,
    shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Homography of a **real pinhole camera** viewing a planar page.

    Returns ``(H, K)`` where ``H`` maps page-texture pixel coordinates to image
    pixel coordinates and ``K`` is the camera intrinsic matrix.

    This matters more than it looks. A homography assembled from four
    hand-chosen corners is not necessarily the image of a rectangle under *any*
    pinhole camera, so geometry that assumes a camera — focal length recovery,
    aspect-ratio recovery, pose estimation — cannot possibly get the right answer
    from it. Building the scene as ``K [r1 r2 t]`` makes the ground truth
    physically realisable, and those methods then have something real to recover.
    """
    pw, ph = page_px
    W, H = image_size
    f = focal_ratio * W
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])

    R = _rotation(*angles_deg)
    tz = f * ph / (fill * H)  # distance that makes the page fill `fill` of the frame
    t = np.array([shift[0] * pw, shift[1] * ph, tz])

    # plane at Z=0: only the first two rotation columns survive
    H_world = K @ np.column_stack([R[:, 0], R[:, 1], t])
    # page texture pixel (u, v) -> world (u - pw/2, v - ph/2)
    T = np.array([[1.0, 0.0, -pw / 2.0], [0.0, 1.0, -ph / 2.0], [0.0, 0.0, 1.0]])
    return H_world @ T, K


def scene_shading(size: tuple[int, int], illum_min: float) -> np.ndarray:
    """The illumination multiplier over a ``(width, height)`` frame.

    A linear ramp from 1.0 at the top-left corner down to ``illum_min`` at the
    bottom-right. Being **affine in (x, y)** is a useful property, not an
    accident: the minimum and maximum over any convex region are attained at its
    vertices, so the illumination ratio across a page can be computed exactly
    from its four corners rather than sampled.
    """
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ramp = 1.0 - (xx / W) * 0.75 - (yy / H) * 0.25
    return (illum_min + (1.0 - illum_min) * ramp).astype(np.float32)


def shading_at(points: np.ndarray, size: tuple[int, int], illum_min: float) -> np.ndarray:
    """Evaluate :func:`scene_shading` at arbitrary ``(x, y)`` points."""
    W, H = size
    p = np.asarray(points, np.float64).reshape(-1, 2)
    ramp = 1.0 - (p[:, 0] / W) * 0.75 - (p[:, 1] / H) * 0.25
    return illum_min + (1.0 - illum_min) * ramp


#: A small fixed vocabulary. Fixed rather than random strings so a page is
#: reproducible from its seed, and word-shaped rather than lorem ipsum so the
#: stroke-width distribution resembles real text.
#: Body-text geometry for :func:`render_text_page`. Thickness is the parameter
#: that decides how hard binarisation is -- see that function's docstring.
BODY_SCALE = 0.46
BODY_THICKNESS = 2

_WORDS = (
    "the", "image", "threshold", "contour", "page", "detect", "classical",
    "vision", "gradient", "pixel", "region", "binarise", "homography", "scan",
    "otsu", "sauvola", "adaptive", "shadow", "illumination", "method", "metric",
    "recover", "perspective", "corner", "document", "measure", "compare", "edge",
)


def render_text_page(
    size: tuple[int, int] = (400, 560), rng: np.random.Generator | None = None
) -> np.ndarray:
    """A page of **real rendered glyphs**, not stand-in strokes.

    ``size`` is ``(width, height)``.

    The earlier version of this drew each line of "text" as a single 4-pixel
    ``cv2.line``. That was geometrically adequate — it gave a text mask to score
    IoU against — but it made every output look like ruled paper, and it made
    the binarisation comparison far too easy: a 4 px solid bar survives almost
    any threshold.

    Real glyphs are **thin, disconnected and anti-aliased**, which is what makes
    binarisation genuinely hard and what the comparison in project 01 is
    supposed to be measuring. Strokes here are 2 px against the old 4 px bars.

    🚨 **Stroke thickness, not font size, decides the difficulty.** Measured at
    flat light, Otsu / Sauvola / oracle text IoU:

    ==============  =====  =======  =======
    body thickness  Otsu   Sauvola  oracle
    ==============  =====  =======  =======
    1 px            0.510  0.517    0.767
    2 px            0.901  0.842    0.932
    ==============  =====  =======  =======

    A 1 px anti-aliased stroke is roughly half ambiguous grey, so *no* method can
    score well and the comparison stops discriminating between them. 2 px is what
    body text actually measures in a page photographed at this resolution, and it
    leaves the methods separable -- which is the point of the scene.

    Rendered with Hershey vector fonts, which ship inside OpenCV, so this still
    needs no download and no font file.
    """
    rng = _rng(0) if rng is None else rng
    page_w, page_h = size
    page = np.full((page_h, page_w), 245, np.uint8)

    margin = 30
    ink = 45

    # title, in a heavier face so the page has a realistic range of stroke widths
    cv2.putText(
        page, "CLASSICAL VISION", (margin, 52),
        cv2.FONT_HERSHEY_DUPLEX, 0.62, ink, 1, cv2.LINE_AA,
    )
    cv2.line(page, (margin, 64), (page_w - margin, 64), 120, 1)

    # body: words laid out with real wrapping, so line lengths vary naturally
    y = 96
    line_height = 24
    while y < page_h - 120:
        x = margin
        while True:
            word = str(_WORDS[int(rng.integers(0, len(_WORDS)))])
            (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, BODY_SCALE, BODY_THICKNESS)
            if x + tw > page_w - margin:
                break
            cv2.putText(
                page, word, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, BODY_SCALE, ink, BODY_THICKNESS, cv2.LINE_AA,
            )
            x += tw + 7
        y += line_height

    # a boxed block, which gives contour-based methods a rectangle INSIDE the
    # page to be confused by -- the page border must still win
    box_top = page_h - 96
    cv2.rectangle(page, (margin, box_top), (page_w - margin, page_h - 34), 110, 1)
    cv2.putText(
        page, "total   1284.50", (margin + 12, box_top + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, ink, 1, cv2.LINE_AA,
    )
    cv2.putText(
        page, "checked by  o.c.v.", (margin + 12, box_top + 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ink, 1, cv2.LINE_AA,
    )
    return page


def document_scene(
    size: tuple[int, int] = (900, 700),
    seed: int | None = 0,
    illum_min: float = 0.62,
    focal_ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A flat "page" photographed at an angle on a cluttered desk.

    ``size`` is ``(width, height)`` — the cv2 convention, not numpy's
    ``(rows, cols)``. The two are transposed and neither library complains, so
    the order is spelled out here and unpacked explicitly.

    ``illum_min`` is the brightness multiplier at the **darkest** corner of the
    frame; the brightest corner is always 1.0. So ``1.0`` is perfectly flat
    studio light and ``0.15`` is a hand casting a deep shadow over half the page.
    It is a parameter rather than a constant because the illumination ratio is
    what decides whether a global threshold can work at all: once the shadowed
    paper is darker than the sunlit text, no single global cut exists.

    Returns ``(photo, flat_page, corners)`` where ``corners`` is the exact
    ``(x, y)`` position of the four page corners in the photo, in
    top-left / top-right / bottom-right / bottom-left order. Those corners are
    the ground truth a document scanner is scored against.
    """
    rng = _rng(seed)
    W, H = size

    page_h, page_w = 560, 400
    page = render_text_page((page_w, page_h), rng)
    page = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB)

    desk = np.zeros((H, W, 3), np.uint8)
    desk[:, :] = (96, 74, 58)
    noise = rng.normal(0, 9, desk.shape)
    desk = to_uint8(to_float(desk) + noise / 255.0)
    for _ in range(7):  # desk clutter, so contour finding has to discriminate
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.rectangle(desk, (x, y), (x + 70, y + 40), (60, 50, 44), -1)

    # Pose the page in front of a real camera. Poses are resampled until all four
    # corners land inside the frame, so no scene is silently cropped.
    src = np.float32([[0, 0], [page_w - 1, 0], [page_w - 1, page_h - 1], [0, page_h - 1]])
    M, dst = None, None
    for _ in range(60):
        angles = (
            float(rng.uniform(-26, 26)),  # tilt away from / towards the camera
            float(rng.uniform(-26, 26)),  # tilt left / right
            float(rng.uniform(-14, 14)),  # roll
        )
        shift = (float(rng.uniform(-0.10, 0.10)), float(rng.uniform(-0.10, 0.10)))
        candidate, _K = camera_homography(
            (page_w, page_h), (W, H), focal_ratio=focal_ratio, angles_deg=angles, shift=shift
        )
        corners = cv2.perspectiveTransform(src.reshape(-1, 1, 2), candidate).reshape(-1, 2)
        margin = 20
        if (
            corners[:, 0].min() > margin
            and corners[:, 0].max() < W - margin
            and corners[:, 1].min() > margin
            and corners[:, 1].max() < H - margin
        ):
            M, dst = candidate, corners.astype(np.float32)
            break
    if M is None:  # fall back to a gentle head-on pose rather than fail
        M, _K = camera_homography(
            (page_w, page_h), (W, H), focal_ratio=focal_ratio, angles_deg=(12.0, -10.0, 4.0)
        )
        dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.float32)

    warped = cv2.warpPerspective(page, M, (W, H))
    page_mask = cv2.warpPerspective(
        np.full((page_h, page_w), 255, np.uint8), M, (W, H)
    )

    photo = desk.copy()
    photo[page_mask > 0] = warped[page_mask > 0]

    # a soft illumination gradient, the thing that breaks global thresholding
    shade = scene_shading((W, H), illum_min)
    photo = to_uint8(to_float(photo) * shade[..., None])
    photo = to_uint8(to_float(photo) + rng.normal(0, 3 / 255, photo.shape))

    return photo, page, dst
