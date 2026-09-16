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
    """A copy-move forgery, with masks for the pasted region and for both copies.

    Two masks, because there are two defensible answers to "which pixels are the
    forgery":

    ``mask``
        The **pasted** pixels only. This is what was actually changed, and the
        only thing an editor would call the forgery.
    ``mask_both``
        The pasted pixels **and the region they were copied from**. This is what
        any copy-move detector can actually report.

    The difference is not a technicality. After the paste, the two regions are
    *identical*: same content, same noise, same compression history. Nothing in
    the image distinguishes the original from the copy, so a detector that finds
    a duplicated pair is finished — deciding which half is the forgery needs
    outside information (lighting, perspective, a second photo) that pixel
    statistics do not contain.

    Scoring against ``mask`` therefore caps precision at about 0.5 no matter how
    good the detector is, and measures the ambiguity rather than the method.
    Projects here score against ``mask_both`` and say so.
    """

    image: np.ndarray
    mask: np.ndarray
    mask_both: np.ndarray
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
    textured_source: bool = True,
    feather: int = 0,
) -> Forgery:
    """Copy a square patch and paste it elsewhere, optionally rotated/scaled.

    The returned masks mark the pasted pixels exactly (and, in ``mask_both``,
    the region they came from), which makes detection a scoreable segmentation
    problem instead of an eyeball test.

    ``textured_source``
        Sample the source patch from a *textured* part of the image rather than
        uniformly at random. A patch of empty sky duplicated into another patch
        of empty sky is not a detectable forgery by any of these methods — and
        it is not one a forger would make either, since there is nothing there to
        hide. Leaving this off makes the scores depend mostly on where the RNG
        happened to look.

    ``feather``
        Blend the paste over this many pixels at its border. A hard-edged paste
        leaves a discontinuity that has nothing to do with duplication, and any
        edge detector would find it — so a detector can look good here for the
        wrong reason. Feathering removes that shortcut at the cost of making the
        mask's boundary approximate.
    """
    rng = _rng(seed)
    h, w = img.shape[:2]
    out = img.copy()

    if textured_source:
        # local standard deviation, subsampled on the stride the search uses
        g = to_float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img)
        # anchor=(0, 0) so the window runs DOWN-RIGHT from each pixel. The default
        # anchor centres it, which would have scored the window centred on (x, y)
        # and then cropped the patch starting AT (x, y) — half a patch away from
        # the texture that was measured.
        mean = cv2.boxFilter(g, -1, (size, size), anchor=(0, 0), borderType=cv2.BORDER_ISOLATED)
        sq = cv2.boxFilter(g * g, -1, (size, size), anchor=(0, 0), borderType=cv2.BORDER_ISOLATED)
        valid = np.sqrt(np.maximum(sq - mean * mean, 0.0))[: h - size + 1, : w - size + 1]
        # choose among the top decile, at random, so it is still varied per seed
        cut = float(np.percentile(valid, 90.0))
        ys, xs = np.nonzero(valid >= cut)
        pick = int(rng.integers(0, len(ys)))
        sy, sx = int(ys[pick]), int(xs[pick])
    else:
        sx = int(rng.integers(0, max(1, w - size)))
        sy = int(rng.integers(0, max(1, h - size)))

    # Sample a source window big enough that the rotated/scaled crop is filled
    # with REAL image content. Rotating a size x size patch in place and letting
    # BORDER_REFLECT invent the corners means ~20% of a 15-degree "forgery" is not
    # a duplicate of anything, which silently caps recall for every method and
    # makes the rotation sweep unreadable.
    # max(size, ...) because for scale > 1 the source region is SMALLER than the
    # paste, but the warp buffer still has to be big enough to crop size x size
    # out of. Without the clamp, scale=1.5 asked for a 93 px window and then
    # cropped 96 px from it, producing a 2x2 array and a broadcast error.
    need = max(size, int(np.ceil(size * np.sqrt(2) / max(scale, 1e-3)))) + 2
    need = min(need, min(h, w))
    cx, cy = sx + size / 2.0, sy + size / 2.0
    cx = float(np.clip(cx, need / 2.0, w - need / 2.0))
    cy = float(np.clip(cy, need / 2.0, h - need / 2.0))
    sx, sy = int(round(cx - size / 2.0)), int(round(cy - size / 2.0))

    x0, y0 = int(round(cx - need / 2.0)), int(round(cy - need / 2.0))
    x0, y0 = max(0, min(w - need, x0)), max(0, min(h - need, y0))
    window = img[y0 : y0 + need, x0 : x0 + need]

    if angle_deg or scale != 1.0:
        m = cv2.getRotationMatrix2D((need / 2.0, need / 2.0), angle_deg, scale)
        rotated = cv2.warpAffine(window, m, (need, need), flags=cv2.INTER_LINEAR)
        off = (need - size) // 2
        patch = rotated[off : off + size, off : off + size].copy()
    else:
        patch = img[sy : sy + size, sx : sx + size].copy()

    for _ in range(64):  # find a destination that does not overlap the source
        dx = int(rng.integers(0, max(1, w - size)))
        dy = int(rng.integers(0, max(1, h - size)))
        if abs(dx - sx) > size or abs(dy - sy) > size:
            break

    if feather > 0:
        alpha = np.zeros((size, size), np.float32)
        alpha[feather:-feather, feather:-feather] = 1.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather / 2.0)
        a = alpha[..., None] if img.ndim == 3 else alpha
        region = out[dy : dy + size, dx : dx + size].astype(np.float32)
        out[dy : dy + size, dx : dx + size] = (
            a * patch.astype(np.float32) + (1.0 - a) * region
        ).astype(np.uint8)
    else:
        out[dy : dy + size, dx : dx + size] = patch

    mask = np.zeros((h, w), np.uint8)
    mask[dy : dy + size, dx : dx + size] = 255

    # The source footprint is the destination square carried back through the
    # transform — a ROTATED square, not an axis-aligned one. Marking an
    # axis-aligned box instead would score a correct detector as over-detecting
    # by (1 - 2/pi) of the box at 45 degrees.
    mask_both = mask.copy()
    if angle_deg or scale != 1.0:
        theta = np.deg2rad(angle_deg)
        r = size / (2.0 * scale)
        corners = np.array([[-r, -r], [r, -r], [r, r], [-r, r]], np.float64)
        rot = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64
        )
        pts = (corners @ rot.T) + np.array([sx + size / 2.0, sy + size / 2.0])
        cv2.fillPoly(mask_both, [np.round(pts).astype(np.int32)], 255)
    else:
        mask_both[sy : sy + size, sx : sx + size] = 255
    return Forgery(out, mask, mask_both, (sx, sy), (dx, dy), size, angle_deg, scale)


def add_scratches(
    img: np.ndarray,
    n_strokes: int = 14,
    thickness: int = 3,
    blotches: int = 6,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw emulsion scratches and dust onto a print.

    Returns ``(damaged, mask)``. The mask is the exact inpainting target, so a
    restoration can be scored against the untouched original rather than judged
    by eye.

    **The damage is not one constant value.** An earlier version painted every
    damaged pixel pure 255, which made damage *detection* a solved problem: the
    rule ``pixel >= 250`` recovered the mask almost exactly, and any detector
    that reasoned about shape instead of brightness looked worse than that rule.
    That is a property of the generator, not of the detectors. Here a scratch is
    a bright value drawn per stroke and jittered per pixel, some blotches are
    dark dust rather than bright emulsion loss, and a few strokes are *darker*
    than the image — so brightness alone no longer separates damage from a
    genuine highlight.
    """
    rng = _rng(seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    value = np.zeros((h, w), np.float32)

    for _ in range(n_strokes):
        stroke = np.zeros((h, w), np.uint8)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        pts = [(x, y)]
        for _ in range(int(rng.integers(3, 7))):  # a wandering polyline, not a straight line
            x = int(np.clip(x + rng.integers(-w // 6, w // 6), 0, w - 1))
            y = int(np.clip(y + rng.integers(-h // 8, h // 8), 0, h - 1))
            pts.append((x, y))
        cv2.polylines(stroke, [np.array(pts, np.int32)], False, 255, thickness)
        # one stroke in five is a dark crease, not a bright emulsion scratch
        level = float(rng.integers(20, 70) if rng.random() < 0.2 else rng.integers(195, 256))
        value[stroke > 0] = level
        mask |= stroke

    for _ in range(blotches):
        blob = np.zeros((h, w), np.uint8)
        cv2.circle(
            blob,
            (int(rng.integers(0, w)), int(rng.integers(0, h))),
            int(rng.integers(4, 12)),
            255,
            -1,
        )
        level = float(rng.integers(10, 60) if rng.random() < 0.4 else rng.integers(200, 256))
        value[blob > 0] = level
        mask |= blob

    sel = mask > 0
    jitter = rng.normal(0.0, 8.0, size=(h, w)).astype(np.float32)
    level = np.clip(value + jitter, 0.0, 255.0)

    damaged = img.copy()
    if damaged.ndim == 3:
        damaged[sel] = level[sel, None].astype(np.uint8)
    else:
        damaged[sel] = level[sel].astype(np.uint8)
    return damaged, mask


# Cyan dye is the least stable, then magenta, then yellow — so the blue channel
# loses the most and the print drifts warm. These are the surviving fractions.
FADE_GAIN = np.array([0.90, 0.82, 0.62], np.float32)
# Paper yellows as it oxidises, and the yellowed base sets the new black point.
FADE_PAPER = np.array([1.00, 0.94, 0.76], np.float32)


def fade_photo(
    img: np.ndarray,
    contrast: float = 0.62,
    lift: float = 0.22,
    saturation: float = 0.72,
    seed: int | None = 0,
) -> np.ndarray:
    """Age a photograph the way time ages one: desaturated, flat, lifted and warm.

    Four separable things happen to a colour print, and this applies them in the
    order they physically occur:

    1. **Dye loss desaturates** the image toward its own luminance.
    2. **Each dye layer fades at its own rate** (``FADE_GAIN``) — a per-channel
       gain, which is what turns the picture warm.
    3. **Contrast compresses** as the density range collapses.
    4. **The paper yellows**, lifting the black point by a tinted ``lift``.

    Steps 2-4 are all *diagonal* — independent per channel — so a per-channel
    stretch can invert them. Step 1 is not: desaturation is a rank-reducing mix
    across channels and no per-channel curve can undo it. That split is the whole
    point of the model, and project 05 measures exactly how much of the loss sits
    on each side of it.

    Unlike :func:`add_scratches` there is no mask here: this is a tone problem,
    not an inpainting problem, and the methods that fix one do nothing for the
    other.
    """
    f = to_float(img)
    if f.ndim == 3:
        gray = (f @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        f = gray + saturation * (f - gray)          # 1. dye loss
        f = f * FADE_GAIN                           # 2. unequal layer fading
    f = f * contrast                                # 3. density range collapses
    f = f + lift * (FADE_PAPER if f.ndim == 3 else 1.0)  # 4. yellowed paper base
    grain = _rng(seed).normal(0.0, 0.010, size=f.shape).astype(np.float32)
    return to_uint8(f + grain)


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


#: The kinds of document `render_text_page` can produce, with the page shape each
#: one naturally has. A document scanner meets all of these, and they stress it
#: differently: a form is full of internal rectangles that a contour detector can
#: mistake for the page, a receipt is narrow and sparse, a two-column article has
#: a very different ink distribution from a letter.
PAGE_KINDS = {
    "letter": (400, 560),
    "receipt": (260, 620),
    "form": (560, 400),
    "article": (460, 600),
}


def render_text_page(
    size: tuple[int, int] = (400, 560),
    rng: np.random.Generator | None = None,
    kind: str = "letter",
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

    if kind == "receipt":
        # a till receipt: narrow, sparse, right-aligned figures, dashed rules.
        # Very different ink density from a letter, and a different page shape.
        cv2.putText(page, "CLASSICAL MART", (margin, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, ink, 1, cv2.LINE_AA)
        y = 78
        for _ in range(int(rng.integers(9, 14))):
            item = str(_WORDS[int(rng.integers(0, len(_WORDS)))])[:11]
            price = f"{rng.uniform(0.6, 24.0):.2f}"
            cv2.putText(page, item, (margin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, ink, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(price, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(page, price, (page_w - margin - tw, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, ink, 1, cv2.LINE_AA)
            y += 26
        for yy in (y + 6, y + 40):
            for x0 in range(margin, page_w - margin, 10):
                cv2.line(page, (x0, yy), (x0 + 5, yy), 120, 1)
        cv2.putText(page, "TOTAL", (margin, y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, ink, 1, cv2.LINE_AA)
        cv2.putText(page, f"{rng.uniform(40, 180):.2f}", (page_w - margin - 70, y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, ink, 1, cv2.LINE_AA)
        return page

    if kind == "form":
        # a ruled table: the hard case for contour detection, because the page
        # now contains many strong rectangles of its own and the page border
        # still has to win.
        cv2.putText(page, "INSPECTION RECORD", (margin, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, ink, 1, cv2.LINE_AA)
        top, rows_n = 70, 9
        row_h = (page_h - top - 40) // rows_n
        col_x = [margin, margin + 150, margin + 270, page_w - margin]
        for r in range(rows_n + 1):
            yy = top + r * row_h
            cv2.line(page, (margin, yy), (page_w - margin, yy), 110, 1)
        for cx in col_x:
            cv2.line(page, (cx, top), (cx, top + rows_n * row_h), 110, 1)
        for r in range(rows_n):
            yy = top + r * row_h + int(row_h * 0.68)
            cv2.putText(page, str(_WORDS[int(rng.integers(0, len(_WORDS)))])[:12],
                        (col_x[0] + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ink, 1, cv2.LINE_AA)
            cv2.putText(page, f"{rng.uniform(0.1, 9.9):.2f}",
                        (col_x[1] + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ink, 1, cv2.LINE_AA)
            cv2.putText(page, "PASS" if rng.random() > 0.3 else "FAIL",
                        (col_x[2] + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ink, 1, cv2.LINE_AA)
        return page

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
    page_image: np.ndarray | None = None,
    page_kind: str = "letter",
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

    if page_image is None:
        page_w, page_h = PAGE_KINDS.get(page_kind, PAGE_KINDS["letter"])
        page = render_text_page((page_w, page_h), rng, kind=page_kind)
        page = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB)
    else:
        # A REAL document, posed by the same synthetic camera. This is what lets
        # the project show genuine document variety — a newspaper sudoku, a page
        # of printed prose, a form — while keeping the exact corner ground truth
        # that a real photograph of a page cannot give you. The content is real;
        # only the pose and the lighting are ours, and those are the two things
        # being measured.
        page = page_image if page_image.ndim == 3 else cv2.cvtColor(page_image, cv2.COLOR_GRAY2RGB)
        longest = 560.0 / max(page.shape[:2])
        page = cv2.resize(page, None, fx=longest, fy=longest, interpolation=cv2.INTER_AREA)
        page_h, page_w = page.shape[:2]

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
    margin = 20

    # Widen the field of view until the page fits, instead of giving up and
    # posing it off-frame. A supplied real document can have any aspect ratio —
    # `printed_text` is 2.2:1 — and at the default focal ratio a wide page simply
    # cannot be tilted and still land inside the frame. The old fallback posed it
    # anyway, so the corners ran off the edge and EVERY detector "failed" on a
    # scene that was never solvable. That is a broken sample, not a hard one.
    for focal in (focal_ratio, focal_ratio * 0.78, focal_ratio * 0.60, focal_ratio * 0.46):
        for _ in range(60):
            angles = (
                float(rng.uniform(-26, 26)),  # tilt away from / towards the camera
                float(rng.uniform(-26, 26)),  # tilt left / right
                float(rng.uniform(-14, 14)),  # roll
            )
            shift = (float(rng.uniform(-0.10, 0.10)), float(rng.uniform(-0.10, 0.10)))
            candidate, _K = camera_homography(
                (page_w, page_h), (W, H), focal_ratio=focal, angles_deg=angles, shift=shift
            )
            corners = cv2.perspectiveTransform(src.reshape(-1, 1, 2), candidate).reshape(-1, 2)
            if (
                corners[:, 0].min() > margin
                and corners[:, 0].max() < W - margin
                and corners[:, 1].min() > margin
                and corners[:, 1].max() < H - margin
            ):
                M, dst = candidate, corners.astype(np.float32)
                break
        if M is not None:
            break
    if M is None:  # fall back to a gentle head-on pose rather than fail
        M, _K = camera_homography(
            (page_w, page_h), (W, H), focal_ratio=focal_ratio * 0.46, angles_deg=(8.0, -6.0, 3.0)
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


# --------------------------------------------------------------------------- #
# coins of known denomination
# --------------------------------------------------------------------------- #
#
# Counting coins is answerable from `skimage.data.coins`, a scan of old Greek
# coins. IDENTIFYING them is not: nobody recorded what those coins are worth, so
# a denomination reported against that image would be unfalsifiable.
#
# These scenes exist for exactly that reason. The mint publishes the diameter of
# every circulating coin, so placing one of known value gives a scene where the
# count, the diameter in millimetres AND the denomination are known by
# construction — and a wrong answer can be shown to be wrong.

#: Indian circulating coins, 2011 series onward, with published diameters in mm.
#:
#: Two facts here do the work, and neither is an artefact of the generator:
#:
#: * **1 and 5 differ by 1.07 mm.** At ordinary framing that is a pixel or two,
#:   so telling them apart is a measurement problem before it is a
#:   classification problem.
#: * **10 and 20 are both 27.00 mm.** In the hand they are told apart by the
#:   20's twelve-sided edge, not by size, so *no* diameter-based method can
#:   separate them. A confusion matrix showing that is reporting the coinage,
#:   not a bug.
RUPEE_COINS_MM = {
    1: 21.93,
    2: 25.00,
    5: 23.00,
    10: 27.00,
    20: 27.00,
}


def coin_scene(
    size: tuple[int, int] = (720, 540),
    counts: dict[int, int] | None = None,
    mm_per_px: float = 0.42,
    background: str = "felt",
    illum_min: float = 1.0,
    touching: bool = False,
    seed: int | None = 0,
):
    """Render coins of known denomination and return the scene with its truth.

    Returns ``(image, coins)``; each coin is a dict with ``denomination``,
    ``centre_xy``, ``diameter_px`` and ``diameter_mm``.

    ``mm_per_px`` sets the scale, so a caller can ask how identification decays
    as the camera moves back. That is the only honest way to report a
    diameter-based classifier, whose accuracy is a function of how many pixels a
    millimetre is worth rather than a fixed property of the method.

    ``touching`` places coins deliberately in contact, which is what breaks
    connected components and is the reason watershed is in this project.
    """
    rng = _rng(seed)
    W, H = size
    # Four of each of the five circulating coins, 20 in all. The 20-rupee
    # coins are in the default on purpose: they share the 10's 27.00 mm
    # diameter exactly, so a scene without them lets a size-based reader look
    # better than the coinage allows.
    counts = counts or {1: 4, 2: 4, 5: 4, 10: 4, 20: 4}

    img = _coin_background(size, background, rng)

    wanted: list[int] = []
    for denom, n in counts.items():
        wanted.extend([denom] * n)
    rng.shuffle(wanted)

    placed: list[dict] = []
    # How close two coins may come, as a fraction of the smaller radius.
    # Negative means they overlap, which is what genuine contact looks like
    # once both coins have a rim and a shadow.
    gap = -0.06 if touching else 0.12
    for denom in wanted:
        d_mm = RUPEE_COINS_MM[denom]
        r_px = 0.5 * d_mm / mm_per_px
        for attempt in range(400):
            if touching and placed:
                # Deliberately push each new coin up against one already down.
                # Sampling uniformly and merely *permitting* overlap does not
                # produce it: there is far more free felt than contact, so a
                # uniform sampler lands in the gaps every time and the scene
                # comes out as twenty well-separated discs. Connected components
                # then segments it perfectly and the whole reason watershed is
                # in this project disappears.
                anchor = placed[int(rng.integers(len(placed)))]
                ox, oy = anchor["centre_xy"]
                reach = r_px + anchor["diameter_px"] / 2
                reach += gap * min(r_px, anchor["diameter_px"] / 2)
                theta = float(rng.uniform(0, 2 * np.pi))
                cx = ox + reach * float(np.cos(theta))
                cy = oy + reach * float(np.sin(theta))
                if not (r_px + 4 <= cx <= W - r_px - 4 and r_px + 4 <= cy <= H - r_px - 4):
                    continue
            else:
                cx = float(rng.uniform(r_px + 4, W - r_px - 4))
                cy = float(rng.uniform(r_px + 4, H - r_px - 4))
            ok = True
            for other in placed:
                ox, oy = other["centre_xy"]
                need = r_px + other["diameter_px"] / 2
                need += gap * min(r_px, other["diameter_px"] / 2)
                if (cx - ox) ** 2 + (cy - oy) ** 2 < need**2 - 1e-6:
                    ok = False
                    break
            if ok:
                _draw_coin(img, (cx, cy), r_px, denom, rng)
                placed.append(
                    {
                        "denomination": denom,
                        "centre_xy": (cx, cy),
                        "diameter_px": 2 * r_px,
                        "diameter_mm": d_mm,
                    }
                )
                break

    if illum_min < 1.0:
        img = to_uint8(to_float(img) * scene_shading(size, illum_min)[..., None])
    img = to_uint8(to_float(img) + rng.normal(0, 2.5 / 255, img.shape))
    return img, placed


def _coin_background(size: tuple[int, int], kind: str, rng) -> np.ndarray:
    """A background for a coin scene. Never flat — a flat one is not a test."""
    W, H = size
    if kind == "felt":
        base = np.array([28, 58, 40], np.float32)
        texture = rng.normal(0, 7, (H, W, 1))
    elif kind == "wood":
        base = np.array([132, 96, 58], np.float32)
        grain = np.sin(np.linspace(0, 38, W))[None, :, None] * 7
        texture = grain + rng.normal(0, 4, (H, W, 1))
    elif kind == "paper":
        base = np.array([228, 224, 214], np.float32)
        texture = rng.normal(0, 3, (H, W, 1))
    elif kind == "marble":
        base = np.array([196, 194, 190], np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        veins = np.sin(xx * 0.02 + yy * 0.031) * 9 + np.sin(xx * 0.004 - yy * 0.009) * 12
        texture = veins[..., None] + rng.normal(0, 3, (H, W, 1))
    elif kind == "slate":
        base = np.array([62, 64, 68], np.float32)
        texture = rng.normal(0, 6, (H, W, 1))
    elif kind == "navy":
        base = np.array([26, 34, 72], np.float32)
        texture = rng.normal(0, 5, (H, W, 1))
    elif kind == "denim":
        base = np.array([54, 72, 104], np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        weave = (np.sin(xx * 1.7) + np.sin(yy * 1.7)) * 4
        texture = weave[..., None] + rng.normal(0, 4, (H, W, 1))
    else:
        raise ValueError(f"unknown background {kind!r}")
    return np.clip(base[None, None, :] + texture, 0, 255).astype(np.uint8)


#: Face tone per denomination, so the scene is not twenty identical discs.
#:
#: The floor here is load-bearing. An earlier version put the 10-rupee coin at
#: 150 with a stronger shading gradient, which took its far edge to 102 against
#: an Otsu threshold of 105 -- so every 10-rupee coin came out fragmented and
#: the watershed dropped all five of them, scoring 15 of 20 on a scene where the
#: coins were not even touching. The tones still differ; none of them reaches
#: the background.
_COIN_TONE = {1: 190, 2: 200, 5: 182, 10: 174, 20: 210}
#: Brass-coloured denominations get a warm tint; the rest stay steel.
_COIN_WARM = (5, 10)


def _draw_coin(img: np.ndarray, centre, radius_px: float, denom: int, rng) -> None:
    """Draw one coin in place: metal disc, rim, faint relief, soft shadow.

    Drawn rather than pasted from a photograph because the whole point is that
    the *diameter* is exact. A photograph brings its own perspective and its own
    unknown scale, and the millimetre ground truth would be a guess.
    """
    cx, cy = centre
    r = int(round(radius_px))
    if r < 3:
        return
    span = 2 * r + 9
    H, W = img.shape[:2]
    if span > W or span > H:
        return
    x0 = max(0, min(int(round(cx)) - r - 4, W - span))
    y0 = max(0, min(int(round(cy)) - r - 4, H - span))
    patch = img[y0 : y0 + span, x0 : x0 + span]
    lc = (span // 2, span // 2)

    # a drop shadow, offset down-right, so the coins sit on the surface
    shadow = np.zeros((span, span), np.float32)
    cv2.circle(shadow, (lc[0] + 3, lc[1] + 3), r, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.5, r * 0.10))
    patch[:] = to_uint8(to_float(patch) * (1.0 - 0.45 * shadow[..., None]))

    # the metal, lit from the upper left so the disc reads three-dimensional
    tone = _COIN_TONE[denom]
    warm = (
        np.array([1.00, 0.96, 0.84], np.float32)
        if denom in _COIN_WARM
        else np.ones(3, np.float32)
    )
    yy, xx = np.mgrid[0:span, 0:span].astype(np.float32)
    lit = 1.0 - 0.09 * ((xx - lc[0]) + (yy - lc[1])) / max(r, 1)
    face = np.clip(tone * lit, 40, 255)[..., None] * warm[None, None, :]

    disc = np.zeros((span, span), np.float32)
    cv2.circle(disc, lc, r, 1.0, -1)
    # the rim: a brighter ring just inside the edge, which is what a real coin
    # has and what gives an edge detector something to find
    rim = np.zeros((span, span), np.float32)
    cv2.circle(rim, lc, r, 1.0, max(1, int(r * 0.10)))
    face = face + rim[..., None] * 26

    # a suggestion of relief, so the interior is not a flat disc that any
    # threshold separates perfectly
    relief = cv2.GaussianBlur(rng.normal(0, 6, (span, span)).astype(np.float32), (0, 0), 1.2)
    face = face + relief[..., None] * disc[..., None]

    soft = cv2.GaussianBlur(disc, (0, 0), 0.7)[..., None]
    patch[:] = to_uint8(
        to_float(patch) * (1 - soft) + to_float(np.clip(face, 0, 255).astype(np.uint8)) * soft
    )
