"""Image loading, saving and dtype conversion.

The BGR/RGB rule
----------------
``cv2.imread`` returns **BGR**. ``matplotlib.pyplot.imshow`` expects **RGB**.
Getting this wrong makes every figure look blue and produces no error at all.

This module enforces one convention: **everything in this repo is RGB uint8**.
Conversion to BGR happens only at the moment a cv2 function needs it, and is
handled inside :func:`imwrite`. No project file should ever call
``cv2.cvtColor(..., COLOR_BGR2RGB)`` directly.

The coordinate rule
-------------------
``cv2`` takes points as ``(x, y)``. ``numpy`` indexes as ``[row, col]`` which is
``(y, x)``. They are transposed relative to each other and neither will complain.
Helpers here always document which convention they use.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

# --------------------------------------------------------------------------- #
# dtype conversion
# --------------------------------------------------------------------------- #


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. A float image is passed through."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        return img.astype(np.float32, copy=False)
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8 [0, 255], clipped. A uint8 image is passed through.

    Clipping matters: Retinex, sharpening and deconvolution all overshoot past
    1.0, and a silent wraparound turns a bright highlight into a black hole.
    """
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #


def imread(path: str | Path, gray: bool = False) -> np.ndarray:
    """Read an image from disk **as RGB uint8** (or 2-D grayscale if ``gray``).

    Raises FileNotFoundError rather than returning ``None``, which is what cv2
    does for a missing path and which turns into a confusing ``NoneType`` error
    several lines later.
    """
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read an image at {path}")
    if gray:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: str | Path, img: np.ndarray) -> Path:
    """Write an **RGB** (or grayscale) image to disk, converting to BGR for cv2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_uint8(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"cv2 failed to write {path}")
    return path


# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> 2-D grayscale uint8. Already-gray input is passed through."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """2-D grayscale -> 3-channel RGB, so figures can stack mixed results."""
    if img.ndim == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# --------------------------------------------------------------------------- #
# sample images that ship with scikit-image (no download, ever)
# --------------------------------------------------------------------------- #

#: Names accepted by :func:`sample`, mapped to the ``skimage.data`` loader.
#: Every one of these is bundled with scikit-image, so no project in this repo
#: needs a network connection to produce a result.
_SAMPLES = {
    "astronaut": skdata.astronaut,  # colour portrait, faces, fine detail
    "camera": skdata.camera,        # grayscale, strong edges, the classic test
    "coins": skdata.coins,          # grayscale, touching objects -> watershed
    "chelsea": skdata.chelsea,      # colour cat, fur texture
    "coffee": skdata.coffee,        # colour, saturated, circular shapes
    "page": skdata.page,            # uneven illumination -> Otsu vs Sauvola
    "text": skdata.text,            # binarisation target
    "moon": skdata.moon,            # low contrast -> histogram work
    "brick": skdata.brick,          # texture
    "grass": skdata.grass,          # texture
    "gravel": skdata.gravel,        # texture
    "horse": skdata.horse,          # binary shape + ground-truth mask
    "rocket": skdata.rocket,        # colour, sky/ground split
    "immunohistochemistry": skdata.immunohistochemistry,
    "retina": skdata.retina,        # thin structures, punishing for IoU
    "cell": skdata.cell,            # microscopy, low contrast
}


def sample(name: str = "astronaut", gray: bool = False) -> np.ndarray:
    """Return a bundled sample image as RGB uint8 (or 2-D grayscale).

    ``skimage.data.horse`` is boolean and ``cell``/``camera`` are 2-D; both are
    normalised here so callers always get uint8 with a predictable shape.
    """
    if name not in _SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(_SAMPLES)}")
    img = _SAMPLES[name]()

    if img.dtype == bool:
        img = (~img).astype(np.uint8) * 255  # horse is True on the *background*
    elif img.dtype != np.uint8:
        img = to_uint8(img / max(img.max(), 1))

    if gray:
        return to_gray(img) if img.ndim == 3 else img
    return ensure_rgb(img)


#: Real photographs bundled with the repo, from OpenCV's BSD-licensed sample
#: data. They exist to show a pipeline working on an image nobody constructed
#: for it. They have **no ground truth**, so nothing that needs one may be
#: reported against them -- see assets/real/README.md.
REAL_PHOTOS = {
    "player": ("messi5.jpg", "a footballer on a pitch, real depth and a real crowd"),
    "newspaper": ("sudoku.png", "a newspaper page photographed at an angle"),
    "printed_text": ("imageTextN.png", "a page of clean printed text"),
    "defocused_text": ("text_defocus.jpg", "printed text, defocused"),
}


def real_photo(name: str) -> np.ndarray:
    """Load one of the bundled **real** photographs as RGB uint8.

    Deliberately a separate function from :func:`sample`. The two are used for
    different things and scored differently: a generated scene has exact ground
    truth and gets a PSNR or an IoU, a real photograph has neither and gets
    shown rather than scored. Keeping them apart at the API makes it hard to
    accidentally quote an accuracy for an image that has no answer.
    """
    if name not in REAL_PHOTOS:
        raise KeyError(f"unknown real photo {name!r}; choose from {sorted(REAL_PHOTOS)}")
    filename, _ = REAL_PHOTOS[name]
    path = Path(__file__).resolve().parent.parent / "assets" / "real" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The real photographs live in assets/real/; "
            "see assets/real/README.md for their provenance."
        )
    return imread(path)


def real_photo_names() -> list[str]:
    return sorted(REAL_PHOTOS)


def sample_names() -> list[str]:
    """Sorted list of every bundled sample name."""
    return sorted(_SAMPLES)
