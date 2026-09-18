"""Point transforms: per-pixel mappings, and what a lookup table costs.

The question
------------
A point transform maps every pixel through the same function, independent of its
neighbours. It is the simplest family in computer vision and the one where an
exact, provable statement is available:

> **The claim under test:** any point transform on 8-bit data is *exactly* a
> 256-entry lookup table. Implementing one with numpy arithmetic and implementing
> it with ``cv2.LUT`` must produce **bit-identical** output — and the LUT should
> be dramatically faster, because it replaces per-pixel maths with a memory read.

That is checkable to the bit rather than to a tolerance, which makes it a
correctness test rather than a benchmark.

The second, more interesting claim:

> **Every monotonic point transform destroys information, and exactly how much is
> computable in advance.** Counting how many of the 256 input levels map to
> distinct outputs gives the loss *before any image is processed* — the same
> quantisation argument as project 03, but here in closed form for each curve.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import entropy, psnr, rms_contrast

EPS = 1e-9


# --------------------------------------------------------------------------- #
# the curves, defined as lookup tables
# --------------------------------------------------------------------------- #


def _to_level(values: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and round **half up**, exactly as `shared.io.to_uint8` does.

    This is the one line that makes the project's central claim true. A plain
    ``.astype(np.uint8)`` truncates, and the arithmetic path rounds, so a gamma
    table built by truncation disagreed with the same gamma written as numpy
    maths on **136 of 256 input levels** — every one of them by exactly one.

    Nothing looks wrong when that happens. The image is a level darker in half
    its tones, which is invisible, and a LUT that is "the same function" is
    silently a different one. Truncation also biases every curve downward by half
    a level on average, so a brightening gamma systematically under-brightens.
    """
    return (np.clip(values, 0.0, 255.0) + 0.5).astype(np.uint8)


def lut_identity() -> np.ndarray:
    return np.arange(256, dtype=np.uint8)


def lut_negative() -> np.ndarray:
    """``255 - v``. The only point transform here that is perfectly invertible.

    A bijection on 0-255, so it loses nothing at all — the reference point for
    every other curve's information loss.
    """
    return np.arange(255, -1, -1, dtype=np.uint8)


def lut_power(gamma: float = 0.5, c: float = 1.0) -> np.ndarray:
    """Power-law (gamma) curve: ``c * (v/255)^gamma * 255``.

    ``gamma < 1`` expands the shadows and compresses the highlights; ``gamma > 1``
    does the reverse.

    The usual line is that brightening duplicates levels and that is why it
    bands. Measured, it is the other way round: gamma 0.5 has a maximum collapse
    run of **2** and a maximum output gap of **16**, while gamma 2.2 has a run of
    **15** and a gap of **3**. Brightening bands because it *spreads* adjacent
    shadow levels apart, leaving holes in the output histogram — not because it
    merges them. Darkening merges them, which loses detail without banding.
    See `max_run_length` and `max_output_gap`.
    """
    v = np.arange(256, dtype=np.float64) / 255.0
    return _to_level(c * np.power(v, gamma) * 255.0)


def lut_log(c: float | None = None) -> np.ndarray:
    """Logarithmic: ``c * log(1 + v)``.

    Compresses a wide dynamic range into a displayable one — the reason it is the
    standard way to display a Fourier spectrum, where DC is millions of times
    larger than anything else.
    """
    v = np.arange(256, dtype=np.float64)
    scale = c if c is not None else 255.0 / np.log(256.0)
    return _to_level(scale * np.log1p(v))


def lut_inverse_log(c: float | None = None) -> np.ndarray:
    """The inverse of the log curve — expands where log compressed."""
    v = np.arange(256, dtype=np.float64)
    scale = c if c is not None else 255.0 / np.log(256.0)
    return _to_level(np.expm1(v / scale))


def lut_piecewise_linear(points=((0, 0), (70, 30), (180, 220), (255, 255))) -> np.ndarray:
    """Contrast stretching through control points.

    A steep segment expands that part of the range at the cost of flattening
    elsewhere — the total "steepness budget" is fixed, which is the clearest way
    to see that contrast enhancement always moves contrast rather than creating it.
    """
    xs = np.array([p[0] for p in points], np.float64)
    ys = np.array([p[1] for p in points], np.float64)
    return _to_level(np.interp(np.arange(256), xs, ys))


def lut_threshold(value: int = 127) -> np.ndarray:
    """A hard step. The maximum possible information loss: 256 levels to 2."""
    v = np.arange(256)
    return np.where(v > value, 255, 0).astype(np.uint8)


def lut_posterise(levels: int = 6) -> np.ndarray:
    """Quantise to ``levels`` values — a controllable amount of loss.

    The dial that makes the information-loss claim continuous: 256 levels down to
    2 in whatever steps you like.
    """
    v = np.arange(256, dtype=np.float64)
    step = 255.0 / max(levels - 1, 1)
    return _to_level(np.round(v / step) * step)


def lut_contrast_stretch(low: int = 30, high: int = 220) -> np.ndarray:
    """Linear stretch mapping ``[low, high]`` onto the full range, clipping outside.

    The clipping is the point: anything below ``low`` becomes 0 and is
    unrecoverable. It buys contrast by throwing away the ends.
    """
    v = np.arange(256, dtype=np.float64)
    out = (v - low) * 255.0 / max(high - low, 1)
    return _to_level(out)


LUTS: dict[str, Callable[[], np.ndarray]] = {
    "Identity (control)": lut_identity,
    "Negative": lut_negative,
    "Gamma 0.5 (brighten)": lambda: lut_power(0.5),
    "Gamma 2.2 (darken)": lambda: lut_power(2.2),
    "Log": lut_log,
    "Inverse log": lut_inverse_log,
    "Piecewise linear": lut_piecewise_linear,
    "Contrast stretch": lut_contrast_stretch,
    "Posterise (6 levels)": lambda: lut_posterise(6),
    "Threshold 127": lambda: lut_threshold(127),
}


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 256-entry table. One memory read per pixel, no arithmetic."""
    return cv2.LUT(img, lut)


def apply_arithmetic(img: np.ndarray, name: str) -> np.ndarray:
    """The same transforms written as per-pixel numpy maths.

    Kept deliberately separate so the two implementations can be compared for
    **bit-exact equality**, which is the project's correctness claim.
    """
    f = to_float(img)
    if name == "Identity (control)":
        return img.copy()
    if name == "Negative":
        return to_uint8(1.0 - f)
    if name == "Gamma 0.5 (brighten)":
        return to_uint8(np.power(f, 0.5))
    if name == "Gamma 2.2 (darken)":
        return to_uint8(np.power(f, 2.2))
    raise ValueError(f"no arithmetic version of {name!r}")


# --------------------------------------------------------------------------- #
# bit-plane slicing
# --------------------------------------------------------------------------- #


def bit_plane(img: np.ndarray, bit: int) -> np.ndarray:
    """Extract one bit plane as a binary image.

    The high planes hold the image's structure; the low planes are close to
    noise. Reconstructing from the top few planes shows how few bits actually
    carry the picture, which is the intuition behind bit-depth reduction and,
    ultimately, compression.
    """
    return ((to_gray(img) >> bit) & 1).astype(np.uint8) * 255


def reconstruct_from_planes(img: np.ndarray, planes: int = 4) -> np.ndarray:
    """Rebuild using only the top ``planes`` bits.

    The mask has to be taken modulo 256 before it becomes a uint8: ``0xFF << 4``
    is 4080, and ``np.uint8(4080)`` raises rather than wrapping on modern numpy.
    """
    gray = to_gray(img)
    mask = np.uint8((0xFF << (8 - planes)) & 0xFF)
    # numpy's `&`, not cv2.bitwise_and: OpenCV wants an array or a 4-tuple scalar
    # for its second argument and rejects a bare numpy scalar.
    return gray & mask


# --------------------------------------------------------------------------- #
# information loss, computed from the table alone
# --------------------------------------------------------------------------- #


def levels_surviving(lut: np.ndarray) -> int:
    """How many of the 256 input levels map to distinct outputs.

    Computed from the table with no image involved. This is the exact,
    image-independent statement of what the curve destroys.
    """
    return int(len(np.unique(lut)))


def is_invertible(lut: np.ndarray) -> bool:
    return levels_surviving(lut) == 256


def max_run_length(lut: np.ndarray) -> int:
    """Longest run of input levels collapsed onto a single output.

    This is **detail loss**: many distinct inputs becoming one output, which
    happens wherever the curve is flat. Derived from the table rather than
    observed in an image.

    It is not the same thing as banding, and conflating the two is easy — see
    `max_output_gap`.
    """
    best = run = 1
    for i in range(1, len(lut)):
        run = run + 1 if lut[i] == lut[i - 1] else 1
        best = max(best, run)
    return best


def max_output_gap(lut: np.ndarray) -> int:
    """Largest jump between the outputs of two **adjacent** input levels.

    This is **banding**, and it is a different failure from `max_run_length`.

    * A flat part of the curve maps many inputs to one output: detail is lost,
      and the result is smooth.
    * A steep part maps adjacent inputs far apart: no detail is lost at all, but
      the output histogram has holes in it, and a smooth gradient in the input
      becomes visible steps.

    The two are produced by opposite halves of the same curve, so a single
    "information loss" number cannot tell them apart. Gamma 0.5 has a maximum run
    of 2 and a maximum gap of 16; gamma 2.2 has a run of 21 and a gap of 1. The
    usual claim that *brightening* causes banding is about gaps, and the usual
    measure of loss counts runs.
    """
    return int(np.max(np.abs(np.diff(lut.astype(np.int16)))))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs selected by `tools/select_images.py --axis brightness`,
#: which is where a point transform's effect lives: the same gamma brightens a
#: dark frame and washes out a bright one, and a pool at one end of this axis
#: would report the pictures rather than the curves.
#:
#: The spread is 47 to 189 mean grey. `firewalkers_at_night` is flame against
#: near-black, where a brightening curve has the most to reveal and the most
#: quantisation to reveal it with; `aircrew_on_tarmac` is bright concrete, where
#: the same curve has nothing left to lift.
IMAGES = (
    "firewalkers_at_night",   # mean  47 - the darkest here
    "young_monks_crowding",   #       74
    "golden_pavilion",        #       85
    "clapboard_houses",       #       93
    "three_girls_by_hay",     #      100
    "market_fruit_stall",     #      102
    "bighorn_ram",            #      113
    "chipmunk_on_granite",    #      119
    "horses_in_a_meadow",     #      126
    "woman_by_a_wall",        #      134
    "monk_under_a_tree",      #      149
    "aircrew_on_tarmac",      #      189 - the brightest here
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs.

    Named so that `run.py`, the tests and `infer.py` all read the same pixels.
    """
    from shared import io

    return io.real_photo(name)


def mean_brightness(img: np.ndarray) -> float:
    """The axis the pool was selected on, recomputed here."""
    return float(to_gray(img).mean())
GAMMAS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.2, 3.0)
POSTERISE_LEVELS = (2, 4, 8, 16, 32, 64, 128, 256)


def table_properties():
    """Every curve's information loss, derived from the lookup table alone."""
    rows = []
    for name, builder in LUTS.items():
        lut = builder()
        rows.append(
            {
                "transform": name,
                "levels_surviving": levels_surviving(lut),
                "levels_lost": 256 - levels_surviving(lut),
                "invertible": is_invertible(lut),
                "max_collapse_run": max_run_length(lut),
                "max_output_gap": max_output_gap(lut),
            }
        )
    return rows


def evaluate_on_images(images=IMAGES, runs: int = 5):
    """Measured effect on real images, beside the predicted loss."""
    rows = []
    for name, builder in LUTS.items():
        lut = builder()
        ents, contrasts, psnrs, ms = [], [], [], []
        for image in images:
            img = to_gray(load_scene(image))
            out, timing = timeit(lambda a=img, t=lut: apply_lut(a, t), runs=runs, warmup=1)
            ents.append(entropy(out))
            contrasts.append(rms_contrast(out))
            psnrs.append(psnr(out, img))
            ms.append(timing.median_ms)
        rows.append(
            {
                "transform": name,
                "levels_surviving": levels_surviving(lut),
                "entropy_bits": round(float(np.mean(ents)), 4),
                "rms_contrast": round(float(np.mean(contrasts)), 4),
                "psnr_vs_original": round(float(np.mean(psnrs)), 3),
                "median_ms": round(float(np.median(ms)), 4),
            }
        )
    return rows


def lut_versus_arithmetic(images=IMAGES, runs: int = 7):
    """The correctness and speed claim, together.

    ``identical`` must be True for every row. If a LUT and its arithmetic
    equivalent ever disagree, one of them has a rounding bug — and since this is
    integer output, "close enough" is not a defence.
    """
    rows = []
    for name in ("Identity (control)", "Negative", "Gamma 0.5 (brighten)", "Gamma 2.2 (darken)"):
        lut = LUTS[name]()
        identical, lut_ms, arith_ms = True, [], []
        for image in images:
            img = to_gray(load_scene(image))
            a, t1 = timeit(lambda x=img, l=lut: apply_lut(x, l), runs=runs, warmup=2)
            b, t2 = timeit(lambda x=img, n=name: apply_arithmetic(x, n), runs=runs, warmup=2)
            identical &= bool(np.array_equal(a, b))
            lut_ms.append(t1.median_ms)
            arith_ms.append(t2.median_ms)
        lut_median = float(np.median(lut_ms))
        arith_median = float(np.median(arith_ms))
        rows.append(
            {
                "transform": name,
                "identical": identical,
                "lut_ms": round(lut_median, 4),
                "arithmetic_ms": round(arith_median, 4),
                "speedup": round(arith_median / max(lut_median, EPS), 1),
            }
        )
    return rows


def sweep_gamma(images=IMAGES, gammas=GAMMAS):
    """Gamma against surviving levels, entropy and brightness.

    Predicted loss and measured entropy should move together. Where they do not,
    it is because the image never used the levels that got collapsed — which is
    itself worth seeing.
    """
    rows = []
    for g in gammas:
        lut = lut_power(g)
        ents, brights = [], []
        for image in images:
            img = to_gray(load_scene(image))
            out = apply_lut(img, lut)
            ents.append(entropy(out))
            brights.append(float(np.mean(out)) / 255.0)
        rows.append(
            {
                "gamma": g,
                "levels_surviving": levels_surviving(lut),
                "max_collapse_run": max_run_length(lut),
                "max_output_gap": max_output_gap(lut),
                "entropy_bits": round(float(np.mean(ents)), 4),
                "mean_brightness": round(float(np.mean(brights)), 4),
            }
        )
    return rows


def sweep_posterise(images=IMAGES, levels=POSTERISE_LEVELS):
    """Quantisation from 256 levels down to 2, with entropy tracking the loss."""
    rows = []
    for n in levels:
        lut = lut_posterise(n)
        ents, psnrs = [], []
        for image in images:
            img = to_gray(load_scene(image))
            out = apply_lut(img, lut)
            ents.append(entropy(out))
            psnrs.append(psnr(out, img))
        rows.append(
            {
                "requested_levels": n,
                "actual_levels": levels_surviving(lut),
                "entropy_bits": round(float(np.mean(ents)), 4),
                "psnr_db": round(float(np.mean(psnrs)), 3),
            }
        )
    return rows


def bit_plane_contribution(images=IMAGES):
    """How much of the image lives in each bit.

    Reconstructing from the top N planes and measuring PSNR shows the top four
    bits carry almost everything, and the bottom two are close to noise — the
    empirical basis for every bit-depth reduction scheme.
    """
    rows = []
    for planes in range(1, 9):
        psnrs, ents = [], []
        for image in images:
            img = to_gray(load_scene(image))
            out = reconstruct_from_planes(img, planes)
            psnrs.append(psnr(out, img))
            ents.append(entropy(out))
        rows.append(
            {
                "top_planes_kept": planes,
                "psnr_db": round(float(np.mean(psnrs)), 3),
                "entropy_bits": round(float(np.mean(ents)), 4),
            }
        )
    return rows


def transform(img: np.ndarray, name: str = "Gamma 0.5 (brighten)") -> np.ndarray:
    return apply_lut(img, LUTS[name]())
