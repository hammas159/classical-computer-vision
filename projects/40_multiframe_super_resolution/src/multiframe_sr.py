"""Multi-frame super-resolution: the honest contrast with project 21.

The question
------------
Project 21 showed that single-image interpolation cannot add information — every
method plateaus within a fraction of a dB, because they are all weighted averages
of the same surviving samples.

> **The claim under test:** multiple frames with **sub-pixel offsets** genuinely
> break that plateau, because each frame samples the scene at different positions
> and the union of those samples contains information no single frame has.

That is the whole difference, and it is measurable: the PSNR gain over the best
single-image method, as a function of how many frames are available.

The conditions under which it fails are just as important:

* **Integer-pixel shifts add nothing.** Every frame samples the same grid
  positions, so averaging them reduces noise and recovers no detail. Sweeping
  from integer to sub-pixel offsets shows the effect appearing.
* **Registration error destroys the gain.** The frames must be aligned to a
  fraction of a pixel. Injecting known registration error finds the point where
  the reconstruction becomes worse than doing nothing.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim

EPS = 1e-9


# --------------------------------------------------------------------------- #
# generating the low-resolution stack
# --------------------------------------------------------------------------- #


def make_stack(
    hr: np.ndarray, n_frames: int = 8, scale: int = 3, subpixel: bool = True,
    noise_sigma: float = 2.0, seed: int = 0,
):
    """Shift, blur, decimate — a stack of low-resolution views with known offsets.

    The order matters and is the physically correct one: the shift happens in the
    high-resolution world (the camera moved), *then* the optics blur, *then* the
    sensor samples. Shifting after decimation would only move whole
    low-resolution pixels around and could never produce new sample positions.

    Returns ``(frames, offsets)`` with offsets in **high-resolution pixels**.
    """
    rng = np.random.default_rng(seed)
    from shared import synth

    h, w = hr.shape[:2]
    frames, offsets = [], []
    for i in range(n_frames):
        if i == 0:
            dx, dy = 0.0, 0.0
        elif subpixel:
            dx, dy = rng.uniform(-scale, scale, 2)
        else:
            dx, dy = (rng.integers(-2, 3) * scale, rng.integers(-2, 3) * scale)

        m = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(hr, m, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT)
        lr = synth.downsample_for_sr(shifted, scale=scale)
        if noise_sigma > 0:
            lr = synth.gaussian_noise(lr, sigma=noise_sigma, seed=seed * 100 + i)
        frames.append(lr)
        offsets.append((float(dx), float(dy)))
    return frames, offsets


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def register_phase_correlation(reference: np.ndarray, frame: np.ndarray, scale: int = 1):
    """Sub-pixel shift estimate via phase correlation.

    🚨 A Hanning window is mandatory. Without it the image borders act as a huge
    step discontinuity, whose spectrum swamps the correlation peak — the single
    most common reason a phase-correlation implementation "doesn't work".
    """
    a = to_float(to_gray(reference))
    b = to_float(to_gray(frame))
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a * window, b * window)
    return float(dx) * scale, float(dy) * scale


def register_ecc(reference: np.ndarray, frame: np.ndarray, scale: int = 1):
    """Enhanced Correlation Coefficient alignment — iterative and sub-pixel.

    Optimises a similarity measure invariant to brightness and contrast, so it
    tolerates exposure differences between frames that a plain correlation would
    not.

    **The sign is not negated, and getting that wrong is silent.** With
    ``templateImage=reference`` and ``inputImage=frame``, the translation that
    comes back is the one that takes the reference *to* the frame — which is the
    frame's own offset, the quantity wanted here. Negating it returns a
    displacement of exactly the right size in exactly the wrong direction, and
    the only symptom is a registration error roughly twice the true offset: this
    project reported ECC as 11x worse than phase correlation until the sign was
    checked against known offsets rather than against intuition.
    """
    a = to_float(to_gray(reference))
    b = to_float(to_gray(frame))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            a, b, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6), None, 5,
        )
    except cv2.error:
        return 0.0, 0.0
    return float(warp[0, 2]) * scale, float(warp[1, 2]) * scale


REGISTRATION: dict[str, Callable] = {
    "Phase correlation": register_phase_correlation,
    "ECC": register_ecc,
}


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #


def sr_single_frame(frames, offsets, scale: int = 3, **kwargs):
    """Bicubic upsampling of the first frame — the single-image baseline.

    This is the line multi-frame has to beat, and it is the same operation
    project 21 found could not be improved on by any other interpolator.
    """
    lr = frames[0]
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def sr_average(frames, offsets, scale: int = 3, **kwargs):
    """Upsample and average, ignoring the offsets.

    The control that isolates **denoising** from **resolution**. It should beat
    the single frame purely because averaging N frames cuts noise by sqrt(N) —
    and it should recover no extra detail at all, because misaligned frames blur
    each other.
    """
    ups = [
        cv2.resize(f, (f.shape[1] * scale, f.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        for f in frames
    ]
    return to_uint8(np.mean([to_float(u) for u in ups], axis=0))


def sr_shift_and_add(frames, offsets, scale: int = 3, registered=None, **kwargs):
    """Place every frame's samples on the high-resolution grid, then normalise.

    The core multi-frame idea. Each low-resolution pixel is a sample of the scene
    at a known sub-pixel position; scattering them onto a finer grid and dividing
    by the per-cell count fills in positions no single frame observed.

    Cells that no frame happened to sample are filled by interpolation afterwards
    — with few frames those gaps are the dominant error, which is why the gain
    grows with frame count rather than appearing all at once.
    """
    shifts = registered if registered is not None else offsets
    lr_h, lr_w = frames[0].shape[:2]
    hr_h, hr_w = lr_h * scale, lr_w * scale

    channels = 3 if frames[0].ndim == 3 else 1
    accum = np.zeros((hr_h, hr_w, channels), np.float64)
    counts = np.zeros((hr_h, hr_w, 1), np.float64)

    for frame, (dx, dy) in zip(frames, shifts):
        f = to_float(frame).reshape(lr_h, lr_w, channels)
        ys, xs = np.mgrid[0:lr_h, 0:lr_w]
        # the frame was shifted by (dx, dy) before sampling, so its sample k
        # corresponds to high-resolution position k*scale - (dx, dy)
        hx = np.round(xs * scale - dx).astype(np.int64)
        hy = np.round(ys * scale - dy).astype(np.int64)
        valid = (hx >= 0) & (hx < hr_w) & (hy >= 0) & (hy < hr_h)
        np.add.at(accum, (hy[valid], hx[valid]), f[valid])
        np.add.at(counts, (hy[valid], hx[valid]), 1.0)

    filled = counts[..., 0] > 0
    out = np.zeros_like(accum)
    out[filled] = accum[filled] / counts[filled]

    # fill unsampled cells from the bicubic upsample of the reference frame
    fallback = to_float(
        cv2.resize(frames[0], (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    ).reshape(hr_h, hr_w, channels)
    out[~filled] = fallback[~filled]

    result = to_uint8(out if channels == 3 else out[..., 0])
    return cv2.GaussianBlur(result, (0, 0), 0.5, borderType=cv2.BORDER_REFLECT)


def sr_iterative_back_projection(
    frames, offsets, scale: int = 3, iterations: int = 15, registered=None, **kwargs
):
    """Refine an estimate by enforcing consistency with every observed frame.

    Start from shift-and-add, then repeatedly: simulate each low-resolution frame
    from the current estimate, compare with what was actually captured, and push
    the residuals back. It is the multi-frame analogue of the back-projection in
    project 21 — but with N constraints instead of one, which is exactly why it
    can do more here.
    """
    from shared import synth

    shifts = registered if registered is not None else offsets
    estimate = to_float(sr_shift_and_add(frames, offsets, scale, registered=registered))
    hr_h, hr_w = estimate.shape[:2]

    for _ in range(iterations):
        total = np.zeros_like(estimate)
        for frame, (dx, dy) in zip(frames, shifts):
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                to_uint8(estimate), m, (hr_w, hr_h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
            )
            simulated = synth.downsample_for_sr(shifted, scale=scale)
            h = min(simulated.shape[0], frame.shape[0])
            w = min(simulated.shape[1], frame.shape[1])
            residual = to_float(frame[:h, :w]) - to_float(simulated[:h, :w])

            up = cv2.resize(residual, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
            back = np.float32([[1, 0, -dx], [0, 1, -dy]])
            up = cv2.warpAffine(up, back, (hr_w, hr_h),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            total += up.reshape(estimate.shape)

        estimate = np.clip(estimate + (0.4 / len(frames)) * total, 0.0, 1.0)
    return to_uint8(estimate)


METHODS: dict[str, Callable] = {
    "Single frame (bicubic)": sr_single_frame,
    "Naive average": sr_average,
    "Shift-and-add": sr_shift_and_add,
    "Iterative back-projection": sr_iterative_back_projection,
}


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs selected by `tools/select_images.py --axis detail`, then
#: ordered by this module's own `detail()` — Laplacian variance of the greyscale
#: image. The two do not agree in scale (the selector measures its own way), so
#: the numbers quoted here are the ones `detail()` returns, which is what the
#: README and `infer.py` report.
#:
#: Detail is the axis that decides whether super-resolution has anything to
#: recover: a smooth scene has no high-frequency content for extra samples to
#: reveal, so a pool at one end of this axis would report the pictures rather
#: than the methods. The spread is 194 to 6048, a factor of 31.
IMAGES = (
    "ladybird_on_a_leaf",     # detail  194 - a beetle on a smooth leaf, the control
    "pintail_at_dusk",        #         266 - a duck on still water
    "carved_stone_relief",    #         665
    "swallow_tailed_gulls",   #         673
    "toadstool_in_moss",      #        1150
    "three_schoolchildren",   #        1497
    "skier_on_a_slope",       #        2112
    "deer_in_bare_woods",     #        2181
    "otters_on_gravel",       #        2726
    "two_at_a_wagon",         #        2754
    "foxes_under_a_ledge",    #        4129
    "mayan_stone_carving",    #        6048 - dense carved relief
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs, used as the high-resolution truth.

    Named so that `run.py`, the tests and `infer.py` all read the same pixels —
    the low-resolution frames are generated from these, so the ground truth
    depends on them.
    """
    from shared import io

    return io.real_photo(name)


def detail(img: np.ndarray) -> float:
    """Laplacian variance — the axis the pool was selected on.

    Recomputed here so the README's numbers come from the project rather than
    from the selection tool, and so `infer.py` can say in advance whether a
    photograph has anything for extra frames to recover.
    """
    return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())
FRAME_COUNTS = (1, 2, 4, 8, 16, 32)
REGISTRATION_ERRORS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
SCALES = (2, 3, 4)


def _prepare(image: str, scale: int):
    hr = load_scene(image)
    h = (hr.shape[0] // scale) * scale
    w = (hr.shape[1] // scale) * scale
    return hr[:h, :w]


def evaluate_methods(
    n_frames: int = 8, scale: int = 3, images=IMAGES, subpixel: bool = True,
    noise_sigma: float = 2.0, runs: int = 1,
):
    """Score every reconstruction against the true high-resolution original."""
    acc = {n: {"psnr": [], "ssim": [], "ms": []} for n in METHODS}
    for i, image in enumerate(images):
        hr = _prepare(image, scale)
        frames, offsets = make_stack(
            hr, n_frames=n_frames, scale=scale, subpixel=subpixel,
            noise_sigma=noise_sigma, seed=i,
        )
        for name, fn in METHODS.items():
            out, timing = timeit(
                lambda f=fn: f(frames, offsets, scale=scale), runs=runs, warmup=0
            )
            out = out[: hr.shape[0], : hr.shape[1]]
            acc[name]["psnr"].append(psnr(out, hr))
            acc[name]["ssim"].append(ssim(out, hr))
            acc[name]["ms"].append(timing.median_ms)

    rows = [
        {
            "method": n,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for n, a in acc.items()
    ]
    baseline = next(r["psnr_db"] for r in rows if r["method"] == "Single frame (bicubic)")
    for r in rows:
        r["gain_over_single_db"] = round(r["psnr_db"] - baseline, 3)
    return rows


def sweep_frame_count(counts=FRAME_COUNTS, scale: int = 3, images=IMAGES):
    """The headline curve: how much does each extra frame buy?

    Expected to rise steeply at first and then saturate — once the
    high-resolution grid is densely sampled, more frames only reduce noise.
    """
    rows = []
    for n in counts:
        scored = evaluate_methods(n_frames=n, scale=scale, images=images)
        row: dict[str, float | int] = {"frames": n}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def subpixel_vs_integer(scale: int = 3, images=IMAGES, n_frames: int = 8):
    """The control that proves *sub-pixel* offsets are what matter.

    With integer-pixel shifts every frame samples the same grid positions, so
    multi-frame reconstruction should collapse to the naive average and the gain
    should vanish. If it does not, the effect being measured is not what it
    claims to be.
    """
    rows = []
    for label, sub in (("Sub-pixel offsets", True), ("Integer-pixel offsets", False)):
        scored = evaluate_methods(n_frames=n_frames, scale=scale, images=images, subpixel=sub)
        row: dict[str, float | str] = {"offsets": label}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def sweep_registration_error(
    errors=REGISTRATION_ERRORS, scale: int = 3, images=IMAGES, n_frames: int = 8
):
    """Where alignment error destroys the benefit.

    Known noise is added to the true offsets, so the reconstruction is given
    *deliberately* wrong alignment. The crossing point where multi-frame drops
    below the single-frame baseline is the practical tolerance.
    """
    rng = np.random.default_rng(0)
    rows = []
    for err in errors:
        per = {n: [] for n in METHODS}
        for i, image in enumerate(images):
            hr = _prepare(image, scale)
            frames, offsets = make_stack(hr, n_frames=n_frames, scale=scale, seed=i)
            noisy = [(dx + rng.normal(0, err), dy + rng.normal(0, err)) for dx, dy in offsets]
            for name, fn in METHODS.items():
                out = fn(frames, offsets, scale=scale, registered=noisy)
                per[name].append(psnr(out[: hr.shape[0], : hr.shape[1]], hr))
        row: dict[str, float] = {"registration_error_px": err}
        for name, vals in per.items():
            row[name] = round(float(np.mean(vals)), 3)
        rows.append(row)
    return rows


def evaluate_registration(scale: int = 3, images=IMAGES, n_frames: int = 8):
    """Can the offsets be recovered from the frames themselves?

    Everything above assumes the shifts are known. In practice they must be
    estimated, and the error of that estimate feeds straight into the sweep
    above — so this measures the input to that tolerance.
    """
    rows = []
    for name, fn in REGISTRATION.items():
        errors = []
        for i, image in enumerate(images):
            hr = _prepare(image, scale)
            frames, offsets = make_stack(hr, n_frames=n_frames, scale=scale, seed=i)
            for frame, (dx, dy) in zip(frames[1:], offsets[1:]):
                ex, ey = fn(frames[0], frame, scale=scale)
                errors.append(float(np.hypot(ex - dx, ey - dy)))
        rows.append(
            {
                "method": name,
                "mean_offset_error_px": round(float(np.mean(errors)), 4),
                "median_offset_error_px": round(float(np.median(errors)), 4),
            }
        )
    return rows


def sweep_scale(scales=SCALES, images=IMAGES, n_frames: int = 16):
    """How far can multi-frame push the scale factor before it too plateaus?"""
    rows = []
    for s in scales:
        scored = evaluate_methods(n_frames=n_frames, scale=s, images=images)
        best = max(scored, key=lambda r: r["psnr_db"])
        single = next(r for r in scored if r["method"] == "Single frame (bicubic)")
        rows.append(
            {
                "scale": s,
                "single_frame_db": single["psnr_db"],
                "best_method": best["method"],
                "best_db": best["psnr_db"],
                "gain_db": round(best["psnr_db"] - single["psnr_db"], 3),
            }
        )
    return rows


def reconstruct(frames, offsets, method: str = "Iterative back-projection", scale: int = 3):
    return METHODS[method](frames, offsets, scale=scale)
