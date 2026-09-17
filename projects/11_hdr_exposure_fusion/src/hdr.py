"""HDR exposure fusion: combining a bracket, and what it cannot get back.

The question
------------
A single exposure cannot hold both a bright window and a dark room. Bracketing
takes several, and fusion combines them. Every write-up shows the result and
declares it better.

> **Better than what, by how much, and against what ceiling?**

That question needs a reference image, and a real bracketed set has none: nobody
recorded what the scene actually looked like. So the bracket here is **generated
from one photograph**, which is therefore the exact answer, and every method is
scored against it.

Two separable questions the same scene answers
----------------------------------------------
1. **Is fusing several frames better than taking one good one?** The middle
   exposure is kept as a control. If a method cannot beat "just take the
   photograph", the bracket bought nothing.
2. **How much of the scene did the bracket fail to record at all?** A pixel
   clipped to white in every frame, or buried under the noise floor in every
   frame, is gone. No fusion method recovers it, and that fraction is the
   ceiling — not an algorithm's failure.

Why tone mapping needs two columns
----------------------------------
Merging a bracket produces a *radiance map* in arbitrary units. Tone mapping
squeezes it back into 8 bits, and every operator makes its own choice about
overall brightness. Scored with raw PSNR that choice dominates the result, and
the table measures preference rather than recovery. So both are reported: PSNR
as-is, and PSNR after matching the output's mean to the reference. The gap
between the two columns is the tone mapper's brightness decision, and it belongs
to the operator rather than to the fusion.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim
from shared import synth

EPS = 1e-6


# --------------------------------------------------------------------------- #
# fusion methods
# --------------------------------------------------------------------------- #


def fuse_mertens(frames: list[np.ndarray], times: np.ndarray) -> np.ndarray:
    """Mertens exposure fusion — blend the LDR frames directly, no radiance map.

    Weights each pixel of each frame by three cues the paper calls contrast,
    saturation and well-exposedness, then blends over a Laplacian pyramid. It
    never estimates radiance and never needs the exposure times, which is why it
    is the method that works when the times are unknown or wrong.
    """
    # uint8 frames, NOT floats scaled to [0,1]. Handed 0-1 floats OpenCV's
    # Mertens returns an image with a maximum around 0.004 -- black, and no
    # error. It scored 6.5 dB and SSIM 0.006 that way, which reads as a broken
    # method rather than a wrong call.
    merged = cv2.createMergeMertens().process(list(frames))
    return to_uint8(np.clip(merged, 0, 1))


#: One-entry cache for the Debevec radiance map.
#:
#: Recovering the camera response and merging costs about 1.9 s, and the three
#: tone-mapping methods below differ ONLY in what they do afterwards. Without
#: this, a six-image run spends 35 s recomputing an identical radiance map three
#: times per image, and the timing column reports the calibration rather than
#: the operator being compared.
_RADIANCE_CACHE: dict = {}


def _debevec_radiance(frames: list[np.ndarray], times: np.ndarray) -> np.ndarray:
    """Recover a linear radiance map with Debevec & Malik's calibration."""
    key = (frames[0].tobytes()[:2048], len(frames), tuple(np.round(times, 6)))
    cached = _RADIANCE_CACHE.get(key)
    if cached is not None:
        return cached
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    response = cv2.createCalibrateDebevec().process(bgr, times=times.copy())
    hdr = cv2.createMergeDebevec().process(bgr, times=times.copy(), response=response)
    _RADIANCE_CACHE.clear()
    _RADIANCE_CACHE[key] = hdr
    return hdr


def _tonemap(hdr_bgr: np.ndarray, operator) -> np.ndarray:
    mapped = operator.process(hdr_bgr.copy())
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
    return cv2.cvtColor(to_uint8(np.clip(mapped, 0, 1)), cv2.COLOR_BGR2RGB)


def fuse_debevec_reinhard(frames, times) -> np.ndarray:
    """Debevec radiance, then Reinhard's photographic tone reproduction."""
    return _tonemap(_debevec_radiance(frames, times), cv2.createTonemapReinhard(gamma=2.2))


def fuse_debevec_drago(frames, times) -> np.ndarray:
    """Debevec radiance, then Drago's adaptive logarithmic mapping."""
    return _tonemap(_debevec_radiance(frames, times), cv2.createTonemapDrago(gamma=2.2))


def fuse_debevec_mantiuk(frames, times) -> np.ndarray:
    """Debevec radiance, then Mantiuk's contrast-domain operator."""
    return _tonemap(_debevec_radiance(frames, times), cv2.createTonemapMantiuk(gamma=2.2))


def fuse_mean(frames: list[np.ndarray], times: np.ndarray) -> np.ndarray:
    """Plain per-pixel average of the encoded frames — the naive control.

    Here because it is what "combine the exposures" means if you do not think
    about it, and because it is the clearest demonstration of why weighting
    matters: a clipped highlight contributes 255 to the average with exactly the
    same authority as a correctly exposed pixel.
    """
    return to_uint8(np.mean([to_float(f) for f in frames], axis=0))


def take_middle_frame(frames: list[np.ndarray], times: np.ndarray) -> np.ndarray:
    """Just use the reference exposure — the "did you need a bracket" control.

    Not a fusion method. It is the thing every fusion method has to beat before
    the extra frames, the extra shutter time and the alignment risk are worth
    anything at all.
    """
    return frames[len(frames) // 2]


METHODS: dict[str, Callable[[list[np.ndarray], np.ndarray], np.ndarray]] = {
    "Mertens fusion": fuse_mertens,
    "Debevec + Reinhard": fuse_debevec_reinhard,
    "Debevec + Drago": fuse_debevec_drago,
    "Debevec + Mantiuk": fuse_debevec_mantiuk,
    "Mean of frames (control)": fuse_mean,
    "Middle exposure only (control)": take_middle_frame,
}

#: Controls are excluded when the question is "which fusion method", and
#: included when the question is "was fusing worth it". Naming them once stops
#: the two questions being answered with the same list by accident.
CONTROLS = ("Mean of frames (control)", "Middle exposure only (control)")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def match_mean(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``img`` so its mean luminance matches ``reference``.

    A tone mapper chooses its own overall brightness, and raw PSNR scores that
    choice far more heavily than it scores the detail actually recovered. This
    removes the choice so the two can be reported separately.

    Bisected on the *clipped* result rather than solved in closed form, because
    scaling then clipping is not linear: the closed-form factor systematically
    overshoots whenever any pixel saturates.
    """
    target = float(to_gray(reference).mean())
    lo, hi = 0.05, 20.0
    f = to_float(img)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        got = float(to_gray(to_uint8(np.clip(f * mid, 0, 1))).mean())
        if got < target:
            lo = mid
        else:
            hi = mid
    return to_uint8(np.clip(f * 0.5 * (lo + hi), 0, 1))


def score(output: np.ndarray, reference: np.ndarray) -> dict:
    """PSNR as-is, PSNR after brightness matching, and SSIM."""
    return {
        "psnr_db": round(psnr(output, reference), 3),
        "psnr_matched_db": round(psnr(match_mean(output, reference), reference), 3),
        "ssim": round(ssim(output, reference), 4),
    }


def contribution_map(frames: list[np.ndarray]) -> np.ndarray:
    """Which frame is best exposed at each pixel, as an index image.

    This is the picture that makes a bracket legible: it shows the fusion
    choosing the dark frame for the window and the bright frame for the shadow,
    and it shows the regions where *no* frame was well exposed.

    "Best exposed" is the Mertens well-exposedness term on its own — a Gaussian
    centred on mid-grey — because that is the cue actually doing the work.
    """
    weights = []
    for f in frames:
        g = to_float(to_gray(f))
        weights.append(np.exp(-((g - 0.5) ** 2) / (2 * 0.2**2)))
    return np.argmax(np.stack(weights), axis=0).astype(np.int32)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Six photographs with different dynamic range. A bracket only helps where a
#: single frame cannot hold the scene, so a pool of six evenly lit images would
#: measure the one case where the answer is "it does not".
IMAGES = (
    "lake_shrine",
    "stone_arch",
    "harbour_boat",
    "windmills",
    "rocky_coast",
    "temple_dragon",
)


def load_scene(name: str) -> np.ndarray:
    from shared import io

    if name in io.REAL_PHOTOS:
        return io.real_photo(name)
    return io.sample(name)


def evaluate_methods(images=IMAGES, stops=synth.EXPOSURE_STOPS, runs: int = 1):
    """Score every method against the photograph the bracket was made from."""
    acc = {name: {"psnr": [], "matched": [], "ssim": [], "ms": []} for name in METHODS}
    single = {"psnr": [], "ssim": []}
    losses = []

    for name in images:
        frames, times, reference = synth.hdr_bracket(load_scene(name), stops=stops)
        losses.append(synth.bracket_loss(frames, stops)["unrecoverable"])
        mid = frames[len(frames) // 2]
        single["psnr"].append(psnr(mid, reference))
        single["ssim"].append(ssim(mid, reference))

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(frames, times), runs=runs, warmup=0)
            s = score(out, reference)
            acc[method]["psnr"].append(s["psnr_db"])
            acc[method]["matched"].append(s["psnr_matched_db"])
            acc[method]["ssim"].append(s["ssim"])
            acc[method]["ms"].append(timing.median_ms)

    rows = [
        {
            "method": name,
            "psnr_db": round(float(np.mean(v["psnr"])), 3),
            "psnr_matched_db": round(float(np.mean(v["matched"])), 3),
            "ssim": round(float(np.mean(v["ssim"])), 4),
            "median_ms": round(float(np.mean(v["ms"])), 3),
        }
        for name, v in acc.items()
    ]
    stats = {
        "middle_exposure_psnr": round(float(np.mean(single["psnr"])), 3),
        "middle_exposure_ssim": round(float(np.mean(single["ssim"])), 4),
        "mean_unrecoverable": round(float(np.mean(losses)), 6),
    }
    return rows, stats


#: Bracket sizes to sweep, as slices of the five-frame set. Answering "how many
#: frames do you actually need" rather than assuming five.
#:
#: One frame is deliberately absent. Debevec's method recovers the camera
#: response curve from how the SAME pixel changes across exposures, and with one
#: exposure there is no change to fit — OpenCV's Mantiuk operator fails outright
#: with `fabs(dprod) > 0` on the degenerate radiance map that results. The
#: one-frame case is not missing from this project: it is the
#: "Middle exposure only" control in the method table, which is exactly what a
#: one-frame bracket is.
BRACKET_SIZES = (2, 3, 5)


def sweep_bracket_size(images=IMAGES, sizes=BRACKET_SIZES):
    """Does a wider bracket keep paying?

    The subsets are centred on the reference exposure, so a 3-frame bracket is
    the middle three and not the darkest three — otherwise the sweep would
    measure exposure choice rather than bracket width.
    """
    full = list(synth.EXPOSURE_STOPS)
    rows = []
    for n in sizes:
        start = (len(full) - n) // 2
        stops = tuple(full[start : start + n])
        per_method = {m: [] for m in METHODS if m not in CONTROLS}
        loss = []
        for name in images:
            frames, times, reference = synth.hdr_bracket(load_scene(name), stops=stops)
            loss.append(synth.bracket_loss(frames, stops)["unrecoverable"])
            for method in per_method:
                per_method[method].append(psnr(METHODS[method](frames, times), reference))
        row = {"frames": n, "stops": list(stops),
               "unrecoverable": round(float(np.mean(loss)), 6)}
        row.update({m: round(float(np.mean(v)), 3) for m, v in per_method.items()})
        rows.append(row)
    return rows


def evaluate_clipping_ceiling(images=IMAGES):
    """What a bracket cannot record, as a function of how wide it is.

    Reported separately from the method table because it is not a property of
    any method: a pixel saturated in every frame is simply not in the data.
    """
    rows = []
    for spread in (0.0, 1.0, 2.0, 3.0, 4.0):
        stops = (-spread, 0.0, spread) if spread > 0 else (0.0,)
        lost = []
        for name in images:
            frames, _, _ = synth.hdr_bracket(load_scene(name), stops=stops)
            lost.append(synth.bracket_loss(frames, stops))
        rows.append(
            {
                "stop_spread": spread,
                "frames": len(stops),
                "blown_everywhere": round(float(np.mean([l["blown_everywhere"] for l in lost])), 6),
                "crushed_everywhere": round(
                    float(np.mean([l["crushed_everywhere"] for l in lost])), 6
                ),
                "unrecoverable": round(float(np.mean([l["unrecoverable"] for l in lost])), 6),
            }
        )
    return rows
