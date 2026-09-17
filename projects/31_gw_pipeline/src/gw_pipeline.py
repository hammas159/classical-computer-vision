"""The Gonzalez & Woods eight-stage enhancement pipeline, stage by stage.

The question
------------
The classic worked example from *Digital Image Processing* chains Laplacian
sharpening, Sobel gradient, smoothing, masking and a power-law transform into
eight steps that produce a well-known result on a bone scan.

It is presented as a recipe. Nobody measures the steps.

> **The claim under test:** not every stage earns its place. Removing one stage
> at a time and re-measuring gives each stage's individual contribution, and some
> of them should turn out to contribute nearly nothing — or to matter only
> because of what follows them.

This is an **ablation study**, which is a different shape of experiment from a
method comparison: one pipeline, N variants, each missing one piece.

Two extra questions the same machinery answers:

* **Does order matter?** Several stages are filters that nearly commute. Running
  a few permutations shows whether the prescribed order is load-bearing.
* **Is the Laplacian sign right?** Stage 2 adds the Laplacian to the original.
  Whether that sharpens or blurs depends on the kernel's centre sign — the
  classic trap, measured here rather than asserted.
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, rms_contrast, ssim

EPS = 1e-9

#: Positive-centre Laplacian. With this kernel the sharpened image is
#: ``original + laplacian``. With the negated kernel it is ``original -
#: laplacian``. Mixing them up blurs instead of sharpening, silently.
LAPLACIAN = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)


# --------------------------------------------------------------------------- #
# the eight stages
# --------------------------------------------------------------------------- #


def stage_a_original(img: np.ndarray) -> np.ndarray:
    """(a) The input. Kept as a named stage so the ablation indexing matches the book."""
    return to_gray(img).copy()


def stage_b_laplacian(gray: np.ndarray) -> np.ndarray:
    """(b) Laplacian of the original — the second derivative, scaled for display."""
    lap = cv2.filter2D(to_float(gray), cv2.CV_32F, LAPLACIAN)
    return to_uint8(cv2.normalize(lap, None, 0, 1, cv2.NORM_MINMAX))


def stage_c_sharpened(gray: np.ndarray, weight: float = 1.0, wrong_sign: bool = False):
    """(c) Original **plus** the Laplacian: the sharpened image.

    ``wrong_sign`` flips the kernel to demonstrate the failure mode. It produces
    a softened image that looks merely "a bit flat" rather than obviously broken,
    which is exactly why the bug survives review.
    """
    kernel = -LAPLACIAN if wrong_sign else LAPLACIAN
    lap = cv2.filter2D(to_float(gray), cv2.CV_32F, kernel)
    return to_uint8(to_float(gray) + weight * lap)


def stage_d_sobel(gray: np.ndarray) -> np.ndarray:
    """(d) Sobel gradient magnitude.

    The Laplacian responds strongly to noise because it is a second derivative;
    the Sobel gradient is a first derivative and is comparatively calm. That
    difference is the entire reason the pipeline computes both.
    """
    g = to_float(gray)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return to_uint8(cv2.normalize(np.sqrt(dx * dx + dy * dy), None, 0, 1, cv2.NORM_MINMAX))


def stage_e_smoothed_sobel(sobel: np.ndarray, ksize: int = 5) -> np.ndarray:
    """(e) Box-smoothed Sobel — this is the stage doing the quiet, essential work.

    Smoothing turns the gradient from a thin edge map into a *region mask* that
    marks where edges are. Used directly at stage (f) it would multiply the
    sharpened image by a one-pixel-wide ridge and destroy everything else.
    """
    return cv2.blur(sobel, (ksize, ksize))


def stage_f_mask(sharpened: np.ndarray, smoothed_sobel: np.ndarray) -> np.ndarray:
    """(f) Product of (c) and (e): sharpening gated by where the edges are.

    This is the pipeline's central idea — apply the noisy, aggressive Laplacian
    sharpening only where the calmer Sobel says there is real structure.
    """
    return to_uint8(to_float(sharpened) * to_float(smoothed_sobel))


def stage_g_sum(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(g) Original plus the masked sharpening."""
    return to_uint8(to_float(gray) + to_float(mask))


def stage_h_power_law(img: np.ndarray, gamma: float = 0.5, c: float = 1.0) -> np.ndarray:
    """(h) Power-law transform to expand the darker tones.

    Purely a tone curve: it adds no detail, it redistributes what is there. Its
    ablation should therefore change perceived brightness a great deal and
    structural similarity very little — a clean demonstration of the difference.
    """
    return to_uint8(c * np.power(to_float(img), gamma))


STAGE_NAMES = (
    "a_original",
    "b_laplacian",
    "c_sharpened",
    "d_sobel",
    "e_smoothed_sobel",
    "f_mask",
    "g_sum",
    "h_power_law",
)


def run_pipeline(
    img: np.ndarray,
    skip: str | None = None,
    gamma: float = 0.5,
    smooth_ksize: int = 5,
    wrong_sign: bool = False,
    order: tuple[str, ...] | None = None,
):
    """Run the full pipeline, optionally omitting one stage.

    Returns ``(final, stages)``. Skipping is implemented as a *pass-through*
    rather than a deletion, so the downstream stages still receive an input of
    the right kind and the ablation isolates that stage's contribution instead of
    breaking the chain.
    """
    if skip is not None and skip not in ABLATABLE:
        # A typo'd stage name used to be ignored, so the "ablation" silently ran
        # the full pipeline and reported it as a row -- a table of six identical
        # numbers that looks like every stage contributing nothing.
        raise ValueError(
            f"unknown stage {skip!r}; ablatable stages are {sorted(ABLATABLE)}"
        )

    stages: dict[str, np.ndarray] = {}

    a = stage_a_original(img)
    stages["a_original"] = a

    b = stage_b_laplacian(a)
    stages["b_laplacian"] = b

    c = a.copy() if skip == "c_sharpened" else stage_c_sharpened(a, wrong_sign=wrong_sign)
    stages["c_sharpened"] = c

    d = stage_d_sobel(a)
    stages["d_sobel"] = d

    # skipping (e) means using the raw gradient as the mask -- the specific
    # mistake this stage exists to prevent
    e = d.copy() if skip == "e_smoothed_sobel" else stage_e_smoothed_sobel(d, smooth_ksize)
    stages["e_smoothed_sobel"] = e

    f = c.copy() if skip == "f_mask" else stage_f_mask(c, e)
    stages["f_mask"] = f

    g = a.copy() if skip == "g_sum" else stage_g_sum(a, f)
    stages["g_sum"] = g

    h = g.copy() if skip == "h_power_law" else stage_h_power_law(g, gamma=gamma)
    stages["h_power_law"] = h

    return h, stages


# --------------------------------------------------------------------------- #
# what "better" means without a ground truth
# --------------------------------------------------------------------------- #


def acutance(img: np.ndarray) -> float:
    """Mean gradient magnitude — how sharp the image actually is."""
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def detail_visibility(img: np.ndarray, dark_percentile: float = 30.0) -> float:
    """Local contrast within the *darkest* regions.

    The pipeline's stated purpose is revealing detail hidden in shadow, so the
    honest measure is local contrast restricted to the dark areas — a global
    contrast number would be dominated by the bright regions the pipeline is not
    trying to fix.
    """
    g = to_gray(img)
    threshold = np.percentile(g, dark_percentile)
    dark = g <= threshold
    if not dark.any():
        return 0.0
    local = cv2.Laplacian(to_float(g), cv2.CV_32F, ksize=3)
    return float(np.std(local[dark]))


def measure(img: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    gray_ref = to_gray(reference)
    return {
        "acutance": acutance(img),
        "dark_detail": detail_visibility(img),
        "rms_contrast": rms_contrast(img),
        "ssim_vs_original": ssim(img, gray_ref),
        "psnr_vs_original": psnr(img, gray_ref),
    }


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs spanning **mean brightness** 36 to 110, capped low on
#: purpose. The book's worked example is a bone scan: low contrast with the
#: detail buried in the dark regions, which is exactly what this pipeline's
#: final power-law stage exists to lift. A pool of well-exposed photographs
#: would give it nothing to do. Selected by
#: `tools/select_images.py --axis brightness --max 110`.
IMAGES = (
    "diver_dark_reef",      # brightness  36 — detail buried in shadow
    "red_canoes",           # brightness  65
    "parasol_boat",         # brightness  73
    "model_red_black",      # brightness  80
    "sphinx_and_pyramid",   # brightness  85
    "elder_in_shawl",       # brightness  89
    "whitewashed_harbour",  # brightness  92
    "glass_tower_tulips",   # brightness  96
    "waterfall_cliff",      # brightness 101
    "cougar_and_kitten",    # brightness 104 — low contrast throughout
    "geisha_street",        # brightness 107
    "collared_lizard",      # brightness 110
)


def load_scene(name: str):
    from shared import io

    return io.real_photo(name)
GAMMAS = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
SMOOTH_SIZES = (1, 3, 5, 9, 15)
ABLATABLE = ("c_sharpened", "e_smoothed_sobel", "f_mask", "g_sum", "h_power_law")


def ablation(images=IMAGES, gamma: float = 0.5):
    """Remove one stage at a time and measure what changed.

    The full pipeline is the reference row. A stage whose removal barely moves
    any column was not contributing, and the pipeline could drop it.
    """
    from shared import io

    rows = []
    configs = [("Full pipeline", None)] + [(f"Without {s}", s) for s in ABLATABLE]

    for label, skip in configs:
        acc = {k: [] for k in ("acutance", "dark_detail", "rms_contrast",
                               "ssim_vs_original", "psnr_vs_original")}
        for name in images:
            img = load_scene(name)
            final, _ = run_pipeline(img, skip=skip, gamma=gamma)
            for k, v in measure(final, img).items():
                acc[k].append(v)
        rows.append(
            {
                "configuration": label,
                "acutance": round(float(np.mean(acc["acutance"])), 5),
                "dark_detail": round(float(np.mean(acc["dark_detail"])), 5),
                "rms_contrast": round(float(np.mean(acc["rms_contrast"])), 4),
                "ssim_vs_original": round(float(np.mean(acc["ssim_vs_original"])), 4),
                "psnr_vs_original": round(float(np.mean(acc["psnr_vs_original"])), 3),
            }
        )

    # express each ablation as a delta from the full pipeline, which is what
    # actually answers "what did this stage contribute"
    full = rows[0]
    for r in rows[1:]:
        r["acutance_delta"] = round(r["acutance"] - full["acutance"], 5)
        r["dark_detail_delta"] = round(r["dark_detail"] - full["dark_detail"], 5)
    return rows


def laplacian_sign_test(images=IMAGES):
    """Measure the sign trap instead of warning about it.

    Correct sign should raise acutance above the original; the wrong sign should
    lower it. Both against the same input, so the comparison is exact.
    """
    from shared import io

    rows = []
    for label, wrong in (("Correct sign (add +8 centre)", False), ("Wrong sign (add -8 centre)", True)):
        acut, ssims = [], []
        for name in images:
            img = load_scene(name)
            gray = to_gray(img)
            out = stage_c_sharpened(gray, wrong_sign=wrong)
            acut.append(acutance(out))
            ssims.append(ssim(out, gray))
        rows.append(
            {
                "configuration": label,
                "acutance": round(float(np.mean(acut)), 5),
                "ssim_vs_original": round(float(np.mean(ssims)), 4),
            }
        )

    baseline = []
    for name in images:
        baseline.append(acutance(to_gray(load_scene(name))))
    rows.insert(
        0,
        {
            "configuration": "Original (no sharpening)",
            "acutance": round(float(np.mean(baseline)), 5),
            "ssim_vs_original": 1.0,
        },
    )
    return rows


def sweep_gamma(images=IMAGES, gammas=GAMMAS):
    """Stage (h) is a tone curve: it should move brightness a lot, structure little."""
    from shared import io

    rows = []
    for g in gammas:
        acc = {"dark_detail": [], "rms_contrast": [], "ssim_vs_original": []}
        for name in images:
            img = load_scene(name)
            final, _ = run_pipeline(img, gamma=g)
            m = measure(final, img)
            for k in acc:
                acc[k].append(m[k])
        rows.append(
            {
                "gamma": g,
                "dark_detail": round(float(np.mean(acc["dark_detail"])), 5),
                "rms_contrast": round(float(np.mean(acc["rms_contrast"])), 4),
                "ssim_vs_original": round(float(np.mean(acc["ssim_vs_original"])), 4),
            }
        )
    return rows


def sweep_smoothing(images=IMAGES, sizes=SMOOTH_SIZES):
    """Stage (e)'s kernel size. At size 1 the mask is the raw gradient — the
    failure this stage exists to prevent, measured rather than described."""
    from shared import io

    rows = []
    for k in sizes:
        acc = {"acutance": [], "dark_detail": [], "ssim_vs_original": []}
        for name in images:
            img = load_scene(name)
            final, _ = run_pipeline(img, smooth_ksize=k)
            m = measure(final, img)
            for key in acc:
                acc[key].append(m[key])
        rows.append(
            {
                "smooth_ksize": k,
                "acutance": round(float(np.mean(acc["acutance"])), 5),
                "dark_detail": round(float(np.mean(acc["dark_detail"])), 5),
                "ssim_vs_original": round(float(np.mean(acc["ssim_vs_original"])), 4),
            }
        )
    return rows


#: The eight-stage pipeline and the one-line alternatives it is supposed to
#: beat, in one place so the figures, the tables and `infer.py` all measure
#: exactly the same thing.
VARIANTS = {
    "G&W 8-stage pipeline": lambda a: run_pipeline(a)[0],
    "CLAHE only": lambda a: cv2.createCLAHE(3.0, (8, 8)).apply(to_gray(a)),
    "Unsharp mask only": lambda a: to_uint8(
        to_float(to_gray(a))
        + 1.0 * (to_float(to_gray(a)) - cv2.GaussianBlur(to_float(to_gray(a)), (0, 0), 1.5))
    ),
    "Gamma 0.5 only": lambda a: stage_h_power_law(to_gray(a), 0.5),
    "Original (control)": lambda a: to_gray(a),
}


def apply_variant(img: np.ndarray, method: str = "G&W 8-stage pipeline"):
    """Run one of `VARIANTS` by name."""
    if method not in VARIANTS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[method](img)


def compare_to_simple_alternatives(images=IMAGES):
    """Eight stages against one-line alternatives.

    The uncomfortable question every pipeline should be asked: does CLAHE, in a
    single call, get most of the way there? If it does, that is worth knowing.
    """
    rows = []
    for label, fn in VARIANTS.items():
        acc = {k: [] for k in ("acutance", "dark_detail", "rms_contrast", "ssim_vs_original")}
        for name in images:
            img = load_scene(name)
            out = fn(img)
            m = measure(out, img)
            for k in acc:
                acc[k].append(m[k])
        rows.append(
            {
                "method": label,
                "acutance": round(float(np.mean(acc["acutance"])), 5),
                "dark_detail": round(float(np.mean(acc["dark_detail"])), 5),
                "rms_contrast": round(float(np.mean(acc["rms_contrast"])), 4),
                "ssim_vs_original": round(float(np.mean(acc["ssim_vs_original"])), 4),
            }
        )
    return rows


def enhance(img: np.ndarray, gamma: float = 0.5):
    return run_pipeline(img, gamma=gamma)
