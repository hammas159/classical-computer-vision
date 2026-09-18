"""Demosaicing and the camera ISP: reconstructing colour from a Bayer mosaic.

The question
------------
A camera sensor has one colour filter per photosite. Two thirds of the colour
data at every pixel is **never captured** — it is interpolated. That is the
largest single act of invention in the entire imaging pipeline and it happens
before anyone sees the picture.

> **The claim under test:** demosaicing error is not spread evenly. It
> concentrates on **high-frequency, high-saturation edges**, where neighbouring
> photosites of the same colour are far apart in the scene. Measuring PSNR over
> the whole image hides this almost completely; measuring it on edge pixels
> alone shows the real failure.

The mosaic is created here from a full-colour original, so the ground truth is
the original itself — a rare case where the "before" image is exactly what the
algorithm is trying to recover.

The Bayer pattern has **twice as many green photosites** as red or blue, because
the eye's luminance response peaks in green. Every good demosaicing algorithm
exploits that asymmetry, which is why bilinear interpolation — which does not —
is the weakest method here.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim

EPS = 1e-9

#: Bayer pattern layouts. The name gives the top-left 2x2 arrangement.
PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


# --------------------------------------------------------------------------- #
# mosaicing — building the sensor data
# --------------------------------------------------------------------------- #


def mosaic(img: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Discard two of the three colour samples at every pixel.

    This is what the sensor physically does. The result is a single-channel image
    where the meaning of each value depends on its position.
    """
    h, w = img.shape[:2]
    out = np.zeros((h, w), np.uint8)
    idx = {"R": 0, "G": 1, "B": 2}
    layout = {
        "RGGB": (("R", "G"), ("G", "B")),
        "BGGR": (("B", "G"), ("G", "R")),
        "GRBG": (("G", "R"), ("B", "G")),
        "GBRG": (("G", "B"), ("R", "G")),
    }[pattern]

    for dy in (0, 1):
        for dx in (0, 1):
            channel = idx[layout[dy][dx]]
            out[dy::2, dx::2] = img[dy::2, dx::2, channel]
    return out


def channel_masks(shape, pattern: str = "RGGB"):
    """Boolean masks for which pixels carry which colour."""
    h, w = shape[:2]
    masks = {c: np.zeros((h, w), bool) for c in "RGB"}
    layout = {
        "RGGB": (("R", "G"), ("G", "B")),
        "BGGR": (("B", "G"), ("G", "R")),
        "GRBG": (("G", "R"), ("B", "G")),
        "GBRG": (("G", "B"), ("R", "G")),
    }[pattern]
    for dy in (0, 1):
        for dx in (0, 1):
            masks[layout[dy][dx]][dy::2, dx::2] = True
    return masks


# --------------------------------------------------------------------------- #
# demosaicing methods
# --------------------------------------------------------------------------- #


def demosaic_nearest(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Replicate the nearest same-colour sample. The floor.

    Produces obvious colour fringing on every edge, and is here to establish what
    "no interpolation at all" costs.
    """
    masks = channel_masks(raw.shape, pattern)
    out = np.zeros((*raw.shape, 3), np.uint8)
    for i, c in enumerate("RGB"):
        m = masks[c].astype(np.uint8)
        known = raw * m
        # dilate the known samples outward to fill the gaps
        filled = cv2.dilate(known, np.ones((3, 3), np.uint8))
        out[..., i] = np.where(masks[c], raw, filled)
    return out


def demosaic_bilinear(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Average the available same-colour neighbours.

    Correct on flat regions and wrong on edges, because it interpolates each
    channel **independently** — so R, G and B edges end up in slightly different
    places and the result is a coloured fringe. Every better method exists to fix
    exactly that.
    """
    masks = channel_masks(raw.shape, pattern)
    f = to_float(raw)
    out = np.zeros((*raw.shape, 3), np.float32)

    kernels = {
        "R": np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32),
        "B": np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32),
        "G": np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], np.float32),
    }
    for i, c in enumerate("RGB"):
        known = f * masks[c]
        out[..., i] = cv2.filter2D(known, -1, kernels[c], borderType=cv2.BORDER_REFLECT)
    return to_uint8(np.clip(out, 0, 1))


def demosaic_opencv(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """OpenCV's bilinear demosaic.

    🚨 OpenCV's ``COLOR_BayerXX2RGB`` constants are named for the **second row,
    second column** of the pattern, not the first. ``RGGB`` therefore maps to
    ``COLOR_BayerBG2RGB``, which reads like a bug and is not. Getting it wrong
    swaps red and blue and looks like a colour-space error somewhere else
    entirely.
    """
    codes = {
        "RGGB": cv2.COLOR_BayerBG2RGB,
        "BGGR": cv2.COLOR_BayerRG2RGB,
        "GRBG": cv2.COLOR_BayerGB2RGB,
        "GBRG": cv2.COLOR_BayerGR2RGB,
    }
    return cv2.cvtColor(raw, codes[pattern])


def demosaic_opencv_vng(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Variable Number of Gradients — interpolate along edges, not across them.

    Examines gradients in eight directions and averages only along the directions
    where the image is smooth. This is the first method here that is *edge aware*,
    so it should show its advantage precisely in the edge-only metric.
    """
    codes = {
        "RGGB": cv2.COLOR_BayerBG2RGB_VNG,
        "BGGR": cv2.COLOR_BayerRG2RGB_VNG,
        "GRBG": cv2.COLOR_BayerGB2RGB_VNG,
        "GBRG": cv2.COLOR_BayerGR2RGB_VNG,
    }
    return cv2.cvtColor(raw, codes[pattern])


def demosaic_opencv_ea(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Edge-aware demosaicing."""
    codes = {
        "RGGB": cv2.COLOR_BayerBG2RGB_EA,
        "BGGR": cv2.COLOR_BayerRG2RGB_EA,
        "GRBG": cv2.COLOR_BayerGB2RGB_EA,
        "GBRG": cv2.COLOR_BayerGR2RGB_EA,
    }
    return cv2.cvtColor(raw, codes[pattern])


def demosaic_malvar(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Malvar-He-Cutler: linear interpolation with **cross-channel correction**.

    The key idea, and the reason it beats bilinear by several dB for the same
    cost: when interpolating red at a green site, it uses the *green* gradient at
    that site as a correction term. Colour channels in natural images are highly
    correlated, so green — which is sampled twice as densely — carries usable
    information about where red's edges are.

    Still a single linear filter. All of the gain comes from the insight, not
    from extra computation.
    """
    masks = channel_masks(raw.shape, pattern)
    f = to_float(raw)

    out = demosaic_bilinear(raw, pattern).astype(np.float32) / 255.0

    # gradient-correction kernels from the original paper, scaled by 1/8
    g_at_rb = np.array(
        [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]],
        np.float32,
    ) / 8.0
    rb_at_g_row = np.array(
        [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]], np.float32
    ) / 8.0
    rb_at_g_col = rb_at_g_row.T
    rb_at_br = np.array(
        [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]], np.float32
    ) / 8.0

    green_est = cv2.filter2D(f, -1, g_at_rb, borderType=cv2.BORDER_REFLECT)
    rb_row = cv2.filter2D(f, -1, rb_at_g_row, borderType=cv2.BORDER_REFLECT)
    rb_col = cv2.filter2D(f, -1, rb_at_g_col, borderType=cv2.BORDER_REFLECT)
    rb_diag = cv2.filter2D(f, -1, rb_at_br, borderType=cv2.BORDER_REFLECT)

    non_green = masks["R"] | masks["B"]
    out[..., 1][non_green] = green_est[non_green]

    # red at green sites: use the row or column form depending on which
    # neighbours are red, and the diagonal form at blue sites
    for target, other in (("R", "B"), ("B", "R")):
        ti = "RGB".index(target)
        at_green = masks["G"]
        same_row = np.zeros_like(at_green)
        same_row[:, :] = False
        # a green site shares its row with `target` if the pixel left/right is target
        left = np.roll(masks[target], 1, axis=1)
        same_row = at_green & (left | np.roll(masks[target], -1, axis=1))
        same_col = at_green & ~same_row

        out[..., ti][same_row] = rb_row[same_row]
        out[..., ti][same_col] = rb_col[same_col]
        out[..., ti][masks[other]] = rb_diag[masks[other]]

    return to_uint8(np.clip(out, 0, 1))


METHODS: dict[str, Callable] = {
    "Nearest neighbour": demosaic_nearest,
    "Bilinear (own)": demosaic_bilinear,
    "Bilinear (OpenCV)": demosaic_opencv,
    "Malvar (cross-channel)": demosaic_malvar,
    "VNG (gradient)": demosaic_opencv_vng,
    "Edge-aware": demosaic_opencv_ea,
}


# --------------------------------------------------------------------------- #
# the rest of the ISP
# --------------------------------------------------------------------------- #


def isp_pipeline(raw: np.ndarray, pattern: str = "RGGB", method: str = "Malvar (cross-channel)",
                 white_balance: bool = True, denoise: bool = True, gamma: float = 1 / 2.2):
    """A minimal image signal processor: demosaic, white balance, denoise, gamma.

    The order is the one real cameras use and it is not arbitrary. Demosaicing
    first because everything downstream needs three channels; white balance
    before gamma because the gains are physically multiplicative in **linear**
    light and applying them after a tone curve is mathematically wrong.
    """
    out = METHODS[method](raw, pattern)

    if white_balance:
        f = to_float(out)
        means = f.reshape(-1, 3).mean(axis=0)
        out = to_uint8(f * (float(means.mean()) / np.maximum(means, EPS)))

    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

    if gamma and gamma != 1.0:
        out = to_uint8(np.power(to_float(out), gamma))
    return out


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def edge_mask(img: np.ndarray, dilate: int = 3) -> np.ndarray:
    """Pixels near a strong edge — where demosaicing actually fails."""
    edges = cv2.Canny(to_gray(img), 60, 160)
    return cv2.dilate(edges, np.ones((dilate, dilate), np.uint8)) > 0


def psnr_on_mask(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """PSNR restricted to a region.

    Whole-image PSNR is dominated by the flat majority, where every method is
    correct. Restricting to edges is what makes the differences visible.
    """
    sel = mask if mask.dtype == bool else mask > 0
    if not sel.any():
        return float("inf")
    a = to_float(pred)[sel]
    b = to_float(truth)[sel]
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse <= EPS else float(10.0 * np.log10(1.0 / mse))


def colour_fringing(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean chroma error on edge pixels, in Lab a/b units.

    The artefact by its actual name. Luminance error on an edge is blur; *chroma*
    error on an edge is fringing, and only the second is a demosaicing failure.
    """
    sel = mask if mask.dtype == bool else mask > 0
    if not sel.any():
        return 0.0
    a = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB).astype(np.float64)
    b = cv2.cvtColor(truth, cv2.COLOR_RGB2LAB).astype(np.float64)
    diff = np.sqrt((a[..., 1] - b[..., 1]) ** 2 + (a[..., 2] - b[..., 2]) ** 2)
    return float(diff[sel].mean())


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs ranked by **this project's own failure condition**: the
#: share of pixels that are both an edge and saturated, which is exactly where
#: demosaicing error is claimed to concentrate. No stock axis in
#: `tools/select_images.py` measures it, and nothing else predicts where the
#: methods will separate.
#:
#: The spread is 0.0% to 70.8%. The sea stacks are grey surf with no saturated
#: edge anywhere — the control, on which every method should agree — and the
#: flounder is saturated texture across the whole frame.
IMAGES = (
    "sea_stacks",           # saturated edges  0.0% - the control
    "kangaroo_resting",     #                  0.3%
    "laden_donkey",         #                  1.5%
    "two_in_headscarves",   #                  2.7%
    "bison_in_snow",        #                  4.5%
    "mono_lake_tufa",       #                  6.9%
    "drying_racks",         #                  9.3%
    "diver_and_coral",      #                 14.2%
    "lynx_on_birch",        #                 16.5%
    "villa_on_the_lake",    #                 23.1%
    "sandstone_ladder",     #                 28.5%
    "flounder_on_gravel",   #                 70.8% - saturated texture everywhere
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs, used as the ground truth.

    The mosaic is made from this, so the "before" image is exactly what the
    algorithm is trying to recover.
    """
    from shared import io

    return io.real_photo(name)


def saturated_edge_share(img: np.ndarray, chroma_threshold: float = 40.0) -> float:
    """Percentage of pixels that are both an edge and saturated.

    The project's claim in one number, and the axis the pool is ordered by:
    demosaicing has to guess two thirds of the colour at every pixel, and the
    guess is hardest where neighbouring photosites of the same colour see
    genuinely different scene content *and* the colours are far from grey.
    """
    edges = edge_mask(img) > 0
    f = img.astype(np.float32)
    chroma = np.sqrt(((f - f.mean(axis=2, keepdims=True)) ** 2).sum(axis=2))
    return float((edges & (chroma > chroma_threshold)).mean() * 100.0)
NOISE_LEVELS = (0.0, 2.0, 5.0, 10.0, 20.0)


def evaluate_methods(images=IMAGES, pattern: str = "RGGB", noise_sigma: float = 0.0, runs: int = 3):
    """Whole-image and edge-only scores for every method.

    The two columns are the finding: they should rank the methods differently and
    the gap between them should be several dB.
    """
    from shared import synth

    acc = {n: {"psnr": [], "edge_psnr": [], "fringe": [], "ssim": [], "ms": []} for n in METHODS}

    for i, name in enumerate(images):
        truth = load_scene(name)
        raw = mosaic(truth, pattern)
        if noise_sigma > 0:
            raw = synth.gaussian_noise(raw, sigma=noise_sigma, seed=i)
        edges = edge_mask(truth)

        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(raw, pattern), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, truth))
            acc[method]["edge_psnr"].append(psnr_on_mask(out, truth, edges))
            acc[method]["fringe"].append(colour_fringing(out, truth, edges))
            acc[method]["ssim"].append(ssim(out, truth))
            acc[method]["ms"].append(timing.median_ms)

    rows = [
        {
            "method": m,
            "psnr_db": round(float(np.mean(a["psnr"])), 3),
            "edge_psnr_db": round(float(np.mean(a["edge_psnr"])), 3),
            "colour_fringing": round(float(np.mean(a["fringe"])), 3),
            "ssim": round(float(np.mean(a["ssim"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 3),
        }
        for m, a in acc.items()
    ]
    for r in rows:
        r["edge_penalty_db"] = round(r["psnr_db"] - r["edge_psnr_db"], 3)
    return rows


def compare_patterns(images=IMAGES, patterns=PATTERNS, method: str = "Malvar (cross-channel)"):
    """Does the Bayer layout matter? It should not — the pattern is a relabelling.

    All four arrangements have the same green density and the same geometry. If
    the scores differ materially, something is indexing the pattern wrongly, so
    this doubles as a correctness check.
    """

    rows = []
    for pattern in patterns:
        scores = []
        for name in images:
            truth = load_scene(name)
            raw = mosaic(truth, pattern)
            scores.append(psnr(METHODS[method](raw, pattern), truth))
        rows.append({"pattern": pattern, "psnr_db": round(float(np.mean(scores)), 3)})
    return rows


def sweep_noise(images=IMAGES, levels=NOISE_LEVELS):
    """Sensor noise interacts with demosaicing: interpolating noise spreads it.

    An edge-aware method makes decisions based on gradients, and noise corrupts
    those gradients — so the sophisticated methods may lose their advantage, or
    even invert it, as noise rises.
    """
    rows = []
    for sigma in levels:
        scored = evaluate_methods(images=images, noise_sigma=sigma, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma}
        for r in scored:
            row[r["method"]] = r["edge_psnr_db"]
        rows.append(row)
    return rows


def green_density_argument(images=IMAGES):
    """Why green is sampled twice as often, as a measurement.

    Reconstruct each channel independently and compare. Green should be
    substantially more accurate than red or blue for every method, purely because
    it has twice the samples — which is the justification for the whole Bayer
    layout.
    """

    rows = []
    for method, fn in METHODS.items():
        per_channel = {c: [] for c in "RGB"}
        for name in images:
            truth = load_scene(name)
            raw = mosaic(truth, "RGGB")
            out = fn(raw, "RGGB")
            for i, c in enumerate("RGB"):
                per_channel[c].append(psnr(out[..., i], truth[..., i]))
        rows.append(
            {
                "method": method,
                **{f"{c}_psnr_db": round(float(np.mean(v)), 3) for c, v in per_channel.items()},
                "green_advantage_db": round(
                    float(np.mean(per_channel["G"]))
                    - float(np.mean(per_channel["R"] + per_channel["B"])) ,
                    3,
                ),
            }
        )
    return rows


def isp_ablation(images=IMAGES):
    """Turn each ISP stage off and measure its contribution."""
    from shared import synth

    configs = {
        "Full ISP": {},
        "No white balance": {"white_balance": False},
        "No denoise": {"denoise": False},
        "No gamma": {"gamma": 1.0},
        "Demosaic only": {"white_balance": False, "denoise": False, "gamma": 1.0},
    }
    rows = []
    for label, kwargs in configs.items():
        scores = []
        for i, name in enumerate(images):
            truth = load_scene(name)
            raw = synth.gaussian_noise(mosaic(truth), sigma=5.0, seed=i)
            out = isp_pipeline(raw, **kwargs)
            scores.append(psnr(out, truth))
        rows.append({"configuration": label, "psnr_db": round(float(np.mean(scores)), 3)})
    return rows


def demosaic(raw: np.ndarray, method: str = "Malvar (cross-channel)", pattern: str = "RGGB"):
    return METHODS[method](raw, pattern)
