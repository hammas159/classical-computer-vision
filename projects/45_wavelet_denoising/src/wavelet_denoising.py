"""Wavelet denoising: soft vs hard thresholding, and how to pick the threshold.

The question
------------
Project 13 compared spatial denoisers. This one asks a narrower question with a
sharper answer:

> **The claim under test:** wavelet denoising works because natural images are
> *sparse* in the wavelet domain — a few large coefficients carry the signal and
> many small ones carry the noise — while white noise is **not** sparse: it
> spreads evenly across all coefficients. So thresholding separates them.

That is measurable directly: compute the coefficient sparsity of a clean image
and of pure noise and compare. If the premise fails, nothing downstream matters.

Two design decisions carry the rest:

* **Soft vs hard thresholding.** Hard keeps coefficients unchanged above the
  threshold and zeroes the rest, which is discontinuous and produces ringing.
  Soft *shrinks* everything toward zero, which is continuous and biases the
  large coefficients. The trade is bias against artefacts.
* **The threshold itself.** VisuShrink uses one universal value; BayesShrink
  adapts per subband. Since the noise sigma is known here, an **oracle**
  threshold — the best possible, found by search — bounds what any rule could
  achieve.
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
# the transform
# --------------------------------------------------------------------------- #

#: Haar analysis filters. Implemented directly rather than via PyWavelets so the
#: project has no dependency beyond the repo's own, and so the decomposition is
#: visible rather than a library call.
def haar_step(channel: np.ndarray):
    """One level of 2-D Haar decomposition: returns (LL, LH, HL, HH).

    Rows then columns, each split into a sum (low-pass) and a difference
    (high-pass), both scaled by 1/sqrt(2) so the transform is **orthonormal**.
    Orthonormality is what makes the noise sigma the same in the coefficient
    domain as in the image domain — the entire basis of the threshold formulas
    below.
    """
    c = channel.astype(np.float32)
    h, w = c.shape
    c = c[: h - h % 2, : w - w % 2]

    even_r, odd_r = c[0::2, :], c[1::2, :]
    low_r = (even_r + odd_r) / np.sqrt(2.0)
    high_r = (even_r - odd_r) / np.sqrt(2.0)

    def split_cols(x):
        return (x[:, 0::2] + x[:, 1::2]) / np.sqrt(2.0), (x[:, 0::2] - x[:, 1::2]) / np.sqrt(2.0)

    ll, lh = split_cols(low_r)
    hl, hh = split_cols(high_r)
    return ll, lh, hl, hh


def haar_inverse_step(ll, lh, hl, hh):
    """Invert one Haar level."""
    def merge_cols(a, b):
        h, w = a.shape
        out = np.zeros((h, w * 2), np.float32)
        out[:, 0::2] = (a + b) / np.sqrt(2.0)
        out[:, 1::2] = (a - b) / np.sqrt(2.0)
        return out

    low_r = merge_cols(ll, lh)
    high_r = merge_cols(hl, hh)
    h, w = low_r.shape
    out = np.zeros((h * 2, w), np.float32)
    out[0::2, :] = (low_r + high_r) / np.sqrt(2.0)
    out[1::2, :] = (low_r - high_r) / np.sqrt(2.0)
    return out


def decompose(channel: np.ndarray, levels: int = 3):
    """Multi-level decomposition. Returns (final LL, [(LH, HL, HH), ...])."""
    coeffs = []
    current = channel.astype(np.float32)
    for _ in range(levels):
        if min(current.shape) < 4:
            break
        ll, lh, hl, hh = haar_step(current)
        coeffs.append((lh, hl, hh))
        current = ll
    return current, coeffs


def reconstruct(ll: np.ndarray, coeffs) -> np.ndarray:
    current = ll
    for lh, hl, hh in reversed(coeffs):
        h = min(current.shape[0], lh.shape[0])
        w = min(current.shape[1], lh.shape[1])
        current = haar_inverse_step(
            current[:h, :w], lh[:h, :w], hl[:h, :w], hh[:h, :w]
        )
    return current


# --------------------------------------------------------------------------- #
# thresholding rules
# --------------------------------------------------------------------------- #


def threshold_hard(coeffs: np.ndarray, t: float) -> np.ndarray:
    """Keep coefficients above ``t`` unchanged, zero the rest.

    Discontinuous at the threshold, which is exactly what produces the
    characteristic ringing: a coefficient hovering near ``t`` flips between its
    full value and zero between neighbouring positions.
    """
    return np.where(np.abs(coeffs) > t, coeffs, 0.0)


def threshold_soft(coeffs: np.ndarray, t: float) -> np.ndarray:
    """Shrink every coefficient toward zero by ``t``.

    Continuous, so no ringing — but every surviving coefficient is now smaller
    than it should be, which systematically blurs the result. Bias in exchange
    for smoothness.
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - t, 0.0)


SHRINKAGE: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "Hard": threshold_hard,
    "Soft": threshold_soft,
}


# --------------------------------------------------------------------------- #
# threshold selection
# --------------------------------------------------------------------------- #


def estimate_sigma_mad(hh: np.ndarray) -> float:
    """Noise sigma from the finest diagonal subband, via the median absolute deviation.

    The finest HH band is almost pure noise in a natural image, and MAD/0.6745 is
    a robust sigma estimator that a few large coefficients cannot skew. This is
    how wavelet denoisers work without being told the noise level.
    """
    return float(np.median(np.abs(hh)) / 0.6745)


def threshold_visushrink(coeffs: np.ndarray, sigma: float) -> float:
    """Universal threshold ``sigma * sqrt(2 ln N)``.

    Chosen so that, with high probability, *no* pure-noise coefficient survives.
    That is a strong guarantee and it is why VisuShrink over-smooths: it is tuned
    to remove all noise rather than to minimise error.
    """
    n = coeffs.size
    return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def threshold_bayesshrink(coeffs: np.ndarray, sigma: float) -> float:
    """BayesShrink: ``sigma^2 / sigma_signal``, computed per subband.

    Adapts to each subband's own signal energy, so detail-rich bands keep a lower
    threshold. Where the signal variance is smaller than the noise variance the
    band is assumed to be all noise and is zeroed entirely.
    """
    var_total = float(np.mean(coeffs**2))
    var_signal = max(var_total - sigma**2, 0.0)
    if var_signal <= EPS:
        return float(np.abs(coeffs).max())
    return float(sigma**2 / np.sqrt(var_signal))


THRESHOLD_RULES: dict[str, Callable[[np.ndarray, float], float]] = {
    "VisuShrink": threshold_visushrink,
    "BayesShrink": threshold_bayesshrink,
}


# --------------------------------------------------------------------------- #
# the denoisers
# --------------------------------------------------------------------------- #


def denoise_wavelet(
    img: np.ndarray, rule: str = "BayesShrink", shrink: str = "Soft",
    levels: int = 3, sigma: float | None = None, fixed_threshold: float | None = None,
) -> np.ndarray:
    """Wavelet denoise one image, per channel.

    Only the detail subbands are thresholded. The final LL band carries the
    image's low-frequency content — thresholding it would remove the picture
    rather than the noise.
    """
    colour = img.ndim == 3
    channels = [img[..., c] for c in range(3)] if colour else [img]
    out = []

    for channel in channels:
        f = to_float(channel)
        ll, coeffs = decompose(f, levels=levels)
        if not coeffs:
            out.append(f)
            continue

        s = sigma if sigma is not None else estimate_sigma_mad(coeffs[0][2])
        shrink_fn = SHRINKAGE[shrink]

        new_coeffs = []
        for lh, hl, hh in coeffs:
            band = []
            for sub in (lh, hl, hh):
                t = fixed_threshold if fixed_threshold is not None else THRESHOLD_RULES[rule](sub, s)
                band.append(shrink_fn(sub, t))
            new_coeffs.append(tuple(band))

        restored = reconstruct(ll, new_coeffs)
        h, w = f.shape
        padded = np.zeros_like(f)
        rh, rw = min(h, restored.shape[0]), min(w, restored.shape[1])
        padded[:rh, :rw] = restored[:rh, :rw]
        padded[rh:, :] = f[rh:, :]
        padded[:, rw:] = f[:, rw:]
        out.append(padded)

    stacked = np.stack(out, axis=-1) if colour else out[0]
    return to_uint8(np.clip(stacked, 0.0, 1.0))


def estimate_noise(img: np.ndarray) -> float:
    """Noise sigma in 0-255 levels, from the MAD of the finest diagonal subband.

    The same estimate BayesShrink uses. Every method in this project is given it,
    because a comparison in which one family adapts to the noise and the others
    run on fixed constants is a comparison of tuning effort.
    """
    ll, coeffs = decompose(to_float(to_gray(img)), levels=1)
    return float(estimate_sigma_mad(coeffs[0][2]) * 255.0)


#: Measured optima, from `sweep_spatial_parameters`. Both were badly mistuned in
#: the first version of this project and both handicaps favoured the wavelets:
#: NLM ran at h=10 where 16 was best at sigma 25 (a 1.6 dB penalty) and the
#: Gaussian at 1.5 where 0.8 was best (0.9 dB).
#:
#: NLM's optimum is a near-constant fraction of the **true** noise — 0.60, 0.64
#: and 0.60 at sigma 10, 25 and 50 — which is worth stating because the advice
#: usually quoted is h ~ sigma. It is not; it is about 0.6 sigma.
#:
#: Against the MAD *estimate* the same optima are 0.62, 0.89 and 0.96, and the
#: drift is not NLM's: `estimate_noise` under-reads by more the noisier the image
#: gets (see `sigma_estimation_accuracy`). Every method here is tuned against the
#: estimate, because that is all a real denoiser has, so this constant is a
#: compromise across the range rather than any one level's optimum.
NLM_H_PER_SIGMA = 0.85

#: The Gaussian's optimum grows roughly with the square root of the noise (0.6,
#: 0.8, 1.4 at sigma 10, 25, 50), which this constant times sqrt(estimate)
#: reproduces to within a fifth of a level.
GAUSSIAN_SIGMA_PER_ROOT = 0.20

#: The bilateral filter's range width wants to be several times the noise —
#: 3.1, 4.5 and 5.1 times the estimate at sigma 10, 25 and 50. Too narrow and it
#: treats noise as edges and preserves it; the first version of this project used
#: a fixed 50, which is right at one noise level out of three.
BILATERAL_SIGMA_PER_ESTIMATE = 4.5


def denoise_gaussian(img: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Spatial Gaussian — the baseline from the same family of assumptions.

    With ``sigma=None`` the width is set from the image's own noise estimate, so
    it adapts exactly as BayesShrink does.
    """
    if sigma is None:
        sigma = max(0.3, GAUSSIAN_SIGMA_PER_ROOT * np.sqrt(estimate_noise(img)))
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def denoise_bilateral(img: np.ndarray, colour_sigma: float | None = None) -> np.ndarray:
    """Bilateral — an edge-preserving average, range width set from the noise."""
    if colour_sigma is None:
        colour_sigma = max(10.0, BILATERAL_SIGMA_PER_ESTIMATE * estimate_noise(img))
    return cv2.bilateralFilter(img, 9, float(colour_sigma), 9)


def denoise_nlm(img: np.ndarray, h: float | None = None) -> np.ndarray:
    if h is None:
        h = max(3.0, NLM_H_PER_SIGMA * estimate_noise(img))
    h = float(h)
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def denoise_none(img: np.ndarray) -> np.ndarray:
    return img.copy()


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Do nothing (control)": denoise_none,
    "Gaussian (spatial)": denoise_gaussian,
    "Bilateral (spatial)": denoise_bilateral,
    "Non-local means (spatial)": denoise_nlm,
    "Wavelet VisuShrink hard": lambda a: denoise_wavelet(a, "VisuShrink", "Hard"),
    "Wavelet VisuShrink soft": lambda a: denoise_wavelet(a, "VisuShrink", "Soft"),
    "Wavelet BayesShrink hard": lambda a: denoise_wavelet(a, "BayesShrink", "Hard"),
    "Wavelet BayesShrink soft": lambda a: denoise_wavelet(a, "BayesShrink", "Soft"),
}


# --------------------------------------------------------------------------- #
# the premise, tested directly
# --------------------------------------------------------------------------- #


def sparsity(coeffs: np.ndarray, fraction: float = 0.1) -> float:
    """Fraction of total energy held by the largest ``fraction`` of coefficients.

    The direct measure of sparsity. A natural image should concentrate most of
    its energy in a small minority of coefficients; white noise should spread it
    evenly, giving a value close to ``fraction`` itself.
    """
    flat = np.sort(np.abs(coeffs.ravel()))[::-1]
    k = max(1, int(len(flat) * fraction))
    total = float(np.sum(flat**2))
    return float(np.sum(flat[:k] ** 2) / max(total, EPS))


def test_sparsity_premise(images=None, levels: int = 3):
    """Compare the sparsity of real images with that of pure noise.

    If the gap is not large, wavelet denoising has no basis and everything else
    in this project is measuring something else.
    """
    images = IMAGES if images is None else images
    rng = np.random.default_rng(0)
    rows = []
    for name in images:
        gray = to_float(to_gray(load_scene(name)))
        _, coeffs = decompose(gray, levels=levels)
        detail = np.concatenate([np.concatenate([c.ravel() for c in band]) for band in coeffs])
        rows.append({"signal": name, "top10pct_energy": round(sparsity(detail), 4)})

    noise = rng.normal(0, 0.1, (512, 512)).astype(np.float32)
    _, ncoeffs = decompose(noise, levels=levels)
    ndetail = np.concatenate([np.concatenate([c.ravel() for c in band]) for band in ncoeffs])
    rows.append({"signal": "white noise", "top10pct_energy": round(sparsity(ndetail), 4)})
    return rows


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

#: Twelve photographs ranked by **this project's own premise**: the share of
#: wavelet energy carried by the largest 10% of coefficients, which is what
#: `sparsity()` measures. No stock axis in `tools/select_images.py` measures it,
#: and nothing else predicts how well wavelet denoising can work — the method
#: exists because signal is sparse in this basis and noise is not, so a pool that
#: did not span sparsity would not test the claim.
#:
#: The spread is 0.688 to 0.989. The packhorse bridge is a frame of dense
#: woodland, where almost nothing is sparse; the hawk is a bird on a bare branch
#: against plain sky, where 10% of the coefficients carry 99.3% of the energy.
IMAGES = (
    "packhorse_bridge",         # sparsity 0.688 - dense woodland, the hard case
    "black_panther",            #          0.788
    "lone_tree_on_a_hill",      #          0.804
    "church_spire",             #          0.832
    "warthogs_drinking",        #          0.855
    "wallaby_and_joey",         #          0.875
    "warbler_at_the_nest",      #          0.891
    "four_children_on_a_wall",  #          0.905
    "cormorants_nesting",       #          0.922
    "woman_among_roses",        #          0.923
    "taj_mahal_reflected",      #          0.947
    "hawk_on_a_branch",         #          0.989 - a bird on plain sky
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's photographs, used as the clean truth.

    Named so that `run.py`, the tests and `infer.py` all read the same pixels —
    noise is added to these, so the ground truth depends on them.
    """
    from shared import io

    return io.real_photo(name)


def image_sparsity(img: np.ndarray, levels: int = 3, fraction: float = 0.1) -> float:
    """The share of detail-coefficient energy in the largest `fraction` of them.

    The axis the pool is ordered by, and the project's premise in one number.
    Recomputed here so the README's figures come from the project rather than
    from a selection script.
    """
    ll, coeffs = decompose(to_float(to_gray(img)), levels=levels)
    flat = np.concatenate([c.ravel() for level in coeffs for c in level])
    return sparsity(flat, fraction=fraction)
NOISE_LEVELS = (5.0, 10.0, 20.0, 35.0, 50.0)
LEVEL_COUNTS = (1, 2, 3, 4, 5)


def evaluate_methods(noise_sigma: float = 25.0, images=IMAGES, runs: int = 3):
    """Score every method, wavelet and spatial, on the same noisy images."""
    from shared import synth

    acc = {n: {"psnr": [], "ssim": [], "ms": []} for n in METHODS}
    noisy_stats = []

    for i, name in enumerate(images):
        clean = load_scene(name)
        noisy = synth.gaussian_noise(clean, sigma=noise_sigma, seed=i)
        noisy_stats.append(psnr(noisy, clean))
        for method, fn in METHODS.items():
            out, timing = timeit(lambda f=fn: f(noisy), runs=runs, warmup=1)
            acc[method]["psnr"].append(psnr(out, clean))
            acc[method]["ssim"].append(ssim(out, clean))
            acc[method]["ms"].append(timing.median_ms)

    return (
        [
            {
                "method": m,
                "psnr_db": round(float(np.mean(a["psnr"])), 3),
                "ssim": round(float(np.mean(a["ssim"])), 4),
                "median_ms": round(float(np.median(a["ms"])), 3),
            }
            for m, a in acc.items()
        ],
        {"noisy_input_psnr": round(float(np.mean(noisy_stats)), 3)},
    )


def oracle_threshold(noise_sigma: float = 25.0, images=IMAGES, shrink: str = "Soft",
                     candidates=np.linspace(0.005, 0.25, 25)):
    """The best achievable threshold, found by search.

    Bounds what any selection rule could do. If VisuShrink and BayesShrink both
    sit well below this, the gap is the cost of choosing a threshold without
    knowing the answer — which is the honest way to judge a selection rule.
    """
    from shared import synth

    best_t, best_psnr = None, -np.inf
    for t in candidates:
        scores = []
        for i, name in enumerate(images):
            clean = load_scene(name)
            noisy = synth.gaussian_noise(clean, sigma=noise_sigma, seed=i)
            out = denoise_wavelet(noisy, shrink=shrink, fixed_threshold=float(t))
            scores.append(psnr(out, clean))
        mean = float(np.mean(scores))
        if mean > best_psnr:
            best_psnr, best_t = mean, float(t)
    return {"best_threshold": round(best_t, 5), "best_psnr_db": round(best_psnr, 3)}


def sweep_noise(levels=NOISE_LEVELS, images=IMAGES):
    """Where wavelet methods overtake spatial ones, if they do."""
    rows = []
    for sigma in levels:
        scored, noisy = evaluate_methods(noise_sigma=sigma, images=images, runs=1)
        row: dict[str, float] = {"noise_sigma": sigma, "noisy_input": noisy["noisy_input_psnr"]}
        for r in scored:
            row[r["method"]] = r["psnr_db"]
        rows.append(row)
    return rows


def sweep_levels(counts=LEVEL_COUNTS, noise_sigma: float = 25.0, images=IMAGES):
    """How many decomposition levels help before they stop?

    Each level halves the resolution, so beyond three or four the subbands are too
    small for a reliable sigma estimate and performance should fall off.
    """
    from shared import synth

    rows = []
    for n in counts:
        p, s = [], []
        for i, name in enumerate(images):
            clean = load_scene(name)
            noisy = synth.gaussian_noise(clean, sigma=noise_sigma, seed=i)
            out = denoise_wavelet(noisy, levels=n)
            p.append(psnr(out, clean))
            s.append(ssim(out, clean))
        rows.append(
            {
                "levels": n,
                "psnr_db": round(float(np.mean(p)), 3),
                "ssim": round(float(np.mean(s)), 4),
            }
        )
    return rows


def sigma_estimation_accuracy(levels=NOISE_LEVELS, images=IMAGES):
    """Does the MAD estimator recover the noise sigma it was not told?

    Everything downstream depends on it, so a systematic bias here would
    propagate into every threshold.
    """
    from shared import synth

    rows = []
    for sigma in levels:
        estimates = []
        for i, name in enumerate(images):
            clean = load_scene(name)
            noisy = synth.gaussian_noise(clean, sigma=sigma, seed=i)
            _, coeffs = decompose(to_float(to_gray(noisy)))
            estimates.append(estimate_sigma_mad(coeffs[0][2]) * 255.0)
        rows.append(
            {
                "true_sigma": sigma,
                "estimated_sigma": round(float(np.mean(estimates)), 3),
                "relative_error": round(float(np.mean(estimates) / max(sigma, EPS) - 1.0), 4),
            }
        )
    return rows


def denoise(img: np.ndarray, method: str = "Wavelet BayesShrink soft") -> np.ndarray:
    return METHODS[method](img)
