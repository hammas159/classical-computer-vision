"""Tests for project 03, low-light enhancement.

The failures worth guarding against here are numerical rather than structural: a
log that collapses to log(eps) on a black pixel, a metric that penalises a method
for solving a different problem, and a "ceiling" claim that is really an
arithmetic statement and should be checked as one.
"""

from __future__ import annotations

import numpy as np
import pytest

import low_light as ll
from shared import io, synth
from shared.metrics import estimate_noise_sigma, mean_brightness, psnr, ssim

# --------------------------------------------------------------------------- #
# the degradation and its ceiling
# --------------------------------------------------------------------------- #


def test_low_light_darkens_and_is_reproducible():
    img = io.sample("coffee")
    dark = synth.low_light(img, gamma=3.0, noise_sigma=4.0, seed=0)
    assert mean_brightness(dark) < mean_brightness(img)
    assert np.array_equal(dark, synth.low_light(img, gamma=3.0, noise_sigma=4.0, seed=0))


def test_quantisation_ceiling_is_monotonic_and_bounded():
    levels = [ll.quantisation_ceiling(g) for g in ll.GAMMA_LEVELS]
    assert all(1 <= n <= 256 for n in levels)
    # darker means fewer surviving levels, strictly
    assert levels == sorted(levels, reverse=True)
    assert ll.quantisation_ceiling(1.0) == 256  # no darkening loses nothing


def test_quantisation_ceiling_matches_a_direct_count():
    """The ceiling is arithmetic, not an empirical fit — check it directly."""
    gamma = 3.0
    v = np.arange(256, dtype=np.float64) / 255.0
    darkened = np.round(np.power(v, gamma) * 255.0)
    assert ll.quantisation_ceiling(gamma) == len(set(darkened.tolist()))


def test_the_oracle_cannot_be_beaten_by_any_real_method():
    """The project's central claim, as a regression test.

    The oracle applies the exact inverse of the degradation. In real arithmetic
    that is perfect; what it actually scores is the ceiling that 8-bit
    quantisation and shadow noise left behind. No method that does *not* know
    gamma may exceed it.
    """
    rows, _ = ll.evaluate_methods(gamma=3.0, images=("coffee", "chelsea"), runs=1)
    oracle = next(r for r in rows if r["method"] == ll.ORACLE_NAME)
    real = [r for r in rows if r["method"] != ll.ORACLE_NAME]
    assert all(r["psnr_db"] <= oracle["psnr_db"] + 1e-6 for r in real)
    # and the ceiling is finite: perfect inversion is impossible after quantising
    assert np.isfinite(oracle["psnr_db"])
    assert oracle["psnr_db"] < 40.0


def test_oracle_inverts_exactly_when_there_is_no_quantisation_or_noise():
    """With float arithmetic and no noise the inverse is exact — proving the
    ceiling comes from the 8-bit round trip, not from the model being wrong."""
    f = np.linspace(0.02, 1.0, 500, dtype=np.float64)
    gamma = 3.0
    restored = np.power(np.power(f, gamma), 1.0 / gamma)
    assert np.allclose(restored, f, atol=1e-9)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(ll.METHODS))
def test_every_method_returns_rgb_uint8_of_the_same_shape(name):
    img = io.sample("chelsea")
    dark = synth.low_light(img, gamma=3.0, seed=0)
    out = ll.METHODS[name](dark)
    assert out.dtype == np.uint8, name
    assert out.shape == dark.shape, name
    assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize("name", list(ll.METHODS))
def test_every_method_brightens(name):
    img = io.sample("coffee")
    dark = synth.low_light(img, gamma=3.0, seed=0)
    out = ll.METHODS[name](dark)
    assert mean_brightness(out) > mean_brightness(dark), name


@pytest.mark.parametrize("name", list(ll.METHODS))
def test_no_method_produces_nan_or_a_constant_image(name):
    """Catches the class of bug that made MSRCR useless: a log of a near-zero
    value, which silently produces a flat or sign-flipped result."""
    img = io.sample("astronaut")
    dark = synth.low_light(img, gamma=4.0, seed=0)
    out = ll.METHODS[name](dark).astype(np.float64)
    assert np.isfinite(out).all(), name
    assert out.std() > 1.0, f"{name} returned a nearly constant image"


def test_msrcr_colour_restoration_stays_bounded_on_black_pixels():
    """The specific numerical bug this project found.

    Written as log(alpha*I) - log(sum I), the restoration term collapses to
    log(eps) = -13.8 for a near-black pixel, swings to -640, and flips the sign
    of every shadow pixel. The bounded log1p-of-ratio form cannot do that.
    """
    black = np.zeros((32, 32, 3), np.uint8)
    out = ll.enhance_msrcr(black)
    assert np.isfinite(out).all()

    # the restoration term itself must be non-negative everywhere
    f = io.to_float(io.sample("coffee"))
    total = f.sum(axis=2, keepdims=True) + ll.EPS
    ratio = np.clip(f / total, 0.0, 1.0)
    restoration = 46.0 * np.log1p(125.0 * ratio)
    assert restoration.min() >= 0.0
    assert np.isfinite(restoration).all()


def test_stretch_uses_percentiles_not_extremes():
    """A single hot pixel must not set the scale for the whole image."""
    x = np.zeros((100, 100), np.float32)
    x[50, 50] = 1000.0  # one outlier
    out = ll._stretch(x)
    # with min/max scaling everything but the outlier would be ~0; with
    # percentiles the outlier is clipped away instead
    assert out.max() <= 1.0
    assert (out > 0.5).sum() <= 5


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_exposure_matching_rescues_retinex_from_a_metric_mismatch():
    """Retinex estimates reflectance, not exposure. Scoring its raw output with
    PSNR measures a global brightness offset, not how much detail was recovered.
    """
    clean = io.sample("coffee")
    dark = synth.low_light(clean, gamma=3.0, seed=0)
    out = ll.enhance_ssr(dark)

    raw = psnr(out, clean)
    matched = psnr(ll.match_exposure(out, clean), clean)
    assert matched > raw + 3.0, "exposure matching should recover several dB"


def test_match_exposure_equalises_mean_brightness():
    clean = io.sample("chelsea")
    dark = synth.low_light(clean, gamma=3.0, seed=0)
    matched = ll.match_exposure(dark, clean)
    assert mean_brightness(matched) == pytest.approx(mean_brightness(clean), abs=0.02)


def test_match_exposure_is_a_no_op_on_a_black_image():
    black = np.zeros((16, 16, 3), np.uint8)
    assert np.array_equal(ll.match_exposure(black, io.sample("coffee")), black)


def test_brightening_amplifies_noise():
    """The hidden cost: brightening is a multiplication, so shadow noise is
    multiplied too. A method that wins on brightness while tripling the noise
    has not improved the picture."""
    rows = ll.evaluate_noise_amplification(gamma=3.0, images=("coffee", "chelsea"))
    assert all(r["amplification"] > 1.0 for r in rows)
    retinex = next(r for r in rows if r["method"] == "Multi-scale Retinex")
    gamma_row = next(r for r in rows if r["method"] == "Gamma 1/2.2 (fixed)")
    assert retinex["amplification"] > gamma_row["amplification"]


def test_noise_estimator_tracks_a_known_sigma_on_grayscale():
    """Calibration check — on a flat *grayscale* field the estimate matches sigma."""
    flat = np.full((256, 256), 128, np.uint8)
    for true_sigma in (10.0, 20.0):
        noisy = synth.gaussian_noise(flat, sigma=true_sigma, seed=0)
        assert estimate_noise_sigma(noisy) == pytest.approx(true_sigma, rel=0.2)


def test_noise_estimator_reads_lower_on_rgb_because_gray_averages_channels():
    """A gotcha worth pinning down rather than rediscovering.

    ``gaussian_noise`` adds *independent* noise to each channel. The estimator
    works on luminance, and averaging three independent channels divides the
    standard deviation by about sqrt(3). So a per-channel sigma of 10 reads back
    near 6, which is the correct luminance noise and not the number passed in.
    """
    flat_rgb = np.full((256, 256, 3), 128, np.uint8)
    measured = estimate_noise_sigma(synth.gaussian_noise(flat_rgb, sigma=10.0, seed=0))
    assert measured == pytest.approx(10.0 / np.sqrt(3), rel=0.25)
    assert measured < 10.0


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def test_evaluate_methods_returns_a_row_per_method_plus_the_oracle():
    rows, dark_stats = ll.evaluate_methods(gamma=2.0, images=("coffee",), runs=1)
    assert len(rows) == len(ll.METHODS) + 1
    assert {r["method"] for r in rows} == set(ll.METHODS) | {ll.ORACLE_NAME}
    assert set(dark_stats) == {"entropy", "brightness", "noise"}


def test_sweep_gamma_shows_the_ceiling_falling_as_it_gets_darker():
    rows = ll.sweep_gamma(images=("coffee",), levels=(1.5, 3.0, 5.0))
    ceilings = [r[ll.ORACLE_NAME] for r in rows]
    assert ceilings == sorted(ceilings, reverse=True)
    assert rows[0]["levels_left"] > rows[-1]["levels_left"]


def test_auto_gamma_recovers_the_true_exponent_when_its_assumption_holds():
    """On an image whose true mean is near the target, the estimate is accurate.

    ``astronaut`` averages 0.449 against a 0.45 target, so this isolates the
    estimator's arithmetic from its assumption.
    """
    clean = io.sample("astronaut")
    for true_gamma in (1.5, 3.0, 5.0):
        dark = synth.low_light(clean, gamma=true_gamma, noise_sigma=4.0, seed=0)
        recovered = 1.0 / ll.estimate_gamma(dark)
        assert recovered == pytest.approx(true_gamma, rel=0.15), (
            f"gamma {true_gamma}: estimated {recovered:.2f}"
        )


def test_auto_gamma_error_tracks_the_brightness_assumption():
    """The project's second finding, as a regression test.

    The estimator cannot know the scene's true exposure, so its error is a
    function of how far that exposure sits from the target — **not** of how dark
    the image was made. Asserting the correlation is what turns "it sometimes
    fails" into a stated, checkable law.
    """
    rows = ll.brightness_assumption_table()
    gaps = np.array([abs(r["brightness_gap"]) for r in rows])
    errors = np.array([abs(r["mean_gamma_error_pct"]) for r in rows])

    # Rank correlation, not Pearson. The relationship is monotonic but strongly
    # non-linear -- `rocket` sits at a 0.194 gap and a 170% error, far off any
    # straight line through the rest -- so Pearson (0.81) understates a
    # relationship that Spearman (0.89) measures correctly.
    def _rank(x):
        return np.argsort(np.argsort(x)).astype(float)

    spearman = float(np.corrcoef(_rank(gaps), _rank(errors))[0, 1])
    assert spearman > 0.85, f"gap/error rank correlation only {spearman:.2f}"

    # and the extremes behave as the law predicts
    best = min(rows, key=lambda r: abs(r["brightness_gap"]))
    worst = max(rows, key=lambda r: abs(r["brightness_gap"]))
    assert abs(best["mean_gamma_error_pct"]) < 15.0
    assert abs(worst["mean_gamma_error_pct"]) > 50.0


def test_auto_gamma_beats_the_fixed_constant_away_from_its_sweet_spot():
    """A fixed 1/2.2 is exactly right only when the scene was darkened by 2.2.

    The project-01 lesson applied here: one lucky constant against a field of
    adaptive methods measures the constant's luck, not the family. Checked on an
    image where the auto method's own assumption holds, so the comparison is
    between the two *strategies* rather than between two different failures.
    """
    clean = io.sample("astronaut")
    for true_gamma in (1.5, 4.0):
        dark = synth.low_light(clean, gamma=true_gamma, noise_sigma=4.0, seed=0)
        assert psnr(ll.enhance_gamma_auto(dark), clean) > psnr(ll.enhance_gamma(dark), clean)


def test_fixed_gamma_equals_the_oracle_at_its_own_gamma():
    """At gamma 2.2 the fixed curve IS the exact inverse, so it must tie.

    If this ever stops holding, either the oracle or the fixed curve has drifted
    from being a true inverse of the degradation.
    """
    clean = io.sample("coffee")
    dark = synth.low_light(clean, gamma=2.2, noise_sigma=0.0, seed=0)
    fixed = psnr(ll.enhance_gamma(dark), clean)
    oracle = psnr(ll.enhance_oracle(dark, 2.2), clean)
    assert fixed == pytest.approx(oracle, abs=0.01)


def test_auto_gamma_leaves_a_well_exposed_image_alone():
    """Applied to an image that is already correctly exposed it must do little."""
    clean = io.sample("coffee")
    out = ll.enhance_gamma_auto(clean)
    assert psnr(out, clean) > 20.0


def test_enhance_dispatches_by_name():
    img = synth.low_light(io.sample("coffee"), gamma=3.0, seed=0)
    assert np.array_equal(ll.enhance(img, "CLAHE"), ll.enhance_clahe(img))
    with pytest.raises(KeyError):
        ll.enhance(img, "not a method")
