"""Tests for project 45, wavelet denoising.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_every_method_is_given_the_same_noise_estimate` exists because the spatial
competitors were originally run on fixed constants while the wavelet methods
adapted — a 1.6 dB handicap on non-local means and 0.9 dB on the Gaussian, both
of which flattered the wavelets.
"""

from __future__ import annotations

import numpy as np
import pytest

import wavelet_denoising as wd

from shared import synth
from shared.metrics import psnr


@pytest.fixture(scope="module")
def scored():
    rows, _ = wd.evaluate_methods(noise_sigma=25.0, runs=1)
    return {r["method"]: r for r in rows}


@pytest.fixture(scope="module")
def sweep():
    return {r["noise_sigma"]: r for r in wd.sweep_noise()}


# --------------------------------------------------------------------------- #
# the premise
# --------------------------------------------------------------------------- #


def test_signal_is_sparse_in_this_basis_and_noise_is_not():
    """Check the premise before measuring anything that depends on it.

    Wavelet denoising works because a few large coefficients carry the signal
    while white noise spreads evenly. If that gap were not there, nothing
    downstream would mean anything.
    """
    rows = {r["signal"]: r["top10pct_energy"] for r in wd.test_sparsity_premise()}
    noise = rows.pop("white noise")

    assert 0.3 < noise < 0.55, "white noise should be close to the 0.1 of a flat spectrum"
    for name, value in rows.items():
        assert value > noise + 0.2, name


def test_the_pool_spans_sparsity():
    from shared import io

    values = []
    for name in wd.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(wd.image_sparsity(wd.load_scene(name)))

    assert values == sorted(values), "IMAGES should be ordered by sparsity"
    assert min(values) < 0.75 and max(values) > 0.95


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_every_method_is_given_the_same_noise_estimate():
    """Or the comparison measures tuning effort rather than method.

    The spatial denoisers originally ran on fixed constants — h=10 for non-local
    means where 16 was best, sigma=1.5 for the Gaussian where 0.8 was — and both
    handicaps favoured the wavelets. All three now read the image's own MAD
    estimate, which is exactly what BayesShrink uses.
    """
    clean = wd.load_scene(wd.IMAGES[0])
    quiet = synth.gaussian_noise(clean, sigma=5.0, seed=0)
    loud = synth.gaussian_noise(clean, sigma=40.0, seed=0)

    assert wd.estimate_noise(loud) > 2 * wd.estimate_noise(quiet)

    # and the adaptive defaults must actually move with it
    for fn in (wd.denoise_gaussian, wd.denoise_nlm, wd.denoise_bilateral):
        assert not np.array_equal(fn(quiet), fn(loud)), fn.__name__


def test_the_mad_estimator_is_biased_in_both_directions():
    """A real bias, and it matters: BayesShrink's threshold is proportional to it.

    At sigma 5 it reads 5.99 — 20% high, because the image's own fine detail is in
    the same subband and sets a floor. At sigma 50 it reads 30.6 — 39% low,
    because clipping at 0 and 255 flattens that subband. The threshold inherits
    both errors: too aggressive on quiet images, too timid on loud ones.
    """
    rows = sorted(wd.sigma_estimation_accuracy(), key=lambda r: r["true_sigma"])
    ratios = [r["estimated_sigma"] / r["true_sigma"] for r in rows]

    # it OVER-reads when the noise is small, because the image's own fine detail
    # lands in the same subband and sets a floor the estimate cannot go below
    assert ratios[0] > 1.1

    # and under-reads increasingly as the noise grows, because clipping at 0 and
    # 255 flattens the subband it is measured from
    assert ratios[-1] < 0.7
    assert ratios == sorted(ratios, reverse=True)


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_premise_holds_and_the_method_still_loses(scored):
    """The project's headline.

    Signal really is sparse and noise really is not — and a bilateral filter
    still beats every wavelet variant by 2 dB for a quarter of the time. The
    premise being true is not the same as the method being the best available.
    """
    best_wavelet = max((r for r in scored.values() if r["method"].startswith("Wavelet")),
                       key=lambda r: r["psnr_db"])
    bilateral = scored["Bilateral (spatial)"]

    assert bilateral["psnr_db"] > best_wavelet["psnr_db"] + 1.5
    assert bilateral["median_ms"] < best_wavelet["median_ms"]


def test_the_bilateral_filter_wins_at_every_noise_level(sweep):
    for sigma, row in sweep.items():
        best_wavelet = max(row[m] for m in wd.METHODS if m.startswith("Wavelet"))
        assert row["Bilateral (spatial)"] > best_wavelet, sigma


def test_at_low_noise_most_methods_are_worse_than_doing_nothing(sweep):
    """The control that says when not to denoise at all."""
    quiet = sweep[min(sweep)]
    control = quiet["Do nothing (control)"]

    harmful = [m for m in wd.METHODS
               if m != "Do nothing (control)" and quiet[m] < control]
    assert len(harmful) >= 4, harmful
    assert "Wavelet VisuShrink soft" in harmful
    assert "Wavelet VisuShrink hard" in harmful


def test_soft_against_hard_depends_on_the_threshold_rule(scored):
    """So neither 'soft is better' nor 'hard is better' is a statement.

    Paired with VisuShrink's universal threshold, hard wins by 0.97 dB. Paired
    with BayesShrink's per-subband one, soft wins by 2.04. The two design
    choices are not independent.
    """
    visu_soft = scored["Wavelet VisuShrink soft"]["psnr_db"]
    visu_hard = scored["Wavelet VisuShrink hard"]["psnr_db"]
    bayes_soft = scored["Wavelet BayesShrink soft"]["psnr_db"]
    bayes_hard = scored["Wavelet BayesShrink hard"]["psnr_db"]

    assert visu_hard > visu_soft
    assert bayes_soft > bayes_hard
    assert (bayes_soft - bayes_hard) > 1.0


def test_visushrink_is_far_too_aggressive(scored, sweep):
    """Its universal threshold is chosen so noise is *never* exceeded, which
    guarantees it removes signal as well."""
    for rule in ("Wavelet VisuShrink soft", "Wavelet VisuShrink hard"):
        assert scored[rule]["psnr_db"] < scored["Wavelet BayesShrink soft"]["psnr_db"]

    quiet = sweep[min(sweep)]
    assert quiet["Wavelet VisuShrink soft"] < quiet["Do nothing (control)"] - 5.0


def test_no_single_threshold_matches_an_adaptive_rule():
    """The best global threshold, searched with the clean images in hand, loses.

    Which is the argument for BayesShrink stated as a number rather than as a
    motivation.
    """
    oracle = wd.oracle_threshold(noise_sigma=25.0)
    rows, _ = wd.evaluate_methods(noise_sigma=25.0, runs=1)
    bayes = next(r["psnr_db"] for r in rows if r["method"] == "Wavelet BayesShrink soft")

    assert oracle["best_psnr_db"] < bayes
    assert oracle["best_threshold"] > 0


def test_nlm_wants_about_six_tenths_of_the_noise_not_all_of_it():
    """The advice usually quoted is h ~ sigma. Measured, it is 0.6 sigma."""
    clean = [wd.load_scene(n) for n in wd.IMAGES[:4]]
    noisy = [synth.gaussian_noise(c, sigma=25.0, seed=0) for c in clean]

    at_sigma = float(np.mean([psnr(wd.denoise_nlm(n, h=25.0), c)
                              for n, c in zip(noisy, clean)]))
    at_six_tenths = float(np.mean([psnr(wd.denoise_nlm(n, h=15.0), c)
                                   for n, c in zip(noisy, clean)]))
    assert at_six_tenths > at_sigma + 0.5


# --------------------------------------------------------------------------- #
# the transform itself
# --------------------------------------------------------------------------- #


def test_the_haar_transform_is_invertible():
    """Decompose then reconstruct must return the original to floating error."""
    from shared.io import to_float, to_gray

    g = to_float(to_gray(wd.load_scene(wd.IMAGES[0])))
    # cropped to even dimensions: one Haar step halves each axis, so an odd size
    # loses its last row or column. `denoise_wavelet` handles that itself — this
    # checks the transform, not the pipeline.
    g = g[: g.shape[0] // 8 * 8, : g.shape[1] // 8 * 8]

    for levels in (1, 2, 3):
        ll, coeffs = wd.decompose(g, levels=levels)
        back = wd.reconstruct(ll, coeffs)
        assert back.shape == g.shape, levels
        assert float(np.abs(back - g).max()) < 1e-5, levels


def test_the_pipeline_returns_the_original_size_even_for_odd_dimensions():
    img = wd.load_scene(wd.IMAGES[0])
    assert img.shape[0] % 2 == 1 or img.shape[1] % 2 == 1, "pick an odd-sized photo"
    assert wd.denoise_wavelet(img).shape == img.shape


def test_soft_thresholding_shrinks_and_hard_thresholding_does_not():
    coeffs = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], np.float32)

    hard = wd.threshold_hard(coeffs, 1.0)
    soft = wd.threshold_soft(coeffs, 1.0)

    assert np.array_equal(hard, np.array([-3.0, 0.0, 0.0, 0.0, 3.0], np.float32))
    assert np.allclose(soft, np.array([-2.0, 0.0, 0.0, 0.0, 2.0], np.float32))
    assert np.abs(soft).max() < np.abs(hard).max(), "soft biases the large coefficients"


def test_more_levels_help_until_they_do_not():
    rows = sorted(wd.sweep_levels(), key=lambda r: r["levels"])
    key = next(k for k in rows[0] if k != "levels")
    values = [r[key] for r in rows]
    assert values[1] > values[0], "one level is not enough"
    assert max(values) - values[-1] < 1.0, "and it plateaus rather than collapsing"
