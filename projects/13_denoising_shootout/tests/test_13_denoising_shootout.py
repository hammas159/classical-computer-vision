"""Tests for project 13, the denoising shootout.

Most of these pin a *finding* rather than a number, so that a later "improvement"
which quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import denoising as dn
from shared import io
from shared.metrics import psnr

IMAGES = ("astronaut", "camera")


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind,level,_label", dn.NOISE_TYPES)
def test_every_noise_model_actually_damages_the_image(kind, level, _label):
    clean = io.sample("astronaut")
    noisy = dn.make_noisy(clean, kind, level, seed=0)
    assert noisy.shape == clean.shape and noisy.dtype == np.uint8
    assert psnr(noisy, clean) < 25.0


def test_only_impulse_noise_produces_ISOLATED_extreme_pixels():
    """The three models differ in kind, not just in degree — which is the premise.

    Counting extreme pixels alone does not show it: `astronaut` has 11.2% of its
    pixels at 0 or 255 with no noise at all, and Poisson clipping adds another
    1.7%. Requiring the extreme pixel to also disagree with its own 5x5 median
    separates isolated speckle from solid dark regions, and the margin is 300x.
    """
    import cv2

    def isolated(image):
        g = io.to_gray(image)
        extreme = (g == 0) | (g == 255)
        return float((extreme & (cv2.absdiff(g, cv2.medianBlur(g, 5)) > 60)).mean())

    for name in ("astronaut", "camera", "brick"):
        clean = io.sample(name)
        scores = {k: isolated(dn.make_noisy(clean, k, lv, seed=0)) for k, lv, _ in dn.NOISE_TYPES}
        assert scores["salt_pepper"] > 0.03, (name, scores)
        assert scores["gaussian"] < 0.001, (name, scores)
        assert scores["poisson"] < 0.001, (name, scores)
        assert isolated(clean) < 0.001, name


def test_an_unknown_noise_kind_raises():
    with pytest.raises(ValueError):
        dn.make_noisy(io.sample("camera"), "not a noise model", 1.0)


# --------------------------------------------------------------------------- #
# the filters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(dn.METHODS))
def test_every_filter_returns_rgb_uint8_of_the_same_shape(name):
    noisy = dn.make_noisy(io.sample("coffee"), "gaussian", 25.0, seed=0)
    out = dn.METHODS[name](noisy)
    assert out.dtype == np.uint8 and out.shape == noisy.shape


def test_the_control_returns_its_input_unchanged():
    noisy = dn.make_noisy(io.sample("camera"), "gaussian", 25.0, seed=0)
    assert np.array_equal(dn.denoise_identity(noisy), noisy)


@pytest.mark.parametrize("name", [m for m in dn.METHODS if not m.startswith("Do nothing")])
def test_every_filter_beats_doing_nothing_at_this_noise_level(name):
    """At sigma 25 every filter here should help. The level sweep shows where that stops."""
    clean = io.sample("astronaut")
    noisy = dn.make_noisy(clean, "gaussian", 25.0, seed=0)
    assert psnr(dn.tuned_call(name, noisy, "gaussian"), clean) > psnr(noisy, clean)


def test_the_wiener_filter_leaves_a_clean_image_nearly_alone():
    """Its whole idea is adaptivity: no local variance excess means no smoothing."""
    clean = io.sample("coffee")
    assert psnr(dn.denoise_wiener(clean), clean) > 30.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_three_noise_models_have_three_different_winners():
    """The project's headline, pinned.

    "There is no best denoiser" is the claim, and this is what makes it a
    measurement rather than a slogan.
    """
    rows = dn.compare_noise_types_tuned(images=dn.IMAGES)
    winners = set()
    for row in rows:
        scores = {k: v for k, v in row.items() if k not in ("noise", "Do nothing (control)")}
        winners.add(max(scores, key=scores.get))
    assert len(winners) == len(rows) == 3


def test_median_dominates_impulse_noise_and_loses_on_gaussian():
    """The role reversal, pinned in both directions.

    Half of this is folklore everyone repeats; the other half is the part nobody
    mentions, and it is the same size of effect.
    """
    rows = dn.compare_noise_types_tuned(images=dn.IMAGES)
    by_noise = {r["noise"]: r for r in rows}
    sp = by_noise["Salt & pepper 6%"]
    gauss = by_noise["Gaussian sigma=25"]

    others_sp = [v for k, v in sp.items() if k not in ("noise", "Median", "Do nothing (control)")]
    assert sp["Median"] > max(others_sp) + 5.0

    others_g = [v for k, v in gauss.items() if k not in ("noise", "Median", "Do nothing (control)")]
    assert gauss["Median"] <= min(others_g) + 0.1  # worst or tied-worst


def test_the_bilateral_filter_reverses_the_median_result():
    rows = dn.compare_noise_types_tuned(images=dn.IMAGES)
    by_noise = {r["noise"]: r for r in rows}
    sp, gauss = by_noise["Salt & pepper 6%"], by_noise["Gaussian sigma=25"]
    # best on Gaussian, near the bottom on impulse
    assert gauss["Bilateral"] > gauss["Median"]
    assert sp["Bilateral"] < sp["Median"] - 5.0


def test_tuning_transfers_for_only_one_filter():
    """The project's second finding, pinned.

    A six-image grid search over one free parameter is about as small as
    overfitting gets. It still happens: fitted on three images and scored on
    three others, most filters' "tuned" parameter is worth nothing and one is
    worth 4 dB.
    """
    rows = dn.transfer_check()
    gains = {r["method"]: r["transfer_gain_db"] for r in rows}
    assert gains["Bilateral"] > 2.0
    assert sum(1 for g in gains.values() if g > 0.5) == 1
    assert min(gains.values()) < 0  # at least one fitted parameter LOSES on held-out data


def test_the_expensive_filter_is_orders_of_magnitude_slower():
    rows, _ = dn.evaluate_methods(images=IMAGES, runs=1)
    by_name = {r["method"]: r for r in rows}
    assert by_name["Non-local means"]["median_ms"] > 100 * by_name["Gaussian"]["median_ms"]


def test_denoising_can_be_worse_than_doing_nothing_at_low_noise():
    """The practically useful part: knowing when not to bother."""
    rows = dn.sweep_level(images=IMAGES, levels=(5.0,))
    row = rows[0]
    hurt = [m for m in dn.METHODS if not m.startswith("Do nothing") and row[m] < row["noisy_input"]]
    assert hurt, "at sigma 5 at least one filter should cost more than the noise does"


def test_the_level_sweep_degrades_monotonically_for_the_input():
    rows = dn.sweep_level(images=IMAGES, levels=(5.0, 20.0, 50.0))
    inputs = [r["noisy_input"] for r in rows]
    assert inputs == sorted(inputs, reverse=True)


# --------------------------------------------------------------------------- #
# the stored tuning
# --------------------------------------------------------------------------- #


def test_every_noise_model_has_a_tuned_parameter_for_every_filter():
    for kind, _, _ in dn.NOISE_TYPES:
        assert kind in dn.TUNED
        for method in dn.METHODS:
            if method.startswith("Do nothing"):
                continue
            assert method in dn.TUNED[kind], (kind, method)


def test_tuned_call_actually_uses_the_stored_parameter():
    noisy = dn.make_noisy(io.sample("camera"), "salt_pepper", 0.06, seed=0)
    param, value = dn.TUNED["salt_pepper"]["Median"]
    assert np.array_equal(
        dn.tuned_call("Median", noisy, "salt_pepper"), dn.denoise_median(noisy, **{param: value})
    )


def test_tuned_call_falls_back_to_the_default_for_an_unknown_noise_kind():
    noisy = dn.make_noisy(io.sample("camera"), "gaussian", 25.0, seed=0)
    assert np.array_equal(dn.tuned_call("Median", noisy, "unknown"), dn.denoise_median(noisy))


def test_denoise_dispatches_by_name():
    noisy = dn.make_noisy(io.sample("camera"), "gaussian", 25.0, seed=0)
    assert np.array_equal(dn.denoise(noisy, "Median"), dn.denoise_median(noisy))
    with pytest.raises(KeyError):
        dn.denoise(noisy, "not a filter")
