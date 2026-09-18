"""Tests for project 37, template matching.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

The invariance claims here are algebra, not empirical tendencies, so two of the
tests check the formula directly with no image search involved — if ZNCC of a
patch against `a*patch + b` is not 1.0, nothing downstream is worth measuring.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import template_matching as tm

from shared.io import to_float, to_uint8


@pytest.fixture(scope="module")
def offsets():
    return {r["offset"]: r for r in tm.sweep_brightness_offset()}


@pytest.fixture(scope="module")
def gains():
    return {r["gain"]: r for r in tm.sweep_brightness_gain()}


# --------------------------------------------------------------------------- #
# the algebra, with no search involved
# --------------------------------------------------------------------------- #


def test_zncc_of_a_patch_against_a_linear_transform_of_itself_is_exactly_one():
    """Subtracting the mean removes b; dividing by the deviation removes a.

    This is arithmetic, so it holds to five decimal places or the implementation
    is wrong. NCC only manages the second half, which is the whole difference
    between the two.
    """
    rng = np.random.default_rng(0)
    patch = to_uint8(rng.random((32, 32)).astype(np.float32))

    for gain, offset in ((1.0, 0.0), (0.6, 0.0), (0.6, 0.2), (0.4, 0.1)):
        transformed = to_uint8(to_float(patch) * gain + offset)
        zncc = float(cv2.matchTemplate(transformed, patch, cv2.TM_CCOEFF_NORMED)[0, 0])
        assert zncc == pytest.approx(1.0, abs=1e-4), (gain, offset)


def test_ncc_survives_a_gain_and_not_an_offset_in_the_formula():
    rows = {(r["gain"], r["offset"]): r for r in tm.verify_formula_invariance()}

    pure_gain = rows[(0.6, 0.0)]
    assert pure_gain["NCC (CCORR_NORMED)"] == pytest.approx(1.0, abs=1e-4)

    with_offset = rows[(1.0, 0.2)]
    assert with_offset["NCC (CCORR_NORMED)"] < 0.99
    assert with_offset["ZNCC (CCOEFF_NORMED)"] > with_offset["NCC (CCORR_NORMED)"]


def test_taking_the_max_of_a_sqdiff_map_would_find_the_worst_location():
    """SQDIFF is a distance. The min/max choice is not cosmetic."""
    scene, template, truth = tm.make_case(tm.IMAGES[0], seed=0)
    result = cv2.matchTemplate(scene, template, cv2.TM_SQDIFF)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)

    assert tm.localisation_error(min_loc, truth) == 0.0
    assert tm.localisation_error(max_loc, truth) > 50.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_raw_cross_correlation_fails_on_undegraded_photographs():
    """The control that shows normalising is not a refinement but the method.

    Unnormalised correlation is maximised by whatever region has the largest
    magnitude, so it finds the brightest patch rather than the matching one. It
    localises about one template in nine with nothing wrong with the scene at
    all, 160 px out on average.
    """
    rows = {r["method"]: r for r in tm.evaluate_methods(runs=1)}
    cc = rows["Cross-correlation"]

    assert cc["success_rate"] < 0.2
    assert cc["mean_error_px"] > 100

    for name in ("SSD (SQDIFF)", "NCC (CCORR_NORMED)", "ZNCC (CCOEFF_NORMED)"):
        assert rows[name]["success_rate"] == 1.0, name


def test_ncc_is_exactly_scale_invariant_and_ssd_is_not(gains):
    """A tenth of the brightness, and NCC does not notice."""
    darkest = min(gains)
    assert darkest <= 0.1

    assert gains[darkest]["NCC (CCORR_NORMED)"] == 1.0
    assert gains[darkest]["ZNCC (CCOEFF_NORMED)"] == 1.0
    assert gains[darkest]["SSD normalised"] == 0.0
    assert gains[darkest]["SSD (SQDIFF)"] < 0.2


def test_only_zncc_survives_an_offset_and_it_takes_a_negative_one_to_show_it(offsets):
    """The project's headline, and the sign of the offset is the whole trick.

    Brightening clips at white and destroys the structure for every method
    equally — at +0.4 NCC and ZNCC both sit at 0.958 and the difference between
    them is invisible. Darkening clips only in the shadows, and there NCC falls
    to 0.458 while ZNCC holds 1.000 at every level.
    """
    assert offsets[-0.4]["ZNCC (CCOEFF_NORMED)"] == 1.0
    assert offsets[-0.4]["NCC (CCORR_NORMED)"] < 0.6
    assert offsets[-0.4]["SSD (SQDIFF)"] < 0.4

    negatives = sorted(o for o in offsets if o < 0)
    for o in negatives:
        assert offsets[o]["ZNCC (CCOEFF_NORMED)"] == 1.0, o

    # a positive offset of the same size hides the difference entirely
    assert offsets[0.4]["NCC (CCORR_NORMED)"] == pytest.approx(
        offsets[0.4]["ZNCC (CCOEFF_NORMED)"], abs=0.05)


def test_zncc_and_ncc_localise_identically_and_differ_357x_in_confidence():
    """Which is why the response surface is reported and not only the error.

    On an undegraded scene both localise every template perfectly. ZNCC's peak
    stands 434x above the surrounding mean and NCC's 1.22x — so a score
    threshold means something for one of them and nothing for the other.
    """
    clean = {r["method"]: r for r in tm.evaluate_methods(runs=1)}
    assert clean["NCC (CCORR_NORMED)"]["success_rate"] == \
        clean["ZNCC (CCOEFF_NORMED)"]["success_rate"] == 1.0

    sharp = {r["method"]: r["peak_to_mean"] for r in tm.response_sharpness()}
    assert sharp["ZNCC (CCOEFF_NORMED)"] > 100 * sharp["NCC (CCORR_NORMED)"]
    assert sharp["ZNCC (CCOEFF_NORMED)"] == max(sharp.values())


def test_scale_is_the_failure_no_scoring_function_fixes():
    """Every method collapses together, which is what says it is not about scoring."""
    rows = {r["scale"]: r for r in tm.sweep_scale()}
    worst = max(rows, key=lambda s: abs(s - 1.0))

    for name in tm.METHODS:
        assert rows[1.0][name] >= rows[worst][name], name
    assert max(rows[worst][m] for m in tm.METHODS) < 0.2


def test_a_pyramid_search_fixes_scale_completely_and_costs_20x():
    rows = tm.multiscale_benefit()
    for r in rows:
        assert r["multi_scale_success"] == 1.0, r["true_scale"]
        assert r["scale_estimate_error"] < 0.01, r["true_scale"]

    worst = min(rows, key=lambda r: r["single_scale_success"])
    assert worst["single_scale_success"] < 0.2
    assert worst["multi_ms"] > 10 * worst["single_ms"]


def test_noise_is_the_degradation_template_matching_barely_notices():
    """A 64x64 template is a 4096-pixel average, and averaging is what kills noise."""
    rows = sorted(tm.sweep_noise(), key=lambda r: r["noise_sigma"])
    for name in ("SSD (SQDIFF)", "NCC (CCORR_NORMED)", "ZNCC (CCOEFF_NORMED)"):
        assert rows[-1][name] > 0.8, name


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_the_true_location_is_exact_on_an_undegraded_scene():
    for name in tm.IMAGES[:4]:
        scene, template, truth = tm.make_case(name, seed=0)
        patch = scene[truth[1]:truth[1] + template.shape[0],
                      truth[0]:truth[0] + template.shape[1]]
        assert np.array_equal(patch, template), name


def test_the_template_is_not_degraded_only_the_scene():
    """The realistic direction: the template is what you stored."""
    plain = tm.make_case(tm.IMAGES[0], seed=0)
    dark = tm.make_case(tm.IMAGES[0], brightness_offset=-0.4, seed=0)

    assert np.array_equal(plain[1], dark[1]), "the template must be untouched"
    assert dark[0].mean() < plain[0].mean() - 40


def test_the_pool_spans_the_entropy_axis_it_was_selected_on():
    from shared import io

    values = []
    for name in tm.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(tm.entropy(tm.load_scene(name)))

    assert len(set(tm.IMAGES)) == len(tm.IMAGES)
    assert min(values) < 6.0, "no flat scene to make matching ambiguous"
    assert max(values) > 7.8
