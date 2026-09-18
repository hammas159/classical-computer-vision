"""Tests for project 39, white balance.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_the_clipped_highlight_scene_is_built_in_the_right_order` exists because the
first version painted the white highlight and *then* applied the cast, which made
the highlight carry the illuminant perfectly — white-patch scored 0.0 degrees on
the scene designed to break it.
"""

from __future__ import annotations

import numpy as np
import pytest

import white_balance as wb


@pytest.fixture(scope="module")
def failures():
    return {r["scene"]: r for r in wb.assumption_failure_tests()}


@pytest.fixture(scope="module")
def casts():
    return {r["cast"]: r for r in wb.compare_casts()}


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_the_clipped_highlight_scene_is_built_in_the_right_order():
    """The cast goes on first; the blown highlight is painted over it.

    A blown highlight is clipped at the sensor, which happens *after* the light
    has been multiplied in, so its channel ratios are 1:1:1 and carry nothing
    about the illuminant. Painting it first and casting afterwards produces the
    opposite: a perfect white reference, which is white-patch's ideal case.
    """
    image, truth = wb.blown_highlight_scene(seed=0)
    centre = image[image.shape[0] // 2, image.shape[1] // 2]

    assert (centre == 255).all(), "the highlight must be clipped in every channel"
    assert wb.angular_error(truth, np.ones(3)) > 5.0, "and the light must not be white"

    # the brightest pixel therefore says 'neutral' while the light is warm
    assert wb.angular_error(wb.est_white_patch(image, percentile=100.0), truth) > 10.0


def test_a_known_cast_is_recoverable_in_principle():
    """The truth is the gains that were applied, so an oracle would score zero."""
    for name in wb.IMAGES[:4]:
        cast, clean, truth = wb.make_case(name, gains=(1.35, 1.0, 0.65))
        assert wb.angular_error(truth, truth) == pytest.approx(0.0, abs=1e-9)
        assert wb.angular_error(np.ones(3), truth) > 5.0, name
        assert cast.shape == clean.shape


def test_the_pool_spans_the_axis_it_was_selected_on():
    from shared import io

    deviations = []
    for name in wb.IMAGES:
        assert name in io.REAL_PHOTOS, name
        deviations.append(wb.grey_deviation(wb.load_scene(name)))

    assert len(set(wb.IMAGES)) == len(wb.IMAGES)
    assert min(deviations) < 1.5, "no scene where grey-world's assumption nearly holds"
    assert max(deviations) > 25.0, "no scene where it cannot hold"


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_a_single_clipped_highlight_reduces_white_patch_to_doing_nothing(failures):
    """The project's sharpest result, and it is an exact equality.

    White-patch takes the brightest pixel as the illuminant. Given a clipped
    highlight that pixel is (255, 255, 255), so the method concludes the light is
    white and applies no correction — landing on precisely the do-nothing
    control, to three decimal places.
    """
    row = failures["Clipped highlight, warm light (breaks white-patch)"]
    assert row["White-patch (true max)"] == pytest.approx(
        row["Do nothing (control)"], abs=0.01)

    # the percentile variant exists for exactly this and it works
    assert row["White-patch (99th pct)"] < 0.6 * row["White-patch (true max)"]
    # and the averaging methods barely notice one circle
    assert row["Grey-world"] < 0.5 * row["White-patch (true max)"]


def test_a_dominant_colour_breaks_grey_world_and_not_the_edge_methods(failures):
    """Under a neutral light, so any error is a hallucinated illuminant."""
    row = failures["Dominant colour, neutral light (breaks grey-world)"]
    assert row["Do nothing (control)"] == 0.0
    assert row["Grey-world"] > 20.0
    assert row["Grey-edge (p=6)"] < 0.2 * row["Grey-world"]


def test_grey_world_error_grows_with_how_much_the_assumption_is_violated():
    """1.17 degrees at 0% dominance, 26.21 at 80% — a straight line through it."""
    rows = sorted(wb.sweep_dominance(), key=lambda r: r["dominant_fraction"])
    errors = [r["Grey-world"] for r in rows]

    assert errors == sorted(errors), "it should be monotone in the violation"
    assert errors[-1] > 20 * errors[0]

    edge = [r["Grey-edge (p=6)"] for r in rows]
    assert max(edge) - min(edge) < 3.0, "grey-edge should barely move"


def test_every_method_makes_an_almost_uncast_image_worse(casts):
    """The control that says when not to run any of this.

    Under a light that is already nearly white, doing nothing scores 0.94
    degrees and every estimator here scores worse — grey-world by a factor of
    ten. These methods have no way to decide there is nothing to correct.
    """
    row = casts["daylight (neutral)"]
    control = row["Do nothing (control)"]
    assert control < 1.5

    worse = [m for m in wb.ESTIMATORS
             if m != "Do nothing (control)" and row[m] > control]
    assert len(worse) == len(wb.ESTIMATORS) - 1, row
    assert row["Grey-world"] > 5 * control


def test_the_usual_minkowski_default_is_set_for_the_wrong_method():
    """p=6 is quoted everywhere. It suits shades-of-grey and costs grey-edge 43%."""
    rows = {r["p"]: r for r in wb.sweep_minkowski_p()}

    best_sg = min(rows, key=lambda p: rows[p]["shades_of_grey_deg"])
    best_ge = min(rows, key=lambda p: rows[p]["grey_edge_deg"])
    assert best_sg >= 6.0
    assert best_ge <= 4.0, "grey-edge wants a low exponent"

    at_six = rows[6.0]["grey_edge_deg"]
    assert at_six > 1.3 * rows[best_ge]["grey_edge_deg"]


def test_noise_breaks_the_brightest_pixel_and_not_the_mean():
    """Because noise invents bright pixels, and that is white-patch's whole input.

    Grey-world averages over every pixel, and zero-mean noise does not move a
    mean — its error is flat or slightly better under noise. White-patch reads
    one pixel and gets steadily worse.
    """
    rows = sorted(wb.sweep_noise(), key=lambda r: r["noise_sigma"])
    assert rows[-1]["White-patch (true max)"] > 1.5 * rows[0]["White-patch (true max)"]
    assert rows[-1]["Grey-world"] <= rows[0]["Grey-world"] + 0.5

    # and again the percentile variant is the mitigation
    assert rows[-1]["White-patch (99th pct)"] < rows[-1]["White-patch (true max)"]


def test_angular_error_and_psnr_can_rank_differently():
    """They are different questions: estimating the light, and applying it.

    Reported side by side because a method can estimate the illuminant slightly
    better and still produce a worse-looking image, and neither column is the
    other's proxy.
    """
    rows = {r["method"]: r for r in wb.evaluate_estimators(runs=1)}
    real = [m for m in rows if m != "Do nothing (control)"]

    by_angle = sorted(real, key=lambda m: rows[m]["angular_error_deg"])
    by_psnr = sorted(real, key=lambda m: -rows[m]["psnr_db"])
    assert by_angle != by_psnr, "if these ever agree exactly, say so instead"

    for m in real:
        assert rows[m]["psnr_db"] > rows["Do nothing (control)"]["psnr_db"], m


# --------------------------------------------------------------------------- #
# the estimators themselves
# --------------------------------------------------------------------------- #


def test_every_estimator_returns_a_positive_three_vector():
    cast, _, _ = wb.make_case(wb.IMAGES[0], gains=(1.35, 1.0, 0.65))
    for name, fn in wb.ESTIMATORS.items():
        v = fn(cast)
        assert v.shape == (3,), name
        assert np.isfinite(v).all(), name
        assert (v > 0).all(), name


def test_the_control_estimates_a_white_light_by_definition():
    cast, _, _ = wb.make_case(wb.IMAGES[0], gains=(1.35, 1.0, 0.65))
    v = wb.normalise_illuminant(wb.est_none(cast))
    assert v == pytest.approx(wb.normalise_illuminant(np.ones(3)), abs=1e-9)


def test_correcting_by_the_true_illuminant_recovers_the_original():
    """The oracle. If this is not near-exact, the correction is wrong, not the estimate."""
    for name in wb.IMAGES[:3]:
        cast, clean, truth = wb.make_case(name, gains=(1.35, 1.0, 0.65))
        restored = wb.apply_correction(cast, truth)
        error = np.abs(restored.astype(float) - clean.astype(float)).mean()
        assert error < 12.0, f"{name}: {error:.2f} mean absolute error"
        assert error < np.abs(cast.astype(float) - clean.astype(float)).mean()
