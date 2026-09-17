"""Tests for project 18, optical flow.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

Two of them exist because the harness was wrong in a way that looked like a
result, which is the failure mode this whole comparison is supposed to detect.
"""

from __future__ import annotations

import numpy as np
import pytest

import optical_flow as of
from shared import synth
from shared.metrics import endpoint_error

MARGIN = 16


def epe(pred, truth) -> float:
    return endpoint_error(pred[MARGIN:-MARGIN, MARGIN:-MARGIN],
                          truth[MARGIN:-MARGIN, MARGIN:-MARGIN])


# --------------------------------------------------------------------------- #
# the harness -- both of these were broken, and both looked like findings
# --------------------------------------------------------------------------- #


def test_the_truth_is_the_forward_flow():
    """`make_pair` must return the flow that carries frame 1 *to* frame 2.

    `synth.warp_by_flow` is a backward warp -- it samples the input at
    ``(x + dx, y + dy)`` -- so warping by ``f`` moves image content by ``-f``.
    Passing the field straight through scored every method against the exact
    negation of the truth, and **every one of them then scored worse than
    predicting no motion at all**: DIS came out at 8.93 px on a 4 px
    displacement instead of 0.013 px.

    That is a table full of plausible-looking numbers where the harness, not
    any method, was wrong. Checked here against DIS, which is accurate enough
    that a sign error cannot hide in it.
    """
    first, second, truth = of.make_pair("carved_mask_thatch", 4.0)
    pred = of.flow_dis(first, second)
    assert epe(pred, truth) < 0.1
    assert epe(pred, -truth) > 5.0   # and the wrong sign is unmistakably wrong


def test_the_gradient_is_scaled_to_a_real_derivative():
    """`cv2.Sobel(ksize=3)` is unnormalised: its output is 8x the derivative.

    LK and Horn-Schunck both divide a temporal difference by a spatial one, so
    an 8x-too-large gradient yields flow 8x too *small* -- pointing the right
    way, which is what made it look like a method that half-works. A true
    1.00 px displacement came back as 0.123 px, or 1/8.1.
    """
    assert of.SOBEL_SCALE == pytest.approx(1 / 8)

    first = of.load_scene("carved_mask_thatch")
    flow = synth.synthetic_flow(first.shape[:2], magnitude=0.5, kind="translation")
    second = synth.warp_by_flow(first, -flow)
    got = of.flow_lucas_kanade_dense(first, second)[100:-100, 100:-100]
    # within 20% of the true 0.5 px, rather than within 20% of 0.0625 px
    assert float(np.median(got[..., 0])) == pytest.approx(0.5, rel=0.2)


# --------------------------------------------------------------------------- #
# the operators
# --------------------------------------------------------------------------- #


def test_no_motion_is_reported_as_no_motion():
    img = of.load_scene("bighorn_rock")
    for name, fn in of.METHODS.items():
        flow = fn(img, img.copy())
        assert float(np.abs(flow).max()) < 0.5, f"{name} invented motion"


def test_every_method_returns_a_full_flow_field():
    first, second, truth = of.make_pair("stone_archway", 2.0)
    for name, fn in of.METHODS.items():
        flow = fn(first, second)
        assert flow.shape == truth.shape, name
        assert np.isfinite(flow).all(), f"{name} produced NaN or inf"


def test_an_unknown_flow_kind_raises():
    with pytest.raises(ValueError, match="unknown flow kind"):
        synth.synthetic_flow((64, 64), kind="swirl")


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_plain_lucas_kanade_dies_between_one_and_two_pixels():
    """The project's headline, and the number the textbooks do not give.

    "LK fails for large motion" is in every textbook. Measured here, the
    threshold is **1 px**: at 1 px its EPE is well under the displacement, and
    at 2 px it is already most of the way to what predicting zero scores.
    """
    images = of.IMAGES[:4]
    at_1 = of.evaluate_methods(magnitude=1.0, images=images)
    at_2 = of.evaluate_methods(magnitude=2.0, images=images)

    def rel(rows, name):
        return next(r["epe_relative"] for r in rows if r["method"] == name)

    assert rel(at_1, "Lucas-Kanade (dense)") < 0.5    # working
    assert rel(at_2, "Lucas-Kanade (dense)") > 0.6    # broken
    assert rel(at_2, "Predict zero (control)") == pytest.approx(1.118, abs=0.01)


def test_each_pyramid_level_roughly_doubles_the_tractable_motion():
    """The quantitative claim, confirmed: 1, 4, 8, 16, 32 px for 0-4 levels.

    The step from 0 to 1 level is a factor of 4 rather than 2 — the coarse level
    both halves the displacement *and* blurs away the fine structure that the
    linearisation was tripping over. Every level after that doubles, as the
    2^L story predicts.
    """
    rows = of.sweep_pyramid_levels(images=of.IMAGES[:4])
    limits = [r["max_tractable_px"] for r in rows]

    assert limits[0] == 1.0
    assert all(limit is not None for limit in limits)
    for previous, current in zip(limits, limits[1:]):
        assert current >= 2 * previous


def test_dis_is_the_only_method_that_survives_a_32_pixel_displacement():
    """And by two orders of magnitude, which is worth stating plainly."""
    rows = of.evaluate_methods(magnitude=32.0, images=of.IMAGES[:4])
    scored = {r["method"]: r["epe_relative"] for r in rows}

    assert scored["DIS"] < 0.02
    for name in ("Lucas-Kanade (dense)", "LK pyramid (3 levels)", "Horn-Schunck"):
        assert scored[name] > 0.5, name


def test_horn_schunck_is_unfinished_at_the_textbook_iteration_count():
    """Not a failed method — an unconverged one, and the difference matters.

    Jacobi iteration propagates information one pixel per sweep, so a global
    method on a 481 px frame needs hundreds of sweeps. At the textbook 100 it
    scores near the do-nothing control; the error keeps falling all the way to
    3000, at roughly 30x the cost.
    """
    rows = of.sweep_hs_iterations(images=of.IMAGES[:4])
    epes = [r["epe_px"] for r in rows]
    times = [r["median_ms"] for r in rows]

    assert epes == sorted(epes, reverse=True), "more iterations must not hurt"
    assert epes[-1] < 0.5 * epes[1]        # 3000 is at least twice as good as 100
    assert times[-1] > 20 * times[1]       # and costs at least 20x as much
    assert of.HS_ITERS > 100               # so the table does not quote the 100


def test_every_method_beats_predicting_zero_at_a_small_displacement():
    """The floor. A method scoring above this line is worse than useless."""
    rows = of.evaluate_methods(magnitude=1.0, images=of.IMAGES[:4])
    control = next(r for r in rows if r["method"].startswith("Predict zero"))
    for r in rows:
        if r["method"] != control["method"]:
            assert r["epe_px"] < control["epe_px"], r["method"]
