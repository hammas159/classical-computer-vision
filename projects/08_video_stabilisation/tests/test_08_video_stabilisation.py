"""Tests for project 08, video stabilisation.

The two worth reading first are `test_the_smoother_matters_more_than_the_estimator`
and `test_a_stabiliser_invents_motion_in_footage_that_never_moved`. The first is
the project's result: motion estimation is not the bottleneck, and swapping the
path model changes the output eight times as much as swapping the estimator. The
second is the control nobody runs — on a tripod shot, where the true answer is
exactly zero, ECC accumulates 25 px of drift.

`test_a_known_warp_is_recovered_exactly` is the regression test for two sign
errors that a round trip caught: the rotation read back out of the affine was
negated, and phase correlation's translation was negated on top of that, which
made it score worse than reporting no motion at all.

The tests are deliberately run on short segments and subsets of the estimators;
`run.py` is what covers all twelve segments.
"""

from __future__ import annotations

import numpy as np
import pytest

import stabilise as st

pytestmark = pytest.mark.skipif(
    not st.video_available(),
    reason="vtest.avi not cached; run `python tools/fetch_assets.py --set video`",
)

FAST = ("Features + LK + RANSAC", "Phase correlation", "Block matching")


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_smoother_matters_more_than_the_estimator():
    """The estimator is not the bottleneck; the path model is."""
    limits = st.what_limits_the_result(starts=st.STARTS[:4])
    assert limits["smoother_matters_more"], limits
    assert limits["smoother_spread"] > 4 * limits["estimator_spread"], limits


def test_the_causal_smoother_is_the_worst_one():
    """A Kalman filter can only lag, and lag is what a viewer sees as shake."""
    limits = st.what_limits_the_result(starts=st.STARTS[:4])
    by_smoother = limits["varying_the_smoother"]
    assert (by_smoother["Kalman (causal)"]
            > 3 * by_smoother["Gaussian (sigma=8)"]), by_smoother


def test_a_stabiliser_invents_motion_in_footage_that_never_moved():
    """The truth here is exactly zero and nothing reports zero."""
    rows = {r["estimator"]: r for r in st.zero_motion_control(starts=st.STARTS[:4])}
    assert rows["No motion (control)"]["invented_motion_px_per_frame"] == 0.0
    for name in FAST:
        assert rows[name]["invented_motion_px_per_frame"] > 0.0, name
    # and because the path is an integral, the invention accumulates
    assert max(rows[n]["worst_accumulated_drift_px"] for n in FAST) > 1.0, rows


def test_every_estimator_beats_reporting_no_motion():
    """If one did not, it would be adding nothing over the control."""
    rows = {r["estimator"]: r for r in st.evaluate_estimators(starts=st.STARTS[:4])}
    control = rows["No motion (control)"]["translation_error_px"]
    for name in FAST:
        assert rows[name]["translation_error_px"] < control / 5, name


def test_stability_is_bought_with_field_of_view():
    """A wider smoother is steadier and crops more, monotonically."""
    rows = st.crop_versus_stability(starts=st.STARTS[:3])
    jitters = [r["residual_jitter"] for r in rows]
    crops = [r["crop"] for r in rows]
    assert jitters == sorted(jitters, reverse=True), rows
    assert crops == sorted(crops), rows
    assert jitters[0] > 5 * jitters[-1]


def test_phase_correlation_cannot_see_rotation():
    """Stated as a property rather than hidden as a weakness."""
    rows = {r["estimator"]: r for r in st.evaluate_estimators(starts=st.STARTS[:3])}
    control = rows["No motion (control)"]["rotation_error_deg"]
    assert rows["Phase correlation"]["rotation_error_deg"] == pytest.approx(
        control, rel=1e-6)
    assert rows["Features + LK + RANSAC"]["rotation_error_deg"] < control / 5


def test_block_matching_fails_when_the_motion_leaves_its_search_window():
    """A fixed +/-24 px search is a hard limit, and the sweep finds it."""
    rows = {r["translation_step_px"]: r for r in
            st.sweep_shake(starts=st.STARTS[:3], amounts=(2.0, 16.0))}
    small, large = rows[2.0], rows[16.0]
    assert small["Block matching"] < small["Features + LK + RANSAC"] * 5
    assert large["Block matching"] > large["Features + LK + RANSAC"] * 2


# --------------------------------------------------------------------------- #
# the two sign errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pose", [(5.0, 0.0, 0.0), (0.0, 4.0, 0.0),
                                  (0.0, 0.0, 2.0), (3.0, -2.0, 1.5)])
def test_a_known_warp_is_recovered_exactly(pose):
    """Warp a frame by a known pose and estimate it back.

    Both sign errors this project shipped in its first version fail here: the
    rotation came back negated, and phase correlation's translation did too.
    """
    frame = st.load_frames(100, 1)[0]
    warped = st.apply_shake([frame, frame],
                            np.array([[0, 0, 0], pose], float))[1]
    for name in ("Features + LK + RANSAC", "ECC (direct)", "Block matching"):
        got = st.ESTIMATORS[name](frame, warped)
        assert np.allclose(got, pose, atol=0.15), (name, pose, got)


def test_phase_correlation_recovers_translation_with_the_right_sign():
    frame = st.load_frames(100, 1)[0]
    warped = st.apply_shake([frame, frame],
                            np.array([[0, 0, 0], [5.0, -3.0, 0.0]], float))[1]
    got = st.ESTIMATORS["Phase correlation"](frame, warped)
    assert np.allclose(got[:2], [5.0, -3.0], atol=0.3), got


def test_the_rotation_sign_convention_round_trips():
    """A pure algebra check, with no image in it."""
    for angle in (-3.0, -0.5, 0.5, 3.0):
        M = st.to_matrix((0.0, 0.0, angle))
        got = st._pose_from_affine(M, (480, 640))
        assert got[2] == pytest.approx(angle, abs=1e-6), (angle, got)


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


def test_the_shake_is_a_shake_and_not_a_pan():
    """An earlier version wandered 277 px across a 768 px frame."""
    path = st.camera_path(st.SEGMENT, seed=0)
    assert float(np.abs(path[:, :2]).max()) < 60.0, path[:, :2].max()
    assert float(path[:, :2].std()) > 1.0


def test_the_truth_is_the_matrix_that_was_used_not_an_estimate():
    import inspect

    source = inspect.getsource(st.shaken_segment)
    assert "camera_path" in source and "apply_shake" in source
    assert "estimate" not in source


def test_the_original_clip_really_is_static():
    """The zero-motion control means nothing if the camera moved.

    Checked against the frames themselves rather than assumed: the median
    absolute difference between frames far apart is small outside the regions
    where people walk.
    """
    frames = st.load_frames(10, 40)
    first = st.to_gray(frames[0]).astype(np.float32)
    last = st.to_gray(frames[-1]).astype(np.float32)
    diff = np.abs(first - last)
    # most of the frame is unchanged; a camera that moved would change all of it
    assert float(np.median(diff)) < 6.0, float(np.median(diff))


def test_the_effective_path_is_the_smooth_one_when_estimation_is_perfect():
    """The identity the whole pipeline rests on."""
    truth = st.camera_path(40, seed=3)
    smoothed = st.smooth_gaussian(truth, sigma=8.0)
    effective = st.effective_path(truth, truth, smoothed)
    assert np.allclose(effective, smoothed)


def test_jitter_ignores_a_smooth_pan():
    """Acceleration, not velocity: a steady pan is not shake."""
    n = 60
    pan = np.stack([np.linspace(0, 50, n), np.zeros(n), np.zeros(n)], axis=1)
    shaky = pan + np.random.default_rng(0).normal(0, 1.0, (n, 3))
    assert st.jitter(pan) < 1e-9
    assert st.jitter(shaky) > 0.5


def test_the_crop_grows_with_the_correction():
    shape = (576, 768, 3)
    small = np.tile(np.array([[1.0, 1.0, 0.1]]), (20, 1))
    large = np.tile(np.array([[20.0, 20.0, 2.0]]), (20, 1))
    assert st.required_crop(small, shape) < st.required_crop(large, shape)
    assert st.required_crop(np.zeros((20, 3)), shape) == pytest.approx(0.0)


def test_composing_two_affines_is_apply_b_then_a():
    a = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 5.0]])
    point = np.array([1.0, 2.0, 1.0])
    assert np.allclose(st._compose(a, b) @ point, [11.0, 7.0])


def test_stabilise_returns_one_frame_per_input():
    _, shaken, _ = st.shaken_segment(st.STARTS[0], seed=0, count=12)
    out, estimated, smoothed = st.stabilise(shaken, "Features + LK + RANSAC",
                                            "Gaussian (sigma=8)")
    assert len(out) == len(shaken)
    assert estimated.shape == (len(shaken), 3)
    assert smoothed.shape == (len(shaken), 3)
