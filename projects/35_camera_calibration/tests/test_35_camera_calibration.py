"""Tests for project 35, camera calibration.

The two worth reading first are
`test_the_reported_error_and_the_real_one_move_in_opposite_directions` and
`test_the_lowest_reported_error_comes_with_a_nonsense_principal_point`. Between
them they pin the project's result: the number `cv2.calibrateCamera` returns —
the one every calibration write-up quotes — gets *better* as the calibration gets
worse, in two independent ways.

`test_straightness_is_not_what_calibration_optimises` is the reason the project
can say that at all: a check that lives outside the objective function.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import calibration as cal

pytestmark = pytest.mark.skipif(
    not cal.assets_available(),
    reason="calibration images not cached; run "
           "`python tools/fetch_assets.py --set calibration`",
)


@pytest.fixture(scope="module")
def models():
    return cal.evaluate_models()


@pytest.fixture(scope="module")
def sweep():
    return cal.sweep_view_count(trials=3)


@pytest.fixture(scope="module")
def full():
    return cal.full_calibration()


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_reported_error_and_the_real_one_move_in_opposite_directions(sweep):
    """Three views: lowest reported error in the project, worst real one."""
    rows = {r["views"]: r for r in sweep}
    three, ten = rows[3], rows[10]
    assert three["rms_fitted"] < ten["rms_fitted"], (three, ten)
    assert three["rms_held_out"] > ten["rms_held_out"], (three, ten)


def test_more_coefficients_always_lower_the_reported_error(models):
    """Which is what makes it useless as a model-selection criterion."""
    rows = models["rows"]
    by_count = sorted(rows, key=lambda r: r["coefficients"])
    fitted = [r["rms_fitted"] for r in by_count]
    assert fitted == sorted(fitted, reverse=True), by_count


def test_the_held_out_error_stops_improving_long_before_the_reported_one(models):
    """Past four coefficients the extra parameters buy nothing real."""
    rows = {r["coefficients"]: r for r in models["rows"]}
    gain_reported = rows[4]["rms_fitted"] - rows[14]["rms_fitted"]
    gain_real = rows[4]["rms_held_out"] - rows[14]["rms_held_out"]
    assert gain_reported > 0
    assert gain_real < gain_reported
    assert abs(gain_real) < 0.02, rows


def test_the_lowest_reported_error_comes_with_a_nonsense_principal_point(models):
    """Ten extra parameters move cx 67 px to buy 0.003 px of held-out error.

    The image is 640 px wide, so its centre is 320. Every model with four or
    fewer distortion coefficients puts cx within a few pixels of 343; the
    14-coefficient one puts it at 276.
    """
    rows = {r["model"]: r for r in models["rows"]}
    everything = rows["Everything (14 coefficients, control)"]
    default = rows["k1, k2, p1, p2 (OpenCV default)"]
    assert everything["rms_fitted"] < default["rms_fitted"]
    assert abs(everything["cx"] - default["cx"]) > 30, (everything, default)
    assert abs(default["rms_held_out"] - everything["rms_held_out"]) < 0.02


def test_a_calibration_from_five_views_is_not_one_number(full):
    """The same lens, the same thirteen photographs, twelve subsets of five."""
    rows = cal.intrinsic_stability(trials=8)
    k1 = [r["k1"] for r in rows]
    cx = [r["cx"] for r in rows]
    assert (max(k1) - min(k1)) / abs(np.mean(k1)) > 0.05, k1
    assert max(cx) - min(cx) > 3.0, cx


def test_adding_views_tightens_the_answer(sweep):
    rows = {r["views"]: r for r in sweep}
    assert rows[3]["fx_spread"] > 3 * rows[10]["fx_spread"], rows


# --------------------------------------------------------------------------- #
# the out-of-objective check
# --------------------------------------------------------------------------- #


def test_straightness_is_not_what_calibration_optimises():
    """`calibrateCamera` minimises reprojection error and nothing else.

    If straightness were part of the objective it could not be used to judge the
    result. This test is the guard on that claim: the function is called with
    the board's own corner correspondences, and there is no straightness term.
    """
    import inspect

    source = inspect.getsource(cal.calibrate)
    assert "cv2.calibrateCamera" in source
    assert "straight" not in source.lower()


def test_undistortion_actually_straightens_the_board(full):
    """7.5x, on a quantity nothing in the fit was aiming at."""
    assert full["straightness_before_px"] > 5 * full["straightness_px"]
    assert full["straightness_px"] < 0.15


def test_a_model_with_no_distortion_terms_leaves_the_bend_in():
    views = cal.usable_views("left")
    none = cal.calibrate(views, "left", cal.MODELS["No distortion model (control)"])
    default = cal.calibrate(views, "left", cal.MODELS["k1, k2, p1, p2 (OpenCV default)"])
    assert np.allclose(none["dist"], 0.0)
    bent = cal.line_straightness(none["K"], none["dist"], views)
    straight = cal.line_straightness(default["K"], default["dist"], views)
    assert bent > 5 * straight, (bent, straight)


# --------------------------------------------------------------------------- #
# the part that works
# --------------------------------------------------------------------------- #


def test_the_stereo_baseline_is_stable_across_subsets():
    """A physical distance between two lenses cannot change between subsets."""
    rows = cal.baseline_stability(trials=6)
    values = [r["baseline_squares"] for r in rows]
    assert np.std(values) / np.mean(values) < 0.02, values


def test_the_two_cameras_are_nearly_parallel():
    """A rig this rotated would not be usable, so a large angle means a bug."""
    stereo = cal.stereo_calibration()
    assert stereo["rotation_deg"] < 2.0, stereo
    # the translation is almost entirely along x, which is what a stereo rig is
    tx, ty, tz = stereo["translation"]
    assert abs(tx) > 20 * max(abs(ty), abs(tz)), stereo


def test_the_focal_lengths_agree_between_axes(full):
    """Square pixels: fx and fy should match to well under a percent."""
    assert abs(full["fx"] - full["fy"]) / full["fx"] < 0.005


def test_the_principal_point_is_near_the_image_centre(full):
    w, h = cal.image_size("left")
    assert abs(full["cx"] - w / 2) < 0.1 * w
    assert abs(full["cy"] - h / 2) < 0.1 * h


def test_the_lens_is_barrel_distorted(full):
    """k1 negative is barrel; a positive k1 here would mean a sign error."""
    assert full["k1"] < -0.1


# --------------------------------------------------------------------------- #
# the inputs
# --------------------------------------------------------------------------- #


def test_thirteen_views_not_fourteen():
    """left10/right10 are 404 upstream. Stated, not silently skipped."""
    assert 10 not in cal.VIEW_IDS
    assert len(cal.usable_views("left")) == 13
    assert len(cal.usable_views("right")) == 13


def test_the_board_geometry_is_exact():
    objp = cal.object_points()
    assert objp.shape == (cal.BOARD[0] * cal.BOARD[1], 3)
    assert np.allclose(objp[:, 2], 0.0)
    grid = objp[:, :2].reshape(cal.BOARD[1], cal.BOARD[0], 2)
    steps = np.diff(grid[0, :, 0])
    assert np.allclose(steps, cal.SQUARE)


def test_corners_are_refined_to_sub_pixel():
    coarse = cal.find_corners("left", 1, refine=False)
    fine = cal.find_corners("left", 1, refine=True)
    assert coarse is not None and fine is not None
    moved = np.abs(fine - coarse).max()
    assert 0 < moved < 3.0, moved


def test_held_out_scoring_resolves_the_pose():
    """A held-out view has no pose from the calibration; scoring one without
    re-solving it would be measuring the wrong thing."""
    import inspect

    source = inspect.getsource(cal.reprojection_error)
    assert "solvePnP" in source


def test_every_model_is_the_same_call_with_different_flags():
    import inspect

    source = inspect.getsource(cal.calibrate)
    assert source.count("cv2.calibrateCamera") == 1
    for name, flags in cal.MODELS.items():
        assert isinstance(flags, int), name
