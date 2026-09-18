"""Tests for project 46, epipolar geometry.

The one worth reading first is
`test_one_parameter_beats_every_projective_estimate`. A control that assumes a
rectified rig and fits a single constant vertical offset — by taking a median —
places the 702 chessboard corners closer to their epipolar lines than RANSAC,
LMedS or either eight-point variant. On a nearly parallel rig, estimating a
general projective F from noisy features is worse than using the structure you
already have.

`test_ransac_reports_a_high_inlier_rate_on_a_degenerate_configuration` is the
other one: points on a single plane cannot determine F at all, and RANSAC returns
an 81% inlier rate while failing to.
"""

from __future__ import annotations

import numpy as np
import pytest

import epipolar as ep

pytestmark = pytest.mark.skipif(
    not ep.assets_available(),
    reason="calibration images not cached; run "
           "`python tools/fetch_assets.py --set calibration`",
)


@pytest.fixture(scope="module")
def overall():
    return {r["estimator"]: r for r in ep.evaluate()}


@pytest.fixture(scope="module")
def truth():
    return ep.calibration_truth()


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_one_parameter_beats_every_projective_estimate(overall):
    """Seven degrees of freedom, estimated from features, lose to one median."""
    control = overall["Assume rectified + offset (1 parameter, control)"]
    for name in ("RANSAC", "LMedS", "8-point, normalised (Hartley)",
                 "8-point, raw pixels", "7-point"):
        assert control["board_median_px"] < overall[name]["board_median_px"], name


def test_the_one_parameter_control_works_because_the_rig_is_nearly_rectified(truth):
    """Its whole content is that the vertical disparity is nearly constant."""
    dy = truth["board_right"][:, 1] - truth["board_left"][:, 1]
    assert abs(dy.mean()) > 5.0         # there IS an offset, so zero parameters is wrong
    assert dy.std() < 2.0               # and it is nearly constant, so one is nearly right


def test_the_zero_parameter_control_is_wrong_by_exactly_that_offset(truth):
    """`assume_rectified` ignores the offset, so it should score about its size."""
    dy = truth["board_right"][:, 1] - truth["board_left"][:, 1]
    F, _ = ep.assume_rectified(truth["board_left"], truth["board_right"])
    error = ep.board_error(F)
    assert abs(error - abs(dy.mean())) < 1.0, (error, dy.mean())


def test_the_calibration_truth_shares_no_measurement_with_the_estimators():
    """It comes from stereoCalibrate on board corners; they see scene features."""
    import inspect

    source = inspect.getsource(ep.calibration_truth)
    assert "stereoCalibrate" in source
    assert "SIFT" not in source and "match_features" not in source

    estimator_source = inspect.getsource(ep.match_features)
    assert "find_corners" not in estimator_source


def test_the_truth_is_good_enough_to_be_a_truth(truth):
    """0.145 px over 702 corners, which is the floor everything else is read against."""
    assert ep.board_error(truth["F"]) < 0.3
    assert len(truth["board_left"]) > 600


def test_normalisation_improves_every_single_pair():
    """Hartley 1997. Same code, same correspondences, one conditioning step."""
    rows = ep.normalisation_effect()
    assert len(rows) >= 10
    for r in rows:
        assert r["raw_px"] > r["normalised_px"], r
        assert r["ratio"] > 1.0, r


def test_ransac_reports_a_high_inlier_rate_on_a_degenerate_configuration():
    """Correspondences on one plane cannot determine F, and nothing says so."""
    rows = ep.planar_degeneracy()
    board_only = [r["board only"] for r in rows if r["board only"] == r["board only"]]
    excluded = [r["board excluded"] for r in rows
                if r["board excluded"] == r["board excluded"]]
    rates = [r["board_only_inlier_rate"] for r in rows
             if r.get("board_only_inlier_rate") == r.get("board_only_inlier_rate")]
    assert np.median(board_only) > 3 * np.median(excluded), (board_only, excluded)
    assert np.median(rates) > 0.7, rates


def test_the_inlier_rate_improves_as_the_answer_gets_worse():
    """It is a function of the threshold, and the threshold is a free parameter."""
    rows = ep.inliers_are_not_quality()
    lo, hi = rows[0], rows[-1]
    assert hi["inlier_rate"] > lo["inlier_rate"]
    assert hi["board_px"] > lo["board_px"]


def test_the_epipole_is_not_a_usable_metric_on_this_rig(truth):
    """A near-parallel rig puts the epipole tens of image widths away.

    Reported rather than quietly dropped: a project that scored epipole distance
    here would be reporting noise with four significant figures.
    """
    e = ep.epipole(truth["F"])
    w, _ = (640, 480)
    assert abs(e[0]) > 20 * w, e


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #


def test_the_raw_eight_point_algorithm_is_ours_not_opencvs():
    """OpenCV always normalises internally, so the broken version has to be written."""
    import inspect

    source = inspect.getsource(ep._eight_point)
    assert "cv2.findFundamentalMat" not in source
    assert "np.linalg.svd" in source


def test_every_estimated_f_is_rank_two():
    """A fundamental matrix is singular by construction; the linear solution is not."""
    p1, p2 = ep.correspondences(ep.usable_pairs()[0], "SIFT", exclude_board=True)
    for name in ("8-point, raw pixels", "8-point, normalised (Hartley)"):
        F, _ = ep.ESTIMATORS[name](p1, p2, 1.0)
        s = np.linalg.svd(F, compute_uv=False)
        assert s[2] / s[0] < 1e-10, (name, s)


def test_normalisation_puts_the_centroid_at_the_origin():
    points = np.random.default_rng(0).uniform(0, 640, (50, 2))
    q, T = ep._normalise(points)
    assert np.allclose(q.mean(axis=0), 0.0, atol=1e-9)
    assert abs(float(np.mean(np.linalg.norm(q, axis=1))) - np.sqrt(2)) < 1e-9
    assert T.shape == (3, 3)


def test_the_symmetric_distance_is_symmetric():
    p1, p2 = ep.correspondences(ep.usable_pairs()[0], "SIFT", exclude_board=True)
    F, _ = ep.opencv_ransac(p1, p2, 1.0)
    a = ep.symmetric_epipolar_distance(F, p1, p2)
    b = ep.symmetric_epipolar_distance(F.T, p2, p1)
    assert abs(a - b) < 1e-6, (a, b)


def test_the_truth_f_gives_itself_zero_error(truth):
    a = ep.symmetric_epipolar_distance(truth["F"], truth["board_left"],
                                       truth["board_right"])
    assert a == pytest.approx(ep.board_error(truth["F"]))


def test_excluding_the_board_actually_removes_matches():
    index = ep.usable_pairs()[0]
    everything, _ = ep.correspondences(index, "SIFT", exclude_board=False)
    without, _ = ep.correspondences(index, "SIFT", exclude_board=True)
    assert len(without) < len(everything)
    share = ep.board_share(index, "SIFT")
    assert share["on_board"] == len(everything) - len(without)


def test_the_seven_point_subset_is_spread_over_the_image():
    """Seven correspondences from one corner determine nothing useful."""
    points = np.random.default_rng(1).uniform(0, 640, (200, 2))
    idx = ep._spread_subset(points, 7)
    chosen = points[idx]
    assert len(set(idx.tolist())) == 7
    assert np.ptp(chosen[:, 0]) > 0.5 * np.ptp(points[:, 0])


def test_a_random_eight_is_worse_than_a_fitted_eight(overall):
    assert (overall["Random 8 matches (control)"]["board_median_px"]
            > overall["8-point, normalised (Hartley)"]["board_median_px"])
