"""Tests for project 06: lane detection.

They pin the result (what the ROI is worth, what the whole pipeline is worth
over guessing, and that neither of the two scores ranks the methods correctly on
its own), the mechanism (contrast predicts disagreement), the construction (the
region of interest is written in fractions, and the pixel literal is kept only
to be measured), and the geometry helpers that everything else rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import lanes as ln  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ln.available(),
    reason="run `python tools/fetch_assets.py --set lanes` to fetch the photographs")


# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #


def test_there_are_at_least_twelve_photographs():
    assert len(ln.image_names()) >= 12


def test_the_photographs_come_from_two_cameras():
    """The two resolutions are the point of this set, not an inconvenience.

    Every claim about the region of interest depends on there being more than
    one camera here.
    """
    cameras = {ln.camera_of(n) for n in ln.image_names()}
    assert len(cameras) == 2, cameras
    for camera in cameras:
        assert sum(1 for n in ln.image_names() if ln.camera_of(n) == camera) >= 5


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def test_vanishing_point_of_two_known_lines():
    """A line from (0,100) to (100,0) and one from (200,100) to (100,0) meet at
    (100, 0) -- checked against arithmetic rather than against another run."""
    vp = ln.vanishing_point(((0.0, 100.0, 100.0, 0.0), (200.0, 100.0, 100.0, 0.0)))
    assert vp == pytest.approx((100.0, 0.0), abs=1e-6)


def test_parallel_lines_have_no_vanishing_point():
    assert ln.vanishing_point(((0.0, 0.0, 0.0, 100.0), (50.0, 0.0, 50.0, 100.0))) is None


def test_a_missing_line_makes_the_answer_implausible():
    shape = (720, 1280, 3)
    assert not ln.plausible((None, (700.0, 446.0, 1100.0, 720.0)), shape)
    assert not ln.plausible((None, None), shape)


def test_lines_leaning_the_same_way_are_implausible():
    """Two lines both leaning left are a kerb and its shadow, not a lane."""
    shape = (720, 1280, 3)
    both_left = ((600.0, 446.0, 300.0, 720.0), (700.0, 446.0, 400.0, 720.0))
    assert not ln.plausible(both_left, shape)


def test_the_fixed_guess_is_plausible_by_construction():
    """The do-nothing control has to pass the geometry check.

    If it did not, the geometry check would be doing the control's job and the
    comparison would be measuring nothing.
    """
    for name in ln.image_names():
        img = ln.load(name)
        assert ln.plausible(ln.detect_fixed(img), img.shape), name


def test_lane_width_is_a_fraction_and_is_resolution_free():
    """The same road at two resolutions must give the same width.

    This is what makes the two camera groups comparable at all.
    """
    lanes = ((0.4 * 1280, 446.0, 0.2 * 1280, 720.0),
             (0.6 * 1280, 446.0, 0.8 * 1280, 720.0))
    half = tuple((a / 2, b / 2, c / 2, d / 2) for a, b, c, d in lanes)
    assert ln.lane_width(lanes, (720, 1280, 3)) == pytest.approx(0.6)
    assert ln.lane_width(half, (360, 640, 3)) == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# the region of interest is written in fractions
# --------------------------------------------------------------------------- #


def test_the_roi_is_fractional():
    """Every ROI coordinate is in [0, 1].

    A pixel crept into this constant once and the trapezoid silently became a
    sliver on one of the two cameras; this asserts the type of the constant, not
    just its value.
    """
    for x, y in ln.ROI:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, (x, y)


def test_the_fractional_roi_covers_the_same_share_of_both_cameras():
    shares = []
    for camera in sorted({ln.camera_of(n) for n in ln.image_names()}):
        name = next(n for n in ln.image_names() if ln.camera_of(n) == camera)
        shares.append((ln.roi_mask(ln.load(name).shape) > 0).mean())
    assert abs(shares[0] - shares[1]) < 0.01, shares


def test_the_pixel_roi_covers_far_less_of_the_bigger_frame():
    """The failure this project measures, asserted as a property of the mask.

    The pixel literal was written for 960x540. On 1280x720 it covers roughly
    half as much of the frame, and nothing warns about it.
    """
    small = next(n for n in ln.image_names() if ln.camera_of(n) == "960x540")
    big = next(n for n in ln.image_names() if ln.camera_of(n) == "1280x720")
    covers_small = (ln.roi_mask_pixels(ln.load(small).shape) > 0).mean()
    covers_big = (ln.roi_mask_pixels(ln.load(big).shape) > 0).mean()
    assert covers_small > 1.5 * covers_big, (covers_small, covers_big)


def test_the_pixel_roi_still_returns_lines_on_the_wrong_camera():
    """The failure is silent, which is the whole reason it is worth a project.

    A detector that raised or returned nothing would be found in a day.
    """
    big = [n for n in ln.image_names() if ln.camera_of(n) == "1280x720"]
    returned = sum(1 for n in big
                   if all(l is not None for l in ln.detect_pixel_roi(ln.load(n))))
    assert returned >= len(big) // 2, returned


def test_the_pixel_roi_costs_nothing_on_the_camera_it_was_written_for():
    rows = {r["camera"]: r for r in ln.roi_in_pixels()}
    same = rows["960x540"]
    assert same["fractional_plausible"] == same["pixel_plausible"]
    assert same["fractional_vp_spread"] == pytest.approx(same["pixel_vp_spread"],
                                                         rel=0.05)


def test_the_pixel_roi_costs_something_on_the_other_one():
    rows = {r["camera"]: r for r in ln.roi_in_pixels()}
    other = rows["1280x720"]
    assert other["pixel_vp_spread"] > 1.5 * other["fractional_vp_spread"], other


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_consistency_alone_ranks_the_do_nothing_control_first():
    """The headline negative result, asserted.

    The fixed guess returns the same two lines for every frame, so its spread is
    exactly zero and it wins the consistency score outright. A project that
    reported only this number would conclude that not looking at the road is the
    best lane detector available.
    """
    rows = {r["detector"]: r for r in ln.evaluate_all()}
    fixed = rows["Fixed guess (control)"]
    assert fixed["vp_x_spread"] == pytest.approx(0.0, abs=1e-6)
    assert fixed["width_spread"] == pytest.approx(0.0, abs=1e-6)
    best = min(rows.values(), key=lambda r: r["vp_x_spread"])
    assert best["detector"] == "Fixed guess (control)"


def test_the_recorded_warp_ranks_the_do_nothing_control_last():
    """And the second number is what rescues the comparison.

    The yaw is applied here, so it is known exactly, and a detector that ignores
    the image cannot follow it.
    """
    errors = {d: ln.yaw_repeatability(d)["median_error_pct"] for d in ln.DETECTORS}
    assert max(errors, key=errors.get) == "Fixed guess (control)"
    for d in ln.REAL_DETECTORS:
        assert errors[d] < 0.5 * errors["Fixed guess (control)"], (d, errors[d])


def test_every_real_pipeline_beats_both_controls_on_the_warp():
    errors = {d: ln.yaw_repeatability(d)["median_error_pct"] for d in ln.DETECTORS}
    worst_real = max(errors[d] for d in ln.REAL_DETECTORS)
    assert worst_real < errors["Bright pixels in the ROI (control)"]
    assert worst_real < errors["Fixed guess (control)"]


def test_the_roi_control_matches_the_full_pipeline_on_one_camera():
    """What the four steps after the trapezoid are worth, where they are worth least.

    On the 960x540 frames the control -- no colour test, no edges, no Hough --
    lands within about a percent of the full five-step pipeline.
    """
    rows = {r["camera"]: r for r in ln.control_gap_by_camera()}
    assert rows["960x540"]["median_gap_pct"] < 1.0
    assert rows["960x540"]["max_gap_pct"] < 2.0


def test_the_roi_control_does_not_match_it_on_the_other_camera():
    """And where they are worth most. The two halves of the claim are separate
    tests so a change that breaks one is not hidden by the other."""
    rows = {r["camera"]: r for r in ln.control_gap_by_camera()}
    assert rows["1280x720"]["median_gap_pct"] > 5 * rows["960x540"]["median_gap_pct"]


def test_removing_the_roi_is_the_largest_single_ablation():
    """Which of the four steps is the algorithm."""
    rows = {r["variant"]: r for r in ln.ablate()}
    full = rows["Full pipeline"]["vp_x_spread_pct"]
    damage = {k: v["vp_x_spread_pct"] - full
              for k, v in rows.items() if k != "Full pipeline"}
    assert max(damage, key=damage.get) == "without the region of interest"
    assert damage["without the region of interest"] > 3 * full


def test_removing_the_edge_detector_changes_almost_nothing():
    """Canny is the step every tutorial spends the most words on, and on this
    data it is worth less than a tenth of a percent of frame width."""
    rows = {r["variant"]: r for r in ln.ablate()}
    delta = abs(rows["without the edge detector"]["vp_x_spread_pct"]
                - rows["Full pipeline"]["vp_x_spread_pct"])
    assert delta < 0.1, delta


def test_the_slope_filter_matters_for_width_and_not_for_the_vanishing_point():
    """The two scores are not measuring the same thing, and this is where it shows.

    Dropping the slope band leaves the vanishing point almost where it was and
    makes the lane width wander five times as far -- horizontal segments from
    the road's own cross-markings enter both fits symmetrically.
    """
    rows = {r["variant"]: r for r in ln.ablate()}
    full, without = rows["Full pipeline"], rows["without the slope filter"]
    assert without["vp_x_spread_pct"] < 1.5 * full["vp_x_spread_pct"]
    assert without["width_spread"] > 3 * full["width_spread"]


# --------------------------------------------------------------------------- #
# the mechanism
# --------------------------------------------------------------------------- #


def test_paint_contrast_predicts_disagreement():
    """The difficulty axis is measured from the image before any detector runs,
    so it predicts rather than explains."""
    per = ln.per_image()
    contrast = np.array([r["paint_contrast"] for r in per])
    disagree = np.array([r["disagreement_pct"] for r in per])
    assert float(np.corrcoef(contrast, disagree)[0, 1]) < -0.5


def test_shadow_predicts_disagreement_too():
    per = ln.per_image()
    shadow = np.array([r["shadow_fraction"] for r in per])
    disagree = np.array([r["disagreement_pct"] for r in per])
    assert float(np.corrcoef(shadow, disagree)[0, 1]) > 0.5


def test_the_hardest_frames_are_the_low_contrast_ones():
    per = sorted(ln.per_image(), key=lambda r: r["paint_contrast"])
    hardest = {r["image"] for r in per[:3]}
    worst = {r["image"] for r in sorted(per, key=lambda r: -r["disagreement_pct"])[:4]}
    assert len(hardest & worst) >= 2, (hardest, worst)


# --------------------------------------------------------------------------- #
# the recorded warp is really recorded
# --------------------------------------------------------------------------- #


def test_the_yaw_homography_moves_the_horizon_and_not_the_bottom_edge():
    shape = (720, 1280, 3)
    H = ln.yaw_homography(shape, 40.0)
    bottom = ln._apply(H, 640.0, 719.0)
    high = ln._apply(H, 640.0, 0.55 * 720)
    assert bottom[0] == pytest.approx(640.0, abs=1e-3)
    assert high[0] == pytest.approx(680.0, abs=1e-3)


def test_a_zero_yaw_is_the_identity():
    """A regression test for the sign and the anchor rows together: with no
    shift the homography must not move anything."""
    H = ln.yaw_homography((720, 1280, 3), 0.0)
    for x, y in ((100.0, 700.0), (640.0, 400.0), (1100.0, 500.0)):
        assert ln._apply(H, x, y) == pytest.approx((x, y), abs=1e-3)
