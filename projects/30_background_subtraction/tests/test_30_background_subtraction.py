"""Tests for project 30, background subtraction.

The one worth reading first is
`test_the_ranking_does_not_survive_the_truth_threshold`. It pins the project's
result: four settings of the oracle, one stop apart, and the method that wins
changes. The data does not decide which background subtractor is best; where the
threshold on *truth* goes decides it.

The rest pin the oracle itself (that it is an empty plaza, that an independent
detector confirms it finds people, that it is contaminated on one frame and this
is reported rather than fixed), the two controls, and the individual findings.
"""

from __future__ import annotations

import numpy as np
import pytest

import backsub as bs

pytestmark = pytest.mark.skipif(
    not bs.video_available(),
    reason="vtest.avi not cached; run `python tools/fetch_assets.py --set video`",
)


@pytest.fixture(scope="module")
def oracle():
    return {r["method"]: r for r in bs.evaluate_against_oracle()}


@pytest.fixture(scope="module")
def arms():
    return {r["method"]: r for r in bs.sustained_versus_instantaneous()}


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_ranking_does_not_survive_the_truth_threshold():
    """Two different methods win at four settings of the oracle.

    A looser oracle keeps only solid bodies, which is what the mixture models
    find; a tighter one keeps faint change too, which is what the high-recall
    methods find. Neither threshold is wrong, and that is the point.
    """
    rows = bs.oracle_threshold_sensitivity()
    winners = {max(bs.REAL_METHODS, key=lambda m: r[m]) for r in rows}
    assert len(winners) >= 2, rows

    loose = next(r for r in rows if r["oracle_multiple"] == 4.0)
    tight = next(r for r in rows if r["oracle_multiple"] == 1.5)
    assert loose["KNN"] > loose["Median background"]
    assert tight["Median background"] > tight["KNN"]


def test_the_ranking_also_depends_on_what_kind_of_change_you_mean(arms):
    """Instantaneous against sustained, with the same methods and the same clip."""
    fd = arms["Frame difference"]
    knn = arms["KNN"]
    assert fd["instantaneous_iou"] > 3 * knn["instantaneous_iou"]
    assert knn["sustained_iou"] > 4 * knn["instantaneous_iou"]
    # about 4x, and pinned at 3.5 because KNN's own IoU moves in the third
    # decimal between runs and 4.0 sits inside that wobble
    assert knn["difference"] > 3.5 * fd["difference"]


def test_pixel_accuracy_is_beaten_by_doing_nothing(oracle):
    """Foreground is under 4% of the frame, so the trivial answer scores 0.96."""
    nothing = oracle["All background (control)"]
    assert nothing["pixel_accuracy"] > 0.95
    assert nothing["iou"] == 0.0
    beaten = [m for m in bs.REAL_METHODS
              if oracle[m]["pixel_accuracy"] < nothing["pixel_accuracy"]]
    assert len(beaten) >= 2, {m: oracle[m]["pixel_accuracy"] for m in bs.REAL_METHODS}


def test_the_other_control_is_useless_in_the_opposite_direction(oracle):
    everything = oracle["All foreground (control)"]
    assert everything["recall"] == pytest.approx(1.0)
    assert everything["precision"] < 0.05
    assert everything["pixel_accuracy"] < 0.05


def test_a_better_background_model_is_not_a_better_mask():
    """The median's recall rises to 0.98 and its IoU falls, because the threshold
    is a fixed number of grey levels and does not move with the model."""
    rows = bs.history_precision_recall()
    lo, hi = rows[0], rows[-1]
    assert hi["recall"] > lo["recall"] + 0.15
    assert hi["precision"] < lo["precision"] - 0.15
    assert hi["iou"] < lo["iou"]


def test_the_mixture_models_are_precise_and_the_simple_ones_are_not(oracle):
    for name in ("MOG2", "KNN"):
        assert oracle[name]["precision"] > 0.85, name
        assert oracle[name]["recall"] < 0.6, name
    for name in ("Running average", "Median background"):
        assert oracle[name]["recall"] > 0.8, name
        assert oracle[name]["precision"] < 0.5, name


def test_every_method_is_weaker_on_a_boundary_than_in_an_interior():
    for row in bs.interior_versus_edge():
        assert row["edge_recall"] < row["interior_recall"], row


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_the_empty_scene_really_is_empty():
    """If people survived the median, the oracle would be scoring them as background.

    Checked with the pedestrian detector rather than by eye: HOG finds people in
    every one of the twelve target frames and none in the median.
    """
    import cv2

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    empty = bs.empty_scene()
    boxes, _ = hog.detectMultiScale(cv2.cvtColor(empty, cv2.COLOR_RGB2BGR),
                                    winStride=(8, 8), padding=(16, 16), scale=1.05,
                                    hitThreshold=0.5)
    assert len(boxes) == 0, f"{len(boxes)} people survived the median"


def test_the_oracle_and_an_independent_detector_agree():
    """HOG works from gradients in one frame; the oracle works from time.

    They share no information, so agreement is evidence rather than circularity.
    """
    rows = bs.oracle_versus_hog()
    total = sum(r["hog_detections"] for r in rows)
    found = sum(r["hog_detections_found_by_oracle"] for r in rows)
    assert total > 20
    assert found / total > 0.8, rows


def test_one_frame_is_contaminated_and_it_is_reported_not_dropped():
    """Frame 140's oracle mask is a field of 8x8 blocks over the flat tarmac.

    The clip is lossily compressed and the oracle's threshold is in grey levels,
    so a frame where the codec spent fewer bits reads as foreground. It stays in
    the twelve, and this test is what stops it being quietly removed later.
    """
    rows = {r["frame"]: r for r in bs.oracle_frame_outliers()}
    bad = rows[140]
    assert bad["components"] > 3 * np.median([r["components"] for r in rows.values()])
    assert bad["share_in_blobs_over_400px"] < 0.8
    for frame, row in rows.items():
        if frame != 140:
            assert row["share_in_blobs_over_400px"] > 0.7, row


def test_the_noise_floor_is_measured_not_chosen():
    floor = bs.measure_noise_floor(frames=60)
    assert 1.0 <= floor <= 12.0
    assert floor == bs.measure_noise_floor(frames=60)


# --------------------------------------------------------------------------- #
# the composite arm
# --------------------------------------------------------------------------- #


def test_truth_is_not_the_pasted_rectangle():
    """Only a third to a half of each rectangle is an observable change.

    An unchanged patch of grass inside the box is not something any method could
    detect, and scoring it would measure the ability to detect nothing.
    """
    rows = bs.truth_coverage()
    shares = [r["changed_share"] for r in rows]
    assert max(shares) < 0.7
    assert min(shares) > 0.15
    for r in rows:
        assert r["changed_px"] < r["rectangle_px"]


def test_the_composite_only_differs_inside_the_rectangle():
    composite, truth, original = bs.swap_composite(*bs.SWAPS[0])
    x, y, w, h = bs.SWAPS[0][2]
    outside = np.ones(original.shape[:2], bool)
    outside[y:y + h, x:x + w] = False
    assert np.array_equal(composite[outside], original[outside])
    assert not (truth[outside] > 0).any()


def test_both_halves_of_a_composite_are_photographs():
    """Nothing is drawn: every pixel comes from one frame or the other."""
    target, donor, rect = bs.SWAPS[3]
    composite, _, original = bs.swap_composite(target, donor, rect)
    donor_frame = bs.load_frame(donor)
    x, y, w, h = rect
    assert np.array_equal(composite[y:y + h, x:x + w], donor_frame[y:y + h, x:x + w])
    assert np.array_equal(composite[:y], original[:y])


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def test_frame_differencing_ignores_everything_but_the_last_frame():
    """Which is why its line in the history sweep is flat."""
    rows = bs.sweep_history_against_oracle(lengths=(5, 60, 240), frames=bs.FRAMES[:3])
    values = [r["Frame difference"] for r in rows]
    assert len(set(values)) == 1, values


def test_the_mixture_models_do_need_history():
    rows = {r["history"]: r for r in
            bs.sweep_history_against_oracle(lengths=(5, 60), frames=bs.FRAMES[:6])}
    assert rows[60]["MOG2"] > rows[5]["MOG2"]
    assert rows[60]["KNN"] > rows[5]["KNN"]


def test_every_method_gets_the_same_morphology_and_the_same_threshold():
    """Otherwise the comparison is of post-processing, not of models."""
    import inspect

    for name in ("Frame difference", "Running average", "Median background"):
        source = inspect.getsource(bs.METHODS[name])
        assert "noise_floor() if threshold is None" in source, name
        assert "_clean(" in source, name


def test_the_shadow_flag_costs_recall_and_it_is_measured():
    for row in bs.shadow_handling(swaps=bs.SWAPS[:3]):
        assert (row["mask_share_shadows_included"]
                > row["mask_share_shadows_excluded"]), row


def test_the_controls_do_exactly_what_they_say():
    frame = bs.load_frame(bs.FRAMES[0])
    assert bs.all_background([frame], frame).max() == 0
    assert bs.all_foreground([frame], frame).min() == 255
