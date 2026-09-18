"""Tests for project 29, object tracking.

The one worth reading first is `test_three_metrics_give_three_different_winners`.
Template matching has the best mean IoU, MOSSE stays on the target longest, and
optical flow has the smallest centre error — three reasonable ways to score the
same twelve runs, three different answers.

`test_mosse_finds_its_peak_at_the_right_place` is the regression test for the bug
that cost this project a day: the target Gaussian is built with `ifftshift`, so
zero displacement is index (0,0) and the response wraps. Subtracting half the box
instead put every step half a box out and scored a good tracker (0.567) below the
do-nothing control (0.248).
"""

from __future__ import annotations

import numpy as np
import pytest

import tracking as tk

pytestmark = pytest.mark.skipif(
    not tk.video_available(),
    reason="vtest.avi not cached; run `python tools/fetch_assets.py --set video`",
)


@pytest.fixture(scope="module")
def overall():
    return {r["tracker"]: r for r in tk.evaluate()}


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_three_metrics_give_three_different_winners(overall):
    real = [r for r in overall.values() if "control" not in r["tracker"]]
    by_iou = max(real, key=lambda r: r["mean_iou"])["tracker"]
    by_survival = max(real, key=lambda r: r["survival_rate"])["tracker"]
    by_centre = min(real, key=lambda r: r["centre_error_px"])["tracker"]
    assert len({by_iou, by_survival, by_centre}) == 3, (by_iou, by_survival, by_centre)


def test_accuracy_and_persistence_do_not_order_the_trackers_the_same_way():
    """They correlate, but they are not the same axis.

    Measured from frame 1: frame 0 is handed to every tracker and scores 1.0,
    and counting it put CamShift top of the accuracy column on the strength of
    the single frame it was given.
    """
    rows = [r for r in tk.survival_versus_accuracy() if "control" not in r["tracker"]]
    by_alive = [r["tracker"] for r in sorted(rows, key=lambda r: -r["iou_while_alive"])]
    by_survival = [r["tracker"] for r in sorted(rows, key=lambda r: -r["survival_rate"])]
    assert by_alive != by_survival, rows


def test_the_do_nothing_control_beats_a_real_tracker(overall):
    """A tracker that cannot beat a box nailed to the first frame is not tracking."""
    static = overall["Static box (control)"]
    worse = [name for name, r in overall.items()
             if "control" not in name and r["mean_iou"] < static["mean_iou"]]
    assert worse, {n: r["mean_iou"] for n, r in overall.items()}
    assert "CamShift" in worse


def test_the_colour_trackers_were_given_almost_nothing():
    """CamShift's score is a property of the scene, not only of the method.

    Both hue trackers need the back-projection to be brighter on the target than
    off it. On the colourful targets here the ratio is under 2, because the plaza
    responds too; on the dark-coated ones there is barely any hue at all.
    """
    contrast = tk.backprojection_contrast()
    saturation = tk.target_saturation()
    assert np.median([r["ratio"] for r in contrast]) < 3.0, contrast
    assert min(r["mean_saturation"] for r in saturation) < 40, saturation


def test_adding_a_kalman_filter_to_the_flow_tracker_makes_it_worse(overall):
    """Smoothing helps a noisy measurement and hurts a confidently wrong one."""
    lk = overall["Optical flow (LK)"]
    kalman = overall["LK + Kalman"]
    assert kalman["survival_rate"] < lk["survival_rate"]
    assert kalman["mean_iou"] < lk["mean_iou"]


def test_a_short_comparison_measures_initialisation(overall):
    """At ten frames the trackers are much closer together than at fifty."""
    rows = {r["length"]: r for r in tk.sweep_run_length(lengths=(10, 50),
                                                        starts=tk.STARTS[:6])}
    real = [t for t in tk.TRACKERS if "control" not in t]

    def spread(row):
        values = [row[t] for t in real]
        return max(values) - min(values)

    assert spread(rows[10]) < spread(rows[50]), rows


# --------------------------------------------------------------------------- #
# the MOSSE bug
# --------------------------------------------------------------------------- #


def test_mosse_finds_its_peak_at_the_right_place():
    """Correlating the first patch with its own filter must give zero shift.

    This is the invariant the shipped bug broke. With the wrong wrap the peak of
    a self-correlation reads as a displacement of half the box.
    """
    frames, truth = tk.truth_chain(tk.STARTS[0], 3)
    first = truth[0]
    boxes = tk.track_mosse(frames[:2], first)
    x0, y0, w, h = boxes[0]
    x1, y1, _, _ = boxes[1]
    assert abs(x1 - x0) < w / 3, (boxes, w)
    assert abs(y1 - y0) < h / 3, (boxes, h)


def test_mosse_beats_the_do_nothing_control(overall):
    """The bug put it below. This is the test that would have caught it."""
    assert (overall["MOSSE (from scratch)"]["mean_iou"]
            > 2 * overall["Static box (control)"]["mean_iou"])


def test_the_mosse_window_tapers_to_zero_at_the_edges():
    """Without it the FFT's wrap-around edge is a strong artificial gradient."""
    patch = np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.float32)
    windowed = tk._mosse_window(patch)
    assert abs(float(windowed[0].mean())) < 1e-6
    assert abs(float(windowed[:, 0].mean())) < 1e-6
    assert float(np.abs(windowed[12:20, 12:20]).mean()) > 0.1


# --------------------------------------------------------------------------- #
# the truth chain
# --------------------------------------------------------------------------- #


def test_the_runs_were_chosen_because_a_target_can_be_followed():
    """Not evenly spaced: the first version's chains were one or two frames long."""
    for row in tk.chain_quality():
        assert row["frames_tracked"] >= 40, row


def test_the_chain_reports_its_own_merges():
    """Two people walking together become one blob, and that is not hidden."""
    rows = tk.chain_quality()
    merged = [r for r in rows if r["merge_events"] > 0]
    assert merged, rows
    for r in merged:
        assert r["max_area_jump"] > 1.8, r


def test_the_jump_limit_scales_with_the_box():
    """A fixed pixel limit cuts the near chains or lets the far ones teleport."""
    import inspect

    source = inspect.getsource(tk.truth_chain)
    assert "jump_per_height * last[3]" in source


def test_the_empty_scene_has_nobody_in_it():
    """If people survived the median, every truth box would be wrong."""
    import cv2

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(cv2.cvtColor(tk.empty_scene(), cv2.COLOR_RGB2BGR),
                                    winStride=(8, 8), padding=(16, 16), scale=1.05,
                                    hitThreshold=0.5)
    assert len(boxes) == 0


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #


def test_every_tracker_returns_one_box_per_frame():
    frames, truth = tk.truth_chain(tk.STARTS[1], 12)
    for name, fn in tk.TRACKERS.items():
        boxes = fn(frames, truth[0])
        assert len(boxes) == len(frames), name
        for b in boxes:
            assert b is None or len(b) == 4, (name, b)


def test_every_box_stays_inside_the_frame():
    frames, truth = tk.truth_chain(tk.STARTS[2], 20)
    h, w = frames[0].shape[:2]
    for name, fn in tk.TRACKERS.items():
        for box in fn(frames, truth[0]):
            if box is None:
                continue
            x, y, bw, bh = box
            assert 0 <= x and 0 <= y, (name, box)
            assert x + bw <= w and y + bh <= h, (name, box)


def test_the_static_control_really_does_not_move():
    frames, truth = tk.truth_chain(tk.STARTS[0], 10)
    boxes = tk.static_box(frames, truth[0])
    assert len(set(boxes)) == 1


def test_mean_shift_cannot_resize_and_camshift_can():
    """Which is the entire difference between them, and it cuts both ways."""
    frames, truth = tk.truth_chain(tk.STARTS[3], 30)
    ms = tk.track_meanshift(frames, truth[0])
    cs = tk.track_camshift(frames, truth[0])
    assert len({(b[2], b[3]) for b in ms}) == 1
    assert len({(b[2], b[3]) for b in cs}) > 1


def test_box_iou_is_symmetric_and_bounded():
    a, b = (0, 0, 10, 10), (5, 5, 10, 10)
    assert tk.box_iou(a, b) == pytest.approx(tk.box_iou(b, a))
    assert tk.box_iou(a, a) == pytest.approx(1.0)
    assert tk.box_iou(a, (100, 100, 10, 10)) == 0.0
