"""Tests for project 54: motion-triggered security alert.

They pin the result (a perfect mask plus the naive rule is unusable; the cooldown
is the decisive knob; a rate-matched random control catches far fewer), the
construction (the zone is verified still frame by frame rather than assumed; an
intrusion span that overlaps real activity is refused; the planted frames are
excluded from the false-alert count), and the decision layer itself, which is
pure arithmetic on a signal and is tested against hand-worked examples rather
than against the clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import alert as al  # noqa: E402

pytestmark = pytest.mark.skipif(
    not al.available(),
    reason="run `python tools/fetch_assets.py --set video` to fetch the clip")


# --------------------------------------------------------------------------- #
# the decision layer, on signals written by hand
# --------------------------------------------------------------------------- #


def test_any_motion_fires_on_every_frame_over_threshold():
    signal = np.array([0, 0, 900, 900, 900, 0, 900])
    assert al.alerts_from(signal, 600, 1, 0).tolist() == [2, 3, 4, 6]


def test_minimum_area_ignores_small_blobs():
    signal = np.array([100, 100, 900, 100])
    assert al.alerts_from(signal, 600, 1, 0).tolist() == [2]
    assert al.alerts_from(signal, 50, 1, 0).tolist() == [0, 1, 2, 3]


def test_persistence_requires_consecutive_frames():
    """A single frame of motion is a bird. Three in a row is a person."""
    signal = np.array([900, 0, 900, 900, 900, 0])
    assert al.alerts_from(signal, 600, 3, 0).tolist() == [4]
    assert al.alerts_from(signal, 600, 1, 0).tolist() == [0, 2, 3, 4]


def test_persistence_resets_on_a_gap():
    signal = np.array([900, 900, 0, 900, 900])
    assert al.alerts_from(signal, 600, 3, 0).tolist() == []


def test_cooldown_collapses_a_run_into_one_alert():
    signal = np.array([900] * 20)
    assert al.alerts_from(signal, 600, 1, 50).tolist() == [0]
    assert len(al.alerts_from(signal, 600, 1, 0)) == 20


def test_cooldown_permits_a_second_alert_once_it_expires():
    signal = np.array([900] * 30)
    assert al.alerts_from(signal, 600, 1, 10).tolist() == [0, 11, 22]


def test_an_empty_signal_raises_nothing():
    assert al.alerts_from(np.zeros(50), 600, 1, 0).tolist() == []


# --------------------------------------------------------------------------- #
# scoring is at the level of events, and excludes planted frames
# --------------------------------------------------------------------------- #


def test_a_correct_alert_during_an_intrusion_is_not_counted_as_false():
    """A real bug, kept as a test.

    `quiet` is measured on the unmodified clip, so it still contains the frames
    an intruder was later planted into. Without subtracting the spans, every
    correct alert was also counted as a false one, and the first run of this
    project reported 214 false alerts where there are 12.
    """
    present = np.zeros(100, bool)
    present[40:50] = True
    quiet = np.arange(100)          # the whole clip looks quiet before planting
    fired = np.array([45])          # one alert, right in the middle of the intrusion
    s = al.score_alerts(fired, present, quiet)
    assert s["detected"] == 1
    assert s["false_alerts"] == 0


def test_an_alert_on_a_verified_quiet_frame_is_false():
    present = np.zeros(100, bool)
    present[40:50] = True
    s = al.score_alerts(np.array([5]), present, np.arange(100))
    assert s["detected"] == 0 and s["false_alerts"] == 1


def test_latency_is_measured_from_the_frame_the_intruder_enters():
    present = np.zeros(100, bool)
    present[40:60] = True
    s = al.score_alerts(np.array([47, 52]), present, np.arange(100))
    assert s["median_latency_frames"] == 7
    assert s["median_latency_seconds"] == pytest.approx(7 / al.FPS)


def test_alerts_during_unlabelled_real_activity_are_neither_right_nor_wrong():
    """The honest third bucket. The zone has real people in it in about 4% of
    frames, and those frames carry no label, so an alert there is counted
    separately rather than being called correct or incorrect."""
    present = np.zeros(100, bool)
    present[40:50] = True
    quiet = np.arange(0, 30)        # frames 30-39 and 50-99 are unlabelled
    s = al.score_alerts(np.array([10, 45, 80]), present, quiet)
    assert s["false_alerts"] == 1 and s["detected"] == 1 and s["unscored"] == 1


def test_present_spans_finds_contiguous_runs():
    present = np.array([0, 1, 1, 0, 0, 1, 0], bool)
    assert al.present_spans(present) == [(1, 2), (5, 5)]


def test_a_span_touching_the_end_is_closed():
    assert al.present_spans(np.array([0, 1, 1], bool)) == [(1, 2)]


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def test_never_alert_has_a_perfect_false_rate_and_no_security():
    present = np.zeros(100, bool)
    present[40:50] = True
    s = al.score_alerts(al.control_never(np.zeros(100)), present, np.arange(100))
    assert s["false_alerts"] == 0
    assert s["missed"] == 1 and s["detected"] == 0


def test_always_alert_catches_everything_and_is_useless():
    present = np.zeros(100, bool)
    present[40:50] = True
    s = al.score_alerts(al.control_always(np.zeros(100)), present, np.arange(100))
    assert s["detected"] == 1 and s["missed"] == 0
    assert s["false_alerts"] == 90


def test_the_random_control_is_rate_matched_and_reproducible():
    a = al.control_random(np.zeros(500), count=12, seed=0)
    b = al.control_random(np.zeros(500), count=12, seed=0)
    assert len(a) == 12 and np.array_equal(a, b)
    assert not np.array_equal(a, al.control_random(np.zeros(500), count=12, seed=1))


# --------------------------------------------------------------------------- #
# the truth had to be earned
# --------------------------------------------------------------------------- #


def test_the_zone_is_mostly_still_but_not_empty():
    """The fact that forced the verified-quiet arm to exist.

    If the zone were genuinely empty the project could assume it; it is not, so
    every false alert is counted only where the oracle confirms stillness.
    """
    activity = al.zone_activity()
    busy = al.busy_frames(activity)
    assert 0 < len(busy) < 0.15 * len(activity), len(busy)


def test_the_zone_is_far_quieter_than_the_busy_part_of_the_scene():
    """Why this zone and not another: it was chosen by measurement."""
    activity = al.zone_activity()
    assert (activity >= al.QUIET_AREA).mean() < 0.10


def test_an_intrusion_overlapping_real_activity_is_refused():
    """A guard nobody can see firing is a guard nobody can trust, so the list
    deliberately contains one span that must be rejected."""
    plan = al.plan_intrusions()
    refused = [r for r in plan if not r["usable"]]
    assert len(refused) >= 1
    for r in refused:
        assert r["overlaps_real_activity"] > 0


def test_most_intrusions_are_planted():
    plan = al.plan_intrusions()
    assert sum(r["usable"] for r in plan) >= 5


def test_the_planted_clip_differs_from_the_original_only_inside_the_spans():
    """The construction, asserted: outside an intrusion the frame is untouched."""
    activity = al.zone_activity()
    seq, present = al.planted_clip(activity=activity)
    original = al.frames()
    spans = al.present_spans(present)
    planted = {i for a, b in spans for i in range(a, b + 1)}
    for i in (0, 60, 200, 350, 600, 790):
        if i not in planted:
            assert np.array_equal(seq[i], original[i]), i
    a, b = spans[0]
    assert not np.array_equal(seq[(a + b) // 2], original[(a + b) // 2])


def test_the_planted_intruder_is_inside_the_zone():
    activity = al.zone_activity()
    seq, present = al.planted_clip(activity=activity)
    spans = al.present_spans(present)
    a, b = spans[0]
    mid = (a + b) // 2
    assert al.largest_blob_in_zone(al.oracle_mask(seq[mid])) > al.QUIET_AREA


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def _signal_and_truth():
    activity = al.zone_activity()
    seq, present = al.planted_clip(activity=activity)
    signal = np.array([al.largest_blob_in_zone(al.oracle_mask(f)) for f in seq])
    return signal, present, al.quiet_frames(activity), al.present_spans(present)


def test_a_perfect_mask_with_the_naive_rule_is_unusable():
    """The headline. Nothing is wrong with the pixels; the rule is wrong."""
    signal, present, quiet, spans = _signal_and_truth()
    area, persistence, cooldown = al.RULES["Any motion in the zone"]
    s = al.score_alerts(al.alerts_from(signal, area, persistence, cooldown),
                        present, quiet, spans)
    assert s["detected"] == len(spans)
    assert s["alerts"] > 20 * len(spans)


def test_the_cooldown_is_the_decisive_knob():
    """Area and persistence together cut the alert count by well under half;
    adding a cooldown cuts it by more than tenfold, and misses nothing."""
    signal, present, quiet, spans = _signal_and_truth()
    naive = al.score_alerts(al.alerts_from(signal, *al.RULES["Any motion in the zone"]),
                            present, quiet, spans)
    persist = al.score_alerts(al.alerts_from(signal, *al.RULES["+ persistence 8"]),
                              present, quiet, spans)
    cooled = al.score_alerts(al.alerts_from(signal, *al.RULES["+ cooldown 50"]),
                             present, quiet, spans)
    assert persist["alerts"] > 0.4 * naive["alerts"]
    assert cooled["alerts"] < 0.15 * persist["alerts"]
    assert cooled["missed"] == 0


def test_too_much_cooldown_starts_missing_intrusions():
    """Quieter stops being better, and the project reports where."""
    signal, present, quiet, spans = _signal_and_truth()
    long = al.score_alerts(al.alerts_from(signal, 600, 8, 200), present, quiet, spans)
    assert long["missed"] > 0


def test_firing_the_right_number_of_times_is_not_enough():
    """What the rate-matched control is for: it fires as often as the tuned rule
    and catches far fewer, so the tuned rule is firing at the right *moments*."""
    signal, present, quiet, spans = _signal_and_truth()
    cooled = al.score_alerts(al.alerts_from(signal, *al.RULES["+ cooldown 50"]),
                             present, quiet, spans)
    matched = al.score_alerts(al.control_random(signal, count=cooled["alerts"]),
                              present, quiet, spans)
    assert matched["alerts"] == cooled["alerts"]
    assert matched["detected"] < cooled["detected"]
